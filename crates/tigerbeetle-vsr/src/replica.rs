// =============================================================================
// Replica — the VSR consensus engine
// =============================================================================
//
// Ported from `src/vsr/replica.zig` (~12,400 lines).
//
// This is the heart of TigerBeetle: it drives the Viewstamped Replication
// protocol, manages the prepare/commit pipeline, handles view changes,
// coordinates checkpoints, and owns all subsystems (journal, grid,
// state_machine, clock, client_sessions, client_replies).
//
// The full async state machine (commit_dispatch, on_message, view changes)
// is deferred — this module currently provides the struct layout, quorum
// computation, and construction logic so that other modules can reference it.

use tigerbeetle_core::constants;

use crate::command::Command;
use crate::message_header;
use crate::message_header::TypedHeader;

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/// Replica status — the high-level phase of the VSR protocol.
///
/// Upstream: `src/vsr/replica.zig:54` (`Status`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    /// Normal operation: processing requests, preparing, committing.
    Normal = 0,
    /// View change in progress: collecting ExitView/JoinView quorums.
    ViewChange = 1,
    /// Recovering from a crash: replaying the WAL.
    Recovering = 2,
    /// Recovering the head: finding the latest committed op.
    RecoveringHead = 3,
}

// ---------------------------------------------------------------------------
// Replica
// ---------------------------------------------------------------------------

/// The VSR Replica — owns all consensus state and subsystems.
///
/// This is a simplified skeleton. The full struct (~490 lines of fields in
/// upstream) will be built up incrementally as subsystems are completed.
///
/// Upstream: `src/vsr/replica.zig:144` (`ReplicaType`).
pub struct Replica {
    // ── Identity & cluster geometry ──────────────────────────────────────
    pub cluster: u128,
    pub replica_index: u16,
    pub replica_count: u16,

    // ── VSR consensus state ──────────────────────────────────────────────
    pub view: u32,
    pub log_view: u32,
    pub status: Status,
    /// Head operation (latest known op).
    pub op: u64,
    /// Minimum committed op (all replicas have committed ≤ this).
    pub commit_min: u64,
    /// Maximum op the primary has committed (may not yet be known to backups).
    pub commit_max: u64,

    // ── Prepare pipeline (primary only) ──────────────────────────────────
    /// The pipeline of inflight prepares.
    pub pipeline_queue: PipelineQueue,
    /// Bitset tracking which replicas have sent prepare_ok for each pipeline slot.
    /// `ok_from_all_replicas[slot]` is a bitmask of replica indices.
    pub ok_from_all_replicas: Vec<u64>,
    /// Monotonically increasing timestamp for prepares.
    pub prepare_timestamp: u64,
    /// The prepare currently being committed (primary).
    pub commit_prepare: Option<u64>, // op of the prepare being committed

    // ── Commit pipeline ──────────────────────────────────────────────────
    /// Current stage of the commit pipeline state machine.
    pub commit_stage: CommitStage,
    /// Whether commit_dispatch is currently executing (reentrancy guard).
    pub commit_dispatch_entered: bool,

    // ── Subsystems ───────────────────────────────────────────────────────
    /// The write-ahead log (WAL): suspend slot geometry, header ring, dirty/
    /// faulty recovery bits, and on-disk checksums.
    pub journal: crate::journal::Journal,
    // grid: Grid,
    // state_machine: StateMachine,
    // clock: Clock,
    // client_sessions: ClientSessions,
    // client_replies: ClientReplies,
    // message_bus: MessageBus,

    // ── Timeouts (tick counts) ──────────────────────────────────────────
    pub ping_timeout: Timeout,
    pub prepare_timeout: Timeout,
    pub commit_message_timeout: Timeout,
    pub view_change_status_timeout: Timeout,
    pub exit_view_message_timeout: Timeout,
    pub exit_view_window_timeout: Timeout,
    pub primary_abdicate_timeout: Timeout,

    // ── Fault detection ──────────────────────────────────────────────────
    /// EWMA fault detector for commit heartbeats from the primary.
    pub commit_fault: FaultDetector,
    /// Timestamp of the last fresh commit heartbeat received.
    pub heartbeat_timestamp: u64,
    /// Bitmask of replicas that have sent ExitView for the current view.
    pub exit_view_from_all_replicas: u64,
    /// Whether the primary has started abdicating.
    pub primary_abdicating: bool,
}

// ---------------------------------------------------------------------------
// Timeout
// ---------------------------------------------------------------------------

/// A simple tick-based timeout counter.
///
/// Upstream uses `vsr.Timeout` struct with similar semantics.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timeout {
    /// Ticks remaining until the timeout fires.
    pub ticks: u32,
    /// Whether the timeout is currently active.
    pub active: bool,
}

impl Timeout {
    #[must_use]
    pub const fn start(ticks: u32) -> Self {
        Self { ticks, active: true }
    }

    /// Advance by one tick. Returns `true` if the timeout has fired.
    pub fn tick(&mut self) -> bool {
        if !self.active {
            return false;
        }
        self.ticks = self.ticks.saturating_sub(1);
        if self.ticks == 0 {
            self.active = false;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self, ticks: u32) {
        self.ticks = ticks;
        self.active = true;
    }

    pub fn stop(&mut self) {
        self.active = false;
    }
}

// ---------------------------------------------------------------------------
// FaultDetector — sliding-window EWMA of signal intervals
// ---------------------------------------------------------------------------

/// Tardiness level reported by the fault detector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tardiness {
    /// elapsed ≤ 1.5× ewma — primary is alive.
    Green,
    /// 1.5× < elapsed ≤ 3× ewma — signal may be delayed.
    Yellow,
    /// elapsed > 3× ewma — primary likely dead.
    Red,
}

/// Sans-IO sliding-window EWMA of inter-signal intervals.
///
/// Two signals feed it: `on_prepare` (new prepares replicated) and
/// `on_commit` (commit heartbeats).  The `tardy()` method compares
/// elapsed time since the last signal against the EWMA to classify
/// liveness as green/yellow/red.
///
/// Upstream: `src/vsr/fault_detector.zig:1` (`FaultDetector`).
#[derive(Clone, Copy, Debug)]
pub struct FaultDetector {
    /// Minimum interval clamp (milliseconds).
    interval_min: u64,
    /// Maximum interval clamp (milliseconds).
    interval_max: u64,
    /// EWMA of the interval between signals (milliseconds).
    interval_ewma: u64,
    /// Timestamp of the last signal (milliseconds).
    last_signal_timestamp: u64,
}

impl FaultDetector {
    /// Create a new fault detector.
    ///
    /// # Panics
    ///
    /// Panics if `interval_min >= interval_max`.
    ///
    /// Upstream: `src/vsr/fault_detector.zig:40` (`init`).
    #[must_use]
    pub fn new(now: u64, interval_min: u64, interval_max: u64) -> Self {
        assert!(interval_min < interval_max);
        Self { interval_min, interval_max, interval_ewma: interval_max, last_signal_timestamp: now }
    }

    /// Record a signal at the given monotonic timestamp (milliseconds).
    ///
    /// Upstream: `src/vsr/fault_detector.zig:57` (`signal`).
    pub fn signal(&mut self, now: u64) {
        let elapsed = now.saturating_sub(self.last_signal_timestamp);
        let clamped = elapsed.clamp(self.interval_min, self.interval_max);

        // EWMA: new = (4 * old + sample) / 5
        self.interval_ewma = (self.interval_ewma * 4 + clamped) / 5;
        self.last_signal_timestamp = now;
    }

    /// Classify the current tardiness based on elapsed time since last signal.
    ///
    /// Upstream: `src/vsr/fault_detector.zig:86` (`tardy`).
    #[must_use]
    pub fn tardy(&self, now: u64) -> Tardiness {
        let elapsed = now.saturating_sub(self.last_signal_timestamp);

        if elapsed * 2 <= self.interval_ewma * 3 {
            Tardiness::Green
        } else if elapsed <= self.interval_ewma * 3 {
            Tardiness::Yellow
        } else {
            Tardiness::Red
        }
    }
}

// ---------------------------------------------------------------------------
// CommitStage — async commit pipeline state machine
// ---------------------------------------------------------------------------

/// Progress through the commit pipeline for one prepare.
///
/// Each variant represents a stage in the pipeline. Stages that need async
/// I/O (prefetch, stall, compact, checkpoint_*) return `.pending` from the
/// dispatch; `.ready` means synchronous completion.
///
/// Upstream: `src/vsr/replica.zig:83` (`CommitStage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitStage {
    /// Not committing.
    Idle,
    /// Get the next prepare to commit from the pipeline.
    Start,
    /// Break out of the commit loop if there is nothing to commit.
    CheckPrepare,
    /// Load required data from LSM tree on disk into memory.
    Prefetch,
    /// Primary delays committing as backpressure to let backups catch up.
    Stall,
    /// Ensure that ClientReplies has at least one Write available.
    ReplySetup,
    /// Execute state machine logic.
    Execute,
    /// Every `VSR_CHECKPOINT_OPS`, mark the current checkpoint as durable.
    CheckpointDurable,
    /// Run one beat of LSM compaction.
    Compact,
    /// Every `VSR_CHECKPOINT_OPS`, persist the current state to disk.
    CheckpointData,
    /// Update the superblock.
    CheckpointSuperblock,
}

// ---------------------------------------------------------------------------
// Quorum computation
// ---------------------------------------------------------------------------

/// VSR quorum sizes for a given `replica_count`.
///
/// Upstream: `src/vsr.zig` (`quorums` function).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quorum {
    /// Replication quorum: how many prepare_ok messages to commit.
    pub replication: u16,
    /// View-change quorum: how many ExitView messages to start a new view.
    pub view_change: u16,
    /// Nack-prepare quorum: how many nack_prepare messages to nack.
    pub nack_prepare: u16,
    /// Simple majority.
    pub majority: u16,
}

/// Compute VSR quorums for a given replica count.
///
/// Upstream: `src/vsr.zig` — `quorums` function.
#[must_use]
pub const fn quorums(replica_count: u16) -> Quorum {
    // From upstream:
    // quorum_replication = @divFloor(replica_count, 2) + 1
    // quorum_view_change = @divFloor(replica_count, 2) + 1
    // quorum_nack_prepare = @divFloor(replica_count, 2)
    let half = replica_count / 2;
    Quorum { replication: half + 1, view_change: half + 1, nack_prepare: half, majority: half + 1 }
}

// ---------------------------------------------------------------------------
// Replica impl
// ---------------------------------------------------------------------------

impl Replica {
    /// Create a new replica. This is the simplified constructor — the full
    /// `init` in upstream is ~380 lines and initializes all subsystems.
    ///
    /// For now, only the consensus state is initialized; subsystems
    /// (journal, grid, state_machine, etc.) are created externally and
    /// assigned after construction.
    ///
    /// # Panics
    ///
    /// Panics if `replica_index >= replica_count` or `replica_count == 0`.
    #[must_use]
    pub fn new(cluster: u128, replica_index: u16, replica_count: u16) -> Self {
        assert!(replica_index < replica_count);
        assert!(replica_count > 0);

        let quorum = quorums(replica_count);
        assert!(quorum.replication > 0);
        assert!(quorum.view_change > 0);

        Self {
            cluster,
            replica_index,
            replica_count,
            view: 0,
            log_view: 0,
            status: Status::Recovering,
            op: 0,
            commit_min: 0,
            commit_max: 0,
            pipeline_queue: PipelineQueue::default(),
            ok_from_all_replicas: Vec::new(),
            prepare_timestamp: 0,
            commit_prepare: None,
            commit_stage: CommitStage::Idle,
            commit_dispatch_entered: false,
            ping_timeout: Timeout::default(),
            prepare_timeout: Timeout::default(),
            commit_message_timeout: Timeout::default(),
            view_change_status_timeout: Timeout::default(),
            exit_view_message_timeout: Timeout::default(),
            exit_view_window_timeout: Timeout::default(),
            primary_abdicate_timeout: Timeout::default(),
            commit_fault: FaultDetector::new(0, 100, 2_000),
            heartbeat_timestamp: 0,
            exit_view_from_all_replicas: 0,
            primary_abdicating: false,
            journal: crate::journal::Journal::new(cluster, replica_index),
        }
    }

    /// Returns the quorum sizes for this replica's cluster.
    #[must_use]
    pub fn quorum(&self) -> Quorum {
        quorums(self.replica_count)
    }

    /// Returns `true` if this replica is the current primary for its view.
    ///
    /// Upstream: `src/vsr.zig` — `primary_index`.
    #[must_use]
    pub fn is_primary(&self) -> bool {
        self.primary_index() == self.replica_index
    }

    /// Returns the index of the primary replica for the current view.
    ///
    /// Upstream: `src/vsr.zig` — `primary_index`.
    #[must_use]
    pub fn primary_index(&self) -> u16 {
        Self::primary_index_for_view(self.view, self.replica_count)
    }

    /// Returns the index of the replica that would be primary for a given view.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // view % replica_count < u16::MAX
    pub fn primary_index_for_view(view: u32, replica_count: u16) -> u16 {
        (view % u32::from(replica_count)) as u16
    }

    /// Returns the checkpoint trigger op: the op at which the next checkpoint
    /// should be initiated.
    ///
    /// Upstream: `vsr.checkpoint_next_trigger`.
    #[must_use]
    pub fn op_checkpoint_next_trigger(&self) -> u64 {
        // Every `checkpoint_ops` ops we trigger a checkpoint.
        // checkpoint_ops = constants.vsr_checkpoint_ops
        // trigger = checkpoint_ops - 1 (the op that triggers the checkpoint)
        // The trigger is the last op in the current checkpoint interval.
        let checkpoint_ops = constants::VSR_CHECKPOINT_OPS as u64;
        let interval = checkpoint_ops;
        // Find the next trigger >= self.commit_min.
        let base = self.commit_min / interval;
        (base + 1) * interval
    }

    /// Advance commit_max monotonically.
    ///
    /// Upstream: `src/vsr/replica.zig:4188` (`advance_commit_max`).
    pub fn advance_commit_max(&mut self, commit: u64) {
        if commit > self.commit_max {
            self.commit_max = commit;
        }
    }

    /// Begin preparing a client request on the primary.
    ///
    /// The incoming request message is rewritten in-place as a Prepare:
    /// header is replaced with a Prepare header carrying the new op,
    /// commit tip, parent checksum (hash-chain link), and timestamp.
    ///
    /// Returns `Ok(op)` if the prepare was accepted, or `Err(RejectReason)`
    /// if it could not be processed.
    ///
    /// # Errors
    /// Returns [`PrepareReject`] if the replica is not primary, not in normal
    /// status, behind on commits, or the pipeline is full.
    ///
    /// # Panics
    /// Panics if `operation` is `.reserved` or `.root`.
    ///
    /// Upstream: `src/vsr/replica.zig:7283` (`primary_pipeline_prepare`).
    pub fn primary_pipeline_prepare(
        &mut self,
        client: u128,
        request: u64,
        operation: crate::Operation,
        body_size: u32,
    ) -> Result<u64, PrepareReject> {
        if !self.is_primary() {
            return Err(PrepareReject::NotPrimary);
        }
        if self.status != Status::Normal {
            return Err(PrepareReject::NotNormal);
        }
        if self.commit_min != self.commit_max {
            return Err(PrepareReject::BehindCommit);
        }

        let pipeline_limit = constants::PIPELINE_PREPARE_QUEUE_MAX as usize
            + constants::PIPELINE_REQUEST_QUEUE_MAX as usize;
        if self.pipeline_queue.prepare_queue.len() >= pipeline_limit {
            return Err(PrepareReject::PipelineFull);
        }

        // The incoming request is rewritten as a Prepare addressing `self.op + 1`;
        // `self.op` is advanced below, exactly as upstream's `on_prepare` does for
        // the primary's self-sent prepare message.
        assert_ne!(operation, crate::Operation::RESERVED, "operation != .reserved");
        assert_ne!(operation, crate::Operation::ROOT, "operation != .root");
        let op = self.op + 1;

        // Timestamp: monotonic, at least prepare_timestamp + 1.
        self.prepare_timestamp = self.prepare_timestamp.max(self.commit_max) + 1;
        let timestamp = self.prepare_timestamp;

        // The hash-chain parent is the checksum of the journal's current head.
        let parent =
            self.journal.header_with_op(self.op).map_or(0, message_header::Prepare::checksum);

        let mut header = message_header::Prepare {
            cluster: self.cluster,
            #[allow(clippy::cast_possible_truncation)] // replica_count ≤ u8::MAX
            replica: self.replica_index as u8,
            view: self.view,
            op,
            commit: self.commit_max,
            timestamp,
            parent,
            operation,
            ..message_header::Prepare::default()
        };
        header.set_checksum_body(&[]);
        header.set_checksum();

        // Advance op and record the prepare in the journal (upstream `on_prepare`):
        self.op = op;
        self.journal.set_header_as_dirty(&header);

        let prepare = PipelinePrepare {
            op,
            checksum: header.checksum(),
            acks_received: 0,
            ok_quorum_received: false,
        };
        self.pipeline_queue.prepare_queue.push(prepare);
        self.ok_from_all_replicas.push(0);

        // Start timeouts if the pipeline was previously empty.
        if self.pipeline_queue.prepare_queue.len() == 1 {
            self.prepare_timeout = Timeout::start(constants::PIPELINE_PREPARE_QUEUE_MAX);
        }

        let _ = (client, request, body_size);
        // TODO(port): serialize the prepare message, send to backups.

        Ok(op)
    }

    /// Handle a prepare_ok from a backup.
    ///
    /// Tracks the ack, counts quorum, and triggers commit when reached.
    ///
    /// Upstream: `src/vsr/replica.zig:2248` (`on_prepare_ok`).
    pub fn on_prepare_ok(&mut self, op: u64, checksum: u128, replica: u16) -> PrepareOkResult {
        if !self.is_primary() {
            return PrepareOkResult::Ignored;
        }
        if self.status != Status::Normal {
            return PrepareOkResult::Ignored;
        }

        // Find the pipeline slot for this op.
        let Some(slot) = self.pipeline_queue.prepare_queue.iter().position(|p| p.op == op) else {
            return PrepareOkResult::UnknownOp;
        };

        let prepare = &mut self.pipeline_queue.prepare_queue[slot];
        if prepare.checksum != 0 && prepare.checksum != checksum {
            return PrepareOkResult::ChecksumMismatch;
        }

        // Set the checksum if not yet set.
        if prepare.checksum == 0 {
            prepare.checksum = checksum;
        }

        // Mark this replica's ack.
        let bit = 1u64 << replica;
        if self.ok_from_all_replicas[slot] & bit != 0 {
            return PrepareOkResult::DuplicateAck;
        }
        self.ok_from_all_replicas[slot] |= bit;
        prepare.acks_received += 1;

        // Check quorum.
        let quorum = u64::from(quorums(self.replica_count).replication);
        if self.ok_from_all_replicas[slot] >= (1u64 << quorum) - 1 {
            // Quorum reached! Stop prepare timeout if pipeline is drained.
            prepare.ok_quorum_received = true;
            if self.pipeline_queue.prepare_queue.len() <= 1 {
                self.prepare_timeout.stop();
            }
            return PrepareOkResult::QuorumReached { op, checksum };
        }

        PrepareOkResult::AckCounted
    }

    /// Commit a single op on the backup.
    ///
    /// Advances commit_min and calls execute on the state machine.
    ///
    /// # Panics
    /// Panics if `op` is not within `(commit_min, op]` (upstream asserts).
    ///
    /// Upstream: `src/vsr/replica.zig:4893` (`execute_op`).
    pub fn commit_op(&mut self, op: u64) {
        assert!(op <= self.op);
        assert!(op > self.commit_min);

        self.commit_min = op;
        self.advance_commit_max(op);

        // TODO(port): execute the operation on the state machine, build Reply.

        // Simulate the async prepare write completing: the header is already in
        // the journal (dirty); the on-disk copy accompanies commit.
        // DEVIATION: upstream marks a prepare written when `write_prepare`'s
        // callback fires (before commit, on equal disks). Without a message bus /
        // I/O pool we complete the write synchronously at commit.
        // TODO(port): remove when `write_prepare`/`write_prepare_callback` ports.
        if let Some(header) = self.journal.header_with_op(op).copied() {
            let slot = crate::journal::Journal::slot_for_op(op);
            self.journal.dirty.clear(slot);
            self.journal.prepare_inhabited[slot.index] = true;
            self.journal.prepare_checksums[slot.index] = header.checksum();
        }
    }

    /// Returns the number of prepares in the pipeline.
    #[must_use]
    pub fn pipeline_pending(&self) -> usize {
        self.pipeline_queue.prepare_queue.len()
    }

    /// Returns `true` if the pipeline is empty.
    #[must_use]
    pub fn pipeline_is_empty(&self) -> bool {
        self.pipeline_queue.prepare_queue.is_empty()
    }

    /// Pop the committed head from the pipeline (primary).
    ///
    /// Called after quorum is reached and the op is committed.
    pub fn pop_committed(&mut self) -> Option<PipelinePrepare> {
        if self.pipeline_queue.prepare_queue.is_empty() {
            return None;
        }
        let prepare = self.pipeline_queue.prepare_queue.remove(0);
        self.ok_from_all_replicas.remove(0);
        Some(prepare)
    }
}

// ---------------------------------------------------------------------------
// Commit pipeline — async state machine for committing prepares
// ---------------------------------------------------------------------------

impl Replica {
    /// Enter the commit pipeline. Called when commit_min < commit_max and
    /// there is at least one prepare ready to commit.
    ///
    /// # Panics
    ///
    /// Panics if `commit_dispatch_entered` is already `true` or if
    /// `commit_stage` is not `Idle`.
    ///
    /// Upstream: `src/vsr/replica.zig:4541` (`commit_dispatch_enter`).
    pub fn commit_dispatch_enter(&mut self) {
        assert!(!self.commit_dispatch_entered);
        assert_eq!(self.commit_stage, CommitStage::Idle);
        self.commit_dispatch();
    }

    /// Resume the commit pipeline after an async I/O completes.
    ///
    /// # Panics
    ///
    /// Panics if `commit_stage` is `Idle` or if `commit_dispatch_entered`
    /// is `false`.
    ///
    /// Upstream: `src/vsr/replica.zig:4546` (`commit_dispatch_resume`).
    pub fn commit_dispatch_resume(&mut self) {
        assert_ne!(self.commit_stage, CommitStage::Idle);
        assert!(self.commit_dispatch_entered);
        self.commit_dispatch_entered = false;
        self.commit_dispatch();
    }

    /// Cancel the commit pipeline (e.g. on state sync).
    ///
    /// # Panics
    ///
    /// Panics if `commit_stage` is `Idle` or if `commit_dispatch_entered`
    /// is `false`.
    ///
    /// Upstream: `src/vsr/replica.zig:4553` (`commit_dispatch_cancel`).
    pub fn commit_dispatch_cancel(&mut self) {
        assert_ne!(self.commit_stage, CommitStage::Idle);
        assert!(self.commit_dispatch_entered);
        self.commit_prepare = None;
        self.commit_stage = CommitStage::Idle;
        self.commit_dispatch_entered = false;
    }

    /// The main commit loop. Processes up to `VSR_CHECKPOINT_OPS` prepares
    /// per call, then returns.
    ///
    /// Upstream: `src/vsr/replica.zig:4374` (`commit_dispatch`).
    fn commit_dispatch(&mut self) {
        assert!(!self.commit_dispatch_entered);
        self.commit_dispatch_entered = true;

        loop {
            if self.commit_stage == CommitStage::Idle {
                self.commit_stage = CommitStage::Start;
                assert!(self.commit_prepare.is_none());
                let ready = self.commit_start();
                if !ready {
                    return; // async: pending
                }
            }

            if self.commit_stage == CommitStage::Start {
                self.commit_stage = CommitStage::CheckPrepare;
                if self.commit_prepare.is_none() {
                    break; // nothing to commit
                }
            }

            assert!(self.commit_prepare.is_some());

            if self.commit_stage == CommitStage::CheckPrepare {
                self.commit_stage = CommitStage::Prefetch;
                let ready = self.commit_prefetch();
                if !ready {
                    return;
                }
            }

            if self.commit_stage == CommitStage::Prefetch {
                self.commit_stage = CommitStage::Stall;
                let ready = self.commit_stall();
                if !ready {
                    return;
                }
            }

            if self.commit_stage == CommitStage::Stall {
                self.commit_stage = CommitStage::ReplySetup;
                let ready = self.commit_reply_setup();
                if !ready {
                    return;
                }
            }

            if self.commit_stage == CommitStage::ReplySetup {
                self.commit_stage = CommitStage::Execute;
                self.commit_execute();
            }

            if self.commit_stage == CommitStage::Execute {
                self.commit_stage = CommitStage::CheckpointDurable;
                let ready = self.commit_checkpoint_durable();
                if !ready {
                    return;
                }
            }

            if self.commit_stage == CommitStage::CheckpointDurable {
                self.commit_stage = CommitStage::Compact;
                let ready = self.commit_compact();
                if !ready {
                    return;
                }
            }

            if self.commit_stage == CommitStage::Compact {
                self.commit_stage = CommitStage::CheckpointData;
                let ready = self.commit_checkpoint_data();
                if !ready {
                    return;
                }
            }

            if self.commit_stage == CommitStage::CheckpointData {
                self.commit_stage = CommitStage::CheckpointSuperblock;
                let ready = self.commit_checkpoint_superblock();
                if !ready {
                    return;
                }
            }

            if self.commit_stage == CommitStage::CheckpointSuperblock {
                self.commit_stage = CommitStage::Idle;
                self.commit_finish();
            }

            assert!(self.commit_prepare.is_none());
            assert_eq!(self.commit_stage, CommitStage::Idle);
        }

        assert_eq!(self.commit_stage, CommitStage::CheckPrepare);
        assert!(self.commit_prepare.is_none());
        self.commit_stage = CommitStage::Idle;
        self.commit_dispatch_entered = false;
    }

    /// Stage: Start — get the next prepare to commit.
    ///
    /// On the primary in Normal status, pops from the pipeline.
    /// On backups, reads from journal (async — deferred to Phase 3).
    ///
    /// Upstream: `src/vsr/replica.zig:4565` (`commit_start`).
    fn commit_start(&mut self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::Start);
        assert!(self.commit_prepare.is_none());

        if self.status == Status::Normal && self.is_primary() {
            self.commit_start_pipeline();
            true
        } else {
            // TODO(port): commit_start_journal — read from journal (async).
            // For now, return ready (nothing to commit from journal).
            true
        }
    }

    /// Primary: take the head of the pipeline as the prepare to commit.
    ///
    /// Upstream: `src/vsr/replica.zig:4578` (`commit_start_pipeline`).
    fn commit_start_pipeline(&mut self) {
        assert_eq!(self.commit_stage, CommitStage::Start);
        assert!(self.commit_prepare.is_none());
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());

        let Some(head) = self.pipeline_queue.prepare_queue.first() else {
            return; // nothing in the pipeline
        };

        if !head.ok_quorum_received {
            // TODO(port): handle via on_prepare_timeout.
            return;
        }

        let count = self.ok_from_all_replicas.first().map_or(0, |bits| bits.count_ones() as usize);
        assert!(u64::from(self.quorum().replication) <= count as u64);
        assert!(count <= self.replica_count as usize);

        self.commit_prepare = Some(head.op);
    }

    /// Stage: Prefetch — load required data from LSM tree into memory.
    ///
    /// Upstream: `src/vsr/replica.zig:4715` (`commit_prefetch`).
    fn commit_prefetch(&self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::Prefetch);
        assert!(self.commit_prepare.is_some());
        // TODO(port): state_machine.prefetch — async I/O. For now, ready.
        true
    }

    /// Stage: Stall — primary backpressure to let backups catch up.
    ///
    /// Upstream: `src/vsr/replica.zig:4777` (`commit_stall`).
    fn commit_stall(&self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::Stall);
        // TODO(port): commit_stall_timeout — async backpressure. For now, ready.
        true
    }

    /// Stage: ReplySetup — ensure ClientReplies has a Write slot.
    ///
    /// Upstream: `src/vsr/replica.zig:4877` (`commit_reply_setup`).
    fn commit_reply_setup(&self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::ReplySetup);
        // TODO(port): client_replies.ready() — async. For now, ready.
        true
    }

    /// Stage: Execute — run state machine logic on the committed prepare.
    ///
    /// After execution on the primary, pops the committed prepare from the
    /// pipeline and the next request from the request queue.
    ///
    /// Upstream: `src/vsr/replica.zig:4893` (`commit_execute`),
    /// `src/vsr/replica.zig:5344` (`execute_op`).
    fn commit_execute(&mut self) {
        assert_eq!(self.commit_stage, CommitStage::Execute);
        let Some(op) = self.commit_prepare else {
            panic!("commit_execute requires commit_prepare");
        };
        assert_eq!(self.commit_min + 1, op);

        // Execute on state machine.
        // TODO(port): state_machine.execute(commit_prepare.body_used())
        self.commit_min = op;
        self.advance_commit_max(op);
        assert!(self.commit_min <= self.commit_max);

        if self.status == Status::Normal && self.is_primary() {
            assert_eq!(self.commit_min, self.commit_max);

            // Pop the committed prepare from the pipeline (primary only).
            let popped = self.pop_committed();
            assert!(popped.is_some_and(|p| p.op == op));

            // Pop the next request to prepare (if any).
            // TODO(port): pass full request message to primary_pipeline_prepare.
            self.pipeline_queue.request_queue.pop();
        }
    }

    /// Stage: CheckpointDurable — mark the checkpoint as durable.
    ///
    /// Upstream: `src/vsr/replica.zig:4967` (`commit_checkpoint_durable`).
    fn commit_checkpoint_durable(&self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::CheckpointDurable);
        // TODO(port): grid.checkpoint_durable — async. For now, ready.
        true
    }

    /// Stage: Compact — run one beat of LSM compaction.
    ///
    /// Upstream: `src/vsr/replica.zig:4943` (`commit_compact`).
    fn commit_compact(&self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::Compact);
        // TODO(port): state_machine.compact — async. For now, ready.
        true
    }

    /// Stage: CheckpointData — persist the current state to disk.
    ///
    /// Upstream: `src/vsr/replica.zig:4989` (`commit_checkpoint_data`).
    fn commit_checkpoint_data(&self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::CheckpointData);
        // TODO(port): checkpoint_data — async. For now, ready.
        true
    }

    /// Stage: CheckpointSuperblock — update the superblock.
    ///
    /// Upstream: `src/vsr/replica.zig` (`commit_checkpoint_superblock`).
    fn commit_checkpoint_superblock(&self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::CheckpointSuperblock);
        // TODO(port): superblock.checkpoint — async. For now, ready.
        true
    }

    /// Stage: Finish — cleanup after committing one prepare.
    ///
    /// Upstream: `src/vsr/replica.zig:5278` (`commit_finish`).
    fn commit_finish(&mut self) {
        assert_eq!(self.commit_stage, CommitStage::Idle);
        assert!(self.commit_prepare.is_some());
        // TODO(port): message_bus.unref(commit_prepare.message), commit_started = null.
        self.commit_prepare = None;
    }
}

// ---------------------------------------------------------------------------
// Fault detection — ping/pong, heartbeats, view changes
// ---------------------------------------------------------------------------

impl Replica {
    /// Called every tick to check the commit heartbeat fault detector.
    ///
    /// - **Primary, yellow/red**: resets commit_message_timeout, sends extra
    ///   Commit to re-smooth the interval.
    /// - **Backup, red**: sends ExitView to begin view change and re-signals
    ///   the fault detector to avoid repeated ExitView.
    ///
    /// # Panics
    ///
    /// Panics if `self.status` is not `Normal`.
    ///
    /// Upstream: `src/vsr/replica.zig:1591` (`tick_normal_heartbeat_fault`).
    pub fn tick_normal_heartbeat_fault(&mut self, now: u64) {
        assert_eq!(self.status, Status::Normal);

        let tardy = self.commit_fault.tardy(now);
        if tardy == Tardiness::Green {
            return; // Everything is fine!
        }

        if self.is_primary() {
            // See FaultDetector smoothing test for why resetting the timeout is
            // critical here.
            self.commit_message_timeout.reset(constants::COMMIT_MESSAGE_TIMEOUT);
            self.send_commit(now);
        } else if tardy == Tardiness::Yellow {
            // A slight delay which could be caused by a natural drop in the
            // load, so wait some more for a Commit message from the primary.
        } else {
            self.send_exit_view();
            self.commit_fault.signal(now);
        }
        assert_ne!(self.commit_fault.tardy(now), Tardiness::Red);
    }

    /// Send a Commit heartbeat to all replicas (primary only).
    ///
    /// Signals the fault detector even while abdicating, to maintain the
    /// invariant that a replica doesn't let commit_fault go red without action.
    ///
    /// # Panics
    ///
    /// Panics if the replica is not a Normal primary or if
    /// `commit_min != commit_max`.
    ///
    /// Upstream: `src/vsr/replica.zig:11342` (`send_commit`).
    pub fn send_commit(&mut self, now: u64) {
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());
        assert_eq!(self.commit_min, self.commit_max);

        self.commit_fault.signal(now);
        if self.primary_abdicating {
            assert!(self.primary_abdicate_timeout.active);
            return;
        }

        // TODO(port): broadcast Commit{commit_max, checkpoint_op, timestamp: now} to replicas.
        let _ = now;
    }

    /// Handle a ping message — reply with a pong.
    ///
    /// Upstream: `src/vsr/replica.zig:1849` (`on_ping`).
    ///
    /// DEVIATION: upstream extracts `ping_timestamp_monotonic` from the typed
    /// Ping header.  This stub accepts the raw Header; typed field access is
    /// deferred until the typed-header integration is complete.
    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn on_ping(&self, _header: &message_header::Header) -> PingReply {
        // TODO(port): extract ping_timestamp_monotonic from typed Ping header.
        PingReply { ping_timestamp_monotonic: 0, pong_timestamp_wall: 0 }
    }

    /// Handle a pong message — feed clock learning.
    ///
    /// Upstream: `src/vsr/replica.zig:1898` (`on_pong`).
    #[allow(clippy::unused_self)]
    pub fn on_pong(&mut self, _header: &message_header::Header) {
        // TODO(port): clock.learn(m0, t1, m2) with the three-clock exchange.
    }

    /// Handle a commit heartbeat from the primary (backup only).
    ///
    /// Updates heartbeat timestamp and signals the fault detector.
    ///
    /// Upstream: `src/vsr/replica.zig:2428` (`on_commit` heartbeat path).
    ///
    /// DEVIATION: upstream reads `header.timestamp` and `header.commit` from the
    /// typed Commit header.  This stub takes the values directly.
    pub fn on_commit_heartbeat(&mut self, view: u32, vrs_timestamp: u64, commit: u64, now: u64) {
        if self.is_primary() {
            return;
        }
        if self.status != Status::Normal {
            return;
        }
        if view != self.view {
            return;
        }

        // Fresh heartbeat: monotonically increasing VSR timestamp.
        if vrs_timestamp > self.heartbeat_timestamp {
            self.heartbeat_timestamp = vrs_timestamp;
            // Feed the fault detector with the monotonic clock time.
            self.commit_fault.signal(now);
            // Clear our own bit in exit_view_from_all_replicas (rescind EV).
            self.exit_view_from_all_replicas &= !(1u64 << self.replica_index);
        }

        self.advance_commit_max(commit);
    }

    /// Begin a view change by broadcasting ExitView.
    ///
    /// Upstream: `src/vsr/replica.zig:8762` (`send_exit_view`).
    pub fn send_exit_view(&mut self) {
        self.status = Status::ViewChange;
        // Set our own bit.
        self.exit_view_from_all_replicas |= 1u64 << self.replica_index;
        // Start the exit-view window timeout (5s) and message timeout (500ms).
        self.exit_view_window_timeout = Timeout::start(constants::EXIT_VIEW_WINDOW_TIMEOUT);
        self.exit_view_message_timeout = Timeout::start(constants::EXIT_VIEW_MESSAGE_TIMEOUT);
        // TODO(port): broadcast ExitView{view} to all replicas + self loopback.
    }

    /// Handle an ExitView message from another replica.
    ///
    /// Collects ExitView messages; when quorum is reached, transitions to
    /// view change status for the next view.
    ///
    /// Upstream: `src/vsr/replica.zig:2549` (`on_exit_view`).
    pub fn on_exit_view(&mut self, header: &message_header::Header) {
        if self.status != Status::Normal && self.status != Status::ViewChange {
            return;
        }
        if header.view < self.view {
            return;
        }

        let replica = header.replica;
        let bit = 1u64 << replica;
        self.exit_view_from_all_replicas |= bit;

        let quorum = u64::from(quorums(self.replica_count).view_change);
        if self.exit_view_from_all_replicas >= (1u64 << quorum) - 1 {
            // Quorum reached — transition to view change.
            let new_view = self.view + 1;
            self.transition_to_view_change_status(new_view);
        }
    }

    /// Transition to view change status.
    ///
    /// Upstream: `src/vsr/replica.zig` (`transition_to_view_change_status`).
    fn transition_to_view_change_status(&mut self, new_view: u32) {
        self.status = Status::ViewChange;
        self.view = new_view;
        self.log_view = new_view;
        self.exit_view_from_all_replicas = 0;
        self.pipeline_queue = PipelineQueue::default();
        self.ok_from_all_replicas.clear();
        self.view_change_status_timeout = Timeout::start(constants::VIEW_CHANGE_STATUS_TIMEOUT);
        // TODO(port): send JoinView to all replicas.
    }

    /// Called every tick — advances all timeouts and the fault detector.
    ///
    /// Upstream: `src/vsr/replica.zig:1532` (`tick`).
    pub fn tick(&mut self, now: u64) {
        if self.status == Status::Normal {
            self.tick_normal_heartbeat_fault(now);
        }

        if self.ping_timeout.tick() {
            self.on_ping_timeout();
        }
        if self.prepare_timeout.tick() {
            self.on_prepare_timeout();
        }
        if self.commit_message_timeout.tick() {
            self.on_commit_message_timeout(now);
        }
        if self.exit_view_message_timeout.tick() {
            self.on_exit_view_message_timeout();
        }
        if self.exit_view_window_timeout.tick() {
            self.on_exit_view_window_timeout();
        }
        if self.view_change_status_timeout.tick() {
            self.on_view_change_status_timeout();
        }
        if self.primary_abdicate_timeout.tick() {
            self.on_primary_abdicate_timeout();
        }
    }

    /// Timeout: broadcast a Ping to all replicas.
    ///
    /// Upstream: `src/vsr/replica.zig:3567` (`on_ping_timeout`).
    fn on_ping_timeout(&mut self) {
        self.ping_timeout.reset(constants::PING_TIMEOUT);
        // TODO(port): broadcast Ping with view_durable(), checkpoint_id/op, release info.
    }

    /// Timeout: the primary re-sends pending prepares or issues a Commit
    /// heartbeat.
    ///
    /// # Panics
    ///
    /// Panics if the replica is not a Normal primary.
    ///
    /// Upstream: `src/vsr/replica.zig:3608` (`on_prepare_timeout`).
    pub fn on_prepare_timeout(&mut self) {
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());
        self.prepare_timeout.reset(constants::PREPARE_TIMEOUT);
        if self.pipeline_queue.prepare_queue.is_empty() {
            // Nothing pending — nothing to re-send.
        } else {
            // TODO(port): find pipeline slot without a full prepare_ok quorum and
            // re-send the prepare (upstream: `primary_pipeline_pending` +
            // `on_repair` for the journal-write case).
        }
    }

    /// Timeout: the primary sends a Commit heartbeat at a fixed cadence.
    ///
    /// # Panics
    ///
    /// Panics if the replica is not a Normal primary or if
    /// `commit_min != commit_max`.
    ///
    /// Upstream: `src/vsr/replica.zig:3691` (`on_commit_message_timeout`).
    pub fn on_commit_message_timeout(&mut self, now: u64) {
        self.commit_message_timeout.reset(constants::COMMIT_MESSAGE_TIMEOUT);
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());
        assert_eq!(self.commit_min, self.commit_max);
        self.send_commit(now);
    }

    /// Timeout: reset the ExitView window, keeping only our own bit.
    ///
    /// # Panics
    ///
    /// Panics unless the status is `Normal` or `ViewChange` and some replica
    /// has been observed exiting the view.
    ///
    /// Upstream: `src/vsr/replica.zig:3701` (`on_exit_view_window_timeout`).
    pub fn on_exit_view_window_timeout(&mut self) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert_ne!(self.exit_view_from_all_replicas, 0);
        self.exit_view_window_timeout.stop();

        // Don't reset our own EV; it will be reset if/when we receive a heartbeat.
        let exit_view = self.exit_view_from_all_replicas & (1u64 << self.replica_index) != 0;
        self.exit_view_from_all_replicas = 0;
        if exit_view {
            self.exit_view_from_all_replicas = 1u64 << self.replica_index;
        }
    }

    /// Timeout: re-send ExitView if our bit is still set after the window.
    ///
    /// # Panics
    ///
    /// Panics unless the status is `Normal` or `ViewChange`.
    ///
    /// Upstream: `src/vsr/replica.zig:3715` (`on_exit_view_message_timeout`).
    pub fn on_exit_view_message_timeout(&mut self) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        self.exit_view_message_timeout.reset(constants::EXIT_VIEW_MESSAGE_TIMEOUT);

        if self.exit_view_from_all_replicas & (1u64 << self.replica_index) != 0 {
            self.send_exit_view();
        }
    }

    /// Timeout: in view change, re-signal ExitView to keep the view-change
    /// quorum alive.
    ///
    /// # Panics
    ///
    /// Panics unless the status is `ViewChange`.
    ///
    /// Upstream: `src/vsr/replica.zig:3727` (`on_view_change_status_timeout`).
    pub fn on_view_change_status_timeout(&mut self) {
        assert_eq!(self.status, Status::ViewChange);
        self.view_change_status_timeout.reset(constants::VIEW_CHANGE_STATUS_TIMEOUT);
        self.send_exit_view();
    }

    /// Timeout: the primary starts abdicating after `PRIMARY_ABDICATE_TIMEOUT`
    /// without a prepare_ok quorum.
    ///
    /// # Panics
    ///
    /// Panics if the replica is not a Normal primary.
    ///
    /// Upstream: `src/vsr/replica.zig:3678` (`on_primary_abdicate_timeout`).
    pub fn on_primary_abdicate_timeout(&mut self) {
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());
        self.primary_abdicate_timeout.reset(constants::PRIMARY_ABDICATE_TIMEOUT);
        self.primary_abdicating = true;
    }
}

/// Reply to a ping message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PingReply {
    pub ping_timestamp_monotonic: u64,
    pub pong_timestamp_wall: u64,
}

// ---------------------------------------------------------------------------
// Prepare pipeline — result types
// ---------------------------------------------------------------------------

/// Reason a prepare was rejected by the primary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepareReject {
    NotPrimary,
    NotNormal,
    BehindCommit,
    PipelineFull,
}

/// Result of processing a prepare_ok message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepareOkResult {
    /// Quorum reached — the prepare can be committed.
    QuorumReached { op: u64, checksum: u128 },
    /// Ack counted, quorum not yet reached.
    AckCounted,
    /// Duplicate ack from same replica — ignored.
    DuplicateAck,
    /// Op not found in pipeline.
    UnknownOp,
    /// Checksum mismatch — stale or conflicting prepare.
    ChecksumMismatch,
    /// Not primary or not normal status.
    Ignored,
}

// ---------------------------------------------------------------------------
// Pipeline — queue (primary) and cache (backup/view-change)
// ---------------------------------------------------------------------------

/// Pipeline queue used by the primary in normal status.
///
/// Two ring buffers: `prepare_queue` (inflight prepares) and
/// `request_queue` (accepted client requests not yet preparing).
///
/// Upstream: `src/vsr/replica.zig:12094` (`PipelineQueue`).
#[derive(Clone, Debug, Default)]
pub struct PipelineQueue {
    /// Inflight prepares in order (ring buffer of up to `PIPELINE_PREPARE_QUEUE_MAX`).
    pub prepare_queue: Vec<PipelinePrepare>,
    /// Accepted requests not yet preparing.
    pub request_queue: Vec<PipelineRequest>,
}

/// A prepare in the pipeline.
#[derive(Clone, Copy, Debug)]
pub struct PipelinePrepare {
    pub op: u64,
    pub checksum: u128,
    /// Number of prepare_ok responses received.
    pub acks_received: u16,
    /// Whether a quorum of prepare_ok messages has been received.
    pub ok_quorum_received: bool,
}

/// A client request queued on the primary.
#[derive(Clone, Copy, Debug)]
pub struct PipelineRequest {
    pub client: u128,
    pub request: u64,
}

/// Pipeline cache used by backups and during view changes.
///
/// A fixed array indexed by `op % capacity`.
///
/// Upstream: `src/vsr/replica.zig:12324` (`PipelineCache`).
#[derive(Clone, Debug, Default)]
pub struct PipelineCache {
    entries: Vec<Option<PipelinePrepare>>,
}

impl PipelineCache {
    /// Create an empty cache with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);
        Self { entries }
    }

    /// Look up a cached prepare by op + checksum.
    #[must_use]
    pub fn find(&self, op: u64, checksum: u128) -> Option<&PipelinePrepare> {
        #[allow(clippy::cast_possible_truncation)]
        let index = (op as usize) % self.entries.len();
        self.entries[index].as_ref().filter(|p| p.op == op && p.checksum == checksum)
    }

    /// Insert a prepare into the cache, returning the evicted entry (if any).
    pub fn insert(&mut self, prepare: PipelinePrepare) -> Option<PipelinePrepare> {
        #[allow(clippy::cast_possible_truncation)]
        let index = (prepare.op as usize) % self.entries.len();
        self.entries[index].replace(prepare)
    }
}

// ---------------------------------------------------------------------------
// Message dispatch
// ---------------------------------------------------------------------------

impl Replica {
    /// Main message dispatch entry point.
    ///
    /// Validates the header, then dispatches to the appropriate handler
    /// based on the command type.
    ///
    /// Upstream: `src/vsr/replica.zig:1729` (`on_message`).
    pub fn on_message(&mut self, header: &message_header::Header) {
        // Validate cluster ID.
        if header.cluster != self.cluster {
            return; // Drop message from wrong cluster.
        }

        // Validate checksum (upstream verifies `header.valid_checksum()`).

        match header.command {
            Command::Request => self.on_request(header),
            Command::Prepare => self.on_prepare(header),
            Command::Commit => self.on_commit(header),
            // TODO(port): remaining message handlers.
            Command::PrepareOk
            | Command::Reply
            | Command::Ping
            | Command::Pong
            | Command::PingClient
            | Command::ExitView
            | Command::JoinView
            | Command::View
            | Command::Headers
            | Command::GetView
            | Command::GetHeaders
            | Command::GetPrepare
            | Command::GetReply
            | Command::GetBlocks
            | Command::Deprecated12
            | Command::Deprecated21
            | Command::Deprecated22
            | Command::Deprecated23
            | Command::Reserved
            | Command::PongClient
            | Command::Eviction
            | Command::Block => {}
        }
    }

    /// Handle a client request (primary only, normal status).
    ///
    /// Upstream: `src/vsr/replica.zig:1944` (`on_request`).
    pub fn on_request(&mut self, header: &message_header::Header) {
        if !self.is_primary() {
            return; // Only primary handles requests.
        }
        if self.status != Status::Normal {
            return;
        }

        // Upstream: if the prepare queue is full, queue the request for later.
        // For now, log that we received the request.
        let _ = header;
        // TODO(port): primary_pipeline_prepare — begin preparing the request.
    }

    /// Handle a prepare message from the primary.
    ///
    /// Upstream: `src/vsr/replica.zig:2021` (`on_prepare`).
    pub fn on_prepare(&mut self, header: &message_header::Header) {
        // Upstream: verify cluster, view, and op range.
        if header.view < self.view {
            return; // Stale prepare.
        }

        if self.is_primary() {
            return; // Primary doesn't process its own prepares via on_prepare.
        }

        // Backup path:
        // 1. If op > commit_min and journal lacks the prepare → replicate.
        // 2. If stale → on_repair.
        // 3. Advance commit_max from header's commit field.
        // 4. Cache in pipeline cache.
        // 5. Advance op, write to journal, send prepare_ok.

        // TODO(port): full backup prepare processing.
        let _ = header;
    }

    /// Handle a commit message from the primary (backup only).
    ///
    /// Upstream: `src/vsr/replica.zig:2396` (`on_commit`).
    pub fn on_commit(&mut self, header: &message_header::Header) {
        if self.is_primary() {
            return; // Primary doesn't receive commit messages.
        }
        if self.status != Status::Normal {
            return;
        }
        if header.view != self.view {
            return; // Stale commit.
        }

        // Upstream: advance commit_max, update heartbeat timestamp, commit journal.
        // TODO(port): advance_commit_max and commit_journal.
        let _ = header;
    }

    /// Returns `true` if we should ignore this request message.
    ///
    /// Upstream: `src/vsr/replica.zig:1930` (`ignore_request_message`).
    #[must_use]
    pub fn ignore_request_message(&self, header: &message_header::Header) -> bool {
        // Ignore if not from the primary's current view.
        header.view != self.view || !self.is_primary() || self.status != Status::Normal
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn quorums_1_replica() {
        let q = quorums(1);
        assert_eq!(q.replication, 1);
        assert_eq!(q.view_change, 1);
        assert_eq!(q.nack_prepare, 0);
        assert_eq!(q.majority, 1);
    }

    #[test]
    fn quorums_3_replicas() {
        let q = quorums(3);
        assert_eq!(q.replication, 2);
        assert_eq!(q.view_change, 2);
        assert_eq!(q.nack_prepare, 1);
        assert_eq!(q.majority, 2);
    }

    #[test]
    fn quorums_5_replicas() {
        let q = quorums(5);
        assert_eq!(q.replication, 3);
        assert_eq!(q.view_change, 3);
        assert_eq!(q.nack_prepare, 2);
        assert_eq!(q.majority, 3);
    }

    #[test]
    fn quorums_6_replicas() {
        let q = quorums(6);
        assert_eq!(q.replication, 4);
        assert_eq!(q.view_change, 4);
        assert_eq!(q.nack_prepare, 3);
        assert_eq!(q.majority, 4);
    }

    #[test]
    fn primary_index_wraps() {
        assert_eq!(Replica::primary_index_for_view(0, 3), 0);
        assert_eq!(Replica::primary_index_for_view(1, 3), 1);
        assert_eq!(Replica::primary_index_for_view(2, 3), 2);
        assert_eq!(Replica::primary_index_for_view(3, 3), 0);
        assert_eq!(Replica::primary_index_for_view(4, 3), 1);
    }

    #[test]
    fn is_primary() {
        let r = Replica::new(0, 0, 3);
        assert!(r.is_primary());

        let r = Replica::new(0, 1, 3);
        assert!(!r.is_primary());
    }

    #[test]
    fn timeout_tick() {
        let mut t = Timeout::start(3);
        assert!(!t.tick()); // 2 remaining
        assert!(!t.tick()); // 1 remaining
        assert!(t.tick()); // fires
        assert!(!t.active);
    }

    #[test]
    fn timeout_stop() {
        let mut t = Timeout::start(3);
        t.stop();
        assert!(!t.tick());
    }

    #[test]
    fn timeout_reset() {
        let mut t = Timeout::start(3);
        t.tick();
        t.tick();
        t.reset(5);
        assert!(t.active);
        assert_eq!(t.ticks, 5);
    }

    #[test]
    fn replica_construction() {
        let r = Replica::new(0xDEAD, 0, 3);
        assert_eq!(r.cluster, 0xDEAD);
        assert_eq!(r.replica_index, 0);
        assert_eq!(r.replica_count, 3);
        assert_eq!(r.status, Status::Recovering);
        assert_eq!(r.op, 0);
        assert_eq!(r.commit_min, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn replica_index_out_of_range() {
        let _ = Replica::new(0, 3, 3);
    }

    #[test]
    fn pipeline_cache_insert_and_find() {
        let mut cache = PipelineCache::new(4);
        let p =
            PipelinePrepare { op: 10, checksum: 0xAB, acks_received: 0, ok_quorum_received: false };
        assert!(cache.insert(p).is_none());
        assert!(cache.find(10, 0xAB).is_some());
        assert!(cache.find(10, 0xCD).is_none());
        assert!(cache.find(11, 0xAB).is_none());
    }

    #[test]
    fn pipeline_cache_eviction() {
        let mut cache = PipelineCache::new(2);
        let p1 =
            PipelinePrepare { op: 0, checksum: 1, acks_received: 0, ok_quorum_received: false };
        let p2 =
            PipelinePrepare { op: 2, checksum: 2, acks_received: 0, ok_quorum_received: false };
        // op=0 maps to index 0, op=2 maps to index 0 (2 % 2 == 0).
        assert!(cache.insert(p1).is_none());
        let evicted = cache.insert(p2);
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().op, 0);
        assert!(cache.find(0, 1).is_none());
        assert!(cache.find(2, 2).is_some());
    }

    #[test]
    fn on_message_wrong_cluster() {
        let mut r = Replica::new(0xDEAD, 0, 3);
        r.status = Status::Normal;
        let mut header = message_header::Header::empty();
        header.cluster = 0xBEEF;
        header.command = Command::Request;
        header.set_checksum();
        // Should silently drop — no panic, no state change.
        r.on_message(&header);
        assert_eq!(r.status, Status::Normal);
    }

    #[test]
    fn on_request_ignored_on_backup() {
        let mut r = Replica::new(0, 1, 3); // replica 1, not primary
        r.status = Status::Normal;
        let mut header = message_header::Header::empty();
        header.cluster = 0;
        header.command = Command::Request;
        header.view = 0;
        header.set_checksum();
        r.on_request(&header);
        // No state change — backup ignores request.
    }

    #[test]
    fn advance_commit_max_monotonic() {
        let mut r = Replica::new(0, 0, 3);
        r.advance_commit_max(5);
        assert_eq!(r.commit_max, 5);
        r.advance_commit_max(3); // Stale — should not decrease.
        assert_eq!(r.commit_max, 5);
        r.advance_commit_max(10);
        assert_eq!(r.commit_max, 10);
    }

    #[test]
    fn primary_pipeline_prepare_basic() {
        let mut r = Replica::new(0, 0, 3); // primary
        r.status = Status::Normal;
        let op = r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();
        assert_eq!(op, 1);
        assert_eq!(r.op, 1);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 1);
        assert!(r.prepare_timestamp > 0);
    }

    #[test]
    fn primary_pipeline_prepare_rejects_backup() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let result = r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64);
        assert_eq!(result, Err(PrepareReject::NotPrimary));
    }

    #[test]
    fn primary_pipeline_prepare_chain() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let op1 = r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();
        let op2 = r.primary_pipeline_prepare(2, 101, crate::Operation(6), 64).unwrap();
        assert_eq!(op1, 1);
        assert_eq!(op2, 2);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 2);
        // Timestamps must be strictly increasing.
        assert!(r.prepare_timestamp > 0);
    }

    #[test]
    fn on_prepare_ok_quorum_3() {
        let mut r = Replica::new(0, 0, 3); // primary, quorum=2
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();
        let checksum = r.journal.header_with_op(1).unwrap().checksum();

        // Replica 1 acks.
        let result = r.on_prepare_ok(1, checksum, 1);
        assert_eq!(result, PrepareOkResult::AckCounted);

        // Replica 2 acks → quorum reached.
        let result = r.on_prepare_ok(1, checksum, 2);
        assert!(matches!(result, PrepareOkResult::QuorumReached { op: 1, .. }));
    }

    #[test]
    fn on_prepare_ok_duplicate_ignored() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();
        let checksum = r.journal.header_with_op(1).unwrap().checksum();

        let _ = r.on_prepare_ok(1, checksum, 1);
        let result = r.on_prepare_ok(1, checksum, 1); // duplicate
        assert_eq!(result, PrepareOkResult::DuplicateAck);
    }

    #[test]
    fn commit_op_advances_commit_min() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.op = 5;
        r.commit_op(3);
        assert_eq!(r.commit_min, 3);
        assert_eq!(r.commit_max, 3);
        r.commit_op(5);
        assert_eq!(r.commit_min, 5);
        assert_eq!(r.commit_max, 5);
    }

    // ── Journal wiring ───────────────────────────────────────────────────

    #[test]
    fn primary_pipeline_prepare_wires_journal_chain() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();
        r.primary_pipeline_prepare(2, 101, crate::Operation(6), 64).unwrap();

        // Op 0 was never prepared — the journal ring still holds a reserved header.
        assert_eq!(r.journal.header_with_op(0), None);

        let h1 = *r.journal.header_with_op(1).unwrap();
        assert_eq!(h1.op, 1);
        assert_eq!(h1.commit, 0);
        assert_eq!(h1.parent, 0); // first entry — hash chain starts at 0
        assert_eq!(h1.view, 0);
        assert_eq!(r.pipeline_queue.prepare_queue[0].checksum, h1.checksum());

        // Dirty until the prepare write "completes":
        assert!(r.journal.dirty.bit(crate::journal::Journal::slot_for_op(1)));

        let h2 = *r.journal.header_with_op(2).unwrap();
        assert_eq!(h2.op, 2);
        assert_eq!(h2.parent, h1.checksum()); // hash chain links op 2 → op 1
        assert_eq!(h2.commit, 0); // nothing committed yet
        assert_eq!(r.pipeline_queue.prepare_queue[1].checksum, h2.checksum());

        assert_eq!(r.journal.op_maximum(), 2);
    }

    #[test]
    fn commit_op_marks_journal_prepare_written() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();
        let header = *r.journal.header_with_op(1).unwrap();
        let slot = crate::journal::Journal::slot_for_op(1);

        // Not yet written: dirty + uninhabited.
        assert!(r.journal.dirty.bit(slot));
        assert!(!r.journal.has_prepare(&header));

        r.commit_op(1);

        assert!(!r.journal.dirty.bit(slot));
        assert!(r.journal.prepare_inhabited[slot.index]);
        assert_eq!(r.journal.prepare_checksums[slot.index], header.checksum());
        assert!(r.journal.has_prepare(&header));
    }

    #[test]
    #[should_panic = "operation != .reserved"]
    fn primary_pipeline_prepare_rejects_reserved_operation() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let _ = r.primary_pipeline_prepare(1, 100, crate::Operation::RESERVED, 64);
    }

    #[test]
    fn pop_committed_drains_pipeline() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();
        r.primary_pipeline_prepare(2, 101, crate::Operation(6), 64).unwrap();
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 2);

        let popped = r.pop_committed();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().op, 1);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 1);

        let popped = r.pop_committed();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().op, 2);
        assert!(r.pipeline_queue.prepare_queue.is_empty());

        assert!(r.pop_committed().is_none());
    }

    // ── FaultDetector tests ──────────────────────────────────────────────

    #[test]
    fn fault_detector_green_when_no_signal() {
        let fd = FaultDetector::new(0, 100, 2_000);
        assert_eq!(fd.tardy(1000), Tardiness::Green);
    }

    #[test]
    fn fault_detector_converges_to_interval() {
        let mut fd = FaultDetector::new(0, 100, 2_000);

        // EWMA starts at interval_max (2000).
        assert_eq!(fd.interval_ewma, 2_000);

        // Signal at t=500: elapsed=500, clamped=500, ewma = (2000*4+500)/5 = 1700
        fd.signal(500);
        assert_eq!(fd.interval_ewma, 1_700);

        // Signal at t=1000: elapsed=500, ewma = (1700*4+500)/5 = 1460
        fd.signal(1_000);
        assert_eq!(fd.interval_ewma, 1_460);

        // After many signals, converges to 500.
        for i in 2..20 {
            fd.signal(i * 500);
        }
        assert!(fd.interval_ewma < 600);
    }

    #[test]
    fn fault_detector_green_yellow_red() {
        let mut fd = FaultDetector::new(0, 100, 2_000);

        // Signal at t=500, ewma starts at 2000, first sample: (2000*4+500)/5=1700
        fd.signal(500);

        // Green: elapsed(100)*2=200 ≤ ewma(1700)*3=5100
        assert_eq!(fd.tardy(600), Tardiness::Green);

        // Yellow: elapsed(2900)*2=5800 > 5100, but elapsed(2900) ≤ 5100
        assert_eq!(fd.tardy(3_400), Tardiness::Yellow);

        // Red: elapsed(6000)*2=12000 > ewma(1700)*3=5100, and elapsed(6000) > 5100
        assert_eq!(fd.tardy(6_500), Tardiness::Red);
    }

    #[test]
    fn fault_detector_signal_resets_tardiness() {
        let mut fd = FaultDetector::new(0, 100, 2_000);
        fd.signal(500); // ewma = (2000*4+500)/5 = 1700

        // Red at t=6500
        assert_eq!(fd.tardy(6_500), Tardiness::Red);

        // Signal resets — now green again
        fd.signal(6_500);
        assert_eq!(fd.tardy(6_500), Tardiness::Green);
    }

    #[test]
    fn on_commit_heartbeat_advances_commit_max() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;

        r.on_commit_heartbeat(r.view, 100, 42, 100);
        assert_eq!(r.commit_max, 42);
        assert_eq!(r.heartbeat_timestamp, 100);
    }

    #[test]
    fn on_commit_heartbeat_rejects_wrong_view() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.on_commit_heartbeat(r.view + 1, 100, 42, 100);
        assert_eq!(r.commit_max, 0);
    }

    #[test]
    fn on_commit_heartbeat_rejects_stale_timestamp() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.on_commit_heartbeat(r.view, 100, 42, 100);
        // Stale VSR timestamp (50 < 100) — heartbeat_timestamp stays, but commit advances.
        r.on_commit_heartbeat(r.view, 50, 99, 200);
        assert_eq!(r.heartbeat_timestamp, 100);
        assert_eq!(r.commit_max, 99); // commit=99 > commit_max=42
    }

    #[test]
    fn on_commit_heartbeat_rejects_primary() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.on_commit_heartbeat(r.view, 100, 42, 100);
        assert_eq!(r.commit_max, 0);
    }

    #[test]
    fn on_commit_heartbeat_signals_fault_detector() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.on_commit_heartbeat(r.view, 100, 42, 500);
        assert_eq!(r.commit_fault.tardy(500), Tardiness::Green);
    }

    #[test]
    fn send_exit_view_sets_status_and_bit() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        assert_eq!(r.status, Status::Normal);

        r.send_exit_view();
        assert_eq!(r.status, Status::ViewChange);
        assert_ne!(r.exit_view_from_all_replicas & (1u64 << 1), 0);
    }

    #[test]
    fn on_exit_view_collects_quorum() {
        let mut r = Replica::new(0, 0, 3);

        // Primary already in view change with own bit set.
        r.status = Status::ViewChange;
        r.exit_view_from_all_replicas = 1u64 << 0;

        // Simulate ExitView from replica 1.
        let header =
            message_header::Header { view: r.view, replica: 1, ..message_header::Header::empty() };
        r.on_exit_view(&header);
        assert_eq!(r.status, Status::ViewChange);
        assert_eq!(r.view, 1);

        // Simulate ExitView from replica 2 in the new view — already transitioned.
        let header =
            message_header::Header { view: r.view, replica: 2, ..message_header::Header::empty() };
        r.on_exit_view(&header);
        assert_eq!(r.status, Status::ViewChange);
    }

    #[test]
    fn on_exit_view_ignores_old_view() {
        let mut r = Replica::new(0, 0, 3);
        r.view = 5;

        let header =
            message_header::Header { view: 3, replica: 1, ..message_header::Header::empty() };
        r.on_exit_view(&header);
        assert_eq!(r.exit_view_from_all_replicas, 0);
    }

    #[test]
    fn tick_normal_heartbeat_fault_backup_red_triggers_exit() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        // Simulate: one signal long ago, now very stale.
        r.commit_fault.signal(0);
        r.tick_normal_heartbeat_fault(10_000);
        assert_eq!(r.status, Status::ViewChange);
    }

    #[test]
    fn tick_normal_heartbeat_fault_backup_red_resignals() {
        // The backup must re-signal after ExitView, or it would be red again next tick.
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.commit_fault.signal(0);
        r.tick_normal_heartbeat_fault(10_000);
        assert_eq!(r.status, Status::ViewChange);
        assert_ne!(r.commit_fault.tardy(10_000), Tardiness::Red);
    }

    #[test]
    fn tick_normal_heartbeat_fault_primary_yellow_resends_commit() {
        // Primary that stops receiving prepare_ok feed becomes yellow; it re-sends
        // an extra Commit and resets commit_message_timeout to re-smooth.
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        // Two signals 500ms apart, then silence to reach yellow.
        r.commit_fault.signal(0);
        r.commit_fault.signal(500);
        r.commit_message_timeout = Timeout::start(100);
        r.tick_normal_heartbeat_fault(5_000); // yellow (elapsed 4500 > 1.5×ewma≈1425? )
        // After send_commit, fault detector is re-signaled at now → green.
        assert_eq!(r.commit_fault.tardy(5_000), Tardiness::Green);
        assert!(r.commit_message_timeout.active);
    }

    #[test]
    fn tick_dispatches_commit_message_timeout_to_send_commit() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.commit_message_timeout = Timeout::start(1);

        r.tick(100);
        assert_eq!(r.commit_message_timeout.ticks, constants::COMMIT_MESSAGE_TIMEOUT);
        assert!(r.commit_message_timeout.active);
        // send_commit signals the fault detector → green at the same timestamp.
        assert_eq!(r.commit_fault.tardy(100), Tardiness::Green);
    }

    #[test]
    fn tick_dispatches_primary_abdicate_timeout() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_abdicate_timeout = Timeout::start(1);
        assert!(!r.primary_abdicating);

        r.tick(0);
        assert!(r.primary_abdicating);
        assert_eq!(r.primary_abdicate_timeout.ticks, constants::PRIMARY_ABDICATE_TIMEOUT);
        assert!(r.primary_abdicate_timeout.active);
    }

    #[test]
    fn tick_dispatches_ping_timeout() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.ping_timeout = Timeout::start(1);

        r.tick(0);
        assert_eq!(r.ping_timeout.ticks, constants::PING_TIMEOUT);
    }

    #[test]
    fn on_exit_view_window_timeout_keeps_own_bit() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.exit_view_from_all_replicas = 0b111; // all three replicas saw ExitView
        r.exit_view_window_timeout = Timeout::start(1);

        r.tick(0);
        assert_eq!(r.exit_view_from_all_replicas, 0b001);
        assert!(!r.exit_view_window_timeout.active);
    }

    #[test]
    fn on_exit_view_window_timeout_clears_if_own_bit_unset() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::ViewChange;
        r.exit_view_from_all_replicas = 0b110; // others only — own bit not set

        r.on_exit_view_window_timeout();
        assert_eq!(r.exit_view_from_all_replicas, 0);
    }

    #[test]
    fn on_exit_view_message_timeout_resends_exit_view() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::ViewChange;
        r.exit_view_from_all_replicas = 1u64 << 1;

        r.on_exit_view_message_timeout();
        // send_exit_view reaffirms the ViewChange status and restarts the window.
        assert_eq!(r.status, Status::ViewChange);
        assert_ne!(r.exit_view_from_all_replicas & (1u64 << 1), 0);
    }

    #[test]
    fn on_view_change_status_timeout_resignals_exit_view() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::ViewChange;
        r.exit_view_from_all_replicas = 0; // cleared after transition

        r.on_view_change_status_timeout();
        assert_eq!(r.view_change_status_timeout.ticks, constants::VIEW_CHANGE_STATUS_TIMEOUT);
        assert_eq!(r.status, Status::ViewChange);
        assert_ne!(r.exit_view_from_all_replicas & (1u64 << 1), 0);
    }

    // ── Commit pipeline tests ────────────────────────────────────────────

    #[test]
    fn commit_dispatch_empty_pipeline_noop() {
        // Primary with nothing to commit: pipeline empty, no-op.
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;

        r.commit_dispatch_enter();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert!(!r.commit_dispatch_entered);
        assert!(r.commit_prepare.is_none());
        assert_eq!(r.commit_min, 0);
    }

    #[test]
    fn commit_dispatch_pipeline_without_quorum_noop() {
        // Prepare in pipeline but no quorum yet: no-op.
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0).unwrap();

        r.commit_dispatch_enter();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert_eq!(r.commit_min, 0);
    }

    #[test]
    fn commit_dispatch_pipeline_with_quorum_executes() {
        // Primary with a quorum'd prepare: commit executes fully.
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let op = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0).unwrap();

        // Quorum = 2 (replica_count 3). Self + one backup.
        let checksum = r.journal.header_with_op(op).unwrap().checksum();
        r.on_prepare_ok(op, checksum, 0); // self
        r.on_prepare_ok(op, checksum, 1); // replica 1 → quorum

        assert!(r.pipeline_queue.prepare_queue[0].ok_quorum_received);
        assert_eq!(r.commit_min, 0);
        assert_eq!(r.commit_max, 0);

        r.commit_dispatch_enter();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert_eq!(r.commit_min, op);
        assert_eq!(r.commit_max, op);
        assert!(r.commit_prepare.is_none());
        assert!(r.pipeline_queue.prepare_queue.is_empty());
    }

    #[test]
    fn commit_dispatch_commits_multiple_prepares() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;

        let op1 = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0).unwrap();
        let op2 = r.primary_pipeline_prepare(2, 2, crate::Operation::NOOP, 0).unwrap();
        let op3 = r.primary_pipeline_prepare(3, 3, crate::Operation::NOOP, 0).unwrap();

        // Quorum all three.
        for op in [op1, op2, op3] {
            let checksum = r.journal.header_with_op(op).unwrap().checksum();
            r.on_prepare_ok(op, checksum, 0);
            r.on_prepare_ok(op, checksum, 1);
        }

        r.commit_dispatch_enter();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert_eq!(r.commit_min, op3);
        assert!(r.pipeline_queue.prepare_queue.is_empty());
    }

    #[test]
    fn commit_dispatch_backup_does_not_commit() {
        // Backup in Normal status: no pipeline, nothing to commit.
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.commit_min = 0;
        r.commit_max = 1; // knows about a commit but has no prepare

        // commit_start_journal stub: no prepare available → ready, nothing to commit.
        r.commit_dispatch_enter();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert_eq!(r.commit_min, 0);
    }

    #[test]
    fn commit_dispatch_cancel_resets_state() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let op = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0).unwrap();
        let checksum = r.journal.header_with_op(op).unwrap().checksum();
        r.on_prepare_ok(op, checksum, 0);
        r.on_prepare_ok(op, checksum, 1);

        // Enter dispatch (sets commit_dispatch_entered via internal state).
        r.commit_dispatch_enter();
        // After the loop, state is reset already.

        // Manually simulate being mid-commit to test cancel.
        r.commit_stage = CommitStage::Execute;
        r.commit_prepare = Some(op);
        r.commit_dispatch_entered = true;
        r.commit_dispatch_cancel();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert!(!r.commit_dispatch_entered);
        assert!(r.commit_prepare.is_none());
    }
}
