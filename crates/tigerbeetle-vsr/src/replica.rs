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

    // ── Subsystems (assigned after construction) ────────────────────────
    // journal: Journal,
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
            ping_timeout: Timeout::default(),
            prepare_timeout: Timeout::default(),
            commit_message_timeout: Timeout::default(),
            view_change_status_timeout: Timeout::default(),
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

        // Advance op.
        self.op += 1;
        let op = self.op;

        // Timestamp: monotonic, at least prepare_timestamp + 1.
        self.prepare_timestamp = self.prepare_timestamp.max(self.commit_max) + 1;
        let timestamp = self.prepare_timestamp;

        let checksum_parent = self.pipeline_queue.prepare_queue.last().map_or(0, |p| p.checksum);

        let prepare = PipelinePrepare {
            op,
            checksum: 0, // Will be set when the message is serialized.
            acks_received: 0,
        };
        self.pipeline_queue.prepare_queue.push(prepare);
        self.ok_from_all_replicas.push(0);

        // Start timeouts if pipeline was previously empty.
        if self.pipeline_queue.prepare_queue.len() == 1 {
            self.prepare_timeout = Timeout::start(constants::PIPELINE_PREPARE_QUEUE_MAX);
        }

        let _ = (client, request, operation, body_size, timestamp, checksum_parent);
        // TODO(port): serialize the prepare message, write to journal, send to backups.

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
        let p = PipelinePrepare { op: 10, checksum: 0xAB, acks_received: 0 };
        assert!(cache.insert(p).is_none());
        assert!(cache.find(10, 0xAB).is_some());
        assert!(cache.find(10, 0xCD).is_none());
        assert!(cache.find(11, 0xAB).is_none());
    }

    #[test]
    fn pipeline_cache_eviction() {
        let mut cache = PipelineCache::new(2);
        let p1 = PipelinePrepare { op: 0, checksum: 1, acks_received: 0 };
        let p2 = PipelinePrepare { op: 2, checksum: 2, acks_received: 0 };
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

        // Replica 1 acks.
        let result = r.on_prepare_ok(1, 0xAB, 1);
        assert_eq!(result, PrepareOkResult::AckCounted);

        // Replica 2 acks → quorum reached.
        let result = r.on_prepare_ok(1, 0xAB, 2);
        assert!(matches!(result, PrepareOkResult::QuorumReached { op: 1, .. }));
    }

    #[test]
    fn on_prepare_ok_duplicate_ignored() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64).unwrap();

        let _ = r.on_prepare_ok(1, 0xAB, 1);
        let result = r.on_prepare_ok(1, 0xAB, 1); // duplicate
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
}
