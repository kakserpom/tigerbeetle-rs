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
use tigerbeetle_core::stdx::Instant;
use tigerbeetle_core::stdx::prng::Prng;

use crate::command::Command;
use crate::grid::{Event, Grid, ReadBlockResult, ReadOptions};
use crate::message_header;
use crate::message_header::{GetBlocks, TypedHeader};
use crate::repair_budget::{RepairBudgetGrid, RepairBudgetOptions};
use crate::storage::MemoryStorage;

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

/// One in-flight `GetBlocks` serve read on behalf of a remote replica
/// (upstream `replica.grid_reads[i]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridServeRead {
    pub token: u32,
    pub destination: u16,
    pub address: u64,
    pub checksum: u128,
}

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
    /// The timestamp of the most recently executed prepare (upstream
    /// `state_machine.commit_timestamp`; kept on the replica until the state
    /// machine is wired into `commit_execute`).
    pub commit_timestamp: u64,
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
    /// The grid (block storage): cache, stash, free set, read/write IOPS.
    ///
    /// `None` until [`Self::mount_grid`] attaches an opened grid — upstream owns
    /// the grid from construction and runs it once the superblock is open.
    ///
    /// DEVIATION: upstream constructs the grid over a real data file and opens it
    /// from the superblock. The sans-IO replica carries no superblock, so the
    /// owner mounts an externally-opened grid (with an attached
    /// [`crate::grid::SuperBlockView`]) before grid repairs can run.
    ///
    /// Just like upstream's `grid.callback == .cancel` gate, `None` short-circuits
    /// all grid repair paths (`on_get_blocks`, `on_block`, `send_get_blocks`).
    pub grid: Option<Grid>,
    /// The storage backing [`Self::grid`].
    ///
    /// DEVIATION: upstream owns one `Storage` (`superblock.storage`) shared by
    /// WAL and grid I/O. The sans-IO replica owns a concrete in-memory storage
    /// for the grid; the grid's APIs take `&mut dyn Storage`, so both are handed
    /// out together via [`Self::grid_mut`].
    pub grid_storage: Option<MemoryStorage>,
    // state_machine: StateMachine,
    // clock: Clock,
    // client_replies: ClientReplies,
    // message_bus: MessageBus,
    /// The client sessions table: the latest committed reply header per active
    /// client (upstream `client_sessions`).
    pub client_sessions: crate::client_sessions::ClientSessions,
    /// The committed client replies, persisted to the client-replies zone of
    /// the data file (upstream `client_replies`).
    ///
    /// Writes are only possible once a storage is mounted via
    /// [`Self::grid_storage`] (see the field's DEVIATION note); reads are
    /// deferred until the GetReply handler is ported.
    pub client_replies: crate::client_replies::ClientReplies,

    // ── Timeouts (tick counts) ──────────────────────────────────────────
    pub ping_timeout: Timeout,
    pub prepare_timeout: Timeout,
    pub commit_message_timeout: Timeout,
    pub view_change_status_timeout: Timeout,
    pub exit_view_message_timeout: Timeout,
    pub exit_view_window_timeout: Timeout,
    pub primary_abdicate_timeout: Timeout,
    pub journal_repair_timeout: Timeout,
    /// Drives periodic `GetBlocks` grid-repair requests (upstream
    /// `grid_repair_timeout`).
    pub grid_repair_timeout: Timeout,

    // ── Grid repair bookkeeping ──────────────────────────────────────────
    /// Per-replica budget of inflight block requests (upstream
    /// `grid_repair_message_budget`).
    pub grid_repair_message_budget: RepairBudgetGrid,
    /// Deterministic PRNG for repair-selection shuffles (upstream `prng`).
    pub prng: Prng,
    /// The most recent monotonic clock value, in nanoseconds, used to stamp
    /// grid repair budget requests.
    ///
    /// DEVIATION: upstream reads `self.clock.monotonic()` on demand; the
    /// sans-IO replica has no clock, so the owner's tick provides it.
    pub monotonic_now: u64,
    /// In-flight `GetBlocks` serve reads, keyed by grid read token, indexed
    /// before every request so a block is only ever fetched once per remote
    /// replica (upstream `replica.grid_reads`).
    ///
    /// DEVIATION: upstream preallocates a fixed pool of
    /// `constants.grid_repair_reads_max` read slots. This port keeps a plain
    /// `Vec` bounded by the same limit.
    pub grid_serve_reads: Vec<GridServeRead>,

    // ── Fault detection ──────────────────────────────────────────────────
    /// EWMA fault detector for commit heartbeats from the primary.
    pub commit_fault: FaultDetector,
    /// Timestamp of the last fresh commit heartbeat received.
    pub heartbeat_timestamp: u64,
    /// Bitmask of replicas that have sent ExitView for the current view.
    pub exit_view_from_all_replicas: u64,
    /// The JoinView headers: the journal suffix from `op` down to `commit_max`
    /// (descending op), broadcast in JoinView messages so the next primary can
    /// rebuild the log. Upstream `vsr.Headers.ViewChangeArray`.
    pub join_view_headers: Vec<message_header::Prepare>,
    /// JoinView messages collected for the current view change, indexed by
    /// replica. The new primary's own slot is filled when it broadcasts (modulo
    /// upstream's synchronous loopback). Upstream `join_view_from_all_replicas`.
    pub join_view_from_all_replicas: Vec<Option<crate::jv_quorum::JoinedView>>,
    /// Whether the JoinView quorum has been collected and validated: the new
    /// log is established for `self.view`.
    ///
    /// Upstream `src/vsr/replica.zig:144` (`join_view_quorum`).
    pub join_view_quorum: bool,
    /// The headers attached to the next View message: the journal suffix from
    /// `op` down (descending op), plus at most two checkpoint-boundary headers.
    /// Upstream `view_headers` (`vsr.Headers.ViewChangeArray`).
    pub view_headers: Vec<message_header::Prepare>,
    /// Whether the primary has started abdicating.
    pub primary_abdicating: bool,

    // ── Repair: head advancement ─────────────────────────────────────────
    /// Correlation nonce echoed in `GetView`/`View` exchanges so a backup can
    /// match the primary's reply to its request (upstream `self.nonce`).
    pub nonce: u128,

    // ── Outbound messages (sans-IO) ──────────────────────────────────────
    /// Outbound messages awaiting delivery.
    ///
    /// DEVIATION: upstream broadcasts through `message_bus` immediately. This
    /// sans-IO skeleton instead queues outbound messages here; the integration
    /// layer drains the queue.
    pub send_queue: Vec<crate::message::Message>,
    /// Client-directed outbound messages (PongClient/Eviction) awaiting delivery.
    ///
    /// DEVIATION: upstream routes client-directed messages to each client's
    /// address via `message_bus`; the sans-IO skeleton owns no client transport,
    /// so these are queued separately for the integration layer.
    pub client_send_queue: Vec<crate::message::Message>,
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
    /// Number of times the timeout has fired. Upstream's `attempts` (used to
    /// trigger an unconditional repair every 50 fires). Never reset; wraps for
    /// decoys only.
    pub attempts: u64,
}

impl Timeout {
    #[must_use]
    pub const fn start(ticks: u32) -> Self {
        Self { ticks, active: true, attempts: 0 }
    }

    /// Advance by one tick. Returns `true` if the timeout has fired.
    pub fn tick(&mut self) -> bool {
        if !self.active {
            return false;
        }
        self.ticks = self.ticks.saturating_sub(1);
        if self.ticks == 0 {
            self.active = false;
            self.attempts += 1;
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

    /// Reset the detector to a fresh EWMA seeded at `now`.
    ///
    /// # Panics
    ///
    /// Panics if `now` precedes the previous signal timestamp (the VSR clock
    /// must never move backwards).
    ///
    /// Upstream: `src/vsr/fault_detector.zig:126` (`reset`).
    pub fn reset(&mut self, now: u64) {
        assert!(now >= self.last_signal_timestamp);
        self.last_signal_timestamp = now;
        self.interval_ewma = self.interval_max;
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
///
/// Flexible Paxos: `quorum_replication_max` caps the replication quorum below a
/// majority; the view-change quorum is then raised to compensate
/// (`quorum_replication + quorum_view_change > replica_count`), optimizing the
/// common (replication) case at the expense of the rarer (view-change) case.
///
/// # Panics
/// Panics if `replica_count == 0`.
#[must_use]
pub const fn quorums(replica_count: u16) -> Quorum {
    assert!(replica_count > 0);
    // For replica_count=2, quorum_replication=2 even though 1 would intersect.
    // This improves durability of small clusters and avoids special-casing a
    // single-replica view change.
    let replication = if replica_count == 2 {
        2
    } else {
        // div_ceil(replica_count, 2), capped at quorum_replication_max.
        let div_ceil = replica_count.div_ceil(2);
        let cap = constants::QUORUM_REPLICATION_MAX as u16;
        if cap < div_ceil { cap } else { div_ceil }
    };
    // The view-change quorum may be more expensive to make the replication
    // quorum cheaper (see constants.rs:QUORUM_REPLICATION_MAX).
    let view_change = if replica_count == 2 { 2 } else { replica_count - replication + 1 };
    // How many nack_prepare messages (about a given op) to NACK; this is enough
    // to guarantee that the replication quorum was not reached (i.e. the op was
    // not committed), because `nack_prepare + quorum_replication > replica_count`.
    let nack_prepare = replica_count - replication + 1;
    // Simple majority. Upstream: div_ceil(n, 2) + (is_even(n)).
    let majority = replica_count.div_ceil(2) + if replica_count.is_multiple_of(2) { 1 } else { 0 };
    Quorum { replication, view_change, nack_prepare, majority }
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
    // `replica_index`/`replica_count` cast to the `u8` repair-budget indices;
    // `replica_count ≤ u8::MAX` (constrained by cluster tuples in `constants`).
    #[allow(clippy::cast_possible_truncation)]
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
            commit_timestamp: 0,
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
            journal_repair_timeout: Timeout::default(),
            grid_repair_timeout: Timeout::default(),
            grid_repair_message_budget: RepairBudgetGrid::new(RepairBudgetOptions {
                replica_index: replica_index as u8,
                replica_count: replica_count as u8,
            }),
            prng: Prng::from_seed(u64::from(replica_index)),
            monotonic_now: 0,
            commit_fault: FaultDetector::new(0, 100, 2_000),
            heartbeat_timestamp: 0,
            exit_view_from_all_replicas: 0,
            primary_abdicating: false,
            journal: crate::journal::Journal::new(cluster, replica_index),
            grid: None,
            grid_storage: None,
            grid_serve_reads: Vec::new(),
            send_queue: Vec::new(),
            client_send_queue: Vec::new(),
            client_sessions: crate::client_sessions::ClientSessions::new(),
            client_replies: crate::client_replies::ClientReplies::new(replica_index as u8),
            // Upstream boots from a superblock whose checkpoint holds the root
            // prepare at op 0; a fresh replica's JV always includes it.
            join_view_headers: vec![message_header::Prepare::root(cluster)],
            join_view_from_all_replicas: vec![None; constants::REPLICAS_MAX],
            join_view_quorum: false,
            view_headers: Vec::new(),
            // Sans-IO deterministic nonce (upstream uses a random u128);
            // nonzero so `GetView` messages validate.
            nonce: u128::from(replica_index) + 1,
        }
    }

    /// Returns the quorum sizes for this replica's cluster.
    #[must_use]
    pub fn quorum(&self) -> Quorum {
        quorums(self.replica_count)
    }

    /// Attach an opened grid and its backing storage (upstream constructs both
    /// at `Replica` creation from the superblock).
    ///
    /// DEVIATION: the sans-IO replica starts without a grid; [`Self::grid`] is
    /// `None` until the owner mounts one via this method (upstream always has a
    /// grid, whose `grid.callback == .cancel` gates the same code paths until it
    /// is open).
    pub fn mount_grid(&mut self, grid: Grid, storage: MemoryStorage) {
        self.grid = Some(grid);
        self.grid_storage = Some(storage);
    }

    /// The mounted grid and its storage, as a mutable pair.
    ///
    /// # Panics
    /// Panics if the grid is not mounted (upstream asserts the grid exists).
    pub fn grid_mut(&mut self) -> (&mut Grid, &mut MemoryStorage) {
        let grid = self
            .grid
            .as_mut()
            .unwrap_or_else(|| unreachable!("grid is mounted before grid repair paths run"));
        let storage = self.grid_storage.as_mut().unwrap_or_else(|| {
            unreachable!("grid storage is mounted before grid repair paths run")
        });
        (grid, storage)
    }

    /// The grid and its storage, as an immutable pair.
    #[must_use]
    pub fn grid_pair(&self) -> Option<(&Grid, &MemoryStorage)> {
        Some((self.grid.as_ref()?, self.grid_storage.as_ref()?))
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

    /// The op of the completed checkpoint of the working superblock.
    ///
    /// Upstream: `src/vsr/replica.zig:6770` (`op_checkpoint`).
    #[must_use]
    pub fn op_checkpoint(&self) -> u64 {
        // TODO(port): the working superblock's checkpoint op. Sans-IO: 0.
        0
    }

    /// The smallest op that may not be discarded (`op_checkpoint + 1`).
    ///
    /// Upstream: `src/vsr/replica.zig:6786` (`op_repair_min`).
    #[must_use]
    pub fn op_repair_min(&self) -> u64 {
        self.op_checkpoint() + 1
    }

    /// The largest op the WAL may hold: everything up to and including the
    /// checkpoint being synced to (or with no sync, the local checkpoint's
    /// prepare_max).
    ///
    /// Upstream: `src/vsr/replica.zig:6778` (`op_prepare_max_sync`).
    #[must_use]
    pub fn op_prepare_max_sync(&self) -> u64 {
        // TODO(port): `op_checkpoint_sync` when syncing; sans-IO the working
        // checkpoint is 0, so this is the prepare_max of checkpoint 0.
        self.op_checkpoint() + constants::VSR_CHECKPOINT_OPS as u64 - 1
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

    /// Advance `commit_max` monotonically.
    ///
    /// Upstream: `src/vsr/replica.zig:4188` (`advance_commit_max`).
    pub fn advance_commit_max(&mut self, commit: u64) {
        if commit > self.commit_max {
            self.commit_max = commit;
        }
    }

    /// Establish a new head op and commit number (`op`, `commit_max`).
    ///
    /// Truncates every op above `op` from the journal. Uncommitted ops may not
    /// survive a view change, but committed ops are never truncated; `commit_max`
    /// is never rewound because `commit_min` represents what we have already
    /// applied to the state machine.
    ///
    /// Upstream: `src/vsr/replica.zig:9619` (`set_op_and_commit_max`).
    fn set_op_and_commit_max(&mut self, op: u64, commit_max: u64) {
        assert!(self.status == Status::ViewChange || self.status == Status::Normal);
        assert!(op <= self.op_prepare_max_sync());
        // `maybe(op >= self.commit_max)` — bounded by pipelining:
        // the intersection property only requires that all possibly committed
        // operations survive into the new view.
        if op < self.op.min(self.op_prepare_max_sync()) {
            assert!(op >= commit_max.max(self.commit_max));
            assert!(self.op <= op + u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX));
        }
        assert!(self.commit_min <= self.commit_max);

        self.op = op;
        self.journal.remove_entries_from(self.op + 1);

        // Crucially, we must never rewind `commit_max` (and then `commit_min`)
        // because `commit_min` represents what we have already applied to our
        // state machine:
        self.commit_max = self.commit_max.max(commit_max);
        assert!(self.commit_max >= self.commit_min);
        assert!(
            self.commit_max
                >= self.op.saturating_sub(u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX))
        );
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
        request: u32,
        operation: crate::Operation,
        body_size: u32,
        request_checksum: u128,
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
            replica: self.replica_u8(),
            view: self.view,
            // DEVIATION: sans-IO has no client Request message to inherit
            // `release` from (upstream: replica.zig:7386); the replica's
            // minimum supported release is used. The `request_checksum` is the
            // client request's checksum, passed by the caller (upstream:
            // replica.zig:7391) — it lets a repeated request be matched against
            // the stored reply.
            release: crate::multiversion::Release::MINIMUM,
            parent,
            client,
            request_checksum,
            op,
            commit: self.commit_max,
            timestamp,
            request,
            operation,
            ..message_header::Prepare::default()
        };
        header.set_checksum_body(&[]);
        header.set_checksum();

        // The primary "self-sends" the prepare through the shared accept path
        // (upstream: `primary_pipeline_prepare` → `on_prepare`), which advances
        // `op` and records the header in the journal:
        let result = self.on_prepare(&header);
        assert_eq!(result, OnPrepareResult::Accepted);

        let prepare = PipelinePrepare {
            op,
            checksum: header.checksum(),
            client,
            acks_received: 0,
            ok_quorum_received: false,
        };
        self.pipeline_queue.prepare_queue.push(prepare);
        self.ok_from_all_replicas.push(0);

        // Upstream counts the primary's own prepare_ok toward a quorum of
        // prepare_oks ("including ourself", replica.zig:2293): the primary's
        // journal write completes on the loopback as `write_prepare_callback`
        // → `send_prepare_ok`. Contribute that self-ack here.
        let result = self.on_prepare_ok(op, header.checksum(), self.replica_index);
        assert!(matches!(
            result,
            PrepareOkResult::AckCounted
                | PrepareOkResult::QuorumReached { .. }
                | PrepareOkResult::DuplicateAck
        ));

        // Start timeouts if the pipeline was previously empty.
        if self.pipeline_queue.prepare_queue.len() == 1 {
            self.prepare_timeout = Timeout::start(constants::PIPELINE_PREPARE_QUEUE_MAX);
        }

        let _ = body_size;

        // Replicate the prepare to every backup (upstream: `replicate` →
        // `send_message_to_other_replicas_and_standbys`, replica.zig:8550-8552).
        // DEVIATION: upstream serializes the request body into the message;
        // sans-IO bodies are deferred, so a header-only prepare is broadcast
        // (receivers dispatch on the header alone).
        let mut message = crate::message::Message::new();
        message.set_header(&header);
        for _ in 1..self.replica_count {
            self.send_queue.push(message.clone());
        }

        Ok(op)
    }

    /// The head of the prepare pipeline that has not yet reached a quorum of
    /// prepare_oks, with its slot index.
    ///
    /// Upstream: `src/vsr/replica.zig:7502` (`primary_pipeline_pending`).
    fn primary_pipeline_pending(&self) -> Option<(usize, &PipelinePrepare)> {
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());
        self.pipeline_queue
            .prepare_queue
            .iter()
            .enumerate()
            .find(|(_, prepare)| !prepare.ok_quorum_received)
    }

    /// Accept a Prepare replicated from the primary (or self-sent on the
    /// primary). Advances `op` by exactly one and records the header as dirty
    /// in the journal.
    ///
    /// Out-of-order prepares (a gap of more than one op) are rejected with
    /// [`OnPrepareResult::FutureOp`]; the caller recovers via
    /// `jump_to_newer_op_in_normal_status`. Only the concurrent `repair()` gap
    /// filling is deferred.
    ///
    /// # Panics
    /// Panics if the replica is not `.normal`, if the header's cluster/view/
    /// replica do not match ours, if the operation is `.reserved`/`.root`, or if
    /// the header breaks the journal hash chain within this view.
    ///
    /// Upstream: `src/vsr/replica.zig:2021` (`on_prepare`).
    pub fn on_prepare(&mut self, header: &message_header::Prepare) -> OnPrepareResult {
        assert_eq!(self.status, Status::Normal);
        assert_eq!(header.cluster, self.cluster);
        assert_eq!(header.view, self.view);
        assert!(u64::from(header.replica) < u64::from(self.replica_count));
        assert_ne!(header.operation, crate::Operation::RESERVED);
        assert_ne!(header.operation, crate::Operation::ROOT);

        if header.op <= self.op {
            return OnPrepareResult::Stale;
        }
        if header.op > self.op + 1 {
            return OnPrepareResult::FutureOp;
        }
        assert_eq!(header.op, self.op + 1);

        // The parent must link to our current head (an older/newer entry may be
        // a whole journal's worth of ops behind due to ring wrapping):
        if let Some(previous) = self.journal.previous_entry(header) {
            assert_eq!(previous.checksum(), header.parent, "hash chain break in view");
        }

        // Advance op and record the prepare in the journal:
        self.op = header.op;
        self.journal.set_header_as_dirty(header);

        OnPrepareResult::Accepted
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

    /// Wire handler: route a PrepareOk message to `on_prepare_ok`.
    ///
    /// After counting the ack, drives the commit dispatch so the primary can
    /// commit the op (or the next queued op) once a quorum of prepare_oks has
    /// arrived — upstream `on_prepare_ok` finishes with `commit_pipeline()`.
    fn on_prepare_ok_message(&mut self, message: &crate::message::Message) {
        let Some(prepare_ok) = message.header::<message_header::PrepareOk>() else {
            return; // Command mismatch or malformed header.
        };
        // DEVIATION: upstream validates checksums at the message_bus receive
        // path; sans-IO messages are constructed locally, so guard anyway.
        if !prepare_ok.valid_checksum() || !prepare_ok.valid_checksum_body(message.body_used()) {
            return;
        }
        if prepare_ok.cluster != self.cluster {
            return;
        }
        assert!(u16::from(prepare_ok.replica) < self.replica_count);
        // Upstream `ignore_prepare_ok` only counts acks in normal status for the
        // current view (older/newer-view acks are dropped; replica.zig:6076-6096).
        if self.status != Status::Normal || prepare_ok.view != self.view {
            return;
        }

        match self.on_prepare_ok(
            prepare_ok.op,
            prepare_ok.prepare_checksum,
            u16::from(prepare_ok.replica),
        ) {
            PrepareOkResult::Ignored
            | PrepareOkResult::UnknownOp
            | PrepareOkResult::ChecksumMismatch
            | PrepareOkResult::DuplicateAck => {}
            PrepareOkResult::AckCounted | PrepareOkResult::QuorumReached { .. } => {
                if self.status == Status::Normal && self.is_primary() {
                    self.commit_dispatch_enter();
                }
            }
        }
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
            self.commit_start_journal()
        }
    }

    /// Enter the commit pipeline to commit committed ops from the journal
    /// (backup only).
    ///
    /// Upstream: `src/vsr/replica.zig:4310` (`commit_journal`).
    fn commit_journal(&mut self) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(!self.is_primary());
        assert!(self.commit_min <= self.commit_max);
        assert!(self.commit_min <= self.op);

        // We have already committed this far:
        if self.commit_max == self.commit_min {
            return;
        }

        // Guard against multiple concurrent invocations:
        if self.commit_stage != CommitStage::Idle {
            return;
        }

        self.commit_dispatch_enter();
    }

    /// Backup: take the next committed op from the journal.
    ///
    /// Commits forward while `commit_min < commit_max`, as long as the next op
    /// is already journaled. The prepare *body* read is async and deferred with
    /// the message bus — the header's op is all the pipeline needs for now.
    ///
    /// Upstream: `src/vsr/replica.zig:4606` (`commit_start_journal`).
    fn commit_start_journal(&mut self) -> bool {
        assert_eq!(self.commit_stage, CommitStage::Start);
        assert!(self.commit_prepare.is_none());
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(!self.is_primary());
        assert!(self.commit_min <= self.commit_max);
        assert!(self.commit_min <= self.op);

        // We may receive commit numbers for ops we do not yet have
        // (`commit_max > self.op`):
        if self.commit_min < self.commit_max && self.commit_min < self.op {
            let op = self.commit_min + 1;
            // The prepare header must be present; without it there is nothing to
            // commit yet (a stale/nonexistent entry leaves the pipeline idle).
            if self.journal.header_with_op(op).is_none() {
                return true;
            }
            // TODO(port): `valid_hash_chain` check (deferred).
            // TODO(port): async prepare body read (`journal.read_prepare`).
            self.commit_prepare = Some(op);
        }
        true
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

    /// Whether this replica is responsible for replying to the client whose op
    /// is being executed: the primary always replies; among the backups exactly
    /// one, selected deterministically by the op, so that a client retrying
    /// against another replica does not get a duplicate-request race
    /// (upstream `execute_op_reply_to_client`, replica.zig:5328).
    ///
    /// # Panics
    /// Panics if the replica count exceeds 256 (upstream `replica_count` is a
    /// `u8`; the `u16` in this port only widens the domain).
    #[allow(clippy::cast_possible_truncation)] // replica_count ≤ 256 (asserted)
    fn execute_op_reply_to_client(&self, op: u64) -> bool {
        if self.is_primary() {
            return true;
        }
        if self.replica_count == 1 {
            return false;
        }
        assert!(self.replica_count <= u16::from(u8::MAX) + 1);
        let mut prng = tigerbeetle_core::stdx::prng::Prng::from_seed(op);
        // Upstream: `range_inclusive(u8, 1, replica_count - 1)`.
        let offset = 1_u8 + prng.gen_int_inclusive_u8(self.replica_count as u8 - 2);
        let backup = (self.primary_index() + u16::from(offset)) % self.replica_count;
        backup == self.replica_index
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

        let Some(prepare) = self.journal.header_with_op(op).copied() else {
            unreachable!("the op being committed is journaled");
        };

        // Execute on state machine.
        // TODO(port): state_machine.execute(commit_prepare.body_used())

        // Track what the state machine has executed: every prepare's timestamp
        // must strictly advance the previous one, which `<` therefore pins the
        // primary's `prepare_timestamp` (upstream `execute_op` asserts
        // `state_machine.commit_timestamp < prepare.header.timestamp`, and the
        // AOF-recovery exception is moot sans-IO — replica.zig:5441-5445).
        assert!(self.commit_timestamp < prepare.timestamp);
        self.commit_timestamp = prepare.timestamp;

        // Construct the client reply from the committed prepare and update the
        // client sessions table (upstream `execute_op`: replica.zig:5391-5523).
        // Runs on the primary and backups alike: any replica can answer a
        // client's GetReply.
        let reply = Self::build_reply(&prepare);

        match reply.operation {
            crate::Operation::REGISTER => self.client_table_entry_create(&reply),
            crate::Operation::PULSE | crate::Operation::UPGRADE => {
                assert_eq!(reply.client, 0);
            }
            _ => self.client_table_entry_update(&reply),
        }

        // Reply to the client: the primary always; exactly one backup per op
        // (selected deterministically). Pulse/upgrade have no client.
        if reply.client != 0 && self.execute_op_reply_to_client(reply.op) {
            let mut message = crate::message::Message::new();
            message.set_header(&reply);
            self.send_reply_message_to_client(&message);
        }

        self.commit_op(op);
        assert!(self.commit_min <= self.commit_max);

        if self.status == Status::Normal && self.is_primary() {
            assert_eq!(self.commit_min, self.commit_max);

            // Pop the committed prepare from the pipeline (primary only).
            let popped = self.pop_committed();
            assert!(popped.is_some_and(|p| p.op == op));

            // Pop the next request to prepare (if any) and prepare it now that
            // the pipeline has room (upstream: replica.zig:4901-4904).
            if let Some(request) = self.pipeline_queue.request_queue.pop() {
                let result = self.primary_pipeline_prepare(
                    request.client,
                    request.request,
                    request.operation,
                    0, // DEVIATION: sans-IO bodies are deferred; unused.
                    request.request_checksum,
                );
                assert!(result.is_ok(), "popped request must be preparable");
            }
        }
    }

    /// Construct a client `Reply` for a committed prepare.
    ///
    /// The reply's `operation`, `client`, `request` and `request_checksum`
    /// echo the prepare; `commit` is the prepare's op; `context` is the reply's
    /// stable-with-view checksum so a retransmitted reply stays valid
    /// (upstream replica.zig:5466-5486).
    ///
    /// # Panics
    ///
    /// Panics if the prepare's operation is `.root` or `.reserved`.
    ///
    /// DEVIATION: upstream sizes the reply according to the state machine's
    /// result (`.register` bodies carry a `RegisterResult`; state-machine
    /// operations carry a `StateMachine.Result`). sans-IO the state machine
    /// execute and the request/result bodies are deferred, so only the reply
    /// header is built: empty-bodied except for `.register`, whose
    /// `RegisterResult` size is preserved (the `batch_size_limit` contents are
    /// zeroed — upstream copies them from the register request body).
    fn build_reply(prepare: &message_header::Prepare) -> message_header::Reply {
        assert_ne!(prepare.operation, crate::Operation::ROOT);
        assert_ne!(prepare.operation, crate::Operation::RESERVED);

        let mut reply = message_header::Reply {
            cluster: prepare.cluster,
            replica: prepare.replica,
            view: prepare.view,
            release: prepare.release,
            op: prepare.op,
            commit: prepare.op,
            timestamp: prepare.timestamp,
            client: prepare.client,
            request: prepare.request,
            operation: prepare.operation,
            request_checksum: prepare.request_checksum,
            size: message_header::SIZE_U32,
            ..message_header::Reply::default()
        };
        // DEVIATION: the reply body is deferred (see above), so its checksum
        // covers the zeroed `RegisterResult` that this build produces (the
        // `batch_size_limit` contents are zeroed — upstream copies them from
        // the register request body).
        if reply.operation == crate::Operation::REGISTER {
            reply.size += message_header::REGISTER_RESULT_SIZE_U32;
            reply.set_checksum_body(&[0_u8; message_header::REGISTER_RESULT_SIZE_U32 as usize]);
        } else {
            reply.set_checksum_body(&[]);
        }
        // `context` is the reply's checksum computed with a fixed view,
        // allowing the reply to be retransmitted in a newer view
        // (upstream replica.zig:5484-5485).
        reply.context = reply.calculate_checksum();
        reply.set_checksum();
        reply
    }

    /// Record a newly registered client's session on commit.
    ///
    /// The register op's commit number becomes the client's session number.
    ///
    /// # Panics
    ///
    /// Panics unless the reply is a valid register reply, or when the session
    /// table is already full and has no eviction candidate (upstream asserts).
    ///
    /// Upstream: `src/vsr/replica.zig:5692` (`client_table_entry_create`).
    fn client_table_entry_create(&mut self, reply: &message_header::Reply) {
        assert_eq!(reply.command, crate::command::Command::Reply);
        assert_eq!(reply.operation, crate::Operation::REGISTER);
        assert_ne!(reply.client, 0);
        assert_eq!(reply.op, reply.commit);
        assert_eq!(reply.size, message_header::SIZE_U32 + message_header::REGISTER_RESULT_SIZE_U32);

        let session = reply.commit; // The commit number becomes the session number.
        let request = reply.request;
        // The `0` commit number is reserved for the cluster `.root` operation.
        assert_ne!(session, 0);
        assert_eq!(request, 0);

        let clients_max = constants::CLIENTS_MAX as usize;
        let clients = self.client_sessions.count();
        assert!(clients <= clients_max);
        if clients == clients_max {
            let evictee = self.client_sessions.evictee();
            self.client_sessions.remove(evictee);
            assert_eq!(self.client_sessions.count(), clients_max - 1);
        }

        let slot = self.client_sessions.put(session, reply);
        assert!(self.client_sessions.count() <= clients_max);

        // Persist the reply body to the client-replies zone (a register reply
        // is the only body-ful reply our operations produce).
        //
        // DEVIATION: upstream writes unconditionally. The sans-IO replica has
        // no storage until the owner mounts one via `grid_storage` (see its
        // DEVIATION note); without it the reply lives only in the in-memory
        // sessions trailer.
        if let Some(storage) = self.grid_storage.as_mut() {
            let mut message = crate::message::Message::new();
            // The body is already zeroed by `Message::new` (the zeroed
            // `RegisterResult` that `build_reply`'s checksum covers).
            message.set_header(reply);
            self.client_replies.write_reply(
                storage,
                slot,
                message,
                crate::client_replies::WriteTrigger::Commit,
            );
        }
    }

    /// Update a registered client's latest reply on commit.
    ///
    /// If the client's session was evicted while preparing, there is nothing to
    /// update (the next request will receive an eviction from the primary).
    ///
    /// Upstream: `src/vsr/replica.zig:5750` (`client_table_entry_update`).
    fn client_table_entry_update(&mut self, reply: &message_header::Reply) {
        assert_eq!(reply.command, crate::command::Command::Reply);
        assert_ne!(reply.operation, crate::Operation::REGISTER);
        assert_ne!(reply.client, 0);
        assert_eq!(reply.op, reply.commit);
        assert_ne!(reply.commit, 0);
        assert_ne!(reply.request, 0);

        if let Some(entry) = self.client_sessions.get_mut(reply.client) {
            assert_eq!(entry.header.command, crate::command::Command::Reply);
            assert_eq!(entry.header.op, entry.header.commit);
            assert!(entry.header.commit >= entry.session);
            assert_eq!(entry.header.client, reply.client);
            assert_eq!(entry.header.request + 1, reply.request);
            assert!(entry.header.op < reply.op);
            assert!(entry.header.commit < reply.commit);
            assert_eq!(entry.header.release.value, reply.release.value);

            entry.header = *reply;
        }

        // The session was evicted while preparing; nothing to do. The next
        // request will receive an eviction from the primary.
        let Some(slot) = self.client_sessions.get_slot_for_header(reply) else {
            return;
        };

        // A body-less reply needs no storage: the header lives safely in the
        // `client_sessions` trailer, so the slot's reply is removed
        // (upstream replica.zig:5773-5776).
        if reply.size == message_header::SIZE_U32 {
            self.client_replies.remove_reply(slot);
            return;
        }

        // DEVIATION: upstream writes unconditionally; the sans-IO replica
        // persists only when a storage is mounted (see `grid_storage`'s
        // DEVIATION note).
        let Some(storage) = self.grid_storage.as_mut() else {
            return;
        };
        let mut message = crate::message::Message::new();
        message.set_header(reply);
        self.client_replies.write_reply(
            storage,
            slot,
            message,
            crate::client_replies::WriteTrigger::Commit,
        );
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

        // Upstream takes the checksum from the superblock checkpoint when
        // `commit_max` is the checkpoint op, otherwise from the journal head.
        // The superblock is not yet ported, so for the checkpoint op (= 0) we
        // fall back to the root prepare's checksum.
        let commit_checksum = if let Some(entry) = self.journal.header_with_op(self.commit_max) {
            entry.checksum()
        } else {
            assert_eq!(self.commit_max, 0);
            message_header::Prepare::root(self.cluster).checksum()
        };

        let mut commit = message_header::Commit::default();
        commit.cluster = self.cluster;
        // DEVIATION: upstream sets `view`/`replica`/`timestamp_monotonic` from
        // the current header and the synchronous clock. The sans-IO skeleton
        // has no clock, so the caller's `now` is used directly as the
        // timestamp (units: ms rather than upstream's ns).
        commit.view = self.view;
        commit.replica = self.replica_u8();
        commit.commit = self.commit_max;
        commit.commit_checksum = commit_checksum;
        commit.timestamp_monotonic = now;
        // TODO(port): checkpoint_op/checkpoint_id from the superblock.
        commit.set_checksum_body(&[]);
        commit.set_checksum();
        // Broadcast to every backup (upstream: `send_commit` →
        // `send_header_to_other_replicas_and_standbys`, replica.zig:11370).
        for _ in 1..self.replica_count {
            self.enqueue_header(&commit);
        }
    }

    /// Enqueue an outbound message for the integration layer.
    ///
    /// DEVIATION: upstream hands the message to `message_bus` immediately; the
    /// sans-IO skeleton records it in [`Self::send_queue`] instead.
    fn enqueue_header<H: TypedHeader>(&mut self, typed: &H) {
        let mut message = crate::message::Message::new();
        message.set_header(typed);
        self.send_queue.push(message);
    }

    /// Handle a ping message — reply with a pong.
    ///
    /// Pings let replicas synchronize cluster time and probe for connectivity.
    /// Only replicas in `Normal`/`ViewChange` status reply, and misdirected
    /// pings (from ourselves) are dropped.
    ///
    /// Upstream: `src/vsr/replica.zig:1849` (`on_ping`).
    pub fn on_ping(&mut self, message: &crate::message::Message) {
        let Some(ping) = message.header::<message_header::Ping>() else {
            return;
        };
        if !ping.valid_checksum() || !ping.valid_checksum_body(message.body_used()) {
            return;
        }
        if ping.invalid_header().is_some() {
            return;
        }
        if self.status != Status::Normal && self.status != Status::ViewChange {
            return;
        }
        if ping.replica == self.replica_u8() {
            return; // Misdirected message (self).
        }
        // TODO(port): multiversion `upgrade_targets` tracking from the ping's
        // view/checkpoint/release list (upstream replica.zig:1871).

        // DEVIATION: upstream uses `view_durable()` (the on-disk view) so that
        // pongs aren't dropped while the view is being updated, and `self.release`
        // for the multiversion version; the sans-IO replica has neither superblock
        // nor multiversion support, so the in-memory view and the minimum release
        // stand in (matching the Prepare header construction).
        let mut reply = message_header::Pong {
            cluster: self.cluster,
            replica: self.replica_u8(),
            view: self.view,
            release: crate::multiversion::Release::MINIMUM,
            // Copy the ping's monotonic timestamp and add our own wall-clock sample.
            ping_timestamp_monotonic: ping.ping_timestamp_monotonic,
            // DEVIATION: upstream samples `clock.realtime()` here; sans-IO has no
            // wall clock, so the owner's latest monotonic `now` stands in.
            pong_timestamp_wall: self.monotonic_now,
            size: u32::try_from(message_header::SIZE)
                .unwrap_or_else(|_| unreachable!("SIZE fits u32")),
            ..message_header::Pong::default()
        };
        reply.set_checksum_body(&[]);
        reply.set_checksum();
        self.enqueue_header(&reply);
    }

    /// Handle a pong message — feed clock learning.
    ///
    /// Upstream: `src/vsr/replica.zig:1898` (`on_pong`).
    pub fn on_pong(&mut self, message: &crate::message::Message) {
        let Some(pong) = message.header::<message_header::Pong>() else {
            return;
        };
        if !pong.valid_checksum() || !pong.valid_checksum_body(message.body_used()) {
            return;
        }
        if pong.invalid_header().is_some() {
            return;
        }
        if pong.replica == self.replica_u8() {
            return; // Misdirected message (self).
        }
        // Ignore clocks of standbys.
        if u16::from(pong.replica) < self.replica_count {
            // TODO(port): `clock.learn(replica, ping_timestamp_monotonic,
            // pong_timestamp_wall, monotonic_now)` and `prepare_timeout.set_rtt_ns`
            // from the measured round-trip time (needs the clock/io infrastructure).
        }
    }

    /// Handle a client's PingClient — reply with a PongClient (time sync), or
    /// evict the client if it is unknown or unsupported.
    ///
    /// Upstream: `src/vsr/replica.zig:1919` (`on_ping_client`).
    pub fn on_ping_client(&mut self, message: &crate::message::Message) {
        let Some(ping_client) = message.header::<message_header::PingClient>() else {
            return;
        };
        if !ping_client.valid_checksum() || !ping_client.valid_checksum_body(message.body_used()) {
            return;
        }
        if ping_client.invalid_header().is_some() {
            return;
        }
        // `PingClient::invalid_header` guarantees `client != 0` (upstream also
        // asserts it).

        if self.ignore_ping_client(&ping_client) {
            return;
        }

        let mut reply = message_header::PongClient {
            cluster: self.cluster,
            // DEVIATION: upstream reports `log_view_durable()`; sans-IO has no
            // superblock, so the in-memory log view stands in.
            view: self.log_view,
            replica: self.replica_u8(),
            release: crate::multiversion::Release::MINIMUM,
            // Echo the client's monotonic timestamp back for clock synchronization.
            ping_timestamp_monotonic: ping_client.ping_timestamp_monotonic,
            size: u32::try_from(message_header::SIZE)
                .unwrap_or_else(|_| unreachable!("SIZE fits u32")),
            ..message_header::PongClient::default()
        };
        reply.set_checksum_body(&[]);
        reply.set_checksum();
        self.send_header_to_client(ping_client.client, &reply);
    }

    /// Whether a PingClient must be dropped (or its client evicted) rather than
    /// answered with a PongClient.
    ///
    /// Upstream: `src/vsr/replica.zig:6003` (`ignore_ping_client`).
    fn ignore_ping_client(&mut self, ping_client: &message_header::PingClient) -> bool {
        assert_eq!(ping_client.command, crate::command::Command::PingClient);
        assert_ne!(ping_client.client, 0);

        // DEVIATION: upstream drops PingClient from standbys; the sans-IO replica
        // does not support standby configuration, so that branch never applies.

        // If the client is not in the sessions table, a nonzero session means its
        // register hasn't landed yet. An up-to-date primary that has already
        // committed the register evicts the client, forcing a fresh register
        // (upstream replica.zig:6014-6025).
        if self.client_sessions.get(ping_client.client).is_none()
            && ping_client.session != 0
            && self.status == Status::Normal
            && self.is_primary()
            && self.commit_min >= ping_client.session
        {
            self.send_eviction_message_to_client(
                ping_client.client,
                message_header::Reason::NoSession,
            );
            return true;
        }

        // DEVIATION: upstream bounds-checks the client's release against
        // `release_client_min` and `self.release` (multiversion); sans-IO
        // supports only the minimum release, so clients must speak exactly it.
        if ping_client.release.value < crate::multiversion::Release::MINIMUM.value {
            if self.status == Status::Normal && self.is_primary() {
                self.send_eviction_message_to_client(
                    ping_client.client,
                    message_header::Reason::ClientReleaseTooLow,
                );
            }
            return true;
        }
        if ping_client.release.value > crate::multiversion::Release::MINIMUM.value {
            if self.status == Status::Normal && self.is_primary() {
                self.send_eviction_message_to_client(
                    ping_client.client,
                    message_header::Reason::ClientReleaseTooHigh,
                );
            }
            return true;
        }

        false
    }

    /// Send a committed reply back to its client (upstream
    /// `send_reply_message_to_client`, replica.zig:8946).
    ///
    /// If the reply was committed in an older view it is retransmitted with
    /// the current `log_view` (`context`, the stable-with-view checksum, is
    /// preserved), so the client never resumes against a primary that has
    /// fallen behind (upstream replica.zig:8964-8987).
    ///
    /// DEVIATION: upstream sends through the message bus to the client's
    /// address. The sans-IO port pushes the reply onto `client_send_queue`,
    /// which the integration layer must pair with the client.
    ///
    /// # Panics
    /// Panics if the message is not a reply, is addressed to the reserved
    /// `client == 0`, or carries a view newer than `self.view` (upstream
    /// asserts all three).
    fn send_reply_message_to_client(&mut self, reply: &crate::message::Message) {
        let Some(header) = reply.header::<message_header::Reply>() else {
            return;
        };
        assert_eq!(header.command, crate::command::Command::Reply);
        assert_ne!(header.client, 0);
        assert!(header.view <= self.view);

        if header.view == self.log_view {
            self.client_send_queue.push(reply.clone());
            return;
        }

        // Cold path: bump the view on a copy (upstream replica.zig:8968-8986).
        let mut header = header;
        header.view = self.log_view;
        header.set_checksum();
        let mut copy = crate::message::Message::new();
        copy.set_header(&header);
        let size = usize::try_from(header.size).unwrap_or_else(|_| unreachable!("size fits usize"));
        copy.buffer_mut()[message_header::SIZE..size].copy_from_slice(reply.body_used());
        self.client_send_queue.push(copy);
    }

    /// Send an Eviction message to a client (primary only).
    ///
    /// # Panics
    /// Panics unless the replica is `Normal` and primary (upstream asserts).
    ///
    /// Upstream: `src/vsr/replica.zig:8921`
    /// (`send_eviction_message_to_client`).
    fn send_eviction_message_to_client(&mut self, client: u128, reason: message_header::Reason) {
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());

        let mut eviction = message_header::Eviction {
            cluster: self.cluster,
            // DEVIATION: upstream reports `log_view_durable()`; sans-IO uses the
            // in-memory log view.
            view: self.log_view,
            replica: self.replica_u8(),
            release: crate::multiversion::Release::MINIMUM,
            client,
            reason_ordinal: reason as u8,
            size: u32::try_from(message_header::SIZE)
                .unwrap_or_else(|_| unreachable!("SIZE fits u32")),
            ..message_header::Eviction::default()
        };
        eviction.set_checksum_body(&[]);
        eviction.set_checksum();
        self.send_header_to_client(client, &eviction);
    }

    /// Send a client-directed header (PongClient or Eviction) to `client`.
    ///
    /// # Panics
    /// Panics if the header's cluster differs, if the header is not
    /// client-directed, or if the header's view is newer than `log_view`
    /// (upstream asserts all three; `view <= log_view_durable()`).
    ///
    /// Upstream: `src/vsr/replica.zig:8988` (`send_header_to_client`).
    fn send_header_to_client<T: TypedHeader>(&mut self, _client: u128, header: &T) {
        // `_client` documents the routing intent; the message carries no client
        // address (the PongClient header has no `client` field), so the
        // integration layer must pair it with the triggering PingClient.
        let frame = header.frame();
        assert_eq!(frame.cluster, self.cluster);
        assert!(frame.view <= self.log_view);
        assert!(
            frame.command == crate::command::Command::PongClient
                || frame.command == crate::command::Command::Eviction
        );

        let mut message = crate::message::Message::new();
        message.set_header(header);
        self.client_send_queue.push(message);
    }

    /// `self.replica_index` as the wire `u8` replica field.
    ///
    /// Truncation is safe: `replica_count ≤ u8::MAX` (constrained by cluster
    /// tuples in `constants`), so each replica index fits in a byte.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    fn replica_u8(&self) -> u8 {
        self.replica_index as u8
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
        // DEVIATION: upstream broadcasts + loopbacks to self, and only flips to
        // ViewChange once the ExitView quorum arrives in `on_exit_view`. The
        // sans-IO skeleton skips the loopback and records our own bit here, but
        // keeps `status` Normal so the same quorum-triggered transition runs.
        let mut exit_view = message_header::ExitView {
            cluster: self.cluster,
            view: self.view,
            replica: self.replica_u8(),
            ..message_header::ExitView::default()
        };
        // Set our own bit (upstream: the loopback processed by on_exit_view).
        self.exit_view_from_all_replicas |= 1u64 << self.replica_index;
        // Start the exit-view window timeout (5s) and message timeout (500ms).
        self.exit_view_window_timeout = Timeout::start(constants::EXIT_VIEW_WINDOW_TIMEOUT);
        self.exit_view_message_timeout = Timeout::start(constants::EXIT_VIEW_MESSAGE_TIMEOUT);

        exit_view.set_checksum_body(&[]);
        exit_view.set_checksum();
        self.enqueue_header(&exit_view);
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
    /// Upstream: `src/vsr/replica.zig:10160` (`transition_to_view_change_status`).
    fn transition_to_view_change_status(&mut self, new_view: u32) {
        assert!(new_view > self.view);
        // Upstream rebuilds the JoinView headers while still in normal status
        // (they describe the journal of the *previous* log view). `log_view`
        // deliberately stays at the last normal view — the JV message carries
        // it, and it is only advanced once a new log is established.
        if self.status == Status::Normal {
            self.update_join_view_headers();
        }
        self.status = Status::ViewChange;
        self.view = new_view;
        self.exit_view_from_all_replicas = 0;
        // Do not let messages from the previous (aborted) view change count
        // towards this one — the quorum-intersection property depends on it
        // (upstream `reset_quorum_join_view`).
        self.join_view_from_all_replicas.fill(None);
        self.join_view_quorum = false;
        self.pipeline_queue = PipelineQueue::default();
        self.ok_from_all_replicas.clear();
        self.view_change_status_timeout = Timeout::start(constants::VIEW_CHANGE_STATUS_TIMEOUT);
        // Stop driving repairs while the journal may be inconsistent with the
        // new view (upstream `replica.zig:10231`).
        self.journal_repair_timeout.stop();
        self.grid_repair_timeout.stop();
        self.send_join_view();
    }

    /// Transition from `ViewChange` to `Normal`, once a new log has been
    /// established for `view_new`.
    ///
    /// # Panics
    ///
    /// Panics unless the replica is in `ViewChange` status with the new log
    /// established — a backup whose journal carries `op`, or the new primary
    /// whose journal is contiguously clean and whose pipeline holds the
    /// survivor prepares.
    ///
    /// Upstream: `src/vsr/replica.zig:10056`
    /// (`transition_to_normal_from_view_change_status`).
    fn transition_to_normal_from_view_change_status(&mut self, view_new: u32, now: u64) {
        assert_eq!(self.status, Status::ViewChange);
        assert!(
            self.commit_max
                >= self.op.saturating_sub(u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX))
        );
        assert!(view_new >= self.view);
        assert!(self.journal.header_with_op(self.op).is_some());
        assert!(!self.primary_abdicating);
        // DEVIATION: upstream asserts `view_headers.command == .view` here (a
        // Headers array records the command of the stored messages); the
        // sans-IO `Vec<Prepare>` carries no command, and `on_view_set_journal`
        // has just installed the View's headers.
        assert!(!self.prepare_timeout.active);
        assert!(!self.primary_abdicate_timeout.active);

        self.status = Status::Normal;
        self.commit_fault.reset(now);

        if self.is_primary() {
            assert_eq!(self.view, view_new);
            assert_eq!(self.log_view, view_new);
            assert_eq!(self.commit_min, self.commit_max);
            // DEVIATION: upstream asserts `primary_journal_repaired()` and
            // `pipeline == .queue` (with the survivors asserted contiguous and
            // clean). Sans-IO the journal is already verified, and the pipeline
            // was rebuilt by `primary_start_view_as_the_new_primary`.
            assert_eq!(self.commit_max + self.pipeline_queue.prepare_queue.len() as u64, self.op);

            self.ping_timeout = Timeout::start(constants::PING_TIMEOUT);
            // DEVIATION: upstream additionally stops join_view_message_timeout
            // and get_view_message_timeout (sans-IO timeouts do not exist).
            self.commit_message_timeout = Timeout::start(constants::COMMIT_MESSAGE_TIMEOUT);
            self.exit_view_window_timeout.stop();
            self.exit_view_message_timeout = Timeout::start(constants::EXIT_VIEW_MESSAGE_TIMEOUT);
            self.view_change_status_timeout.stop();

            // Do not reset the pipeline as there may be uncommitted ops to
            // drive to completion (upstream `replica.zig:10095`).
            if !self.pipeline_queue.prepare_queue.is_empty() {
                self.prepare_timeout = Timeout::start(constants::PIPELINE_PREPARE_QUEUE_MAX);
                self.primary_abdicate_timeout = Timeout::start(constants::PRIMARY_ABDICATE_TIMEOUT);
            }
        } else {
            // DEVIATION: upstream does not set view/log_view when we recovered
            // into the same view we crashed in (log_view == view_new, via
            // recovering_head); sans-IO replicas only reach this from a View
            // whose log_view is the previous one, so the assignment always runs.
            self.view = view_new;
            self.log_view = view_new;
            // TODO(port): `view_durable_update()` needs the superblock.
            self.ping_timeout = Timeout::start(constants::PING_TIMEOUT);
            self.commit_message_timeout.stop();
            self.exit_view_window_timeout.stop();
            self.exit_view_message_timeout = Timeout::start(constants::EXIT_VIEW_MESSAGE_TIMEOUT);
            self.view_change_status_timeout.stop();

            // Upstream asserts `pipeline == .cache` (and `get_view_message_timeout`
            // ticking, which does not exist sans-IO); the skeleton's only pipeline
            // is the primary's queue, emptied at transition_to_view_change_status.
            assert!(self.pipeline_queue.prepare_queue.is_empty());
        }

        // DEVIATION: upstream additionally starts get_view_message_timeout and
        // repair_sync_timeout, and resets `commit_mins` / `head_ops` EWMA history.
        self.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        // Upstream starts the grid repair timeout with `100 / tick_ms` ticks
        // (replica.zig:9921); sans-IO ticks are unitless, so it uses the same
        // fixed cadence as the journal repair timeout.
        self.grid_repair_timeout = Timeout::start(constants::GRID_REPAIR_TIMEOUT);

        self.heartbeat_timestamp = 0;
        // A replica reports its own ExitView only while it thinks the primary
        // is faulty (`reset_quorum_exit_view`).
        self.exit_view_from_all_replicas = 0;
        // Upstream `reset_quorum_join_view` (also clears the flag on a primary that
        // collected the quorum it just transitioned out of):
        self.join_view_from_all_replicas.fill(None);
        self.join_view_quorum = false;
    }

    /// Become a `Normal` primary over the quorum's log (the new primary's tail
    /// of `on_join_view`).
    ///
    /// Upstream, the View is broadcast and the pipeline rebuilt only after CTRL
    /// journal repairs complete (`repair()` → `primary_start_view_as_the_new_primary`,
    /// `replica.zig:5983`). Sans-IO journals are clean by construction, so this
    /// runs synchronously after the View broadcast: the survivor prepares
    /// (`commit_max+1..=op`) are loaded into the pipeline, we return to Normal,
    /// and our own prepare_oks are contributed toward the re-confirmation
    /// quorum for each survivor.
    ///
    /// # Panics
    ///
    /// Panics if the replica is not the newly-elected primary in `ViewChange`
    /// status, if its journal is not fully repaired, or if the journal is
    /// missing a survivor header. The journal-repaired panic is defensive
    /// (matching upstream `primary_start_view_as_the_new_primary`): dirty ops
    /// are nacked and truncated by the CTRL quorum before this runs, so it is
    /// unreachable through [`Self::on_join_view`] by construction.
    fn primary_start_view_as_the_new_primary(&mut self, now: u64) {
        assert_eq!(self.status, Status::ViewChange);
        assert!(self.is_primary());
        assert_eq!(self.view, self.log_view);
        assert!(self.join_view_quorum);
        // Upstream: `assert(self.primary_repair_pipeline() == .done)` and
        // `assert(self.primary_journal_repaired())`.
        // DEVIATION: the CTRL pipeline-repair phase (`primary_repair_pipeline`,
        // its read callbacks and timeout) collapses in sans-IO — the journal
        // holds only headers with no WAL read/completion model, so the rebuild
        // below is the synchronous `primary_repair_pipeline_done` and the
        // `.done` branch is taken unconditionally.
        assert!(self.primary_journal_repaired());
        assert_eq!(self.commit_min, self.commit_max);
        assert!(self.commit_max <= self.op);

        // Rebuild the pipeline from the survivor prepares (upstream
        // `primary_repair_pipeline_done`, sans-IO: nothing to verify).
        self.pipeline_queue = PipelineQueue::default();
        self.ok_from_all_replicas.clear();
        // A surviving op's header must string together with its predecessor
        // (upstream asserts `journal_header.parent == parent` for each op):
        let mut parent = match self.journal.header_with_op(self.commit_max) {
            Some(header) => header.checksum(),
            // DEVIATION: sans-IO journals hold no op-0 root (no superblock), so
            // a chain anchored at commit_max == 0 starts from a zero parent.
            None if self.commit_max == 0 => 0,
            None => panic!("missing committed anchor {}", self.commit_max),
        };
        for op in self.commit_max + 1..=self.op {
            let Some(header) = self.journal.header_with_op(op).copied() else {
                panic!("new primary journal must hold survivor op {op}");
            };
            assert_eq!(header.parent, parent);
            assert_eq!(self.commit_max + self.pipeline_queue.prepare_queue.len() as u64, op - 1);
            self.pipeline_queue.prepare_queue.push(PipelinePrepare {
                op,
                checksum: header.checksum(),
                client: header.client,
                acks_received: 0,
                ok_quorum_received: false,
            });
            parent = header.checksum();
            self.ok_from_all_replicas.push(0);
        }
        assert_eq!(self.commit_max + self.pipeline_queue.prepare_queue.len() as u64, self.op);

        self.transition_to_normal_from_view_change_status(self.view, now);

        // Contribute our own prepare_oks (upstream
        // `send_prepare_oks_after_view_change`, with the primary's prepare_oks
        // self-routed through `on_prepare_ok`).
        let survivors: Vec<(u64, u128)> = self
            .pipeline_queue
            .prepare_queue
            .iter()
            .map(|prepare| (prepare.op, prepare.checksum))
            .collect();
        for (op, checksum) in survivors {
            let result = self.on_prepare_ok(op, checksum, self.replica_index);
            assert!(
                matches!(result, PrepareOkResult::AckCounted | PrepareOkResult::DuplicateAck),
                "self prepare_ok cannot quorum a survivor op"
            );
        }
    }

    /// Rebuild the JoinView headers from the journal suffix.
    ///
    /// The array runs from `op` down to `commit_max` (descending op); the head
    /// entry's op is what `send_join_view` advertises as `op`.
    ///
    /// Upstream: `src/vsr/replica.zig:10254` (`update_join_view_headers`).
    ///
    /// DEVIATION: only the transition-from-normal path is ported. The
    /// `.recovering` and view-change repair paths (stitching the prior view's
    /// headers, blanks for absent entries) are deferred.
    fn update_join_view_headers(&mut self) {
        assert_eq!(self.status, Status::Normal);
        assert_eq!(self.view, self.log_view);

        self.join_view_headers.clear();
        let mut op = self.op;
        loop {
            // Only op 0 may be absent: upstream's journal always contains the
            // root prepare (from the superblock checkpoint) at op 0, while the
            // sans-IO skeleton's journal starts empty until prepares land.
            let header = if let Some(header) = self.journal.header_with_op(op) {
                *header
            } else {
                assert!(op <= self.commit_max && op == 0);
                message_header::Prepare::root(self.cluster)
            };
            self.join_view_headers.push(header);
            if op <= self.commit_max {
                break;
            }
            op -= 1;
        }
        assert!(self.join_view_headers.len() <= constants::PIPELINE_PREPARE_QUEUE_MAX as usize + 1);
        assert_eq!(self.join_view_headers[0].op, self.op);
    }

    /// Send a JoinView message to all replicas.
    ///
    /// The message advertises the replica's log (`op`, `commit_min`), the view
    /// being joined and the `log_view` the headers belong to, plus present/nack
    /// bitsets so the new primary can run CTRL repair on the uncommitted
    /// suffix.
    ///
    /// Upstream: `src/vsr/replica.zig:8783` (`send_join_view`).
    #[allow(clippy::cast_possible_truncation)] // counts are bounded by u128::BITS
    fn send_join_view(&mut self) {
        assert_eq!(self.status, Status::ViewChange);
        assert!(self.replica_count > 1);
        assert!(self.view > self.log_view);
        assert!(!self.join_view_headers.is_empty());

        // Collect nacks and presence bits per header index. Mirrors upstream:
        // - nacked headers are truncated by the new primary,
        // - present = on disk (`prepare_inhabited` + matching checksum),
        // - neither → the new primary waits for more JVs before deciding.
        let mut nack_bitset: u128 = 0;
        let mut present_bitset: u128 = 0;
        for (i, header) in self.join_view_headers.iter().enumerate() {
            assert!(i < u128::BITS as usize);
            let bit = 1_u128 << (i as u32);
            let slot = crate::journal::Journal::slot_for_op(header.op);
            let journal_header = self.journal.header_with_op(header.op);
            let dirty = self.journal.dirty.bit(slot);
            let faulty = self.journal.faulty.bit(slot);

            // Nack case 1: no prepare at all, and not due to a fault.
            if journal_header.is_none() && !faulty {
                nack_bitset |= bit;
            }
            if let Some(journal_header) = journal_header {
                // Nack case 2: in memory but not yet durable (dirty), matching.
                if journal_header.checksum() == header.checksum && dirty && !faulty {
                    nack_bitset |= bit;
                }
                // Nack case 3: a *different* prepare — safe to nack even if faulty.
                if journal_header.checksum() != header.checksum {
                    nack_bitset |= bit;
                }
                // Present: the prepare is on disk.
                let on_disk = self.journal.prepare_inhabited[slot.index]
                    && self.journal.prepare_checksums[slot.index] == header.checksum;
                if on_disk {
                    assert_eq!(journal_header.checksum(), header.checksum);
                    present_bitset |= bit;
                }
            }
        }

        // Encode the JV headers into the message body.
        let mut body = Vec::with_capacity(self.join_view_headers.len() * message_header::SIZE);
        for header in &self.join_view_headers {
            body.extend_from_slice(&header.to_wire());
        }

        let mut join_view = message_header::JoinView {
            cluster: self.cluster,
            replica: self.replica_u8(),
            view: self.view,
            log_view: self.log_view,
            checkpoint_op: 0, // TODO(port): op_checkpoint from the superblock.
            op: self.join_view_headers[0].op,
            commit_min: self.commit_min,
            nack_bitset,
            present_bitset,
            size: (message_header::SIZE + body.len()) as u32,
            ..message_header::JoinView::default()
        };
        join_view.set_checksum_body(&body);
        join_view.set_checksum();

        let mut message = crate::message::Message::new();
        message.set_body(&body);
        message.set_header(&join_view);
        self.send_queue.push(message);

        // Record our own JV for the quorum. Upstream loopbacks to `self` and
        // `on_join_view` fills the slot; the sans-IO skeleton emulates that
        // here. Only the new primary ever consumes its own slot — backups
        // never collect JVs (upstream `ignore_view_change_message` asserts
        // `join_view_from_all_replicas` is all null for them).
        if Self::primary_index_for_view(self.view, self.replica_count) == self.replica_index {
            let slot = usize::from(self.replica_index);
            assert!(self.join_view_from_all_replicas[slot].is_none());
            self.join_view_from_all_replicas[slot] = Some(crate::jv_quorum::JoinedView {
                header: join_view,
                headers: self.join_view_headers.clone(),
            });
        }
    }

    /// The replica's `view_headers` must be primed first
    /// ([`Self::primary_update_view_headers`]).
    ///
    /// # Panics
    ///
    /// Panics unless the replica is the primary with `view == log_view`.
    ///
    /// Upstream: `src/vsr/replica.zig:5795` (`create_view_message`).
    #[allow(clippy::cast_possible_truncation)] // body length is bounded by the fixed buffer
    fn make_view_message(&mut self, nonce: u128) -> crate::message::Message {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.is_primary());
        assert_eq!(self.view, self.log_view);
        assert!(self.commit_min <= self.op);

        self.primary_update_view_headers();
        assert_eq!(self.view_headers[0].op, self.op);

        // DEVIATION: upstream serializes `vsr.CheckpointState` (1024 bytes, the
        // working superblock checkpoint) as the View body prefix; sans-IO has
        // no superblock yet, so a zeroed prefix keeps the wire layout stable.
        let body_len =
            constants::CHECKPOINT_STATE_SIZE + self.view_headers.len() * message_header::SIZE;
        let mut view = message_header::View {
            cluster: self.cluster,
            replica: self.replica_u8(),
            view: self.view,
            checkpoint_op: 0, // TODO(port): op_checkpoint from the superblock.
            op: self.op,
            commit_max: self.commit_max,
            nonce,
            size: (message_header::SIZE + body_len) as u32,
            ..message_header::View::default()
        };

        let mut body = Vec::with_capacity(body_len);
        body.extend_from_slice(&[0; constants::CHECKPOINT_STATE_SIZE]);
        for header in &self.view_headers {
            body.extend_from_slice(&header.to_wire());
        }
        view.set_checksum_body(&body);
        view.set_checksum();

        let mut message = crate::message::Message::new();
        message.set_body(&body);
        message.set_header(&view);
        message
    }

    /// Build and enqueue the View message announcing the new log to the other
    /// replicas (the new primary's first broadcast of the view).
    ///
    /// # Panics
    ///
    /// Panics unless the replica is the primary with `view == log_view`.
    ///
    /// Upstream: `src/vsr/replica.zig:9499` (`primary_send_view`).
    fn primary_send_view(&mut self) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.is_primary());
        assert!(self.primary_journal_headers_repaired());

        let message = self.make_view_message(0);
        self.send_queue.push(message);
    }

    /// Whether this (primary) replica's log is a valid hash chain from
    /// `op_repair_min` up to the head — the CTRL precondition for sending Views
    /// and starting the view (`primary_send_view` must not advertise a broken
    /// log).
    ///
    /// # Panics
    ///
    /// Panics unless the replica is the primary with `view == log_view`, or a
    /// view-changing primary without a collected JoinView quorum.
    ///
    /// Upstream: `src/vsr/replica.zig:7999` (`primary_journal_headers_repaired`).
    fn primary_journal_headers_repaired(&self) -> bool {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.is_primary());
        assert_eq!(self.view, self.log_view);
        if self.status == Status::ViewChange {
            assert!(self.join_view_quorum);
        }
        self.valid_hash_chain_between(self.op_repair_min(), self.op)
    }

    /// Whether every op the journal holds in `op_repair_min()..=op` has had its
    /// prepare written to the WAL (no dirty slots) — the companion half of
    /// [`Self::primary_journal_headers_repaired`]: the new primary may only
    /// advertise and start a view once its own uncommitted prepares are durable,
    /// so a subsequent crash cannot truncate them.
    ///
    /// # Panics
    ///
    /// Panics unless the replica is the primary with `view == log_view`, a
    /// view-changing primary without a collected JoinView quorum, or holds an
    /// entry in `op_repair_min()..=op` whose header is absent.
    ///
    /// Upstream: `src/vsr/replica.zig:7983` (`primary_journal_prepares_repaired`).
    fn primary_journal_prepares_repaired(&self) -> bool {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.is_primary());
        assert_eq!(self.view, self.log_view);
        if self.status == Status::ViewChange {
            assert!(self.join_view_quorum);
        }
        for op in self.op_repair_min()..=self.op {
            let Some(header) = self.journal.header_with_op(op) else {
                panic!("primary journal missing header {op}");
            };
            let slot = self.journal.slot_for_header(header);
            if self.journal.dirty.bit(slot) {
                return false;
            }
        }
        true
    }

    /// Whether the primary's journal is fully repaired: headers hash-chain
    /// connected up to the head *and* every prepare written. Without both, the
    /// primary must not send Views or start the view as the new primary.
    ///
    /// Upstream: `src/vsr/replica.zig:7973` (`primary_journal_repaired`).
    fn primary_journal_repaired(&self) -> bool {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.is_primary());
        assert_eq!(self.view, self.log_view);
        if self.status == Status::ViewChange {
            assert!(self.join_view_quorum);
        }
        self.primary_journal_headers_repaired() && self.primary_journal_prepares_repaired()
    }

    /// Whether the journal holds a contiguous, connected hash chain over
    /// `op_min..=op_max` (each op's `parent` matching its predecessor's
    /// checksum; `op_min` is the verified anchor).
    ///
    /// # Panics
    ///
    /// Panics unless `op_max == self.op` (checking a sub-head range would risk
    /// committing a forked chain that a new primary reordered).
    ///
    /// Upstream: `src/vsr/replica.zig:11004` (`valid_hash_chain_between`).
    fn valid_hash_chain_between(&self, op_min: u64, op_max: u64) -> bool {
        assert!(op_min <= op_max);
        assert_eq!(op_max, self.op);
        // DEVIATION: upstream asserts `op_max >= op_checkpoint()` and that the
        // op after the checkpoint connects to the superblock checkpoint header;
        // sans-IO `op_checkpoint()` is 0 with no superblock to connect to.
        let Some(head) = self.journal.header_with_op(op_max).copied() else {
            return false;
        };
        let mut b = head;
        let mut op = op_max;
        while op > op_min {
            op -= 1;
            let Some(a) = self.journal.header_with_op(op).copied() else {
                return false;
            };
            assert_eq!(a.op + 1, b.op); // guaranteed by the slot arithmetic
            if a.checksum() != b.parent {
                return false;
            }
            b = a;
        }
        true
    }

    /// Send a PrepareOk for every op in `commit_max+1..=op` that we have
    /// journaled — after a view change, so the new primary can commit the
    /// (possibly uncommitted) ops that survived into the new log.
    ///
    /// Upstream: `src/vsr/replica.zig:8715` (`send_prepare_oks_after_view_change`)
    /// and :8730 (`send_prepare_oks_from`).
    fn send_prepare_oks_after_view_change(&mut self) {
        assert_eq!(self.status, Status::Normal);
        self.send_prepare_oks_from(self.commit_max + 1);
    }

    /// Send a PrepareOk for every op in `op_start..=op` that we have journaled.
    ///
    /// Upstream: `src/vsr/replica.zig:8730` (`send_prepare_oks_from`).
    fn send_prepare_oks_from(&mut self, op_start: u64) {
        let mut op = op_start;
        while op <= self.op {
            if let Some(header) = self.journal.header_with_op(op).copied() {
                self.send_prepare_ok(&header);
            }
            op += 1;
        }
    }

    /// Send a PrepareOk acknowledging `prepare` to the primary.
    ///
    /// Upstream: `src/vsr/replica.zig:7600` (`send_prepare_ok`).
    fn send_prepare_ok(&mut self, prepare: &message_header::Prepare) {
        assert!(prepare.valid_checksum());
        assert!(prepare.invalid().is_none());
        assert_eq!(prepare.command, Command::Prepare);

        let mut prepare_ok = message_header::PrepareOk {
            cluster: self.cluster,
            replica: self.replica_u8(),
            view: self.view,
            client: prepare.client,
            // The previous prepare's checksum, and the checksum being acked.
            parent: prepare.parent,
            prepare_checksum: prepare.checksum,
            checkpoint_id: 0, // TODO(port): working superblock checkpoint_id.
            commit_min: self.commit_min,
            op: prepare.op,
            timestamp: prepare.timestamp,
            request: prepare.request,
            operation: prepare.operation,
            ..message_header::PrepareOk::default()
        };
        assert!(u16::from(prepare_ok.replica) < self.replica_count);
        prepare_ok.set_checksum_body(&[]);
        prepare_ok.set_checksum();
        self.enqueue_header(&prepare_ok);
    }

    /// The journal header for `op`, falling back to the root prepare when
    /// `op == 0`: the sans-IO journal may not hold the root until the first
    /// prepare lands, whereas upstream's always does (superblock checkpoint).
    fn journal_header_or_root(&self, op: u64) -> message_header::Prepare {
        if let Some(header) = self.journal.header_with_op(op) {
            *header
        } else {
            assert_eq!(op, 0);
            message_header::Prepare::root(self.cluster)
        }
    }

    /// Rebuild [`Self::view_headers`]: the unbroken journal suffix from the
    /// head op downward (capacity permitting), plus at most two
    /// checkpoint-boundary headers to help backups repair across `op_hook`s.
    ///
    /// # Panics
    ///
    /// Panics if the journal is missing a header within the suffix range (an
    /// invariant violation; upstream `header_with_op(op).?` un-wraps the same).
    ///
    /// Upstream: `src/vsr/replica.zig:5939` (`primary_update_view_headers`)
    /// and :5853 (`update_view_headers`).
    fn primary_update_view_headers(&mut self) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert_eq!(self.view, self.log_view);
        assert!(self.is_primary());

        let op_max = self.op;
        // The oldest op that may still be in the journal
        // (`op_max + 1 -| journal_slot_count`, equivalently).
        let journal_start = op_max.saturating_sub(u64::from(constants::JOURNAL_SLOT_COUNT) - 1);
        let header_break = self.journal.find_latest_headers_break_between(journal_start, op_max);
        // The suffix must be an unbroken chain; a break would bound it further.
        let op_min = match header_break {
            Some(break_range) => break_range.op_max + 1,
            None => journal_start,
        };

        self.view_headers.clear();
        let mut op = op_max + 1;
        while op > 0 && self.view_headers.len() < constants::VIEW_CHANGE_HEADERS_SUFFIX_MAX as usize
        {
            op -= 1;
            self.view_headers.push(self.journal_header_or_root(op));
        }
        assert!(self.view_headers.len() + 2 <= constants::VIEW_HEADERS_MAX as usize);

        // Include the headers at the preceding checkpoint triggers (`op_hook`)
        // so backups can repair across a checkpoint boundary (at most 2).
        for op_hook in [
            op_max.saturating_sub(constants::VSR_CHECKPOINT_OPS as u64),
            op_max.saturating_sub(constants::VSR_CHECKPOINT_OPS as u64 * 2),
        ] {
            if op > op_hook && op_hook >= op_min {
                op = op_hook;
                self.view_headers.push(self.journal_header_or_root(op));
            }
        }

        // Ops run strictly descending (the checkpoint hooks may jump down).
        let mut previous: Option<u64> = None;
        for header in &self.view_headers {
            if let Some(previous) = previous {
                assert!(header.op < previous);
            }
            previous = Some(header.op);
        }
    }

    /// Called every tick — advances all timeouts and the fault detector.
    ///
    /// Upstream: `src/vsr/replica.zig:1532` (`tick`).
    pub fn tick(&mut self, now: u64) {
        self.monotonic_now = now;
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
        if self.journal_repair_timeout.tick() {
            self.on_journal_repair_timeout();
        }
        if self.grid_repair_timeout.tick() {
            self.on_grid_repair_timeout();
        }
    }

    /// Timeout: broadcast a Ping to all replicas.
    ///
    /// Upstream: `src/vsr/replica.zig:3567` (`on_ping_timeout`).
    fn on_ping_timeout(&mut self) {
        self.ping_timeout.reset(constants::PING_TIMEOUT);

        // DEVIATION: upstream uses `view_durable()` (replica.zig:3574) and
        // `self.release`, and `checkpoint_id` comes from the superblock's
        // checkpoint; the sans-IO replica has no superblock or multiversion
        // support (matching the Prepare header construction). `checkpoint_id = 0`
        // is not validated by `Ping::invalid_header`.
        let mut ping = message_header::Ping {
            cluster: self.cluster,
            replica: self.replica_u8(),
            view: self.view,
            release: crate::multiversion::Release::MINIMUM,
            checkpoint_id: 0,
            checkpoint_op: self.op_checkpoint(),
            // Upstream samples `clock.monotonic()`; the owner's tick provides it.
            ping_timestamp_monotonic: self.monotonic_now,
            // DEVIATION: upstream bundles the multiversion release list (up to
            // `vsr_releases_max`); sans-IO builds with the single minimum release.
            release_count: 1,
            ..message_header::Ping::default()
        };

        // Body: up to `vsr_releases_max` Release entries, the bundled minimum
        // first and the rest zeroed (upstream `ping_message_release_list`).
        let body_len = size_of::<crate::multiversion::Release>()
            * usize::try_from(constants::VSR_RELEASES_MAX)
                .unwrap_or_else(|_| unreachable!("vsr_releases_max fits usize"));
        let mut body = vec![0_u8; body_len];
        body[..size_of::<crate::multiversion::Release>()]
            .copy_from_slice(&crate::multiversion::Release::MINIMUM.value.to_le_bytes());
        ping.size = u32::try_from(message_header::SIZE + body.len())
            .unwrap_or_else(|_| unreachable!("ping size is far below u32::MAX"));
        ping.set_checksum_body(&body);
        ping.set_checksum();

        // Broadcast to every other replica (upstream
        // `send_message_to_other_replicas_and_standbys`, replica.zig:3605).
        for _ in 1..self.replica_count {
            let mut message = crate::message::Message::new();
            message.set_header(&ping);
            message.set_body(&body);
            self.send_queue.push(message);
        }
    }

    /// Timeout: the primary re-sends pending prepares or issues a Commit
    /// heartbeat.
    ///
    /// Re-sends the first prepare without a full prepare_ok quorum to every
    /// backup that has not yet acked it, walking the replicas in ring order
    /// (a lost prepare or prepare_ok is thus retried at a fixed cadence).
    ///
    /// # Panics
    ///
    /// Panics if the replica is not a Normal primary, or if a pending prepare
    /// is missing from the primary's own journal.
    ///
    /// Upstream: `src/vsr/replica.zig:3608` (`on_prepare_timeout`).
    pub fn on_prepare_timeout(&mut self) {
        assert_eq!(self.status, Status::Normal);
        assert!(self.is_primary());
        self.prepare_timeout.reset(constants::PREPARE_TIMEOUT);

        let Some((slot, prepare)) = self.primary_pipeline_pending() else {
            // Nothing unquorum'd is pending; stop ticking until the next prepare.
            self.prepare_timeout.stop();
            return;
        };

        // The peers that have not yet acked this prepare, in ring order
        // (upstream builds `waiting` at replica.zig:3627-3639).
        let mut waiting = Vec::new();
        for ring in 1..self.replica_count {
            let replica = (self.replica_index + ring) % self.replica_count;
            if self.ok_from_all_replicas[slot] & (1 << replica) == 0 {
                waiting.push(replica);
            }
        }
        // The primary's own prepare_ok is always set (see
        // `primary_pipeline_prepare`), so a pending prepare necessarily has
        // some unacked backup.
        assert!(!waiting.is_empty(), "pending prepare has a quorum already");

        // DEVIATION: upstream re-sends `prepare.message`, which carries the
        // request body; sans-IO bodies are deferred, so the header-only
        // prepare is replayed (receivers dispatch on the header alone).
        let prepare_op = prepare.op;
        let Some(prepare_header) = self.journal.header_with_op(prepare_op) else {
            panic!("primary's journal must hold its own pending prepare op {prepare_op}")
        };
        let mut message = crate::message::Message::new();
        message.set_header(prepare_header);
        for _ in 0..waiting.len() {
            self.send_queue.push(message.clone());
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

/// Result of accepting a Prepare message (`on_prepare`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnPrepareResult {
    /// The prepare advanced the log (by one op) and was recorded in the journal.
    Accepted,
    /// `header.op <= self.op` — duplicate or already superseded.
    Stale,
    /// `header.op > self.op + 1` — a gap; `jump_to_newer_op` is deferred.
    FutureOp,
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

impl PipelineQueue {
    /// Whether the prepare queue has reached capacity (upstream
    /// `prepare_queue.full()`).
    #[must_use]
    pub fn prepare_queue_full(&self) -> bool {
        self.prepare_queue.len() >= constants::PIPELINE_PREPARE_QUEUE_MAX as usize
    }
}

/// A prepare in the pipeline.
#[derive(Clone, Copy, Debug)]
pub struct PipelinePrepare {
    pub op: u64,
    pub checksum: u128,
    /// The client the prepare serves (used to wait for a session's register to
    /// commit — upstream `pipeline.queue.message_by_client`).
    pub client: u128,
    /// Number of prepare_ok responses received.
    pub acks_received: u16,
    /// Whether a quorum of prepare_ok messages has been received.
    pub ok_quorum_received: bool,
}

/// A client request queued on the primary.
#[derive(Clone, Copy, Debug)]
pub struct PipelineRequest {
    pub client: u128,
    pub request: u32,
    pub request_checksum: u128,
    pub operation: crate::Operation,
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
    /// Dispatches an incoming message to the appropriate handler.
    ///
    /// `now` is the monotonic clock time (milliseconds) used by handlers that
    /// feed the fault detector; the sans-IO skeleton has no wall clock.
    ///
    /// Upstream: message receive path (`message_bus` → per-command handlers).
    pub fn on_message(&mut self, message: &crate::message::Message, now: u64) {
        self.monotonic_now = now;
        // Parse the base header frame (no command-specific validation).
        let some_frame: Option<&[u8; message_header::SIZE]> = message.frame().try_into().ok();
        let Some(frame) = some_frame else {
            return;
        };
        let Some(header) = message_header::Header::from_wire(frame) else {
            return;
        };

        // Validate cluster ID.
        if header.cluster != self.cluster {
            return; // Drop message from wrong cluster.
        }

        // Validate checksum (upstream verifies `header.valid_checksum()`).

        match header.command {
            Command::Request => self.on_request(&header),
            Command::Prepare => self.on_prepare_message(&header),
            Command::Commit => self.on_commit(&header, now),
            Command::ExitView => self.on_exit_view(&header),
            Command::JoinView => self.on_join_view(message, now),
            Command::View => self.on_view(message, now),
            Command::PrepareOk => self.on_prepare_ok_message(message),
            Command::GetPrepare => self.on_get_prepare(&header),
            Command::GetHeaders => self.on_get_headers(&header),
            Command::Headers => self.on_headers(message),
            Command::GetView => self.on_get_view(message),
            Command::GetBlocks => self.on_get_blocks(message),
            Command::Block => self.on_block(message),
            Command::Ping => self.on_ping(message),
            Command::Pong => self.on_pong(message),
            Command::PingClient => self.on_ping_client(message),
            Command::Reply => self.on_reply(message),
            Command::GetReply => self.on_get_reply(message),
            // PongClient/Eviction are client-directed and misdirected here
            // (upstream replica.zig:1822 warns and drops); the remaining
            // commands are still unported. Drop all of them.
            Command::PongClient
            | Command::Eviction
            | Command::Deprecated12
            | Command::Deprecated21
            | Command::Deprecated22
            | Command::Deprecated23
            | Command::Reserved => {}
        }
    }

    /// Handle a `GetBlocks` request from a remote replica: read each requested
    /// block and reply with a `Block` message when each read completes.
    ///
    /// # Panics
    /// Panics if the request body is empty or not aligned to `BlockRequest`
    /// (callers validate `GetBlocks::invalid_header` before dispatching).
    ///
    /// Upstream: `src/vsr/replica.zig:3324` (`on_get_blocks`).
    pub fn on_get_blocks(&mut self, message: &crate::message::Message) {
        let Some(get_blocks) = message.header::<GetBlocks>() else {
            return;
        };
        if !get_blocks.valid_checksum() || !get_blocks.valid_checksum_body(message.body_used()) {
            return;
        }
        if get_blocks.invalid_header().is_some() {
            return;
        }

        if get_blocks.replica == self.replica_u8() {
            return; // Misdirected message (self).
        }

        // Upstream also drops for `standby()`; the sans-IO port has no standby flag.
        if self.grid.is_none() {
            return; // Upstream `grid.callback == .cancel`.
        }

        let destination = u16::from(get_blocks.replica);
        let request_bytes = message.body_used();
        assert!(!request_bytes.is_empty()); // Guaranteed by `invalid_header`.

        // Disjoint field borrows: the grid and its storage are separate from the
        // serve-read ledger.
        let (Some(grid), Some(storage)) = (&mut self.grid, &mut self.grid_storage) else {
            return;
        };
        for request in request_bytes.as_chunks::<{ size_of::<crate::BlockRequest>() }>().0 {
            let mut checksum_bytes = [0_u8; 16];
            checksum_bytes.copy_from_slice(&request[0..16]);
            let mut address_bytes = [0_u8; 8];
            address_bytes.copy_from_slice(&request[16..24]);
            let block_checksum = u128::from_le_bytes(checksum_bytes);
            let block_address = u64::from_le_bytes(address_bytes);

            if self.grid_serve_reads.iter().any(|read| {
                read.destination == destination
                    && read.address == block_address
                    && read.checksum == block_checksum
            }) {
                continue; // Already reading this block for this replica.
            }
            if self.grid_serve_reads.len() >= usize::from(constants::GRID_REPAIR_READS_MAX) {
                return; // Upstream: ignore the remaining requests.
            }

            let token = grid.read_block(
                storage,
                block_address,
                block_checksum,
                true, // Coherent read (local storage only).
                ReadOptions { cache_read: true, cache_write: false },
            );
            self.grid_serve_reads.push(GridServeRead {
                token,
                destination,
                address: block_address,
                checksum: block_checksum,
            });
        }
    }

    /// Drive the mounted grid forward and handle its completion events.
    ///
    /// # Panics
    /// Panics if the grid is not mounted (no-op: nothing to poll).
    pub fn poll_grid(&mut self) {
        let events = {
            let (Some(grid), Some(storage)) = (&mut self.grid, &mut self.grid_storage) else {
                return;
            };
            grid.poll(storage);
            grid.take_events()
        };
        for event in events {
            self.on_grid_event(event);
        }
    }

    /// Drive pending client-reply writes and reads to completion (upstream:
    /// the storage callback halves of `client_replies`, settled in the IO
    /// event loop).
    ///
    /// Completed reply-repair reads are dispatched here: a found reply is sent
    /// to the requesting replica; a corrupt or unexpected reply marks the slot
    /// faulty so it can be repaired via [`Self::on_reply`].
    ///
    /// DEVIATION: the sans-IO replica has no storage until one is mounted via
    /// `grid_storage` (see that field's DEVIATION note); without it there is
    /// nothing to poll.
    ///
    /// # Panics
    /// Panics if a reply-repair read's destination replica is missing.
    pub fn poll_client_replies(&mut self) {
        let Some(storage) = self.grid_storage.as_mut() else {
            return;
        };
        self.client_replies.poll(storage);
        for event in self.client_replies.take_events() {
            match event {
                crate::client_replies::Event::ReadReply {
                    slot,
                    outcome,
                    destination_replica,
                    ..
                } => {
                    match outcome {
                        crate::client_replies::ReadOutcome::Found(reply)
                        | crate::client_replies::ReadOutcome::ResolvedByWrite(reply) => {
                            // `Some(destination)` means the read was issued for
                            // a GetReply from that replica (send it there);
                            // `None` means a duplicate request's repeat-reply
                            // (client-directed — send it to the client).
                            match destination_replica {
                                Some(destination) => {
                                    assert!(u16::from(destination) < self.replica_count);
                                    self.send_reply_to_replica(&reply);
                                }
                                None => self.send_reply_message_to_client(&reply),
                            }
                        }
                        crate::client_replies::ReadOutcome::Corrupt
                        | crate::client_replies::ReadOutcome::Unexpected => {
                            self.client_replies.mark_faulty(slot);
                        }
                    }
                }
                crate::client_replies::Event::Ready
                | crate::client_replies::Event::CheckpointDone => {
                    // These fire only when `ready()`/`checkpoint()` waiters are
                    // registered, which the sans-IO replica does not do yet.
                }
            }
        }
    }

    fn on_grid_event(&mut self, event: Event) {
        if let Event::ReadDone { token, address, checksum, result, valid_location } = event {
            self.on_grid_read_done(token, address, checksum, result, valid_location);
        }
        // Other completions are drained by the owner directly.
    }

    /// Reply to a `GetBlocks` read with the block contents.
    ///
    /// Upstream: `src/vsr/replica.zig:3410` (`on_get_blocks_read_block`).
    fn on_grid_read_done(
        &mut self,
        token: u32,
        address: u64,
        checksum: u128,
        result: ReadBlockResult,
        valid_location: Option<u32>,
    ) {
        let Some(index) = self.grid_serve_reads.iter().position(|read| read.token == token) else {
            return;
        };
        let read = self.grid_serve_reads.remove(index);

        if result != ReadBlockResult::Valid {
            return;
        }
        let Some(location) = valid_location else {
            return;
        };

        // Upstream asserts `read.message.header.checksum == grid_read.checksum`
        // — the reply *is* the block, so that is the block's own content checksum.
        assert_eq!(read.checksum, checksum);
        assert_eq!(read.address, address);

        let block_bytes = {
            let grid =
                self.grid.as_mut().unwrap_or_else(|| unreachable!("grid mounted for a serve read"));
            assert_eq!(grid.block(location).len(), constants::BLOCK_SIZE);
            grid.block(location).to_vec()
        };

        // The reply *is* the block: the block's own header (command=block,
        // cluster, address, checksum) occupies the message header slot.
        assert_eq!(block_bytes.len(), constants::BLOCK_SIZE);
        let mut reply = crate::message::Message::new();
        reply.buffer_mut()[..constants::BLOCK_SIZE].copy_from_slice(&block_bytes);
        self.send_queue.push(reply);
    }

    /// Send a stored client reply to a replica that requested it.
    ///
    /// DEVIATION: upstream `send_message_to_replica` (replica.zig:8972) opens
    /// a connection to the destination. The sans-IO port has no message bus,
    /// so the reply is pushed to the shared outbox (`send_queue`) and the
    /// integration layer must pair it with the `GetReply`'s `replica` field.
    fn send_reply_to_replica(&mut self, reply: &crate::message::Message) {
        assert_eq!(
            reply.header::<message_header::Reply>().map(|header| header.command),
            Some(crate::command::Command::Reply),
        );
        self.send_queue.push(reply.clone());
    }

    /// Serve a `GetReply` for a client's committed reply (reply repair).
    ///
    /// Responds with the stored `command=reply` message when the reply is
    /// available and valid; otherwise stays silent (upstream docs:
    /// "Protocol: Repair Client Replies").
    ///
    /// The reply is served from RAM when a write to its slot is still in
    /// flight; otherwise it is read from the client-replies zone, and the
    /// read's outcome (`Event::ReadReply`) is dispatched by
    /// [`Self::poll_client_replies`].
    ///
    /// Upstream: `src/vsr/replica.zig:3213` (`on_get_reply`).
    ///
    /// # Panics
    /// Panics if the session entry's client does not match the `GetReply`, the
    /// session holds a body-less reply, or the reply op does not match the
    /// `GetReply`'s (all asserted invariants of the session table, upstream
    /// replica.zig:3225-3238).
    pub fn on_get_reply(&mut self, message: &crate::message::Message) {
        let Some(get_reply) = message.header::<message_header::GetReply>() else {
            return;
        };
        if !get_reply.valid_checksum() || !get_reply.valid_checksum_body(message.body_used()) {
            return;
        }
        if get_reply.invalid_header().is_some() {
            return;
        }
        // `GetReply::invalid_header` guarantees `reply_client != 0`.

        // `ignore_repair_message` (replica.zig:6106): reply repair runs from
        // normal/view-change status, and `GetReply`'s `view` is always 0 so
        // the view gates do not apply.
        if !matches!(self.status, Status::Normal | Status::ViewChange) {
            return;
        }
        if get_reply.replica == self.replica_u8() {
            return; // Misdirected (upstream warns and drops).
        }

        let Some(entry) = self.client_sessions.get(get_reply.reply_client) else {
            return; // Client not in the sessions table.
        };
        assert_eq!(entry.header.client, get_reply.reply_client);

        if entry.header.checksum != get_reply.reply_checksum {
            return; // The session has advanced past (or never held) this reply.
        }
        // Body-less replies live only in the sessions trailer, never in the
        // client-replies zone, and are requested by checksum alone:
        assert_ne!(entry.header.size, message_header::SIZE_U32);
        assert_eq!(entry.header.op, get_reply.reply_op);

        let Some(slot) = self.client_sessions.get_slot_for_client(get_reply.reply_client) else {
            unreachable!("session entry exists (fetched above)");
        };

        // Serve from RAM if a write to the slot is still in flight
        // (upstream `read_reply_sync`, replica.zig:6665).
        if let Some(in_flight) = self.client_replies.write_in_flight_latest(slot, entry).cloned() {
            self.send_reply_to_replica(&in_flight);
            return;
        }

        // Otherwise queue an async read; the reply is sent once the read
        // completes (upstream `read_reply`, replica.zig:3251).
        let Some(storage) = self.grid_storage.as_mut() else {
            // DEVIATION: the sans-IO replica has no storage until one is
            // mounted via `grid_storage` (see that field's DEVIATION note).
            return;
        };
        if let Err(crate::client_replies::ReadError::Busy) =
            self.client_replies.read_reply(storage, slot, entry, Some(get_reply.replica))
        {
            // Upstream: ignore the GetReply when client_replies is busy.
        }
    }

    /// Handle a reply sent by another replica (reply repair, the receiving
    /// half of `on_get_reply`): rewrite a corrupt/missing reply into the
    /// client-replies zone (upstream `src/vsr/replica.zig:2342` `on_reply`).
    pub fn on_reply(&mut self, message: &crate::message::Message) {
        let Some(reply) = message.header::<message_header::Reply>() else {
            return;
        };
        if !reply.valid_checksum() || !reply.valid_checksum_body(message.body_used()) {
            return;
        }
        if reply.invalid_header().is_some() {
            return;
        }

        // The reply is only useful if it matches the session table's expected
        // reply *and* the slot is known to be corrupt/missing:
        let Some(entry) = self.client_sessions.get(reply.client) else {
            return; // Client not in the sessions table.
        };
        if entry.header.checksum != reply.checksum {
            return; // The session has a different (newer/older) reply in mind.
        }
        let Some(slot) = self.client_sessions.get_slot_for_header(&reply) else {
            unreachable!("session entry exists (fetched above)");
        };
        if !self.client_replies.reply_is_faulty(slot) {
            return; // Nothing to repair.
        }
        if !self.client_replies.ready_sync() {
            return; // Upstream: ignore while busy.
        }

        let Some(storage) = self.grid_storage.as_mut() else {
            // DEVIATION: see `grid_storage`'s DEVIATION note.
            return;
        };
        let reply_message = message.clone();
        self.client_replies.write_reply(
            storage,
            slot,
            reply_message,
            crate::client_replies::WriteTrigger::Repair,
        );
    }

    /// Request missing blocks from `destination` (upstream
    /// `src/vsr/replica.zig:11229` `send_get_blocks`).
    ///
    /// # Panics
    /// Panics if the destination is self, the grid is not mounted, or (in a
    /// multi-replica cluster) the destination's repair budget is exhausted.
    #[allow(clippy::field_reassign_with_default)] // typed header: reserved fields private
    pub fn send_get_blocks(&mut self, destination: u8) {
        assert!(self.grid_repair_timeout.active);
        assert!(self.grid.is_some()); // Upstream `grid.callback != .cancel`.
        assert_ne!(destination, self.replica_u8());

        if self.replica_count > 1 {
            assert!(
                self.grid_repair_message_budget.budget_available(destination)
                    >= u32::from(constants::GRID_REPAIR_REQUEST_MAX)
            );
        }

        // The slot split in upstream trades requests between
        // `blocks_missing.faulty_blocks` and `read_global_queue`. This port has
        // no BlockMissing bookkeeping (DEVIATION), so only parked coherent
        // reads are requested, and the whole buffer goes to `read_global_queue`.
        let now = Instant { ns: self.monotonic_now };
        let mut requested = Vec::new();
        {
            let grid = self
                .grid
                .as_ref()
                .unwrap_or_else(|| unreachable!("grid is mounted (asserted above)"));
            for (address, checksum) in grid
                .global_reads()
                .into_iter()
                .take(usize::from(constants::GRID_REPAIR_REQUEST_MAX))
            {
                assert!(!grid.free_set_is_free(address)); // Upstream replica.zig:11321.
                if self.grid_repair_message_budget.decrement(
                    crate::BlockReference { checksum, address },
                    destination,
                    now,
                ) {
                    requested.push((address, checksum));
                }
            }
        }

        if requested.is_empty() {
            return;
        }

        let body: Vec<u8> = requested
            .iter()
            .flat_map(|(address, checksum)| {
                let mut request = vec![0_u8; size_of::<crate::BlockRequest>()];
                request[0..16].copy_from_slice(&checksum.to_le_bytes());
                request[16..24].copy_from_slice(&address.to_le_bytes());
                request
            })
            .collect();

        let mut header = message_header::GetBlocks::default();
        header.cluster = self.cluster;
        header.replica = self.replica_u8();
        header.size = u32::try_from(message_header::SIZE + body.len())
            .unwrap_or_else(|_| unreachable!("get_blocks size is far below u32::MAX"));
        header.set_checksum_body(&body);
        header.set_checksum();

        let mut message = crate::message::Message::new();
        message.set_header(&header);
        message.set_body(&body);
        self.send_queue.push(message);
    }

    /// Handle a repair Block from a remote replica: fulfill parked coherent
    /// reads and persist the block (upstream `src/vsr/replica.zig:3453`
    /// `on_block`).
    pub fn on_block(&mut self, message: &crate::message::Message) {
        let Some(block) = message.header::<message_header::Block>() else {
            return;
        };
        if !block.valid_checksum() || !block.valid_checksum_body(message.body_used()) {
            return;
        }
        if block.invalid_header().is_some() {
            return;
        }

        if self.grid.is_none() {
            return; // Upstream `grid.callback == .cancel`.
        }

        // The block may be shorter than `block_size`; pad with zeros so the
        // grid can store a full block (upstream copies into a full `BlockPtr`).
        let size = block.size as usize;
        let mut full_block = vec![0_u8; constants::BLOCK_SIZE];
        full_block[..size].copy_from_slice(&message.buffer()[..size]);
        let address = block.address;
        let checksum = block.checksum;

        let fulfilled = {
            let (Some(grid), Some(storage)) = (&mut self.grid, &mut self.grid_storage) else {
                unreachable!("grid is mounted above");
            };
            let fulfilled = grid.fulfill_block(&full_block);
            if fulfilled {
                // Persist the repair so the block survives restarts.
                // DEVIATION: upstream writes only when the block was in
                // `blocks_missing.faulty_blocks` (`repair_block_waiting`);
                // this port has no BlockMissing tracking, so every fulfilled
                // repair block is durably written.
                let location = grid.get_block();
                grid.block_mut(location).copy_from_slice(&full_block);
                let _ = grid.repair_block(storage, location);
            }
            fulfilled
        };

        if fulfilled {
            self.grid_repair_message_budget.increment(crate::BlockReference { checksum, address });
            if let Some(destination) =
                self.grid_repair_message_budget.next_destination(&mut self.prng)
            {
                self.send_get_blocks(destination);
            }
        }
    }

    /// Periodic grid-repair driver: re-arm the cadence, reap expired budget
    /// requests, and request blocks from a random replica with budget left.
    ///
    /// Upstream: `src/vsr/replica.zig:3815` (`on_grid_repair_timeout`).
    pub fn on_grid_repair_timeout(&mut self) {
        // Re-arm *before* requesting: `Timeout::tick` clears `active` when it
        // fires (mirroring `on_journal_repair_timeout`, which resets before
        // calling `repair()`).
        self.grid_repair_timeout.reset(constants::GRID_REPAIR_TIMEOUT);
        // Upstream uses `reset_with_jitter`; sans-IO is deterministic, so we
        // re-arm with a fixed cadence (DEVIATION).
        self.grid_repair_message_budget.reap_expired_requests(Instant { ns: self.monotonic_now });

        if self.grid.is_some()
            && let Some(destination) =
                self.grid_repair_message_budget.next_destination(&mut self.prng)
        {
            self.send_get_blocks(destination);
        }
    }

    /// Whether a JoinView/View message should be dropped before processing.
    ///
    /// # Panics
    ///
    /// Panics if the message is neither a JoinView nor a View, or if the typed
    /// header cannot be decoded (callers must have validated command + checksum
    /// first).
    ///
    /// Upstream: `src/vsr/replica.zig:6805`
    /// (`ignore_view_change_message`).
    fn ignore_view_change_message(&self, message: &crate::message::Message) -> bool {
        // Callers must have validated command + checksum on the frame already
        // (via `on_message` / the per-command guards); a decodable frame is an
        // internal invariant, but a malformed one is safely ignored.
        let Ok(frame) = <&[u8; message_header::SIZE]>::try_from(message.frame()) else {
            return true;
        };
        let Some(base) = message_header::Header::from_wire(frame) else {
            return true;
        };
        assert!(
            base.command == Command::JoinView || base.command == Command::View,
            "only view-change messages reach ignore_view_change_message"
        );
        assert_ne!(self.status, Status::Recovering); // Single node clusters have no view changes.
        assert!(u16::from(base.replica) < self.replica_count);

        if base.view < self.view {
            return true; // Older view.
        }

        match base.command {
            Command::View => {
                let Some(view) = message.header::<message_header::View>() else {
                    return true;
                };
                // This may be caused by faults in the network topology.
                if u16::from(base.replica) == self.replica_index {
                    return true;
                }
                // Syncing replicas must be careful about receiving View messages,
                // since they may have fast-forwarded their commit_max via their
                // checkpoint target (never the case sans-IO, checkpoint_op = 0).
                if view.commit_max < self.op_checkpoint() {
                    return true;
                }
                false
            }
            Command::JoinView => {
                let Some(join_view) = message.header::<message_header::JoinView>() else {
                    return true;
                };
                assert!(join_view.view > 0, "the initial view is already zero");
                // DEVIATION: upstream ignores JVs from standbys (`standby()`),
                // which the sans-IO skeleton does not model.
                if self.status == Status::RecoveringHead {
                    return true;
                }
                if self.status == Status::Normal && join_view.view == self.view {
                    return true; // View already started.
                }
                if self.join_view_quorum {
                    return true; // Quorum received already.
                }
                if Self::primary_index_for_view(self.view, self.replica_count) != self.replica_index
                {
                    // A backup: it staged its own JV for broadcast in
                    // `transition_to_view_change_status` but does not process
                    // incoming JVs — it awaits a View from the new primary.
                    for jv in &self.join_view_from_all_replicas {
                        assert!(jv.is_none());
                    }
                    return true;
                }
                false
            }
            _ => unreachable!(),
        }
    }

    /// Handle a JoinView message — only meaningful for the primary of
    /// `self.view` during a view change.
    ///
    /// Collects the JVs into [`Self::join_view_from_all_replicas`], runs the
    /// `JVQuorum.quorum_headers` algorithm, and once it completes: installs the
    /// canonical log, advances `log_view`, and broadcasts a View message.
    ///
    /// # Panics
    ///
    /// Panics on invariant violations (e.g. a broadcast JV from a replica
    /// outside the cluster, or a JV for a view we have moved past).
    ///
    /// Upstream: `src/vsr/replica.zig:2608` (`on_join_view`),
    /// `primary_receive_join_view` (:4045), and
    /// `primary_set_log_from_join_view_messages` (:9705).
    pub fn on_join_view(&mut self, message: &crate::message::Message, now: u64) {
        let Some(join_view) = message.header::<message_header::JoinView>() else {
            return; // Command mismatch or malformed header.
        };
        // DEVIATION: upstream validates checksums at the message_bus receive
        // path; sans-IO messages are constructed locally, so guard anyway.
        if !join_view.valid_checksum() || !join_view.valid_checksum_body(message.body_used()) {
            return;
        }
        if self.ignore_view_change_message(message) {
            return;
        }

        assert!(self.replica_count > 1);
        assert_eq!(self.status, Status::ViewChange);
        assert_eq!(join_view.view, self.view);
        assert!(!self.join_view_quorum);
        assert!(u16::from(join_view.replica) < self.replica_count);

        let body = message.body_used();
        self.primary_receive_join_view(&join_view, body);

        // The new primary's own slot is filled by the broadcast loopback:
        assert!(self.join_view_from_all_replicas[usize::from(self.replica_index)].is_some());
        crate::jv_quorum::verify(&self.join_view_from_all_replicas);

        let quorum = self.quorum();
        let result = crate::jv_quorum::quorum_headers(
            &self.join_view_from_all_replicas,
            crate::jv_quorum::QuorumOptions {
                replica_count: self.replica_count,
                quorum_view_change: quorum.view_change,
                quorum_nack_prepare: quorum.nack_prepare,
            },
        );
        let op_head = match result {
            crate::jv_quorum::QuorumHeadersResult::AwaitingQuorum => return,
            crate::jv_quorum::QuorumHeadersResult::AwaitingRepair
            | crate::jv_quorum::QuorumHeadersResult::CompleteInvalid => {
                // TODO(port): upstream logs these and starts `journal`-request
                // repairs (`primary_log_join_view_quorum` + `repair()`); sans-IO
                // journals have no faults to repair, so there is nothing to do.
                return;
            }
            crate::jv_quorum::QuorumHeadersResult::CompleteValid { op_head, .. } => op_head,
        };

        // TODO(port): an op_checkpoint lagging more than `replica_count - 1`
        // views forfeits the view change to let a checkpoint-ahead replica be
        // primary (upstream `on_join_view`). Deferred: no superblock yet.

        assert!(!self.join_view_quorum);
        self.join_view_quorum = true;

        self.primary_set_log_from_join_view_messages();

        // Still ViewChange, but our prior log_view headers may have been
        // replaced; disambiguate them for a subsequent JV if we never reach
        // Normal (upstream `replica.zig:2732`).
        self.log_view = self.view;

        assert_eq!(self.op, op_head);
        assert!(self.op >= self.commit_max);
        assert!(
            self.prepare_timestamp
                >= match self.journal.header_with_op(self.op) {
                    Some(h) => h.timestamp,
                    None => panic!("journal missing head op {}", self.op),
                }
        );

        // DEVIATION: upstream keeps the replica in `view_change` and drives the
        // CTRL loop from `repair()` (waiting for the journal's prepares, then
        // `primary_repair_pipeline` → `primary_start_view_as_the_new_primary`).
        // Sans-IO journals are clean by construction and the pipeline rebuild is
        // synchronous, so we broadcast the View (only once the log's hash chain
        // is verified, via `primary_send_view`) and transition immediately.
        self.primary_send_view();
        self.primary_start_view_as_the_new_primary(now);
    }

    /// Decode the body of a JoinView message into prepare headers.
    ///
    /// Returns `None` if the body is empty or not a whole number of headers.
    fn decode_join_view_headers(body: &[u8]) -> Option<Vec<message_header::Prepare>> {
        if body.is_empty() || !body.len().is_multiple_of(message_header::SIZE) {
            return None;
        }
        let mut headers = Vec::with_capacity(body.len() / message_header::SIZE);
        for frame in body.as_chunks::<{ message_header::SIZE }>().0 {
            headers.push(message_header::Prepare::from_wire(frame)?);
        }
        Some(headers)
    }

    /// Record a received JoinView in the quorum slot of its sender.
    ///
    /// A duplicate JV is kept only if it is strictly newer (higher checkpoint,
    /// then commit) than the one already recorded — upstream
    /// `replica.zig:4061`.
    fn primary_receive_join_view(&mut self, join_view: &message_header::JoinView, body: &[u8]) {
        let slot = usize::from(join_view.replica);
        if let Some(existing) = &self.join_view_from_all_replicas[slot] {
            let replace = existing.header.checkpoint_op < join_view.checkpoint_op
                || (existing.header.checkpoint_op == join_view.checkpoint_op
                    && existing.header.commit_min < join_view.commit_min);
            if !replace {
                return; // Keep the more up-to-date duplicate.
            }
        }
        let Some(headers) = Self::decode_join_view_headers(body) else {
            return; // Malformed body.
        };
        self.join_view_from_all_replicas[slot] =
            Some(crate::jv_quorum::JoinedView { header: *join_view, headers });
    }

    /// Install the new view's log from the JV quorum (as the new primary).
    ///
    /// Sets the head op, truncates ops above it, installs the canonical headers
    /// (high → low op), and preserves the highest possible `commit_max`.
    ///
    /// # Panics
    ///
    /// Panics if the JV quorum has not been collected, or if the quorum no
    /// longer yields a complete valid result.
    ///
    /// Upstream: `src/vsr/replica.zig:9705`
    /// (`primary_set_log_from_join_view_messages`).
    fn primary_set_log_from_join_view_messages(&mut self) {
        assert_eq!(self.status, Status::ViewChange);
        assert!(self.view > self.log_view);
        assert!(self.is_primary());
        assert!(self.replica_count > 1);
        assert!(self.join_view_quorum);
        assert!(self.join_view_from_all_replicas[usize::from(self.replica_index)].is_some());
        crate::jv_quorum::verify(&self.join_view_from_all_replicas);

        // The `prepare_timestamp` prevents the primary's clock from running
        // backwards; advance before discarding uncommitted timestamps.
        let timestamp_max = crate::jv_quorum::timestamp_max(&self.join_view_from_all_replicas);
        self.prepare_timestamp = self.prepare_timestamp.max(timestamp_max);

        let quorum = self.quorum();
        let result = crate::jv_quorum::quorum_headers(
            &self.join_view_from_all_replicas,
            crate::jv_quorum::QuorumOptions {
                replica_count: self.replica_count,
                quorum_view_change: quorum.view_change,
                quorum_nack_prepare: quorum.nack_prepare,
            },
        );
        let crate::jv_quorum::QuorumHeadersResult::CompleteValid { op_head, headers, .. } = result
        else {
            unreachable!("quorum_headers is CompleteValid once join_view_quorum is set");
        };

        // We must never rewind `commit_max`: `commit_min` represents what we
        // have already applied to the state machine (upstream
        // `set_op_and_commit_max`).
        let commit_max_quorum = crate::jv_quorum::commit_max(&self.join_view_from_all_replicas);

        // "`replica.op` exists" invariant may be broken briefly between
        // `set_op_and_commit_max()` and `replace_header(header_head)`
        // (upstream `replica.zig:9750`).
        self.set_op_and_commit_max(op_head, commit_max_quorum);
        assert!(self.commit_max <= self.op);
        self.replace_header(&headers[0]);
        assert!(self.journal.header_with_op(self.op).is_some());

        // Install the remaining canonical headers high → low.
        for header in &headers[1..] {
            assert!(header.op < self.op);
            self.replace_header(header);
        }
        assert!(self.journal.header_with_op(self.commit_max).is_some());

        // Uncanonical (older log_view) JVs may hold committed headers no
        // canonical JV carries; install them. Uncommitted headers from these
        // JVs would be verified and written by `repair_header` upstream —
        // deferred to the I/O increment. Collect first to satisfy the borrow
        // checker (the JV quorum is borrowed while `replace_header` mutates).
        let mut uncanonical_committed: Vec<message_header::Prepare> = Vec::new();
        for jv in crate::jv_quorum::jvs_uncanonical(&self.join_view_from_all_replicas) {
            for header in &jv.headers {
                if crate::jv_quorum::jv_header_type(header) != crate::jv_quorum::JvHeaderType::Valid
                {
                    continue;
                }
                if header.op <= jv.header.commit_min {
                    uncanonical_committed.push(*header);
                } else {
                    // TODO(port): `repair_header` for the uncommitted uncanonical
                    // headers; sans-IO journals have nothing to repair.
                }
            }
        }
        for header in &uncanonical_committed {
            self.replace_header(header);
        }
    }

    /// Install a reconstructible header into the journal, marking it dirty so the
    /// WAL write is re-issued — unless we already have it exactly: that would
    /// trigger a repair and delay the view change, or worse, prevent repairs to
    /// another replica when we have the op.
    ///
    /// # Panics
    ///
    /// Panics if the header is invalid, or would advance the op.
    ///
    /// Upstream: `src/vsr/replica.zig:8490` (`replace_header`).
    fn replace_header(&mut self, header: &message_header::Prepare) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(header.valid_checksum());
        assert!(header.invalid().is_none());
        assert_eq!(header.command, Command::Prepare);
        assert!(header.view <= self.view);
        assert!(header.op <= self.op);
        // Never advance the op.
        assert!(header.op <= self.op_prepare_max_sync());
        // TODO(port): the `header.op == op_checkpoint() + 1` parent assertion
        // needs the superblock checkpoint header (upstream :8514).
        if header.op < self.op_repair_min() {
            return; // Restart-recovery would never use this header.
        }
        if !self.journal.has_header(header) {
            self.journal.set_header_as_dirty(header);
        }
    }

    /// Handle a View message from the primary of `header.view`.
    ///
    /// The View defines the new view's log suffix (and, upstream, its
    /// checkpoint). A backup replaces its journal suffix, returns to `Normal`
    /// status, floods prepare_oks for any ops it can still contribute, and
    /// commits everything it now knows to be committed.
    ///
    /// # Panics
    ///
    /// Panics on invariant violations (e.g. a View from a non-primary, or one
    /// that advertises an op above the pipeline window).
    ///
    /// Upstream: `src/vsr/replica.zig:2759` (`on_view`).
    pub fn on_view(&mut self, message: &crate::message::Message, now: u64) {
        let Some(view) = message.header::<message_header::View>() else {
            return; // Command mismatch or malformed header.
        };
        // DEVIATION: upstream validates checksums at the message_bus receive
        // path; sans-IO messages are constructed locally, so guard anyway.
        if !view.valid_checksum() || !view.valid_checksum_body(message.body_used()) {
            return;
        }
        if self.ignore_view_change_message(message) {
            return;
        }

        // The recovering_head fast-forward path is deferred; sans-IO replicas
        // only reach this handler from `view_change` or `normal`.
        assert!(self.status == Status::ViewChange || self.status == Status::Normal);

        if view.view == self.log_view && view.op < self.op {
            // We were already in this view prior to receiving the View.
            assert_eq!(self.status, Status::Normal);
            return;
        }

        if self.view < view.view {
            self.transition_to_view_change_status(view.view);
        }
        if self.status == Status::Normal {
            assert!(!self.is_primary());
            assert_eq!(self.view, self.log_view);
        }
        assert_eq!(self.view, view.view);

        // The View message may be from a primary that hasn't yet committed up
        // to its commit_max.
        assert_eq!(
            u16::from(view.replica),
            Self::primary_index_for_view(view.view, self.replica_count)
        );
        assert!(view.commit_max >= view.checkpoint_op);
        assert!(view.op >= view.commit_max);
        assert!(view.op - view.commit_max <= u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX));

        // Sans-IO: `on_view_set_checkpoint` never triggers (op_checkpoint = 0,
        // checkpoints are deferred with the superblock), so the journal update
        // path always runs — without cancellation for state sync.
        self.on_view_set_journal(&view, message.body_used(), now);
    }

    /// Install the View's log suffix and return to `Normal` status as a backup.
    ///
    /// Upstream: `src/vsr/replica.zig:2913`
    /// (`on_view_set_journal`), which also handles the `.recovering_head`
    /// and (deferred) `.normal`-joining paths.
    fn on_view_set_journal(&mut self, view: &message_header::View, body: &[u8], now: u64) {
        assert!(self.status == Status::ViewChange || self.status == Status::Normal);
        assert_eq!(
            u16::from(view.replica),
            Self::primary_index_for_view(view.view, self.replica_count)
        );
        assert!(view.commit_max >= view.checkpoint_op);
        assert!(view.op >= view.commit_max);

        let Some(view_headers) = Self::decode_view_headers(body) else {
            return; // Body does not fit a checkpoint prefix plus whole headers.
        };
        assert_eq!(view_headers[0].op, view.op);
        // The suffix runs descending; the checkpoint hooks may jump down.
        assert!(view_headers[0].op >= view_headers[view_headers.len() - 1].op);

        {
            // Replace our log with the suffix from View. There is no sync/kick
            // (checkpoint_op = 0), so `op_prepare_max_sync` is huge and the
            // first (head) header always fits — but keep the upstream search
            // shape: the new head is the first header not ruled out by op.
            let mut head: Option<u64> = None;
            for header in &view_headers {
                assert!(header.commit <= view.commit_max);
                if header.op <= self.op_prepare_max_sync()
                    && (self.log_view < self.view
                        || (self.log_view == self.view && header.op >= self.op))
                {
                    head = Some(header.op);
                    break;
                }
            }
            if let Some(op) = head {
                self.set_op_and_commit_max(op, view.commit_max);
                assert_eq!(self.op, op);
                assert!(self.commit_max >= view.commit_max);
            } else {
                assert_eq!(self.log_view, self.view);
                assert!(self.op > self.op_prepare_max_sync());
                self.advance_commit_max(view.commit_max);
            }
            assert!(self.commit_min <= self.commit_max);

            for header in &view_headers {
                if header.op <= self.op_prepare_max_sync() {
                    self.replace_header(header);
                }
            }
        }

        // Remember the new log suffix for repairs within the view
        // (upstream `view_headers.replace(.view, view_headers)`).
        self.view_headers.clear();
        self.view_headers.extend_from_slice(&view_headers);

        match self.status {
            Status::ViewChange => {
                self.transition_to_normal_from_view_change_status(view.view, now);
                self.send_prepare_oks_after_view_change();
                self.commit_journal();
            }
            Status::Normal => {
                // DEVIATION: upstream re-broadcasts prepare_oks / commits only
                // in the view_change path (and the deferred recovering_head
                // path); a normal-status replica joining mid-view has already
                // done both.
            }
            _ => unreachable!(),
        }

        assert_eq!(self.status, Status::Normal);
        assert_eq!(view.view, self.log_view);
        assert_eq!(view.view, self.view);
        assert!(!self.is_primary());

        // DEVIATION: upstream optionally starves the verification pipeline when
        // state syncing; sans-IO journals are clean by construction. When idle
        // it re-runs `repair()`, which we do too: a jumped head leaves a gap
        // below it that the repair pass now fills via GetHeaders/GetPrepare.
        self.repair();
    }

    /// Decode the body of a View message into prepare headers.
    ///
    /// The body starts with a `CHECKPOINT_STATE_SIZE` prefix (the checkpoint a
    /// View advertises) followed by whole prepare-header frames. Returns `None`
    /// if the suffix is empty or not a whole number of headers.
    fn decode_view_headers(body: &[u8]) -> Option<Vec<message_header::Prepare>> {
        let headers = body.get(constants::CHECKPOINT_STATE_SIZE..)?;
        Self::decode_headers_bytes(headers)
    }

    /// Decode a message body as a sequence of whole prepare-header frames.
    ///
    /// Returns `None` if the body is empty or not a whole number of headers.
    fn decode_headers_bytes(body: &[u8]) -> Option<Vec<message_header::Prepare>> {
        if body.is_empty() || !body.len().is_multiple_of(message_header::SIZE) {
            return None;
        }
        let mut out = Vec::with_capacity(body.len() / message_header::SIZE);
        for frame in body.as_chunks::<{ message_header::SIZE }>().0 {
            out.push(message_header::Prepare::from_wire(frame)?);
        }
        Some(out)
    }

    /// Decode the body of a `Headers` message into prepare headers.
    fn decode_headers_body(
        message: &crate::message::Message,
    ) -> Option<Vec<message_header::Prepare>> {
        Self::decode_headers_bytes(message.body_used())
    }

    /// Handle a client request (primary only, normal status).
    ///
    /// A validated request is prepared immediately when the pipeline is not
    /// full, else it is queued on the request queue and prepared once earlier
    /// commits execute (upstream `commit_execute`).
    ///
    /// # Panics
    ///
    /// Panics if the primary has an uncommitted prepare (upstream asserts
    /// `commit_min == commit_max`, replica.zig:1950).
    ///
    /// Upstream: `src/vsr/replica.zig:1944` (`on_request`).
    pub fn on_request(&mut self, header: &message_header::Header) {
        if !self.is_primary() || self.status != Status::Normal {
            return;
        }
        let Some(request) = header.into_typed::<message_header::Request>() else {
            return; // Command mismatch or invalid header.
        };
        if !request.valid_checksum() || request.invalid_header().is_some() {
            return;
        }
        if self.ignore_request_message(&request) {
            return;
        }

        // The primary must be fully caught up to accept a request
        // (upstream: replica.zig:1950).
        assert_eq!(self.commit_min, self.commit_max);

        if self.pipeline_queue.prepare_queue_full() {
            self.pipeline_queue.request_queue.push(PipelineRequest {
                client: request.client,
                request: request.request,
                request_checksum: request.checksum,
                operation: request.operation,
            });
        } else {
            let body_size = request.size - message_header::SIZE_U32;
            let result = self.primary_pipeline_prepare(
                request.client,
                request.request,
                request.operation,
                body_size,
                request.checksum,
            );
            assert!(result.is_ok(), "valid primary-visible request must prepare");
        }
    }

    /// Handle a Prepare message from the primary — decode and forward to
    /// [`Self::on_prepare`], then ack the accepted prepare on the backup.
    ///
    /// Upstream: `src/vsr/replica.zig:2021` (`on_prepare`); the ack is sent
    /// from `write_prepare_callback` → `send_prepare_ok` (replica.zig:11225).
    pub fn on_prepare_message(&mut self, header: &message_header::Header) {
        let Some(prepare) = header.into_typed::<message_header::Prepare>() else {
            return; // Command mismatch or invalid header.
        };
        // Upstream ignores prepares sent outside our normal current-view state
        // (older/newer view, view-change status); raw `on_prepare` asserts a
        // matching view, so filter first (upstream: replica.zig:2066-2086).
        if self.status != Status::Normal || prepare.view != self.view {
            return;
        }
        match self.on_prepare(&prepare) {
            OnPrepareResult::Accepted => {
                // A backup acks the accepted prepare; the primary contributes
                // its own prepare_ok on the pipeline path instead (we do not
                // self-ack here).
                if !self.is_primary() {
                    self.send_prepare_ok(&prepare);
                    // Opportunistically repair any gaps this prepare exposed
                    // (upstream defer: replica.zig:2111-2116).
                    self.repair();
                }
            }
            OnPrepareResult::Stale => {
                // We already hold the op: it may be a re-broadcast of a prepare
                // whose ack was lost — refresh the journal and republish the
                // ack if we have it clean (upstream `on_repair`).
                self.on_repair(&prepare);
            }
            OnPrepareResult::FutureOp => {
                // The primary has moved more than one op past our head: advance
                // `op` so the skipped prepares slot in as they are repaired
                // (upstream repairs the gap concurrently via `repair()`).
                self.jump_to_newer_op_in_normal_status(prepare.op);
            }
        }
    }

    /// Advance `op` to `op - 1`, skipping a contiguous range of missing ops.
    ///
    /// When a backup receives a prepare more than one op ahead of its head,
    /// the missing ops in between are unknown, but the head can still be moved
    /// forward so that the intervening prepares repair in one by one (they are
    /// repaired on demand — upstream via `repair()`).
    ///
    /// # Panics
    ///
    /// Panics if the replica is not `.normal`, is the primary, is asked to jump
    /// by one op or fewer, would restore an already-committed op, or would
    /// overwrite an op the WAL cannot hold (all replicated from upstream's
    /// asserts; the caller routes only current-view headers here).
    ///
    /// Upstream: `src/vsr/replica.zig:6933`
    /// (`jump_to_newer_op_in_normal_status`).
    pub fn jump_to_newer_op_in_normal_status(&mut self, op: u64) {
        assert_eq!(self.status, Status::Normal);
        assert!(!self.is_primary());
        assert!(op > self.op + 1);
        // We may have learned of a higher `commit_max` through a commit message
        // before jumping to a newer op: still reject coming at/below `commit_min`.
        assert!(op > self.commit_min);
        // Never overwrite an op that still needs to be checkpointed.
        // DEVIATION: sans-IO the checkpoint is always durable at 0.
        assert!(op <= self.op_prepare_max_sync());

        self.op = op - 1;
        assert!(self.op >= self.commit_min);
        assert_eq!(self.op + 1, op);
    }

    /// Whether this replica may currently repair (request or serve repairs).
    ///
    /// # Panics
    ///
    /// Panics if in `.view_change` with a quorum but not the would-be primary
    /// (an invariant of the transition, upstream replica.zig:8444).
    ///
    /// Upstream: `src/vsr/replica.zig:8440` (`repairs_allowed`).
    #[must_use]
    pub fn repairs_allowed(&self) -> bool {
        match self.status {
            Status::ViewChange => {
                // Becoming primary requires a repair of the journal first; a
                // backup mid-transition must stay quiet (upstream 8442-8447).
                if self.join_view_quorum {
                    assert_eq!(
                        Replica::primary_index_for_view(self.view, self.replica_count),
                        u16::from(self.replica_u8())
                    );
                    true
                } else {
                    false
                }
            }
            Status::Normal => true,
            _ => false,
        }
    }

    /// The largest op that may be requested during repair.
    ///
    /// # Panics
    ///
    /// Panics if the head has moved beyond what the WAL can hold (an invariant
    /// that the checkpoint/commit machinery enforces).
    ///
    /// Upstream: `src/vsr/replica.zig:7243` (`op_repair_max`).
    #[must_use]
    pub fn op_repair_max(&self) -> u64 {
        assert!(self.op >= self.op_checkpoint());
        assert!(self.op <= self.op_prepare_max_sync());
        assert!(self.op <= self.commit_max + u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX));
        self.commit_max.min(self.op_prepare_max_sync().max(self.op))
    }

    /// Choose a peer to send repair requests to.
    ///
    /// DEVIATION: upstream rotates through peers with a PRNG to spread load;
    /// sans-IO we deterministically pick the next replica index.
    ///
    /// # Panics
    ///
    /// Panics if this is a solo replica (nothing to repair from) or the cluster
    /// is larger than `u8::MAX` replicas.
    ///
    /// Upstream: `src/vsr/replica.zig:4268` (`choose_any_other_replica`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // replica_count <= u8::MAX asserted above
    pub fn choose_any_other_replica(&self) -> u8 {
        assert!(self.replica_count > 1);
        assert!(self.replica_count <= u16::from(u8::MAX) + 1);
        let other = (u16::from(self.replica_u8()) + 1) % self.replica_count;
        assert_ne!(other, u16::from(self.replica_u8()));
        other as u8
    }

    /// Request the prepare for `op` (whose header we hold, dirty) from a peer.
    ///
    /// Upstream: `src/vsr/replica.zig:8311` (`repair_prepare`).
    ///
    /// DEVIATION: no write budget, pipeline cache, or `journal.writing`
    /// consultation; upstream further has a mode that requests an
    /// *unknown*-checksum prepare with `view` + checksum 0, which the view-0
    /// explicit-checksum `GetPrepare` form of this port cannot express —
    /// unknown-checksum ops are covered by the `GetHeaders` break path instead
    /// (they repair in as a header first, then get their body here).
    ///
    /// # Panics
    ///
    /// Panics if not `.normal`/`.view_change` or if repairs are not allowed.
    ///
    /// Upstream: `src/vsr/replica.zig:8311` (`repair_prepare`).
    fn repair_prepare(&mut self, op: u64) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.repairs_allowed());
        let Some(header) = self.journal.header_with_op(op).copied() else {
            return;
        };
        let to = self.choose_any_other_replica();
        self.send_get_prepare(to, op, header.checksum());
    }

    /// Repair every op in `op_min..=op_max` that lacks a clean prepare.
    ///
    /// # Panics
    ///
    /// Panics if repairs are not allowed, if the range strays outside
    /// `[op_repair_min, op]`, or if `op_min > op_max`.
    ///
    /// Upstream: `src/vsr/replica.zig:8220` (`repair_prepares_between`).
    fn repair_prepares_between(&mut self, op_min: u64, op_max: u64) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.repairs_allowed());
        assert!(op_min >= self.op_repair_min());
        assert!(op_min <= op_max);
        assert!(op_max <= self.op);

        // DEVIATION: no IOP/write budget to schedule against — request every op
        // whose slot is absent or dirty (the repair-message budget is deferred).
        for op in op_min..=op_max {
            let missing_or_dirty =
                self.journal.slot_with_op(op).is_none_or(|slot| self.journal.dirty.bit(slot));
            if missing_or_dirty {
                self.repair_prepare(op);
            }
        }
    }

    /// Drop dirty WAL entries that fall outside the `[op_repair_min, op]`
    /// repair window, so that `repair()` can eventually finish.
    ///
    /// An out-of-bounds dirty slot is either:
    /// - an op `<= op_checkpoint` that was committed before checkpointing, but
    ///   whose WAL entry was found corrupt after recovering from a crash, or
    /// - (indistinguishably) an op `> self.op` that was truncated, and is now
    ///   corrupt.
    ///
    /// In-bounds slots are left alone: the `repair_prepares_between` invocations
    /// that run before this function either already repaired them, or are
    /// waiting for a `GetPrepare` reply.
    ///
    /// # Panics
    ///
    /// Panics if repairs are not allowed, or if the journal head is more than a
    /// full slot count ahead of the commit frontier.
    ///
    /// Upstream: `src/vsr/replica.zig:8257` (`repair_clean_out_of_bound_prepares`).
    fn repair_clean_out_of_bound_prepares(&mut self) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.repairs_allowed());
        assert!(self.op >= self.commit_min);
        assert!(self.op - self.commit_min <= u64::from(constants::JOURNAL_SLOT_COUNT));

        // The repair window's slots, inclusive:
        let slots_repaired = crate::journal::SlotRange {
            head: crate::journal::Slot::for_op(self.op_repair_min()),
            tail: self
                .journal
                .slot_with_op(self.op)
                .unwrap_or_else(|| panic!("slot_with_op({}) failed", self.op)),
        };

        for slot_index in 0..u64::from(constants::JOURNAL_SLOT_COUNT) {
            let slot = crate::journal::Slot::for_op(slot_index);
            if slots_repaired.head == slots_repaired.tail || slots_repaired.contains(slot) {
                // In-bounds: handled by the repair_prepares_between invocations
                // before this function is invoked.
            } else if self.journal.dirty.bit(slot) {
                // Out-of-bounds dirty slots cannot be repaired (there is no
                // valid sibling to hash-chain to); drop the entry.
                self.journal.remove_entry(slot);
            }
        }
    }

    /// Periodic repair driver: re-arm our repair cadence and run a pass.
    ///
    /// Upstream also reaps expired `journal_repair_message_budget` requests;
    /// the sans-IO port has no message budgets.
    ///
    /// # Panics
    ///
    /// Panics if the replica is not `.normal`/`.view_change`.
    ///
    /// Upstream: `src/vsr/replica.zig:3771` (`on_journal_repair_timeout`).
    pub fn on_journal_repair_timeout(&mut self) {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        // Upstream uses `reset_with_jitter`; sans-IO is deterministic, so we
        // re-arm with a fixed cadence (DEVIATION).
        self.journal_repair_timeout.reset(constants::JOURNAL_REPAIR_TIMEOUT);
        self.repair();
    }

    /// Detect missing/divergent headers and prepares in the journal and repair
    /// them from peers.
    ///
    /// Runs only while [`Self::journal_repair_timeout`] is active: in normal
    /// operation it fires every `JOURNAL_REPAIR_TIMEOUT` ticks, and
    /// opportunistically after a prepare is accepted
    /// ([`Self::on_prepare_message`]) and after every `Headers` response
    /// ([`Self::on_headers`]).
    ///
    /// DEVIATION (deferred): only the grid `GetBlocks` repair step is not
    /// ported yet. The view-change primary's `primary_send_view` step is folded
    /// into [`Self::on_join_view`], which transitions to Normal synchronously
    /// once the quorum log is installed.
    ///
    /// # Panics
    ///
    /// Panics if not `.normal`/`.view_change`, if the journal head advances
    /// further than a pipeline beyond the commit frontier, or if the journal
    /// head op is absent.
    ///
    /// Upstream: `src/vsr/replica.zig:7544` (`repair`).
    pub fn repair(&mut self) {
        if !self.journal_repair_timeout.active {
            // Upstream: `if (!self.journal_repair_timeout.ticking) return;`
            // (replica.zig:7545). Guards against repair traffic before the
            // replica has completed its status transition.
            return;
        }

        // Grid repair step (upstream replica.zig:7552): request remote blocks
        // for parked coherent reads. Unlike the journal steps below this runs
        // before the status asserts, matching upstream, where it precedes
        // `state_machine_opened`.
        if self.grid.is_some()
            && let Some(destination) =
                self.grid_repair_message_budget.next_destination(&mut self.prng)
        {
            self.send_get_blocks(destination);
        }

        assert!(self.status == Status::Normal || self.status == Status::ViewChange);
        assert!(self.repairs_allowed());

        assert!(self.op_checkpoint() <= self.op);
        assert!(self.op_checkpoint() <= self.commit_min);
        assert!(self.commit_min <= self.op);
        assert!(self.commit_min <= self.commit_max);
        assert!(
            self.commit_max
                >= self.op.saturating_sub(u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX))
        );
        assert!(self.journal.header_with_op(self.op).is_some());

        // Request outstanding possibly committed headers to advance our op
        // number. This handles the case of an idle cluster, where a backup
        // will not otherwise advance (upstream 7598-7616).
        if self.op < self.op_repair_max()
            || (self.status == Status::Normal
                && self.view_headers.first().is_some_and(|header| self.op < header.op))
        {
            assert!(self.replica_count > 1);
            assert!(!self.is_primary());
            self.send_get_view();
        }

        // The op is from the current view; anything that hash-chains to it is
        // worth repairing (upstream 7608-7619). Wait for view-change primaries.

        let repair_op_max: u64 = if self.journal_repair_timeout.attempts != 0
            && self.journal_repair_timeout.attempts.is_multiple_of(50)
        {
            // Every 50 timeouts, unconditionally repair — allows backups to
            // repair journal faults in an idle cluster where the head op
            // does not progress (replica.zig:7625).
            self.op
        } else if self.status == Status::ViewChange {
            // View-changing replicas repair unconditionally.
            self.op
        } else {
            // Missing prepares within a pipeline of ops from the head may
            // arrive via normal replication; wait for them (upstream 7635).
            self.op.saturating_sub(u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX))
        };

        // Request any missing or disconnected headers:
        let header_break =
            self.journal.find_latest_headers_break_between(self.op_repair_min(), self.op);
        if let Some(range) = header_break {
            assert!(self.replica_count > 1);
            assert!(range.op_min >= self.op_repair_min());
            assert!(range.op_max < self.op);

            if range.op_min <= repair_op_max {
                let op_max = range.op_max.min(repair_op_max);
                assert!(range.op_min <= op_max);
                // Pessimistically request extra headers: sends are inexpensive
                // and it may save extra round-trips (upstream 7672-7676).
                let to = self.choose_any_other_replica();
                self.send_get_headers(to, range.op_min, op_max);
            }
        }

        // First priority is committing further; only then repairing committed
        // prepares at risk of eviction (upstream 7682-7686):
        if self.commit_min < repair_op_max {
            self.repair_prepares_between(self.commit_min + 1, repair_op_max);
        }
        if self.op_repair_min() <= self.commit_min {
            self.repair_prepares_between(self.op_repair_min(), self.commit_min);
        }

        self.repair_clean_out_of_bound_prepares();

        if self.commit_min < self.commit_max {
            // Commit what we can even eagerly — discovering missing prepares
            // drives further repairs (upstream 7697-7705).
            assert!(self.replica_count > 1);
            self.commit_journal();
        }
    }

    /// Handle a repair Prepare: one we already hold, or one that may replace a
    /// faulty/dirty entry. Repairs may never advance `self.op`.
    ///
    /// Upstream `on_repair` is reached from `on_prepare` for prepares that are
    /// older (`view < self.view`) or that we already hold (`op <= self.op`,
    /// current view). The sans-IO port reaches it only from
    /// [`Self::on_prepare_message`] via [`OnPrepareResult::Stale`] (view
    /// matches, `op <= self.op`).
    ///
    /// If the prepare is already journaled and clean, we resend our prepare_ok:
    /// the primary's copy of that ack may have been lost, and republishing it
    /// lets the quorum make progress. Otherwise the prepare is (re-)installed
    /// as dirty and, on a backup, a commit pass is attempted.
    ///
    /// DEVIATION: upstream also consults the pipeline-cache and the
    /// repair-message budget, and writes the repaired prepare through
    /// `write_prepare`; sans-IO installs the header via `repair_header` only.
    ///
    /// # Panics
    ///
    /// Panics if `self.status` is not `Status::Normal` (the repair path only
    /// runs in a normal-op context, matching the lone caller in the port).
    ///
    /// Upstream: `src/vsr/replica.zig:2455` (`on_repair`).
    pub fn on_repair(&mut self, header: &message_header::Prepare) {
        assert_eq!(self.status, Status::Normal);
        if header.view > self.view {
            return; // From a view we have not joined.
        }
        if header.op > self.op {
            return; // Repairs may never advance `self.op`.
        }

        if self.journal.has_prepare(header) {
            // Duplicate and clean: republish the lost prepare_ok (upstream
            // 2510-2515).
            self.send_prepare_ok(header);
            return;
        }

        if self.repair_header(header) {
            // Optimistically try to commit now that the prepare is (re-)staged
            // (upstream 2539-2541); primaries wait for the pipeline instead.
            if !self.is_primary() {
                self.commit_journal();
            }
        }
    }

    /// Request a peer to re-serve the prepare for `op`.
    ///
    /// Uses the explicit-checksum form of `GetPrepare` (`view == 0`), which
    /// the responder can satisfy without referencing its own view of the log:
    /// `checksum` is typically known to the requester from the hash chain of a
    /// later prepare (its `parent`) or from a `Commit` (`commit_checksum`).
    ///
    /// DEVIATION: upstream (`repair_prepare`, replica.zig:8387) omits the
    /// checksum and sets `view = self.view` when it has no journal entry for
    /// the op, letting the responder resolve the op in its own log. That form
    /// is unsatisfiable for `view == 0` (the responder treats `view == 0` as
    /// requiring the explicit checksum), so the port always sends the explicit
    /// checksum. Upstream also picks `to` via the repair-message budget
    /// (`journal_repair_message_budget.decrement`); sans-IO has no budgets, the
    /// caller supplies the destination. The flat `send_queue` carries no
    /// routing metadata, so the single queued `GetPrepare` must be delivered to
    /// `to`.
    ///
    /// # Panics
    ///
    /// Panics if the replica is single, or if `to` is not a valid replica.
    ///
    /// Upstream: `src/vsr/replica.zig:8311` (`repair_prepare`).
    pub fn send_get_prepare(&mut self, to: u8, op: u64, checksum: u128) {
        assert!(u16::from(to) < self.replica_count);
        assert_ne!(to, self.replica_u8());
        let mut get_prepare = message_header::GetPrepare {
            cluster: self.cluster,
            view: 0,
            replica: self.replica_u8(),
            prepare_op: op,
            prepare_checksum: checksum,
            ..message_header::GetPrepare::default()
        };
        get_prepare.set_checksum_body(&[]);
        get_prepare.set_checksum();
        self.enqueue_header(&get_prepare);
    }

    /// Serve a prepare from our journal in response to a `GetPrepare` request.
    ///
    /// The requester references the prepare either by an explicit checksum
    /// (`view == 0`) or by op alone (`view != 0`, `prepare_checksum == 0`), in
    /// which case the checksum is resolved from our own entry for the op. If we
    /// do not hold the requested checksum — because we are missing the op or
    /// hold a different one — we do not answer, and the requester will retry
    /// against another peer.
    ///
    /// DEVIATION: upstream (replica.zig:3084-3157) serves the pipeline first,
    /// then reads the WAL prepare with `read_prepare_with_op_and_checksum` so
    /// the response carries the prepare body. The sans-IO journal is in-memory
    /// with deferred bodies, so the stored header is served directly; the
    /// recipient validates it via the hash chain as usual. The flat
    /// `send_queue` carries no routing metadata, so the single queued `Prepare`
    /// must be delivered to `get_prepare.replica`.
    ///
    /// # Panics
    ///
    /// Panics if `replica_count <= 1` or if the request claims to originate
    /// from this replica.
    ///
    /// Upstream: `src/vsr/replica.zig:3056` (`on_get_prepare`).
    pub fn on_get_prepare(&mut self, header: &message_header::Header) {
        let Some(get_prepare) = header.into_typed::<message_header::GetPrepare>() else {
            return; // Command mismatch or invalid header.
        };
        assert!(self.replica_count > 1);
        assert_ne!(get_prepare.replica, self.replica_u8());

        // Resolve the checksum the requester expects (upstream 3064-3079):
        let checksum = if get_prepare.view == 0 {
            get_prepare.prepare_checksum
        } else {
            assert_eq!(get_prepare.prepare_checksum, 0);
            let Some(entry) = self.journal.header_with_op(get_prepare.prepare_op) else {
                return; // We don't hold the op the requester assumes we hold.
            };
            entry.checksum()
        };

        // Only answer if we hold the exact prepare requested:
        let Some(prepare) =
            self.journal.header_with_op_and_checksum(get_prepare.prepare_op, checksum)
        else {
            return;
        };
        self.enqueue_header(&prepare.clone());
    }

    /// Request a peer to re-serve the headers for `[op_min, op_max]` (both
    /// inclusive), up to `GET_HEADERS_MAX` of them, as a `Headers` message.
    ///
    /// DEVIATION: upstream (`repair`, replica.zig:7666-7678) chooses the peer
    /// and requests pessimistically beyond the known break; sans-IO has no
    /// `repair()` orchestration yet, so the caller supplies the range and the
    /// destination. As with the other repair requests, the flat `send_queue`
    /// carries no routing metadata: the single queued `GetHeaders` must be
    /// delivered to `to`.
    ///
    /// # Panics
    ///
    /// Panics if `to` is not a valid replica index, if `to` is this replica,
    /// or if `op_min > op_max`.
    ///
    /// Upstream: `src/vsr/replica.zig:7666-7678` (the `GetHeaders` sender).
    pub fn send_get_headers(&mut self, to: u8, op_min: u64, op_max: u64) {
        assert!(u16::from(to) < self.replica_count);
        assert_ne!(to, self.replica_u8());
        assert!(op_min <= op_max);
        // `GetHeaders` requires `view == 0` (message_header.rs):
        let mut get_headers = message_header::GetHeaders {
            cluster: self.cluster,
            view: 0,
            replica: self.replica_u8(),
            op_min,
            op_max,
            ..message_header::GetHeaders::default()
        };
        get_headers.set_checksum_body(&[]);
        get_headers.set_checksum();
        self.enqueue_header(&get_headers);
    }

    /// Serve the prepare headers in `[op_min, op_max]` (both inclusive) that we
    /// hold in the journal, as a `Headers` message to the requester.
    ///
    /// Serves at most `GET_HEADERS_MAX` headers. If the range yields nothing we
    /// stay silent so the requester does not mistake an empty body for a valid
    /// response.
    ///
    /// DEVIATION: upstream (replica.zig:3184-3191) copies into a message
    /// buffer; sans-IO builds a `Vec`/`Message`. The flat `send_queue` carries
    /// no routing metadata, so the single queued `Headers` must be delivered to
    /// `get_headers.replica`.
    ///
    /// # Panics
    ///
    /// Panics if `replica_count <= 1` or if the request claims to originate
    /// from this replica.
    ///
    /// Upstream: `src/vsr/replica.zig:3159` (`on_get_headers`).
    #[allow(clippy::cast_possible_truncation)] // body length is bounded by GET_HEADERS_MAX frames
    pub fn on_get_headers(&mut self, header: &message_header::Header) {
        let Some(get_headers) = header.into_typed::<message_header::GetHeaders>() else {
            return; // Command mismatch or invalid header.
        };
        assert!(self.replica_count > 1);
        assert_ne!(get_headers.replica, self.replica_u8());

        // `op_min`/`op_max` are both inclusive; the journal never copies more
        // than `GET_HEADERS_MAX` headers into `dest` (upstream 3181-3191).
        let mut dest = vec![message_header::Prepare::default(); constants::GET_HEADERS_MAX];
        let copied = self.journal.copy_latest_headers_between(
            get_headers.op_min,
            get_headers.op_max,
            &mut dest,
        );
        if copied == 0 {
            return; // Nothing in range; stay silent (upstream 3194-3201).
        }

        let mut headers = message_header::Headers {
            cluster: self.cluster,
            view: self.view,
            replica: self.replica_u8(),
            ..message_header::Headers::default()
        };
        let frames: Vec<u8> =
            dest[..copied].iter().flat_map(message_header::Prepare::to_wire).collect();

        let size = message_header::SIZE + frames.len();
        headers.size = size as u32;
        headers.set_checksum_body(&frames);
        headers.set_checksum();

        let mut response = crate::message::Message::new();
        response.set_header(&headers);
        response.set_body(&frames);
        self.send_queue.push(response);
    }

    /// Request the primary of the current view to serve us a fresh `View`, so we
    /// can advance our head op when we are too far behind to GetPrepare.
    ///
    /// DEVIATION: upstream (`repair`, replica.zig:7598-7616) sends `GetView`
    /// through the message bus to `primary_index(view)`. The flat `send_queue`
    /// carries no routing metadata: the single queued `GetView` must be
    /// delivered to the primary.
    ///
    /// # Panics
    ///
    /// Panics if this is a solo replica or this replica is the primary.
    ///
    /// Upstream: the `GetView` sender behind `src/vsr/replica.zig:7598`.
    pub fn send_get_view(&mut self) {
        assert!(self.replica_count > 1);
        assert!(!self.is_primary());

        let mut get_view = message_header::GetView {
            cluster: self.cluster,
            view: self.view,
            replica: self.replica_u8(),
            nonce: self.nonce,
            ..message_header::GetView::default()
        };
        get_view.set_checksum_body(&[]);
        get_view.set_checksum();
        self.enqueue_header(&get_view);
    }

    /// Serve a `GetView` request: reply to the requester with a fresh `View`
    /// carrying our op/commit_max and echoing `get_view.nonce`.
    ///
    /// DEVIATION: the flat `send_queue` carries no routing metadata, so the
    /// single queued `View` must be delivered to `get_view.replica`.
    ///
    /// # Panics
    ///
    /// Panics unless this replica is a normal-status primary whose
    /// `view == log_view`, and the request matches our view and did not
    /// originate from this replica.
    ///
    /// Upstream: `src/vsr/replica.zig:3022` (`on_get_view`).
    pub fn on_get_view(&mut self, message: &crate::message::Message) {
        let Some(get_view) = message.header::<message_header::GetView>() else {
            return; // Command mismatch or malformed header.
        };
        if !get_view.valid_checksum() || !get_view.valid_checksum_body(message.body_used()) {
            return;
        }
        assert_eq!(self.status, Status::Normal);
        assert_eq!(self.view, self.log_view);
        assert_eq!(get_view.view, self.view);
        assert_ne!(get_view.replica, self.replica_u8());
        assert!(self.is_primary());

        let view = self.make_view_message(get_view.nonce);
        self.send_queue.push(view);
    }

    /// Handle a `Headers` message: try to repair every prepare header it
    /// carries, then re-run repair of the anything outstanding.
    ///
    /// # Panics
    ///
    /// Panics if a valid `Headers` message carries an undersized body (never
    /// happens: `Headers::invalid` already requires a body; the assert mirrors
    /// upstream replica.zig:3309).
    ///
    /// Upstream: `src/vsr/replica.zig:3300` (`on_headers`).
    #[allow(clippy::cast_possible_truncation)] // SIZE is a const that fits in u32
    pub fn on_headers(&mut self, message: &crate::message::Message) {
        let Some(headers) = message.header::<message_header::Headers>() else {
            return; // Invalid header or wrong command.
        };
        let Some(received) = Self::decode_headers_body(message) else {
            return; // Body is not a whole number of headers.
        };
        assert!(headers.size > message_header::SIZE as u32);

        for header in &received {
            self.repair_header(header);
        }
        // Repair whatever the newly-installed headers left outstanding: dirty
        // prepares above commit_min still need their bodies, and earlier breaks
        // can be discovered by re-scanning (upstream 3321).
        self.repair();
    }

    /// Decide whether to insert or update a prepare header received via
    /// repair (from a `Headers` message or a late-prepare).
    ///
    /// A repair may never advance or replace `self.op`; it only backfills
    /// behind the head. The hash-chain-connection check protects the
    /// prepare_ok promise we made to the primary: confusing a broken entry with
    /// a valid one would let a divergent op leak into a view change.
    ///
    /// DEVIATION: upstream also rejects headers preceding `op_repair_min()`
    /// only when a checkpoint could still reference them; sans-IO has
    /// `op_checkpoint == 0`, so the guard is against slot-wrapping garbage
    /// only. The `checkpoint_id`-for-op assertion is likewise skipped.
    ///
    /// Returns whether the header was (re-)installed as dirty.
    ///
    /// Upstream: `src/vsr/replica.zig:7799` (`repair_header`).
    fn repair_header(&mut self, header: &message_header::Prepare) -> bool {
        assert!(self.status == Status::Normal || self.status == Status::ViewChange);

        if header.view > self.view {
            return false; // From a view we have not joined.
        }
        if header.op > self.op {
            return false; // Would advance the hash chain head.
        }
        if header.op == self.op && !self.journal.has_header(header) {
            return false; // Would replace the hash chain head.
        }
        if header.op < self.op_repair_min() {
            return false; // Precedes the (ring) wrap this op belongs to.
        }

        if self.journal.has_header(header) && self.journal.has_prepare(header) {
            return false; // Already clean.
        }

        // Replacing an entry whose view differs from the head's requires
        // proving the replacement connects the chain up to the head:
        let head_view_departed = match self.journal.header_with_op(self.op) {
            Some(head) => head.view != header.view,
            None => true,
        };
        if head_view_departed && !self.repair_header_would_connect_hash_chain(header) {
            return false; // Would break the chain; divergent or stale.
        }

        self.journal.set_header_as_dirty(header);
        true
    }

    /// Whether replacing our entry at `header.op` (which may be a stale or
    /// diverged op) with `header` would allow the hash chain to connect through
    /// to `self.op`.
    ///
    /// Upstream: `src/vsr/replica.zig:7945` (`repair_header_would_connect_hash_chain`).
    fn repair_header_would_connect_hash_chain(&self, header: &message_header::Prepare) -> bool {
        let mut entry = *header;
        while entry.op < self.op {
            let Some(next) = self.journal.next_entry(&entry) else {
                return false;
            };
            if entry.checksum != next.parent {
                return false;
            }
            entry = *next;
        }
        entry.op == self.op
    }

    /// Handle a Commit message from the primary (backup only, normal status).
    ///
    /// Feeds the fault detector, verifies the committed checksum against the
    /// journal (when present), advances `commit_max`, and enters the commit
    /// pipeline to execute the op from the journal.
    ///
    /// # Panics
    ///
    /// Panics if a present committed entry's checksum disagrees with the Commit
    /// message while our hash chain up to the head is intact (upstream:
    /// `commit checksum verification failed`). A mismatch with a broken chain is
    /// tolerated while the journal is still repairing.
    ///
    /// Upstream: `src/vsr/replica.zig:2396` (`on_commit`).
    pub fn on_commit(&mut self, header: &message_header::Header, now: u64) {
        let Some(commit) = header.into_typed::<message_header::Commit>() else {
            return; // Command mismatch or invalid header.
        };
        if self.is_primary() {
            return; // Primary does not receive Commit messages.
        }
        if self.status != Status::Normal {
            return;
        }
        if commit.view != self.view {
            return; // Stale or future Commit.
        }
        assert_eq!(
            u64::from(commit.replica),
            u64::from(Self::primary_index_for_view(commit.view, self.replica_count))
        );

        self.on_commit_heartbeat(commit.view, commit.timestamp_monotonic, commit.commit, now);

        // We may not always have the latest commit entry, but if we do, our
        // checksum must match (upstream `replica.zig:2436-2449`):
        if let Some(entry) = self.journal.header_with_op(commit.commit) {
            if entry.checksum() == commit.commit_checksum {
                // Verified.
            } else if self.valid_hash_chain_between(commit.commit, self.op) {
                // Our own chain from `commit` to the head is intact, so the
                // primary's checksum is simply wrong.
                panic!("commit checksum verification failed");
            } else {
                // We may still be repairing after receiving the View message;
                // skip verification until the chain reconnects.
            }
        }

        self.commit_journal();
    }

    /// Returns `true` if a client request should be ignored: dropped, or
    /// answered with an eviction.
    ///
    /// Subset of upstream's checks that are meaningful sans-IO — standby
    /// discrimination, backup forwarding (`ignore_request_message_backup`), the
    /// request body / operation / register-body safeguards, and the
    /// upgrade/duplicate/preparing checks are all deferred (the port has no
    /// standby mode, backup forwarding, `request_size_limit`, or mounted state
    /// machine / session table yet). `Request::invalid_header` already rejects
    /// malformed requests, so the evictions below are the ones a valid header
    /// can still trigger.
    ///
    /// Upstream: `src/vsr/replica.zig:6242` (`ignore_request_message`).
    fn ignore_request_message(&mut self, request: &message_header::Request) -> bool {
        assert_eq!(request.command, crate::command::Command::Request);
        assert!(self.is_primary());
        assert_eq!(self.status, Status::Normal);

        // A buggy client may send a view higher than one the cluster has seen.
        // Err on the side of safety and drop such requests (upstream 6256-6265).
        if request.view > self.view {
            return true;
        }

        // Unsupported client release versions are evicted (upstream 6276-6304).
        // DEVIATION: sans-IO the replica runs exactly `Release::MINIMUM`, and
        // `Request::invalid_header` rejects `release == 0`, so — MINIMUM being
        // `(0,0,1)` — the too-low branch is unreachable on a valid header; it
        // is kept for symmetry with upstream. The replica's own `release` field
        // is likewise pinned to `MINIMUM`.
        if request.release.value < crate::multiversion::Release::MINIMUM.value {
            self.send_eviction_message_to_client(
                request.client,
                message_header::Reason::ClientReleaseTooLow,
            );
            return true;
        }
        if request.release.value > crate::multiversion::Release::MINIMUM.value {
            self.send_eviction_message_to_client(
                request.client,
                message_header::Reason::ClientReleaseTooHigh,
            );
            return true;
        }

        // Compatibility safeguard (upstream 6357-6379): `Request::invalid_header`
        // accepts a `.register` without a body (legacy clients), so such a
        // request must be evicted rather than silently dropped. The only sizes
        // invalid_header admits for a register are `SIZE` and
        // `SIZE + @sizeOf(RegisterRequest)`, so a header-only request is
        // precisely the legacy no-body one.
        if request.operation == crate::Operation::REGISTER
            && request.size == message_header::SIZE_U32
        {
            self.send_eviction_message_to_client(
                request.client,
                message_header::Reason::InvalidRequestBodySize,
            );
            return true;
        }

        // Stale duplicates and resends of already-committed requests (upstream
        // `ignore_request_message_duplicate`, replica.zig:6498). For a client
        // with a session, the request number is ordered against the latest
        // committed reply: older requests and request collisions are dropped,
        // an exact resend re-serves the stored reply, and a brand-new request
        // is accepted only when the client proves it received the last reply.
        if let Some(entry) = self.client_sessions.get(request.client) {
            assert_eq!(entry.header.command, crate::command::Command::Reply);
            assert_eq!(entry.header.client, request.client);
            assert_ne!(entry.header.client, 0);

            // A (repeated) register skips the session checks and falls
            // through to the request-number checks below, so the register
            // reply can be re-served (upstream replica.zig:6516-6517).
            if request.operation != crate::Operation::REGISTER {
                if entry.session > request.session {
                    // The client is borrowing a session that is no longer its
                    // own (it was evicted and reused, then a stale request
                    // arrived).
                    self.send_eviction_message_to_client(
                        request.client,
                        message_header::Reason::SessionTooLow,
                    );
                    return true;
                }
                // Sessions are immutable; a newer session is a client bug.
                if entry.session < request.session {
                    return true;
                }
            }

            if entry.header.release.value != request.release.value {
                self.send_eviction_message_to_client(
                    request.client,
                    message_header::Reason::SessionReleaseMismatch,
                );
                return true;
            }

            if entry.header.request > request.request {
                return true; // Older request.
            }
            if entry.header.request == request.request {
                if request.checksum == entry.header.request_checksum {
                    assert_eq!(entry.header.operation, request.operation);
                    // DEVIATION: the entry is copied out so that the
                    // repeat-reply helper can take `&mut self` (see its note).
                    self.on_request_repeat_reply(request, entry.header);
                }
                // Else: request collision (client bug) — drop.
                return true;
            }
            if entry.header.request + 1 == request.request {
                if request.parent == entry.header.context {
                    return false; // New request; the client acked the last reply.
                }
                // The client may only have one request inflight.
                return true;
            }
            return true; // Newer request (client/network bug).
        }
        if request.operation == crate::Operation::REGISTER {
            return false; // New session.
        }
        // The client's register may still be in the pipeline; a follow-up
        // request must wait for it to commit (upstream replica.zig:6599).
        if self.pipeline_queue.prepare_queue.iter().any(|prepare| prepare.client == request.client)
        {
            return true;
        }
        if request.client == 0 {
            assert_eq!(request.request, 0); // Pulse/upgrade (invalid_header enforces).
            return false;
        }
        // No session: the client must register first.
        self.send_eviction_message_to_client(request.client, message_header::Reason::NoSession);
        true
    }

    /// Re-serve the stored reply to a client that resends an already-committed
    /// request (upstream `on_request_repeat_reply`, replica.zig:6631).
    ///
    /// A body-less reply lives in the `client_sessions` trailer and is rebuilt
    /// directly. A body-ful reply (register) is served from the in-flight write
    /// if one is pending, else read from the client-replies zone; the read
    /// surfaces as a [`crate::client_replies::Event::ReadReply`] with
    /// `destination_replica = None`, and `poll_client_replies` routes it to the
    /// client.
    ///
    /// Streams the stored reply header instead of `&Entry` so this method may
    /// take `&mut self` (the caller already borrows `client_sessions`);
    /// the client-replies calls below re-lookup the session by client.
    ///
    /// # Panics
    /// Panics if `request` does not match the stored reply, or if the session's
    /// client-replies slot is unknown (upstream asserts).
    fn on_request_repeat_reply(
        &mut self,
        request: &message_header::Request,
        reply_header: message_header::Reply,
    ) {
        assert_eq!(request.request, reply_header.request);
        assert_eq!(request.checksum, reply_header.request_checksum);
        assert_eq!(self.status, Status::Normal);

        // DEVIATION: sans-IO has no message pool; a header-only reply is built
        // into a fresh `Message` (upstream `create_message_from_header`).
        if reply_header.size == message_header::SIZE_U32 {
            let mut message = crate::message::Message::new();
            message.set_header(&reply_header);
            self.send_reply_message_to_client(&message);
            return;
        }

        let Some(slot) = self.client_sessions.get_slot_for_client(request.client) else {
            panic!("session slot must exist for a registered client");
        };
        let entry = crate::client_sessions::Entry { session: 0, header: reply_header };
        if let Some(message) = self.client_replies.write_in_flight_latest(slot, &entry).cloned() {
            self.send_reply_message_to_client(&message);
            return;
        }

        let Some(storage) = self.grid_storage.as_mut() else {
            // DEVIATION: the sans-IO replica has no storage until one is
            // mounted via `grid_storage` (see that field's DEVIATION note).
            return;
        };
        if let Err(crate::client_replies::ReadError::Busy) =
            self.client_replies.read_reply(storage, slot, &entry, None)
        {
            // Upstream: the client must retry when client_replies is busy.
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used, // mirrors grid.rs's test module
        clippy::field_reassign_with_default, // typed headers keep reserved fields private
        clippy::cast_possible_truncation // test helpers build wire sizes that fit u32
    )]

    use super::*;
    use crate::Zone;
    use crate::grid::GridOptions;
    use crate::message::Message;
    use crate::message_header::BlockType;
    use crate::multiversion::Release;
    use tigerbeetle_core::constants::BLOCK_SIZE;
    use tigerbeetle_lsm::free_set::SHARD_BITS;

    const CLUSTER: u128 = 0xDEAD;

    fn test_grid() -> Grid {
        Grid::new(GridOptions {
            cache_blocks_count: 64,
            stash_blocks_count: 12,
            read_iops_max: 2,
            write_iops_max: 2,
            // Free-set bootstrap: two shards worth of addresses (must be a
            // multiple of `SHARD_BITS`; grid.rs uses the same sizing).
            free_set_blocks_count: Some(2 * SHARD_BITS),
            free_set_blocks_capacity: None,
        })
    }

    fn test_storage() -> MemoryStorage {
        MemoryStorage::new(Zone::Grid.start() + 64 * BLOCK_SIZE as u64)
    }

    fn mount_test_grid(r: &mut Replica) {
        r.mount_grid(test_grid(), test_storage());
    }

    fn write_block_header(buffer: &mut [u8], address: u64, body_len: usize) -> u128 {
        let mut header = message_header::Block::default();
        header.cluster = CLUSTER;
        header.size = (message_header::SIZE + body_len) as u32;
        header.release = Release { value: 1 };
        header.address = address;
        header.block_type_ordinal = BlockType::FreeSet as u8;

        let body = vec![0xAB_u8; body_len];
        header.checksum_body = header.calculate_checksum_body(&body);
        header.set_checksum();
        buffer[..message_header::SIZE].copy_from_slice(&header.to_wire());
        buffer[message_header::SIZE..message_header::SIZE + body.len()].copy_from_slice(&body);
        header.checksum
    }

    /// Creates a full-size block for a freshly acquired address, returns the
    /// `(address, checksum, bytes)` — the write completed and the block is cached.
    fn write_block(grid: &mut Grid, storage: &mut MemoryStorage) -> (u64, u128, Vec<u8>) {
        let reservation = grid.reserve(1);
        let address = grid.acquire(reservation);

        let mut bytes = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut bytes, address, BLOCK_SIZE - message_header::SIZE);

        let location = grid.get_block();
        grid.block_mut(location).copy_from_slice(&bytes);
        grid.create_block(storage, address, location);
        grid.poll(storage);
        let _ = grid.take_events();
        assert!(grid.cached_location(address).is_some());
        (address, checksum, bytes)
    }

    /// Builds a `GetBlocks` message from `source` requesting `requests` (address, checksum).
    fn get_blocks_message(source: u8, requests: &[(u64, u128)]) -> Message {
        let mut header = message_header::GetBlocks::default();
        header.cluster = CLUSTER;
        header.replica = source;

        let mut body = Vec::with_capacity(requests.len() * size_of::<crate::BlockRequest>());
        for (address, checksum) in requests {
            body.extend_from_slice(&checksum.to_le_bytes());
            body.extend_from_slice(&address.to_le_bytes());
            body.extend_from_slice(&[0_u8; 8]);
        }
        header.size = (message_header::SIZE + body.len()) as u32;
        header.set_checksum_body(&body);
        header.set_checksum();

        let mut message = Message::new();
        message.set_header(&header);
        message.set_body(&body);
        message
    }

    /// Builds a valid `Ping` message from `source`, echoing the minimum release
    /// (upstream `ping_message_release_list`).
    fn ping_message(source: u8, ping_timestamp_monotonic: u64) -> Message {
        let mut header = message_header::Ping::default();
        header.cluster = CLUSTER;
        header.replica = source;
        header.release = crate::multiversion::Release::MINIMUM;
        header.checkpoint_op = 0;
        header.ping_timestamp_monotonic = ping_timestamp_monotonic;
        header.release_count = 1;

        let body_len =
            size_of::<crate::multiversion::Release>() * constants::VSR_RELEASES_MAX as usize;
        let mut body = vec![0_u8; body_len];
        body[..size_of::<crate::multiversion::Release>()]
            .copy_from_slice(&crate::multiversion::Release::MINIMUM.value.to_le_bytes());
        header.size = (message_header::SIZE + body.len()) as u32;
        header.set_checksum_body(&body);
        header.set_checksum();

        let mut message = Message::new();
        message.set_header(&header);
        message.set_body(&body);
        message
    }

    /// Builds a valid `Pong` message from `source`.
    fn pong_message(source: u8) -> Message {
        let mut header = message_header::Pong::default();
        header.cluster = CLUSTER;
        header.replica = source;
        header.release = crate::multiversion::Release::MINIMUM;
        header.ping_timestamp_monotonic = 7;
        header.pong_timestamp_wall = 9;
        header.size = message_header::SIZE as u32;
        header.set_checksum_body(&[]);
        header.set_checksum();

        let mut message = Message::new();
        message.set_header(&header);
        message
    }

    /// Builds a valid `PingClient` message from `client`.
    fn ping_client_message(client: u128, session: u64, ping_timestamp: u64) -> Message {
        let mut header = message_header::PingClient::default();
        header.cluster = CLUSTER;
        header.client = client;
        header.release = crate::multiversion::Release::MINIMUM;
        header.ping_timestamp_monotonic = ping_timestamp;
        header.session = session;
        header.size = message_header::SIZE as u32;
        header.set_checksum_body(&[]);
        header.set_checksum();

        let mut message = Message::new();
        message.set_header(&header);
        message
    }

    /// Builds a valid (minimum size, no body) `Request` message from `client`.
    /// Builds a valid (minimum size, no body) `Request` message from `client`
    /// with the arbitrary session `42` (upstream clients number sessions per
    /// register commit; a dedicated helper parameter is available when a
    /// session must match an existing one).
    fn request_message(client: u128, request: u32, operation: crate::Operation) -> Message {
        request_message_full(client, request, operation, 42, 0, message_header::SIZE as u32)
    }

    /// Like [`request_message`], with explicit `session`, `parent` and `size`.
    ///
    /// A `size` larger than the header size carries a zeroed body (used by
    /// body-ful register requests; on_request does not inspect the contents,
    /// but the header's checksum chain must cover them).
    fn request_message_full(
        client: u128,
        request: u32,
        operation: crate::Operation,
        session: u64,
        parent: u128,
        size: u32,
    ) -> Message {
        let mut header = message_header::Request::default();
        header.cluster = CLUSTER;
        header.client = client;
        header.session = session;
        header.request = request;
        header.operation = operation;
        header.release = crate::multiversion::Release::MINIMUM;
        header.parent = parent;
        header.view = 0;
        header.size = size;
        let body: Vec<u8> = vec![0; size.saturating_sub(message_header::SIZE as u32) as usize];
        header.set_checksum_body(&body);
        header.set_checksum();

        let mut message = Message::new();
        message.set_header(&header);
        if !body.is_empty() {
            message.set_body(&body);
        }
        message
    }

    /// Commits a register for `client` at the next op, returning that op
    /// (its commit number becomes the client's session number).
    fn register_client(r: &mut Replica, client: u128) -> u64 {
        let op = r.primary_pipeline_prepare(client, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        r.commit_max = op;
        r.commit_prepare = Some(op);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        assert_eq!(r.client_sessions.get(client).expect("registered session").session, op,);
        r.send_queue.clear();
        r.client_send_queue.clear();
        op
    }

    #[test]
    fn grid_mount_and_repair_budget_construction() {
        let mut r = Replica::new(CLUSTER, 1, 3);
        assert!(r.grid.is_none());
        assert!(r.grid_storage.is_none());

        // The grid repair budget covers every remote replica (`replica_count - 1`)
        // with a full `GetBlocks` worth of budget each.
        assert_eq!(r.grid_repair_message_budget.budget_available(0), 5);
        assert_eq!(r.grid_repair_message_budget.budget_available(2), 5);

        mount_test_grid(&mut r);

        assert!(r.grid.is_some());
        assert!(r.grid_storage.is_some());
    }

    #[test]
    fn on_get_blocks_serves_a_requested_block() {
        let mut r = Replica::new(CLUSTER, 0, 2);
        mount_test_grid(&mut r);
        let (address, checksum, expected) = {
            let (grid, storage) = r.grid_mut();
            write_block(grid, storage)
        };

        let request = get_blocks_message(1, &[(address, checksum)]);
        r.on_message(&request, 0);

        // The read is in flight until pumped:
        assert_eq!(r.grid_serve_reads.len(), 1);

        r.poll_grid();

        // No grid reads left; exactly one Block reply was sent.
        assert!(r.grid_serve_reads.is_empty());
        assert_eq!(r.send_queue.len(), 1);

        let reply = &r.send_queue[0];
        // The reply *is* the block: buffer[0..block_size], its own header parsed
        // as the message header (upstream `on_get_blocks_read_block`).
        assert_eq!(reply.buffer(), expected.as_slice());
        assert_eq!(reply.size_raw(), BLOCK_SIZE as u32);
        let header = reply.header::<message_header::Block>().expect("reply is a block");
        assert!(header.valid_checksum());
        assert!(header.valid_checksum_body(reply.body_used()));
        assert_eq!(header.cluster, CLUSTER);
        assert_eq!(header.address, address);
        assert_eq!(header.release, Release { value: 1 });
        assert_eq!(header.block_type_ordinal, BlockType::FreeSet as u8);
        assert_eq!(header.size, BLOCK_SIZE as u32);
        assert_eq!(reply.body_used().len(), BLOCK_SIZE - message_header::SIZE);
    }

    #[test]
    fn on_get_blocks_dedupes_inflight_and_within_message() {
        let mut r = Replica::new(CLUSTER, 0, 2);
        mount_test_grid(&mut r);
        let (address, checksum, expected) = {
            let (grid, storage) = r.grid_mut();
            write_block(grid, storage)
        };

        // The same block twice in one message, then again in a second message.
        let request = get_blocks_message(1, &[(address, checksum), (address, checksum)]);
        let request_again = get_blocks_message(1, &[(address, checksum)]);
        r.on_message(&request, 0);
        r.on_message(&request_again, 0);

        assert_eq!(r.grid_serve_reads.len(), 1, "one outstanding read, deduped twice");

        r.poll_grid();

        assert!(r.grid_serve_reads.is_empty());
        assert_eq!(r.send_queue.len(), 1, "one reply, not three");
        assert_eq!(r.send_queue[0].buffer(), expected.as_slice());
    }

    #[test]
    fn on_get_blocks_ignores_bad_cluster() {
        let mut r = Replica::new(CLUSTER, 0, 2);
        mount_test_grid(&mut r);

        // Wrong cluster: dropped in `on_message` before any handler runs.
        let mut header = message_header::GetBlocks::default();
        header.cluster = CLUSTER + 1;
        header.replica = 1;
        header.size = (message_header::SIZE + size_of::<crate::BlockRequest>()) as u32;
        let mut body = Vec::with_capacity(size_of::<crate::BlockRequest>());
        body.extend_from_slice(&0xAB_u128.to_le_bytes());
        body.extend_from_slice(&1_u64.to_le_bytes());
        body.extend_from_slice(&[0_u8; 8]);
        header.set_checksum_body(&body);
        header.set_checksum();
        let mut request = Message::new();
        request.set_header(&header);
        request.set_body(&body);

        r.on_message(&request, 0);
        assert!(r.send_queue.is_empty());
        assert!(r.grid_serve_reads.is_empty());
    }

    #[test]
    fn on_get_blocks_ignores_misdirected_self() {
        let mut r = Replica::new(CLUSTER, 0, 2);
        assert!(r.grid.is_none());
        // Not mounted: ignored (upstream `grid.callback == .cancel`) regardless of direction.
        let request = get_blocks_message(1, &[(1, 0xAB)]);
        r.on_message(&request, 0);
        assert!(r.send_queue.is_empty());
        assert!(r.grid_serve_reads.is_empty());

        // Misdirected (to self): ignored even with a mounted grid.
        mount_test_grid(&mut r);
        let request = get_blocks_message(0, &[(1, 0xAB)]);
        r.on_message(&request, 0);
        assert!(r.send_queue.is_empty());
        assert!(r.grid_serve_reads.is_empty());
    }

    /// Issue a coherent read for `(address, checksum)` and pump the grid until
    /// the read parks in the read-global queue (i.e. repairable-from-remote).
    fn park_read(r: &mut Replica, address: u64, checksum: u128) {
        let (grid, storage) = r.grid_mut();
        grid.read_block(
            storage,
            address,
            checksum,
            true,
            ReadOptions { cache_read: true, cache_write: false },
        );
        grid.poll(storage);
        assert_eq!(grid.read_global_queue_len(), 1);
    }

    #[test]
    fn on_block_repairs_parked_reads_and_persists() {
        // Two replicas in one cluster: r0 holds the block, r1's coherent read
        // for it is parked in the grid read-global queue (nothing stored
        // locally), so the grid repair timeout requests it over the
        // GetBlocks/Block protocol.
        let mut r0 = Replica::new(CLUSTER, 0, 2);
        mount_test_grid(&mut r0);
        let (address, checksum, expected) = {
            let (grid, storage) = r0.grid_mut();
            write_block(grid, storage)
        };

        let mut r1 = Replica::new(CLUSTER, 1, 2);
        mount_test_grid(&mut r1);
        // r1's free set must also consider the address allocated for its
        // coherent read to be legal (`read_block` asserts non-free).
        {
            let (grid, _storage) = r1.grid_mut();
            let reservation = grid.reserve(1);
            assert_eq!(grid.acquire(reservation), address);
        }
        r1.status = Status::Normal;
        r1.grid_repair_timeout = Timeout::start(constants::GRID_REPAIR_TIMEOUT);

        // Park the coherent read: storage holds no block at `address`.
        park_read(&mut r1, address, checksum);

        // The grid repair timeout picks a peer with budget left and requests.
        r1.on_grid_repair_timeout();
        assert_eq!(r1.send_queue.len(), 1);
        let get_blocks = r1.send_queue.remove(0);
        let request = get_blocks.header::<message_header::GetBlocks>().expect("get_blocks");
        assert!(request.valid_checksum());
        assert!(request.valid_checksum_body(get_blocks.body_used()));
        assert_eq!(request.replica, 1);
        assert_eq!(request.cluster, CLUSTER);

        // r0 serves it: the reply message *is* the block.
        r0.on_message(&get_blocks, 0);
        r0.poll_grid();
        assert_eq!(r0.send_queue.len(), 1);
        let reply = r0.send_queue.remove(0);
        assert_eq!(reply.buffer(), expected.as_slice());

        // r1 receives the repair block: the parked read is fulfilled, the
        // budget is replenished, and nothing further needs requesting.
        r1.on_message(&reply, 0);
        assert_eq!(r1.send_queue.len(), 0);
        assert_eq!(r1.grid_mut().0.read_global_queue_len(), 0, "parked read fulfilled");
        // Complete the durable repair-block write to storage.
        r1.poll_grid();

        // A fresh coherent read now resolves from storage (the block persisted).
        let (grid, storage) = r1.grid_mut();
        let token = grid.read_block(
            storage,
            address,
            checksum,
            true,
            ReadOptions { cache_read: true, cache_write: false },
        );
        grid.poll(storage);
        let events = grid.take_events();
        assert!(
            events.iter().any(|event| matches!(
                event,
                Event::ReadDone { token: done_token, result: ReadBlockResult::Valid, .. }
                    if *done_token == token
            )),
            "fresh read of the repaired block succeeds"
        );
    }

    #[test]
    fn on_grid_repair_timeout_fires_repair_roundtrip_over_ticks() {
        // Same topology as `on_block_repairs_parked_reads_and_persists`, but the
        // requester drives the repair through the real timeout path (`tick()`
        // firing `grid_repair_timeout`), not by calling the handler directly.
        let mut r0 = Replica::new(CLUSTER, 0, 2);
        mount_test_grid(&mut r0);
        let (address, checksum, expected) = {
            let (grid, storage) = r0.grid_mut();
            write_block(grid, storage)
        };

        let mut r1 = Replica::new(CLUSTER, 1, 2);
        mount_test_grid(&mut r1);
        {
            let (grid, _storage) = r1.grid_mut();
            let reservation = grid.reserve(1);
            assert_eq!(grid.acquire(reservation), address);
        }
        r1.status = Status::Normal;
        r1.grid_repair_timeout = Timeout::start(constants::GRID_REPAIR_TIMEOUT);
        park_read(&mut r1, address, checksum);

        // Reactor loop: the grid repair timeout fires after `GRID_REPAIR_TIMEOUT`
        // ticks and r1 requests the parked block. The cluster clock stands still
        // for the grid path, so both replicas tick with `now = 0`.
        for _ in 0..constants::GRID_REPAIR_TIMEOUT {
            r0.tick(0);
            r1.tick(0);
        }
        assert_eq!(r1.send_queue.len(), 1, "grid repair timeout fired");
        let get_blocks = r1.send_queue.remove(0);

        // r0 serves the reply.
        r0.on_message(&get_blocks, 0);
        r0.poll_grid();
        assert_eq!(r0.send_queue.len(), 1);
        let reply = r0.send_queue.remove(0);
        assert_eq!(reply.buffer(), expected.as_slice());

        // r1 consumes it: the parked read is fulfilled, nothing further is
        // requested, and the block is durably written.
        r1.on_message(&reply, 0);
        assert_eq!(r1.send_queue.len(), 0, "no further blocks to request");
        assert_eq!(r1.grid_mut().0.read_global_queue_len(), 0, "parked read fulfilled");
        r1.poll_grid();

        let (grid, storage) = r1.grid_mut();
        let token = grid.read_block(
            storage,
            address,
            checksum,
            true,
            ReadOptions { cache_read: true, cache_write: false },
        );
        grid.poll(storage);
        let events = grid.take_events();
        assert!(
            events.iter().any(|event| matches!(
                event,
                Event::ReadDone { token: done_token, result: ReadBlockResult::Valid, .. }
                    if *done_token == token
            )),
            "fresh read of the repaired block succeeds"
        );
    }

    #[test]
    fn send_get_blocks_requests_at_most_grid_repair_request_max_blocks() {
        // Five parked reads but a per-destination budget of four requests per
        // `GetBlocks` message: only `GRID_REPAIR_REQUEST_MAX` are requested, the
        // fifth stays parked for a later round.
        let mut r = Replica::new(CLUSTER, 1, 2);
        mount_test_grid(&mut r);
        let requests: Vec<(u64, u128)> =
            (1_u64..=5).map(|address| (address, 0x1000_u128 + u128::from(address))).collect();
        {
            let (grid, storage) = r.grid_mut();
            let reservation = grid.reserve(requests.len());
            for (address, _) in &requests {
                assert_eq!(grid.acquire(reservation), *address);
            }
            for (address, checksum) in &requests {
                grid.read_block(
                    storage,
                    *address,
                    *checksum,
                    true,
                    ReadOptions { cache_read: true, cache_write: false },
                );
            }
            grid.poll(storage);
            assert_eq!(grid.read_global_queue_len(), 5);
        }
        r.status = Status::Normal;
        r.grid_repair_timeout = Timeout::start(constants::GRID_REPAIR_TIMEOUT);

        r.on_grid_repair_timeout();
        assert_eq!(r.send_queue.len(), 1);
        let get_blocks = r.send_queue.remove(0);
        let request = get_blocks.header::<message_header::GetBlocks>().expect("get_blocks");
        assert!(request.valid_checksum());
        assert!(
            request.valid_checksum_body(get_blocks.body_used()),
            "request body carries exactly the requested blocks"
        );
        assert_eq!(
            request.size as usize,
            message_header::SIZE
                + usize::from(constants::GRID_REPAIR_REQUEST_MAX)
                    * size_of::<crate::BlockRequest>()
        );

        // Requests do not drain the global queue — parked reads are removed only
        // by a fulfilling `Block` (see `on_block_repairs_parked_reads_and_persists`).
        // Here the cap shows as the message size and the budget drawdown:
        // four of the five blocks are requested, leaving one request of budget.
        assert_eq!(r.grid_mut().0.read_global_queue_len(), 5, "still parked until fulfilled");
        assert_eq!(r.grid_repair_message_budget.budget_available(0), 1);
    }

    #[test]
    fn grid_repair_expiry_restores_budget_and_resends() {
        // Same five-parked-read topology as the cap test. Once the four
        // outstanding requests expire (GRID_REPAIR_EXPIRY later), the budget is
        // restored and the grid repair timeout re-requests the blocks.
        let mut r = Replica::new(CLUSTER, 1, 2);
        mount_test_grid(&mut r);
        let requests: Vec<(u64, u128)> =
            (1_u64..=5).map(|address| (address, 0x1000_u128 + u128::from(address))).collect();
        {
            let (grid, storage) = r.grid_mut();
            let reservation = grid.reserve(requests.len());
            for (address, _) in &requests {
                assert_eq!(grid.acquire(reservation), *address);
            }
            for (address, checksum) in &requests {
                grid.read_block(
                    storage,
                    *address,
                    *checksum,
                    true,
                    ReadOptions { cache_read: true, cache_write: false },
                );
            }
            grid.poll(storage);
            assert_eq!(grid.read_global_queue_len(), 5);
        }
        r.status = Status::Normal;
        r.grid_repair_timeout = Timeout::start(constants::GRID_REPAIR_TIMEOUT);

        // First round (t=0): four blocks requested, one request of budget left.
        r.on_grid_repair_timeout();
        assert_eq!(r.send_queue.len(), 1);
        assert_eq!(r.grid_repair_message_budget.budget_available(0), 1);

        // A round before expiry (t=0): the requests are still outstanding and
        // the budget is below GRID_REPAIR_REQUEST_MAX, so nothing is re-sent.
        r.send_queue.clear();
        r.on_grid_repair_timeout();
        assert!(r.send_queue.is_empty(), "budget exhausted until the requests expire");

        // After expiry (t = GRID_REPAIR_EXPIRY + 1ms): the budget is restored
        // and the blocks are re-requested from the parked reads.
        r.monotonic_now = 251_000_000;
        r.on_grid_repair_timeout();
        assert_eq!(r.send_queue.len(), 1, "expired requests are re-issued");
        assert_eq!(r.grid_repair_message_budget.budget_available(0), 1);
        let get_blocks = r.send_queue.remove(0);
        let request = get_blocks.header::<message_header::GetBlocks>().expect("get_blocks");
        assert!(request.valid_checksum());
        assert_eq!(
            request.size as usize,
            message_header::SIZE
                + usize::from(constants::GRID_REPAIR_REQUEST_MAX)
                    * size_of::<crate::BlockRequest>()
        );
    }

    #[test]
    fn quorums_1_replica() {
        let q = quorums(1);
        assert_eq!(q.replication, 1);
        assert_eq!(q.view_change, 1);
        assert_eq!(q.nack_prepare, 1);
        assert_eq!(q.majority, 1);
    }

    #[test]
    fn quorums_3_replicas() {
        let q = quorums(3);
        assert_eq!(q.replication, 2);
        assert_eq!(q.view_change, 2);
        assert_eq!(q.nack_prepare, 2);
        assert_eq!(q.majority, 2);
    }

    #[test]
    fn quorums_4_replicas() {
        let q = quorums(4);
        assert_eq!(q.replication, 2);
        assert_eq!(q.view_change, 3);
        assert_eq!(q.nack_prepare, 3);
        assert_eq!(q.majority, 3);
    }

    #[test]
    fn quorums_5_replicas() {
        let q = quorums(5);
        assert_eq!(q.replication, 3);
        assert_eq!(q.view_change, 3);
        assert_eq!(q.nack_prepare, 3);
        assert_eq!(q.majority, 3);
    }

    #[test]
    fn quorums_6_replicas() {
        let q = quorums(6);
        assert_eq!(q.replication, 3);
        assert_eq!(q.view_change, 4);
        assert_eq!(q.nack_prepare, 4);
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
        let p = PipelinePrepare {
            op: 10,
            checksum: 0xAB,
            client: 1,
            acks_received: 0,
            ok_quorum_received: false,
        };
        assert!(cache.insert(p).is_none());
        assert!(cache.find(10, 0xAB).is_some());
        assert!(cache.find(10, 0xCD).is_none());
        assert!(cache.find(11, 0xAB).is_none());
    }

    #[test]
    fn pipeline_cache_eviction() {
        let mut cache = PipelineCache::new(2);
        let p1 = PipelinePrepare {
            op: 0,
            checksum: 1,
            client: 1,
            acks_received: 0,
            ok_quorum_received: false,
        };
        let p2 = PipelinePrepare {
            op: 2,
            checksum: 2,
            client: 1,
            acks_received: 0,
            ok_quorum_received: false,
        };
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
        let mut message = crate::message::Message::new();
        message.frame_mut().copy_from_slice(&header.to_wire());
        // Should silently drop — no panic, no state change.
        r.on_message(&message, 0);
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
    fn on_request_primary_prepares_immediately() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        assert!(r.is_primary());

        // An unregistered client's register starts a new session and is
        // accepted (upstream `ignore_request_message`: new session). A body-ful
        // register (the legacy body-less variant is evicted upstream).
        let register_size = message_header::SIZE_U32 + message_header::REGISTER_RESULT_SIZE_U32;
        let request = request_message_full(1, 0, crate::Operation::REGISTER, 0, 0, register_size);
        r.on_message(&request, 0);

        assert_eq!(r.op, 1);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 1);
        assert_eq!(r.pipeline_queue.request_queue.len(), 0);
        assert_eq!(r.pipeline_queue.prepare_queue[0].op, 1);
        // The primary contributes its own prepare_ok.
        assert_eq!(r.pipeline_queue.prepare_queue[0].acks_received, 1);
        // And the prepare is broadcast to both backups (2).
        assert_eq!(r.send_queue.len(), 2);

        let header = r.journal.header_with_op(1).expect("journal head");
        assert_eq!(header.client, 1);
        assert_eq!(header.request, 0);
        assert_eq!(header.operation, crate::Operation::REGISTER);
        // The prepare echoes the request's checksum (for repeat-reply matching).
        assert_eq!(
            header.request_checksum,
            request.header::<message_header::Request>().unwrap().checksum
        );
    }

    #[test]
    fn on_request_queues_when_pipeline_full() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Register client 2 and commit one body-less noop (request 1), so a
        // request `2` is its next (new) request with a real session. Request
        // numbers are strictly sequential per client (upstream
        // `client_table_entry_update` asserts `request + 1`).
        let op = r.primary_pipeline_prepare(2, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        let op = r.primary_pipeline_prepare(2, 1, crate::Operation::NOOP, 0, 0).unwrap();
        assert_eq!(op, 2);
        r.commit_max = 2;
        r.commit_prepare = Some(2);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        let session = r.client_sessions.get(2).expect("client 2 session");
        let (session_number, parent) = (session.session, session.header.context);

        for _ in 0..constants::PIPELINE_PREPARE_QUEUE_MAX {
            let op = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
            assert!(op > 0);
        }
        let prepare_count = constants::PIPELINE_PREPARE_QUEUE_MAX as usize;
        assert_eq!(r.pipeline_queue.prepare_queue.len(), prepare_count);

        // Client 2's next request (request 2, parent = last reply's context)
        // passes the duplicate checks but finds the pipeline full.
        let request = request_message_full(
            2,
            2,
            crate::Operation::NOOP,
            session_number,
            parent,
            message_header::SIZE as u32,
        );
        // Drain the broadcast noise from the register/request-1 setup above.
        r.send_queue.clear();
        r.on_message(&request, 0);

        assert_eq!(r.pipeline_queue.prepare_queue.len(), prepare_count);
        assert_eq!(r.pipeline_queue.request_queue.len(), 1);
        assert_eq!(r.pipeline_queue.request_queue[0].client, 2);
        assert_eq!(r.pipeline_queue.request_queue[0].request, 2);
        assert_eq!(
            r.pipeline_queue.request_queue[0].request_checksum,
            request.header::<message_header::Request>().unwrap().checksum
        );
        // No broadcast for a queued request.
        assert_eq!(r.send_queue.len(), 0);
    }

    #[test]
    fn on_request_commit_execute_drains_request_queue() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        assert!(r.is_primary());
        assert!(r.commit_min == 0 && r.commit_max == 0);

        // Pipeline a first op and queue a second client request behind it.
        let op = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.advance_commit_max(1);
        r.pipeline_queue.request_queue.push(PipelineRequest {
            client: 7,
            request: 9,
            request_checksum: 0,
            operation: crate::Operation::NOOP,
        });

        // Drive the execute stage manually: commit_prepare = op 1.
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        // The queued request was prepared as op 2.
        assert_eq!(r.op, 2);
        assert_eq!(r.pipeline_queue.request_queue.len(), 0);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 1);
        assert_eq!(r.pipeline_queue.prepare_queue[0].op, 2);
        let header = r.journal.header_with_op(2).expect("journal op 2");
        assert_eq!(header.client, 7);
        assert_eq!(header.request, 9);
        assert_eq!(header.operation, crate::Operation::NOOP);
    }

    #[test]
    fn commit_execute_register_creates_session() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // A register op (request 0, per Request::invalid_header) at commit 1.
        let op = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        assert_eq!(r.client_sessions.count(), 1);
        let entry = r.client_sessions.get(7).expect("registered client");
        // The register op's commit number becomes the session number.
        assert_eq!(entry.session, 1);
        assert_eq!(entry.header.operation, crate::Operation::REGISTER);
        assert_eq!(entry.header.client, 7);
        assert_eq!(entry.header.op, 1);
        assert_eq!(entry.header.commit, 1);
        assert_eq!(entry.header.request, 0);
        assert_eq!(
            entry.header.size,
            message_header::SIZE_U32 + message_header::REGISTER_RESULT_SIZE_U32
        );
        assert_ne!(entry.header.context, 0, "stable-with-view reply checksum");
        assert!(entry.header.valid_checksum());
    }

    #[test]
    fn commit_execute_noop_updates_registered_session() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Register client 7 (commit 1).
        let op = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        let entry = r.client_sessions.get(7).expect("registered client");
        assert_eq!(entry.session, 1);
        assert_eq!(entry.header.request, 0);

        // Client 7's next request (request 1, noop) commits as op 2.
        let op = r.primary_pipeline_prepare(7, 1, crate::Operation::NOOP, 0, 0).unwrap();
        assert_eq!(op, 2);
        r.commit_max = 2;
        r.commit_prepare = Some(2);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        let entry = r.client_sessions.get(7).expect("registered client");
        assert_eq!(r.client_sessions.count(), 1);
        // The session number stays the register's commit; the header is the
        // client's latest committed reply.
        assert_eq!(entry.session, 1);
        assert_eq!(entry.header.operation, crate::Operation::NOOP);
        assert_eq!(entry.header.request, 1);
        assert_eq!(entry.header.op, 2);
        assert_eq!(entry.header.commit, 2);
    }

    #[test]
    fn commit_execute_primary_replies_to_client() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        let op = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        r.commit_max = op;
        r.commit_prepare = Some(op);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        // The primary always replies to the client.
        assert_eq!(r.client_send_queue.len(), 1);
        let reply = r.client_send_queue[0].header::<message_header::Reply>().expect("reply");
        assert_eq!(reply.client, 7);
        assert_eq!(reply.commit, 1);
        assert_eq!(reply.request, 0);
        assert_eq!(reply.op, reply.commit);
        assert_eq!(reply.view, r.log_view);
        assert_eq!(reply.replica, r.replica_u8());
        assert!(reply.valid_checksum());
        assert!(reply.valid_checksum_body(r.client_send_queue[0].body_used()));
    }

    #[test]
    fn commit_execute_pulse_does_not_reply_to_client() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // A pulse has no client, so nothing is queued for a client.
        let op = r.primary_pipeline_prepare(0, 0, crate::Operation::PULSE, 0, 0).unwrap();
        r.commit_max = op;
        r.commit_prepare = Some(op);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        assert_eq!(r.client_send_queue.len(), 0, "pulse has no client to reply to");
    }

    #[test]
    fn execute_op_reply_to_client_backups_are_deterministic() {
        // The primary always replies.
        let r0 = Replica::new(CLUSTER, 0, 3);
        assert!(r0.execute_op_reply_to_client(1));

        // Exactly one backup replies per op, selected deterministically.
        let r1 = Replica::new(CLUSTER, 1, 3);
        let r2 = Replica::new(CLUSTER, 2, 3);
        for op in 1..=16 {
            let a = r1.execute_op_reply_to_client(op);
            let b = r2.execute_op_reply_to_client(op);
            assert_ne!(a, b, "replicas 1 and 2 must pick exactly one replier (op {op})");
            let selected_backup = if a { 1 } else { 2 };
            let mut prng = tigerbeetle_core::stdx::prng::Prng::from_seed(op);
            let offset = 1_u8 + prng.gen_int_inclusive_u8(1);
            assert_eq!(
                selected_backup,
                u16::from(offset) % 3,
                "selection matches the upstream PRNG (op {op})"
            );
        }
    }

    #[test]
    fn send_reply_message_to_client_bumps_stale_view() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.view = 2;
        r.log_view = 1;

        let mut header = message_header::Reply::default();
        header.cluster = CLUSTER;
        header.replica = r.replica_u8();
        header.view = 0; // Committed in an older view than `log_view`.
        header.release = crate::multiversion::Release::MINIMUM;
        header.op = 1;
        header.commit = 1;
        header.timestamp = 1;
        header.client = 7;
        header.request = 1;
        header.request_checksum = 0x1234;
        header.operation = crate::Operation::NOOP;
        header.size = message_header::SIZE_U32;
        header.set_checksum_body(&[]);
        header.context = header.calculate_checksum();
        header.set_checksum();
        let mut message = crate::message::Message::new();
        message.set_header(&header);

        r.send_reply_message_to_client(&message);

        assert_eq!(r.client_send_queue.len(), 1);
        let out = r.client_send_queue[0].header::<message_header::Reply>().expect("reply");
        assert_eq!(out.view, 1, "view is bumped to the durable log view");
        assert!(out.valid_checksum());
        // The checksum body and stable-with-view context survive the bump.
        assert_eq!(out.checksum_body, header.checksum_body);
        assert_eq!(out.context, header.context);
        assert_eq!(out.request_checksum, header.request_checksum);
    }

    #[test]
    fn commit_execute_skips_unregistered_client() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Register client 7 (commit 1), then commit a noop for client 8.
        let op = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        let op = r.primary_pipeline_prepare(8, 1, crate::Operation::NOOP, 0, 0).unwrap();
        assert_eq!(op, 2);
        r.commit_max = 2;
        r.commit_prepare = Some(2);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        assert_eq!(r.client_sessions.count(), 1, "unregistered client creates no entry");
        assert!(r.client_sessions.get(8).is_none());
    }

    #[test]
    fn commit_execute_register_writes_reply_to_storage() {
        use crate::storage::{ReadRequest, Storage, zeroed_buffer};

        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        // Register client 7 (commit 1): commits a body-ful reply.
        let op = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        let slot = r.client_sessions.get_slot_for_client(7).expect("client 7 session");
        assert!(!r.client_replies.reply_durable(slot), "reply not yet written");
        r.poll_client_replies();
        assert!(r.client_replies.reply_durable(slot), "reply durably written");

        // Read the reply back from the client-replies zone and verify it is the
        // registered client's zeroed-`RegisterResult` reply.
        let mut storage = r.grid_storage.take().expect("mounted storage");
        storage.read_sectors(ReadRequest {
            zone: Zone::ClientReplies,
            offset_in_zone: 0,
            buffer: zeroed_buffer(crate::message::MESSAGE_SIZE_MAX),
        });
        let written = match storage.next_completion().expect("read completion") {
            crate::storage::Completion::Read(request) => request.buffer,
            crate::storage::Completion::Write(_) => unreachable!("expected a read"),
        };

        let entry = r.client_sessions.get(7).expect("registered client");
        let written_header = message_header::Reply::from_wire(
            written[..message_header::SIZE].try_into().expect("frame length"),
        )
        .expect("reply frame parses");
        assert_eq!(written_header.checksum, entry.header.checksum);
        assert_eq!(written_header.size, 512, "size was {}", written_header.size);
        assert_eq!(&written[message_header::SIZE..512], &[0_u8; 256]);
    }

    #[test]
    fn commit_execute_noop_does_not_write_reply() {
        use crate::storage::Storage;

        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        // Register client 7 (commit 1) then commit its body-less noop (op 2).
        let op = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        r.poll_client_replies();

        let op = r.primary_pipeline_prepare(7, 1, crate::Operation::NOOP, 0, 0).unwrap();
        assert_eq!(op, 2);
        r.commit_max = 2;
        r.commit_prepare = Some(2);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        // A body-less reply is not written: remove_reply only clears the slot's
        // faulty bit, so the storage stays quiet.
        let slot = r.client_sessions.get_slot_for_client(7).expect("client 7 session");
        assert!(r.client_replies.reply_durable(slot));
        r.poll_client_replies();
        let mut storage = r.grid_storage.take().expect("mounted storage");
        assert!(storage.next_completion().is_none(), "noop produced no write");
    }

    #[test]
    fn commit_execute_tracks_monotonic_commit_timestamp() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        let op1 = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        let timestamp1 = r.journal.header_with_op(op1).expect("prepare").timestamp;
        r.commit_max = op1;
        r.commit_prepare = Some(op1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        assert_eq!(r.commit_timestamp, timestamp1);

        // A later op is stamped strictly later, so execution advances too
        // (upstream `state_machine.commit_timestamp < prepare.header.timestamp`).
        let op2 = r.primary_pipeline_prepare(7, 1, crate::Operation::NOOP, 0, 0).unwrap();
        let timestamp2 = r.journal.header_with_op(op2).expect("prepare").timestamp;
        assert!(timestamp2 > timestamp1);
        r.commit_max = op2;
        r.commit_prepare = Some(op2);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        assert_eq!(r.commit_timestamp, timestamp2);
    }

    #[test]
    #[should_panic(expected = "commit_timestamp < prepare.timestamp")]
    fn commit_execute_panics_on_stale_prepare_timestamp() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        assert_eq!(r.primary_pipeline_prepare(7, 0, crate::Operation::NOOP, 0, 0).unwrap(), 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        // The committed prepare's timestamp must advance the previous one; a
        // backwards (or stalled) state machine clock is a bug.
        r.commit_timestamp = u64::MAX;
        r.commit_execute();
    }

    #[test]
    fn execute_op_reply_routing_exactly_one_backup_per_op() {
        // The primary always replies; among the backups exactly one, selected
        // deterministically by the op (upstream `execute_op_reply_to_client`,
        // replica.zig:5328), so a client retrying against another replica never
        // races with the primary's own reply.
        let primary = Replica::new(CLUSTER, 0, 3);
        let backup1 = Replica::new(CLUSTER, 1, 3);
        let backup2 = Replica::new(CLUSTER, 2, 3);

        let mut selected_by_one = 0;
        let mut selected_by_two = 0;
        for op in 1..=256 {
            assert!(primary.execute_op_reply_to_client(op), "primary always replies");

            let one = backup1.execute_op_reply_to_client(op);
            let two = backup2.execute_op_reply_to_client(op);
            assert_ne!(one, two, "op {op}: exactly one backup replies");
            assert_eq!(one, backup1.execute_op_reply_to_client(op), "op {op}: deterministic");
            assert_eq!(two, backup2.execute_op_reply_to_client(op), "op {op}: deterministic");
            selected_by_one += usize::from(one);
            selected_by_two += usize::from(two);
        }

        assert_eq!(selected_by_one + selected_by_two, 256);
        assert!(selected_by_one > 0 && selected_by_two > 0, "both backups serve some ops");
    }

    #[test]
    fn backup_commit_execute_replies_only_when_selected() {
        let mut backup1 = Replica::new(CLUSTER, 1, 3);
        backup1.status = Status::Normal;
        let mut backup2 = Replica::new(CLUSTER, 2, 3);
        backup2.status = Status::Normal;

        // The same committed op lands on both backups through the shared
        // prepare path.
        let h1 = make_prepare_for_replica(CLUSTER, 0, 1, 0, 0);
        for r in [&mut backup1, &mut backup2] {
            deliver_prepare(r, &h1);
            r.commit_max = 1;
            r.commit_journal();
            assert_eq!(r.commit_min, 1);
        }

        // Exactly one of the two backups turns `execute_op_reply_to_client`
        // into a reply delivered to the client.
        let replies = backup1.client_send_queue.len() + backup2.client_send_queue.len();
        assert_eq!(replies, 1, "exactly one backup replies to the client");
    }

    #[test]
    fn commit_execute_skips_session_evicted_while_preparing() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Client 7 registers first (session 1); clients 8..=13 fill the table,
        // leaving client 7 the oldest entry.
        for client in 7_u64..=13 {
            register_client(&mut r, u128::from(client));
        }
        assert_eq!(r.client_sessions.count(), constants::CLIENTS_MAX as usize);

        // A register for a new client sharing the pipeline with client 7's next
        // request: op 8 (client 14's register) commits first — the table is
        // full, so the oldest entry (client 7) is evicted — and op 9 (client
        // 7's request) executes afterwards.
        assert_eq!(r.primary_pipeline_prepare(14, 0, crate::Operation::REGISTER, 0, 0).unwrap(), 8);
        assert_eq!(r.primary_pipeline_prepare(7, 1, crate::Operation::NOOP, 0, 0).unwrap(), 9);
        r.commit_max = 8;
        r.commit_prepare = Some(8);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        assert_eq!(r.client_sessions.get(7), None, "client 7 evicted at capacity");
        assert_eq!(r.client_sessions.get(14).expect("new session").session, 8);
        r.client_send_queue.clear();

        // Op 9 executes with the session already gone: the reply is still
        // delivered, told the client nothing was tracked (upstream
        // `client_table_entry_update` — the next request receives an
        // eviction).
        r.commit_max = 9;
        r.commit_prepare = Some(9);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        assert_eq!(r.op, 9);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 0);
        assert_eq!(r.client_send_queue.len(), 1, "reply still delivered");
        let reply = r
            .client_send_queue
            .last()
            .expect("reply")
            .header::<message_header::Reply>()
            .expect("reply header");
        assert_eq!(reply.client, 7);
        assert_eq!(reply.op, 9);

        // The evicted client is unregistered, so its next request is evicted:
        r.on_message(&request_message(7, 2, crate::Operation::NOOP), 0);
        let evicted = r
            .client_send_queue
            .last()
            .expect("eviction")
            .header::<message_header::Eviction>()
            .expect("eviction header");
        assert_eq!(evicted.reason(), Some(message_header::Reason::NoSession));
        assert_eq!(evicted.client, 7);
    }

    #[test]
    fn on_request_repeat_reply_register_serves_ram_reply() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        // Register client 7 through the client-facing path so the prepare
        // echoes the real request checksum (a resend can then match it).
        let register_size = message_header::SIZE_U32 + message_header::REGISTER_RESULT_SIZE_U32;
        let register = request_message_full(7, 0, crate::Operation::REGISTER, 0, 0, register_size);
        r.on_message(&register, 0);
        assert_eq!(r.op, 1);
        r.send_queue.clear();

        // Commit the register; the reply write to the client-replies zone is
        // still in flight, so the repeat is served from RAM.
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        assert!(r.client_sessions.get(7).is_some());
        assert_eq!(r.client_send_queue.len(), 1, "commit reply");
        r.send_queue.clear();

        // Resend the exact same request: the duplicate is answered with the
        // stored reply without preparing a new op.
        r.on_message(&register, 0);

        let stored = r.client_sessions.get(7).expect("registered session").header;
        assert_eq!(r.client_send_queue.len(), 2, "commit reply + repeat reply");
        let repeat_header = r
            .client_send_queue
            .last()
            .expect("repeat reply")
            .header::<message_header::Reply>()
            .expect("reply header");
        assert_eq!(repeat_header.checksum, stored.checksum);
        assert_eq!(repeat_header.client, 7);
        assert_eq!(repeat_header.size, stored.size);
        assert_eq!(r.op, 1, "duplicate must not re-prepare");
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 0, "committed op was popped");
        assert_eq!(r.send_queue.len(), 0, "duplicate must not broadcast");
    }

    #[test]
    fn on_request_repeat_reply_register_reads_from_disk() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        let register_size = message_header::SIZE_U32 + message_header::REGISTER_RESULT_SIZE_U32;
        let register = request_message_full(7, 0, crate::Operation::REGISTER, 0, 0, register_size);
        r.on_message(&register, 0);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        r.send_queue.clear();
        r.client_send_queue.clear();

        // Drain the write: the reply is now durable, so a resend must read it
        // back from the client-replies zone.
        r.poll_client_replies();

        r.on_message(&register, 0);
        // The async read completes on the next poll and is routed to the client
        // (a `ReadReply` event with `destination_replica = None`).
        r.poll_client_replies();

        let stored = r.client_sessions.get(7).expect("registered session").header;
        assert_eq!(r.client_send_queue.len(), 1);
        let repeat_header = r
            .client_send_queue
            .last()
            .expect("repeat reply")
            .header::<message_header::Reply>()
            .expect("reply header");
        assert_eq!(repeat_header.checksum, stored.checksum);
        assert_eq!(repeat_header.client, 7);
        assert_eq!(r.op, 1, "duplicate must not re-prepare");
    }

    #[test]
    fn on_request_repeat_reply_serves_bodyless_reply() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Register client 7 (request 0), then commit a body-less noop
        // (request 1) driven through the client-facing path so the prepares
        // echo the real request checksums.
        let register_size = message_header::SIZE_U32 + message_header::REGISTER_RESULT_SIZE_U32;
        let register = request_message_full(7, 0, crate::Operation::REGISTER, 0, 0, register_size);
        r.on_message(&register, 0);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        r.send_queue.clear();

        // A new request must carry `parent` equal to the last reply's context.
        let session = r.client_sessions.get(7).expect("session");
        let (session_number, parent) = (session.session, session.header.context);
        let noop = request_message_full(
            7,
            1,
            crate::Operation::NOOP,
            session_number,
            parent,
            message_header::SIZE as u32,
        );
        r.on_message(&noop, 0);
        assert_eq!(r.op, 2);
        r.commit_max = 2;
        r.commit_prepare = Some(2);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        assert_eq!(r.client_send_queue.len(), 2, "register + noop commit replies");
        r.send_queue.clear();

        // Resend the body-less noop: the repeated header-only reply is rebuilt
        // directly from the session trailer.
        r.on_message(&noop, 0);

        let stored = r.client_sessions.get(7).expect("registered session").header;
        assert_eq!(stored.size, message_header::SIZE_U32);
        assert_eq!(r.client_send_queue.len(), 3, "commit replies + repeat reply");
        let repeat_header = r
            .client_send_queue
            .last()
            .expect("repeat reply")
            .header::<message_header::Reply>()
            .expect("reply header");
        assert_eq!(repeat_header.checksum, stored.checksum);
        assert_eq!(repeat_header.size, message_header::SIZE_U32);
        assert_eq!(r.op, 2, "duplicate must not re-prepare");
        assert_eq!(r.send_queue.len(), 0, "duplicate must not broadcast");
    }

    #[test]
    fn on_request_drops_stale_and_conflicting_requests() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Client 7 has committed register (request 0) and request 1.
        r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        r.primary_pipeline_prepare(7, 1, crate::Operation::NOOP, 0, 0).unwrap();
        r.commit_max = 2;
        r.commit_prepare = Some(2);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        assert_eq!(r.op, 2);
        r.client_send_queue.clear();
        r.send_queue.clear();

        let session = r.client_sessions.get(7).expect("session").session;

        // Older request (entry request 1 > 0): dropped.
        r.on_message(
            &request_message_full(
                7,
                0,
                crate::Operation::NOOP,
                session,
                0,
                message_header::SIZE as u32,
            ),
            0,
        );
        // Same request number but a different checksum (request collision):
        // dropped.
        r.on_message(
            &request_message_full(
                7,
                1,
                crate::Operation::NOOP,
                session,
                0,
                message_header::SIZE as u32,
            ),
            0,
        );
        // Request 2 with the wrong parent (the client did not ack the last
        // reply): dropped.
        r.on_message(
            &request_message_full(
                7,
                2,
                crate::Operation::NOOP,
                session,
                0,
                message_header::SIZE as u32,
            ),
            0,
        );
        // Request 3 skips request 2 (newer request): dropped.
        r.on_message(
            &request_message_full(
                7,
                3,
                crate::Operation::NOOP,
                session,
                0,
                message_header::SIZE as u32,
            ),
            0,
        );

        assert_eq!(r.op, 2, "no dropped request may prepare");
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 0);
        assert!(r.client_send_queue.is_empty(), "no reply or eviction for drops");
        assert_eq!(r.send_queue.len(), 0, "no broadcast");
    }

    #[test]
    fn on_request_evicts_missing_and_stale_sessions() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Client 7 is registered (session = commit 1).
        r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();

        // Client 8 has no session: evicted with `no_session`.
        r.on_message(&request_message(8, 1, crate::Operation::NOOP), 0);
        let evicted = r
            .client_send_queue
            .last()
            .expect("eviction")
            .header::<message_header::Eviction>()
            .expect("eviction header");
        assert_eq!(evicted.reason(), Some(message_header::Reason::NoSession));

        // Client 9's register is still in the pipeline; a follow-up request
        // waits for it to commit rather than being evicted.
        r.send_queue.clear();
        r.client_send_queue.clear();
        let op = r.primary_pipeline_prepare(9, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 2);
        r.on_message(&request_message(9, 1, crate::Operation::NOOP), 0);
        assert_eq!(r.op, 2, "waiting for the in-pipeline register: nothing prepared");
        assert!(r.client_send_queue.is_empty(), "no eviction while the register is pending");
    }

    #[test]
    fn on_request_evicts_oldest_registration_at_capacity() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Fill the client table (test_min: 7 slots) with clients 1..=7.
        for client in 1..=constants::CLIENTS_MAX {
            register_client(&mut r, u128::from(client));
        }
        assert_eq!(r.client_sessions.count(), constants::CLIENTS_MAX as usize);

        // The next registration evicts the oldest entry (commit 1): client 1.
        register_client(&mut r, 8);
        assert_eq!(r.client_sessions.count(), constants::CLIENTS_MAX as usize);
        assert_eq!(r.client_sessions.get(1), None, "the oldest registration is evicted");
        assert_eq!(r.client_sessions.get(8).expect("newest session").session, 8);

        // The evicted client is now unknown: its next request is evicted with
        // `no_session`.
        r.on_message(&request_message(1, 1, crate::Operation::NOOP), 0);
        let evicted = r
            .client_send_queue
            .last()
            .expect("eviction")
            .header::<message_header::Eviction>()
            .expect("eviction header");
        assert_eq!(evicted.reason(), Some(message_header::Reason::NoSession));
        assert_eq!(evicted.client, 1);
        assert_eq!(r.op, 8, "no request may prepare");
    }

    #[test]
    fn on_request_evicts_session_too_low_after_slot_reuse() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;

        // Client 1 registers first (session 1), then the table fills around it,
        // and client 8's registration evicts it.
        for client in 1..=constants::CLIENTS_MAX {
            register_client(&mut r, u128::from(client));
        }
        register_client(&mut r, 8);
        assert_eq!(r.client_sessions.get(1), None);

        // Client 1 re-registers: its new session (9) exceeds the old one.
        register_client(&mut r, 1);
        assert_eq!(r.client_sessions.get(1).expect("re-registered session").session, 9);

        // A stale request still carrying the client's pre-eviction session is
        // evicted with `session_too_low`: that session number now belongs to a
        // newer registration of the same client.
        r.on_message(
            &request_message_full(1, 1, crate::Operation::NOOP, 1, 0, message_header::SIZE as u32),
            0,
        );
        let evicted = r
            .client_send_queue
            .last()
            .expect("eviction")
            .header::<message_header::Eviction>()
            .expect("eviction header");
        assert_eq!(evicted.reason(), Some(message_header::Reason::SessionTooLow));
        assert_eq!(evicted.client, 1);
        assert_eq!(r.op, 9, "no request may prepare");
    }

    fn get_reply_message(
        from: u8,
        client: u128,
        checksum: u128,
        op: u64,
    ) -> crate::message::Message {
        let mut header = message_header::GetReply::default();
        header.cluster = CLUSTER;
        header.replica = from;
        header.size = message_header::SIZE_U32;
        header.reply_client = client;
        header.reply_checksum = checksum;
        header.reply_op = op;
        // `GetReply::invalid_header` requires a zero view/release, which the
        // defaults already satisfy.
        header.set_checksum_body(&[]);
        header.set_checksum();
        let mut message = crate::message::Message::new();
        message.set_header(&header);
        message
    }

    fn register_and_persist(r: &mut Replica, client: u128) -> u64 {
        let op = r.primary_pipeline_prepare(client, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        assert_eq!(op, 1);
        r.commit_max = 1;
        r.commit_prepare = Some(1);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        r.poll_client_replies();
        op
    }

    #[test]
    fn on_get_reply_serves_stored_reply() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        register_and_persist(&mut r, 7);
        r.send_queue.clear(); // Clear the primary's prepare-broadcast noise.
        let entry = r.client_sessions.get(7).expect("registered client");
        let expected_client = entry.header.client;
        let expected_checksum = entry.header.checksum;
        let expected_op = entry.header.op;

        // Replica 1 asks for client 7's committed reply.
        let get_reply = get_reply_message(1, expected_client, expected_checksum, expected_op);
        r.on_message(&get_reply, 0);
        // The reply is on disk (not a RAM write), so a reply-repair read is
        // issued and settles on the next poll.
        assert_eq!(r.send_queue.len(), 0);
        r.poll_client_replies();

        assert_eq!(r.send_queue.len(), 1);
        let reply = &r.send_queue[0];
        let header = reply.header::<message_header::Reply>().expect("a reply message");
        assert_eq!(header.client, expected_client);
        assert_eq!(header.checksum, expected_checksum);
        assert_eq!(header.op, expected_op);
        assert_eq!(header.replica, r.replica_u8());
        assert!(header.valid_checksum());
    }

    #[test]
    fn on_get_reply_serves_in_flight_ram_write() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        // Commit the register but leave the reply write in flight.
        let op = r.primary_pipeline_prepare(7, 0, crate::Operation::REGISTER, 0, 0).unwrap();
        r.commit_max = op;
        r.commit_prepare = Some(op);
        r.commit_stage = CommitStage::Execute;
        r.commit_execute();
        let slot = r.client_sessions.get_slot_for_client(7).expect("client 7 session");
        assert!(!r.client_replies.reply_durable(slot), "register write in flight");
        r.send_queue.clear(); // Clear the primary's prepare-broadcast noise.

        let entry = r.client_sessions.get(7).expect("registered client");
        let expected_checksum = entry.header.checksum;
        let expected_op = entry.header.op;
        let get_reply = get_reply_message(1, entry.header.client, expected_checksum, expected_op);
        r.on_message(&get_reply, 0);

        // Served from RAM: no read, no poll needed.
        assert_eq!(r.send_queue.len(), 1);
        let header = r.send_queue[0].header::<message_header::Reply>().expect("a reply message");
        assert_eq!(header.checksum, expected_checksum);
    }

    #[test]
    fn on_get_reply_ignores_unknown_client_and_bad_checksum() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        register_and_persist(&mut r, 7);
        r.send_queue.clear(); // Clear the primary's prepare-broadcast noise.

        // Unknown client.
        r.on_message(&get_reply_message(1, 99, 0xDEAD_BEEF, 1), 0);
        assert_eq!(r.send_queue.len(), 0, "unknown client: no reply");

        // Known client, wrong reply checksum.
        let entry = r.client_sessions.get(7).expect("registered client");
        let expected_op = entry.header.op;
        r.on_message(&get_reply_message(1, 7, 0xDEAD_BEEF, expected_op), 0);
        assert_eq!(r.send_queue.len(), 0, "checksum mismatch: no reply");
        r.poll_client_replies();
        assert_eq!(r.send_queue.len(), 0);
    }

    #[test]
    fn on_reply_repairs_faulty_reply() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        register_and_persist(&mut r, 7);
        let slot = r.client_sessions.get_slot_for_client(7).expect("client 7 session");
        assert!(r.client_replies.reply_durable(slot));
        r.send_queue.clear(); // Clear the primary's prepare-broadcast noise.

        // A clean slot is not rewritten (upstream replica.zig:2377).
        let entry = r.client_sessions.get(7).expect("registered client");
        let reply_header = entry.header;
        let mut reply = crate::message::Message::new();
        reply.set_header(&reply_header);
        r.on_message(&reply, 0);
        assert!(r.client_replies.reply_durable(slot), "clean slot untouched");
        let _repaired_storage = r.grid_storage.take().expect("mounted storage");

        // Simulate a corrupt on-disk reply and repair it from replica 1.
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));
        r.client_replies.mark_faulty(slot);
        assert!(r.client_replies.reply_is_faulty(slot));

        let mut reply = crate::message::Message::new();
        reply.set_header(&reply_header);
        r.on_message(&reply, 0);
        assert!(!r.client_replies.reply_durable(slot), "repair write in flight");
        r.poll_client_replies();
        assert!(r.client_replies.reply_durable(slot), "reply repaired");
        assert!(!r.client_replies.reply_is_faulty(slot));
    }

    #[test]
    fn on_get_reply_ignores_misdirected_or_wrong_status() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.grid_storage = Some(MemoryStorage::new(
            Zone::ClientReplies.start() + 4 * crate::message::MESSAGE_SIZE_MAX as u64,
        ));

        register_and_persist(&mut r, 7);
        r.send_queue.clear(); // Clear the primary's prepare-broadcast noise.
        let entry = r.client_sessions.get(7).expect("registered client");
        let expected_client = entry.header.client;
        let expected_checksum = entry.header.checksum;
        let expected_op = entry.header.op;

        // A GetReply addressed to self is dropped.
        r.on_message(
            &get_reply_message(r.replica_u8(), expected_client, expected_checksum, expected_op),
            0,
        );
        assert_eq!(r.send_queue.len(), 0, "misdirected: no reply");
        r.poll_client_replies();
        assert_eq!(r.send_queue.len(), 0, "no reads were queued");

        // A GetReply arriving while not in normal/view-change status is dropped.
        r.status = Status::Recovering;
        r.on_message(&get_reply_message(1, expected_client, expected_checksum, expected_op), 0);
        assert_eq!(r.send_queue.len(), 0, "recovering: no reply");
    }

    #[test]
    fn on_request_evicts_unsupported_release() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        assert!(r.is_primary());

        let mut request = request_message(1, 1, crate::Operation::NOOP);
        let mut header = request.header::<message_header::Request>().expect("request");
        header.release = crate::multiversion::Release { value: 2 };
        header.set_checksum();
        request.set_header(&header);
        r.on_message(&request, 0);

        assert_eq!(r.pipeline_queue.prepare_queue.len(), 0);
        assert_eq!(r.client_send_queue.len(), 1);
        let eviction =
            r.client_send_queue[0].header::<message_header::Eviction>().expect("eviction");
        assert_eq!(eviction.reason(), Some(message_header::Reason::ClientReleaseTooHigh));
    }

    #[test]
    fn on_request_evicts_register_without_body() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        assert!(r.is_primary());

        let mut header = message_header::Request::default();
        header.cluster = CLUSTER;
        header.client = 5;
        header.session = 0;
        header.request = 0;
        header.operation = crate::Operation::REGISTER;
        header.release = crate::multiversion::Release::MINIMUM;
        header.size = message_header::SIZE as u32;
        header.set_checksum_body(&[]);
        header.set_checksum();
        let mut message = Message::new();
        message.set_header(&header);
        r.on_message(&message, 0);

        assert_eq!(r.pipeline_queue.prepare_queue.len(), 0);
        assert_eq!(r.client_send_queue.len(), 1);
        let eviction =
            r.client_send_queue[0].header::<message_header::Eviction>().expect("eviction");
        assert_eq!(eviction.reason(), Some(message_header::Reason::InvalidRequestBodySize));
    }

    #[test]
    fn on_request_drops_future_view() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        assert!(r.is_primary());

        let mut request = request_message(1, 1, crate::Operation::NOOP);
        let mut header = request.header::<message_header::Request>().expect("request");
        header.view = 1; // Newer than r.view == 0.
        header.set_checksum();
        request.set_header(&header);
        r.on_message(&request, 0);

        assert_eq!(r.pipeline_queue.prepare_queue.len(), 0);
        assert_eq!(r.pipeline_queue.request_queue.len(), 0);
        assert_eq!(r.client_send_queue.len(), 0, "dropped, not evicted");
        assert_eq!(r.send_queue.len(), 0);
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
        let op = r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
        assert_eq!(op, 1);
        assert_eq!(r.op, 1);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 1);
        assert!(r.prepare_timestamp > 0);
    }

    #[test]
    fn primary_pipeline_prepare_rejects_backup() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let result = r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0);
        assert_eq!(result, Err(PrepareReject::NotPrimary));
    }

    #[test]
    fn primary_pipeline_prepare_chain() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let op1 = r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
        let op2 = r.primary_pipeline_prepare(2, 101, crate::Operation(6), 64, 0).unwrap();
        assert_eq!(op1, 1);
        assert_eq!(op2, 2);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 2);
        // Timestamps must be strictly increasing.
        assert!(r.prepare_timestamp > 0);
    }

    #[test]
    fn primary_pipeline_prepare_broadcasts_to_backups() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
        // One broadcast Prepare per backup, carrying the journaled op.
        assert_eq!(r.send_queue.len(), 2);
        for message in &r.send_queue {
            let prepare = message.header::<message_header::Prepare>().unwrap();
            assert_eq!(prepare.op, 1);
            assert_eq!(prepare.checksum(), r.journal.header_with_op(1).unwrap().checksum());
            assert_eq!(u16::from(prepare.replica), 0);
        }
    }

    #[test]
    fn on_prepare_ok_quorum_3() {
        let mut r = Replica::new(0, 0, 3); // primary, quorum=2
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
        let checksum = r.journal.header_with_op(1).unwrap().checksum();

        // The primary contributed its own prepare_ok on prepare.
        assert!(r.pipeline_queue.prepare_queue[0].acks_received >= 1);

        // Replica 1 acks → quorum reached (primary + one backup).
        let result = r.on_prepare_ok(1, checksum, 1);
        assert!(matches!(result, PrepareOkResult::QuorumReached { op: 1, .. }));

        // A repeated ack is not re-counted.
        let result = r.on_prepare_ok(1, checksum, 1);
        assert_eq!(result, PrepareOkResult::DuplicateAck);
    }

    #[test]
    fn on_prepare_ok_duplicate_ignored() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
        let checksum = r.journal.header_with_op(1).unwrap().checksum();

        let _ = r.on_prepare_ok(1, checksum, 1);
        let result = r.on_prepare_ok(1, checksum, 1); // duplicate
        assert_eq!(result, PrepareOkResult::DuplicateAck);
    }

    #[test]
    fn primary_pipeline_pending_tracks_unquorumed_head() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
        r.primary_pipeline_prepare(2, 2, crate::Operation::NOOP, 0, 0).unwrap();

        // Both are pending (self-ack alone is not a quorum).
        let (slot, pending) = r.primary_pipeline_pending().unwrap();
        assert_eq!(slot, 0);
        assert_eq!(pending.op, 1);
        assert!(!pending.ok_quorum_received);

        // Quorum op 1 (self + replica 1): the new head is op 2.
        let checksum = r.journal.header_with_op(1).unwrap().checksum();
        r.on_prepare_ok(1, checksum, 1);
        let (slot, pending) = r.primary_pipeline_pending().unwrap();
        assert_eq!(slot, 1);
        assert_eq!(pending.op, 2);

        // Quorum op 2 too: nothing left pending.
        let checksum = r.journal.header_with_op(2).unwrap().checksum();
        r.on_prepare_ok(2, checksum, 1);
        assert!(r.primary_pipeline_pending().is_none());
    }

    #[test]
    fn on_prepare_timeout_retransmits_to_unacked_backups() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let op = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
        r.send_queue.clear(); // the initial broadcast is not under test

        // No backup has acked → both replicas 1 and 2 are waiting; the
        // timeout re-sends the pending prepare to each.
        r.on_prepare_timeout();
        assert_eq!(r.send_queue.len(), 2);
        let checksum = r.journal.header_with_op(op).unwrap().checksum();
        for message in &r.send_queue {
            let prepare = message.header::<message_header::Prepare>().unwrap();
            assert_eq!(prepare.op, op);
            assert_eq!(prepare.checksum(), checksum);
        }
        assert!(r.prepare_timeout.active); // reset, still ticking
        r.send_queue.clear();

        // Once a backup acks, retransmission stops.
        r.on_prepare_ok(op, checksum, 1);
        r.commit_dispatch_enter();
        r.on_prepare_timeout();
        assert!(r.send_queue.is_empty());
        assert!(!r.prepare_timeout.active); // pending none → stopped
    }

    #[test]
    fn prepare_timeout_retransmits_pending_head_only() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
        r.primary_pipeline_prepare(2, 2, crate::Operation::NOOP, 0, 0).unwrap();
        r.send_queue.clear(); // the initial broadcasts are not under test

        // Quorum op 1 and commit it; op 2 remains pending.
        let checksum = r.journal.header_with_op(1).unwrap().checksum();
        r.on_prepare_ok(1, checksum, 1);
        r.commit_dispatch_enter();
        assert_eq!(r.commit_min, 1);
        assert_eq!(r.pipeline_queue.prepare_queue.len(), 1);

        // The timeout re-sends op 2 (still unacked by both backups), not op 1.
        r.on_prepare_timeout();
        assert_eq!(r.send_queue.len(), 2);
        for message in &r.send_queue {
            let prepare = message.header::<message_header::Prepare>().unwrap();
            assert_eq!(prepare.op, 2);
        }
    }

    #[test]
    fn backup_acks_prepare_on_the_wire_and_primary_commits() {
        // The primary prepares op 1 through its pipeline and broadcasts it to
        // both backups; the backup receives it on the wire, accepts it, and
        // acks; the primary counts the ack toward quorum and commits.
        let mut r1 = Replica::new(0, 0, 3); // primary, view 0
        r1.status = Status::Normal;
        r1.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
        assert_eq!(r1.pipeline_queue.prepare_queue.len(), 1);
        assert_eq!(r1.send_queue.len(), 2); // one broadcast Prepare per backup

        let mut r2 = Replica::new(0, 2, 3); // backup
        r2.status = Status::Normal;
        let prepare = r1.send_queue[0].header::<message_header::Prepare>().unwrap();
        assert_eq!(prepare.command, Command::Prepare);
        let prepare_msg = r1.send_queue[0].clone();
        r2.on_message(&prepare_msg, 20_000);
        assert_eq!(r2.op, 1);
        assert_eq!(r2.send_queue.len(), 1); // exactly one PrepareOk
        let ok = r2.send_queue[0].header::<message_header::PrepareOk>().unwrap();
        assert_eq!(ok.op, 1);
        assert_eq!(u16::from(ok.replica), 2);
        assert_eq!(ok.prepare_checksum, prepare.checksum());
        assert_eq!(ok.view, 0);
        assert_eq!(ok.cluster, 0);

        // The primary processes the ack → quorum → commits op 1.
        r1.on_message(&r2.send_queue[0], 20_001);
        assert_eq!(r1.commit_min, 1);
        assert_eq!(r1.commit_max, 1);
        assert!(r1.pipeline_queue.prepare_queue.is_empty());
        assert_eq!(r2.commit_min, 0); // the backup has not committed yet
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
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
        r.primary_pipeline_prepare(2, 101, crate::Operation(6), 64, 0).unwrap();

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
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
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
        let _ = r.primary_pipeline_prepare(1, 100, crate::Operation::RESERVED, 64, 0);
    }

    #[test]
    fn backup_on_prepare_journals_chain() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;

        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        assert_eq!(r.on_prepare(&h1), OnPrepareResult::Accepted);
        assert_eq!(r.op, 1);

        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Accepted);
        assert_eq!(r.op, 2);

        // Hash chain is recorded in the journal:
        assert_eq!(r.journal.header_with_op(2).unwrap().parent, h1.checksum());
        assert_eq!(r.journal.op_maximum(), 2);

        // Duplicate and future ops:
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Stale);
        let h5 = make_prepare_for_replica(0, 0, 5, h2.checksum(), 0);
        assert_eq!(r.on_prepare(&h5), OnPrepareResult::FutureOp);
        assert_eq!(r.op, 2); // unchanged
    }

    #[test]
    #[should_panic = "hash chain break in view"]
    fn backup_on_prepare_panics_on_hash_chain_break() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);
        let h2 = make_prepare_for_replica(0, 0, 2, 0xDEAD, 0); // wrong parent
        r.on_prepare(&h2);
    }

    #[test]
    fn backup_commits_from_journal() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;

        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        assert_eq!(r.on_prepare(&h1), OnPrepareResult::Accepted);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Accepted);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        assert_eq!(r.on_prepare(&h3), OnPrepareResult::Accepted);
        assert_eq!(r.op, 3);

        // Backup learns commit_max=1 and commits op 1 from its journal:
        r.commit_max = 1;
        r.commit_dispatch_enter();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert_eq!(r.commit_min, 1);
        // The prepare write completes with the commit:
        assert!(r.journal.has_prepare(&h1));

        // Then it catches up op by op:
        r.commit_max = 2;
        r.commit_dispatch_enter();
        assert_eq!(r.commit_min, 2);
        assert!(r.journal.has_prepare(&h2));

        r.commit_max = 3;
        r.commit_dispatch_enter();
        assert_eq!(r.commit_min, 3);
        assert!(r.journal.has_prepare(&h3));
    }

    #[test]
    fn backup_does_not_commit_unacquired_op() {
        // commit_max advances to 2, but only op 1 is in the journal:
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);

        r.commit_max = 2;
        r.commit_dispatch_enter();
        // Op 1 commits; op 2 has no header in the journal, so the pipeline stops.
        assert_eq!(r.commit_min, 1);
        assert_eq!(r.commit_stage, CommitStage::Idle);
    }

    #[test]
    fn on_commit_drives_backup_journal_commit() {
        let mut r = Replica::new(0, 1, 3); // backup, primary is replica 0
        r.status = Status::Normal;

        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        assert_eq!(r.on_prepare(&h1), OnPrepareResult::Accepted);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Accepted);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        assert_eq!(r.on_prepare(&h3), OnPrepareResult::Accepted);

        // Primary commits ops 1..3 and broadcasts a Commit message:
        let commit = make_commit_message(0, 0, 3, h3.checksum(), 100);
        r.on_message(&commit, 1_000);

        assert_eq!(r.heartbeat_timestamp, 100);
        assert_eq!(r.commit_min, 3);
        assert_eq!(r.commit_max, 3);
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert!(r.journal.has_prepare(&h3));
    }
    #[test]
    fn on_repair_resends_prepare_ok_for_clean_duplicate() {
        // Backup r1 accepted h1 and committed it (slot clean). The primary
        // re-sends h1 (its copy of our ack may be lost): on_repair republishes
        // the prepare_ok instead of dropping the duplicate.
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        deliver_prepare(&mut r, &h1); // Accepted -> acks h1
        assert_eq!(r.send_queue.len(), 1);

        r.commit_max = 1;
        r.commit_journal();
        assert_eq!(r.commit_min, 1);
        assert!(r.journal.has_prepare(&h1));

        // The identical re-send goes through the Stale -> on_repair path:
        deliver_prepare(&mut r, &h1);
        assert_eq!(r.send_queue.len(), 2);
        let ack = r.send_queue.pop().unwrap().header::<message_header::PrepareOk>().unwrap();
        assert_eq!(ack.prepare_checksum, h1.checksum);
        assert_eq!(ack.op, 1);
        assert_eq!(ack.replica, 1);
    }

    #[test]
    fn on_repair_refreshes_dirty_copy_and_commits() {
        // A dirty copy is not a duplicate: on_repair re-stages the header and
        // (as a backup) attempts the commit pass, which finishes the op here.
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        deliver_prepare(&mut r, &h1); // Accepted -> acks h1
        assert_eq!(r.send_queue.len(), 1);

        // The primary committed op 1 but its Commit broadcast has not arrived
        // yet. It re-sends h1; our slot is still dirty (write pending):
        r.commit_max = 1;
        deliver_prepare(&mut r, &h1); // Stale -> on_repair -> stale copy
        assert_eq!(r.send_queue.len(), 1); // no re-ack for a dirty copy
        assert_eq!(r.commit_min, 1); // commit_journal finished the op
        assert!(r.journal.has_prepare(&h1)); // now clean

        // With the slot clean, a third re-send republishes the ack:
        deliver_prepare(&mut r, &h1);
        assert_eq!(r.send_queue.len(), 2);
    }

    #[test]
    fn on_repair_ignores_newer_view_and_future_ops() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        assert_eq!(r.on_prepare(&h1), OnPrepareResult::Accepted);
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Accepted);
        assert_eq!(r.op, 2);

        // A newer view is refused (we have not joined it):
        let newer_view = make_prepare_for_replica(0, 1, 1, 0, 0);
        r.on_repair(&newer_view);
        assert!(r.send_queue.is_empty());
        assert_eq!(r.journal.header_with_op(1).unwrap().checksum(), h1.checksum());

        // An op beyond the head may not be repaired in:
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        r.on_repair(&h3);
        assert!(r.send_queue.is_empty());
        assert!(r.journal.header_with_op(3).is_none());
    }

    #[test]
    fn future_prepare_jumps_head_then_gap_repairs_in() {
        // Primary is at op 3; backup r1 misses op 2. The op-3 prepare arrives
        // first: the head jumps to op 2, the re-sent op 3 is accepted across
        // the empty slot, and op 2 then repairs in below the head.
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        deliver_prepare(&mut r, &h1);
        assert_eq!(r.op, 1);

        deliver_prepare(&mut r, &h3); // FutureOp -> jump to op 2
        assert_eq!(r.op, 2);

        // The (re-sent) op 3 is now exactly one ahead: it is accepted across
        // the gap, since the parent-link check is skipped for a slot that was
        // never written (upstream replica.zig:2211-2215).
        deliver_prepare(&mut r, &h3);
        assert_eq!(r.op, 3);
        assert_eq!(r.journal.header_with_op(3).unwrap().checksum(), h3.checksum());

        // The missing op 2 repairs in below the head (as a GetPrepare response:
        // a Prepare with op < self.op goes through on_repair):
        deliver_prepare(&mut r, &h2);
        assert_eq!(r.op, 3); // repairs never advance the head
        assert_eq!(r.journal.header_with_op(2).unwrap().checksum(), h2.checksum());
    }

    #[test]
    fn repair_requests_headers_across_a_journal_break() {
        // Backup r1 committed ops 1..=5; Commit messages for 6..=11 were
        // processed, but those prepares were then lost from the journal. The
        // head jumps past them to op 15. With pipeline=4 the repair window is
        // [1..=11]: the 6..=14 break yields a GetHeaders request for 6..=11.
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        // repair() is gated on the journal-repair timeout being active (the
        // upstream `journal_repair_timeout` gate); arm it to run passes.
        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        let mut parent = 0;
        for op in 1..=5 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            r.on_prepare(&h);
        }
        for op in 1..=5 {
            r.commit_op(op);
        }
        // Commit messages advanced the frontier past the (still missing)
        // prepares 6..=11:
        r.commit_min = 11;
        r.commit_max = 11;

        let h15 = make_prepare_for_replica(0, 0, 15, parent, 0);
        deliver_prepare(&mut r, &h15); // FutureOp -> jump to op 14
        assert_eq!(r.op, 14);
        deliver_prepare(&mut r, &h15); // op 15 == op+1: accepted across the gap
        assert_eq!(r.op, 15);

        // The accept already ran a repair pass; drain and drive a fresh one to
        // assert on its output in isolation.
        r.send_queue.clear();
        r.repair();
        assert_eq!(r.send_queue.len(), 1);
        let get_headers =
            r.send_queue.pop().unwrap().header::<message_header::GetHeaders>().unwrap();
        assert_eq!(get_headers.replica, 1);
        assert_eq!(get_headers.view, 0);
        assert_eq!(get_headers.op_min, 6);
        assert_eq!(get_headers.op_max, 11);
    }

    #[test]
    fn repair_heals_gap_through_headers_then_prepare() {
        // The full loop: r1 detects the break, GetHeaders from r0 installs the
        // missing headers, and the follow-up repair pass turns each dirty
        // recovered header into a GetPrepare for its body.
        let mut primary = Replica::new(0, 0, 3);
        primary.status = Status::Normal;
        let mut parent = 0;
        let mut h_by_op = std::collections::HashMap::new();
        for op in 1..=15 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            primary.on_prepare(&h);
            h_by_op.insert(op, h);
        }

        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        // Arm the repair timeout so the internal repair passes after accepts
        // and after Headers run (see the gap test above).
        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        let mut parent = 0;
        for op in 1..=5 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            r.on_prepare(&h);
        }
        for op in 1..=5 {
            r.commit_op(op);
        }
        // Committed 6..=11, then lost (see the gap test above):
        r.commit_min = 11;
        r.commit_max = 11;
        let h15 = h_by_op[&15];
        deliver_prepare(&mut r, &h15); // jump to 14
        deliver_prepare(&mut r, &h15); // accept across the gap
        assert_eq!(r.op, 15);
        r.send_queue.clear();

        // Pass 1: GetHeaders(6..=11).
        r.repair();
        let get_headers =
            r.send_queue.pop().unwrap().header::<message_header::GetHeaders>().unwrap();
        assert_eq!(get_headers.op_min, 6);
        assert_eq!(get_headers.op_max, 11);

        // Primary serves headers 6..=11 (newest first); backup installs them.
        let mut request = crate::message::Message::new();
        request.set_header(&get_headers);
        primary.on_message(&request, 100);
        let response = primary.send_queue.pop().unwrap();
        r.on_message(&response, 100);

        assert_eq!(r.send_queue.len(), 6); // one GetPrepare per recovered op
        let mut repair_ops: Vec<u64> = r
            .send_queue
            .iter()
            .map(|m| m.header::<message_header::GetPrepare>().unwrap().prepare_op)
            .collect();
        repair_ops.sort_unstable();
        assert_eq!(repair_ops, [6, 7, 8, 9, 10, 11]);
        for m in &r.send_queue {
            let request = m.header::<message_header::GetPrepare>().unwrap();
            assert_eq!(request.replica, 1);
            let expected = &h_by_op[&request.prepare_op];
            assert_eq!(request.prepare_checksum, expected.checksum);
        }

        // Pass 2: serve the op-6 body; the backup's journal now holds the
        // repaired header+prepare for op 6.
        let index = r
            .send_queue
            .iter()
            .position(|m| m.header::<message_header::GetPrepare>().unwrap().prepare_op == 6)
            .unwrap();
        let get_prepare =
            r.send_queue.remove(index).header::<message_header::GetPrepare>().unwrap();
        assert_eq!(get_prepare.prepare_op, 6);
        let mut request = crate::message::Message::new();
        request.set_header(&get_prepare);
        primary.on_message(&request, 100);
        let prepare = primary.send_queue.pop().unwrap();
        assert_eq!(prepare.header::<message_header::Prepare>().unwrap().op, 6);
        r.on_message(&prepare, 100);
        assert_eq!(r.journal.header_with_op(6).unwrap().checksum(), h_by_op[&6].checksum);
    }

    #[test]
    fn repair_is_quiet_when_journal_intact() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let mut parent = 0;
        for op in 1..=15 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            r.on_prepare(&h);
        }
        for op in 1..=11 {
            r.commit_op(op);
        }
        assert_eq!(r.commit_min, 11);

        // Arm the repair timeout (repair() is gated on it being active).
        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        r.repair(); // contiguous and clean where it matters: nothing asked for
        assert!(r.send_queue.is_empty());
    }

    #[test]
    fn repair_cleans_dirty_slots_out_of_bounds() {
        // A crash-recovered replica may hold a dirty WAL entry that is out of
        // bounds of the `[op_repair_min, op]` window (op <= op_checkpoint, or an
        // op beyond self.op that was truncated). Such entries cannot be
        // repaired — `repair()` must drop them, while leaving in-bounds dirty
        // slots alone.
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let mut parent = 0;
        for op in 1..=15 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            r.on_prepare(&h);
        }
        for op in 1..=11 {
            r.commit_op(op);
        }

        // In-bounds dirty slot: repaired via GetPrepare.
        let in_bounds = crate::journal::Slot::for_op(6);
        r.journal.dirty.set(in_bounds);
        // Out-of-bounds dirty slot: a stray corrupt entry far from the window
        // (op 25 > self.op; slot 25 in the 32-slot test journal).
        let out_of_bounds = crate::journal::Slot::for_op(25);
        r.journal.dirty.set(out_of_bounds);
        assert!(r.journal.dirty.bit(in_bounds));
        assert!(r.journal.dirty.bit(out_of_bounds));

        // Arm the repair timeout (repair() is gated on it being active).
        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        r.repair();

        assert!(!r.journal.dirty.bit(out_of_bounds));
        assert!(r.journal.dirty.bit(in_bounds)); // still waiting for a GetPrepare
        let get_prepares: Vec<u64> = r
            .send_queue
            .iter()
            .map(|m| m.header::<message_header::GetPrepare>().unwrap().prepare_op)
            .collect();
        assert_eq!(get_prepares, [6]);
    }

    #[test]
    fn repair_requests_get_view_from_primary_when_op_behind() {
        // A backup whose head trails its commit frontier (`op < op_repair_max`)
        // asks the primary for a fresh View, so it can advance the head even in
        // an idle cluster (upstream replica.zig:7598).
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        prepare_and_commit_suffix(&mut r, 0, 8, 8);
        // Commit messages advanced the frontier ahead of the head:
        r.commit_max = 16;
        assert_eq!(r.op, 8);
        assert!(r.op < r.op_repair_max());

        // Arm the repair timeout (repair() is gated on it being active).
        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        r.repair();
        assert_eq!(r.send_queue.len(), 1);
        let get_view = r.send_queue.pop().unwrap().header::<message_header::GetView>().unwrap();
        assert_eq!(get_view.replica, 1);
        assert_eq!(get_view.view, 0);
        assert_eq!(get_view.nonce, r.nonce);
    }

    #[test]
    fn get_view_advances_head_and_triggers_gap_repair() {
        // Full loop: a backup behind the primary's head requests a View, the
        // primary replies with its op/commit_max (echoing the nonce), the backup
        // jumps its head (bounded by op_prepare_max_sync), and the repair pass
        // then targets the gap below.
        let mut primary = Replica::new(0, 0, 3);
        primary.status = Status::Normal;
        let mut parent = 0;
        let mut h_by_op = std::collections::HashMap::new();
        for op in 1..=18 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            primary.on_prepare(&h);
            h_by_op.insert(op, h);
        }
        for op in 1..=15 {
            primary.commit_op(op);
        }
        assert_eq!(primary.op, 18);
        assert_eq!(primary.commit_max, 15);

        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        // Arm the repair timeout before the View lands: the tail of `on_view`
        // re-runs the (gated) repair pass to fill the gap below the jumped
        // head.
        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        prepare_and_commit_suffix(&mut r, 0, 6, 6);
        r.commit_max = 15; // knows of commits beyond its head
        assert_eq!(r.op, 6);

        // Step 1: backup requests the view.
        r.repair();
        let get_view = r.send_queue.pop().unwrap();
        assert_eq!(get_view.header::<message_header::GetView>().unwrap().nonce, r.nonce);

        // Step 2: the primary replies with a View echoing our nonce.
        primary.on_message(&get_view, 100);
        let view = primary.send_queue.pop().unwrap();
        assert_eq!(primary.send_queue.len(), 0);
        let view_header = view.header::<message_header::View>().unwrap();
        assert_eq!(view_header.view, 0);
        assert_eq!(view_header.replica, 0);
        assert_eq!(view_header.nonce, r.nonce);
        assert_eq!(view_header.op, 18);
        assert_eq!(view_header.commit_max, 15);

        // Step 3: the backup jumps its head and stages the new log suffix.
        r.on_message(&view, 100);
        assert_eq!(r.op, 18);
        assert_eq!(r.commit_max, 15);
        assert_eq!(r.status, Status::Normal);
        // The staged suffix (descending from the new head) plus the checkpoint
        // hook at op 0 (with no real checkpoints, saturating to the root).
        assert_eq!(r.view_headers.first().unwrap().op, 18);
        assert_eq!(r.view_headers[constants::VIEW_CHANGE_HEADERS_SUFFIX_MAX as usize - 1].op, 14);
        assert!(r.view_headers.len() <= constants::VIEW_HEADERS_MAX as usize);
        // The jumped-op suffix headers replaced the journal's (absent) slots:
        for op in 14..=18 {
            let slot = crate::journal::Journal::slot_for_op(op);
            assert!(r.journal.header_with_op(op).is_some());
            assert!(r.journal.dirty.bit(slot));
            assert_eq!(r.journal.header_with_op(op).unwrap().checksum(), h_by_op[&op].checksum());
        }

        // Step 4: the repair pass (run at the end of on_view) targets the hole
        // the jump left below it, and the committed-but-only-headered op 14.
        assert_eq!(r.send_queue.len(), 2);
        let get_headers = r.send_queue.remove(0).header::<message_header::GetHeaders>().unwrap();
        assert_eq!(get_headers.replica, 1);
        assert_eq!(get_headers.op_min, 7);
        assert_eq!(get_headers.op_max, 13);
        let get_prepare =
            r.send_queue.pop().unwrap().header::<message_header::GetPrepare>().unwrap();
        assert_eq!(get_prepare.prepare_op, 14);
    }

    #[test]
    fn repair_is_gated_until_the_journal_repair_timeout_is_active() {
        // `repair()` is only reached when the repair timeout is ticking
        // (upstream `replica.zig:7545`); without it the pass emits nothing.
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        prepare_and_commit_suffix(&mut r, 0, 5, 5);
        r.commit_max = 16; // knows of commits beyond its head
        assert_eq!(r.op, 5);
        assert!(r.op < r.op_repair_max());

        // Gate closed: no timeout armed.
        r.repair();
        assert!(r.send_queue.is_empty());
        assert!(!r.journal_repair_timeout.active);

        // Gate open: arming the timeout lets the pass run.
        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        r.repair();
        assert_eq!(r.send_queue.len(), 1);
        let get_view = r.send_queue.pop().unwrap().header::<message_header::GetView>().unwrap();
        assert_eq!(get_view.nonce, r.nonce);
    }

    #[test]
    fn journal_repair_timeout_drives_repair_from_tick() {
        // The tick loop fires the repair timeout, which re-arms itself and
        // runs a repair pass — no manual `repair()` call needed.
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        prepare_and_commit_suffix(&mut r, 0, 8, 8);
        r.commit_max = 12; // knows of commits beyond its head

        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        for _ in 0..constants::JOURNAL_REPAIR_TIMEOUT - 1 {
            r.tick(100);
        }
        assert!(r.journal_repair_timeout.active); // not fired yet
        assert_eq!(r.send_queue.len(), 0);

        r.tick(100); // the timeout fires: re-arms and repairs
        assert!(r.journal_repair_timeout.active); // re-armed
        assert_eq!(r.journal_repair_timeout.attempts, 1);
        assert_eq!(r.send_queue.len(), 1);
        let get_view = r.send_queue.pop().unwrap().header::<message_header::GetView>().unwrap();
        assert_eq!(get_view.replica, 1);
        assert_eq!(get_view.nonce, r.nonce);
    }

    #[test]
    fn every_fiftieth_repair_pass_is_unconditional() {
        // In normal status the repair window is [commit_min+1, op - pipeline];
        // a gap outside that window goes unrequested until the 50th timeout
        // fire, which repairs unconditionally (upstream `replica.zig:7625`).
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let mut parent = 0;
        for op in 1..=5 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            r.on_prepare(&h);
        }
        for op in 1..=5 {
            r.commit_op(op);
        }
        // Committed 6..=7, then lost; the head jumps across the hole to op 8
        // (see the committed-loss pattern above).
        let h8 = make_prepare_for_replica(0, 0, 8, parent, 0);
        deliver_prepare(&mut r, &h8); // jump to op 7
        deliver_prepare(&mut r, &h8); // accept at op 8
        assert_eq!(r.op, 8);
        assert_eq!(r.commit_min, 5);
        r.send_queue.clear();

        r.journal_repair_timeout = Timeout::start(constants::JOURNAL_REPAIR_TIMEOUT);
        // With pipeline=4 the bounded window is `op - 4 = 4`, so the 6..=7
        // break's range is not requested, and the dirty head op 8 is out of
        // `[commit_min+1, repair_op_max]`:
        for _fire in 1..=49 {
            for _ in 0..constants::JOURNAL_REPAIR_TIMEOUT {
                r.tick(100);
            }
            assert_eq!(r.send_queue.len(), 0);
        }
        assert_eq!(r.journal_repair_timeout.attempts, 49);

        // The 50th fire repairs unconditionally: GetHeaders for the 6..=7 gap
        // and a GetPrepare for the dirty head op 8.
        for _ in 0..constants::JOURNAL_REPAIR_TIMEOUT {
            r.tick(100);
        }
        assert_eq!(r.journal_repair_timeout.attempts, 50);
        assert_eq!(r.send_queue.len(), 2);
        let get_headers = r.send_queue.remove(0).header::<message_header::GetHeaders>().unwrap();
        assert_eq!(get_headers.op_min, 6);
        assert_eq!(get_headers.op_max, 7);
        let get_prepare =
            r.send_queue.pop().unwrap().header::<message_header::GetPrepare>().unwrap();
        assert_eq!(get_prepare.prepare_op, 8);
    }

    #[test]
    fn on_commit_stale_timestamp_skips_heartbeat() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);

        r.on_commit_heartbeat(0, 50, 1, 100); // sets heartbeat_timestamp = 50

        // An older Commit (timestamp 40) must not move the heartbeat backward,
        // and commit_max still advances, so op 1 commits:
        let commit = make_commit_message(0, 0, 1, h1.checksum(), 40);
        r.on_message(&commit, 200);
        assert_eq!(r.heartbeat_timestamp, 50);
        assert_eq!(r.commit_min, 1);
    }

    #[test]
    fn on_commit_ignored_on_primary() {
        let mut r = Replica::new(0, 0, 3); // primary
        r.status = Status::Normal;
        let commit = make_commit_message(0, 0, 1, 0, 100);
        r.on_message(&commit, 200);
        assert_eq!(r.commit_max, 0);
        assert_eq!(r.commit_min, 0);
    }

    #[test]
    fn primary_send_commit_broadcasts_real_commit_header() {
        let mut r = Replica::new(0, 0, 3); // primary
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);
        r.commit_op(1); // commit_min == commit_max == 1

        r.send_commit(1_000);
        assert_eq!(r.send_queue.len(), 2); // one Commit per backup
        for message in &r.send_queue {
            let commit = message.header::<message_header::Commit>().unwrap();
            assert_eq!(commit.commit, 1);
            assert_eq!(commit.commit_checksum, h1.checksum());
            assert_eq!(commit.timestamp_monotonic, 1_000);
            assert_eq!(commit.view, 0);
            assert_eq!(commit.replica, 0);
        }
    }

    #[test]
    fn primary_commit_reaches_backup_through_send_queue() {
        let mut primary = Replica::new(0, 0, 3);
        primary.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        primary.on_prepare(&h1);
        primary.commit_op(1);

        // Primary broadcasts a Commit; the backup receives it through the
        // sans-IO send_queue.
        primary.send_commit(1_000);
        let commit = primary.send_queue.pop().unwrap();

        let mut backup = Replica::new(0, 1, 3);
        backup.status = Status::Normal;
        assert_eq!(backup.on_prepare(&h1), OnPrepareResult::Accepted);
        backup.on_message(&commit, 1_100);
        assert_eq!(backup.commit_min, 1);
        assert_eq!(backup.commit_max, 1);
        assert!(backup.journal.has_prepare(&h1));
    }

    #[test]
    fn on_get_prepare_serves_prepare_matching_explicit_checksum() {
        let mut r = Replica::new(0, 0, 3); // primary
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        r.on_prepare(&h1);
        r.on_prepare(&h2);

        // Replica 2 asks for op 2 by reference to its checksum (view == 0):
        let mut get_prepare = message_header::GetPrepare {
            cluster: 0,
            replica: 2,
            prepare_op: 2,
            prepare_checksum: h2.checksum(),
            ..message_header::GetPrepare::default()
        };
        get_prepare.set_checksum_body(&[]);
        get_prepare.set_checksum();
        let mut message = crate::message::Message::new();
        message.set_header(&get_prepare);
        r.on_message(&message, 100);

        assert_eq!(r.send_queue.len(), 1);
        let served = r.send_queue[0].header::<message_header::Prepare>().unwrap();
        assert_eq!(served.op, 2);
        assert_eq!(served.checksum(), h2.checksum());
        assert_eq!(served.parent, h1.checksum());
        assert_eq!(served.replica, 0);
    }

    #[test]
    fn on_get_prepare_resolves_op_in_own_journal_for_view_nonzero() {
        let mut r = Replica::new(0, 0, 3); // primary
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        r.on_prepare(&h1);
        r.on_prepare(&h2);

        // No explicit checksum; the responder resolves op 2 in its own log:
        let mut get_prepare = message_header::GetPrepare {
            cluster: 0,
            replica: 2,
            view: 7,
            prepare_op: 2,
            ..message_header::GetPrepare::default()
        };
        get_prepare.set_checksum_body(&[]);
        get_prepare.set_checksum();
        let mut message = crate::message::Message::new();
        message.set_header(&get_prepare);
        r.on_message(&message, 100);

        assert_eq!(r.send_queue.len(), 1);
        let served = r.send_queue[0].header::<message_header::Prepare>().unwrap();
        assert_eq!(served.op, 2);
        assert_eq!(served.checksum(), h2.checksum());
    }

    #[test]
    fn on_get_prepare_is_silent_when_request_cannot_be_satisfied() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);

        // Wrong checksum for an op we do hold:
        let mut get_prepare = message_header::GetPrepare {
            cluster: 0,
            replica: 2,
            prepare_op: 1,
            prepare_checksum: 0xDEAD,
            ..message_header::GetPrepare::default()
        };
        get_prepare.set_checksum_body(&[]);
        get_prepare.set_checksum();
        let mut message = crate::message::Message::new();
        message.set_header(&get_prepare);
        r.on_message(&message, 100);
        assert!(r.send_queue.is_empty());

        // A view != 0 request for an op we do not hold:
        let mut get_prepare = message_header::GetPrepare {
            cluster: 0,
            replica: 2,
            view: 7,
            prepare_op: 2,
            ..message_header::GetPrepare::default()
        };
        get_prepare.set_checksum_body(&[]);
        get_prepare.set_checksum();
        let mut message = crate::message::Message::new();
        message.set_header(&get_prepare);
        r.on_message(&message, 100);
        assert!(r.send_queue.is_empty());
    }

    #[test]
    fn get_prepare_repairs_single_missed_op_and_backup_commits() {
        // Primary r0 prepares and commits ops 1..=2:
        let mut primary = Replica::new(0, 0, 3);
        primary.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        assert_eq!(primary.on_prepare(&h1), OnPrepareResult::Accepted);
        assert_eq!(primary.on_prepare(&h2), OnPrepareResult::Accepted);
        primary.commit_op(1);
        primary.commit_op(2);
        assert_eq!(primary.commit_min, 2);
        assert_eq!(primary.commit_max, 2);

        // Backup r2 only ever received op 1. It later sees a prepare for a
        // future op 3 (out of order); the gap at op 2 is rejected, but the
        // future op's parent reveals op 2's checksum:
        let mut backup = Replica::new(0, 2, 3);
        backup.status = Status::Normal;
        assert_eq!(backup.on_prepare(&h1), OnPrepareResult::Accepted);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        assert_eq!(backup.on_prepare(&h3), OnPrepareResult::FutureOp);
        assert_eq!(backup.op, 1);

        // The backup asks the primary to re-serve op 2:
        backup.send_get_prepare(0, 2, h3.parent);
        let get_prepare = backup.send_queue.pop().unwrap();
        assert!(get_prepare.header::<message_header::GetPrepare>().is_some());

        // The primary answers from its journal:
        primary.on_message(&get_prepare, 50);
        assert_eq!(primary.send_queue.len(), 1);
        let rehearsed = primary.send_queue.pop().unwrap();
        let served = rehearsed.header::<message_header::Prepare>().unwrap();
        assert_eq!(served.op, 2);
        assert_eq!(served.checksum(), h2.checksum());

        // The backup installs the repaired prepare and acks it, then commits:
        backup.on_message(&rehearsed, 100);
        assert_eq!(backup.op, 2);
        let commit = make_commit_message(0, 0, 2, h2.checksum(), 100);
        backup.on_message(&commit, 200);
        assert_eq!(backup.commit_min, 2);
        assert_eq!(backup.commit_max, 2);
        // The commit marks the repaired prepare's slot clean:
        assert!(backup.journal.has_prepare(&h2));
    }

    #[test]
    fn on_get_headers_serves_contiguous_range_descending() {
        let mut r = Replica::new(0, 0, 3); // primary
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        r.on_prepare(&h1);
        r.on_prepare(&h2);
        r.on_prepare(&h3);

        // Replica 2 asks for headers 1..=5:
        let mut get_headers = message_header::GetHeaders {
            cluster: 0,
            replica: 2,
            op_min: 1,
            op_max: 5,
            ..message_header::GetHeaders::default()
        };
        get_headers.set_checksum_body(&[]);
        get_headers.set_checksum();
        let mut request = crate::message::Message::new();
        request.set_header(&get_headers);
        r.on_message(&request, 100);
        assert_eq!(r.send_queue.len(), 1);
        let response = r.send_queue.pop().unwrap();
        let headers = response.header::<message_header::Headers>().unwrap();
        assert_eq!(headers.size as usize, message_header::SIZE * 4); // header + 3
        let decoded = Replica::decode_headers_body(&response).unwrap();
        let ops: Vec<u64> = decoded.iter().map(|h| h.op).collect();
        // copy_latest_headers_between copies newest first:
        assert_eq!(ops, [3, 2, 1]);
        for (served, expected) in decoded.iter().zip([&h3, &h2, &h1]) {
            assert_eq!(served.checksum(), expected.checksum());
        }
        assert_eq!(headers.view, r.view);
        assert_eq!(headers.replica, 0);
    }

    #[test]
    fn on_get_headers_is_silent_without_headers_in_range() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        r.on_prepare(&h1);
        r.on_prepare(&h2);

        // Replica 2 asks for an op-range the journal does not hold:
        let mut get_headers = message_header::GetHeaders {
            cluster: 0,
            replica: 2,
            op_min: 4,
            op_max: 6,
            ..message_header::GetHeaders::default()
        };
        get_headers.set_checksum_body(&[]);
        get_headers.set_checksum();
        let mut request = crate::message::Message::new();
        request.set_header(&get_headers);
        r.on_message(&request, 100);
        assert!(r.send_queue.is_empty());
    }

    #[test]
    fn on_get_headers_bounded_by_index_get_max() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let mut parent = 0;
        for op in 1..=30 {
            let h = make_prepare_for_replica(0, 0, op, parent, 0);
            parent = h.checksum();
            r.on_prepare(&h);
        }
        assert_eq!(r.op, 30);

        let mut get_headers = message_header::GetHeaders {
            cluster: 0,
            replica: 2,
            op_min: 1,
            op_max: 30,
            ..message_header::GetHeaders::default()
        };
        get_headers.set_checksum_body(&[]);
        get_headers.set_checksum();
        let mut request = crate::message::Message::new();
        request.set_header(&get_headers);
        r.on_message(&request, 100);
        let response = r.send_queue.pop().unwrap();
        let decoded = Replica::decode_headers_body(&response).unwrap();
        assert_eq!(decoded.len(), constants::GET_HEADERS_MAX);
    }

    #[test]
    fn on_headers_refreshes_dirty_headers_and_skips_clean() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        assert_eq!(r.on_prepare(&h1), OnPrepareResult::Accepted);
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Accepted);

        // A repair of a dirty (writes pending) prepare refreshes the slot:
        assert!(r.repair_header(&h1));
        assert!(r.repair_header(&h2));
        assert!(r.journal.has_header(&h1));
        assert!(r.journal.has_header(&h2));

        // Once committed (clean), the identical headers are no-ops:
        r.commit_max = 2;
        r.commit_dispatch_enter();
        assert_eq!(r.commit_min, 2);
        assert!(!r.repair_header(&h1));
        assert!(!r.repair_header(&h2));
    }

    #[test]
    fn repair_header_refuses_to_move_or_break_the_head() {
        let mut r = Replica::new(0, 1, 3); // backup
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        assert_eq!(r.on_prepare(&h1), OnPrepareResult::Accepted);
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Accepted);
        assert_eq!(r.op, 2);

        // A newer view is never installed:
        let newer_view = make_prepare_for_replica(0, 1, 1, 0, 0);
        assert!(!r.repair_header(&newer_view));

        // An op ahead of the head may not be repaired in:
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        assert!(!r.repair_header(&h3));

        // A different prepare for the head op would change the hash chain head:
        let divergent = make_prepare_for_replica(0, 0, 2, 0xDEAD, 0);
        assert!(!r.repair_header(&divergent));
        // The original head (and journal) is untouched:
        assert_eq!(r.journal.header_with_op(2).unwrap().checksum(), h2.checksum());
    }

    #[test]
    fn get_headers_round_trip_refreshes_backup_range() {
        // Primary r0 committed ops 1..=3:
        let mut primary = Replica::new(0, 0, 3);
        primary.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        primary.on_prepare(&h1);
        primary.on_prepare(&h2);
        primary.on_prepare(&h3);
        primary.commit_op(1);
        primary.commit_op(2);
        primary.commit_op(3);

        // Backup r2 received 1..=3 but wants to confirm the chain (its slots
        // are still dirty — writes pending):
        let mut backup = Replica::new(0, 2, 3);
        backup.status = Status::Normal;
        backup.on_prepare(&h1);
        backup.on_prepare(&h2);
        backup.on_prepare(&h3);

        // The exchange: backup asks for the range, primary serves it, backup
        // repairs its dirty copies. Op range is served newest-first.
        backup.send_get_headers(0, 1, 5);
        primary.on_message(&backup.send_queue.pop().unwrap(), 50);
        assert_eq!(primary.send_queue.len(), 1);
        let headers = primary.send_queue.pop().unwrap();
        backup.on_message(&headers, 100);

        assert_eq!(backup.op, 3);
        // The dirty slots were refreshed, and the chain head never moved:
        assert_eq!(backup.journal.header_with_op(1).unwrap().checksum(), h1.checksum());
        assert_eq!(backup.journal.header_with_op(2).unwrap().checksum(), h2.checksum());
        assert_eq!(backup.journal.header_with_op(3).unwrap().checksum(), h3.checksum());
        assert!(backup.send_queue.is_empty());
    }

    #[test]
    #[should_panic = "commit checksum verification failed"]
    fn on_commit_panics_on_checksum_mismatch() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);

        // Commit claims op 1 but with the wrong checksum:
        let commit = make_commit_message(0, 0, 1, 0xDEAD, 100);
        r.on_message(&commit, 200);
    }

    #[test]
    fn on_commit_skips_checksum_verification_while_repairing() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        r.on_prepare(&h2);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        r.on_prepare(&h3);
        assert_eq!(r.op, 3);

        // Lose op 2's header while still holding the head (op 3), so our chain
        // from op 1 up to the head is broken — exactly the mid-repair state
        // where a mismatched commit checksum must be tolerated.
        r.journal.remove_entry(crate::journal::Journal::slot_for_op(2));
        assert!(!r.valid_hash_chain_between(1, r.op));

        // Commit claims op 1 (whose entry we hold) with a bogus checksum:
        let commit = make_commit_message(0, 0, 1, 0xDEAD, 100);
        r.on_message(&commit, 200);

        // No panic: commit_max advanced (op 1 applies; op 2 is still absent).
        assert_eq!(r.heartbeat_timestamp, 100);
        assert_eq!(r.commit_max, 1);
        assert_eq!(r.commit_min, 1);
        assert!(r.journal.header_with_op(2).is_none());
    }

    /// Build a checksum-valid Prepare header as a primary would, for feeding
    /// `on_prepare` from the primary's point of view.
    fn make_prepare_for_replica(
        cluster: u128,
        view: u32,
        op: u64,
        parent: u128,
        commit: u64,
    ) -> message_header::Prepare {
        assert!(op > commit);
        let mut header = message_header::Prepare {
            cluster,
            view,
            op,
            parent,
            commit,
            release: crate::multiversion::Release::MINIMUM,
            client: 1,
            timestamp: op,
            request: 1,
            operation: crate::Operation::NOOP,
            ..message_header::Prepare::default()
        };
        header.set_checksum_body(&[]);
        header.set_checksum();
        header
    }

    /// Deliver a Prepare through the full dispatch path (`on_message`).
    fn deliver_prepare(r: &mut Replica, prepare: &message_header::Prepare) {
        let mut message = crate::message::Message::new();
        message.set_header(prepare);
        r.on_message(&message, 100);
    }

    /// Build a checksum-valid Commit message from the primary.
    fn make_commit_message(
        cluster: u128,
        view: u32,
        commit: u64,
        commit_checksum: u128,
        timestamp_monotonic: u64,
    ) -> crate::message::Message {
        let mut c = message_header::Commit::default();
        c.cluster = cluster;
        c.view = view;
        c.replica = 0; // primary replica index for view 0
        c.commit = commit;
        c.commit_checksum = commit_checksum;
        c.timestamp_monotonic = timestamp_monotonic;
        c.set_checksum_body(&[]);
        c.set_checksum();
        let mut message = crate::message::Message::new();
        message.set_header(&c);
        message
    }

    #[test]
    fn pop_committed_drains_pipeline() {
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        r.primary_pipeline_prepare(1, 100, crate::Operation(6), 64, 0).unwrap();
        r.primary_pipeline_prepare(2, 101, crate::Operation(6), 64, 0).unwrap();
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
        // Status stays Normal until the ExitView quorum arrives (`on_exit_view`).
        assert_eq!(r.status, Status::Normal);
        assert_eq!(r.send_queue.len(), 1);
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
        // The replica stays Normal — the ExitView quorum has not arrived yet.
        assert_eq!(r.status, Status::Normal);
        assert_eq!(r.send_queue.len(), 1);
        assert_eq!(r.exit_view_from_all_replicas, 1 << 1);
    }

    #[test]
    fn tick_normal_heartbeat_fault_backup_red_resignals() {
        // The backup must re-signal after ExitView, or it would be red again next tick.
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.commit_fault.signal(0);
        r.tick_normal_heartbeat_fault(10_000);
        assert_eq!(r.status, Status::Normal);
        assert_ne!(r.commit_fault.tardy(10_000), Tardiness::Red);
    }

    #[test]
    fn tick_normal_heartbeat_fault_backup_red_broadcasts_exit_view() {
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        r.commit_fault.signal(0);
        r.tick_normal_heartbeat_fault(10_000);

        assert_eq!(r.status, Status::Normal);
        assert_eq!(r.send_queue.len(), 1);
        let exit_view = r.send_queue[0].header::<message_header::ExitView>().unwrap();
        assert_eq!(exit_view.view, 0);
        assert_eq!(exit_view.replica, 1);
        assert_eq!(r.exit_view_from_all_replicas, 1 << 1);
    }

    #[test]
    fn view_change_reached_by_exchange_of_exit_views() {
        // Backups 1 and 2 both lose the primary (0) and broadcast ExitView for
        // view 0. When they exchange those messages they reach the view-change
        // quorum (2 of 3) and transition to view 1.
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        r1.commit_fault.signal(0);
        r1.tick_normal_heartbeat_fault(10_000);

        let mut r2 = Replica::new(0, 2, 3);
        r2.status = Status::Normal;
        r2.commit_fault.signal(0);
        r2.tick_normal_heartbeat_fault(10_000);

        assert_eq!(r1.status, Status::Normal);
        assert_eq!(r2.status, Status::Normal);
        assert_eq!(r1.send_queue.len(), 1);
        assert_eq!(r2.send_queue.len(), 1);

        // Exchange the queued ExitViews. Each replica already has its own bit
        // set, so a second distinct bit reaches the quorum and triggers the
        // transition to view 1.
        r1.on_message(&r2.send_queue[0], 10_001);
        r2.on_message(&r1.send_queue[0], 10_001);

        // Each replica has now joined view 1; `log_view` keeps point at view 0,
        // the last view the log is known to be valid in (upstream semantics).
        assert_eq!(r1.view, 1);
        assert_eq!(r1.log_view, 0);
        assert_eq!(r1.status, Status::ViewChange);
        assert_eq!(r2.view, 1);
        assert_eq!(r2.log_view, 0);
    }

    #[test]
    fn join_view_message_carries_journal_suffix_and_bitsets() {
        // Backups 1 and 2 prepare ops 1..3, commit 1, then lose the primary.
        let mut r = Replica::new(0, 1, 3);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        assert_eq!(r.on_prepare(&h1), OnPrepareResult::Accepted);
        assert_eq!(r.on_prepare(&h2), OnPrepareResult::Accepted);
        assert_eq!(r.on_prepare(&h3), OnPrepareResult::Accepted);
        r.on_message(&make_commit_message(0, 0, 1, h1.checksum(), 100), 100);
        assert_eq!(r.commit_min, 1);

        // Lose the primary; expect ExitView then (from transition) JoinView.
        r.commit_fault.signal(0);
        r.tick_normal_heartbeat_fault(10_000);
        assert_eq!(r.send_queue.len(), 1); // ExitView

        // Reach the view-change quorum with a second ExitView, joining view 1.
        let mut peer = Replica::new(0, 2, 3);
        peer.status = Status::Normal;
        peer.commit_fault.signal(0);
        peer.tick_normal_heartbeat_fault(10_000);
        r.on_message(&peer.send_queue[0], 10_001);
        assert_eq!(r.view, 1);
        assert_eq!(r.send_queue.len(), 2); // ExitView + JoinView

        let join_view = r.send_queue[1].header::<message_header::JoinView>().unwrap();
        assert_eq!(join_view.view, 1);
        assert_eq!(join_view.log_view, 0);
        assert_eq!(join_view.replica, 1);
        assert_eq!(join_view.op, 3); // head of the journal suffix
        assert_eq!(join_view.commit_min, 1);

        // Body holds [3, 2, 1] (descending op).
        let body = r.send_queue[1].body_used();
        assert_eq!(body.len(), 3 * message_header::SIZE);
        let mut decoded = Vec::new();
        for i in 0..3 {
            let chunk: &[u8; message_header::SIZE] =
                body[i * message_header::SIZE..(i + 1) * message_header::SIZE].try_into().unwrap();
            decoded.push(message_header::Prepare::from_wire(chunk).unwrap());
        }
        assert_eq!(decoded[0], h3);
        assert_eq!(decoded[1], h2);
        assert_eq!(decoded[2], h1);

        // The committed op is durable (present); the uncommitted suffix is
        // dirty, so it is nacked to give the new primary freedom to truncate.
        assert_eq!(join_view.present_bitset, 1 << 2);
        assert_eq!(join_view.nack_bitset, (1 << 0) | (1 << 1));
    }

    /// Prepare chained ops `1..=op_max` (view 0) and commit op `committed` on a
    /// Normal replica, returning the prepared headers in ascending-op order.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
    fn prepare_and_commit_suffix(
        r: &mut Replica,
        cluster: u128,
        op_max: u64,
        committed: u64,
    ) -> Vec<message_header::Prepare> {
        assert!(committed >= 1);
        assert!(committed <= op_max);
        let mut parent = 0;
        let mut headers = Vec::new();
        for op in 1..=op_max {
            let header = make_prepare_for_replica(cluster, 0, op, parent, 0);
            parent = header.checksum();
            assert_eq!(r.on_prepare(&header), OnPrepareResult::Accepted);
            headers.push(header);
        }
        r.on_message(
            &make_commit_message(
                cluster,
                0,
                committed,
                headers[committed as usize - 1].checksum(),
                100,
            ),
            100,
        );
        headers
    }

    /// Simulate the WAL writes for ops `committed+1..=op_max` completing, so the
    /// journal records them as durable/present (upstream `write_prepare` +
    /// `write_prepare_callback`). Sans-IO, `commit_op` is the only other writer
    /// of the durable bits.
    fn mark_suffix_durable(r: &mut Replica, committed: u64, op_max: u64) {
        for op in committed + 1..=op_max {
            let header = r.journal.header_with_op(op).copied().unwrap();
            let slot = crate::journal::Journal::slot_for_op(op);
            r.journal.dirty.clear(slot);
            r.journal.prepare_inhabited[slot.index] = true;
            r.journal.prepare_checksums[slot.index] = header.checksum();
        }
    }

    #[test]
    fn on_join_view_new_primary_installs_log_and_sends_view() {
        // Replica 1 is the new primary (view 1). It and replica 2 both prepared
        // ops 1..3 and committed op 1 before the old primary (0) failed. The
        // uncommitted suffix (ops 2,3) is dirty on both, so the CTRL quorum
        // nacks and truncates it: the new log is [1].
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        let headers = prepare_and_commit_suffix(&mut r1, 0, 3, 1);
        r1.transition_to_view_change_status(1);
        assert_eq!(r1.send_queue.len(), 1); // JoinView
        assert!(r1.join_view_from_all_replicas[1].is_some());

        let mut r2 = Replica::new(0, 2, 3);
        r2.status = Status::Normal;
        prepare_and_commit_suffix(&mut r2, 0, 3, 1);
        r2.transition_to_view_change_status(1);
        let peer_jv = r2.send_queue.pop().unwrap();

        r1.on_message(&peer_jv, 20_000);

        // Quorum reached; log established; View broadcast; the new primary
        // returned to Normal (its quorum flag reset by the transition).
        assert!(!r1.join_view_quorum);
        assert_eq!(r1.status, Status::Normal);
        assert!(r1.is_primary());
        assert_eq!(r1.log_view, 1);
        assert_eq!(r1.view, 1);
        assert_eq!(r1.op, 1); // nacked suffix truncated
        assert_eq!(r1.commit_max, 1);
        assert_eq!(r1.commit_min, 1);
        assert_eq!(r1.send_queue.len(), 2); // JoinView + View

        let view = r1.send_queue[1].header::<message_header::View>().unwrap();
        assert_eq!(view.view, 1);
        assert_eq!(view.replica, 1);
        assert_eq!(view.op, 1);
        assert_eq!(view.commit_max, 1);

        // Body: zeroed CheckpointState prefix + the surviving headers descending.
        let body = r1.send_queue[1].body_used();
        assert_eq!(body.len(), constants::CHECKPOINT_STATE_SIZE + 2 * message_header::SIZE);
        assert_eq!(
            &body[..constants::CHECKPOINT_STATE_SIZE],
            &[0; constants::CHECKPOINT_STATE_SIZE]
        );
        let start = constants::CHECKPOINT_STATE_SIZE;
        let head: &[u8; message_header::SIZE] =
            body[start..start + message_header::SIZE].try_into().unwrap();
        assert_eq!(message_header::Prepare::from_wire(head).unwrap(), headers[0]);
        let root: &[u8; message_header::SIZE] = body
            [start + message_header::SIZE..start + 2 * message_header::SIZE]
            .try_into()
            .unwrap();
        assert_eq!(message_header::Prepare::from_wire(root).unwrap().op, 0);
    }

    #[test]
    fn primary_journal_headers_repaired_requires_contiguous_hash_chain() {
        // Replica 0 is the view-0 primary. Ops 1..3 prepared, op 1 committed.
        let mut r = Replica::new(0, 0, 1);
        r.status = Status::Normal;
        let headers = prepare_and_commit_suffix(&mut r, 0, 3, 1);
        assert!(r.primary_journal_headers_repaired());

        // Punch a hole: lose op 2's header+prepare but keep the head (op 3).
        r.journal.remove_entry(crate::journal::Journal::slot_for_op(2));
        assert!(!r.primary_journal_headers_repaired());

        // Recovering the header from a peer reconnects the chain.
        assert!(r.repair_header(&headers[1]));
        assert_eq!(r.journal.header_with_op(2), Some(&headers[1]));
        assert!(r.primary_journal_headers_repaired());
    }

    #[test]
    #[should_panic(expected = "primary_journal_headers_repaired")]
    fn primary_send_view_panics_when_journal_headers_unrepaired() {
        let mut r = Replica::new(0, 0, 1);
        r.status = Status::Normal;
        prepare_and_commit_suffix(&mut r, 0, 3, 1);
        r.journal.remove_entry(crate::journal::Journal::slot_for_op(2));
        r.primary_send_view();
    }

    #[test]
    fn primary_journal_prepares_repaired_requires_written_prepares() {
        let mut r = Replica::new(0, 0, 1);
        r.status = Status::Normal;
        let h1 = make_prepare_for_replica(0, 0, 1, 0, 0);
        r.on_prepare(&h1);
        let h2 = make_prepare_for_replica(0, 0, 2, h1.checksum(), 0);
        r.on_prepare(&h2);
        let h3 = make_prepare_for_replica(0, 0, 3, h2.checksum(), 0);
        r.on_prepare(&h3);
        // A single-replica primary commits through `commit_op`, not the Commit
        // message (which on_commit ignores for the primary).
        r.commit_op(1);

        // Ops 2,3 were prepared but never written to the WAL — still dirty.
        assert!(!r.primary_journal_prepares_repaired());
        assert!(!r.primary_journal_repaired());

        // Model the WAL write completions; the journal is now fully repaired.
        mark_suffix_durable(&mut r, 1, 3);
        assert!(r.primary_journal_prepares_repaired());
        assert!(r.primary_journal_repaired());
    }

    #[test]
    fn on_join_view_does_nothing_without_quorum() {
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        prepare_and_commit_suffix(&mut r1, 0, 3, 1);
        r1.transition_to_view_change_status(1);
        assert_eq!(r1.send_queue.len(), 1); // only our own JoinView

        // No second JV: quorum (2 of 3) not yet reached — no View, no log set.
        assert!(!r1.join_view_quorum);
        assert_eq!(r1.log_view, 0);
        assert_eq!(r1.send_queue.len(), 1);
    }

    #[test]
    fn on_join_view_ignored_by_backup() {
        // Replica 2 is a backup for view 1; it stages its own JV broadcast but
        // must not process incoming JVs.
        let mut r2 = Replica::new(0, 2, 3);
        r2.status = Status::Normal;
        prepare_and_commit_suffix(&mut r2, 0, 3, 1);
        r2.transition_to_view_change_status(1);
        // Backups never fill their own JV slot (only the new primary does).
        assert!(r2.join_view_from_all_replicas[2].is_none());

        // A peer JV arrives — the backup ignores it.
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        prepare_and_commit_suffix(&mut r1, 0, 3, 1);
        r1.transition_to_view_change_status(1);
        let peer_jv = r1.send_queue.pop().unwrap();
        r2.on_message(&peer_jv, 20_000);

        assert!(!r2.join_view_quorum);
        assert_eq!(r2.log_view, 0);
        // No View emitted: the queue still holds only r2's own staged JoinView.
        assert_eq!(r2.send_queue.len(), 1);
        assert_eq!(r2.send_queue[0].header::<message_header::JoinView>().unwrap().replica, 2);
    }

    #[test]
    fn view_change_reaches_view_message() {
        // End-to-end sans-IO view change: replicas 1 and 2 red-detect the dead
        // primary 0, exchange ExitViews, transition to view 1, and the new
        // primary (1) collects a JV quorum and broadcasts a View.
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        prepare_and_commit_suffix(&mut r1, 0, 3, 1);
        r1.commit_fault.signal(0);
        r1.tick_normal_heartbeat_fault(10_000); // ExitView
        assert_eq!(r1.send_queue.len(), 1);

        let mut r2 = Replica::new(0, 2, 3);
        r2.status = Status::Normal;
        prepare_and_commit_suffix(&mut r2, 0, 3, 1);
        r2.commit_fault.signal(0);
        r2.tick_normal_heartbeat_fault(10_000); // ExitView

        let ev1 = r1.send_queue[0].header::<message_header::ExitView>().unwrap();
        let ev2 = r2.send_queue[0].header::<message_header::ExitView>().unwrap();
        r1.on_message(&r2.send_queue[0], 10_001);
        r2.on_message(&r1.send_queue[0], 10_001);
        assert_eq!(r1.status, Status::ViewChange);
        assert_eq!(r2.status, Status::ViewChange);
        // The ExitViews advertised the old view (0), before the transition.
        assert_eq!(ev1.view, 0);
        assert_eq!(ev2.view, 0);

        // r1 is the new primary. Deliver r2's JoinView to complete its quorum.
        // Each replica's queue holds [ExitView, JoinView].
        assert_eq!(r2.send_queue.len(), 2);
        let jv = r2.send_queue.pop().unwrap();
        assert_eq!(r1.send_queue.len(), 2);
        r1.on_message(&jv, 10_002);
        assert_eq!(r1.send_queue.len(), 3); // ExitView + JoinView + View
        assert_eq!(r1.log_view, 1);

        let view = r1.send_queue[2].header::<message_header::View>().unwrap();
        assert_eq!(view.view, 1);
        assert_eq!(view.op, 1);
        assert_eq!(view.commit_max, 1);

        // Deliver the new primary's View to the backup: it installs the log,
        // returns to Normal, and (with commit_max == op and nothing above it)
        // has nothing to re-ack or commit.
        let r1_view = std::mem::take(&mut r1.send_queue).pop().unwrap();
        r2.on_message(&r1_view, 10_003);
        assert_eq!(r2.status, Status::Normal);
        assert_eq!(r2.view, 1);
        assert_eq!(r2.log_view, 1);
        assert_eq!(r2.op, 1);
        assert_eq!(r2.commit_min, 1);
        assert_eq!(r2.commit_max, 1);
        assert!(!r2.join_view_quorum);
        assert_eq!(r2.exit_view_from_all_replicas, 0);
        // The queue still holds the stale ExitView from the old view (the
        // JoinView was popped and delivered as r1's quorum member).
        assert_eq!(r2.send_queue.len(), 1);
        assert_eq!(r2.send_queue[0].header::<message_header::ExitView>().unwrap().view, 0);
    }

    #[test]
    fn on_view_backup_floods_prepare_oks_for_uncommitted_suffix() {
        // Ops 2,3 survived as durable (present, not nacked), so the new log's
        // head is op 3. The backup installs the log, returns to Normal, and — as
        // the new primary cannot commit ops 2,3 without its prepare_oks — floods
        // a PrepareOk for each of them.
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        let headers = prepare_and_commit_suffix(&mut r1, 0, 3, 1);
        mark_suffix_durable(&mut r1, 1, 3);
        r1.transition_to_view_change_status(1);

        let mut r2 = Replica::new(0, 2, 3);
        r2.status = Status::Normal;
        prepare_and_commit_suffix(&mut r2, 0, 3, 1);
        mark_suffix_durable(&mut r2, 1, 3);
        r2.transition_to_view_change_status(1);
        let peer_jv = r2.send_queue.pop().unwrap();

        r1.on_message(&peer_jv, 20_000);
        assert_eq!(r1.op, 3);
        let view_msg = r1.send_queue.pop().unwrap();
        let view = view_msg.header::<message_header::View>().unwrap();
        assert_eq!(view.op, 3);
        assert_eq!(view.commit_max, 1);

        r2.on_message(&view_msg, 20_001);
        assert_eq!(r2.status, Status::Normal);
        assert_eq!(r2.view, 1);
        assert_eq!(r2.log_view, 1);
        assert_eq!(r2.op, 3);
        assert_eq!(r2.commit_max, 1);
        assert_eq!(r2.commit_min, 1);

        // PrepareOk flood: our staged JoinView was consumed, so the queue holds
        // exactly the two acks, in ascending op order.
        assert_eq!(r2.send_queue.len(), 2);
        let ok2 = r2.send_queue[0].header::<message_header::PrepareOk>().unwrap();
        let ok3 = r2.send_queue[1].header::<message_header::PrepareOk>().unwrap();
        assert_eq!(ok2.op, 2);
        assert_eq!(ok2.prepare_checksum, headers[1].checksum());
        assert_eq!(ok3.op, 3);
        assert_eq!(ok3.prepare_checksum, headers[2].checksum());
        assert_eq!(ok3.replica, 2);
    }

    #[test]
    fn on_view_ignored_stale_or_misdirected() {
        // r2 is already Normal in view 1; a stale View from the previous view
        // and a misdirected self-View are both dropped without side effects.
        let mut r2 = Replica::new(0, 2, 3);
        r2.status = Status::Normal;
        prepare_and_commit_suffix(&mut r2, 0, 3, 1);
        r2.transition_to_view_change_status(1);
        let peer_jv = r2.send_queue.pop().unwrap();
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        prepare_and_commit_suffix(&mut r1, 0, 3, 1);
        r1.transition_to_view_change_status(1);
        r1.on_message(&peer_jv, 20_000);
        let view_msg = r1.send_queue.pop().unwrap();
        assert_eq!(view_msg.header::<message_header::View>().unwrap().view, 1);

        r2.on_message(&view_msg, 20_001);
        assert_eq!(r2.status, Status::Normal);
        assert_eq!(r2.log_view, 1);
        // r2's queue is empty: its JoinView was popped above (as r1's quorum
        // member) and the View produced no reply messages (nothing uncommitted).

        // A stale View from the previous view (0) is dropped in
        // `ignore_view_change_message` before any journal update.
        let mut stale_view = message_header::View {
            cluster: 0,
            replica: 1,
            view: 0, // < r2.view == 1
            op: 1,
            commit_max: 1,
            size: u32::try_from(message_header::SIZE + constants::CHECKPOINT_STATE_SIZE).unwrap(),
            ..message_header::View::default()
        };
        let mut stale_msg = crate::message::Message::new();
        stale_msg.set_body(&[0; constants::CHECKPOINT_STATE_SIZE]);
        stale_msg.set_header(&stale_view);
        let body = stale_msg.body_used();
        stale_view.set_checksum_body(body);
        stale_view.set_checksum();
        stale_msg.set_header(&stale_view);
        r2.on_message(&stale_msg, 20_002);

        // Misdirected: a replica must never process a View it authored.
        let mut self_view = message_header::View {
            cluster: 0,
            replica: 2, // misdirected: sent to itself
            view: 1,
            op: 1,
            commit_max: 1,
            size: u32::try_from(message_header::SIZE + constants::CHECKPOINT_STATE_SIZE).unwrap(),
            ..message_header::View::default()
        };
        let mut self_msg = crate::message::Message::new();
        self_msg.set_body(&[0; constants::CHECKPOINT_STATE_SIZE]);
        self_msg.set_header(&self_view);
        let body = self_msg.body_used();
        self_view.set_checksum_body(body);
        self_view.set_checksum();
        self_msg.set_header(&self_view);
        r2.on_message(&self_msg, 20_003);

        // No message emitted, and the log is untouched.
        assert!(r2.send_queue.is_empty());
        assert_eq!(r2.log_view, 1);
        assert_eq!(r2.op, 1);
    }

    #[test]
    fn view_change_commits_survivors_after_prepare_ok_flood() {
        // Ops 2,3 survived as durable, so the quorum keeps a log headed at op 3.
        // The new primary returns to Normal with the survivors in its pipeline,
        // the backup re-acks them after the View, and the survivors commit —
        // then the commit heartbeat carries the result back to the backup.
        let mut r1 = Replica::new(0, 1, 3);
        r1.status = Status::Normal;
        prepare_and_commit_suffix(&mut r1, 0, 3, 1);
        mark_suffix_durable(&mut r1, 1, 3);
        r1.transition_to_view_change_status(1);

        let mut r2 = Replica::new(0, 2, 3);
        r2.status = Status::Normal;
        prepare_and_commit_suffix(&mut r2, 0, 3, 1);
        mark_suffix_durable(&mut r2, 1, 3);
        r2.transition_to_view_change_status(1);
        let r2_jv = r2.send_queue.pop().unwrap();

        // The new primary transitions to Normal with the survivors in pipeline,
        // having contributed its own prepare_oks toward each.
        r1.on_message(&r2_jv, 20_000);
        assert_eq!(r1.status, Status::Normal);
        assert!(r1.is_primary());
        assert_eq!(r1.view, 1);
        assert_eq!(r1.log_view, 1);
        assert_eq!(r1.op, 3);
        assert_eq!(r1.commit_min, 1);
        assert_eq!(r1.commit_max, 1);
        let survivor_ops: Vec<u64> = r1.pipeline_queue.prepare_queue.iter().map(|p| p.op).collect();
        assert_eq!(survivor_ops, [2, 3]);
        assert_eq!(r1.send_queue.len(), 2); // JoinView + View
        assert!(!r1.join_view_quorum); // reset by the transition to Normal

        // The backup installs the log (op 3) and floods prepare_oks for 2,3.
        r2.on_message(&r1.send_queue[1], 20_001);
        assert_eq!(r2.status, Status::Normal);
        assert_eq!(r2.op, 3);
        assert_eq!(r2.commit_max, 1);
        assert_eq!(r2.send_queue.len(), 2);
        let ok2 = r2.send_queue[0].header::<message_header::PrepareOk>().unwrap();
        let ok3 = r2.send_queue[1].header::<message_header::PrepareOk>().unwrap();
        assert_eq!(ok2.op, 2);
        assert_eq!(ok3.op, 3);

        // Each backup ack completes the quorum (self + r2) of a survivor.
        r1.on_message(&r2.send_queue[0], 20_002);
        assert_eq!(r1.commit_min, 2);
        assert_eq!(r1.commit_max, 2);
        let survivor_ops: Vec<u64> = r1.pipeline_queue.prepare_queue.iter().map(|p| p.op).collect();
        assert_eq!(survivor_ops, [3]);

        r1.on_message(&r2.send_queue[1], 20_003);
        assert_eq!(r1.commit_min, 3);
        assert_eq!(r1.commit_max, 3);
        assert!(r1.pipeline_queue.prepare_queue.is_empty());

        // The commit heartbeat reached the backup, which commits its (clean)
        // op 2,3 from the journal.
        r1.send_commit(20_004);
        let commit = r1.send_queue.pop().unwrap();
        assert!(commit.header::<message_header::Commit>().is_some());
        r2.on_message(&commit, 20_005);
        assert_eq!(r2.commit_min, 3);
        assert_eq!(r2.commit_max, 3);
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
    fn on_ping_replies_with_pong() {
        let mut r = Replica::new(CLUSTER, 0, 2);
        r.status = Status::Normal;
        // The pong echoes the pinger's monotonic timestamp and carries this
        // replica's wall-clock sample (`monotonic_now` sans-IO).
        r.monotonic_now = 5;

        let ping_msg = ping_message(1, 123_456_789);
        r.on_message(&ping_msg, 5);
        assert_eq!(r.send_queue.len(), 1);
        let reply = r.send_queue.remove(0);

        let pong = reply.header::<message_header::Pong>().expect("pong replied");
        assert!(pong.valid_checksum());
        assert!(pong.valid_checksum_body(&[]));
        assert_eq!(pong.invalid_header(), None);
        assert_eq!(pong.cluster, CLUSTER);
        assert_eq!(pong.replica, 0);
        assert_eq!(pong.release, crate::multiversion::Release::MINIMUM);
        assert_eq!(pong.ping_timestamp_monotonic, 123_456_789);
        assert_eq!(pong.pong_timestamp_wall, 5);
    }

    #[test]
    fn on_ping_ignored_when_not_normal_or_misdirected() {
        // A recovering replica does not reply (upstream replica.zig:1851).
        let mut r = Replica::new(CLUSTER, 0, 2);
        let ping = ping_message(1, 11);
        r.on_message(&ping, 0);
        assert!(r.send_queue.is_empty());

        // Misdirected (from self): dropped even when Normal.
        r.status = Status::Normal;
        let self_ping = ping_message(0, 11);
        r.on_message(&self_ping, 0);
        assert!(r.send_queue.is_empty());
    }

    #[test]
    fn on_pong_ignores_misdirected_and_standby() {
        let mut r = Replica::new(CLUSTER, 0, 2);
        // Misdirected (self): dropped.
        let self_pong = pong_message(0);
        r.on_message(&self_pong, 0);
        assert!(r.send_queue.is_empty());
        // A standby's clock is ignored (replica >= replica_count).
        let standby_pong = pong_message(2);
        r.on_message(&standby_pong, 0);
        assert!(r.send_queue.is_empty());
    }

    #[test]
    fn on_ping_timeout_broadcasts_pings() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.ping_timeout = Timeout::start(1);

        r.tick(5);
        assert_eq!(r.send_queue.len(), 2, "one ping to each other replica");
        let body_len =
            size_of::<crate::multiversion::Release>() * constants::VSR_RELEASES_MAX as usize;
        for message in &r.send_queue {
            let ping = message.header::<message_header::Ping>().expect("broadcast ping");
            assert!(ping.valid_checksum());
            assert!(ping.valid_checksum_body(message.body_used()));
            assert_eq!(ping.invalid_header(), None);
            assert_eq!(ping.cluster, CLUSTER);
            assert_eq!(ping.replica, 0);
            assert_eq!(ping.release, crate::multiversion::Release::MINIMUM);
            assert_eq!(ping.checkpoint_op, 0);
            assert_eq!(ping.ping_timestamp_monotonic, 5);
            assert_eq!(message.body_used().len(), body_len);
        }
    }

    #[test]
    fn on_ping_client_replies_with_pong_client() {
        let mut r = Replica::new(CLUSTER, 0, 3);
        assert_eq!(r.client_sessions.count(), 0);
        // session == 0: the client may not be registered yet, but time sync is
        // always answered (upstream only evicts a registered-but-unknown client).
        let ping_msg = ping_client_message(42, 0, 123_456_789);
        r.on_message(&ping_msg, 9);
        assert_eq!(r.send_queue.len(), 0);
        assert_eq!(r.client_send_queue.len(), 1);

        let reply = r.client_send_queue.remove(0);
        let pong_client = reply.header::<message_header::PongClient>().expect("pong_client");
        assert!(pong_client.valid_checksum());
        assert!(pong_client.valid_checksum_body(&[]));
        assert_eq!(pong_client.invalid_header(), None);
        assert_eq!(pong_client.cluster, CLUSTER);
        assert_eq!(pong_client.replica, 0);
        assert_eq!(pong_client.view, r.log_view);
        assert_eq!(pong_client.release, crate::multiversion::Release::MINIMUM);
        assert_eq!(pong_client.ping_timestamp_monotonic, 123_456_789);
    }

    #[test]
    fn on_ping_client_evicts_committed_but_unregistered_client() {
        // The primary has already committed the client's register (commit_min),
        // yet the client is not in the sessions table — it must re-register.
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        assert!(r.is_primary());
        r.commit_min = 100;

        let ping_client = ping_client_message(42, 50, 11);
        r.on_message(&ping_client, 0);
        assert!(r.client_send_queue.len() == 1, "an eviction replaces the pong");
        let message = r.client_send_queue.remove(0);
        let eviction = message.header::<message_header::Eviction>().expect("eviction");
        assert!(eviction.valid_checksum());
        assert_eq!(eviction.invalid_header(), None);
        assert_eq!(eviction.client, 42);
        assert_eq!(eviction.reason(), Some(message_header::Reason::NoSession));
    }

    #[test]
    fn on_ping_client_answers_pong_for_register_in_flight() {
        // The register is still uncommitted (session > commit_min), so the
        // primary cannot evict yet and answers the ping.
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        r.commit_min = 30;

        let ping_client = ping_client_message(42, 50, 11);
        r.on_message(&ping_client, 0);
        assert_eq!(r.client_send_queue.len(), 1);
        let reply = r.client_send_queue.remove(0);
        assert!(reply.header::<message_header::PongClient>().is_some());
    }

    #[test]
    fn on_ping_client_evicts_unsupported_release() {
        // A release above the supported minimum (sans-IO is single-versioned)
        // gets an eviction when the replica is an up-to-date primary.
        let mut r = Replica::new(CLUSTER, 0, 3);
        r.status = Status::Normal;
        let mut ping_client = ping_client_message(42, 0, 11);
        let mut header = ping_client.header::<message_header::PingClient>().expect("ping_client");
        header.release = crate::multiversion::Release { value: 2 };
        header.set_checksum();
        ping_client.set_header(&header);

        r.on_message(&ping_client, 0);
        assert_eq!(r.client_send_queue.len(), 1);
        let message = r.client_send_queue.remove(0);
        let eviction = message.header::<message_header::Eviction>().expect("eviction");
        assert_eq!(eviction.reason(), Some(message_header::Reason::ClientReleaseTooHigh));
    }

    #[test]
    fn on_ping_client_from_non_primary_is_answered_not_evicted() {
        // Backups answer PongClient but never evict; the uncommitted/unknown
        // session path is a no-op for them.
        let mut r = Replica::new(CLUSTER, 1, 3);
        r.status = Status::Normal;
        assert!(!r.is_primary());
        r.commit_min = 100;

        let ping_client = ping_client_message(42, 50, 11);
        r.on_message(&ping_client, 0);
        assert_eq!(r.client_send_queue.len(), 1);
        let reply = r.client_send_queue.remove(0);
        assert!(reply.header::<message_header::PongClient>().is_some());
        assert_eq!(r.client_send_queue.len(), 0);
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
        r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();

        r.commit_dispatch_enter();
        assert_eq!(r.commit_stage, CommitStage::Idle);
        assert_eq!(r.commit_min, 0);
    }

    #[test]
    fn commit_dispatch_pipeline_with_quorum_executes() {
        // Primary with a quorum'd prepare: commit executes fully.
        let mut r = Replica::new(0, 0, 3);
        r.status = Status::Normal;
        let op = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();

        // Quorum = 2 (replica_count 3): the primary's own prepare_ok was
        // contributed on prepare, so one backup ack completes the quorum.
        let checksum = r.journal.header_with_op(op).unwrap().checksum();
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

        let op1 = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
        let op2 = r.primary_pipeline_prepare(2, 2, crate::Operation::NOOP, 0, 0).unwrap();
        let op3 = r.primary_pipeline_prepare(3, 3, crate::Operation::NOOP, 0, 0).unwrap();

        // Quorum all three; the primary's own prepare_ok brings each to
        // quorum with a single backup ack.
        for op in [op1, op2, op3] {
            let checksum = r.journal.header_with_op(op).unwrap().checksum();
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
        let op = r.primary_pipeline_prepare(1, 1, crate::Operation::NOOP, 0, 0).unwrap();
        let checksum = r.journal.header_with_op(op).unwrap().checksum();
        r.on_prepare_ok(op, checksum, 1); // replica 1 → quorum (with the self-ack)

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
