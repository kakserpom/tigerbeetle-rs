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
}

#[cfg(test)]
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
}
