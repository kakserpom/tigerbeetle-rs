//! Track corrupt/missing grid blocks for repair (upstream
//! `src/vsr/grid_blocks_missing.zig`).
//!
//! The collection holds the blocks a replica wants re-fetched from the cluster, keyed by
//! block address:
//! - `repair_block()` is enqueued when a non-repair read encounters a corrupt block (and by
//!   the grid scrubber, not yet ported).
//! - `write_commence()`/`write_complete()` drive a fault through the repairing write back to
//!   storage; on success the fault is removed.
//! - `checkpoint_durable_commence()`/`checkpoint_durable_complete()` abort repairs of blocks
//!   that are about to be freed once the checkpoint is durable.
//!
//! Only the single-block repair machinery is ported. DEVIATION: the LSM-aware table-sync
//! half (`sync_table`, `RepairTable`, `sync_jump_*`, `sync_tables_cancel`,
//! `enqueued_blocks_sync`, the `faulty_tables` queues and `Cause.sync`) is not ported —
//! state-sync is Phase 3. Consequently `FaultyBlock.cause` is always `.repair` and the two
//! maps/`syncing_faulty_blocks` collapse into one.
//!
//! DEVIATION: upstream keys `FaultyBlocks` with an insertion-ordered `AutoArrayHashMap` and
//! rotates the starting index at random when sending requests; this port uses a
//! `BTreeMap` (deterministic address order) because `std::collections::HashMap` iteration
//! order would break bit-for-bit reproducibility (AGENTS.md determinism rules).
//! `send_get_blocks` still applies the same prng rotation, mirroring the observable
//! request-spreading behavior.
//!
//! Invariants (upstream `verify`):
//! - every `faulty_blocks` address is not free,
//! - `faulty_blocks.len() == enqueued_blocks_repair`,
//! - `enqueued_blocks_repair <= blocks_max`.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use tigerbeetle_lsm::free_set::FreeSet;

use crate::BlockRequest;

/// Upstream `FaultyBlock.state` transitions:
/// - initial state is `waiting`,
/// - `waiting → writing` when the block arrives and begins to repair,
/// - `writing → aborting` when the checkpoint becomes durable and the (writing) block is to
///   be freed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultState {
    Waiting,
    Writing,
    Aborting,
}

/// Upstream `FaultyBlock` — `cause` is always `repair` in this port (DEVIATION, see the
/// module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FaultyBlock {
    checksum: u128,
    state: FaultState,
}

/// Upstream `GridBlocksMissing.state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Repairing,
    CheckpointDurable { aborting: u64 },
}

/// The single-block repair subset of upstream `GridBlocksMissing`.
#[derive(Debug)]
pub struct GridBlocksMissing {
    /// Lower bound for the limit of concurrent `repair_block()`s available
    /// (upstream `Options.blocks_max`).
    blocks_max: usize,
    /// Keyed by block address.
    ///
    /// Invariants:
    /// - For every block address in `faulty_blocks`, `¬free_set.is_free(address)`.
    faulty_blocks: BTreeMap<u64, FaultyBlock>,
    /// Number of faults enqueued for repair (upstream `enqueued_blocks_repair`).
    enqueued_blocks_repair: usize,
    state: State,
}

impl GridBlocksMissing {
    /// # Panics
    /// Asserts the capacity arithmetic fits (upstream `init`).
    #[must_use]
    pub fn new(blocks_max: usize) -> Self {
        Self {
            blocks_max,
            faulty_blocks: BTreeMap::new(),
            enqueued_blocks_repair: 0,
            state: State::Repairing,
        }
    }

    /// Upstream `verify` (release-build assertions, always enabled like upstream).
    fn verify(&self) {
        assert_eq!(self.faulty_blocks.len(), self.enqueued_blocks_repair);
        assert!(self.enqueued_blocks_repair <= self.blocks_max);

        let mut aborting = 0;
        for (address, fault) in &self.faulty_blocks {
            assert!(*address > 0);
            aborting += u64::from(fault.state == FaultState::Aborting);
        }
        match self.state {
            State::CheckpointDurable { aborting: count } => assert_eq!(aborting, count),
            State::Repairing => assert_eq!(aborting, 0),
        }
    }

    /// Number of faults currently tracked (upstream `faulty_blocks.count()`).
    #[must_use]
    pub fn count(&self) -> usize {
        self.faulty_blocks.len()
    }

    /// Upstream `fault_at_index`: the request to send for the fault at `fault_index`, or
    /// `None` while the fault is already being written (or aborting).
    ///
    /// Note that returning `None` doesn't necessarily indicate that there are no more
    /// blocks.
    ///
    /// # Panics
    /// Asserts `fault_index` is in range (upstream).
    #[must_use]
    pub fn fault_at_index(&self, fault_index: usize) -> Option<BlockRequest> {
        assert!(!self.faulty_blocks.is_empty());
        assert!(fault_index < self.faulty_blocks.len());

        let (address, fault) = self.faulty_blocks.iter().enumerate().nth(fault_index)?.1;
        match fault.state {
            FaultState::Waiting => Some(BlockRequest {
                block_checksum: fault.checksum,
                block_address: *address,
                reserved: [0; 8],
            }),
            FaultState::Writing | FaultState::Aborting => None,
        }
    }

    /// Count the number of *non-table* block repairs available (upstream
    /// `repair_blocks_available`, with the table-sync reserves dropped).
    #[must_use]
    pub fn repair_blocks_available(&self) -> usize {
        self.blocks_max - self.enqueued_blocks_repair
    }

    /// Queue a faulty block to request from the cluster and repair.
    ///
    /// Enqueuing a duplicate `(address, checksum)` is a no-op.
    ///
    /// # Panics
    /// Asserts a repair slot is available, and that an existing entry for `address` carries
    /// the same `checksum` and is not `aborting` (upstream `enqueue_faulty_block`).
    pub fn repair_block(&mut self, address: u64, checksum: u128) {
        assert!(self.repair_blocks_available() > 0);
        self.verify();

        match self.faulty_blocks.entry(address) {
            Entry::Occupied(fault) => {
                assert_eq!(fault.get().checksum, checksum);
                assert_ne!(fault.get().state, FaultState::Aborting);
                // Upstream `.repair` duplicate → no-op.
            }
            Entry::Vacant(slot) => {
                slot.insert(FaultyBlock { checksum, state: FaultState::Waiting });
                self.enqueued_blocks_repair += 1;
            }
        }
        self.verify();
    }

    /// Whether the given block is queued as `waiting` — i.e. a received repair block would
    /// be durable-written (upstream `block_waiting`).
    #[must_use]
    pub fn block_waiting(&self, address: u64, checksum: u128) -> bool {
        let Some(fault) = self.faulty_blocks.get(&address) else { return false };
        fault.checksum == checksum && fault.state == FaultState::Waiting
    }

    /// Transition a waiting fault to `writing` now that the repairing write has begun
    /// (upstream `write_commence`, called by `Grid.repair_block`).
    ///
    /// # Panics
    /// Asserts the fault is present and waiting with the matching `checksum`.
    pub fn write_commence(&mut self, address: u64, checksum: u128) {
        assert!(self.block_waiting(address, checksum));
        let fault = self
            .faulty_blocks
            .get_mut(&address)
            .unwrap_or_else(|| unreachable!("block_waiting just affirmed the entry"));
        fault.state = FaultState::Writing;
        self.verify();
    }

    /// The repairing write of `address`/`checksum` completed; the fault is healed and
    /// removed (upstream `write_complete`).
    ///
    /// DEVIATION: upstream reads the block header to identify the completed write; the port
    /// receives `(address, checksum)` from the caller (the written block is still at the
    /// grid's write location in the port's plumbing).
    ///
    /// # Panics
    /// Asserts the fault is present with the matching `checksum` and in state `writing` or
    /// `aborting`.
    pub fn write_complete(&mut self, address: u64, checksum: u128) {
        let (fault_checksum, fault_state) = {
            let fault = self
                .faulty_blocks
                .get(&address)
                .unwrap_or_else(|| unreachable!("a completing repair implies a tracked fault"));
            (fault.checksum, fault.state)
        };
        assert_eq!(fault_checksum, checksum);
        assert!(
            fault_state == FaultState::Writing || fault_state == FaultState::Aborting,
            "a completing repair must have begun"
        );

        self.faulty_blocks.remove(&address);
        self.enqueued_blocks_repair -= 1;

        if fault_state == FaultState::Aborting {
            let State::CheckpointDurable { aborting } = &mut self.state else {
                unreachable!("aborting faults imply the checkpoint_durable state");
            };
            *aborting -= 1;
        }
        self.verify();
    }

    /// Cancel in-flight repairs (upstream `cancel`, called by `Grid.cancel`): a `writing`
    /// fault drops back to `waiting` because the write may not take place.
    ///
    /// # Panics
    /// Asserts no fault is `aborting` (upstream `unreachable`).
    pub fn cancel(&mut self) {
        self.verify();
        for fault in self.faulty_blocks.values_mut() {
            match fault.state {
                FaultState::Waiting => {}
                FaultState::Writing => fault.state = FaultState::Waiting,
                FaultState::Aborting => {
                    unreachable!("cancel() must not run while repairs are aborting")
                }
            }
        }
        self.verify();
    }

    /// Abort queued repairs to blocks about to be freed, now that the current checkpoint is
    /// durable (upstream `checkpoint_durable_commence`).
    ///
    /// Waiting faults to be freed are released directly; writing faults to be freed switch
    /// to `aborting` and are awaited by [`GridBlocksMissing::checkpoint_durable_complete`].
    ///
    /// # Panics
    /// Asserts the collection is in `repairing` state and the free set is opened, and that
    /// no tracked address is free.
    pub fn checkpoint_durable_commence(&mut self, free_set: &FreeSet) {
        assert_eq!(self.state, State::Repairing);
        assert!(free_set.opened());
        self.verify();

        let mut to_remove = Vec::new();
        let mut to_abort = Vec::new();
        for (&address, fault) in &self.faulty_blocks {
            assert!(!free_set.is_free(address), "faulty blocks must not be free");
            assert_ne!(fault.state, FaultState::Aborting);
            // `to_be_freed_at_checkpoint_durability` (not `is_released`): the latter also
            // covers blocks released for the *next* checkpoint.
            if free_set.to_be_freed_at_checkpoint_durability(address) {
                match fault.state {
                    FaultState::Waiting => to_remove.push(address),
                    FaultState::Writing => to_abort.push(address),
                    FaultState::Aborting => {
                        unreachable!("no fault is aborting before checkpoint durably begins")
                    }
                }
            }
        }

        for address in to_remove {
            self.faulty_blocks.remove(&address);
            self.enqueued_blocks_repair -= 1;
        }
        for address in &to_abort {
            let fault = self
                .faulty_blocks
                .get_mut(address)
                .unwrap_or_else(|| unreachable!("address was iterated as present"));
            fault.state = FaultState::Aborting;
        }

        self.state = State::CheckpointDurable { aborting: to_abort.len() as u64 };
        self.verify();
    }

    /// Returns `true` when the state≠`waiting` faults for blocks that are staged to be
    /// released have finished (upstream `checkpoint_durable_complete`). All other writes can
    /// safely complete after the checkpoint.
    ///
    /// # Panics
    /// Asserts the collection is in `checkpoint_durable` state.
    pub fn checkpoint_durable_complete(&mut self) -> bool {
        self.verify();
        let complete = match self.state {
            State::CheckpointDurable { aborting } => aborting == 0,
            State::Repairing => {
                unreachable!("checkpoint_durable_complete requires checkpoint_durable state")
            }
        };
        if complete {
            self.state = State::Repairing;
        }
        self.verify();
        complete
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // tests drive the module with unwrap like upstream tests

    use super::GridBlocksMissing;
    use crate::BlockRequest;
    use tigerbeetle_lsm::free_set::FreeSet;

    #[test]
    fn new_is_empty_with_full_repair_budget() {
        let missing = GridBlocksMissing::new(30);
        assert_eq!(missing.count(), 0);
        assert_eq!(missing.repair_blocks_available(), 30);
    }

    #[test]
    fn repair_block_tracks_and_dedups_by_address_checksum() {
        let mut missing = GridBlocksMissing::new(30);

        missing.repair_block(1, 0xAB);
        assert_eq!(missing.count(), 1);
        assert_eq!(missing.repair_blocks_available(), 29);

        // Duplicate: no-op.
        missing.repair_block(1, 0xAB);
        assert_eq!(missing.count(), 1);
        assert_eq!(missing.repair_blocks_available(), 29);

        // A different block fills the second slot.
        missing.repair_block(2, 0xCD);
        assert_eq!(missing.count(), 2);

        // Only waiting faults yield requests.
        assert_eq!(
            missing.fault_at_index(0),
            Some(BlockRequest { block_checksum: 0xAB, block_address: 1, reserved: [0; 8] })
        );
        assert_eq!(
            missing.fault_at_index(1),
            Some(BlockRequest { block_checksum: 0xCD, block_address: 2, reserved: [0; 8] })
        );
    }

    #[test]
    #[should_panic(expected = "left == right")]
    fn duplicate_address_with_different_checksum_panics() {
        let mut missing = GridBlocksMissing::new(30);
        missing.repair_block(1, 0xAB);
        missing.repair_block(1, 0xCD);
    }

    #[test]
    #[should_panic(expected = "repair_blocks_available")]
    fn repair_block_beyond_capacity_panics() {
        let mut missing = GridBlocksMissing::new(2);
        missing.repair_block(1, 0xAB);
        missing.repair_block(2, 0xCD);
        missing.repair_block(3, 0xEF);
    }

    #[test]
    fn write_commence_marks_writing_and_fault_at_index_goes_none() {
        let mut missing = GridBlocksMissing::new(30);
        missing.repair_block(1, 0xAB);

        missing.write_commence(1, 0xAB);
        assert!(!missing.block_waiting(1, 0xAB));
        assert_eq!(missing.fault_at_index(0), None);
        // Still counted against the repair budget until the write completes.
        assert_eq!(missing.count(), 1);
        assert_eq!(missing.repair_blocks_available(), 29);
    }

    #[test]
    fn write_complete_heals_and_removes_the_fault() {
        let mut missing = GridBlocksMissing::new(30);
        missing.repair_block(1, 0xAB);
        missing.write_commence(1, 0xAB);
        missing.write_complete(1, 0xAB);

        assert_eq!(missing.count(), 0);
        assert_eq!(missing.repair_blocks_available(), 30);
    }

    #[test]
    fn cancel_drops_writing_faults_back_to_waiting() {
        let mut missing = GridBlocksMissing::new(30);
        missing.repair_block(1, 0xAB);
        missing.repair_block(2, 0xCD);
        missing.write_commence(1, 0xAB);

        missing.cancel();

        assert!(missing.block_waiting(1, 0xAB), "cancel()ed write may not take place");
        assert!(missing.block_waiting(2, 0xCD));
        assert_eq!(missing.count(), 2);
    }

    fn keeper(free_set: &mut FreeSet) -> u64 {
        let reservation = free_set.reserve(1).unwrap();
        free_set.acquire(reservation).unwrap()
    }

    /// A free set mid-checkpoint-interval with one block staged for release at the
    /// next checkpoint durability: the release happens while the set is durable
    /// (landing it in `blocks_released`), then the interval is marked not-yet-durable.
    /// That is exactly the view `Grid::checkpoint_durable` passes to
    /// [`checkpoint_durable_commence`].
    fn mid_interval_free_set() -> (FreeSet, u64) {
        let mut set = FreeSet::open_empty(4096);
        let reservation = set.reserve(1).unwrap();
        let to_be_freed = set.acquire(reservation).unwrap();
        set.release(to_be_freed);
        set.mark_checkpoint_not_durable();
        (set, to_be_freed)
    }

    #[test]
    fn checkpoint_durable_releases_waiting_faults_to_be_freed() {
        let mut missing = GridBlocksMissing::new(30);
        let (mut free_set, to_be_freed) = mid_interval_free_set();
        let keeper = keeper(&mut free_set);

        missing.repair_block(to_be_freed, 0xAB);
        missing.repair_block(keeper, 0xCD);

        missing.checkpoint_durable_commence(&free_set);
        assert!(
            missing.checkpoint_durable_complete(),
            "no aborting repairs: checkpoint durable completes immediately"
        );

        // The to-be-freed waiting fault was released; the keeper survives.
        assert_eq!(missing.count(), 1);
        assert_eq!(missing.repair_blocks_available(), 29);
        assert!(missing.block_waiting(keeper, 0xCD));
    }

    #[test]
    fn checkpoint_durable_awaits_aborting_writes_of_to_be_freed_blocks() {
        let mut missing = GridBlocksMissing::new(30);
        let (free_set, to_be_freed) = mid_interval_free_set();
        missing.repair_block(to_be_freed, 0xAB);
        // The repairing write is already in flight when the checkpoint becomes durable.
        missing.write_commence(to_be_freed, 0xAB);

        missing.checkpoint_durable_commence(&free_set);
        assert!(!missing.checkpoint_durable_complete(), "aborting write still in flight");
        assert_eq!(missing.count(), 1, "the aborting fault is still tracked");

        // The aborting write completes: the checkpoint can now become durable.
        missing.write_complete(to_be_freed, 0xAB);
        assert!(missing.checkpoint_durable_complete());
        assert_eq!(missing.count(), 0);
    }

    #[test]
    fn checkpoint_durable_preserves_writing_faults_not_to_be_freed() {
        let mut missing = GridBlocksMissing::new(30);
        let (mut free_set, _to_be_freed) = mid_interval_free_set();

        let keeper = keeper(&mut free_set);
        missing.repair_block(keeper, 0xCD);
        missing.write_commence(keeper, 0xCD);

        missing.checkpoint_durable_commence(&free_set);
        assert!(
            missing.checkpoint_durable_complete(),
            "no fault is to-be-freed, so nothing aborts"
        );
        // Complete the write normally afterwards.
        missing.write_complete(keeper, 0xCD);
        assert_eq!(missing.count(), 0);
    }
}
