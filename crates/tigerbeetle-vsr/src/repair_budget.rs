//! Repair budgets: how many repair requests a replica may have inflight per remote replica,
//! and which remote replica to ask next.
//!
//! Upstream: `src/vsr/repair_budget.zig`.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use tigerbeetle_core::constants::{GRID_REPAIR_REQUEST_MAX, REPLICAS_MAX};
use tigerbeetle_core::stdx::prng::{Prng, Ratio, Reservoir, ratio};
use tigerbeetle_core::stdx::{Duration, Instant};

use crate::BlockReference;

/// Maximum inflight `get_prepare` messages per remote replica, at any point of time.
///
/// This is kept small to ensure that even if the budget to a remote replica is saturated
/// by multiple replicas, overflowing the egress `send_queue` (which leads to dropped messages)
/// on the remote replica is unlikely. For example, since the `send_queue` is currently sized
/// to 4 messages, if we were to set this limit to 4 as well, multiple repairing replicas are
/// more likely to overflow the remote replica's send queue.
const REPAIR_MESSAGES_INFLIGHT_COUNT_MAX: usize = 2;

/// Multiple of repair latency used to determine expiry duration, which is the time we wait
/// before restoring the budget for an inflight repair request if the prepare has not arrived.
const REPAIR_LATENCY_MULTIPLE_EXPIRY: u8 = 2;

/// The maximum amount of time we wait before reclaiming the budget for an inflight repair
/// request if the prepare has not arrived.
///
/// Capped at 500ms to avoid an unbounded increase in the tracked repair latency for remote
/// replicas. Specifically, helps avoid the case where a partitioned replica with missing
/// prepares gets into a cycle of requesting prepares, waiting for them to expire, and then
/// increasing the repair latency on expiry.
fn duration_expiry_max() -> Duration {
    Duration::ms(500)
}

#[derive(Clone, Copy)]
pub struct RepairBudgetOptions {
    pub replica_index: u8,
    pub replica_count: u8,
}

#[derive(Debug)]
pub struct RepairBudgetJournal {
    capacity: u32,
    available: u32,

    replica_index: u8,

    /// Tracks the prepare ops requested from each remote replica.
    ///
    /// DEVIATION: upstream uses `std.AutoArrayHashMapUnmanaged` preallocated to
    /// [`REPAIR_MESSAGES_INFLIGHT_COUNT_MAX`] entries; insertion order is never observable
    /// through this API, so `HashMap` preserves the semantics.
    replicas_requested_prepares: Vec<HashMap<u64, Instant>>,

    /// Exponential weighted moving average of the repair latency for each remote replica.
    ///
    /// Repair latency is calculated as the duration elapsed between when a prepare is requested
    /// from a remote replica, and when it is either received from the remote replica (see
    /// [`RepairBudgetJournal::increment`]), or expired (see
    /// [`RepairBudgetJournal::reap_expired_requests`]).
    replicas_repair_latency: Vec<Duration>,

    /// Probability of choosing a random replica with available budget, as opposed to one with the
    /// best repair latency with available budget.
    ///
    /// Experiments ensure that we try alternative repair routes, and avoids potential resonance
    /// wherein we keep requesting from a permanently crashed replica with the best repair latency.
    /// This is because we don't penalize the repair latency once it exceeds `duration_expiry_max`,
    /// so if a crashed replica has the best latency, it may remain that way forever.
    experiment_chance: Ratio,
}

impl RepairBudgetJournal {
    /// # Panics
    /// Panics if `replica_count > u8::MAX` slots cannot be tracked — impossible given the
    /// parameter type; retained to mirror upstream's allocation checks.
    #[must_use]
    pub fn new(options: RepairBudgetOptions) -> Self {
        // Replicas can repair from all replicas but themselves,
        // while standbys can repair from all replicas.
        let remote_replica_count =
            options.replica_count - u8::from(options.replica_index < options.replica_count);

        let mut replicas_requested_prepares = Vec::new();
        for _ in 0..options.replica_count {
            replicas_requested_prepares
                .push(HashMap::with_capacity(REPAIR_MESSAGES_INFLIGHT_COUNT_MAX));
        }

        // Initialize repair latency to 1 ms for all replicas, this gets refined as we start
        // repairing from these replicas. We choose a value lower than the the typical latency
        // between two replicas, so as to not bias replica selection when we have few measurements.
        let replicas_repair_latency = vec![Duration::ms(1); usize::from(options.replica_count)];

        Self {
            capacity: u32::from(remote_replica_count)
                * u32::try_from(REPAIR_MESSAGES_INFLIGHT_COUNT_MAX).unwrap_or(u32::MAX),
            available: u32::from(remote_replica_count)
                * u32::try_from(REPAIR_MESSAGES_INFLIGHT_COUNT_MAX).unwrap_or(u32::MAX),
            replica_index: options.replica_index,
            replicas_requested_prepares,
            replicas_repair_latency,
            experiment_chance: ratio(1, 10),
        }
    }

    /// Returns the index of the replica with the lowest repair latency, and budget availability, if
    /// one exists. Otherwise, returns `None`. For a fraction of ops (guided by
    /// `experiment_chance`), diverges from this heuristic and returns the index of a random replica
    /// with budget availability, using reservoir sampling.
    ///
    /// # Panics
    /// Panics if the capacity is zero or an invariant is violated (upstream asserts).
    pub fn decrement(&mut self, op: u64, now: Instant, prng: &mut Prng) -> Option<u8> {
        assert!(self.capacity > 0);
        // Upstream: `maybe(budget.available == 0)` — a documentation-only assertion.
        self.assert_invariants();

        let experiment = prng.chance(self.experiment_chance);
        let mut experiment_replica_index: Option<u8> = None;
        let mut reservoir = Reservoir::new();

        let mut repair_latency_min: Option<Duration> = None;
        let mut repair_latency_min_replica_index: Option<u8> = None;

        for (replica_index, requested_prepares) in
            self.replicas_requested_prepares.iter().enumerate()
        {
            let replica_index = replica_index_u8(replica_index);
            // Disallow requests to self.
            if replica_index == self.replica_index {
                continue;
            }
            // Enforce per-replica budget.
            if requested_prepares.len() == REPAIR_MESSAGES_INFLIGHT_COUNT_MAX {
                continue;
            }
            // Disallow requesting from a replica from which this op has already been requested.
            if requested_prepares.contains_key(&op) {
                continue;
            }

            let replica_repair_latency = self.replicas_repair_latency[usize::from(replica_index)];

            if repair_latency_min.is_none()
                || replica_repair_latency.ns < repair_latency_min.unwrap_or(Duration { ns: 0 }).ns
            {
                repair_latency_min = Some(replica_repair_latency);
                repair_latency_min_replica_index = Some(replica_index);
            }

            // Reservoir sampling with an arbitrarily chosen weight of 1 for each item suffices
            // our use case, as the goal is to get some degree of randomness during experiments.
            if reservoir.replace(prng, 1) {
                experiment_replica_index = Some(replica_index);
            }
        }
        assert_eq!(repair_latency_min.is_none(), repair_latency_min_replica_index.is_none());
        assert_eq!(repair_latency_min_replica_index.is_none(), experiment_replica_index.is_none());

        let replica_index =
            if experiment { experiment_replica_index } else { repair_latency_min_replica_index };

        if let Some(replica_index) = replica_index {
            assert_ne!(replica_index, self.replica_index);
            let requested_prepares =
                &mut self.replicas_requested_prepares[usize::from(replica_index)];
            assert!(requested_prepares.len() < REPAIR_MESSAGES_INFLIGHT_COUNT_MAX);
            assert!(!requested_prepares.contains_key(&op));
            requested_prepares.insert(op, now);
            self.available -= 1;
        }

        replica_index
    }

    /// Increments the budget by 1 for each replica that this prepare op has been requested from.
    /// Also refines the repair latency for each of these replicas.
    pub fn increment(&mut self, op: u64, now: Instant) {
        self.assert_invariants();

        for (replica_index, requested_prepares) in
            self.replicas_requested_prepares.iter_mut().enumerate()
        {
            if let Some(requested_at) = requested_prepares.remove(&op) {
                self.available += 1;

                // We have no information about the replica that sent this prepare, as the message
                // header stores the index of the primary processed that prepare. Consequently, we
                // refine repair latency for all replicas that this prepare op was requested from.
                // This would lead to some inaccuracy in the latency measurement, but is acceptable
                // since the scenario where a prepare has been requested from multiple replicas is
                // rare in practice. The more common scenario is that we have a large number of
                // prepares missing (for e.g. after state sync, or if a lagging replica transitions
                // to a new checkpoint), in which case we request a unique op from each replica.
                let latency = &mut self.replicas_repair_latency[replica_index];
                *latency = ewma_add_duration(*latency, requested_at.elapsed(now));
            }
        }

        self.assert_invariants();
    }

    pub fn refill(&mut self) {
        self.assert_invariants();

        for requested_prepares in &mut self.replicas_requested_prepares {
            requested_prepares.clear();
        }
        self.available = self.capacity;

        self.assert_invariants();
    }

    /// Iterates through the inflight requests across all remote replicas, and reclaims the budget
    /// for expired requests. Penalizes the replicas for which some expired requests were found,
    /// adding the duration spent waiting for the expired requests to their repair latency.
    ///
    /// Expiry provides resilience to network faults, by ensuring that a dropped packet or the
    /// remote replica crashing doesn't cause an op to get stuck in the queue for a remote replica.
    /// We avoid spurious expiry due to transient network hiccups like increased latency by waiting
    /// for twice the measured repair latency.
    pub fn reap_expired_requests(&mut self, now: Instant) {
        self.assert_invariants();

        for (replica_index, requested_prepares) in
            self.replicas_requested_prepares.iter_mut().enumerate()
        {
            let duration_expiry_ns = core::cmp::min(
                u64::from(REPAIR_LATENCY_MULTIPLE_EXPIRY)
                    * self.replicas_repair_latency[replica_index].ns,
                duration_expiry_max().ns,
            );

            let expired_ops: Vec<u64> = requested_prepares
                .iter()
                .filter(|(_, requested_at)| requested_at.elapsed(now).ns > duration_expiry_ns)
                .map(|(op, _)| *op)
                .collect();

            for op in expired_ops {
                if let Some(requested_at) = requested_prepares.remove(&op) {
                    let latency = &mut self.replicas_repair_latency[replica_index];
                    *latency = ewma_add_duration(*latency, requested_at.elapsed(now));
                    self.available += 1;
                }
            }
        }

        self.assert_invariants();
    }

    fn assert_invariants(&self) {
        assert!(self.available <= self.capacity);
        if usize::from(self.replica_index) < self.replicas_requested_prepares.len() {
            assert!(
                self.replicas_requested_prepares[usize::from(self.replica_index)].is_empty(),
                "own slot must stay empty"
            );
        }

        let mut requested_prepares_count: u32 = 0;
        for requested_prepares in &self.replicas_requested_prepares {
            requested_prepares_count += u32::try_from(requested_prepares.len()).unwrap_or(u32::MAX);
        }
        assert_eq!(self.capacity - self.available, requested_prepares_count);
    }
}

fn ewma_add_duration(old: Duration, new: Duration) -> Duration {
    Duration { ns: (old.ns * 4 + new.ns) / 5 }
}

/// The amount of time we wait before restoring the budget for an
/// inflight repair request if the block has not arrived.
fn grid_duration_expiry() -> Duration {
    Duration::ms(250)
}

/// The amount of time we wait before re-requesting a block
/// which has not yet arrived.
fn grid_duration_retry() -> Duration {
    Duration::ms(100)
}

/// Maximum blocks that can be requested per remote replica.
///
/// We use a small number to ensure that even if the budget to a
/// remote replica is saturated by multiple replicas, overflowing
/// the egress `send_queue` (which leads to dropped messages, and
/// wasted network & storage IO) on the remote replica is unlikely.
/// The +1 allows us to send a full `get_blocks` even when
/// all but one request has been responded to.
const REPLICA_BLOCKS_REQUESTED_MAX: usize = GRID_REPAIR_REQUEST_MAX as usize + 1;

#[derive(Debug)]
pub struct RepairBudgetGrid {
    capacity: u32,
    available: u32,
    replica_index: u8,

    /// Tracks the blocks requested from each remote replica.
    ///
    /// DEVIATION: upstream uses `std.AutoArrayHashMapUnmanaged`; insertion order is never
    /// observable through this API, so `HashMap` preserves the semantics.
    replicas_requested_blocks: Vec<HashMap<BlockReference, Instant>>,
}

impl RepairBudgetGrid {
    #[must_use]
    pub fn new(options: RepairBudgetOptions) -> Self {
        // Replicas can repair from all replicas but themselves,
        // while standbys can repair from all replicas.
        let remote_replica_count =
            options.replica_count - u8::from(options.replica_index < options.replica_count);

        let mut replicas_requested_blocks = Vec::new();
        for _ in 0..options.replica_count {
            replicas_requested_blocks.push(HashMap::with_capacity(REPLICA_BLOCKS_REQUESTED_MAX));
        }

        Self {
            capacity: u32::from(remote_replica_count)
                * u32::try_from(REPLICA_BLOCKS_REQUESTED_MAX).unwrap_or(u32::MAX),
            available: u32::from(remote_replica_count)
                * u32::try_from(REPLICA_BLOCKS_REQUESTED_MAX).unwrap_or(u32::MAX),
            replica_index: options.replica_index,
            replicas_requested_blocks,
        }
    }

    fn assert_invariants(&self) {
        assert!(self.available <= self.capacity);

        if usize::from(self.replica_index) < self.replicas_requested_blocks.len() {
            assert!(
                self.replicas_requested_blocks[usize::from(self.replica_index)].is_empty(),
                "own slot must stay empty"
            );
        }

        let mut requested_blocks_count: u32 = 0;
        for requested_blocks in &self.replicas_requested_blocks {
            requested_blocks_count += u32::try_from(requested_blocks.len()).unwrap_or(u32::MAX);
        }

        assert_eq!(self.available + requested_blocks_count, self.capacity);
    }

    /// Returns the index of a random replica (shuffled by `prng`) with enough budget to receive
    /// a full `get_blocks`, or `None`.
    ///
    /// # Panics
    /// Panics if more than [`REPLICAS_MAX`] replicas are tracked (upstream uses a fixed-size
    /// scratch array).
    #[must_use]
    pub fn next_destination(&mut self, prng: &mut Prng) -> Option<u8> {
        self.assert_invariants();

        let replica_count = self.replicas_requested_blocks.len();
        assert!(replica_count <= REPLICAS_MAX);

        let mut replica_indexes = [0u8; REPLICAS_MAX];
        for (i, replica) in replica_indexes.iter_mut().enumerate().take(replica_count) {
            *replica = replica_index_u8(i);
        }
        prng.shuffle(&mut replica_indexes[..replica_count]);

        replica_indexes[..replica_count].iter().copied().find(|&replica_index| {
            replica_index != self.replica_index
                && self.budget_available(replica_index) >= u32::from(GRID_REPAIR_REQUEST_MAX)
        })
    }

    /// # Panics
    /// Panics if `replica_index` equals our own index (upstream asserts).
    #[must_use]
    pub fn budget_available(&self, replica_index: u8) -> u32 {
        self.assert_invariants();

        assert_ne!(self.replica_index, replica_index);

        let replica_requested_blocks = &self.replicas_requested_blocks[usize::from(replica_index)];

        u32::try_from(REPLICA_BLOCKS_REQUESTED_MAX - replica_requested_blocks.len())
            .unwrap_or(u32::MAX)
    }

    /// # Panics
    /// Panics if the target replica's budget is exhausted, the block address is zero, or
    /// `replica_index` is our own index (upstream asserts).
    pub fn decrement(
        &mut self,
        block_identifier: BlockReference,
        replica_index: u8,
        now: Instant,
    ) -> bool {
        self.assert_invariants();

        assert!(self.available > 0);
        assert!(block_identifier.address > 0);
        assert_ne!(replica_index, self.replica_index);

        let mut duration_since_requested_min: Option<Duration> = None;

        for requested_blocks in &self.replicas_requested_blocks {
            if let Some(&requested_at) = requested_blocks.get(&block_identifier) {
                let duration_since_requested = requested_at.elapsed(now);
                if duration_since_requested_min.is_none()
                    || duration_since_requested.ns
                        < duration_since_requested_min.unwrap_or(Duration { ns: 0 }).ns
                {
                    duration_since_requested_min = Some(duration_since_requested);
                }
            }
        }

        if let Some(duration) = duration_since_requested_min
            && duration.ns < grid_duration_retry().ns
        {
            return false;
        }

        let replica_requested_blocks =
            &mut self.replicas_requested_blocks[usize::from(replica_index)];

        assert!((*replica_requested_blocks).len() < REPLICA_BLOCKS_REQUESTED_MAX);

        match replica_requested_blocks.entry(block_identifier) {
            Entry::Occupied(mut occupied) => {
                occupied.insert(now);
            }
            Entry::Vacant(vacant) => {
                vacant.insert(now);
                self.available -= 1;
            }
        }

        true
    }

    pub fn increment(&mut self, block_identifier: BlockReference) {
        self.assert_invariants();

        // We have no information about the replica that sent
        // this block, as storing each replica's index in the
        // block header would make storage non-deterministic.
        // Consequently, we increase the budget for all replicas
        // this block was requested from. This is safe to do as
        // we only invoke `increment` *once* -- when the replica
        // either uses the block to serve a read that's waiting
        // on this block, or a repair write (see `on_block`).
        for requested_blocks in &mut self.replicas_requested_blocks {
            if requested_blocks.remove(&block_identifier).is_some() {
                self.available += 1;
            }
        }

        self.assert_invariants();
    }

    pub fn refill(&mut self) {
        self.assert_invariants();

        self.available = self.capacity;

        for requested_blocks in &mut self.replicas_requested_blocks {
            requested_blocks.clear();
        }

        self.assert_invariants();
    }

    pub fn reap_expired_requests(&mut self, now: Instant) {
        self.assert_invariants();

        for requested_blocks in &mut self.replicas_requested_blocks {
            let expired_blocks: Vec<BlockReference> = requested_blocks
                .iter()
                .filter(|(_, requested_at)| {
                    requested_at.elapsed(now).ns > grid_duration_expiry().ns
                })
                .map(|(block, _)| *block)
                .collect();

            for block in expired_blocks {
                if requested_blocks.remove(&block).is_some() {
                    self.available += 1;
                }
            }
        }

        self.assert_invariants();
    }
}

/// Cast helper for replica indices: every vector length originates from a `u8`
/// `replica_count`, so the cast cannot truncate.
#[must_use]
fn replica_index_u8(replica_index: usize) -> u8 {
    match u8::try_from(replica_index) {
        Ok(replica_index) => replica_index,
        Err(_) => unreachable!("replica counts originate from a u8 parameter"),
    }
}

// TODO(port): upstream has no unit tests in repair_budget.zig; these smoke tests pin down the
// budget lifecycle until the replica-level tests exercise them end-to-end.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::{
        REPLICA_BLOCKS_REQUESTED_MAX, RepairBudgetGrid, RepairBudgetJournal, RepairBudgetOptions,
    };
    use crate::BlockReference;
    use tigerbeetle_core::constants::{GRID_REPAIR_REQUEST_MAX, REPLICAS_MAX};
    use tigerbeetle_core::stdx::prng::Prng;
    use tigerbeetle_core::stdx::{Duration, Instant};

    #[test]
    fn repair_budget_journal_exhausts_and_restores_budget() {
        let mut prng = Prng::from_seed(42);
        let now = Instant { ns: 1_000 };

        let mut budget =
            RepairBudgetJournal::new(RepairBudgetOptions { replica_index: 0, replica_count: 4 });
        assert_eq!(budget.capacity, 6); // 3 remote replicas x 2 inflight max.

        // Exhaust the whole budget with distinct ops.
        let mut granted = Vec::new();
        for op in 1..=6 {
            if let Some(replica_index) = budget.decrement(op, now, &mut prng) {
                assert_ne!(replica_index, 0);
                granted.push((op, replica_index));
            }
        }
        assert_eq!(granted.len(), 6);
        assert_eq!(budget.available, 0);

        // No budget left, and each replica's queue is full.
        assert_eq!(budget.decrement(7, now, &mut prng), None);

        // A repeated request for an already-requested op is refused even mid-cycle.
        let (requested_op, _) = granted[0];
        assert_eq!(budget.decrement(requested_op, now, &mut prng), None);

        // Receiving one prepare restores one unit and refines that replica's latency.
        budget.increment(requested_op, now.add(Duration::ms(3)));
        assert_eq!(budget.available, 1);

        // Refill clears everything.
        budget.refill();
        assert_eq!(budget.available, budget.capacity);
        assert!(budget.decrement(requested_op, now, &mut prng).is_some());
    }

    #[test]
    fn repair_budget_journal_reaps_expired_requests() {
        let mut prng = Prng::from_seed(43);
        let now = Instant { ns: 0 };

        let mut budget =
            RepairBudgetJournal::new(RepairBudgetOptions { replica_index: 0, replica_count: 2 });

        // Initial latency is 1ms => expiry is min(2 * 1ms, 500ms) = 2ms.
        assert!(
            budget.decrement(100, now, &mut prng).is_some(),
            "the only remote replica must be eligible"
        );
        assert_eq!(budget.available, 1); // Capacity is 2 for a single remote replica.

        // Not yet expired after 2ms exactly (expiry requires strictly greater).
        budget.reap_expired_requests(now.add(Duration::ms(2)));
        assert_eq!(budget.available, 1);

        // Expired after 3ms: budget restored, latency penalized toward 3ms.
        budget.reap_expired_requests(now.add(Duration::ms(3)));
        assert_eq!(budget.available, 2);
        assert_eq!(budget.replicas_repair_latency[1], Duration { ns: 1_400_000 });
    }

    #[test]
    fn repair_budget_grid_tracks_blocks_per_replica() {
        let mut prng = Prng::from_seed(44);
        let now = Instant { ns: 10_000 };

        let mut budget = RepairBudgetGrid::new(RepairBudgetOptions {
            replica_index: 0,
            replica_count: u8::try_from(REPLICAS_MAX).unwrap_or(u8::MAX),
        });
        assert_eq!(
            budget.capacity,
            (u32::try_from(REPLICAS_MAX).unwrap_or(u32::MAX) - 1)
                * u32::try_from(REPLICA_BLOCKS_REQUESTED_MAX).unwrap_or(u32::MAX)
        );

        // Every destination is a valid remote replica with a full get_blocks worth of budget.
        let first = budget
            .next_destination(&mut prng)
            .unwrap_or_else(|| unreachable!("a fresh budget always has a destination"));
        assert_ne!(first, 0);
        assert!(budget.budget_available(first) >= u32::from(GRID_REPAIR_REQUEST_MAX));

        let block = BlockReference { checksum: 77, address: 9 };

        // Send a full get_blocks worth of requests to the chosen replica.
        for i in 0..GRID_REPAIR_REQUEST_MAX {
            assert!(budget.decrement(block_of(i), first, now));
        }
        // One spare slot remains (+1 headroom over a full get_blocks).
        assert_eq!(budget.budget_available(first), 1);

        // An immediate re-request within the retry window is rejected without consuming budget.
        assert!(!budget.decrement(block_of(0), first, now.add(Duration::ms(50))));
        assert_eq!(budget.budget_available(first), 1);

        // After the retry window elapses the block may be re-requested in place.
        assert!(budget.decrement(block_of(0), first, now.add(Duration::ms(150))));
        assert_eq!(budget.budget_available(first), 1);

        // The spare slot takes one more block (the map is now at capacity).
        assert!(budget.decrement(block, first, now));
        assert_eq!(budget.budget_available(first), 0);

        // Block arrival restores budget across every replica it was requested from.
        budget.increment(block);
        assert_eq!(budget.budget_available(first), 1);

        // Expiry reclaims everything that was requested but never answered.
        budget.refill();
        for i in 0..u16::try_from(REPLICA_BLOCKS_REQUESTED_MAX).unwrap_or(u16::MAX) {
            assert!(budget.decrement(block_of(i), first, now));
        }
        budget.reap_expired_requests(now.add(Duration::ms(251)));
        assert_eq!(
            budget.budget_available(first),
            u32::try_from(REPLICA_BLOCKS_REQUESTED_MAX).unwrap_or(u32::MAX)
        );
    }

    fn block_of(i: u16) -> BlockReference {
        BlockReference { checksum: u128::from(i), address: u64::from(i) + 1 }
    }
}
