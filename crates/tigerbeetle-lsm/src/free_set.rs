//! The 0 address is reserved for usage as a sentinel and will never be returned by
//! [`FreeSet::acquire`].
//!
//! Concurrent callers must reserve free blocks before acquiring them to ensure that
//! acquisition order is deterministic despite concurrent jobs acquiring blocks in
//! nondeterministic order.
//!
//! The reservation lifecycle is:
//!
//!   1. Reserve: In deterministic order, each job (e.g. compaction) calls [`FreeSet::reserve`]
//!      to reserve the upper bound of blocks that it may need to acquire to complete.
//!   2. Acquire: The jobs run concurrently. Each job acquires blocks only from its respective
//!      reservation (via [`FreeSet::acquire`]).
//!   3. Forfeit: When a job finishes, it calls [`FreeSet::forfeit`] to drop its reservation.
//!   4. Done: When all pending reservations are forfeited, the reserved (but unacquired) space
//!      is reclaimed.
//!
//! Port of `src/vsr/free_set.zig`. Placed in the `tigerbeetle-lsm` crate because it depends only
//! on core primitives and is consumed by compaction; `tigerbeetle-vsr` (the grid) may depend on
//! this crate per the bottom-up dependency rule.

// Upstream FreeSet is entirely `usize`-based; block *addresses* are `u64` on the wire. The
// conversions below mirror upstream's declared types and are bounded by blocks_count_max().
#![allow(clippy::cast_possible_truncation)]

use tigerbeetle_core::constants;
use tigerbeetle_core::ewah;
use tigerbeetle_core::stdx::bitset::{BitKind, BitSet, Direction};

/// This is logically a range of addresses within the FreeSet, but its actual fields are block
/// indexes for ease of calculation.
///
/// A reservation covers a range of both free and acquired blocks — when it is first created,
/// it is guaranteed to cover exactly as many free blocks as were requested by
/// [`FreeSet::reserve`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    block_base: usize,
    block_count: usize,
    /// An identifier for each reservation cycle, to verify that old reservations are not reused.
    session: usize,
}

// Each shard is 8 cache lines because the CPU line fill buffer can fetch 10 lines in parallel.
// And 8 is fast for division when computing the shard of a block.
// Since the shard is scanned sequentially, the prefetching amortizes the cost of the single
// cache miss. It also reduces the size of the index.
//
// e.g. 10TiB disk ÷ 64KiB/block ÷ 512*8 blocks/shard ÷ 8 shards/byte = 5120B index
const SHARD_CACHE_LINES: usize = 8;

/// Blocks per shard (`FreeSet.shard_bits`).
pub const SHARD_BITS: usize = SHARD_CACHE_LINES * constants::CACHE_LINE_SIZE * 8;

const _: () = assert!(SHARD_BITS == 4096);

/// Temporarily holds blocks released prior durability of the current checkpoint, to be freed
/// when the next checkpoint becomes durable. These blocks are moved to `blocks_released` once
/// the current checkpoint becomes durable.
///
/// DEVIATION: upstream uses an insertion-ordered hash map (`AutoArrayHashMapUnmanaged(u64,
/// void)`); a `Vec<u64>` preserves the same key order with linear-time membership checks,
/// which suffices for the sizes and access patterns involved.
#[derive(Clone, Debug, Default)]
struct ReleasedPriorCheckpointDurability {
    keys: Vec<u64>,
    capacity: usize,
}

impl ReleasedPriorCheckpointDurability {
    fn new(capacity: usize) -> Self {
        Self { keys: Vec::with_capacity(capacity), capacity }
    }

    fn count(&self) -> usize {
        self.keys.len()
    }

    /// Upstream `putAssumeCapacity`.
    ///
    /// # Panics
    /// Panics if the key is already present or the preallocated capacity is exhausted.
    fn put_assume_capacity(&mut self, key: u64) {
        assert!(!self.contains(key));
        assert!(self.keys.len() < self.capacity);
        self.keys.push(key);
    }

    fn contains(&self, key: u64) -> bool {
        self.keys.contains(&key)
    }

    /// Upstream `pop()` — removes the most recently inserted entry.
    fn pop(&mut self) -> Option<u64> {
        self.keys.pop()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReservationState {
    Reserving,
    Forfeiting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitsetKind {
    BlocksAcquired,
    BlocksReleased,
}

/// The free set tracks which grid blocks are acquired and which are free.
///
/// Free set is stored in the grid (see `CheckpointTrailer`) and is not available until the
/// relevant blocks are fetched from disk (or other replicas) and decoded.
///
/// Without the free set, only blocks belonging to the free set might be read and no blocks can
/// be written.
#[derive(Clone, Debug)]
pub struct FreeSet {
    // Free set is stored in the grid and is not available until opened:
    opened: bool,

    /// Whether the current checkpoint is durable.
    checkpoint_durable: bool,

    /// If a shard has any free blocks, the corresponding index bit is zero.
    /// If a shard has no free blocks, the corresponding index bit is one.
    index: BitSet,

    /// The maximum number of blocks the free set is allowed to reserve
    /// (driven by --limit-storage).
    blocks_count_limit: u64,

    /// Set bits indicate acquired blocks; unset bits indicate free blocks.
    blocks_acquired: BitSet,

    /// Set bits indicate blocks released in the current checkpoint, to be freed when the next
    /// checkpoint becomes durable.
    blocks_released: BitSet,

    blocks_released_prior_checkpoint_durability: ReleasedPriorCheckpointDurability,

    /// The number of blocks that are reserved, counting both acquired and free blocks
    /// from the start of `blocks_acquired`.
    /// Alternatively, the index of the first non-reserved block in `blocks_acquired`.
    reservation_blocks: usize,

    /// The number of active reservations.
    reservation_count: usize,

    /// Verify that when the caller transitions from creating reservations to forfeiting them,
    /// all reservations must be forfeited before additional reservations are made.
    reservation_state: ReservationState,

    /// Verifies that reservations are not allocated from or forfeited when they should not be.
    reservation_session: usize,
}

impl FreeSet {
    /// # Panics
    /// Panics if `grid_size_limit` does not yield a whole number of shards.
    #[must_use]
    pub fn new(
        grid_size_limit: usize,
        blocks_released_prior_checkpoint_durability_max: usize,
    ) -> Self {
        let blocks_count = Self::block_count_max(grid_size_limit);
        assert_eq!(blocks_count % SHARD_BITS, 0);
        assert_eq!(blocks_count % tigerbeetle_core::stdx::bitset::WORD_BITS, 0);

        // Every block bit is covered by exactly one index bit.
        let shards_count = blocks_count / SHARD_BITS;
        let index = BitSet::new_empty(shards_count);

        let blocks_acquired = BitSet::new_empty(blocks_count);
        let blocks_released = BitSet::new_empty(blocks_count);

        let released_prior_checkpoint_durability = ReleasedPriorCheckpointDurability::new(
            blocks_released_prior_checkpoint_durability_max
                // `blocks_released` and `blocks_acquired` encoded in the CheckpointTrailer are
                // released at checkpoint (see `mark_checkpoint_not_durable` in grid.zig).
                + 2 * checkpoint_trailer_block_count_for_trailer_size(
                    ewah::encode_size_max(blocks_count / tigerbeetle_core::stdx::bitset::WORD_BITS),
                ),
        );

        assert_eq!(index.count(), 0);
        assert_eq!(blocks_acquired.count(), 0);
        assert_eq!(blocks_released.count(), 0);
        assert_eq!(released_prior_checkpoint_durability.count(), 0);

        Self {
            opened: false,
            checkpoint_durable: false,
            index,
            blocks_count_limit: (grid_size_limit / constants::BLOCK_SIZE) as u64,
            blocks_acquired,
            blocks_released,
            blocks_released_prior_checkpoint_durability: released_prior_checkpoint_durability,
            reservation_blocks: 0,
            reservation_count: 0,
            reservation_state: ReservationState::Reserving,
            reservation_session: 1,
        }
    }

    /// # Panics
    /// Panics if any bitset invariant is violated after clearing.
    pub fn reset(&mut self) {
        self.index = BitSet::new_empty(self.index.len());
        self.blocks_acquired = BitSet::new_empty(self.blocks_acquired.len());
        self.blocks_released = BitSet::new_empty(self.blocks_released.len());
        self.blocks_released_prior_checkpoint_durability.keys.clear();

        self.opened = false;
        self.checkpoint_durable = false;
        self.reservation_blocks = 0;
        self.reservation_count = 0;
        self.reservation_state = ReservationState::Reserving;
        self.reservation_session = self.reservation_session.wrapping_add(1);

        assert_eq!(self.index.count(), 0);
        assert_eq!(self.blocks_acquired.count(), 0);
        assert_eq!(self.blocks_released.count(), 0);
        assert_eq!(self.blocks_released_prior_checkpoint_durability.count(), 0);

        assert!(!self.opened);
    }

    /// Opens a free set. Needs two inputs:
    ///
    ///   - the byte buffers with the ewah-encoded acquired and released bitsets,
    ///   - the list of block addresses used to store both the encoded bitsets in the grid.
    ///
    /// Block addresses themselves are not a part of the encoded bitset for acquired blocks,
    /// see CheckpointTrailer for details.
    ///
    /// # Panics
    /// Panics if the set is already open or the inputs disagree about emptiness.
    pub fn open(
        &mut self,
        encoded_blocks_acquired: &[&[u8]],
        encoded_blocks_released: &[&[u8]],
        free_set_block_addresses_blocks_acquired: &[u64],
        free_set_block_addresses_blocks_released: &[u64],
    ) {
        assert!(!self.opened);
        let encoded_empty =
            encoded_blocks_acquired.is_empty() && encoded_blocks_released.is_empty();
        let addresses_empty = free_set_block_addresses_blocks_acquired.is_empty()
            && free_set_block_addresses_blocks_released.is_empty();
        assert_eq!(encoded_empty, addresses_empty);
        self.decode(BitsetKind::BlocksAcquired, encoded_blocks_acquired);
        self.decode(BitsetKind::BlocksReleased, encoded_blocks_released);
        self.mark_released(free_set_block_addresses_blocks_acquired);
        self.mark_released(free_set_block_addresses_blocks_released);
        self.opened = true;
    }

    /// A shortcut to initialize an empty free set for tests.
    ///
    /// # Panics
    /// Panics if `blocks_count` is not a multiple of [`SHARD_BITS`].
    #[must_use]
    pub fn init_empty(blocks_count: usize) -> Self {
        Self::new(blocks_count * constants::BLOCK_SIZE, 0)
    }

    /// A shortcut to initialize and open an empty free set for tests.
    ///
    /// # Panics
    /// Panics if `blocks_count` is not a multiple of [`SHARD_BITS`] or the opened set is not
    /// entirely free.
    #[must_use]
    pub fn open_empty(blocks_count: usize) -> Self {
        let mut set = Self::init_empty(blocks_count);

        set.open(&[], &[], &[], &[]);
        // Mark checkpoint as durable so tests use blocks_released for block releases.
        // blocks_released_prior_checkpoint_durable is required to ensure correctness across
        // multiple replicas, while tests check the following flows in a single process:
        // * Block acquisition-release
        // * Bitset encoding-decoding
        set.checkpoint_durable = true;

        assert_eq!(set.count_free(), blocks_count);
        assert_eq!(set.count_released(), 0);
        set
    }

    fn verify_index(&self) {
        for shard in 0..self.index.len() {
            assert_eq!(self.find_free_block_in_shard(shard).is_none(), self.index.get(shard));
        }
    }

    /// Returns the number of active reservations.
    ///
    /// # Panics
    /// Panics if the free set has not been opened.
    #[must_use]
    pub fn count_reservations(&self) -> usize {
        assert!(self.opened);
        self.reservation_count
    }

    /// Returns the number of free blocks.
    ///
    /// # Panics
    /// Panics if the free set has not been opened.
    #[must_use]
    pub fn count_free(&self) -> usize {
        assert!(self.opened);
        self.blocks_acquired.capacity() - self.blocks_acquired.count()
    }

    /// Returns the number of acquired blocks.
    ///
    /// # Panics
    /// Panics if the free set has not been opened.
    #[must_use]
    pub fn count_acquired(&self) -> usize {
        assert!(self.opened);
        self.blocks_acquired.count()
    }

    /// Returns the number of released blocks.
    ///
    /// # Panics
    /// Panics if the free set has not been opened.
    #[must_use]
    pub fn count_released(&self) -> usize {
        assert!(self.opened);
        self.blocks_released.count() + self.blocks_released_prior_checkpoint_durability.count()
    }

    /// Returns the address of the highest acquired block.
    ///
    /// # Panics
    /// Panics if the free set has not been opened.
    #[must_use]
    pub fn highest_address_acquired(&self) -> Option<u64> {
        assert!(self.opened);
        if let Some(block) = self.blocks_acquired.iter(BitKind::Set, Direction::Reverse).next() {
            Some(block as u64 + 1)
        } else {
            // All blocks are free.
            assert_eq!(self.blocks_acquired.count(), 0);
            None
        }
    }

    /// Returns the address of the highest released block.
    ///
    /// # Panics
    /// Panics if the free set has not been opened.
    #[must_use]
    pub fn highest_address_released(&self) -> Option<u64> {
        assert!(self.opened);
        if let Some(block) = self.blocks_released.iter(BitKind::Set, Direction::Reverse).next() {
            Some(block as u64 + 1)
        } else {
            assert_eq!(self.count_released(), 0);
            None
        }
    }

    /// Reserve `reserve_count` free blocks. The blocks are not acquired yet.
    ///
    /// Invariants:
    ///
    ///   - If a reservation is returned, it covers exactly `reserve_count` free blocks, along
    ///     with any interleaved already-acquired blocks.
    ///   - Active reservations are exclusive (i.e. disjoint).
    ///     (A reservation is active until [`Self::forfeit`] is called.)
    ///
    /// Returns `None` if there are not enough blocks free and vacant.
    /// Returns a reservation which can be used with [`Self::acquire`]:
    /// - The caller should consider the returned Reservation as opaque and immutable.
    /// - Each [`Self::reserve`] call which returns a reservation must correspond to exactly one
    ///   [`Self::forfeit`] call.
    ///
    /// # Panics
    /// Panics on protocol violations (reserving while forfeiting, zero-count).
    #[must_use]
    pub fn reserve(&mut self, reserve_count: usize) -> Option<Reservation> {
        assert!(self.opened);
        assert_eq!(self.reservation_state, ReservationState::Reserving);
        assert!(reserve_count > 0);

        let shard_start = find_bit(
            &self.index,
            self.reservation_blocks / SHARD_BITS,
            self.index.len(),
            BitKind::Unset,
        )?;

        // The reservation may cover (and ignore) already-acquired blocks due to fragmentation.
        let mut block = std::cmp::max(shard_start * SHARD_BITS, self.reservation_blocks);
        for _ in 0..reserve_count {
            block =
                find_bit(&self.blocks_acquired, block, self.blocks_acquired.len(), BitKind::Unset)?
                    + 1;

            // The free block from the `blocks_acquired` bit set may be past the total number of
            // blocks that this free set is allowed to acquire (see `block_count_max`).
            if block as u64 > self.blocks_count_limit {
                return None;
            }
        }
        let block_base = self.reservation_blocks;
        let block_count = block - self.reservation_blocks;
        self.reservation_blocks += block_count;
        self.reservation_count += 1;

        Some(Reservation { block_base, block_count, session: self.reservation_session })
    }

    /// After invoking `forfeit()`, the reservation must never be used again.
    ///
    /// # Panics
    /// Panics if the reservation belongs to an old session.
    pub fn forfeit(&mut self, reservation: Reservation) {
        assert!(self.opened);
        assert_eq!(self.reservation_session, reservation.session);

        self.reservation_count -= 1;
        if self.reservation_count == 0 {
            // All reservations have been dropped.
            self.reservation_blocks = 0;
            self.reservation_session = self.reservation_session.wrapping_add(1);
            self.reservation_state = ReservationState::Reserving;
        } else {
            self.reservation_state = ReservationState::Forfeiting;
        }
    }

    /// Marks a free block from the reservation as allocated, and returns the address.
    /// The reservation must not have been forfeited yet.
    /// The reservation must belong to the current cycle of reservations.
    ///
    /// Invariants:
    ///
    ///   - An acquired block cannot be acquired again until it has been released and the release
    ///     has been checkpointed.
    ///
    /// Returns `None` if no free block is available in the reservation.
    ///
    /// # Panics
    /// Panics on protocol violations (stale session, out-of-bounds reservation).
    pub fn acquire(&mut self, reservation: Reservation) -> Option<u64> {
        assert!(self.opened);
        assert!(self.reservation_count > 0);
        assert!(reservation.block_count > 0);
        assert!(reservation.block_base < self.reservation_blocks);
        assert!(reservation.block_base + reservation.block_count <= self.reservation_blocks);
        assert_eq!(reservation.session, self.reservation_session);

        let shard_start = find_bit(
            &self.index,
            reservation.block_base / SHARD_BITS,
            (reservation.block_base + reservation.block_count).div_ceil(SHARD_BITS),
            BitKind::Unset,
        )?;
        assert!(!self.index.get(shard_start));

        let reservation_start = std::cmp::max(shard_start * SHARD_BITS, reservation.block_base);
        let reservation_end = reservation.block_base + reservation.block_count;
        let block =
            find_bit(&self.blocks_acquired, reservation_start, reservation_end, BitKind::Unset)?;
        assert!(block >= reservation.block_base);
        assert!(block <= reservation.block_base + reservation.block_count);
        assert!(!self.blocks_acquired.get(block));
        assert!(!self.blocks_released.get(block));
        assert!(!self.blocks_released_prior_checkpoint_durability.contains(block as u64));

        // Even if "shard_start" has free blocks, we might acquire our block from a later shard.
        // (This is possible because our reservation begins part-way through the shard.)
        let shard = block / SHARD_BITS;
        assert!(shard >= shard_start);

        self.blocks_acquired.set(block);
        // Update the index when every block in the shard is acquired.
        if self.find_free_block_in_shard(shard).is_none() {
            self.index.set(shard);
        }
        Some(block as u64 + 1)
    }

    fn find_free_block_in_shard(&self, shard: usize) -> Option<usize> {
        let shard_start = shard * SHARD_BITS;
        let shard_end = shard_start + SHARD_BITS;
        assert!(shard_start < self.blocks_acquired.len());

        find_bit(&self.blocks_acquired, shard_start, shard_end, BitKind::Unset)
    }

    #[must_use]
    pub fn is_free(&self, address: u64) -> bool {
        if self.opened {
            let block = address - 1;
            !self.blocks_acquired.get(block as usize)
        } else {
            // When the free set is not open, conservatively assume that the block is acquired.
            //
            // This path is hit only when the replica opens the free set, reading its blocks from
            // the grid.
            false
        }
    }

    /// # Panics
    /// Panics if the free set has not been opened.
    #[must_use]
    pub fn is_released(&self, address: u64) -> bool {
        assert!(self.opened);
        let block = address - 1;
        self.blocks_released_prior_checkpoint_durability.contains(block)
            || self.blocks_released.get(block as usize)
    }

    /// Returns `true` if the block at the given address would be freed when the current
    /// checkpoint becomes durable (when checkpoint_durable is set to `true`).
    ///
    /// Calling this function is only valid while the current checkpoint is not durable. During
    /// this period, blocks are marked as released in
    /// `blocks_released_prior_checkpoint_durability`; `blocks_released` remains unchanged and
    /// contains blocks released during the previous checkpoint interval.
    ///
    /// # Panics
    /// Panics unless the set is open and the current checkpoint is *not* durable.
    #[must_use]
    pub fn to_be_freed_at_checkpoint_durability(&self, address: u64) -> bool {
        let block = address - 1;

        assert!(self.opened);
        assert!(!self.checkpoint_durable);

        // Block address must be acquired, but is not necessarily released.
        assert!(self.blocks_acquired.get(block as usize));
        assert!(
            !self.blocks_released.get(block as usize)
                || !self.blocks_released_prior_checkpoint_durability.contains(block)
        );

        self.blocks_released.get(block as usize)
    }

    /// Leave the address acquired for now, but free it when the next checkpoint becomes durable.
    /// This ensures that it will not be overwritten during the current checkpoint — the block may
    /// still be needed if we crash and recover from the current checkpoint.
    /// (TODO) If the block was created since the last checkpoint then it's safe to free
    ///        immediately. This may reduce space amplification, especially for smaller datasets.
    ///        (Note: This must be careful not to release while any reservations are held
    ///        to avoid making the reservation's acquire()s nondeterministic).
    ///
    /// # Panics
    /// Panics if the address was not acquired-and-unreleased (upstream asserts the same).
    pub fn release(&mut self, address: u64) {
        assert!(self.opened);

        let block = address - 1;
        assert!(self.blocks_acquired.get(block as usize));
        assert!(!self.blocks_released.get(block as usize));
        assert!(!self.blocks_released_prior_checkpoint_durability.contains(block));

        // `blocks_released` remains unchanged while the current checkpoint is not durable,
        // since it contains blocks released in the previous checkpoint. These blocks must not be
        // freed till the current checkpoint is durable, so as to maintain the durability of these
        // blocks on a commit quorum of replicas.
        if self.checkpoint_durable {
            self.blocks_released.set(block as usize);
        } else {
            self.blocks_released_prior_checkpoint_durability.put_assume_capacity(block);
        }
    }

    /// Mark the given addresses as allocated in the current checkpoint, but free in the next one.
    ///
    /// This is used only when reading a free set from the grid. On disk representation of the
    /// free set doesn't include the blocks storing the free set itself, and these blocks must be
    /// manually patched in after decoding. As the next checkpoint will have a completely different
    /// free set, the blocks can be simultaneously released.
    ///
    /// # Panics
    /// Panics if addresses are unsorted/duplicated or inconsistent (upstream asserts).
    fn mark_released(&mut self, addresses: &[u64]) {
        assert!(!self.opened);
        assert!(!self.checkpoint_durable);

        let mut address_previous: u64 = 0;
        for address in addresses {
            assert!(*address > 0);

            // Assert that addresses are sorted and unique. Sortedness is not a requirement, but
            // a consequence of "first free" allocation algorithm.
            assert!(*address > address_previous);
            address_previous = *address;

            let block = (*address - 1) as usize;

            assert!(!self.blocks_acquired.get(block));
            assert!(!self.blocks_released.get(block));
            assert!(!self.blocks_released_prior_checkpoint_durability.contains(*address - 1));

            self.blocks_acquired.set(block);

            let shard = block / SHARD_BITS;
            // Update the index when every block in the shard is acquired.
            if self.find_free_block_in_shard(shard).is_none() {
                self.index.set(shard);
            }

            self.blocks_released_prior_checkpoint_durability.put_assume_capacity(*address - 1);
        }
    }

    /// Given the address, marks an acquired block as free.
    ///
    /// # Panics
    /// Panics unless the block was acquired-and-released and no reservations are held.
    fn free(&mut self, address: u64) {
        assert!(self.opened);
        assert!(self.checkpoint_durable);

        let block = address - 1;
        assert!(self.blocks_acquired.get(block as usize));
        assert!(self.blocks_released.get(block as usize));
        assert!(!self.blocks_released_prior_checkpoint_durability.contains(block));

        assert_eq!(self.reservation_count, 0);
        assert_eq!(self.reservation_blocks, 0);

        self.index.unset(block as usize / SHARD_BITS);
        self.blocks_acquired.unset(block as usize);
        self.blocks_released.unset(block as usize);
    }

    /// # Panics
    /// Panics if the checkpoint is already non-durable or staged releases exist.
    pub fn mark_checkpoint_not_durable(&mut self) {
        assert!(self.opened);
        assert!(self.checkpoint_durable);
        assert_eq!(self.blocks_released_prior_checkpoint_durability.count(), 0);
        self.checkpoint_durable = false;
    }

    /// Now that the checkpoint is durable on a commit quorum of replicas:
    /// 1. Mark the current checkpoint as durable.
    /// 2. Mark all released blocks in `blocks_released` as free.
    /// 3. Move released blocks from `blocks_released_prior_checkpoint_durability` to
    ///    `blocks_released`.
    ///
    /// # Panics
    /// Panics if the checkpoint is already durable.
    pub fn mark_checkpoint_durable(&mut self) {
        assert!(self.opened);
        assert!(!self.checkpoint_durable);

        self.checkpoint_durable = true;

        // DEVIATION: upstream frees blocks while iterating `blocks_released` (safe there because
        // `free()` only clears bits at/past the cursor); Rust's borrow checker requires
        // collecting the addresses first.
        let released: Vec<u64> = self
            .blocks_released
            .iter(BitKind::Set, Direction::Forward)
            .map(|block| block as u64 + 1)
            .collect();
        for address in released {
            self.free(address);
        }

        assert_eq!(self.blocks_released.count(), 0);

        // Block releases from the current checkpoint that were temporarily recorded in
        // blocks_released_prior_checkpoint_durability can now be moved to blocks_released.
        while let Some(block) = self.blocks_released_prior_checkpoint_durability.pop() {
            self.blocks_released.set(block as usize);
        }
        assert_eq!(self.blocks_released_prior_checkpoint_durability.count(), 0);

        // Index verification is O(blocks.bit_length) so do it only when checkpoint is marked
        // durable, which is also linear (as we free released blocks in `blocks_released`).
        self.verify_index();
    }

    /// Returns the number of blocks that the free set can physically reference via the acquired
    /// and released bitsets. Logically, the limit on the number of blocks that can be acquired by
    /// the free set is imposed by --limit-storage.
    #[must_use]
    pub fn block_count_max(grid_size_limit: usize) -> usize {
        let block_count_limit = grid_size_limit / constants::BLOCK_SIZE;
        block_count_limit.div_ceil(SHARD_BITS) * SHARD_BITS
    }

    /// Returns the maximum number of bytes needed for encoding the acquired/released bitset.
    ///
    /// # Panics
    /// Panics if the two blocksets have different lengths or the block count is misaligned.
    #[must_use]
    pub fn encode_size_max(&self) -> usize {
        assert_eq!(self.blocks_acquired.len(), self.blocks_released.len());

        let blocks_count = self.blocks_acquired.len();
        assert_eq!(blocks_count % SHARD_BITS, 0);
        assert_eq!(blocks_count % tigerbeetle_core::stdx::bitset::WORD_BITS, 0);

        ewah::encode_size_max(blocks_count / tigerbeetle_core::stdx::bitset::WORD_BITS)
    }

    /// Decodes the compressed bitset chunks into the target bitset.
    /// Panics if the encoding is invalid.
    ///
    /// # Panics
    /// Panics if the set is open/checkpoint-durable or the encoding overflows.
    fn decode(&mut self, target_bitset: BitsetKind, source_chunks: &[&[u8]]) {
        assert!(!self.opened);
        assert!(!self.checkpoint_durable);

        let source_size: usize = source_chunks.iter().map(|chunk| chunk.len()).sum();

        let (words, bitset_len) = match target_bitset {
            BitsetKind::BlocksAcquired => {
                let len = self.blocks_acquired.len();
                (self.blocks_acquired.words_mut(), len)
            }
            BitsetKind::BlocksReleased => {
                let len = self.blocks_released.len();
                (self.blocks_released.words_mut(), len)
            }
        };

        let mut decoder = ewah::Decoder::new(words, source_size);

        let mut words_decoded: usize = 0;
        for chunk in source_chunks {
            words_decoded += decoder.decode_chunk(chunk);
        }
        assert!(decoder.done());

        assert!(words_decoded * tigerbeetle_core::stdx::bitset::WORD_BITS <= bitset_len);

        // The encoder does not encode trailing 0s, so everything past words_decoded must be
        // zeroed.
        assert!(words[words_decoded..].iter().all(|&word| word == 0));
        // TODO: uncomment on the next release:
        // if words_decoded > 0 { assert!(words[words_decoded - 1] != 0); }
    }

    /// Decodes the compressed bitset chunks into the acquired and released bitsets.
    ///
    /// # Panics
    /// Panics if the set is not pristine (upstream asserts the same conditions).
    pub fn decode_chunks(
        &mut self,
        source_chunks_blocks_acquired: &[&[u8]],
        source_chunks_blocks_released: &[&[u8]],
    ) {
        assert!(!self.opened);
        assert!(!self.checkpoint_durable);

        // Verify that this FreeSet is entirely unallocated.
        assert_eq!(self.index.count(), 0);
        assert_eq!(self.blocks_acquired.count(), 0);
        assert_eq!(self.blocks_released.count(), 0);
        assert_eq!(self.blocks_released_prior_checkpoint_durability.count(), 0);

        assert_eq!(self.reservation_count, 0);
        assert_eq!(self.reservation_blocks, 0);

        self.decode(BitsetKind::BlocksAcquired, source_chunks_blocks_acquired);
        self.decode(BitsetKind::BlocksReleased, source_chunks_blocks_released);

        for shard in 0..self.index.len() {
            if self.find_free_block_in_shard(shard).is_none() {
                self.index.set(shard);
            }
        }

        self.verify_index();
    }

    /// Encodes one bitset into the given chunks; returns the encoded size in bytes,
    /// excluding trailing zero runs (upstream `encode`, private).
    fn encode_one(&self, source_bitset: BitsetKind, target_chunks: &mut [&mut [u8]]) -> usize {
        assert!(self.opened);
        assert!(self.checkpoint_durable);

        let mut encoder = match source_bitset {
            BitsetKind::BlocksAcquired => ewah::Encoder::new(self.blocks_acquired.words()),
            BitsetKind::BlocksReleased => ewah::Encoder::new(self.blocks_released.words()),
        };

        let mut bytes_encoded_total: u64 = 0;
        let mut finished = false;
        for chunk in target_chunks.iter_mut() {
            let bytes_encoded = encoder.encode_chunk(chunk);
            assert!(bytes_encoded > 0);

            bytes_encoded_total += bytes_encoded as u64;

            if encoder.done() {
                finished = true;
                break;
            }
        }
        assert!(finished, "target_chunks too small for encoding");

        // Don't explicitly encode trailing zeros to ensure that the encoding is the same
        // regardless of the runtime-configurable capacity of the bit set (driven by
        // --limit-storage).
        let bytes_trailing_zero_runs =
            (encoder.trailing_zero_runs_count() * core::mem::size_of::<u64>()) as u64;

        (bytes_encoded_total - bytes_trailing_zero_runs) as usize
    }

    /// Returns `(encoded_size_blocks_acquired, encoded_size_blocks_released)`.
    ///
    /// # Panics
    /// Panics if reservations are outstanding (upstream asserts the same).
    pub fn encode_chunks(
        &self,
        target_chunks_blocks_acquired: &mut [&mut [u8]],
        target_chunks_blocks_released: &mut [&mut [u8]],
    ) -> (usize, usize) {
        assert!(self.opened);
        assert!(self.checkpoint_durable);
        assert_eq!(self.reservation_count, 0);
        assert_eq!(self.reservation_blocks, 0);

        (
            self.encode_one(BitsetKind::BlocksAcquired, target_chunks_blocks_acquired),
            self.encode_one(BitsetKind::BlocksReleased, target_chunks_blocks_released),
        )
    }
}

/// Upstream `vsr.checkpoint_trailer.block_count_for_trailer_size`. Duplicated here (rather than
/// calling `tigerbeetle_vsr::checkpoint_trailer::block_count_for_trailer_size`) to avoid an
/// `lsm → vsr` dependency edge; keep both in sync.
/// The bound only sizes the staged-releases map.
fn checkpoint_trailer_block_count_for_trailer_size(trailer_size: usize) -> usize {
    trailer_size.div_ceil(constants::BLOCK_SIZE)
}

/// Returns the index of the first set/unset bit within the range `bit_min..bit_max`
/// (inclusive…exclusive).
///
/// Word-wise scan mirroring upstream `find_bit` (which pokes the std iterator internals).
fn find_bit(bit_set: &BitSet, bit_min: usize, bit_max: usize, kind: BitKind) -> Option<usize> {
    assert!(bit_max >= bit_min);
    assert!(bit_max <= bit_set.len());

    let want_set = kind == BitKind::Set;
    let words = bit_set.words();
    let word_bits = tigerbeetle_core::stdx::bitset::WORD_BITS;

    let word_start = bit_min / word_bits; // Inclusive.
    let word_offset = bit_min % word_bits;
    let word_end = bit_max.div_ceil(word_bits); // Exclusive.

    for (w, word) in words[word_start..word_end].iter().enumerate() {
        let mut candidate_word = if want_set { *word } else { !*word };
        if w == 0 && word_offset > 0 {
            candidate_word &= !((1_u64 << word_offset) - 1);
        }
        // Mask bits at or beyond `bit_max` in the final word.
        let word_bit_base = (word_start + w) * word_bits;
        if word_bit_base + word_bits > bit_max {
            let keep = bit_max - word_bit_base;
            candidate_word &= (1_u64 << keep) - 1;
        }
        if candidate_word != 0 {
            let bit = word_bit_base + candidate_word.trailing_zeros() as usize;
            assert!(bit >= bit_min && bit < bit_max);
            return Some(bit);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    // Tests mirror upstream's `.?` unwraps on Option-returning APIs.
    #![allow(clippy::unwrap_used)]

    use super::*;
    use tigerbeetle_core::stdx::prng::Prng;

    // DEVIATION: upstream seeds some tests from the system CSPRNG (`std.crypto.random`);
    // fixed seeds keep the port deterministic.

    #[test]
    fn block_shard_count() {
        if constants::BLOCK_SIZE != 64 * 1024 {
            // 64 KiB
            return;
        }
        let blocks_in_tb = (1 << 40) / constants::BLOCK_SIZE;
        test_block_shards_count(5120 * 8, 10 * blocks_in_tb);
        test_block_shards_count(5120 * 8 - 1, 10 * blocks_in_tb - SHARD_BITS);
        test_block_shards_count(1, SHARD_BITS); // Must be at least one index bit.
    }

    fn test_block_shards_count(expect_shards_count: usize, blocks_count: usize) {
        let set = FreeSet::open_empty(blocks_count);
        assert_eq!(expect_shards_count, set.index.len());
    }

    #[test]
    fn highest_address_acquired() {
        let blocks_count = SHARD_BITS;

        let mut set = FreeSet::open_empty(blocks_count);

        {
            let reservation = set.reserve(6).unwrap();
            assert_eq!(None::<u64>, set.highest_address_acquired());
            assert_eq!(Some(1), set.acquire(reservation));
            assert_eq!(Some(2), set.acquire(reservation));
            assert_eq!(Some(3), set.acquire(reservation));
            set.forfeit(reservation);
        }

        assert_eq!(Some(3), set.highest_address_acquired());

        set.release(2);
        set.free(2);
        assert_eq!(Some(3), set.highest_address_acquired());

        set.release(3);
        set.free(3);
        assert_eq!(Some(1), set.highest_address_acquired());

        set.release(1);
        set.free(1);
        assert_eq!(None, set.highest_address_acquired());

        {
            let reservation = set.reserve(6).unwrap();
            assert_eq!(Some(1), set.acquire(reservation));
            assert_eq!(Some(2), set.acquire(reservation));
            assert_eq!(Some(3), set.acquire(reservation));
            set.forfeit(reservation);
        }

        {
            set.release(3);
            assert_eq!(Some(3), set.highest_address_acquired());

            set.free(3);
            assert_eq!(Some(2), set.highest_address_acquired());
        }
    }

    #[test]
    fn acquire_release() {
        test_acquire_release(SHARD_BITS);
        test_acquire_release(2 * SHARD_BITS);
        test_acquire_release(63 * SHARD_BITS);
        test_acquire_release(64 * SHARD_BITS);
        test_acquire_release(65 * SHARD_BITS);
    }

    fn test_acquire_release(blocks_count: usize) {
        // Acquire everything, then release, then acquire again.
        let mut set = FreeSet::open_empty(blocks_count);
        let empty = FreeSet::open_empty(blocks_count);

        {
            let reservation = set.reserve(blocks_count).unwrap();

            for i in 0..blocks_count {
                assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
            }
            assert_eq!(None, set.acquire(reservation));
            set.forfeit(reservation);
        }

        assert_eq!(set.blocks_acquired.len(), set.count_acquired());
        assert_eq!(0, set.count_free());

        {
            for i in 0..blocks_count {
                set.release(i as u64 + 1);
                set.free(i as u64 + 1);
            }
            expect_free_set_equal(&empty, &set);
        }

        assert_eq!(0, set.count_acquired());
        assert_eq!(set.blocks_acquired.len(), set.count_free());

        {
            let reservation = set.reserve(blocks_count).unwrap();

            for i in 0..blocks_count {
                assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
            }
            assert_eq!(None, set.acquire(reservation));
            set.forfeit(reservation);
        }
    }

    #[test]
    fn reserve_acquire() {
        let blocks_count_total = SHARD_BITS;
        let mut set = FreeSet::open_empty(blocks_count_total);

        // At most `blocks_count_total` blocks are initially available for reservation.
        assert!(set.reserve(blocks_count_total + 1).is_none());
        let r1 = set.reserve(blocks_count_total - 1).unwrap();
        let r2 = set.reserve(1).unwrap();
        assert!(set.reserve(1).is_none());
        set.forfeit(r1);
        set.forfeit(r2);

        let mut address: u64 = 1; // Start at 1 because addresses are >0.
        {
            let reservation = set.reserve(2).unwrap();
            assert_eq!(Some(address), set.acquire(reservation));
            assert_eq!(Some(address + 1), set.acquire(reservation));
            assert_eq!(None, set.acquire(reservation));
            set.forfeit(reservation);
        }
        address += 2;

        {
            // Blocks are acquired from the target reservation.
            let reservation_1 = set.reserve(2).unwrap();
            let reservation_2 = set.reserve(2).unwrap();
            assert_eq!(Some(address), set.acquire(reservation_1));
            assert_eq!(Some(address + 2), set.acquire(reservation_2));
            assert_eq!(Some(address + 1), set.acquire(reservation_1));
            assert_eq!(None, set.acquire(reservation_1));
            assert_eq!(Some(address + 3), set.acquire(reservation_2));
            assert_eq!(None, set.acquire(reservation_2));
            set.forfeit(reservation_1);
            set.forfeit(reservation_2);
        }
    }

    #[test]
    fn checkpoint() {
        let blocks_count = SHARD_BITS;
        let mut set = FreeSet::open_empty(blocks_count);

        let empty = FreeSet::open_empty(blocks_count);

        let mut full = FreeSet::open_empty(blocks_count);

        {
            // Acquire all of `full`'s blocks.
            let reservation = full.reserve(blocks_count).unwrap();
            for i in 0..full.blocks_acquired.len() {
                assert_eq!(Some(i as u64 + 1), full.acquire(reservation));
            }
            full.forfeit(reservation);
        }

        {
            // Acquire & stage-release every block.
            let reservation = set.reserve(blocks_count).unwrap();

            for i in 0..set.blocks_acquired.len() {
                assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
                set.release(i as u64 + 1);

                // These count functions treat staged blocks as acquired.
                assert_eq!(i + 1, set.count_acquired());
                assert_eq!(set.blocks_acquired.len() - i - 1, set.count_free());
            }
            // All blocks are still acquired, though staged to release at the next checkpoint.
            assert_eq!(None, set.acquire(reservation));
            set.forfeit(reservation);
        }

        // Perform checkpoint-related operations.
        set.mark_checkpoint_not_durable();
        set.mark_checkpoint_durable();

        expect_free_set_equal(&empty, &set);
        assert_eq!(0, set.blocks_released.count());

        {
            // Allocate & stage-release all blocks again.
            let reservation = set.reserve(blocks_count).unwrap();

            for i in 0..set.blocks_acquired.len() {
                assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
                set.release(i as u64 + 1);
            }
            set.forfeit(reservation);
        }

        let mut set_encoded_blocks_acquired = vec![0_u8; set.encode_size_max()];
        let mut set_encoded_blocks_released = vec![0_u8; set.encode_size_max()];

        let mut set_decoded = FreeSet::init_empty(blocks_count);

        {
            let (encoded_size_blocks_acquired, encoded_size_blocks_released) = set.encode_chunks(
                &mut [&mut set_encoded_blocks_acquired],
                &mut [&mut set_encoded_blocks_released],
            );

            set_decoded.decode_chunks(
                &[&set_encoded_blocks_acquired[..encoded_size_blocks_acquired]],
                &[&set_encoded_blocks_released[..encoded_size_blocks_released]],
            );
            expect_free_set_equal(&set, &set_decoded);
        }

        {
            let (encoded_size_blocks_acquired, encoded_size_blocks_released) = full.encode_chunks(
                &mut [&mut set_encoded_blocks_acquired],
                &mut [&mut set_encoded_blocks_released],
            );

            set_decoded.reset();
            set_decoded.decode_chunks(
                &[&set_encoded_blocks_acquired[..encoded_size_blocks_acquired]],
                &[&set_encoded_blocks_released[..encoded_size_blocks_released]],
            );
            expect_free_set_equal(&full, &set_decoded);
        }
    }

    #[derive(Clone, Copy)]
    enum TestPatternFill {
        UniformOnes,
        UniformZeros,
        Literal,
    }

    struct TestPattern {
        fill: TestPatternFill,
        words: usize,
    }

    fn test_encode(patterns: &[TestPattern]) {
        let mut prng = Prng::from_seed(0x5EED_5EED_5EED_5EED);

        let word_bits = tigerbeetle_core::stdx::bitset::WORD_BITS;
        let blocks_count: usize = patterns.iter().map(|p| p.words * word_bits).sum();

        let mut decoded_expect = FreeSet::open_empty(blocks_count);

        {
            // The `index` will start out one-filled. Every pattern containing a zero will update
            // the corresponding index bit with a zero (probably multiple times) to ensure it ends
            // up synced with `blocks`.
            decoded_expect.index.toggle_all();
            assert_eq!(decoded_expect.index.count(), decoded_expect.index.capacity());

            // Fill the bitset according to the patterns.
            let blocks = decoded_expect.blocks_acquired.words_mut();
            let mut blocks_offset: usize = 0;
            for pattern in patterns {
                for _ in 0..pattern.words {
                    blocks[blocks_offset] = match pattern.fill {
                        TestPatternFill::UniformOnes => u64::MAX,
                        TestPatternFill::UniformZeros => 0,
                        TestPatternFill::Literal => {
                            prng.range_inclusive_usize(1, usize::MAX - 1) as u64
                        }
                    };
                    let index_bit = blocks_offset * word_bits / SHARD_BITS;
                    if !matches!(pattern.fill, TestPatternFill::UniformOnes) {
                        decoded_expect.index.unset(index_bit);
                    }
                    blocks_offset += 1;
                }
            }
            assert_eq!(blocks_offset, blocks.len());
        }

        let mut encoded = vec![0_u8; decoded_expect.encode_size_max()];

        assert_eq!(encoded.len() % core::mem::size_of::<u64>(), 0);
        let encoded_length =
            decoded_expect.encode_one(BitsetKind::BlocksAcquired, &mut [&mut encoded]);

        let mut decoded_actual = FreeSet::init_empty(blocks_count);

        decoded_actual.decode_chunks(&[&encoded[..encoded_length]], &[]);
        expect_free_set_equal(&decoded_expect, &decoded_actual);
    }

    #[test]
    fn encode_decode_encode() {
        // Number of words per shard.
        let shard_bits = SHARD_BITS / tigerbeetle_core::stdx::bitset::WORD_BITS;

        // Uniform.
        test_encode(&[TestPattern { fill: TestPatternFill::UniformOnes, words: shard_bits }]);
        test_encode(&[TestPattern { fill: TestPatternFill::UniformZeros, words: shard_bits }]);
        test_encode(&[TestPattern { fill: TestPatternFill::Literal, words: shard_bits }]);
        test_encode(&[TestPattern {
            fill: TestPatternFill::UniformOnes,
            words: (u16::MAX as usize) + 1,
        }]);

        // Mixed.
        test_encode(&[
            TestPattern { fill: TestPatternFill::UniformOnes, words: shard_bits / 4 },
            TestPattern { fill: TestPatternFill::UniformZeros, words: shard_bits / 4 },
            TestPattern { fill: TestPatternFill::Literal, words: shard_bits / 4 },
            TestPattern { fill: TestPatternFill::UniformOnes, words: shard_bits / 4 },
        ]);

        // Random.
        let mut prng = Prng::from_seed(0x1234_5678_9ABC_DEF0);

        let fills =
            [TestPatternFill::UniformOnes, TestPatternFill::UniformZeros, TestPatternFill::Literal];
        for _ in 0..10 {
            let mut patterns = Vec::with_capacity(shard_bits);
            for _ in 0..shard_bits {
                patterns.push(TestPattern { fill: fills[prng.index(fills.len())], words: 1 });
            }
            test_encode(&patterns);
        }
    }

    fn expect_free_set_equal(a: &FreeSet, b: &FreeSet) {
        expect_bit_set_equal(&a.blocks_acquired, &b.blocks_acquired);
        expect_bit_set_equal(&a.blocks_released, &b.blocks_released);
        expect_bit_set_equal(&a.index, &b.index);

        assert_eq!(
            a.blocks_released_prior_checkpoint_durability.count(),
            b.blocks_released_prior_checkpoint_durability.count(),
        );

        for (address_a, address_b) in a
            .blocks_released_prior_checkpoint_durability
            .keys
            .iter()
            .zip(b.blocks_released_prior_checkpoint_durability.keys.iter())
        {
            assert_eq!(address_a, address_b);
        }
    }

    fn expect_bit_set_equal(a: &BitSet, b: &BitSet) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.words(), b.words());
    }

    #[test]
    fn decode_small_bitset_into_large_bitset() {
        let shard_bits = SHARD_BITS;
        let mut small_set = FreeSet::open_empty(shard_bits);

        {
            // Set up a small bitset (with blocks_count==shard_bits) with no free blocks.
            let reservation = small_set.reserve(small_set.blocks_acquired.len()).unwrap();

            for _ in 0..small_set.blocks_acquired.len() {
                small_set.acquire(reservation);
            }
            small_set.forfeit(reservation);
        }

        let mut small_buffer = vec![0_u8; small_set.encode_size_max()];

        let small_buffer_written =
            small_set.encode_one(BitsetKind::BlocksAcquired, &mut [&mut small_buffer]);

        // Decode the serialized small bitset into a larger bitset
        // (with blocks_count==2*shard_bits).
        let mut big_set = FreeSet::init_empty(2 * shard_bits);

        big_set.decode(BitsetKind::BlocksAcquired, &[&small_buffer[..small_buffer_written]]);
        big_set.opened = true;

        for block in 0..2 * shard_bits {
            let address = block as u64 + 1;
            assert_eq!(shard_bits <= block, big_set.is_free(address));
        }
    }

    #[test]
    fn encode_decode_manual() {
        let encoded_words: [u64; 5] = [
            // Mask 1: run of 2 words of 0s (uniform_bit = 0), then 3 literals
            #[allow(clippy::identity_op)] // keeps the marker layout fields visible
            (0 | (2 << 1) | (3 << 32)),
            0xAAAA_AAAA_AAAA_AAAA, // literal 1
            0x5555_5555_5555_5555, // literal 2
            0xAAAA_AAAA_AAAA_AAAA, // literal 3
            // Mask 2: run of 59 words of 1s, then 0 literals
            //
            // 59 is chosen so that because the blocks_count must be a multiple of the shard size:
            // shard_bits = 4096 bits = 64 words × 64 bits/word = (2+3+59)*64
            1 | ((64 - 5) << 1),
        ];
        let mut encoded_bytes = Vec::new();
        for word in &encoded_words {
            encoded_bytes.extend_from_slice(&word.to_le_bytes());
        }

        let mut decoded_expect = vec![
            0x0000_0000_0000_0000, // run 1
            0x0000_0000_0000_0000,
            0xAAAA_AAAA_AAAA_AAAA, // literal 1
            0x5555_5555_5555_5555, // literal 2
            0xAAAA_AAAA_AAAA_AAAA, // literal 3
        ];
        decoded_expect.resize(64, u64::MAX); // 64 - 5 uniform-one words

        let word_bits = tigerbeetle_core::stdx::bitset::WORD_BITS;
        let blocks_count = decoded_expect.len() * word_bits;

        // Test decode.
        let mut decoded_actual = FreeSet::init_empty(blocks_count);

        decoded_actual.decode(BitsetKind::BlocksAcquired, &[&encoded_bytes]);

        assert_eq!(decoded_expect.len(), decoded_actual.blocks_acquired.words().len());
        assert_eq!(&decoded_expect, decoded_actual.blocks_acquired.words());

        // Test encode.
        let mut encoded_actual = vec![0_u8; decoded_actual.encode_size_max()];

        // Pretend `opened` and `checkpoint_durable` are true as it is asserted in `encode_one`.
        decoded_actual.opened = true;
        decoded_actual.checkpoint_durable = true;
        let encoded_actual_length =
            decoded_actual.encode_one(BitsetKind::BlocksAcquired, &mut [&mut encoded_actual]);
        assert_eq!(encoded_words.len() * core::mem::size_of::<u64>(), encoded_actual_length);
    }

    /// Returns the index of the first set/unset bit within the range `bit_min..bit_max`
    /// (inclusive…exclusive), using a linear scan as the reference model.
    fn find_bit_reference(
        bit_set: &BitSet,
        bit_min: usize,
        bit_max: usize,
        kind: BitKind,
    ) -> Option<usize> {
        (bit_min..bit_max).find(|&bit| bit_set.get(bit) == (kind == BitKind::Set))
    }

    fn test_find_bit(prng: &mut Prng, bit_set: &BitSet, kind: BitKind) {
        let bit_min = prng.int_inclusive_usize(bit_set.len() - 1);
        let bit_max = prng.range_inclusive_usize(bit_min, bit_set.len());

        let bit_actual = find_bit(bit_set, bit_min, bit_max, kind);
        if let Some(bit) = bit_actual {
            assert_eq!(bit_set.get(bit), kind == BitKind::Set);
            assert!(bit >= bit_min);
            assert!(bit < bit_max);
        }

        assert_eq!(bit_actual, find_bit_reference(bit_set, bit_min, bit_max, kind));
    }

    #[test]
    fn find_bit_fuzz() {
        let mut prng = Prng::from_seed(0xF100);

        let word_bits = tigerbeetle_core::stdx::bitset::WORD_BITS;
        for bit_length in 1..=(word_bits * 4) {
            let mut bit_set = BitSet::new_empty(bit_length);

            let p = prng.int_inclusive_usize(100);

            for b in 0..bit_length {
                if p < prng.int_inclusive_usize(100) {
                    bit_set.set(b);
                } else {
                    bit_set.unset(b);
                }
            }

            for _ in 0..20 {
                test_find_bit(&mut prng, &bit_set, BitKind::Set);
            }
            for _ in 20..40 {
                test_find_bit(&mut prng, &bit_set, BitKind::Unset);
            }
        }
    }

    #[test]
    fn acquire_part_way_through_a_shard() {
        let mut set = FreeSet::open_empty(SHARD_BITS * 3);

        let reservation_a = set.reserve(1).unwrap();

        let reservation_b = set.reserve(2 * SHARD_BITS).unwrap();

        // Acquire all of reservation B.
        // At the end, the first shard still has a bit free (reserved by A).
        for i in 0..reservation_b.block_count {
            let address = set.acquire(reservation_b).unwrap();
            assert_eq!(address - 1, reservation_a.block_count as u64 + i as u64);
            set.verify_index();
        }
        assert_eq!(None, set.acquire(reservation_b));

        set.forfeit(reservation_a);
        set.forfeit(reservation_b);
    }

    #[test]
    fn decode_big_bitset_into_small_bitset() {
        let shard_bits = SHARD_BITS;

        let mut big_set = FreeSet::open_empty(2 * shard_bits);

        {
            // Set up a big bitset (with blocks_count==2*shard_bits) with half the blocks free.
            let acquired_block_count = big_set.blocks_acquired.len() / 2;
            let reservation = big_set.reserve(acquired_block_count).unwrap();

            for _ in 0..acquired_block_count {
                big_set.acquire(reservation);
            }
            big_set.forfeit(reservation);
        }

        let mut big_buffer = vec![0_u8; big_set.encode_size_max()];

        let big_buffer_written =
            big_set.encode_one(BitsetKind::BlocksAcquired, &mut [&mut big_buffer]);

        // Decode the serialized big bitset into a smaller bitset (with blocks_count==shard_bits).
        let mut small_set = FreeSet::init_empty(shard_bits);

        small_set.decode(BitsetKind::BlocksAcquired, &[&big_buffer[..big_buffer_written]]);
        for block in 0..shard_bits {
            let address = block as u64 + 1;
            assert!(!small_set.is_free(address));
            assert!(!big_set.is_free(address));
        }
    }
}
