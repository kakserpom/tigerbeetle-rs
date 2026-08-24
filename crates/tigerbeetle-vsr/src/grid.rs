//! The Grid provides access to on-disk blocks (blobs of [`BLOCK_SIZE`] bytes).
//! Each block is identified by an "address" (`u64`, beginning at 1).
//!
//! Port of `src/vsr/grid.zig` — slice 1 covered the in-memory block-management layer
//! (block storage, reference counting, stash, block cache, offset math and read
//! validation); slice 2 adds the IO paths: create/repair writes with a bounded IOP
//! pool and unbounded write queue, reads with merging, cache shortcuts and the
//! remote-repair parking queue, plus `fulfill_block` and free-set plumbing.
//!
//! Remaining upstream surface:
//!
//! TODO(port): `src/vsr/grid.zig` — `open`/`checkpoint`/
//! `mark_checkpoint_not_durable`/`checkpoint_durable`/`cancel` (need the full
//! SuperBlock + CheckpointTrailer), `blocks_missing` repair bookkeeping (currently
//! `repair_block` skips its asserts/hooks), `checkpoint_id`/`checkpoint_durable`
//! stamps on reads/writes, `verify_table`/`assert_coherent`/`madv_dont_dump`.
//!
//! DEVIATION: upstream hands out `*align(sector_size) [block_size]u8` pointers into a
//! single allocation; safe Rust forbids aliasing pointers, so blocks are identified by
//! their index ("location") within [`Grid::blocks`] instead. Pointer-identity assertions
//! become location comparisons with identical semantics. Consequently, resolved reads
//! deliver a location whose bytes the caller must copy before the next [`Grid::poll`]
//! (the location is recycled then), mirroring upstream's "pointer valid during the
//! callback" contract.
//!
//! DEVIATION: callbacks become [`Event`]s drained from [`Grid::take_events`], and the
//! `on_next_tick` deferral for `cache_read` disappears — with deferred user code there
//! is no re-entrancy to protect against, so reads resolve inline with identical
//! observable outcomes.
//!
//! DEVIATION: `fulfill_block()` resolves readers with a copy of the network-provided
//! block held by the grid until the next poll (upstream hands out the message buffer
//! directly).
//!
//! DEVIATION: `reserve()` panics instead of calling `vsr.fatal()`; fatal handling
//! arrives with the server binary.

#![allow(clippy::cast_possible_truncation)] // block counts and sizes fit u32 like upstream

use std::collections::{HashSet, VecDeque};

use tigerbeetle_core::constants::{BLOCK_SIZE, SECTOR_SIZE};
use tigerbeetle_lsm::free_set::{FreeSet, Reservation};
use tigerbeetle_lsm::set_associative_cache::{
    Layout, SetAssociativeCache, SetAssociativeCacheSpec,
};

use crate::Zone;
use crate::command::Command;
use crate::message_header::{self, TypedHeader};
use crate::schema;
use crate::storage::{Completion, ReadRequest, Storage, WriteRequest};

/// Upstream `set_associative_cache_ways = 16` and its cache-line override.
struct GridCacheAddress;

impl Layout for GridCacheAddress {
    // Upstream: `.cache_line_size = 16` — allows running with a much smaller grid cache
    // (256MiB vs 1GiB) instead of being completely optimal.
    const CACHE_LINE_SIZE: u64 = 16;
}

impl SetAssociativeCacheSpec for GridCacheAddress {
    type Key = u64;
    type Value = u64;

    fn key_from_value(value: &u64) -> u64 {
        *value
    }

    fn hash(key: u64) -> u64 {
        assert!(key > 0);
        tigerbeetle_core::stdx::hash::hash_inline_u64(key)
    }
}

type Cache = SetAssociativeCache<GridCacheAddress>;

/// Byte offset of a grid address within the `Grid` zone.
///
/// # Panics
/// Asserts `address > 0` (addresses begin at 1).
#[must_use]
pub fn block_offset(address: u64) -> u64 {
    assert!(address > 0);
    (address - 1) * BLOCK_SIZE as u64
}

/// Smallest sector-multiple that fits `size` bytes.
fn sector_ceil(size: usize) -> usize {
    size.div_ceil(SECTOR_SIZE) * SECTOR_SIZE
}

/// Although we distinguish between the reasons why the block is invalid, upstream only uses
/// this info for logging, not logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadBlockResult {
    /// The block is valid.
    Valid,
    /// Checksum of block header is invalid.
    InvalidChecksum,
    /// Checksum of block body is invalid.
    InvalidChecksumBody,
    /// The block header is valid, but its `header.command` is not `block`
    /// (this is possible due to misdirected IO).
    UnexpectedCommand,
    /// The block is valid, but it is not the block we expected.
    UnexpectedChecksum,
    /// The block is valid, and it is the expected block, but the last sector's padding is
    /// corrupt, so it will be repaired just to be safe.
    InvalidPadding,
}

/// Construction options for [`Grid`] (upstream's anonymous init-options struct).
#[derive(Clone, Copy, Debug)]
pub struct GridOptions {
    pub cache_blocks_count: usize,
    pub stash_blocks_count: usize,
    pub read_iops_max: usize,
    pub write_iops_max: usize,
    /// Test/bootstrap constructor for the free set: `Some(blocks_count)` builds an
    /// opened, entirely-free set (upstream populates the free set from checkpoint
    /// trailers inside `Grid.open()`, which lands with the superblock lifecycle —
    /// TODO(port): src/vsr/grid.zig `open`). Must be a multiple of
    /// [`tigerbeetle_lsm::free_set::SHARD_BITS`].
    pub free_set_blocks_count: Option<usize>,
}

/// Whether an address is currently being written, and why
/// (upstream `Grid.Writing`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Writing {
    Create,
    Repair,
    NotWriting,
}

/// Read options (the coherent flag replaces upstream's callback-kind discrimination:
/// `.from_local_storage` → `coherent = false`, `.from_local_or_global_storage` →
/// `coherent = true`).
#[derive(Clone, Copy, Debug)]
pub struct ReadOptions {
    pub cache_read: bool,
    pub cache_write: bool,
}

/// Completion events drained via [`Grid::take_events`] — the event-queue replacement
/// for upstream's write/read callbacks.
///
/// Locations referenced by an event are only valid until the next [`Grid::poll`]
/// (see the module-level deviation notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    WriteDone {
        token: u32,
        address: u64,
        /// A fresh stash block handed to the caller, replacing the consumed one
        /// (upstream overwrites the caller's `*BlockPtr` in the callback).
        fresh_location: u32,
    },
    ReadDone {
        token: u32,
        address: u64,
        checksum: u128,
        result: ReadBlockResult,
        /// Set iff `result == Valid`: where to copy the block bytes from.
        valid_location: Option<u32>,
    },
}

/// The `(address, checksum)` pair a read expects to find
/// (upstream anonymous `expect` parameter of `read_block_validate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpectBlock {
    pub address: u64,
    pub checksum: u128,
}

/// Validates a freshly-read block against the expected address/checksum
/// (upstream `read_block_validate`; the `Valid` variant carries no pointer here —
/// the caller keeps the buffer it passed in).
///
/// # Panics
/// Asserts internal invariants of a checksum-valid block header (size bounds), mirroring
/// upstream's asserts.
#[must_use]
pub fn read_block_validate(block: &[u8], expect: ExpectBlock) -> ReadBlockResult {
    let frame: &[u8; message_header::SIZE] = block[..message_header::SIZE]
        .try_into()
        .unwrap_or_else(|_| unreachable!("grid blocks are at least HEADER_SIZE bytes"));
    let Some(base) = message_header::Header::from_wire(frame) else {
        unreachable!("generic header frames always parse");
    };

    if !base.valid_checksum() {
        return ReadBlockResult::InvalidChecksum;
    }

    // Upstream reinterprets the first bytes as `vsr.Header.Block` without checking the
    // command byte; only after the checksum does it compare the command explicitly:
    if base.command != Command::Block {
        return ReadBlockResult::UnexpectedCommand;
    }

    let Some(header) = message_header::Block::from_wire(frame) else {
        unreachable!("command was verified as Block");
    };

    assert!(header.size >= message_header::SIZE as u32);
    assert!(header.size <= BLOCK_SIZE as u32);

    if !base.valid_checksum_body(&block[message_header::SIZE..header.size as usize]) {
        return ReadBlockResult::InvalidChecksumBody;
    }

    if header.checksum != expect.checksum {
        return ReadBlockResult::UnexpectedChecksum;
    }

    if !tigerbeetle_core::stdx::zeroed(
        &block[header.size as usize..sector_ceil(header.size as usize)],
    ) {
        return ReadBlockResult::InvalidPadding;
    }

    assert_eq!(header.address, expect.address);
    ReadBlockResult::Valid
}

/// A stack of stash locations with O(1) membership. Order mirrors upstream's
/// `AutoArrayHashMapUnmanaged`: `pop()` removes the most recently pushed entry and
/// `remove()` swaps the last element into the hole (`swapRemove`).
#[derive(Debug, Default)]
struct StashStack {
    order: Vec<u32>,
    members: HashSet<u32>,
}

impl StashStack {
    fn push(&mut self, location: u32) {
        assert!(self.members.insert(location), "location {location} already stashed");
        self.order.push(location);
    }

    fn pop(&mut self) -> Option<u32> {
        let location = self.order.pop()?;
        assert!(self.members.remove(&location));
        Some(location)
    }

    fn remove(&mut self, location: u32) -> bool {
        if self.members.remove(&location) {
            let index = self
                .order
                .iter()
                .position(|&candidate| candidate == location)
                .unwrap_or_else(|| unreachable!("membership set and order are in sync"));
            self.order.swap_remove(index);
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.order.len()
    }
}

/// State of one queued/submitted read (upstream `Grid.Read` minus checkpoint stamps).
#[derive(Debug)]
struct ReadOp {
    token: u32,
    address: u64,
    checksum: u128,
    /// Upstream `callback == .from_local_or_global_storage`.
    coherent: bool,
    // Upstream also stamps `cache_read` here for its next-tick deferral; we resolve
    // inline (module-level deviation), so only `cache_write` survives.
    cache_write: bool,
    /// Reads merged into this one (upstream `resolves: QueueType(ReadPending)`).
    resolves: VecDeque<usize>,
    state: ReadState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadState {
    /// Root in `read_queue`, waiting for an IOP.
    RootQueued,
    /// Root in `read_pending_queue`, waiting for an IOP to free up.
    WaitingIop,
    /// Submitted to storage; awaiting completion.
    Executing,
    /// Parked in `read_global_queue`, awaiting `fulfill_block()`.
    ParkedGlobal,
    /// Attached as a resolver to another read; freed when that read resolves.
    Attached,
}

/// A queued/submitted write (upstream `Grid.Write`; `checkpoint_id` TODO(port)).
#[derive(Debug)]
struct WriteOp {
    token: u32,
    address: u64,
    repair: bool,
    location: u32,
    /// Sector-rounded length actually submitted (`sector_ceil(header.size)`).
    sector_len: usize,
}

/// In-memory half of the Grid plus its IO queues
/// (upstream `GridType` fields minus superblock hooks).
pub struct Grid {
    /// Block contents. Entries correspond to `blocks_references` entries.
    blocks: Vec<Vec<u8>>,
    /// Per-block outstanding-reference counts (saturates at `u8::MAX` like upstream).
    blocks_references: Vec<u8>,

    cache: Cache,
    /// The block cached at cache way-set slot `i` lives in `blocks[cache_locations[i]]`.
    ///
    /// Invariants:
    /// - `cache_locations[i] < blocks.len()`
    /// - `cache_locations[i] != cache_locations[j] iff i != j`
    cache_locations: Vec<u32>,

    /// Free stash locations (no outstanding references), LIFO.
    stash_free: StashStack,
    /// Stash locations handed out via [`Grid::get_block`]/[`Grid::block_ref`].
    stash_used: StashStack,

    /// How many more stash references may be taken.
    ///
    /// Invariants: `stash_available <= stash_blocks_count`,
    /// `stash_available <= stash_free.len()`.
    stash_available: u32,

    /// Blocks reserved for read IOPs; slot `s` reads into `blocks[read_iop_blocks[s]]`
    /// and is replaced with a fresh block on every completion (the reference "burst").
    read_iop_blocks: Vec<u32>,
    /// Read-IOP slots currently unassigned, FIFO.
    read_iops_free: VecDeque<usize>,

    free_set: FreeSet,

    /// Writes executing on storage, by IOP slot.
    writes_exec: Vec<Option<WriteOp>>,
    /// FIFO order of occupied write slots (matches storage completion order).
    write_exec_order: VecDeque<usize>,
    /// Writes waiting for a free IOP slot, FIFO.
    write_queue: VecDeque<WriteOp>,

    /// All live reads, addressed by arena index (recycled via `reads_free`).
    reads: Vec<Option<ReadOp>>,
    reads_free: VecDeque<usize>,
    /// Root reads awaiting/using an IOP, FIFO (merge target for later reads).
    read_queue: VecDeque<usize>,
    /// Roots waiting for an IOP slot, FIFO.
    read_pending_queue: VecDeque<usize>,
    /// Coherent reads parked after invalid results, awaiting remote repair via
    /// [`Grid::fulfill_block`] (upstream `read_global_queue`).
    read_global_queue: VecDeque<usize>,
    /// `(arena index, IOP slot)` of reads executing on storage, FIFO.
    read_exec_order: VecDeque<(usize, usize)>,

    next_write_token: u32,
    next_read_token: u32,

    /// Blocks whose last reference is dropped at the start of the next poll:
    /// consumed read-IOP blocks and `fulfill_block` copies.
    pending_reap: Vec<u32>,

    events: VecDeque<Event>,
}

impl Grid {
    /// # Panics
    /// Asserts `stash_blocks_count > 0` and capacity arithmetic, mirroring upstream init.
    #[must_use]
    pub fn new(options: GridOptions) -> Self {
        assert!(options.stash_blocks_count > 0);

        let blocks_count = options.cache_blocks_count + options.stash_blocks_count;
        let mut blocks = Vec::with_capacity(blocks_count);
        for _ in 0..blocks_count {
            blocks.push(vec![0u8; BLOCK_SIZE]);
        }

        let mut blocks_references = vec![0u8; blocks_count];
        let mut cache_locations = Vec::with_capacity(options.cache_blocks_count);

        let mut stash_free = StashStack::default();
        let mut stash_used = StashStack::default();

        for location in 0..u32::try_from(blocks_count).unwrap_or_else(|_| {
            unreachable!("block count fits u32 like upstream's location indices")
        }) {
            let location_usize = location as usize;
            if location_usize < options.cache_blocks_count {
                cache_locations.push(location);
            } else {
                stash_free.push(location);
            }
        }
        assert_eq!(stash_free.len(), options.stash_blocks_count);

        // Reserve one stash block per potential read IOP (upstream init loop):
        let mut read_iop_blocks = Vec::new();
        for _ in 0..options.read_iops_max {
            let location =
                stash_free.pop().unwrap_or_else(|| unreachable!("stash sized for reads"));
            blocks_references[location as usize] += 1;
            stash_used.push(location);
            read_iop_blocks.push(location);
        }

        let free_set = if let Some(blocks_count) = options.free_set_blocks_count {
            FreeSet::open_empty(blocks_count)
        } else {
            let grid_size_limit = options.cache_blocks_count * BLOCK_SIZE;
            FreeSet::new(grid_size_limit, 0)
        };

        let grid = Self {
            blocks,
            blocks_references,
            cache: Cache::new(options.cache_blocks_count as u64, "grid"),
            cache_locations,
            stash_free,
            stash_used,
            stash_available: u32::try_from(options.stash_blocks_count)
                .unwrap_or_else(|_| unreachable!("stash count fits u32"))
                - u32::try_from(options.read_iops_max)
                    .unwrap_or_else(|_| unreachable!("read iops fit u32")),
            read_iop_blocks,
            read_iops_free: (0..options.read_iops_max).collect(),
            free_set,
            writes_exec: (0..options.write_iops_max).map(|_| None).collect(),
            write_exec_order: VecDeque::new(),
            write_queue: VecDeque::new(),
            reads: Vec::new(),
            reads_free: VecDeque::new(),
            read_queue: VecDeque::new(),
            read_pending_queue: VecDeque::new(),
            read_global_queue: VecDeque::new(),
            read_exec_order: VecDeque::new(),
            next_write_token: 0,
            next_read_token: 0,
            pending_reap: Vec::new(),
            events: VecDeque::new(),
        };
        grid.assert_invariants();
        grid
    }

    fn assert_invariants(&self) {
        assert!(self.stash_available as usize <= self.stash_free.len());
        let mut seen = HashSet::new();
        for &location in &self.cache_locations {
            assert!(location < self.blocks.len() as u32);
            assert!(seen.insert(location));
        }
    }

    /// Return a block from the stash which had no outstanding references.
    ///
    /// # Panics
    /// Panics when the stash has no free blocks (upstream `@panic`).
    pub fn get_block(&mut self) -> u32 {
        let location = self.stash_free.pop().unwrap_or_else(|| panic!("stash has no free blocks"));

        assert_eq!(self.blocks_references[location as usize], 0);
        self.blocks_references[location as usize] += 1;
        self.stash_available -= 1;
        self.stash_used.push(location);

        location
    }

    /// Take an additional reference to a block (upstream `block_ref`; upstream also
    /// validates the block's header checksum, which we keep).
    ///
    /// # Panics
    /// Asserts the block holds a valid header and the location is not in `stash_free`.
    pub fn block_ref(&mut self, location: u32) {
        let header = schema::header_from_block(&self.blocks[location as usize]);
        assert!(header.valid_checksum());

        assert!(!self.stash_free.members.contains(&location));

        if self.blocks_references[location as usize] == 0 {
            // The only way to reference a zero-reference block is if we got it from the
            // cache, not the stash.
            assert!(!self.stash_used.members.contains(&location));
        }

        self.blocks_references[location as usize] += 1;
        self.stash_available -= 1;

        if !self.stash_used.members.contains(&location) {
            self.stash_used.push(location);
        }
    }

    /// Release a reference to a block.
    ///
    /// # Panics
    /// Asserts the reference count is positive.
    pub fn block_unref(&mut self, location: u32) {
        assert!(!self.stash_free.members.contains(&location));

        assert!(self.blocks_references[location as usize] > 0);
        self.blocks_references[location as usize] -= 1;
        self.stash_available += 1;

        if self.blocks_references[location as usize] == 0 && self.stash_used.remove(location) {
            self.stash_free.push(location);
        }

        assert!(self.stash_available as usize <= self.stash_free.len());
    }

    /// Outstanding references to the given block.
    #[must_use]
    pub fn block_references(&self, location: u32) -> u8 {
        self.blocks_references[location as usize]
    }

    /// Insert the address into the cache, and swap the evicted block into the stash
    /// (upstream `cache_upsert`).
    ///
    /// # Panics
    /// Asserts the saved block is referenced, carries `address` in its header, and is
    /// currently a used stash block.
    pub fn cache_upsert(&mut self, block_address: u64, block_save_location: u32) {
        assert!(block_address != 0);

        // The location/block that is being moved from stash to cache:
        assert!(self.blocks_references[block_save_location as usize] > 0);

        {
            let save_header = schema::header_from_block(&self.blocks[block_save_location as usize]);
            assert_eq!(save_header.address, block_address);
        }

        let upserted = self.cache.upsert(&block_address);
        let cache_index = upserted.index;
        assert!(cache_index < self.cache_locations.len());

        // The location/block being moved from cache to stash:
        let block_drop_location = self.cache_locations[cache_index];
        assert_ne!(block_drop_location, block_save_location);

        let block_drop_removed = self.stash_used.remove(block_save_location);
        assert!(block_drop_removed);

        if self.blocks_references[block_drop_location as usize] == 0 {
            self.stash_free.push(block_drop_location);
        } else {
            self.stash_used.push(block_drop_location);
        }
        self.cache_locations[cache_index] = block_save_location;

        if self.blocks_references[block_drop_location as usize] == 0 {
            // This block content won't be used again. We could overwrite the entire thing,
            // but that would be more expensive.
            self.blocks[block_drop_location as usize][..message_header::SIZE].fill(0);
        }
    }

    /// The block currently cached for `address`, if any (its location).
    #[must_use]
    pub fn cached_location(&mut self, address: u64) -> Option<u32> {
        let cache_index = self.cache.get_index(address)?;
        Some(self.cache_locations[cache_index])
    }

    /// Number of cache slots backing this grid (test/introspection helper).
    #[must_use]
    pub fn cache_slots(&self) -> usize {
        self.cache_locations.len()
    }

    /// Total number of blocks (cache + stash) held by this grid.
    #[must_use]
    pub fn blocks_count(&self) -> usize {
        self.blocks.len()
    }

    /// Contents of a block (read-only view for tests and upcoming slices).
    #[must_use]
    pub fn block(&self, location: u32) -> &[u8] {
        &self.blocks[location as usize]
    }

    /// Mutable contents of a block.
    pub fn block_mut(&mut self, location: u32) -> &mut [u8] {
        &mut self.blocks[location as usize]
    }

    /// Abort if there are not enough free blocks to fill the reservation
    /// (upstream aborts via `vsr.fatal(.storage_size_would_exceed_limit)`).
    ///
    /// # Panics
    /// Panics when the free set cannot cover `blocks_count`.
    pub fn reserve(&mut self, blocks_count: usize) -> Reservation {
        self.free_set.reserve(blocks_count).unwrap_or_else(|| {
            panic!(
                "data file would become too large: restart the replica increasing \
                 '--limit-storage' (reservation of {blocks_count} blocks exceeded the limit)"
            )
        })
    }

    /// Forfeit a reservation.
    pub fn forfeit(&mut self, reservation: Reservation) {
        self.free_set.forfeit(reservation);
    }

    /// Returns a just-allocated block.
    ///
    /// # Panics
    /// Panics if more blocks are acquired than reserved (upstream `orelse` unreachable).
    #[must_use]
    pub fn acquire(&mut self, reservation: Reservation) -> u64 {
        self.free_set
            .acquire(reservation)
            .unwrap_or_else(|| panic!("grid.acquire(): reservation exhausted"))
    }

    /// Release addresses, demoting them within the block cache to reduce conflict
    /// misses (upstream `Grid.release`; the blocks stay readable until the next
    /// checkpoint).
    ///
    /// # Panics
    /// Asserts addresses are positive and not under a repairing write.
    pub fn release(&mut self, addresses: &[u64]) {
        for &address in addresses {
            assert!(address > 0);

            // It's safe to release an address that is being read from or written to,
            // as it can only be overwritten in the next checkpoint (when the address
            // is freed and can be reacquired).
            assert!(matches!(self.writing(address, None), Writing::NotWriting | Writing::Create));

            self.cache.demote(address);
            self.free_set.release(address);
        }
    }

    /// Whether `address` is being written to, and why
    /// (upstream `Grid.writing`).
    ///
    /// # Panics
    /// If `expect_location` is given, asserts that no write uses that block.
    #[must_use]
    pub fn writing(&self, address: u64, expect_location: Option<u32>) -> Writing {
        assert!(address > 0);

        let mut result = Writing::NotWriting;
        let mut scan = |op: &WriteOp| {
            if let Some(expect_location) = expect_location {
                assert_ne!(op.location, expect_location);
            }
            if address == op.address {
                assert_eq!(result, Writing::NotWriting, "address {address} already being written");
                result = if op.repair { Writing::Repair } else { Writing::Create };
            }
        };
        for op in &self.write_queue {
            scan(op);
        }
        for op in self.writes_exec.iter().flatten() {
            scan(op);
        }
        result
    }

    /// Write a block for the first time. Consumes the caller's reference on `location`;
    /// on completion the block enters the cache and a fresh stash block is delivered
    /// via [`Event::WriteDone`] (upstream replaces the caller's `*BlockPtr`).
    ///
    /// # Panics
    /// Asserts the block carries `address` in its header, the address is neither being
    /// written, read nor free.
    pub fn create_block(&mut self, storage: &mut dyn Storage, address: u64, location: u32) -> u32 {
        let header = schema::header_from_block(&self.blocks[location as usize]);
        assert_eq!(header.address, address);

        // TODO(port): src/vsr/grid.zig create_block — checkpoint/free-set block-type
        // coupling, cluster/release checks, blocks_missing hooks.
        assert_eq!(
            self.writing(address, Some(location)),
            Writing::NotWriting,
            "address {address} already being written"
        );
        assert!(!self.free_set.is_free(address));
        self.assert_not_reading(address);

        let token = self.next_write_token;
        self.next_write_token += 1;
        self.enqueue_write(
            storage,
            WriteOp { token, address, repair: false, location, sector_len: 0 },
        );
        token
    }

    /// Write a block that should already exist but maybe doesn't (disk fault or state
    /// sync miss). Consumes the caller's reference like [`Grid::create_block`].
    ///
    /// TODO(port): src/vsr/grid.zig repair_block — `blocks_missing` bookkeeping.
    ///
    /// # Panics
    /// Same asserts as [`Grid::create_block`] (the address must not be free either).
    pub fn repair_block(&mut self, storage: &mut dyn Storage, location: u32) -> u32 {
        let header = schema::header_from_block(&self.blocks[location as usize]);
        let address = header.address;

        assert_eq!(
            self.writing(address, Some(location)),
            Writing::NotWriting,
            "address {address} already being written"
        );
        assert!(!self.free_set.is_free(address));
        self.assert_not_reading(address);

        let token = self.next_write_token;
        self.next_write_token += 1;
        self.enqueue_write(
            storage,
            WriteOp { token, address, repair: true, location, sector_len: 0 },
        );
        token
    }

    fn enqueue_write(&mut self, storage: &mut dyn Storage, mut op: WriteOp) {
        let header = schema::header_from_block(&self.blocks[op.location as usize]);
        assert!(header.size > message_header::SIZE as u32);
        assert!(header.size <= BLOCK_SIZE as u32);

        // Zero sector padding (upstream write_block).
        let size = header.size as usize;
        let end = sector_ceil(size);
        self.blocks[op.location as usize][size..end].fill(0);
        op.sector_len = end;

        if let Some(slot) =
            (0..self.writes_exec.len()).find(|&slot| self.writes_exec[slot].is_none())
        {
            self.writes_exec[slot] = Some(op);
            self.submit_write(slot, storage);
            self.write_exec_order.push_back(slot);
        } else {
            self.write_queue.push_back(op);
        }
    }

    fn submit_write(&mut self, slot: usize, storage: &mut dyn Storage) {
        let op =
            self.writes_exec[slot].as_ref().unwrap_or_else(|| unreachable!("slot was just filled"));
        let buffer = self.blocks[op.location as usize][..op.sector_len].to_vec();
        storage.write_sectors(WriteRequest {
            zone: Zone::Grid,
            offset_in_zone: block_offset(op.address),
            buffer,
        });
    }

    fn complete_write(&mut self, request: &WriteRequest, storage: &mut dyn Storage) {
        let slot =
            self.write_exec_order.pop_front().unwrap_or_else(|| unreachable!("FIFO mismatch"));
        let op = self.writes_exec[slot].take().unwrap_or_else(|| unreachable!("slot was occupied"));
        debug_assert_eq!(request.offset_in_zone, block_offset(op.address));
        debug_assert_eq!(request.buffer.len(), op.sector_len);
        assert!(!self.free_set.is_free(op.address));

        // Insert the written block into the cache, then hand the caller a fresh block:
        self.cache_upsert(op.address, op.location);
        // Usually references=1, but reading from the write queue keeps it higher.
        self.block_unref(op.location);
        let fresh_location = self.get_block();

        // Start a queued write before delivering the event, so the queue is never
        // preempted (upstream ordering).
        if let Some(next) = self.write_queue.pop_front() {
            self.writes_exec[slot] = Some(next);
            self.submit_write(slot, storage);
            self.write_exec_order.push_back(slot);
        }

        self.events.push_back(Event::WriteDone {
            token: op.token,
            address: op.address,
            fresh_location,
        });
    }

    /// Fetch the block synchronously from the write queues, if possible
    /// (upstream `read_block_from_write_queues`).
    #[must_use]
    fn read_block_from_write_queues(&self, address: u64, checksum: u128) -> Option<u32> {
        assert!(address > 0);

        let mut block_found_count = 0;
        let mut found = None;
        let mut consider = |location: u32| {
            let header = schema::header_from_block(&self.blocks[location as usize]);
            if address == header.address && checksum == header.checksum {
                block_found_count += 1;
                found = Some(location);
            }
        };
        for op in &self.write_queue {
            consider(op.location);
        }
        for op in self.writes_exec.iter().flatten() {
            consider(op.location);
        }
        assert!(block_found_count <= 1);
        found
    }

    /// Fetch the block synchronously from the grid cache (or the write queues),
    /// if possible. The returned location is only valid until the next Grid write
    /// completes (upstream `read_block_from_cache`).
    ///
    /// TODO(port): src/vsr/grid.zig read_block_from_cache — cluster/release checks and
    /// the `constants.verify` disk crosscheck.
    ///
    /// # Panics
    /// Asserts `address > 0`, and, for coherent reads, that the address is acquired.
    #[must_use]
    pub fn read_block_from_cache(
        &mut self,
        address: u64,
        checksum: u128,
        coherent: bool,
    ) -> Option<u32> {
        assert!(address > 0);

        if coherent {
            assert!(!self.free_set.is_free(address));
        }

        let cache_index = self.cache.get_index(address)?;
        let cache_location = self.cache_locations[cache_index];
        let header = schema::header_from_block(&self.blocks[cache_location as usize]);
        assert_eq!(header.address, address);

        if header.checksum == checksum {
            Some(cache_location)
        } else {
            // An old version may only be cached while a newer version is being written
            // (or was learnt via state sync, TODO(port)).
            self.read_block_from_write_queues(address, checksum)
        }
    }

    /// Request `address`, expecting `checksum`. Returns a token correlating with the
    /// eventual [`Event::ReadDone`] (which may arrive without any [`Grid::poll`] when
    /// resolved from the write queue or cache).
    ///
    /// Coherent reads assert the address is acquired and park in the remote-repair
    /// queue on invalid results; non-coherent reads may probe recently released
    /// addresses and report failures directly.
    ///
    /// # Panics
    /// Asserts `address > 0` and, for coherent reads, that the address is acquired;
    /// asserts two coherent reads of the same address never request different versions.
    #[allow(clippy::too_many_lines)] // mirrors upstream tick_callback step for step
    pub fn read_block(
        &mut self,
        storage: &mut dyn Storage,
        address: u64,
        checksum: u128,
        coherent: bool,
        options: ReadOptions,
    ) -> u32 {
        assert!(address > 0);

        if coherent {
            assert!(!self.free_set.is_free(address));
        }

        // Check the write queue before checking the read queue, since otherwise:
        // 1. Read block. (coherent=false, i.e. via repair)
        // 2. Create block. (start)
        // 3. Read block again. (coherent=true)
        // We must ensure that the second read succeeds instead of queueing behind
        // the first read.
        if let Some(location) = self.read_block_from_write_queues(address, checksum) {
            let token = self.next_read_token;
            self.next_read_token += 1;
            self.events.push_back(Event::ReadDone {
                token,
                address,
                checksum,
                result: ReadBlockResult::Valid,
                valid_location: Some(location),
            });
            return token;
        }

        // Check if a read is already processing/recovering and merge with it.
        // (Don't remote-repair repairs — the block may not belong to our current
        // checkpoint, so local-storage reads skip the global queue.)
        let mut attach_to: Option<usize> = None;
        for &queued in &self.read_queue {
            self.consider_merging(queued, address, checksum, coherent, &mut attach_to);
        }
        if coherent {
            for &queued in &self.read_global_queue {
                self.consider_merging(queued, address, checksum, coherent, &mut attach_to);
            }
        }

        let token = self.next_read_token;
        self.next_read_token += 1;

        if let Some(root) = attach_to {
            let id = self.alloc_read(ReadOp {
                token,
                address,
                checksum,
                coherent,
                cache_write: options.cache_write,
                resolves: VecDeque::new(),
                state: ReadState::Attached,
            });
            let root_op =
                self.reads[root].as_mut().unwrap_or_else(|| unreachable!("queue id is live"));
            root_op.resolves.push_back(id);
            return token;
        }

        if options.cache_read
            && let Some(location) = self.read_block_from_cache(address, checksum, coherent)
        {
            self.events.push_back(Event::ReadDone {
                token,
                address,
                checksum,
                result: ReadBlockResult::Valid,
                valid_location: Some(location),
            });
            return token;
        }

        // Become the "root" read fetching the block from storage.
        let id = self.alloc_read(ReadOp {
            token,
            address,
            checksum,
            coherent,
            cache_write: options.cache_write,
            resolves: VecDeque::new(),
            state: ReadState::RootQueued,
        });
        self.read_queue.push_back(id);

        if let Some(slot) = self.read_iops_free.pop_front() {
            self.submit_read(storage, id, slot);
        } else {
            let op = self.reads[id].as_mut().unwrap_or_else(|| unreachable!("just allocated"));
            op.state = ReadState::WaitingIop;
            self.read_pending_queue.push_back(id);
        }
        token
    }

    /// Merge-decision helper for [`Grid::read_block`] (upstream inline loop).
    fn consider_merging(
        &self,
        queued: usize,
        address: u64,
        checksum: u128,
        coherent: bool,
        attach_to: &mut Option<usize>,
    ) {
        let Some(op) = self.reads[queued].as_ref() else { unreachable!("queued read is live") };
        if op.address != address {
            return;
        }
        if op.checksum == checksum {
            *attach_to = Some(queued);
        } else {
            assert!(
                !(op.coherent && coherent),
                "two different versions of block requested coherently"
            );
        }
    }

    fn alloc_read(&mut self, op: ReadOp) -> usize {
        if let Some(id) = self.reads_free.pop_front() {
            self.reads[id] = Some(op);
            id
        } else {
            self.reads.push(Some(op));
            self.reads.len() - 1
        }
    }

    fn submit_read(&mut self, storage: &mut dyn Storage, id: usize, slot: usize) {
        let address = self.reads[id].as_ref().unwrap_or_else(|| unreachable!("live read")).address;
        let op = self.reads[id].as_mut().unwrap_or_else(|| unreachable!("live read"));
        op.state = ReadState::Executing;
        self.read_exec_order.push_back((id, slot));

        storage.read_sectors(ReadRequest {
            zone: Zone::Grid,
            offset_in_zone: block_offset(address),
            buffer: vec![0u8; BLOCK_SIZE],
        });
    }

    fn complete_read(&mut self, request: &ReadRequest, storage: &mut dyn Storage) {
        let (root_id, slot) =
            self.read_exec_order.pop_front().unwrap_or_else(|| unreachable!("FIFO mismatch"));
        let old_iop_location = self.read_iop_blocks[slot];

        // The block-reference "burst": hold the current read block while acquiring
        // a fresh one for the IOP (upstream read_block_callback).
        let fresh = self.get_block();
        self.read_iop_blocks[slot] = fresh;

        let len = request.buffer.len();
        self.blocks[old_iop_location as usize][..len].copy_from_slice(&request.buffer);

        let expect = {
            let op = self.reads[root_id].as_ref().unwrap_or_else(|| unreachable!("live read"));
            ExpectBlock { address: op.address, checksum: op.checksum }
        };

        // Hand the freed IOP slot to a pending read before resolving callbacks
        // (upstream ordering).
        if let Some(next_id) = self.read_pending_queue.pop_front() {
            self.submit_read(storage, next_id, slot);
        } else {
            self.read_iops_free.push_back(slot);
        }

        let result = read_block_validate(&self.blocks[old_iop_location as usize], expect);

        // Remove the "root" read so that the address is no longer actively locked.
        self.read_queue.retain(|&id| id != root_id);

        let cache_write = {
            let op = self.reads[root_id].as_ref().unwrap_or_else(|| unreachable!("live read"));
            op.cache_write
        };
        if result == ReadBlockResult::Valid && cache_write {
            self.cache_upsert(expect.address, old_iop_location);
        }

        let valid_location = (result == ReadBlockResult::Valid).then_some(old_iop_location);
        self.resolve_read(root_id, result, valid_location);

        // The IOP block survives until the next poll so readers can copy from it
        // (module-level deviation notes).
        self.pending_reap.push(old_iop_location);
    }

    /// Resolve a root read and every read merged into it
    /// (upstream `read_block_resolve`, flattened onto the event queue).
    ///
    /// On an invalid result, coherent readers park in [`Grid::read_global_queue`]
    /// awaiting [`Grid::fulfill_block`]; non-coherent readers observe the failure
    /// directly. Upstream nests resolvers beneath the reparked root; here each parked
    /// reader waits independently — observable behavior is identical because
    /// fulfillment matches per-reader.
    fn resolve_read(
        &mut self,
        root_id: usize,
        result: ReadBlockResult,
        valid_location: Option<u32>,
    ) {
        let mut root = self.reads[root_id].take().unwrap_or_else(|| unreachable!("live read"));

        // Resolve merged reads first (upstream order).
        for resolver_id in std::mem::take(&mut root.resolves) {
            let Some(resolver) = self.reads[resolver_id].as_mut() else {
                unreachable!("attached read is live");
            };
            if resolver.coherent && result != ReadBlockResult::Valid {
                resolver.state = ReadState::ParkedGlobal;
                self.read_global_queue.push_back(resolver_id);
            } else {
                self.events.push_back(Event::ReadDone {
                    token: resolver.token,
                    address: resolver.address,
                    checksum: resolver.checksum,
                    result,
                    valid_location,
                });
                self.reads[resolver_id] = None;
                self.reads_free.push_back(resolver_id);
            }
        }

        if root.coherent && result != ReadBlockResult::Valid {
            root.state = ReadState::ParkedGlobal;
            self.reads[root_id] = Some(root);
            self.read_global_queue.push_back(root_id);
        } else {
            self.events.push_back(Event::ReadDone {
                token: root.token,
                address: root.address,
                checksum: root.checksum,
                result,
                valid_location,
            });
            self.reads_free.push_back(root_id);
        }
    }

    /// Offer a block received from another replica to parked coherent readers
    /// (upstream `fulfill_block`). Returns whether anyone was waiting for it.
    ///
    /// The block is copied; the resolved reader may copy from the copy until the next
    /// [`Grid::poll`].
    ///
    /// # Panics
    /// Asserts `block` parses as a valid block header spanning exactly [`BLOCK_SIZE`].
    pub fn fulfill_block(&mut self, block: &[u8]) -> bool {
        assert_eq!(block.len(), BLOCK_SIZE);
        let header = schema::header_from_block(block);
        // TODO(port): src/vsr/grid.zig fulfill_block — cluster/release checks against
        // the superblock working state.

        let mut matched = None;
        for &parked in &self.read_global_queue {
            let op = self.reads[parked].as_ref().unwrap_or_else(|| unreachable!("live read"));
            if op.checksum == header.checksum && op.address == header.address {
                matched = Some(parked);
                break;
            }
        }

        let Some(parked) = matched else { return false };

        self.read_global_queue.retain(|&id| id != parked);

        let location = self.get_block();
        self.blocks[location as usize].copy_from_slice(block);
        // Hold the reference until the next poll (see deviation notes).
        self.pending_reap.push(location);

        self.resolve_read(parked, ReadBlockResult::Valid, Some(location));
        true
    }

    /// Drive IO completions. Must be called repeatedly until quiescent; events are
    /// collected for [`Grid::take_events`].
    pub fn poll(&mut self, storage: &mut dyn Storage) {
        self.reap();
        while let Some(completion) = storage.next_completion() {
            match completion {
                Completion::Write(request) => self.complete_write(&request, storage),
                Completion::Read(request) => self.complete_read(&request, storage),
            }
        }
    }

    /// Drain accumulated events.
    #[must_use]
    pub fn take_events(&mut self) -> Vec<Event> {
        self.events.drain(..).collect()
    }

    fn reap(&mut self) {
        let locations = std::mem::take(&mut self.pending_reap);
        for location in locations {
            self.block_unref(location);
        }
    }

    /// # Panics
    /// Asserts `address` is not part of any pending/executing read.
    fn assert_not_reading(&self, address: u64) {
        let ids = self
            .read_queue
            .iter()
            .copied()
            .chain(self.read_pending_queue.iter().copied())
            .chain(self.read_global_queue.iter().copied())
            .chain(self.read_exec_order.iter().map(|&(id, _)| id));
        for id in ids {
            let op = self.reads[id].as_ref().unwrap_or_else(|| unreachable!("queued read"));
            assert_ne!(op.address, address, "address {address} is currently being read");
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default // typed headers keep reserved fields private
    )]

    use super::{
        BLOCK_SIZE, Event, ExpectBlock, Grid, GridOptions, ReadBlockResult, ReadOptions, Writing,
        block_offset, read_block_validate,
    };
    use crate::Zone;
    use crate::message_header::{self, TypedHeader};
    use crate::multiversion::Release;
    use crate::storage::MemoryStorage;
    use tigerbeetle_core::constants::SECTOR_SIZE;
    use tigerbeetle_lsm::free_set::SHARD_BITS;

    const CACHE_BLOCKS_COUNT: usize = 64;
    const STASH_BLOCKS_COUNT: usize = 12;
    const READ_IOPS_MAX: usize = 2;
    /// Free-set bootstrap: two shards worth of addresses (must be a multiple of
    /// `SHARD_BITS`).
    const FREE_SET_BLOCKS: usize = 2 * SHARD_BITS;

    fn grid_options(read_iops_max: usize, write_iops_max: usize) -> GridOptions {
        GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: STASH_BLOCKS_COUNT,
            read_iops_max,
            write_iops_max,
            free_set_blocks_count: Some(FREE_SET_BLOCKS),
        }
    }

    fn new_grid() -> Grid {
        Grid::new(grid_options(READ_IOPS_MAX, WRITE_IOPS_MAX))
    }

    fn new_tiny_grid() -> Grid {
        Grid::new(GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: 4,
            read_iops_max: 0,
            write_iops_max: 0,
            free_set_blocks_count: None,
        })
    }

    const WRITE_IOPS_MAX: usize = 2;

    /// A grid wired to an in-memory image of the `Grid` zone.
    struct Env {
        grid: Grid,
        storage: MemoryStorage,
    }

    impl Env {
        fn new(read_iops_max: usize, write_iops_max: usize) -> Self {
            const BLOCKS_CAPACITY: u64 = 64;
            let storage =
                MemoryStorage::new(Zone::Grid.start() + BLOCKS_CAPACITY * BLOCK_SIZE as u64);
            Self { grid: Grid::new(grid_options(read_iops_max, write_iops_max)), storage }
        }

        /// Absolute sector numbers backing `address` (for fault injection).
        fn sectors(address: u64) -> std::collections::HashSet<u64> {
            let base = Zone::Grid.start() + block_offset(address);
            (0..BLOCK_SIZE as u64 / SECTOR_SIZE as u64)
                .map(|step| (base + step * SECTOR_SIZE as u64) / SECTOR_SIZE as u64)
                .collect()
        }
    }

    /// Writes a checksum-valid block header into `buffer` and returns the block's checksum.
    fn write_block_header(buffer: &mut [u8], address: u64, body_len: usize) -> u128 {
        let mut header = message_header::Block::default();
        header.size = (message_header::SIZE + body_len) as u32;
        header.release = Release { value: 1 };
        header.address = address;
        header.block_type_ordinal = message_header::BlockType::FreeSet as u8;

        let body = vec![0xAB_u8; body_len];

        header.checksum_body = header.calculate_checksum_body(&body);
        header.set_checksum();
        buffer[..message_header::SIZE].copy_from_slice(&header.to_wire());
        buffer[message_header::SIZE..message_header::SIZE + body.len()].copy_from_slice(&body);
        header.checksum
    }

    #[test]
    fn init_layout_matches_options() {
        let grid = new_grid();
        assert_eq!(grid.blocks_count(), CACHE_BLOCKS_COUNT + STASH_BLOCKS_COUNT);
        // read_iop reservations consume READ_IOPS_MAX free stash blocks:
        assert_eq!(grid.stash_available as usize, STASH_BLOCKS_COUNT - READ_IOPS_MAX);
        assert_eq!(grid.cache_slots(), CACHE_BLOCKS_COUNT);
    }

    #[test]
    fn get_block_is_lifo_and_counts_references() {
        let mut grid = new_grid();

        let first = grid.get_block();
        let second = grid.get_block();
        assert_ne!(first, second);
        assert_eq!(grid.block_references(first), 1);
        assert_eq!(grid.block_references(second), 1);

        // LIFO reuse:
        grid.block_unref(first);
        grid.block_unref(second);
        assert_eq!(grid.get_block(), second);
        assert_eq!(grid.get_block(), first);
    }

    #[test]
    fn ref_and_unref_round_trip_with_shared_references() {
        let mut grid = new_grid();

        let location = grid.get_block();
        write_block_header(grid.block_mut(location), 7, 16);

        grid.block_ref(location);
        assert_eq!(grid.block_references(location), 2);

        grid.block_unref(location);
        assert_eq!(grid.block_references(location), 1);
        grid.block_unref(location);
        assert_eq!(grid.block_references(location), 0);
    }

    #[should_panic(expected = "stash has no free blocks")]
    #[test]
    fn get_block_panics_when_stash_is_exhausted() {
        let mut grid = new_tiny_grid();
        for _ in 0..=4 {
            grid.get_block();
        }
    }

    #[should_panic(expected = "invalid block header")]
    #[test]
    fn block_ref_rejects_invalid_header() {
        let mut grid = new_grid();
        let location = grid.get_block();
        // Garbage contents (all-zero header has an invalid checksum):
        grid.block_ref(location);
    }

    /// Publishes a fresh block for `address` into the cache and releases our reference,
    /// returning the block's location.
    fn publish(grid: &mut Grid, address: u64) -> u32 {
        let location = grid.get_block();
        write_block_header(grid.block_mut(location), address, 8);
        grid.cache_upsert(address, location);
        grid.block_unref(location);
        location
    }

    #[test]
    fn cache_upsert_keeps_inserted_address_and_distinct_locations() {
        let mut grid = new_grid();

        let mut published: Vec<(u64, u32)> = Vec::new();
        // More inserts than cache slots: eviction is guaranteed by pigeonhole, but which
        // keys are evicted depends on the hash, so assert general invariants instead.
        for address in 1..=(CACHE_BLOCKS_COUNT as u64 + CACHE_BLOCKS_COUNT as u64) {
            let location = publish(&mut grid, address);
            assert_eq!(grid.cached_location(address), Some(location));

            // Cache locations stay distinct:
            let mut seen = std::collections::HashSet::new();
            for slot in 0..grid.cache_slots() {
                let cached = grid.cache_locations[slot];
                assert!(seen.insert(cached));
            }
            published.push((address, location));
        }

        // At least one early address was evicted along the way:
        let evicted_early = published
            .iter()
            .take(CACHE_BLOCKS_COUNT)
            .filter(|&&(address, _)| grid.cached_location(address).is_none())
            .count();
        assert!(evicted_early > 0);
    }

    #[test]
    fn cache_upsert_zeroes_first_evicted_unreferenced_header() {
        let mut grid = new_grid();

        let mut cached_locations: Vec<(u64, u32)> = Vec::new();
        let mut evictee: Option<(u64, u32)> = None;
        for address in 1..=(CACHE_BLOCKS_COUNT as u64 * 2) {
            publish(&mut grid, address);

            // Find the first previously-published address that just got evicted:
            for &(old_address, old_location) in &cached_locations {
                if grid.cached_location(old_address).is_none() {
                    evictee = Some((old_address, old_location));
                    break;
                }
            }
            if let Some((_, old_location)) = evictee {
                // Its header must be zeroed (contents may remain, as upstream clears only
                // the fixed-header region):
                let bytes = grid.block(old_location);
                assert!(
                    bytes[..message_header::SIZE].iter().all(|&byte| byte == 0),
                    "evicted unreferenced block header must be zeroed"
                );
                return;
            }
            cached_locations.push((
                address,
                grid.cached_location(address).unwrap_or_else(|| unreachable!("just published")),
            ));
        }
        panic!("no eviction observed despite exceeding cache capacity");
    }

    #[test]
    fn block_offset_math() {
        assert_eq!(block_offset(1), 0);
        assert_eq!(block_offset(2), BLOCK_SIZE as u64);
        assert_eq!(block_offset(10), 9 * BLOCK_SIZE as u64);
    }

    #[test]
    fn read_block_validate_accepts_a_clean_block() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);
        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum }),
            ReadBlockResult::Valid
        );
    }

    #[test]
    fn read_block_validate_detects_corrupt_frame() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        write_block_header(&mut buffer, 3, 100);
        buffer[message_header::SIZE - 1] ^= 0xFF;

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum: 42 }),
            ReadBlockResult::InvalidChecksum
        );
    }

    #[test]
    fn read_block_validate_detects_corrupt_body() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);
        // Corrupting the body leaves the (frame-only) header checksum intact:
        buffer[message_header::SIZE] ^= 0xFF;

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum }),
            ReadBlockResult::InvalidChecksumBody
        );
    }

    #[test]
    fn read_block_validate_flags_other_commands_as_misdirected_io() {
        // A fully valid reply landed where a block was expected:
        let mut reply = message_header::Reply::default();
        reply.size = (message_header::SIZE + 8) as u32;
        reply.release = Release { value: 1 };
        reply.client = 1;
        reply.op = 1;
        reply.commit = 1;
        reply.timestamp = 1;
        reply.set_checksum();

        let mut buffer = vec![0u8; BLOCK_SIZE];
        buffer[..message_header::SIZE].copy_from_slice(&reply.to_wire());

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum: reply.checksum }),
            ReadBlockResult::UnexpectedCommand
        );
    }

    #[test]
    fn read_block_validate_flags_wrong_block() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum: checksum + 1 }),
            ReadBlockResult::UnexpectedChecksum
        );
    }

    #[test]
    fn read_block_validate_flags_dirty_padding() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);
        let padded_end = (message_header::SIZE + 100).div_ceil(SECTOR_SIZE) * SECTOR_SIZE;
        buffer[padded_end - 1] = 0x1F;

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum }),
            ReadBlockResult::InvalidPadding
        );
    }

    /// Acquires the next free block address from the grid's free set.
    fn acquire_address(grid: &mut Grid) -> u64 {
        let reservation = grid.reserve(1);
        grid.acquire(reservation)
    }

    /// Builds a checksum-valid block for `address` into a fresh stash block, returning
    /// `(location, expected bytes, checksum)`.
    fn build_block(grid: &mut Grid, address: u64) -> (u32, Vec<u8>, u128) {
        let mut expected = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut expected, address, 100);

        let location = grid.get_block();
        grid.block_mut(location).copy_from_slice(&expected);
        (location, expected, checksum)
    }

    fn read_options(cache: bool) -> ReadOptions {
        ReadOptions { cache_read: cache, cache_write: cache }
    }

    #[test]
    fn create_and_read_round_trip_through_storage() {
        let mut env = Env::new(READ_IOPS_MAX, WRITE_IOPS_MAX);
        let address = acquire_address(&mut env.grid);
        let (location, expected, checksum) = build_block(&mut env.grid, address);

        env.grid.create_block(&mut env.storage, address, location);
        env.grid.poll(&mut env.storage);

        // The write completed; the block is cached and on disk.
        assert!(env.grid.cached_location(address).is_some());
        assert_eq!(env.storage.write_ops, 1);

        // Read back from disk (cache disabled):
        let token =
            env.grid.read_block(&mut env.storage, address, checksum, false, read_options(false));
        env.grid.poll(&mut env.storage);

        let events = env.grid.take_events();
        assert_eq!(events.len(), 2, "one WriteDone and one ReadDone");
        match events[0] {
            Event::WriteDone { token: write_token, address: done_address, fresh_location } => {
                assert_eq!(write_token, 0);
                assert_eq!(done_address, address);
                assert_ne!(fresh_location, location);
            }
            other @ Event::ReadDone { .. } => panic!("unexpected first event: {other:?}"),
        }
        match events[1] {
            Event::ReadDone { token: read_token, result, valid_location, .. } => {
                assert_eq!(read_token, token);
                assert_eq!(result, ReadBlockResult::Valid);
                let location = valid_location.unwrap_or_else(|| panic!("valid read has location"));
                assert_eq!(env.grid.block(location), &expected[..]);
            }
            other @ Event::WriteDone { .. } => panic!("unexpected second event: {other:?}"),
        }
        assert_eq!(env.storage.read_ops, 1);
    }

    #[test]
    fn read_hits_write_queue_before_any_poll() {
        let mut env = Env::new(READ_IOPS_MAX, WRITE_IOPS_MAX);
        let address = acquire_address(&mut env.grid);
        let (location, _expected, checksum) = build_block(&mut env.grid, address);

        // Submitted but not yet polled:
        env.grid.create_block(&mut env.storage, address, location);

        let token =
            env.grid.read_block(&mut env.storage, address, checksum, false, read_options(false));
        let events = env.grid.take_events();
        assert!(
            matches!(
                events.as_slice(),
                [Event::ReadDone { token: t, result: ReadBlockResult::Valid, valid_location: Some(loc), .. }]
                    if *t == token && *loc == location
            ),
            "read must resolve synchronously from the write queue: {events:?}"
        );
        // No storage IO happened:
        assert_eq!(env.storage.read_ops, 0);
        assert_eq!(env.storage.write_ops, 1); // submission only
    }

    #[test]
    fn read_merges_into_root_single_storage_read() {
        let mut env = Env::new(READ_IOPS_MAX, WRITE_IOPS_MAX);
        let address = acquire_address(&mut env.grid);
        let (location, _, checksum) = build_block(&mut env.grid, address);
        env.grid.create_block(&mut env.storage, address, location);
        env.grid.poll(&mut env.storage);
        let _ = env.grid.take_events();

        let token_a =
            env.grid.read_block(&mut env.storage, address, checksum, false, read_options(false));
        let token_b =
            env.grid.read_block(&mut env.storage, address, checksum, false, read_options(false));
        env.grid.poll(&mut env.storage);

        let events = env.grid.take_events();
        assert_eq!(events.len(), 2);
        let tokens: Vec<u32> = events
            .iter()
            .map(|event| match event {
                Event::ReadDone { token, .. } => *token,
                other @ Event::WriteDone { .. } => panic!("unexpected event: {other:?}"),
            })
            .collect();
        // Upstream resolves merged reads before the root:
        assert_eq!(tokens, vec![token_b, token_a]);
        // Both reads resolved from a single storage request:
        assert_eq!(env.storage.read_ops, 1);
    }

    #[test]
    fn iop_exhaustion_pends_then_drains() {
        let mut env = Env::new(1, WRITE_IOPS_MAX);
        let address_a = acquire_address(&mut env.grid);
        let (loc_a, _, checksum_a) = build_block(&mut env.grid, address_a);
        env.grid.create_block(&mut env.storage, address_a, loc_a);
        let address_b = acquire_address(&mut env.grid);
        let (loc_b, _, checksum_b) = build_block(&mut env.grid, address_b);
        env.grid.create_block(&mut env.storage, address_b, loc_b);
        env.grid.poll(&mut env.storage);
        let _ = env.grid.take_events();

        let token_a = env.grid.read_block(
            &mut env.storage,
            address_a,
            checksum_a,
            false,
            read_options(false),
        );
        // Second read finds no free IOP slot and pends:
        let token_b = env.grid.read_block(
            &mut env.storage,
            address_b,
            checksum_b,
            false,
            read_options(false),
        );
        assert_eq!(env.storage.read_ops, 1);

        env.grid.poll(&mut env.storage);
        let events = env.grid.take_events();
        assert_eq!(events.len(), 2, "both reads drained after polling");
        assert!(
            events.iter().all(|event| matches!(
                event,
                Event::ReadDone { result: ReadBlockResult::Valid, .. }
            ))
        );
        let tokens: Vec<u32> = events
            .iter()
            .map(|event| match event {
                Event::ReadDone { token, .. } => *token,
                other @ Event::WriteDone { .. } => panic!("unexpected event: {other:?}"),
            })
            .collect();
        assert_eq!(tokens, vec![token_a, token_b]);
    }

    #[test]
    fn corrupt_disk_noncoherent_gets_invalid_result_event() {
        let mut env = Env::new(READ_IOPS_MAX, WRITE_IOPS_MAX);
        let address = acquire_address(&mut env.grid);
        let (location, _, checksum) = build_block(&mut env.grid, address);
        env.grid.create_block(&mut env.storage, address, location);
        env.grid.poll(&mut env.storage);
        let _ = env.grid.take_events();

        // Latent sector error across the whole block:
        env.storage.faulty_sectors.extend(Env::sectors(address));

        env.grid.read_block(&mut env.storage, address, checksum, false, read_options(false));
        env.grid.poll(&mut env.storage);

        let events = env.grid.take_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Event::ReadDone { result: ReadBlockResult::InvalidChecksum, valid_location: None, .. }
        ));
    }

    #[test]
    fn coherent_fault_parks_until_fulfilled_by_network_block() {
        let mut env = Env::new(READ_IOPS_MAX, WRITE_IOPS_MAX);
        let address = acquire_address(&mut env.grid);
        let (location, expected, checksum) = build_block(&mut env.grid, address);
        env.grid.create_block(&mut env.storage, address, location);
        env.grid.poll(&mut env.storage);
        let _ = env.grid.take_events();

        env.storage.faulty_sectors.extend(Env::sectors(address));

        env.grid.read_block(&mut env.storage, address, checksum, true, read_options(false));
        env.grid.poll(&mut env.storage);
        // Parked in the remote-repair queue — no resolution yet:
        assert!(env.grid.take_events().is_empty());

        // An unrelated block satisfies nobody:
        let mut stranger = vec![0u8; BLOCK_SIZE];
        write_block_header(&mut stranger, address + 7, 16);
        assert!(!env.grid.fulfill_block(&stranger));

        // The repaired block resolves the parked reader:
        assert!(env.grid.fulfill_block(&expected));
        let events = env.grid.take_events();
        assert_eq!(events.len(), 1);
        match events[0] {
            Event::ReadDone {
                result: ReadBlockResult::Valid, valid_location: Some(loc), ..
            } => {
                assert_eq!(env.grid.block(loc), &expected[..]);
            }
            ref other => panic!("unexpected event: {other:?}"),
        }
    }

    #[should_panic(expected = "already being written")]
    #[test]
    fn create_twice_same_address_panics() {
        let mut env = Env::new(READ_IOPS_MAX, WRITE_IOPS_MAX);
        let address = acquire_address(&mut env.grid);
        let (first, _, _) = build_block(&mut env.grid, address);
        env.grid.create_block(&mut env.storage, address, first);

        let (second, _, _) = build_block(&mut env.grid, address);
        env.grid.create_block(&mut env.storage, address, second);
    }

    #[test]
    fn release_demotes_from_cache_and_allows_reacquire() {
        let mut env = Env::new(READ_IOPS_MAX, WRITE_IOPS_MAX);
        let address = acquire_address(&mut env.grid);
        let (location, _, _) = build_block(&mut env.grid, address);
        env.grid.create_block(&mut env.storage, address, location);
        env.grid.poll(&mut env.storage);
        let _ = env.grid.take_events();

        assert!(env.grid.cached_location(address).is_some());
        assert_eq!(env.grid.writing(address, None), Writing::NotWriting);

        // Demotion keeps the block readable (upstream: "this does not remove the blocks
        // from the cache") — it only frees the way sooner for future inserts.
        // Released addresses also stay reserved against the current checkpoint, so the
        // next acquisition hands out a fresh block:
        env.grid.release(&[address]);
        let fresh = acquire_address(&mut env.grid);
        assert_ne!(fresh, address, "released blocks are not reusable until checkpoint");
    }

    #[test]
    fn reserve_acquire_forfeit_round_trip() {
        let mut grid = new_grid();

        // Reservations are closed by forfeiting once the caller is done (mirroring the
        // upstream `FreeSet.reserve/acquire` test); acquiring every reserved block does
        // not close the cycle by itself.
        let reservation = grid.reserve(2);
        let first = grid.acquire(reservation);
        let second = grid.acquire(reservation);
        assert_ne!(first, second);
        grid.forfeit(reservation);

        // Released addresses stay reserved against the current checkpoint:
        grid.release(&[first, second]);

        let reservation = grid.reserve(2);
        let next_first = grid.acquire(reservation);
        let next_second = grid.acquire(reservation);
        grid.forfeit(reservation);
        assert_ne!(next_first, next_second);
        assert_ne!(next_first, first);
        assert_ne!(next_first, second);
    }
}
