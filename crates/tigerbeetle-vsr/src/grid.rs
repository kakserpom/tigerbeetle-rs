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
//! TODO(port): `src/vsr/grid.zig` — `cancel` (needs storage next-tick machinery),
//! `blocks_missing` repair bookkeeping (currently `repair_block`/`checkpoint_durable`
//! skip its hooks), `checkpoint_id`/`checkpoint_durable` stamps on reads/writes,
//! `verify_table`/`assert_coherent`/`madv_dont_dump`.
//!
//! DEVIATION: the grid does not own a superblock; the owner pushes a
//! [`SuperBlockView`] snapshot (`cluster`/`release`/`storage_size`) via
//! [`Grid::attach_superblock_view`] before `open()`/`checkpoint()` (upstream reaches
//! into `grid.superblock.working.*`).
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

use tigerbeetle_core::checksum::ChecksumStream;
use tigerbeetle_core::constants::{BLOCK_SIZE, SECTOR_SIZE};
use tigerbeetle_lsm::free_set::{FreeSet, Reservation};
use tigerbeetle_lsm::set_associative_cache::{
    Layout, SetAssociativeCache, SetAssociativeCacheSpec,
};

use crate::Zone;
use crate::checkpoint_trailer::{
    Callback, CheckpointTrailer, Chunk, TrailerType, block_count_for_trailer_size,
};
use crate::command::Command;
use crate::message_header::{self, TypedHeader};
use crate::multiversion::Release;
use crate::schema;
use crate::storage::{Completion, ReadRequest, Storage, WriteRequest};
use crate::superblock::{DATA_FILE_SIZE_MIN, TrailerReference};

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
    /// trailers inside `Grid.open()` — ported, see [`Grid::open`]). Must be a multiple
    /// of [`tigerbeetle_lsm::free_set::SHARD_BITS`].
    pub free_set_blocks_count: Option<usize>,
    /// Address-space capacity (in blocks) of the free set constructed *without* the
    /// bootstrap (`free_set_blocks_count: None`); `None` derives it from
    /// `cache_blocks_count`.
    ///
    /// DEVIATION: upstream sizes the free set from
    /// `superblock.working.vsr_state.checkpoint.storage_size`; our grid does not own
    /// the superblock (see the module docs), so the owner passes the capacity in.
    pub free_set_blocks_capacity: Option<usize>,
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
    /// [`Grid::open`] completed: the free set was loaded from its checkpoint trailers
    /// (upstream invokes the `open` callback).
    OpenDone,
    /// [`Grid::checkpoint`] completed: both free-set trailers are on disk and their
    /// references are available via [`Grid::free_set_checkpoint_references`] (upstream
    /// invokes the `checkpoint` callback; the caller then drives the superblock's own
    /// checkpoint).
    CheckpointDone,
    /// [`Grid::checkpoint_durable`] completed (upstream invokes the `checkpoint_durable`
    /// callback).
    CheckpointDurableDone,
}

/// The `(address, checksum)` pair a read expects to find
/// (upstream anonymous `expect` parameter of `read_block_validate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpectBlock {
    pub address: u64,
    pub checksum: u128,
}

/// Snapshot of the superblock working-state fields the grid reads (upstream
/// `grid.superblock.working.{cluster, vsr_state.checkpoint.release, ...}`). See the
/// module-level deviation note.
///
/// Extended with manifest-checkpoint fields used by [`crate::manifest_log`].
#[derive(Clone, Copy, Debug)]
pub struct SuperBlockView {
    pub cluster: u128,
    /// The release of the current checkpoint (upstream reads it for trailer block headers).
    pub release: Release,
    /// The data-file size of the current checkpoint (cross-checked when opening).
    pub storage_size: u64,

    // Manifest checkpoint state (used by manifest_log).
    /// The total number of manifest blocks at the last checkpoint.
    /// Used by manifest_log.open to know when to stop reading the linked list.
    pub manifest_block_count: u32,
    /// The oldest manifest block in the chain at the last checkpoint.
    /// When open reaches this block, it stops reading.
    pub manifest_oldest_address: u64,
    pub manifest_oldest_checksum: u128,
    /// The newest (most recently appended) manifest block in the chain at the
    /// last checkpoint. `manifest_log.open` starts its linked-list recovery here
    /// and walks back towards the oldest block.
    pub manifest_newest_address: u64,
    pub manifest_newest_checksum: u128,
    /// Whether the given op has already been compacted (upstream `op_compacted`).
    pub op_compacted: bool,
}

/// Which grid-owned checkpoint trailer a state-machine step applies to.
///
/// Upstream has two distinct `CheckpointTrailer` fields with duplicated callbacks; an
/// index keeps the ported step functions shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Acquired,
    Released,
}

/// Upstream `Grid.callback` tag (the callback pointers become events).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridCallback {
    Open,
    Checkpoint,
}

/// The checkpoint-trailer references a grid open/checkpoint consumes/produces
/// (upstream: the superblock working header's `free_set_reference()` pair /
/// `UpdateCheckpoint.free_set_references`).
#[derive(Clone, Copy, Debug)]
pub struct GridOpenReferences {
    pub blocks_acquired: TrailerReference,
    pub blocks_released: TrailerReference,
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

    /// Superblock working-state snapshot (see [`SuperBlockView`]); `None` until attached.
    view: Option<SuperBlockView>,
    /// Upstream `free_set_checkpoint_blocks_acquired`.
    free_set_checkpoint_blocks_acquired: CheckpointTrailer,
    /// Upstream `free_set_checkpoint_blocks_released`.
    free_set_checkpoint_blocks_released: CheckpointTrailer,
    /// The grid-level lifecycle operation in flight, if any (upstream `Grid.callback`).
    callback: Option<GridCallback>,

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
            let capacity_blocks =
                options.free_set_blocks_capacity.unwrap_or(options.cache_blocks_count);
            FreeSet::new(capacity_blocks * BLOCK_SIZE, 0)
        };

        // Upstream sizes the trailers with `free_set.encode_size_max()`.
        let trailer_buffer_size = free_set.encode_size_max();
        assert!(block_count_for_trailer_size(trailer_buffer_size as u64) > 0);

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
            view: None,
            free_set_checkpoint_blocks_acquired: CheckpointTrailer::init(
                TrailerType::FreeSet,
                trailer_buffer_size,
            ),
            free_set_checkpoint_blocks_released: CheckpointTrailer::init(
                TrailerType::FreeSet,
                trailer_buffer_size,
            ),
            callback: None,
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

    /// Whether the given address has been released (freed) in the free set.
    ///
    /// Used by compaction assertions to verify block lifetime invariants.
    #[must_use]
    pub fn free_set_is_released(&self, address: u64) -> bool {
        self.free_set.is_released(address)
    }

    /// Whether the given address is free (never allocated or fully released) in the free set.
    ///
    /// Used by compaction assertions to verify block lifetime invariants.
    #[must_use]
    pub fn free_set_is_free(&self, address: u64) -> bool {
        self.free_set.is_free(address)
    }

    /// Number of coherent reads parked awaiting a block from a remote replica
    /// (upstream `grid.read_global_queue.count()`).
    #[must_use]
    pub fn read_global_queue_len(&self) -> usize {
        self.read_global_queue.len()
    }

    /// The `(address, checksum)` pairs parked in the global repair queue, in
    /// FIFO order (upstream iterating `grid.read_global_queue`).
    #[must_use]
    pub fn global_reads(&self) -> Vec<(u64, u128)> {
        self.read_global_queue
            .iter()
            .map(|&id| {
                let read = self.reads[id].as_ref().unwrap_or_else(|| unreachable!("live read"));
                (read.address, read.checksum)
            })
            .collect()
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

    /// Mutable contents of two distinct blocks (used by the table builder, which writes the
    /// value and index blocks together during a value-block finish).
    ///
    /// # Panics
    /// Asserts the locations differ.
    pub fn blocks_mut2(&mut self, a: u32, b: u32) -> (&mut [u8], &mut [u8]) {
        assert_ne!(a, b);
        let low = u64::from(a.min(b)) as usize;
        let high = u64::from(a.max(b)) as usize;
        let (low_slice, high_slice) = self.blocks[low..=high].split_at_mut(high - low);
        // `high_slice` is the single block at index `high`.
        if a < b {
            (low_slice[0].as_mut_slice(), high_slice[0].as_mut_slice())
        } else {
            (high_slice[0].as_mut_slice(), low_slice[0].as_mut_slice())
        }
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
    /// collected for [`Grid::take_events`]. Also advances any in-flight
    /// open/checkpoint state machine (see [`Event::OpenDone`]/[`Event::CheckpointDone`]).
    pub fn poll(&mut self, storage: &mut dyn Storage) {
        self.reap();
        while let Some(completion) = storage.next_completion() {
            match completion {
                Completion::Write(request) => self.complete_write(&request, storage),
                Completion::Read(request) => self.complete_read(&request, storage),
            }
        }
        self.poll_lifecycle(storage);
    }

    /// Drain accumulated events.
    #[must_use]
    pub fn take_events(&mut self) -> Vec<Event> {
        self.events.drain(..).collect()
    }

    /// True while `Grid::checkpoint` has been started but not yet completed
    /// (its completion clears `self.callback`). Used by the forest to sequence the
    /// manifest flush and the free-set checkpoint during a single `Forest::checkpoint`.
    #[must_use]
    pub fn is_checkpoint_in_flight(&self) -> bool {
        matches!(self.callback, Some(GridCallback::Checkpoint))
    }

    /// Attach the superblock working-state snapshot used by `open`/`checkpoint`
    /// (DEVIATION: upstream reaches into `grid.superblock.working.*`; see
    /// [`SuperBlockView`]).
    ///
    /// # Panics
    /// Panics if a lifecycle operation is in flight (upstream has the same coupling via
    /// `grid.callback`).
    pub fn attach_superblock_view(&mut self, view: SuperBlockView) {
        assert!(self.callback.is_none());
        self.view = Some(view);
    }

    /// The attached superblock working-state snapshot (upstream reads it via
    /// `grid.superblock.working.*`; compaction uses it for output block headers).
    ///
    /// # Panics
    /// Panics if no view has been attached.
    #[must_use]
    pub fn superblock_view(&self) -> SuperBlockView {
        self.view_attached()
    }

    /// The free set (opened by [`Grid::open`], mutated by [`Grid::checkpoint`] et al).
    #[must_use]
    pub fn free_set(&self) -> &FreeSet {
        &self.free_set
    }

    /// The checkpoint-trailer references produced by the last [`Grid::checkpoint`]
    /// (upstream reads them off each trailer for `UpdateCheckpoint.free_set_references`).
    ///
    /// # Panics
    /// Panics if no checkpoint completed or one is in flight (upstream asserts the same
    /// inside `CheckpointTrailer.checkpoint_reference()`).
    #[must_use]
    pub fn free_set_checkpoint_references(&self) -> GridOpenReferences {
        GridOpenReferences {
            blocks_acquired: self.trailer(Slot::Acquired).checkpoint_reference(),
            blocks_released: self.trailer(Slot::Released).checkpoint_reference(),
        }
    }

    /// Load the free set from its checkpoint trailers
    /// (upstream `GridType.open(grid, references)`).
    ///
    /// Completes with [`Event::OpenDone`] once both trailers are read and decoded.
    ///
    /// # Panics
    /// Panics if a lifecycle operation is in flight, the free set is already open, or no
    /// superblock view is attached (mirrors upstream's asserts).
    pub fn open(&mut self, storage: &mut dyn Storage, references: GridOpenReferences) {
        assert!(self.callback.is_none());
        assert!(!self.free_set.opened());
        assert!(!self.free_set.checkpoint_durable());
        let _ = self.view_attached();

        self.callback = Some(GridCallback::Open);

        // Prepare both trailers before starting either one: an empty-size trailer
        // completes synchronously, and preparing upfront keeps its completion from
        // observing the other trailer unstarted (upstream defers via next-tick).
        self.trailer_prepare(Slot::Acquired, references.blocks_acquired);
        self.trailer_prepare(Slot::Released, references.blocks_released);

        self.trailer_start(
            storage,
            Slot::Acquired,
            references.blocks_acquired.last_block_address,
            references.blocks_acquired.last_block_checksum,
        );
        self.trailer_start(
            storage,
            Slot::Released,
            references.blocks_released.last_block_address,
            references.blocks_released.last_block_checksum,
        );
    }

    /// Begin a checkpoint: encode the free set into fresh trailer blocks and write them
    /// to disk (upstream `GridType.checkpoint(grid)`).
    ///
    /// Completes with [`Event::CheckpointDone`]; the references are then available via
    /// [`Grid::free_set_checkpoint_references`].
    ///
    /// # Panics
    /// Panics if a lifecycle operation is in flight, remote repairs are pending, free-set
    /// reservations are outstanding, or no superblock view is attached (mirrors upstream).
    pub fn checkpoint(&mut self, storage: &mut dyn Storage) {
        assert!(self.callback.is_none());
        assert!(self.read_global_queue.is_empty());
        assert_eq!(self.free_set.count_reservations(), 0);
        let view = self.view_attached();
        _ = view;

        // Refresh the trailer blocks, then encode the free set into them to learn the
        // encoded sizes (upstream does this per trailer before computing checksums):
        self.refresh_trailer_blocks(Slot::Acquired);
        self.refresh_trailer_blocks(Slot::Released);

        let (size_acquired, size_released) = self.encode_free_set();
        for (slot, size) in [(Slot::Acquired, size_acquired), (Slot::Released, size_released)] {
            let item_size = self.trailer(slot).trailer_type.item_size();
            assert_eq!(size % item_size as u64, 0);
            let trailer = self.trailer_mut(slot);
            trailer.size = size;
            trailer.size_transferred = 0;
        }

        self.callback = Some(GridCallback::Checkpoint);

        // Mark both trailers checkpointing before starting either (see `open` — upstream
        // relies on next-tick deferral for the same premature-join protection).
        self.trailer_mut(Slot::Acquired).callback = Callback::Checkpoint;
        self.trailer_mut(Slot::Released).callback = Callback::Checkpoint;

        self.trailer_checkpoint(storage, Slot::Acquired);
        self.trailer_checkpoint(storage, Slot::Released);
    }

    /// Roll back to the previous durable checkpoint: the just-written trailer blocks are
    /// released again (upstream `GridType.mark_checkpoint_not_durable`).
    ///
    /// Upstream releases the trailer blocks *after* marking the free set not durable,
    /// since `free_set.release()` asserts the checkpoint is not durable. Order matters:
    /// the released addresses must be freed exactly when the bitset bits clear.
    ///
    /// # Panics
    /// Panics if the current checkpoint is already non-durable.
    pub fn mark_checkpoint_not_durable(&mut self) {
        assert!(self.free_set.checkpoint_durable());
        self.free_set.mark_checkpoint_not_durable();

        let mut addresses = Vec::new();
        addresses.extend(self.trailer_addresses(Slot::Acquired));
        addresses.extend(self.trailer_addresses(Slot::Released));
        self.release(&addresses);
    }

    /// Mark the current checkpoint durable (upstream
    /// `GridType.checkpoint_durable`, which awaits repair writes first).
    ///
    /// TODO(port): src/vsr/grid.zig checkpoint_durable — upstream waits for outstanding
    /// repair writes (`blocks_missing`) here; we require quiet write queues instead.
    ///
    /// Completes with [`Event::CheckpointDurableDone`].
    ///
    /// # Panics
    /// Panics if the checkpoint is already durable or writes are in flight.
    pub fn checkpoint_durable(&mut self) {
        assert!(!self.free_set.checkpoint_durable());
        assert!(self.write_queue.is_empty());
        assert!(self.writes_exec.iter().all(Option::is_none));
        self.free_set.mark_checkpoint_durable();
        self.events.push_back(Event::CheckpointDurableDone);
    }

    fn view_attached(&self) -> SuperBlockView {
        self.view.unwrap_or_else(|| {
            unreachable!("attach_superblock_view() must precede open/checkpoint")
        })
    }

    fn trailer(&self, slot: Slot) -> &CheckpointTrailer {
        match slot {
            Slot::Acquired => &self.free_set_checkpoint_blocks_acquired,
            Slot::Released => &self.free_set_checkpoint_blocks_released,
        }
    }

    fn trailer_mut(&mut self, slot: Slot) -> &mut CheckpointTrailer {
        match slot {
            Slot::Acquired => &mut self.free_set_checkpoint_blocks_acquired,
            Slot::Released => &mut self.free_set_checkpoint_blocks_released,
        }
    }

    /// Upstream `CheckpointTrailer.open`: claim trailer state and a fresh block per
    /// buffer (all `block_count_max` of them, like upstream).
    fn trailer_prepare(&mut self, slot: Slot, reference: TrailerReference) {
        let trailer = self.trailer_mut(slot);
        assert_eq!(trailer.callback, Callback::None);
        assert!(!trailer.attached);
        assert_eq!(trailer.size, 0);
        assert_eq!(trailer.size_transferred, 0);
        assert_eq!(reference.trailer_size % trailer.trailer_type.item_size() as u64, 0);
        let block_count = block_count_for_trailer_size(reference.trailer_size);
        assert!(block_count <= trailer.block_count_max());

        trailer.attached = true;
        trailer.callback = Callback::Open;
        trailer.size = reference.trailer_size;
        trailer.checksum = reference.checksum;

        for index in 0..trailer.locations.len() {
            // Grab a fresh block for every trailer chunk buffer (upstream init loop).
            let location = self.get_block();
            self.trailer_mut(slot).locations[index] = location;
        }
        self.trailer_mut(slot).block_index = block_count;
    }

    fn trailer_start(
        &mut self,
        storage: &mut dyn Storage,
        slot: Slot,
        address: u64,
        checksum: u128,
    ) {
        assert_eq!(self.trailer(slot).callback, Callback::Open);

        if self.trailer(slot).size == 0 {
            assert_eq!(address, 0);
            assert_eq!(checksum, 0);
            // DEVIATION: completes inline instead of on the next tick; `open()` prepared
            // both trailers upfront so the premature join cannot happen.
            self.trailer_open_done(slot);
        } else {
            assert_ne!(address, 0);
            self.trailer_open_read_next(storage, slot, address, checksum);
        }
    }

    /// Read the next (previous on disk) trailer block, counting `block_index` down.
    fn trailer_open_read_next(
        &mut self,
        storage: &mut dyn Storage,
        slot: Slot,
        address: u64,
        checksum: u128,
    ) {
        assert_eq!(self.trailer(slot).callback, Callback::Open);
        assert!(address > 0);
        assert_ne!(checksum, 0);

        let block_count = self.trailer(slot).block_count();
        let trailer = self.trailer_mut(slot);
        trailer.block_index -= 1;
        let index = trailer.block_index as usize;
        trailer.addresses[index] = address;
        trailer.checksums[index] = checksum;

        // Trailer block addresses must be unique within a trailer (they are acquired
        // sequentially during the checkpoint that wrote them).
        for &other in &trailer.addresses[index + 1..block_count as usize] {
            assert_ne!(other, address);
        }

        // `.from_local_or_global_storage` ⇒ coherent, cache_read, no cache_write.
        // Safe before the free set is opened: `is_free()` conservatively answers false.
        let token = self.read_block(
            storage,
            address,
            checksum,
            true,
            ReadOptions { cache_read: true, cache_write: false },
        );
        self.trailer_mut(slot).outstanding_read = Some(token);
    }

    fn reap(&mut self) {
        let locations = std::mem::take(&mut self.pending_reap);
        for location in locations {
            self.block_unref(location);
        }
    }

    /// Upstream `open_read_next_callback`: validate, adopt the read's block, follow
    /// the linked list towards block 0.
    fn trailer_open_read_done(
        &mut self,
        storage: &mut dyn Storage,
        slot: Slot,
        valid_location: u32,
    ) {
        let (index, chunk_size) = {
            let trailer = self.trailer(slot);
            assert_eq!(trailer.callback, Callback::Open);
            assert!(trailer.size_transferred < trailer.size);
            let index = trailer.block_index as usize;
            (index, Chunk::size(trailer.block_index, trailer.block_count(), trailer.size))
        };

        // The block must be one of ours (misdirected IO would have failed validation).
        let header = schema::header_from_block(&self.blocks[valid_location as usize]);
        assert_eq!(
            message_header::BlockType::from_ordinal(header.block_type_ordinal),
            Some(self.trailer(slot).trailer_type.block_type()),
            "unexpected trailer block type"
        );

        // Adopt the read's buffer: release our held block and take a reference to the
        // freshly-read one (upstream swaps the `BlockPtr`s; the extra reference keeps
        // the read-IOP block alive past its poll-time reaping).
        let held = self.trailer(slot).locations[index];
        self.block_unref(held);
        self.block_ref(valid_location);
        self.trailer_mut(slot).locations[index] = valid_location;

        let trailer = self.trailer_mut(slot);
        trailer.size_transferred += u64::from(chunk_size);

        if let Some(next) = schema::TrailerNode::previous(&self.blocks[valid_location as usize]) {
            assert!(index > 0);
            self.trailer_open_read_next(storage, slot, next.address, next.checksum);
        } else {
            assert_eq!(index, 0);
            self.trailer_open_done(slot);
        }
    }

    /// Upstream `open_done`: verify the whole-trailer checksum.
    fn trailer_open_done(&mut self, slot: Slot) {
        {
            let trailer = self.trailer(slot);
            assert_eq!(trailer.callback, Callback::Open);
            assert_eq!(trailer.block_index, 0);
            assert_eq!(trailer.size_transferred, trailer.size);
        }

        let computed = {
            let trailer = self.trailer(slot);
            let mut stream = ChecksumStream::new();
            for chunk in Self::trailer_chunk_bodies(
                &trailer.locations,
                &self.blocks,
                trailer.block_count(),
                trailer.size,
            ) {
                stream.add(chunk);
            }
            stream.checksum()
        };
        assert_eq!(computed, self.trailer(slot).checksum, "trailer checksum mismatch");

        self.trailer_mut(slot).callback = Callback::None;
        self.open_join();
    }

    /// Encoded-trailer chunk bodies for the first `block_count` blocks
    /// (upstream derives `block_bodies[i][0..Chunk.size]`; we slice on demand).
    ///
    /// # Panics
    /// Panics unless `locations` holds at least `block_count` entries.
    fn trailer_chunk_bodies<'a>(
        locations: &[u32],
        blocks: &'a [Vec<u8>],
        block_count: u32,
        trailer_size: u64,
    ) -> Vec<&'a [u8]> {
        locations[..block_count as usize]
            .iter()
            .enumerate()
            .map(|(index, &location)| {
                let end = message_header::SIZE
                    + Chunk::size(index as u32, block_count, trailer_size) as usize;
                &blocks[location as usize][message_header::SIZE..end]
            })
            .collect()
    }

    /// Move the trailer's held blocks out of [`Grid::blocks`] so their bodies can be
    /// handed to `FreeSet::{open,encode_chunks}` without aliasing borrows; pair with
    /// [`Grid::put_trailer_blocks`]. Takes *all* `block_count_max` buffers (the
    /// encoder fills them regardless of the previous encoded size).
    fn take_trailer_blocks(&mut self, slot: Slot) -> Vec<Vec<u8>> {
        let locations = self.trailer(slot).locations.clone();
        locations.iter().map(|&l| std::mem::take(&mut self.blocks[l as usize])).collect()
    }

    fn put_trailer_blocks(&mut self, slot: Slot, taken: Vec<Vec<u8>>) {
        let locations = self.trailer(slot).locations.clone();
        assert_eq!(taken.len(), locations.len());
        for (block, &location) in taken.into_iter().zip(&locations) {
            assert!(self.blocks[location as usize].is_empty(), "taken block was not restored");
            self.blocks[location as usize] = block;
        }
    }

    /// Upstream `open_free_set_callback`: decode both trailers into the free set and
    /// cross-check against the superblock working state.
    fn open_join(&mut self) {
        assert_eq!(self.callback, Some(GridCallback::Open));
        if self.trailer(Slot::Acquired).callback == Callback::Open
            || self.trailer(Slot::Released).callback == Callback::Open
        {
            return;
        }

        let view = self.view_attached();

        let count_acquired = self.trailer(Slot::Acquired).block_count();
        let count_released = self.trailer(Slot::Released).block_count();
        let size_acquired = self.trailer(Slot::Acquired).size;
        let size_released = self.trailer(Slot::Released).size;
        let addresses_acquired =
            self.trailer(Slot::Acquired).addresses[..count_acquired as usize].to_vec();
        let addresses_released =
            self.trailer(Slot::Released).addresses[..count_released as usize].to_vec();

        // Hand the encoded chunks to the free set. The blocks are moved out and back so
        // the immutable chunk slices do not alias the `&mut self.free_set`.
        let taken_acquired = self.take_trailer_blocks(Slot::Acquired);
        let taken_released = self.take_trailer_blocks(Slot::Released);

        let encoded_acquired: Vec<&[u8]> = taken_acquired
            .iter()
            .take(count_acquired as usize)
            .enumerate()
            .map(|(index, block)| {
                let end = message_header::SIZE
                    + Chunk::size(index as u32, count_acquired, size_acquired) as usize;
                &block[message_header::SIZE..end]
            })
            .collect();
        let encoded_released: Vec<&[u8]> = taken_released
            .iter()
            .take(count_released as usize)
            .enumerate()
            .map(|(index, block)| {
                let end = message_header::SIZE
                    + Chunk::size(index as u32, count_released, size_released) as usize;
                &block[message_header::SIZE..end]
            })
            .collect();

        self.free_set.open(
            &encoded_acquired,
            &encoded_released,
            &addresses_acquired,
            &addresses_released,
        );

        self.put_trailer_blocks(Slot::Acquired, taken_acquired);
        self.put_trailer_blocks(Slot::Released, taken_released);

        let highest_address =
            addresses_acquired.iter().chain(addresses_released.iter()).copied().max().unwrap_or(0);
        if view.storage_size == 0 {
            assert_eq!(count_acquired, 0);
            assert_eq!(count_released, 0);
            assert_eq!(highest_address, 0);
        } else {
            assert_eq!(
                DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64,
                view.storage_size
            );
        }

        assert_eq!(self.free_set.count_reservations(), 0);
        // The freshly constructed (unopened) set starts empty, so everything acquired
        // now comes from the decoded trailers:
        assert!(self.free_set.count_acquired() >= count_acquired as usize);
        assert!(self.free_set.count_released() >= (count_acquired + count_released) as usize);

        self.callback = None;
        self.events.push_back(Event::OpenDone);
    }

    /// Swap every held trailer block for a fresh stash block (upstream refreshes the
    /// `BlockPtr`s at the top of `GridType.checkpoint` — all `block_count_max` of them,
    /// since the encoder fills the full buffers).
    fn refresh_trailer_blocks(&mut self, slot: Slot) {
        let count = self.trailer(slot).locations.len();
        for index in 0..count {
            let location = self.trailer(slot).locations[index];
            self.block_unref(location);
            let fresh = self.get_block();
            self.trailer_mut(slot).locations[index] = fresh;
        }
    }

    /// Encode the free set into the trailer blocks; returns
    /// `(encoded_size_blocks_acquired, encoded_size_blocks_released)`
    /// (upstream encodes per trailer inside `GridType.checkpoint`).
    fn encode_free_set(&mut self) -> (u64, u64) {
        // Move the blocks out so the mutable chunk slices do not alias the
        // `&self.free_set` borrow inside `encode_chunks`.
        let mut taken_acquired = self.take_trailer_blocks(Slot::Acquired);
        let mut taken_released = self.take_trailer_blocks(Slot::Released);

        let mut chunks_acquired: Vec<&mut [u8]> =
            taken_acquired.iter_mut().map(|block| block[message_header::SIZE..].as_mut()).collect();
        let mut chunks_released: Vec<&mut [u8]> =
            taken_released.iter_mut().map(|block| block[message_header::SIZE..].as_mut()).collect();

        let (size_acquired, size_released) =
            self.free_set.encode_chunks(&mut chunks_acquired, &mut chunks_released);

        self.put_trailer_blocks(Slot::Acquired, taken_acquired);
        self.put_trailer_blocks(Slot::Released, taken_released);

        (size_acquired as u64, size_released as u64)
    }

    /// Upstream `CheckpointTrailer.checkpoint`: checksum the encoded chunks, claim
    /// addresses and issue every trailer-block write in one pass.
    ///
    /// DEVIATION: upstream defers the zero-size completion to the next tick; here both
    /// trailers are marked checkpointing upfront, so inline completion is safe.
    fn trailer_checkpoint(&mut self, storage: &mut dyn Storage, slot: Slot) {
        assert_eq!(self.trailer(slot).callback, Callback::Checkpoint);
        assert_eq!(self.trailer(slot).size_transferred, 0);

        if self.trailer(slot).size == 0 {
            self.trailer_checkpoint_done(slot);
            return;
        }

        let view = self.view_attached();
        let block_count = self.trailer(slot).block_count();
        let trailer_size = self.trailer(slot).size;
        let block_type = self.trailer(slot).trailer_type.block_type();

        // Checksum the just-encoded chunks (upstream streams them in `grid.checkpoint`):
        let computed = {
            let trailer = self.trailer(slot);
            let mut stream = ChecksumStream::new();
            for chunk in Self::trailer_chunk_bodies(
                &trailer.locations,
                &self.blocks,
                block_count,
                trailer_size,
            ) {
                stream.add(chunk);
            }
            stream.checksum()
        };
        self.trailer_mut(slot).checksum = computed;

        let reservation = self.reserve(block_count as usize);

        for index in 0..block_count as usize {
            let address = self.acquire(reservation);

            // Undefined fields stay zeroed for deterministic output (upstream notes the
            // same about its allocator-backed buffers).
            let metadata_wire = if index == 0 {
                schema::TrailerNode::Metadata {
                    previous_trailer_block_checksum: 0,
                    previous_trailer_block_address: 0,
                }
                .to_wire()
            } else {
                let (previous_address, previous_checksum) = {
                    let trailer = self.trailer(slot);
                    (trailer.addresses[index - 1], trailer.checksums[index - 1])
                };
                schema::TrailerNode::Metadata {
                    previous_trailer_block_checksum: previous_checksum,
                    previous_trailer_block_address: previous_address,
                }
                .to_wire()
            };

            let chunk_size = Chunk::size(index as u32, block_count, trailer_size);
            let location = self.trailer(slot).locations[index];

            let block_checksum = {
                let block = &mut self.blocks[location as usize];
                let mut header = message_header::Block::default();
                header.cluster = view.cluster;
                header.metadata_bytes = metadata_wire;
                header.size = (message_header::SIZE + chunk_size as usize) as u32;
                header.address = address;
                header.snapshot = 0;
                header.release = view.release;
                header.block_type_ordinal = block_type as u8;

                let body = &block[message_header::SIZE..message_header::SIZE + chunk_size as usize];
                header.set_checksum_body(body);
                header.set_checksum();

                block[..message_header::SIZE].copy_from_slice(&header.to_wire());
                header.checksum()
            };

            {
                let trailer = self.trailer_mut(slot);
                trailer.addresses[index] = address;
                trailer.checksums[index] = block_checksum;
            }

            schema::TrailerNode::assert_valid_header(&self.blocks[location as usize]);

            let token = self.create_block(storage, address, location);
            self.trailer_mut(slot).outstanding_writes.push((token, index as u32));
        }
        self.forfeit(reservation);

        self.trailer_mut(slot).block_index = block_count;
    }

    fn trailer_checkpoint_done(&mut self, slot: Slot) {
        {
            let trailer = self.trailer(slot);
            assert_eq!(trailer.callback, Callback::Checkpoint);
            assert_eq!(trailer.block_index, trailer.block_count());
            assert_eq!(trailer.size_transferred, trailer.size);
        }
        self.trailer_mut(slot).callback = Callback::None;
        self.checkpoint_join();
    }

    fn checkpoint_join(&mut self) {
        assert_eq!(self.callback, Some(GridCallback::Checkpoint));
        assert!(self.read_global_queue.is_empty());

        if self.trailer(Slot::Acquired).callback == Callback::Checkpoint
            || self.trailer(Slot::Released).callback == Callback::Checkpoint
        {
            return;
        }

        self.callback = None;
        self.events.push_back(Event::CheckpointDone);
    }

    /// Trailer block addresses recorded by the last checkpoint.
    fn trailer_addresses(&self, slot: Slot) -> Vec<u64> {
        let trailer = self.trailer(slot);
        trailer.addresses[..trailer.block_count() as usize].to_vec()
    }

    /// Route IO completions to the in-flight lifecycle state machine
    /// (upstream dispatches through per-trailer callbacks).
    ///
    /// Events owned by a trailer are consumed; everything else is queued back in order.
    /// The loop terminates when no owned event was consumed — chained reads resolve
    /// inline into [`Grid::events`] and are picked up by the next iteration.
    fn poll_lifecycle(&mut self, storage: &mut dyn Storage) {
        if self.callback.is_none() {
            return;
        }
        loop {
            let mut advanced = false;
            let events = std::mem::take(&mut self.events);
            let mut deferred = VecDeque::with_capacity(events.len());
            for event in events {
                match event {
                    Event::ReadDone {
                        token,
                        result: ReadBlockResult::Valid,
                        valid_location: Some(valid_location),
                        ..
                    } => match self.trailer_slot_owning_read(token) {
                        Some(slot) => {
                            self.trailer_mut(slot).outstanding_read = None;
                            self.trailer_open_read_done(storage, slot, valid_location);
                            advanced = true;
                        }
                        None => deferred.push_back(event),
                    },
                    Event::WriteDone { token, fresh_location, .. } => {
                        match self.trailer_slot_owning_write(token) {
                            Some((slot, index)) => {
                                let trailer = self.trailer_mut(slot);
                                trailer.outstanding_writes.retain(|&(t, _)| t != token);
                                trailer.locations[index as usize] = fresh_location;
                                trailer.size_transferred += u64::from(Chunk::size(
                                    index,
                                    trailer.block_count(),
                                    trailer.size,
                                ));
                                if trailer.size_transferred == trailer.size {
                                    self.trailer_checkpoint_done(slot);
                                }
                                advanced = true;
                            }
                            None => deferred.push_back(event),
                        }
                    }
                    other => deferred.push_back(other),
                }
            }
            self.events.extend(deferred);
            if !advanced {
                break;
            }
        }
    }

    fn trailer_slot_owning_read(&self, token: u32) -> Option<Slot> {
        [Slot::Acquired, Slot::Released]
            .into_iter()
            .find(|&slot| self.trailer(slot).outstanding_read == Some(token))
    }

    fn trailer_slot_owning_write(&self, token: u32) -> Option<(Slot, u32)> {
        for &slot in &[Slot::Acquired, Slot::Released] {
            for &(owned_token, index) in &self.trailer(slot).outstanding_writes {
                if owned_token == token {
                    return Some((slot, index));
                }
            }
        }
        None
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
    use crate::checkpoint_trailer::block_count_for_trailer_size;
    use crate::message_header::{self, TypedHeader};
    use crate::multiversion::Release;
    use crate::storage::MemoryStorage;
    use crate::superblock::DATA_FILE_SIZE_MIN;
    use tigerbeetle_core::constants::SECTOR_SIZE;
    use tigerbeetle_lsm::free_set::SHARD_BITS;

    const CACHE_BLOCKS_COUNT: usize = 64;
    const STASH_BLOCKS_COUNT: usize = 12;
    const READ_IOPS_MAX: usize = 2;
    /// Free-set bootstrap: two shards worth of addresses (must be a multiple of
    /// `SHARD_BITS`).
    const FREE_SET_BLOCKS: usize = 2 * SHARD_BITS;

    /// Storage image size for the single-block-trailer round trip.
    const STORAGE_BLOCKS_CAPACITY: u64 = 256;

    /// Stash headroom for two multi-block trailers plus IO slots.
    const MULTI_BLOCK_STASH: usize = 32;

    fn grid_options(read_iops_max: usize, write_iops_max: usize) -> GridOptions {
        GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: STASH_BLOCKS_COUNT,
            read_iops_max,
            write_iops_max,
            free_set_blocks_count: Some(FREE_SET_BLOCKS),
            free_set_blocks_capacity: None,
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
            free_set_blocks_capacity: None,
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
            other => panic!("unexpected first event: {other:?}"),
        }
        match events[1] {
            Event::ReadDone { token: read_token, result, valid_location, .. } => {
                assert_eq!(read_token, token);
                assert_eq!(result, ReadBlockResult::Valid);
                let location = valid_location.unwrap_or_else(|| panic!("valid read has location"));
                assert_eq!(env.grid.block(location), &expected[..]);
            }
            other => panic!("unexpected second event: {other:?}"),
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
                other => panic!("unexpected event: {other:?}"),
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
                other => panic!("unexpected event: {other:?}"),
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

    /// A grid whose free set is *not* opened yet, sized for `blocks_capacity` addresses.
    fn new_unopened_grid(blocks_capacity: usize) -> Grid {
        Grid::new(GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: STASH_BLOCKS_COUNT,
            read_iops_max: READ_IOPS_MAX,
            write_iops_max: WRITE_IOPS_MAX,
            free_set_blocks_count: None,
            free_set_blocks_capacity: Some(blocks_capacity),
        })
    }

    fn empty_reference() -> crate::superblock::TrailerReference {
        crate::superblock::TrailerReference {
            checksum: tigerbeetle_core::checksum::checksum(&[]),
            last_block_address: 0,
            last_block_checksum: 0,
            trailer_size: 0,
        }
    }

    fn empty_references() -> super::GridOpenReferences {
        super::GridOpenReferences {
            blocks_acquired: empty_reference(),
            blocks_released: empty_reference(),
        }
    }

    /// Polls until `want` matches a drained event (upstream: the callback fires).
    fn drive_until(grid: &mut Grid, storage: &mut MemoryStorage, want: &dyn Fn(&Event) -> bool) {
        for _ in 0..1000 {
            grid.poll(storage);
            if grid.take_events().iter().any(want) {
                return;
            }
        }
        panic!("expected lifecycle event never arrived");
    }

    #[test]
    fn open_with_empty_trailers_loads_an_empty_free_set() {
        let mut storage = MemoryStorage::new(Zone::Grid.start() + 64 * BLOCK_SIZE as u64);
        let mut grid = new_unopened_grid(FREE_SET_BLOCKS);

        grid.attach_superblock_view(super::SuperBlockView {
            cluster: 0xAB,
            release: Release { value: 7 },
            storage_size: 0,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        });
        grid.open(&mut storage, empty_references());
        drive_until(&mut grid, &mut storage, &|event| matches!(event, Event::OpenDone));

        assert!(grid.free_set().opened());
        assert!(!grid.free_set().checkpoint_durable());
        assert_eq!(grid.free_set().count_acquired(), 0);
        assert_eq!(grid.free_set().count_free(), FREE_SET_BLOCKS);
    }

    #[test]
    fn open_panics_without_attached_view_or_when_opened() {
        let mut storage = MemoryStorage::new(Zone::Grid.start() + 64 * BLOCK_SIZE as u64);

        let mut grid = new_unopened_grid(FREE_SET_BLOCKS);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                grid.open(&mut storage, empty_references());
            }))
            .is_err(),
            "open without a superblock view must panic"
        );

        let mut bootstrapped = new_grid();
        bootstrapped.attach_superblock_view(super::SuperBlockView {
            cluster: 0xAB,
            release: Release { value: 7 },
            storage_size: 0,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        });
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bootstrapped.open(&mut storage, empty_references());
            }))
            .is_err(),
            "open on an already-open free set must panic"
        );
    }

    #[test]
    fn checkpoint_round_trip_reopens_from_storage() {
        const USER_BLOCKS: usize = 8;

        let cluster: u128 = 0xAB_CD;
        let release = Release { value: 7 };

        let mut storage =
            MemoryStorage::new(Zone::Grid.start() + STORAGE_BLOCKS_CAPACITY * BLOCK_SIZE as u64);

        // Grid #1: start from an unopened free set and go through the full lifecycle —
        // open (empty trailers, like a fresh format), mark durable, acquire, checkpoint.
        let mut grid = new_unopened_grid(FREE_SET_BLOCKS);
        grid.attach_superblock_view(super::SuperBlockView {
            cluster,
            release,
            storage_size: 0,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        });
        grid.open(&mut storage, empty_references());
        drive_until(&mut grid, &mut storage, &|event| matches!(event, Event::OpenDone));
        grid.checkpoint_durable();

        let reservation = grid.reserve(USER_BLOCKS);
        let acquired: Vec<u64> = (0..USER_BLOCKS).map(|_| grid.acquire(reservation)).collect();
        grid.forfeit(reservation);
        assert_eq!(acquired, (1..=USER_BLOCKS as u64).collect::<Vec<u64>>());

        grid.checkpoint(&mut storage);
        drive_until(&mut grid, &mut storage, &|event| matches!(event, Event::CheckpointDone));

        let references = grid.free_set_checkpoint_references();
        // The released trailer stays empty — nothing was released yet (upstream skips
        // encoding trailing zero runs).
        let blocks_acquired_trailer =
            block_count_for_trailer_size(references.blocks_acquired.trailer_size) as usize;
        let blocks_released_trailer =
            block_count_for_trailer_size(references.blocks_released.trailer_size) as usize;
        assert!(blocks_acquired_trailer > 0);
        assert_eq!(references.blocks_released.trailer_size, 0);
        assert_eq!(
            grid.free_set().count_acquired(),
            USER_BLOCKS + blocks_acquired_trailer + blocks_released_trailer
        );

        // What upstream's superblock checkpoint records as the new storage size:
        let highest_address = references
            .blocks_acquired
            .last_block_address
            .max(references.blocks_released.last_block_address);
        let storage_size_new = DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64;

        // Grid #2: reopen the same storage from the checkpoint references.
        let mut reopened = new_unopened_grid(FREE_SET_BLOCKS);
        reopened.attach_superblock_view(super::SuperBlockView {
            cluster,
            release,
            storage_size: storage_size_new,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        });
        reopened.open(&mut storage, references);
        drive_until(&mut reopened, &mut storage, &|event| matches!(event, Event::OpenDone));

        assert_eq!(reopened.free_set().count_acquired(), USER_BLOCKS + blocks_acquired_trailer);
        for address in 1..=USER_BLOCKS as u64 {
            assert!(!reopened.free_set().is_free(address));
        }
        // The trailer blocks themselves load released (they are rewritten by the next
        // checkpoint before they can be freed) — released implies acquired:
        assert_eq!(
            reopened.free_set().count_released(),
            blocks_acquired_trailer + blocks_released_trailer
        );

        // The loaded checkpoint starts non-durable; run the replica commit-pipeline
        // sequence: durability flip moves the loaded trailer addresses from the staged
        // bucket into `blocks_released`, the next checkpoint rolls them back and
        // rewrites them, and the following flip frees them.
        assert!(!reopened.free_set().checkpoint_durable());
        reopened.checkpoint_durable();
        assert!(reopened.free_set().checkpoint_durable());
        assert_eq!(
            reopened.free_set().count_released(),
            blocks_acquired_trailer + blocks_released_trailer,
            "staged trailer addresses become released"
        );

        // Full cycle, in the replica commit-pipeline order: write the next generation
        // of trailers, roll back the just-superseded checkpoint (releasing the new
        // trailer addresses into the staged bucket), then flip durability — freeing the
        // previous generation and moving the staged releases into `blocks_released`.
        reopened.checkpoint(&mut storage);
        drive_until(&mut reopened, &mut storage, &|event| matches!(event, Event::CheckpointDone));
        let references_second = reopened.free_set_checkpoint_references();
        let blocks_acquired_second =
            block_count_for_trailer_size(references_second.blocks_acquired.trailer_size) as usize;
        let blocks_released_second =
            block_count_for_trailer_size(references_second.blocks_released.trailer_size) as usize;

        reopened.mark_checkpoint_not_durable();
        assert!(!reopened.free_set().checkpoint_durable());

        reopened.checkpoint_durable();
        assert_eq!(
            reopened.free_set().count_released(),
            blocks_acquired_second + blocks_released_second,
            "previous generation freed, current generation staged"
        );
        assert_eq!(
            reopened.free_set().count_acquired(),
            USER_BLOCKS + blocks_acquired_second + blocks_released_second
        );
    }

    #[test]
    fn checkpoint_round_trip_with_multi_block_trailers() {
        // One released address per 64-block word keeps every covered word hybrid-
        // incompressible (neither all-zero nor all-one), so the released trailer
        // exceeds one chunk (`CHUNK_SIZE_MAX`) and spans several blocks. The last
        // shard stays entirely free — room for the trailer block reservations.
        const CAPACITY: usize = 9 * SHARD_BITS;
        const ACQUIRED_SPAN: usize = 8 * SHARD_BITS;
        const WORDS: usize = ACQUIRED_SPAN / 64;
        const { assert!(WORDS * 10 > crate::checkpoint_trailer::CHUNK_SIZE_MAX) };

        let cluster: u128 = 0xFE_ED;
        let release = Release { value: 3 };

        // The image covers the whole address space; only the trailer blocks are ever
        // written to it (bitset operations never touch storage).
        let mut storage =
            MemoryStorage::new(Zone::Grid.start() + CAPACITY as u64 * BLOCK_SIZE as u64);

        let mut grid = Grid::new(GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: MULTI_BLOCK_STASH,
            read_iops_max: READ_IOPS_MAX,
            write_iops_max: WRITE_IOPS_MAX,
            free_set_blocks_count: None,
            free_set_blocks_capacity: Some(CAPACITY),
        });
        grid.attach_superblock_view(super::SuperBlockView {
            cluster,
            release,
            storage_size: 0,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        });
        grid.open(&mut storage, empty_references());
        drive_until(&mut grid, &mut storage, &|event| matches!(event, Event::OpenDone));
        grid.checkpoint_durable();

        // Acquire everything but the free tail, then release one address per word:
        let reservation = grid.reserve(ACQUIRED_SPAN);
        for _ in 0..ACQUIRED_SPAN {
            let _ = grid.acquire(reservation);
        }
        grid.forfeit(reservation);

        let mut scattered = Vec::with_capacity(WORDS);
        for word in 0..WORDS {
            scattered.push((word * 64 + 1) as u64);
        }
        grid.release(&scattered);

        grid.checkpoint(&mut storage);
        drive_until(&mut grid, &mut storage, &|event| matches!(event, Event::CheckpointDone));

        let references = grid.free_set_checkpoint_references();
        assert!(
            references.blocks_released.trailer_size
                > crate::checkpoint_trailer::CHUNK_SIZE_MAX as u64,
            "expected multi-block released trailer, size={}",
            references.blocks_released.trailer_size
        );

        let highest_address = references
            .blocks_acquired
            .last_block_address
            .max(references.blocks_released.last_block_address);
        let storage_size_new = DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64;

        let mut reopened = Grid::new(GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: MULTI_BLOCK_STASH,
            read_iops_max: READ_IOPS_MAX,
            write_iops_max: WRITE_IOPS_MAX,
            free_set_blocks_count: None,
            free_set_blocks_capacity: Some(CAPACITY),
        });
        reopened.attach_superblock_view(super::SuperBlockView {
            cluster,
            release,
            storage_size: storage_size_new,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        });
        reopened.open(&mut storage, references);
        drive_until(&mut reopened, &mut storage, &|event| matches!(event, Event::OpenDone));

        let trailer_blocks_total =
            block_count_for_trailer_size(references.blocks_acquired.trailer_size) as usize
                + block_count_for_trailer_size(references.blocks_released.trailer_size) as usize;
        assert_eq!(reopened.free_set().count_released(), scattered.len() + trailer_blocks_total);
        for &address in &scattered[..8] {
            assert!(!reopened.free_set().is_free(address));
            assert!(!reopened.free_set().is_free(address + 1));
        }

        // Durability flip: frees the previous checkpoint's released blocks (the
        // scattered ones) and stages the loaded trailer addresses into
        // `blocks_released` (freed by the next flip).
        reopened.checkpoint_durable();
        assert_eq!(reopened.free_set().count_released(), trailer_blocks_total);
        for &address in &scattered[..8] {
            assert!(reopened.free_set().is_free(address));
        }
    }
}
