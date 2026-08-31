//! Port of `reference/tigerbeetle/src/lsm/manifest_log.zig` (1259 lines).
//!
//! Maintains a durable manifest log of the latest [`TableInfo`]s for every LSM tree's
//! in-memory manifest.
//!
//! Invariants:
//!
//! * Checkpointing the manifest log must flush all buffered log blocks.
//!
//! * Opening the manifest log must emit only the latest TableInfo's to be inserted.
//!
//! * The latest version of a table must never be dropped from the log through a
//!   compaction, unless the table was removed.
//!
//! * Removes that are recorded in a log block must also queue that log block for
//!   compaction.
//!
//! * Compaction must compact partially full blocks, even where it must rewrite all
//!   entries to the tail end of the log.
//!
//! * If a remove is dropped from the log, then all prior inserts/updates must already
//!   have been dropped.
//!
//! # DEVIATION
//!
//! Upstream uses callback-based async (`Grid.Read`/`Grid.Write`/`Grid.NextTick`).
//! This port uses a state-machine + `poll()`/`on_grid_event()` pattern matching our
//! Grid architecture. The caller drives lifecycle by calling `poll()` in a loop.
//!
//! Upstream stores `*Grid` and `*SuperBlock` pointers; this port receives
//! `&mut Grid` per method call and takes a [`SuperBlockView`] snapshot.
//!
//! DEVIATION: upstream's `on_next_tick` deferral for empty manifests becomes
//! `ManifestLogPhase::OpenPendingDone` — `poll()` invokes the callback on the
//! next pass.

// DEVIATION: upstream constants are comptime; Rust usize/u64 casts are safe on
// 64-bit targets and bounded by small upstream-defined limits.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::{HashMap, HashSet, VecDeque};

use tigerbeetle_core::stdx::div_ceil;
use tigerbeetle_lsm::free_set::Reservation;
use tigerbeetle_lsm::schema::manifest_node::{
    self as mn, Event as EntryEvent, Metadata, TableInfo,
};

use crate::BlockReference;
use crate::grid::{Event, Grid, ReadBlockResult, ReadOptions, SuperBlockView};
use crate::message_header::{self, BlockType, TypedHeader};
use crate::schema::{self, ManifestNode};
use crate::storage::Storage;
use crate::superblock::ManifestReferences;

// ---------------------------------------------------------------------------
// TableExtent
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TableExtent {
    block: u64,
    entry: u32,
}

// ---------------------------------------------------------------------------
// Pace
// ---------------------------------------------------------------------------

/// Compaction pacing constants (upstream `ManifestLog.Pace`).
#[derive(Clone, Copy, Debug)]
pub struct Pace {
    /// "A": max manifest blocks appended per half-bar by table compaction.
    pub half_bar_append_blocks_max: u32,
    /// "C": max manifest blocks to compact (read) per half-bar.
    pub half_bar_compact_blocks_max: u32,
    /// "T": max manifest blocks in a fully-compacted log.
    pub log_blocks_full_max: u64,
    /// Limit of MC(c) as c approaches infinity.
    pub log_blocks_cycle_max: u64,
    /// Absolute upper-bound on manifest block count.
    pub log_blocks_max: u64,
    /// Max number of live tables in the forest.
    pub tables_max: u32,
}

impl Pace {
    /// Compute pacing constants for the given tree count and table limit.
    ///
    /// # Panics
    ///
    /// Panics if `tree_count == 0`, `tables_max == 0`, `tables_max <= tree_count`,
    /// or `compact_extra_blocks == 0`. Also panics if the fixed-point iteration
    /// for `log_blocks_cycle_max` does not converge within 1024 rounds.
    #[must_use]
    pub fn init(tree_count: u32, tables_max: u32, compact_extra_blocks: u32) -> Self {
        assert!(tree_count > 0);
        assert!(tables_max > 0);
        assert!(tables_max > tree_count);
        assert!(compact_extra_blocks > 0);

        #[allow(clippy::cast_possible_truncation)] // constants fit; upstream uses comptime
        let block_entries_max = ManifestNode::ENTRY_COUNT_MAX as u64;
        #[allow(clippy::cast_possible_truncation)] // LSM_LEVELS ≤ 255
        let half_bar_compactions =
            div_ceil(tigerbeetle_core::constants::LSM_LEVELS as usize, 2) as u64;
        #[allow(clippy::cast_possible_truncation)] // LSM_GROWTH_FACTOR ≤ 16
        let compaction_tables_input_max =
            1 + u64::from(tigerbeetle_core::constants::LSM_GROWTH_FACTOR);

        let half_bar_append_entries_max =
            u64::from(tree_count) * half_bar_compactions * (compaction_tables_input_max * 3);

        let half_bar_append_blocks_max =
            div_ceil(half_bar_append_entries_max as usize, block_entries_max as usize) as u32;

        let half_bar_compact_blocks_max = half_bar_append_blocks_max + compact_extra_blocks;
        assert!(half_bar_compact_blocks_max > half_bar_append_blocks_max);

        let log_blocks_full_max = div_ceil(tables_max as usize, block_entries_max as usize) as u64;
        assert!(log_blocks_full_max > 0);

        let mut pace = Self {
            half_bar_append_blocks_max,
            half_bar_compact_blocks_max,
            log_blocks_full_max,
            tables_max,
            log_blocks_cycle_max: 0,
            log_blocks_max: 0,
        };

        pace.log_blocks_cycle_max = pace.log_blocks_cycle_max_fixed_point();

        let burst = u64::from(half_bar_append_blocks_max)
            * div_ceil(log_blocks_full_max as usize + 1, half_bar_compact_blocks_max as usize)
                as u64;
        pace.log_blocks_max = pace.log_blocks_cycle_max + burst;

        assert!(pace.log_blocks_cycle_max > pace.log_blocks_full_max);
        assert!(pace.log_blocks_max > pace.log_blocks_cycle_max);
        pace
    }

    /// Total number of blocks in the ring buffer: one for each of the three phases.
    #[must_use]
    pub fn blocks_count(&self) -> u32 {
        1 + self.half_bar_compact_blocks_max + self.half_bar_append_blocks_max
    }

    /// Number of manifest blocks to compact (read) per half-bar.
    ///
    /// # Panics
    ///
    /// Panics if `tables_count > self.tables_max`.
    #[must_use]
    pub fn half_bar_compact_blocks(&self, log_blocks_count: u32, tables_count: u32) -> u32 {
        assert!(tables_count <= self.tables_max);
        if self.log_blocks_cycle_max
            <= u64::from(log_blocks_count) + u64::from(self.half_bar_append_blocks_max)
        {
            return self.half_bar_compact_blocks_max;
        }
        let target = std::cmp::max(
            1_u64,
            self.log_blocks_cycle_max * u64::from(tables_count) / u64::from(self.tables_max),
        );
        let compact =
            u64::from(self.half_bar_compact_blocks_max) * u64::from(log_blocks_count) / target;
        #[allow(clippy::cast_possible_truncation)] // result ≤ half_bar_compact_blocks_max ≤ u32
        std::cmp::min(self.half_bar_compact_blocks_max, compact as u32)
    }

    fn log_blocks_cycle_max_fixed_point(&self) -> u64 {
        let mut before: u64 = 0;
        for _ in 0..1024 {
            let after = self.log_blocks_full_max
                + u64::from(self.half_bar_append_blocks_max)
                    * div_ceil(before as usize, self.half_bar_compact_blocks_max as usize) as u64;
            if before == after {
                return after;
            }
            before = after;
        }
        panic!("ManifestLog.Pace.log_blocks_cycle_max: no convergence");
    }
}

// ---------------------------------------------------------------------------
// ManifestLogPhase
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ManifestLogPhase {
    Idle,
    OpenReading {
        read_token: u32,
    },
    OpenPendingDone,
    CompactingReading {
        read_token: u32,
        reservation: Reservation,
        blocks_remaining: u32,
    },
    /// Compaction reads complete; awaiting `compact_end()` to forfeit reservation.
    CompactingDone {
        reservation: Reservation,
    },
    Flushing {
        writes_pending: usize,
    },
}

// ---------------------------------------------------------------------------
// ManifestLog
// ---------------------------------------------------------------------------

pub struct ManifestLog {
    superblock: SuperBlockView,
    opened: bool,
    phase: ManifestLogPhase,

    log_block_checksums: VecDeque<u128>,
    log_block_addresses: VecDeque<u64>,

    blocks: Vec<u32>,
    blocks_closed: u8,
    entry_count: u32,

    table_extents: HashMap<u64, TableExtent>,
    tables_removed: HashSet<u64>,

    /// Grid reservation covering the entire manifest ring buffer. Acquired lazily at
    /// [`init_blocks`](Self::init_blocks) (after open) and held for the log's lifetime;
    /// each open block gets an address from it (upstream `grid_reservation`,
    /// manifest_log.zig:148 / 166-168 / 691).
    grid_reservation: Option<Reservation>,

    pace: Pace,
    forest_table_count_max: u32,
    #[allow(dead_code)] // used by upstream logging; TODO(port): add logging framework
    replica_index: Option<usize>,

    pending_callback: Option<Box<dyn FnMut()>>,
    #[allow(clippy::type_complexity)]
    // upstream uses function pointer tuples; boxed closures are simplest
    open_event: Option<Box<dyn FnMut(&TableInfo)>>,
}

impl ManifestLog {
    /// Construct a new manifest log from a superblock snapshot.
    #[must_use]
    pub fn new(superblock: SuperBlockView, pace: Pace, replica_index: Option<usize>) -> Self {
        #[allow(clippy::cast_possible_truncation)] // blocks_count ≤ u32::MAX
        let blocks_count = pace.blocks_count() as usize;
        #[allow(clippy::cast_possible_truncation)] // log_blocks_max ≤ u32::MAX
        let log_blocks_max = pace.log_blocks_max as usize;

        Self {
            forest_table_count_max: pace.tables_max,
            opened: false,
            phase: ManifestLogPhase::Idle,
            log_block_checksums: VecDeque::with_capacity(log_blocks_max),
            log_block_addresses: VecDeque::with_capacity(log_blocks_max),
            blocks: Vec::with_capacity(blocks_count),
            blocks_closed: 0,
            entry_count: 0,
            table_extents: HashMap::with_capacity(pace.tables_max as usize + 1),
            tables_removed: HashSet::with_capacity(pace.tables_max as usize),
            grid_reservation: None,
            pace,
            superblock,
            replica_index,
            pending_callback: None,
            open_event: None,
        }
    }

    /// Mark the manifest ring buffer as ready for appends.
    ///
    /// Upstream pre-allocates the whole `blocks_count` ring buffer at init and reserves that
    /// many grid blocks (manifest_log.zig `init`/`grid_reservation`). This port acquires one
    /// block location per open block on demand in [`acquire_block`](Self::acquire_block) but
    /// still reserves the full ring's worth of grid addresses up front here, so address
    /// availability is guaranteed for the entire lifecycle (upstream reserves `blocks_count`).
    ///
    /// # Panics
    ///
    /// Panics if called more than once, or if blocks have already been allocated.
    pub fn init_blocks(&mut self, grid: &mut Grid) {
        assert!(self.blocks.is_empty());
        assert!(self.grid_reservation.is_none());
        self.grid_reservation = Some(grid.reserve(self.pace.blocks_count() as usize));
    }

    /// Release all grid blocks held by this manifest log.
    pub fn deinit(&mut self, grid: &mut Grid) {
        for &loc in &self.blocks {
            grid.block_unref(loc);
        }
        self.blocks.clear();
        self.log_block_checksums.clear();
        self.log_block_addresses.clear();
        self.table_extents.clear();
        self.tables_removed.clear();
    }

    /// Release all grid blocks and reset to a clean initial state.
    pub fn reset(&mut self, grid: &mut Grid) {
        for &loc in &self.blocks {
            grid.block_unref(loc);
        }
        self.blocks.clear();
        for _ in 0..self.pace.blocks_count() {
            self.blocks.push(grid.get_block());
        }
        self.log_block_checksums.clear();
        self.log_block_addresses.clear();
        self.table_extents.clear();
        self.tables_removed.clear();
        self.blocks_closed = 0;
        self.entry_count = 0;
        self.opened = false;
        self.phase = ManifestLogPhase::Idle;
        self.pending_callback = None;
        self.open_event = None;
    }

    // -----------------------------------------------------------------------
    // Open
    // -----------------------------------------------------------------------

    /// Begin opening the manifest log, reading the linked list of blocks from
    /// the superblock's oldest manifest reference.
    ///
    /// # Panics
    ///
    /// Panics if already opened, or if the phase is not idle.
    pub fn open<F, C>(&mut self, event: F, callback: C, grid: &mut Grid, storage: &mut dyn Storage)
    where
        F: FnMut(&TableInfo) + 'static,
        C: FnMut() + 'static,
    {
        assert!(!self.opened);
        assert!(matches!(self.phase, ManifestLogPhase::Idle));
        assert!(self.log_block_checksums.is_empty());
        assert!(self.blocks.is_empty());
        assert_eq!(self.blocks_closed, 0);
        assert_eq!(self.entry_count, 0);

        self.open_event = Some(Box::new(event));
        self.pending_callback = Some(Box::new(callback));

        let refs = superblock_refs(&self.superblock);
        if refs.empty() {
            self.phase = ManifestLogPhase::OpenPendingDone;
        } else {
            self.open_read_block(
                BlockReference { checksum: refs.newest_checksum, address: refs.newest_address },
                grid,
                storage,
            );
        }
    }

    fn open_read_block(
        &mut self,
        block_ref: BlockReference,
        grid: &mut Grid,
        storage: &mut dyn Storage,
    ) {
        assert!(!self.opened);
        assert!(matches!(self.phase, ManifestLogPhase::OpenReading { .. }));
        assert!(self.log_block_checksums.len() < self.log_block_checksums.capacity());
        assert!(self.blocks.is_empty());
        assert!(block_ref.address > 0);

        self.log_block_checksums.push_front(block_ref.checksum);
        self.log_block_addresses.push_front(block_ref.address);

        let token = grid.read_block(
            storage,
            block_ref.address,
            block_ref.checksum,
            true,
            ReadOptions { cache_read: true, cache_write: true },
        );
        self.phase = ManifestLogPhase::OpenReading { read_token: token };
    }

    fn open_process_block(
        &mut self,
        block_location: u32,
        grid: &mut Grid,
        storage: &mut dyn Storage,
    ) {
        assert!(!self.opened);
        assert!(matches!(self.phase, ManifestLogPhase::OpenReading { .. }));
        assert!(!self.log_block_checksums.is_empty());

        let block_checksum = self.log_block_checksums[0];
        let block_address = self.log_block_addresses[0];

        let block = grid.block(block_location);
        verify_block(block, Some(block_checksum), Some(block_address));

        let metadata = ManifestNode::metadata(block);
        let tables_used = ManifestNode::tables(block);
        assert!(metadata.entry_count > 0);
        assert!(metadata.entry_count as usize <= mn::ENTRY_COUNT_MAX);

        // Iterate entries in reverse (newest first within each block).
        for entry_index in (0..metadata.entry_count).rev() {
            let table = &tables_used[entry_index as usize];
            assert_ne!(table.label.event, EntryEvent::Reserved);
            assert!(table.address > 0);

            if table.label.event == EntryEvent::Remove {
                self.tables_removed.insert(table.address);
            } else if self.tables_removed.contains(&table.address) {
                if table.label.event == EntryEvent::Insert {
                    self.tables_removed.remove(&table.address);
                }
            } else {
                if !self.table_extents.contains_key(&table.address) {
                    self.check_tables_count();
                }
                let entry = self
                    .table_extents
                    .entry(table.address)
                    .or_insert(TableExtent { block: block_address, entry: entry_index });
                if entry.block == block_address
                    && entry.entry == entry_index
                    && let Some(ref mut cb) = self.open_event
                {
                    cb(table);
                }
            }
        }

        // Follow the linked list or finish.
        if self.superblock.manifest_oldest_address == block_address {
            assert_eq!(self.superblock.manifest_oldest_checksum, block_checksum);
            self.open_done();
        } else {
            let prev = ManifestNode::previous(grid.block(block_location));
            if let Some(previous) = prev {
                self.open_read_block(previous, grid, storage);
            } else {
                self.open_done();
            }
        }
    }

    fn open_done(&mut self) {
        assert!(!self.opened);
        assert_eq!(self.log_block_checksums.len(), self.log_block_addresses.len());
        let manifest_block_count = self.superblock.manifest_block_count;
        assert_eq!(self.log_block_checksums.len() as u32, manifest_block_count);
        assert!(self.blocks.is_empty());
        assert_eq!(self.blocks_closed, 0);
        assert_eq!(self.entry_count, 0);

        self.opened = true;
        self.open_event = None;
        self.phase = ManifestLogPhase::Idle;

        if let Some(mut cb) = self.pending_callback.take() {
            cb();
        }
    }

    // -----------------------------------------------------------------------
    // Append
    // -----------------------------------------------------------------------

    /// Append a table entry to the manifest log.
    ///
    /// # Panics
    ///
    /// Panics if the log is not opened, the table's level is out of range,
    /// or the event contradicts the current extent tracking state.
    pub fn append(&mut self, table: &TableInfo, grid: &mut Grid) {
        assert!(self.opened);
        assert!(table.label.level < tigerbeetle_core::constants::LSM_LEVELS);
        assert!(table.address > 0);
        assert!(table.snapshot_min > 0);
        assert!(table.snapshot_max > table.snapshot_min);

        match table.label.event {
            EntryEvent::Reserved => unreachable!(),
            EntryEvent::Insert => assert!(!self.table_extents.contains_key(&table.address)),
            EntryEvent::Update | EntryEvent::Remove => {
                assert!(self.table_extents.contains_key(&table.address));
            }
        }

        self.append_internal(table, grid);
    }

    fn append_internal(&mut self, table: &TableInfo, grid: &mut Grid) {
        assert!(table.label.level < tigerbeetle_core::constants::LSM_LEVELS);
        assert!(table.address > 0);

        if self.entry_count == 0 {
            // Start a fresh open block.
            self.acquire_block(grid);
        }

        let entry_count_max_u32 = mn::ENTRY_COUNT_MAX as u32;
        assert!(self.entry_count < entry_count_max_u32);

        let entry = self.entry_count as usize;
        let block_loc = *self
            .blocks
            .last()
            .unwrap_or_else(|| unreachable!("blocks must be non-empty after acquire_block"));

        {
            let block = grid.block_mut(block_loc);
            let offset = message_header::SIZE + entry * mn::ENTRY_SIZE;
            let mut entry_bytes = [0_u8; mn::ENTRY_SIZE];
            entry_bytes.copy_from_slice(&table.to_wire());
            block[offset..offset + mn::ENTRY_SIZE].copy_from_slice(&entry_bytes);
        }

        let block_address = schema::header_from_block(grid.block(block_loc)).address;

        match table.label.event {
            EntryEvent::Reserved => unreachable!(),
            EntryEvent::Insert | EntryEvent::Update => {
                if !self.table_extents.contains_key(&table.address) {
                    self.check_tables_count();
                }
                let ext = self
                    .table_extents
                    .entry(table.address)
                    .or_insert(TableExtent { block: 0, entry: 0 });
                ext.block = block_address;
                ext.entry = entry as u32;
            }
            EntryEvent::Remove => {
                self.table_extents.remove(&table.address);
            }
        }

        self.entry_count += 1;
        if self.entry_count == entry_count_max_u32 {
            self.close_block(grid);
            assert_eq!(self.entry_count, 0);
        }
    }

    fn check_tables_count(&self) {
        let count = self.table_extents.len();
        assert!(
            count <= self.forest_table_count_max as usize,
            "forest_tables_count would exceed limit \
             (tables_count={} tables_max={}) - \
             please contact the team directly who will be able to assist",
            count,
            self.forest_table_count_max,
        );
    }

    // -----------------------------------------------------------------------
    // Block lifecycle
    // -----------------------------------------------------------------------

    /// Acquire a fresh block location + address to serve as the next open block.
    ///
    /// Mirrors upstream `acquire_block` (manifest_log.zig:878-907): eagerly acquire a grid
    /// address from the reservation and initialize the block header so `close_block` and the
    /// append path can read the address back out of it.
    ///
    /// The `blocks` deque holds only the in-use block locations: closed-but-unflushed blocks
    /// in `blocks[0..blocks_closed]`, plus the current open block at the tail when
    /// `entry_count > 0`. A block is acquired from the grid here on demand and released
    /// back to the grid's stash by [`flush`](Self::flush)/`on_grid_event` after it is written.
    ///
    /// DEVIATION: upstream pre-allocates the whole `blocks_count` ring buffer at `init`,
    /// reserving that many grid blocks up front. This port reserves the same count at
    /// [`init_blocks`](Self::init_blocks) but acquires one block location per open block on
    /// demand, because the Rust `Grid` cannot be stored inside `ManifestLog` (safe-Rust
    /// aliasing) — it is threaded per call — and the `blocks.remove(0)` release pattern below
    /// expects a growing/shrinking deque.
    fn acquire_block(&mut self, grid: &mut Grid) {
        assert_eq!(self.entry_count, 0);
        assert_eq!(self.log_block_checksums.len(), self.log_block_addresses.len());
        assert_eq!(self.blocks.len(), self.blocks_closed as usize);

        // Bounded by the ring-buffer sizing that upstream guarantees via `blocks_count()`.
        assert!(self.blocks.len() < self.pace.blocks_count() as usize);

        let location = grid.get_block();
        let address = grid.acquire(self.grid_reservation.unwrap_or_else(|| {
            unreachable!("grid reservation must be set by init_blocks before any append")
        }));

        let mut header = message_header::Block::default();
        header.cluster = self.superblock.cluster;
        header.address = address;
        // The real size is fixed up by `close_block`; a valid (> HEADER size) placeholder
        // satisfies `header_from_block`'s structural asserts during the open block's appends.
        header.size = tigerbeetle_core::constants::BLOCK_SIZE as u32;
        header.block_type_ordinal = BlockType::Manifest as u8;
        header.release = self.superblock.release;

        {
            let block = grid.block_mut(location);
            block[..message_header::SIZE]
                .copy_from_slice(&message_header::TypedHeader::to_wire(&header));
        }

        self.blocks.push(location);
    }

    fn close_block(&mut self, grid: &mut Grid) {
        assert!(self.opened);
        assert_eq!(self.blocks.len(), self.blocks_closed as usize + 1);
        assert!(self.log_block_checksums.len() < self.log_block_checksums.capacity());

        let block_loc = *self
            .blocks
            .last()
            .unwrap_or_else(|| unreachable!("close_block requires an open block"));
        let entry_count = self.entry_count;
        assert!(entry_count > 0);
        assert!(entry_count as usize <= mn::ENTRY_COUNT_MAX);

        let block_size = ManifestNode::size(entry_count);

        let (checksum, address) = {
            let block = grid.block_mut(block_loc);

            let mut header = message_header::Block::from_wire(
                (&block[..message_header::SIZE])
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("block header must be SIZE bytes")),
            )
            .unwrap_or_else(|| unreachable!("block header must be valid wire format"));
            header.size = block_size;

            let prev_checksum = self.log_block_checksums.back().copied().unwrap_or(0);
            let prev_address = self.log_block_addresses.back().copied().unwrap_or(0);
            header.metadata_bytes = Metadata {
                previous_manifest_block_checksum: prev_checksum,
                previous_manifest_block_address: prev_address,
                entry_count,
            }
            .to_wire();

            block[..message_header::SIZE].copy_from_slice(&header.to_wire());
            let size = block_size as usize;
            for b in &mut block[size..] {
                *b = 0;
            }

            let mut header = message_header::Block::from_wire(
                (&block[..message_header::SIZE])
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("block header must be SIZE bytes")),
            )
            .unwrap_or_else(|| unreachable!("block header must be valid wire format"));
            header.set_checksum_body(&block[message_header::SIZE..size]);
            header.set_checksum();
            block[..message_header::SIZE].copy_from_slice(&header.to_wire());
            (header.checksum(), header.address)
        };

        verify_block(grid.block(block_loc), Some(checksum), Some(address));

        self.log_block_checksums.push_back(checksum);
        self.log_block_addresses.push_back(address);

        self.blocks_closed += 1;
        self.entry_count = 0;
    }

    // -----------------------------------------------------------------------
    // Flush
    // -----------------------------------------------------------------------

    fn flush(&mut self, grid: &mut Grid, storage: &mut dyn Storage) {
        assert!(self.opened);

        let to_write = self.blocks_closed as usize;
        if to_write == 0 {
            self.phase = ManifestLogPhase::Flushing { writes_pending: 0 };
            return;
        }

        let mut writes_pending = 0;
        for i in 0..to_write {
            let loc = self.blocks[i];
            let block = grid.block(loc);
            verify_block(block, None, None);

            let meta = ManifestNode::metadata(block);
            assert!(meta.entry_count > 0);
            let header = schema::header_from_block(block);
            assert!(header.address > 0);

            if i == to_write - 1 {
                assert!(meta.entry_count as usize <= mn::ENTRY_COUNT_MAX);
            } else {
                assert_eq!(meta.entry_count as usize, mn::ENTRY_COUNT_MAX);
            }

            grid.create_block(storage, header.address, loc);
            writes_pending += 1;
        }

        self.phase = ManifestLogPhase::Flushing { writes_pending };
    }

    // -----------------------------------------------------------------------
    // Compact
    // -----------------------------------------------------------------------

    /// Begin a compaction cycle, reading the oldest log blocks and re-appending
    /// any still-live entries.
    ///
    /// # Panics
    ///
    /// Panics if the log is not opened, or if `op < LSM_COMPACTION_OPS`.
    pub fn compact<C>(&mut self, callback: C, op: u64, grid: &mut Grid, storage: &mut dyn Storage)
    where
        C: FnMut() + 'static,
    {
        assert!(self.opened);
        assert_eq!(self.log_block_checksums.len(), self.log_block_addresses.len());
        assert!(op >= tigerbeetle_core::constants::LSM_COMPACTION_OPS as u64,);

        let compact_blocks = std::cmp::min(
            self.pace.half_bar_compact_blocks(
                self.log_block_checksums.len() as u32,
                self.table_extents.len() as u32,
            ),
            self.log_block_checksums.len().saturating_sub(self.blocks_closed as usize) as u32,
        );

        #[allow(clippy::cast_possible_truncation)]
        let reservation = grid.reserve(
            compact_blocks as usize + u64::from(self.pace.half_bar_append_blocks_max) as usize,
        );

        self.pending_callback = Some(Box::new(callback));
        self.flush(grid, storage);

        if compact_blocks == 0 {
            grid.forfeit(reservation);
            self.phase = ManifestLogPhase::Idle;
            if let Some(mut cb) = self.pending_callback.take() {
                cb();
            }
            return;
        }

        self.phase = ManifestLogPhase::CompactingReading {
            read_token: 0,
            reservation,
            blocks_remaining: compact_blocks,
        };
        self.compact_next_read(grid, storage);
    }

    fn compact_next_read(&mut self, grid: &mut Grid, storage: &mut dyn Storage) {
        let remaining = match &self.phase {
            ManifestLogPhase::CompactingReading { blocks_remaining, .. } => *blocks_remaining,
            _ => unreachable!(),
        };

        if remaining == 0 {
            self.compact_done();
            return;
        }

        let oldest_checksum = self.log_block_checksums[0];
        let oldest_address = self.log_block_addresses[0];

        let token = grid.read_block(
            storage,
            oldest_address,
            oldest_checksum,
            true,
            ReadOptions { cache_read: true, cache_write: true },
        );

        if let ManifestLogPhase::CompactingReading {
            ref mut read_token,
            ref mut blocks_remaining,
            ..
        } = self.phase
        {
            *read_token = token;
            *blocks_remaining -= 1;
        }
    }

    fn compact_process_block(
        &mut self,
        block_location: u32,
        grid: &mut Grid,
        storage: &mut dyn Storage,
    ) {
        assert!(matches!(self.phase, ManifestLogPhase::CompactingReading { .. }));

        let oldest_checksum = self
            .log_block_checksums
            .pop_front()
            .unwrap_or_else(|| unreachable!("compaction must have blocks to read"));
        let oldest_address = self
            .log_block_addresses
            .pop_front()
            .unwrap_or_else(|| unreachable!("compaction must have addresses to read"));

        let block = grid.block(block_location);
        verify_block(block, Some(oldest_checksum), Some(oldest_address));

        let meta = ManifestNode::metadata(block);
        let tables = ManifestNode::tables(block);

        for (entry_index, table) in tables.iter().enumerate() {
            let entry = entry_index as u32;
            match table.label.event {
                EntryEvent::Reserved => unreachable!(),
                EntryEvent::Insert | EntryEvent::Update => {
                    if self.table_extents.get(&table.address)
                        == Some(&TableExtent { block: oldest_address, entry })
                    {
                        // Re-append live entry.
                        self.append_internal(table, grid);
                    }
                    // else: stale — dropped.
                }
                EntryEvent::Remove => {
                    // Paired inserts already compacted — safe to drop.
                }
            }
        }

        grid.release(&[oldest_address]);
        let _ = meta; // used only in assertions above

        self.compact_next_read(grid, storage);
    }

    fn compact_done(&mut self) {
        if let ManifestLogPhase::CompactingReading { reservation, .. } =
            std::mem::replace(&mut self.phase, ManifestLogPhase::Idle)
        {
            self.phase = ManifestLogPhase::CompactingDone { reservation };
        }

        if let Some(mut cb) = self.pending_callback.take() {
            cb();
        }
    }

    /// Forfeit the grid reservation acquired at the start of `compact`.
    ///
    /// # Panics
    ///
    /// Panics if the log is not opened.
    pub fn compact_end(&mut self, grid: &mut Grid) {
        assert!(self.opened);
        if let ManifestLogPhase::CompactingDone { reservation } =
            std::mem::replace(&mut self.phase, ManifestLogPhase::Idle)
        {
            grid.forfeit(reservation);
        }
    }

    // -----------------------------------------------------------------------
    // Checkpoint
    // -----------------------------------------------------------------------

    /// Flush all pending blocks and prepare the manifest for a checkpoint.
    ///
    /// # Panics
    ///
    /// Panics if the log is not opened.
    pub fn checkpoint<C>(&mut self, callback: C, grid: &mut Grid, storage: &mut dyn Storage)
    where
        C: FnMut() + 'static,
    {
        assert!(self.opened);
        assert_eq!(self.log_block_checksums.len(), self.log_block_addresses.len());

        if self.entry_count > 0 {
            self.close_block(grid);
            assert_eq!(self.entry_count, 0);
            assert!(self.blocks_closed > 0);
        }
        assert_eq!(self.blocks_closed as usize, self.blocks.len());

        self.pending_callback = Some(Box::new(callback));
        self.flush(grid, storage);

        // When there are zero blocks to flush, `flush` sets
        // `Flushing { writes_pending: 0 }` but no `WriteDone` event fires, so
        // `pending_callback` would never be consumed. Fire it immediately,
        // mirroring the zero-blocks early-return in `compact` (line 721).
        if self.blocks_closed == 0
            && let Some(mut cb) = self.pending_callback.take()
        {
            cb();
        }
    }

    /// Return the manifest references to persist into the superblock checkpoint.
    ///
    /// # Panics
    ///
    /// Panics if the log is not opened, or if blocks are still pending.
    #[must_use]
    pub fn checkpoint_references(&self) -> ManifestReferences {
        assert!(self.opened);
        assert_eq!(self.log_block_checksums.len(), self.log_block_addresses.len());
        assert!(self.blocks.is_empty());
        assert_eq!(self.blocks_closed, 0);
        assert_eq!(self.entry_count, 0);

        if self.log_block_addresses.is_empty() {
            ManifestReferences::default()
        } else {
            ManifestReferences {
                oldest_checksum: *self
                    .log_block_checksums
                    .front()
                    .unwrap_or_else(|| unreachable!("non-empty log must have front")),
                oldest_address: *self
                    .log_block_addresses
                    .front()
                    .unwrap_or_else(|| unreachable!("non-empty log must have front")),
                newest_checksum: *self
                    .log_block_checksums
                    .back()
                    .unwrap_or_else(|| unreachable!("non-empty log must have back")),
                newest_address: *self
                    .log_block_addresses
                    .back()
                    .unwrap_or_else(|| unreachable!("non-empty log must have back")),
                block_count: self.log_block_checksums.len() as u32,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Poll
    // -----------------------------------------------------------------------

    /// Drive the state machine. Call after each `grid.poll()`.
    pub fn poll(&mut self, grid: &mut Grid, storage: &mut dyn Storage) {
        // Handle deferred phases first.
        if matches!(self.phase, ManifestLogPhase::OpenPendingDone) {
            self.open_done();
        }

        for event in grid.take_events() {
            self.on_grid_event(&event, grid, storage);
        }
    }

    fn on_grid_event(&mut self, event: &Event, grid: &mut Grid, storage: &mut dyn Storage) {
        match *event {
            Event::ReadDone { token, result, valid_location, .. } => {
                let consumed = match self.phase {
                    ManifestLogPhase::OpenReading { read_token }
                    | ManifestLogPhase::CompactingReading { read_token, .. } => read_token == token,
                    _ => false,
                };
                if consumed && matches!(result, ReadBlockResult::Valid) {
                    let loc = valid_location
                        .unwrap_or_else(|| unreachable!("valid read must provide location"));
                    match self.phase {
                        ManifestLogPhase::OpenReading { .. } => {
                            self.open_process_block(loc, grid, storage);
                        }
                        ManifestLogPhase::CompactingReading { .. } => {
                            self.compact_process_block(loc, grid, storage);
                        }
                        _ => unreachable!(),
                    }
                }
            }
            Event::WriteDone { .. } => {
                if let ManifestLogPhase::Flushing { ref mut writes_pending } = self.phase {
                    *writes_pending -= 1;
                    if *writes_pending == 0 {
                        for _ in 0..self.blocks_closed {
                            self.blocks.remove(0);
                        }
                        self.blocks_closed = 0;
                        self.phase = ManifestLogPhase::Idle;
                        if let Some(mut cb) = self.pending_callback.take() {
                            cb();
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Adapts the physical [`ManifestLog`] to the LSM layer's [`ManifestLog`] trait seam
/// (`lsm/manifest.rs`), so grooves and trees can attach it during [`Forest::open`].
///
/// The LSM trait's `append(&mut self, &WireTableInfo)` cannot carry a `&mut Grid`, while a
/// physical append must write blocks through the grid. Tree compaction to a manifest isn't
/// wired to a physical log yet (I/O dispatch deferred, see AGENTS), so this adapter's
/// `append` is a documented stub; the grid-bearing physical path remains the inherent
/// [`ManifestLog::append`].
impl tigerbeetle_lsm::manifest::ManifestLog for ManifestLog {
    fn is_opened(&self) -> bool {
        self.opened
    }

    /// DEVIATION: the LSM trait cannot thread `&mut Grid`, and physical block appends are
    /// deferred. Trees don't append to a physical manifest log during compaction in this
    /// port yet, so this is unreachable for now.
    ///
    /// TODO(port): thread the grid so compaction can append physical manifest blocks.
    fn append(&mut self, _entry: &TableInfo) {
        unreachable!("ManifestLog::append (LSM trait) is deferred pending grid threading");
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn verify_block(block: &[u8], expected_checksum: Option<u128>, expected_address: Option<u64>) {
    let header = schema::header_from_block(block);
    assert!(header.valid_checksum());
    assert!(header.valid_checksum_body(&block[message_header::SIZE..header.size as usize]));
    assert_eq!(header.block_type(), Some(BlockType::Manifest));

    if let Some(addr) = expected_address {
        assert_eq!(header.address, addr);
    }
    if let Some(cs) = expected_checksum {
        assert_eq!(header.checksum, cs);
    }

    let meta = ManifestNode::metadata(block);
    assert!(meta.entry_count > 0);
    assert!(meta.entry_count as usize <= ManifestNode::ENTRY_COUNT_MAX);
}

fn superblock_refs(view: &SuperBlockView) -> ManifestReferences {
    ManifestReferences {
        oldest_checksum: view.manifest_oldest_checksum,
        oldest_address: view.manifest_oldest_address,
        newest_checksum: 0,
        newest_address: 0,
        block_count: view.manifest_block_count,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::multiversion::Release;

    fn test_superblock() -> SuperBlockView {
        SuperBlockView {
            cluster: 0xAB,
            release: Release { value: 1 },
            storage_size: 0,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            op_compacted: false,
        }
    }

    #[test]
    fn pace_init() {
        let pace = Pace::init(
            24,
            2_300_000,
            tigerbeetle_core::constants::LSM_MANIFEST_COMPACT_EXTRA_BLOCKS as u32,
        );
        assert!(pace.half_bar_append_blocks_max > 0);
        assert!(pace.half_bar_compact_blocks_max > pace.half_bar_append_blocks_max);
        assert!(pace.log_blocks_full_max > 0);
        assert!(pace.log_blocks_cycle_max > pace.log_blocks_full_max);
        assert!(pace.log_blocks_max > pace.log_blocks_cycle_max);
    }

    #[test]
    fn pace_blocks_count() {
        let pace = Pace::init(1, 100, 5);
        assert_eq!(
            pace.blocks_count(),
            1 + pace.half_bar_compact_blocks_max + pace.half_bar_append_blocks_max
        );
    }

    #[test]
    fn pace_half_bar_compact_blocks_at_limit() {
        let pace = Pace::init(1, 100, 5);
        let result =
            pace.half_bar_compact_blocks(u32::try_from(pace.log_blocks_cycle_max).unwrap(), 50);
        assert_eq!(result, pace.half_bar_compact_blocks_max);
    }

    #[test]
    fn manifest_log_new() {
        let _log = ManifestLog::new(test_superblock(), Pace::init(1, 100, 5), Some(0));
    }
}
