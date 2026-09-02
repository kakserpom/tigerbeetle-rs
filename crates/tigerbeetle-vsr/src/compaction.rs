//! Compaction: moves or merges a table's values from the previous level.
//!
//! Port of `src/lsm/compaction.zig` (`CompactionType`).
//!
//! # Differences from upstream
//!
//! - **No ResourcePool:** The resource pool (block allocation, IOPS management) is deferred
//!   until the I/O subsystem is fully ported (Phase 3). Block fields are `()` placeholders.
//! - **No I/O dispatch:** `compaction_dispatch`, `compaction_dispatch_enter`,
//!   `read_index_block`, `read_value_block`, `write_value_block`, and all callbacks are
//!   deferred. The lifecycle methods (`half_bar_commence`, `half_bar_complete`, `beat_commence`)
//!   are stubs.
//! - **No Tracer:** `grid.trace.count/start/stop` calls are deferred.
//! - **Compaction uses raw pointers:** `*Tree<S>` and `*Grid` are borrowed mutably; upstream
//!   stores `*Tree` and `*Grid` pointers too, so this matches the ownership model.
//! - **ImmutableTableIterator lifetime:** The iterator is stored inline; upstream stores it in a
//!   union. Rust's ownership makes the union unnecessary — the enum variant owns the iterator.
//! - **No log crate:** upstream `log.debug(...)` calls are deferred.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_panics_doc)] // liberally asserting like upstream
#![allow(clippy::manual_assert_eq)] // upstream uses `assert(a == b)` style
#![allow(clippy::struct_excessive_bools)] // matches upstream struct layout
#![allow(clippy::too_many_lines)] // lifecycle methods are long, matching upstream
#![allow(clippy::too_many_arguments)] // matching upstream parameter lists

use core::cmp::Ordering;

use tigerbeetle_core::constants;
use tigerbeetle_lsm::compaction::{
    COMPACTION_TABLES_INPUT_MAX, COMPACTION_TABLES_OUTPUT_MAX, snapshot_min_for_table_output,
};
use tigerbeetle_lsm::direction::Direction;
use tigerbeetle_lsm::free_set::Reservation;
use tigerbeetle_lsm::manifest::{CompactionRange, TableInfoReference, TreeTableInfo};
use tigerbeetle_lsm::table_memory::{
    ImmutableTableIterator, Mutability, Table as TableTrait, Usage,
};

use crate::grid::{Event, Grid};
use crate::storage::Storage;
use crate::table::{BuilderState, DataFinishOptions, IndexFinishOptions, TableBuilder};
use crate::tree::TreeSpec;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Port of upstream `compaction_tables_input_max`.
pub const COMPACTION_TABLES_INPUT_MAX_VSR: usize = COMPACTION_TABLES_INPUT_MAX;

/// Port of upstream `compaction_tables_output_max`.
pub const COMPACTION_TABLES_OUTPUT_MAX_VSR: usize = COMPACTION_TABLES_OUTPUT_MAX;

// ---------------------------------------------------------------------------
// CompactionCounters
// ---------------------------------------------------------------------------

/// Physical IO counters for a compaction (upstream `CompactionCounters`).
///
/// Counters track physical IO and are not fully deterministic across replicas.
/// `in` and `dropped` values can vary between replicas.
///
/// Accounting equation: `out == in - dropped`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionCounters {
    pub in_: u64,
    pub dropped: u64,
    pub out: u64,
}

impl CompactionCounters {
    /// Returns `true` if the accounting equation holds.
    #[must_use]
    pub const fn consistent(self) -> bool {
        self.out == self.in_ - self.dropped
    }
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// Cursor position within level A or B input blocks (upstream `Position`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Position {
    pub index_block: u32,
    pub value_block: u32,
    pub value: u32,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Compaction lifecycle stage (upstream `stage` enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Inactive,
    Beat,
    Paused,
}

/// Progress through the immutable table within a half bar (upstream `level_a_immutable_stage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelAImmutableStage {
    Ready,
    Merge,
    Exhausted,
}

/// Manifest log append operation (upstream `manifest_entries` operation field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestEntryOperation {
    InsertToLevelB,
    MoveToLevelB,
}

/// A queued manifest log entry (upstream entry in `manifest_entries` BoundedArray).
#[derive(Clone, Copy, Debug)]
pub struct ManifestEntry<K: Copy> {
    pub operation: ManifestEntryOperation,
    pub table: TreeTableInfo<K>,
}

/// Discriminates whether level A input is the in-memory immutable table or a disk table
/// (upstream `TableInfoA` tagged union).
///
/// DEVIATION: upstream stores `ImmutableTableIterator` inline in the union. In Rust, the
/// iterator borrows from the tree's immutable table, creating a self-referential lifetime.
/// Instead, this port tracks the variant and re-creates the iterator when needed during
/// the dispatch loop.
pub enum TableInfoA<K: Copy> {
    /// Level A is the immutable table (compact into level 0).
    Immutable,
    /// Level A is a disk table.
    Disk(TableInfoReference<TreeTableInfo<K>>),
}

// ---------------------------------------------------------------------------
// Quotas
// ---------------------------------------------------------------------------

/// Logical progress quotas for compaction pacing (upstream `quotas` struct).
///
/// Quotas track logical progress and determine pacing. They must be deterministic
/// across replicas (unlike physical IO counters).
#[derive(Clone, Copy, Debug, Default)]
pub struct Quotas {
    pub beat: u64,
    pub beat_done: u64,
    pub half_bar: u64,
    pub half_bar_done: u64,
}

impl Quotas {
    /// Returns `true` if the beat quota is exhausted.
    #[must_use]
    pub fn beat_exhausted(&self) -> bool {
        assert!(self.beat_done <= self.half_bar_done);
        assert!(self.half_bar_done <= self.half_bar);
        self.beat_done >= self.beat
    }

    /// Returns `true` if the half-bar quota is exhausted.
    #[must_use]
    pub fn half_bar_exhausted(&self) -> bool {
        assert!(self.half_bar_done <= self.half_bar);
        self.half_bar_done == self.half_bar
    }
}

// ---------------------------------------------------------------------------
// Merge / Copy result types
// ---------------------------------------------------------------------------

/// Result of a copy-with-tombstone-drop operation (upstream `CopyDropTombstonesResult`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CopyDropTombstonesResult {
    pub consumed: u32,
    pub dropped: u32,
    pub produced: u32,
}

/// Result of a merge operation (upstream `MergeResult`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MergeResult {
    pub consumed_a: u32,
    pub consumed_b: u32,
    pub dropped: u32,
    pub produced: u32,
}

// ---------------------------------------------------------------------------
// Pure value functions — hot CPU loops for the data plane
// ---------------------------------------------------------------------------

/// Copy values from `source` to `target`, returning the number copied.
///
/// Port of upstream `values_copy`.
pub fn values_copy<V: Copy>(target: &mut [V], source: &[V]) -> u32 {
    assert!(!source.is_empty());
    assert!(!target.is_empty());

    let len = source.len().min(target.len()) as u32;
    target[..len as usize].copy_from_slice(&source[..len as usize]);
    len
}

/// Copy values from an immutable table iterator to `target`, up to `budget_iterator` values.
///
/// Port of upstream `values_copy_immutable`.
pub fn values_copy_immutable<S: TableTrait>(
    target: &mut [S::Value],
    iterator: &mut ImmutableTableIterator<'_, S>,
    budget_iterator: u32,
) -> u32 {
    assert!(!target.is_empty());

    let mut index_target: u32 = 0;
    while index_target < budget_iterator && index_target < target.len() as u32 {
        let Some(value_in) = iterator.pop() else {
            break;
        };
        target[index_target as usize] = value_in;
        index_target += 1;
    }

    index_target
}

/// Copy values from `source` to `target`, dropping tombstones.
///
/// Port of upstream `values_copy_drop_tombstones`.
pub fn values_copy_drop_tombstones<S: TableTrait>(
    target: &mut [S::Value],
    source: &[S::Value],
) -> CopyDropTombstonesResult {
    assert!(!source.is_empty());
    assert!(!target.is_empty());

    let mut index_source: usize = 0;
    let mut index_target: usize = 0;

    while index_source < source.len() && index_target < target.len() {
        let value_in = &source[index_source];
        index_source += 1;
        if S::tombstone(value_in) {
            assert!(S::USAGE != Usage::SecondaryIndex);
            continue;
        }
        target[index_target] = *value_in;
        index_target += 1;
    }

    let result = CopyDropTombstonesResult {
        consumed: index_source as u32,
        dropped: (index_source - index_target) as u32,
        produced: index_target as u32,
    };
    assert!(result.consumed > 0);
    assert!(result.consumed <= source.len() as u32);
    assert!(result.dropped <= result.consumed);
    assert!(result.produced <= target.len() as u32);
    assert!(result.produced == result.consumed - result.dropped);
    result
}

/// Copy values from an immutable table iterator to `target`, dropping tombstones.
///
/// Port of upstream `values_copy_drop_tombstones_immutable`.
pub fn values_copy_drop_tombstones_immutable<S: TableTrait>(
    target: &mut [S::Value],
    iterator: &mut ImmutableTableIterator<'_, S>,
    budget_iterator: u32,
) -> CopyDropTombstonesResult {
    assert!(!target.is_empty());

    let remaining_before = iterator.count_remaining();
    let dropped_before = iterator.count_dropped();

    let mut index_source: u32 = 0;
    let mut index_target: u32 = 0;

    while index_source < budget_iterator && index_target < target.len() as u32 {
        let Some(value_in) = iterator.pop() else {
            break;
        };
        index_source += 1;
        if S::tombstone(&value_in) {
            assert!(S::USAGE != Usage::SecondaryIndex);
            continue;
        }
        target[index_target as usize] = value_in;
        index_target += 1;
    }

    let consumed_iterator = remaining_before - iterator.count_remaining();
    let dropped_iterator = iterator.count_dropped() - dropped_before;

    let result = CopyDropTombstonesResult {
        consumed: consumed_iterator,
        dropped: (index_source - index_target) + dropped_iterator,
        produced: index_target,
    };
    assert!(result.dropped <= result.consumed);
    assert!(result.produced <= target.len() as u32);
    assert!(result.produced == result.consumed - result.dropped);
    result
}

/// Merge values from level A (immutable iterator) and level B (slice), with level A taking
/// precedence on equal keys. Tombstones may be dropped.
///
/// Port of upstream `values_merge_immutable`.
pub fn values_merge_immutable<S: TableTrait>(
    target: &mut [S::Value],
    iterator_a: &mut ImmutableTableIterator<'_, S>,
    source_b: &[S::Value],
    drop_tombstones: bool,
    budget_iterator: u32,
) -> MergeResult {
    assert!(!source_b.is_empty());
    assert!(!target.is_empty());

    let remaining_before = iterator_a.count_remaining();
    let dropped_before = iterator_a.count_dropped();

    let mut index_source_a: usize = 0;
    let mut index_source_b: usize = 0;
    let mut index_target: usize = 0;

    while index_source_a < budget_iterator as usize
        && index_source_b < source_b.len()
        && index_target < target.len()
    {
        let Some(key_a) = iterator_a.peek() else {
            break;
        };
        let value_b = &source_b[index_source_b];

        match key_a.cmp(&S::key_from_value(value_b)) {
            Ordering::Less => {
                // Pick value from level a.
                index_source_a += 1;
                let Some(value_a) = iterator_a.pop() else {
                    break;
                };
                if drop_tombstones && S::tombstone(&value_a) {
                    assert!(S::USAGE != Usage::SecondaryIndex);
                    continue;
                }
                target[index_target] = value_a;
                index_target += 1;
            }
            Ordering::Greater => {
                // Pick value from level b.
                index_source_b += 1;
                target[index_target] = *value_b;
                index_target += 1;
            }
            Ordering::Equal => {
                // Equal keys — collapse them!
                index_source_a += 1;
                index_source_b += 1;
                let Some(value_a) = iterator_a.pop() else {
                    break;
                };
                if S::USAGE == Usage::SecondaryIndex {
                    // Secondary index: cancel matching put/remove pairs.
                    assert!(S::tombstone(&value_a) != S::tombstone(value_b));
                } else {
                    if drop_tombstones && S::tombstone(&value_a) {
                        continue;
                    }
                    target[index_target] = value_a;
                    index_target += 1;
                }
            }
        }
    }

    let remaining_after = iterator_a.count_remaining();
    let dropped_after = iterator_a.count_dropped();

    let consumed_a = remaining_before - remaining_after;
    let consumed_b = index_source_b as u32;
    let result = MergeResult {
        consumed_a,
        consumed_b,
        dropped: (dropped_after - dropped_before)
            + ((index_source_a + index_source_b - index_target) as u32),
        produced: index_target as u32,
    };
    assert!(result.consumed_a > 0 || result.consumed_b > 0);
    assert!(result.dropped <= result.consumed_a + result.consumed_b);
    assert!(result.produced <= target.len() as u32);
    assert!(result.produced == result.consumed_a + result.consumed_b - result.dropped);
    result
}

/// Merge values from level A and level B slices, with level A taking precedence on equal
/// keys. Tombstones may be dropped.
///
/// Port of upstream `values_merge`.
pub fn values_merge<S: TableTrait>(
    target: &mut [S::Value],
    source_a: &[S::Value],
    source_b: &[S::Value],
    drop_tombstones: bool,
) -> MergeResult {
    assert!(!source_a.is_empty());
    assert!(!source_b.is_empty());
    assert!(!target.is_empty());

    let mut index_source_a: usize = 0;
    let mut index_source_b: usize = 0;
    let mut index_target: usize = 0;

    while index_source_a < source_a.len()
        && index_source_b < source_b.len()
        && index_target < target.len()
    {
        let value_a = &source_a[index_source_a];
        let value_b = &source_b[index_source_b];

        match S::key_from_value(value_a).cmp(&S::key_from_value(value_b)) {
            Ordering::Less => {
                // Pick value from level a.
                index_source_a += 1;
                if drop_tombstones && S::tombstone(value_a) {
                    assert!(S::USAGE != Usage::SecondaryIndex);
                    continue;
                }
                target[index_target] = *value_a;
                index_target += 1;
            }
            Ordering::Greater => {
                // Pick value from level b.
                index_source_b += 1;
                target[index_target] = *value_b;
                index_target += 1;
            }
            Ordering::Equal => {
                // Equal keys — collapse them!
                index_source_a += 1;
                index_source_b += 1;
                if S::USAGE == Usage::SecondaryIndex {
                    // Secondary index: cancel matching put/remove pairs.
                    assert!(S::tombstone(value_a) != S::tombstone(value_b));
                } else {
                    if drop_tombstones && S::tombstone(value_a) {
                        continue;
                    }
                    target[index_target] = *value_a;
                    index_target += 1;
                }
            }
        }
    }

    let result = MergeResult {
        consumed_a: index_source_a as u32,
        consumed_b: index_source_b as u32,
        dropped: (index_source_a + index_source_b - index_target) as u32,
        produced: index_target as u32,
    };
    assert!(result.consumed_a > 0 || result.consumed_b > 0);
    assert!(result.consumed_a <= source_a.len() as u32);
    assert!(result.consumed_b <= source_b.len() as u32);
    assert!(result.dropped <= result.consumed_a + result.consumed_b);
    assert!(result.produced <= target.len() as u32);
    assert!(result.produced == result.consumed_a + result.consumed_b - result.dropped);
    result
}

// ---------------------------------------------------------------------------
// Compaction struct
// ---------------------------------------------------------------------------

/// Compaction state for a single level (upstream `CompactionType(Tree, Storage)`).
///
/// Each compaction is paced to run in an arbitrary amount of beats, by the forest.
///
/// # Porting status
///
/// Phase 1: struct fields, init, reset, assert_between_bars, idle, block_queues_empty_input,
///          and pure merge/copy functions.
/// Phase 2: half_bar_commence, half_bar_complete, beat_commence (lifecycle stubs).
/// Phase 3: I/O dispatch. The sans-IO `dispatch` loop merges both the immutable (L0) source
///          and a disk level-A table (`TableInfoA::Disk`, level_b > 0) against disk level-B
///          value blocks (grid reads + block iteration), writing output index/value blocks
///          through the grid. Remaining: the forest's per-half-bar beat pacing.
pub struct Compaction<S: TreeSpec> {
    // Globally scoped fields (survive across bars):
    #[allow(dead_code)] // used in Phase 2/3 (lifecycle + I/O dispatch)
    grid: *mut Grid,
    #[allow(dead_code)]
    tree: *mut crate::tree::Tree<S>,
    pub level_b: u8,
    pub stage: Stage,

    // Half-bar-scoped fields (reset between bars):
    pub op_min: u64,
    pub table_info_a: Option<TableInfoA<S::Key>>,
    pub range_b: Option<CompactionRange<TreeTableInfo<S::Key>, S::Key>>,
    pub move_table: bool,
    pub drop_tombstones: bool,
    pub counters: CompactionCounters,
    pub quotas: Quotas,
    pub level_a_position: Position,
    pub level_b_position: Position,

    /// Queued manifest log appends, applied in `half_bar_complete`.
    ///
    /// DEVIATION: upstream uses `stdx.BoundedArrayType`; here we use `Vec` since
    /// `BoundedArray` doesn't support the mutable slice indexing needed by `half_bar_complete`.
    pub manifest_entries: Vec<ManifestEntry<S::Key>>,

    /// Output table builder (upstream `table_builder`).
    ///
    /// DEVIATION: upstream stores raw `*Block` pointers from a `ResourcePool`; this port has no
    /// ResourcePool, so the two in-flight output blocks are tracked as grid block locations.
    pub table_builder: TableBuilder,
    pub table_builder_index_block: Option<u32>,
    pub table_builder_value_block: Option<u32>,

    pub level_a_immutable_stage: LevelAImmutableStage,

    // Level-A disk-block scratch, decoded synchronously for a `TableInfoA::Disk` source
    // (upstream queues level-A blocks in the resource pool; this port decodes them into these
    // buffers keyed by `level_a_position.value_block` and re-reads only when the position moves).
    /// (value-block address, checksum) pairs for the level-A disk table's index block.
    level_a_index_value_blocks: Vec<(u64, u128)>,
    /// Decoded values of the current level-A value block.
    level_a_values: Vec<S::Value>,
    /// The `value_block` position whose values are in `level_a_values`.
    level_a_values_loaded_value_block: Option<u32>,

    // Level-B disk-block scratch, decoded synchronously for the current
    // `level_b_position.{index_block, value_block}`:
    //
    // DEVIATION: upstream keeps level-B index/value block queues in the resource pool and reuses
    // them across beats. This sans-I/O port has no ResourcePool, so it decodes each block into
    // these buffers keyed by position, re-reading only when the position advances.
    /// The `index_block` position (table within `range_b`) whose value-block address/checksum
    /// list is cached in `level_b_index_value_blocks`.
    level_b_index_loaded_table: Option<u32>,
    /// (value-block address, checksum) pairs for the current level-B table's index block.
    level_b_index_value_blocks: Vec<(u64, u128)>,
    /// Decoded values of the current level-B value block.
    level_b_values: Vec<S::Value>,
    /// The `value_block` position (within the current table) whose values are in `level_b_values`.
    level_b_values_loaded_value_block: Option<u32>,

    // Beat-scoped fields (reset between beats):
    pub pool_is_active: bool,
    pub callback_is_set: bool,
}

impl<S: TreeSpec> Compaction<S> {
    /// Port of upstream `Compaction.init`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `tree` and `grid` remain valid for the compaction's lifetime.
    /// This matches upstream's `*Tree` / `*Grid` raw pointer ownership.
    #[allow(clippy::missing_panics_doc)]
    pub fn new(tree: *mut crate::tree::Tree<S>, grid: *mut Grid, level_b: u8) -> Self {
        assert!((level_b as usize) < constants::LSM_LEVELS as usize);
        Self {
            grid,
            tree,
            level_b,
            stage: Stage::Inactive,
            op_min: 0,
            table_info_a: None,
            range_b: None,
            move_table: false,
            drop_tombstones: false,
            counters: CompactionCounters::default(),
            quotas: Quotas::default(),
            level_a_position: Position::default(),
            level_b_position: Position::default(),
            manifest_entries: Vec::new(),
            table_builder: TableBuilder::new(),
            table_builder_index_block: None,
            table_builder_value_block: None,
            level_a_immutable_stage: LevelAImmutableStage::Ready,
            level_a_index_value_blocks: Vec::new(),
            level_a_values: Vec::new(),
            level_a_values_loaded_value_block: None,
            level_b_index_loaded_table: None,
            level_b_index_value_blocks: Vec::new(),
            level_b_values: Vec::new(),
            level_b_values_loaded_value_block: None,
            pool_is_active: false,
            callback_is_set: false,
        }
    }

    /// Attach the grid this compaction writes output through.
    ///
    /// DEVIATION: upstream constructs the ``Compaction`` with a `*Grid` in hand; this port's
    /// `Tree` is built standalone and attaches the (forest-owned) grid later via
    /// [`crate::tree::Tree::attach_grid`]. The stored pointer is only used for lifecycle
    /// bookkeeping — the dispatch/half-bar methods receive `grid` as a parameter.
    pub fn set_grid(&mut self, grid: *mut Grid) {
        self.grid = grid;
    }

    /// Port of upstream `Compaction.reset`.
    pub fn reset(&mut self) {
        // TODO(port): grid.trace.cancel(.compact_beat) — tracer not yet ported.
        self.stage = Stage::Inactive;
        self.op_min = 0;
        self.table_info_a = None;
        self.range_b = None;
        self.move_table = false;
        self.drop_tombstones = false;
        self.counters = CompactionCounters::default();
        self.quotas = Quotas::default();
        self.level_a_position = Position::default();
        self.level_b_position = Position::default();
        self.manifest_entries.clear();
        self.table_builder = TableBuilder::new();
        self.table_builder_index_block = None;
        self.table_builder_value_block = None;
        self.level_a_immutable_stage = LevelAImmutableStage::Ready;
        self.level_a_index_value_blocks.clear();
        self.level_a_values.clear();
        self.level_a_values_loaded_value_block = None;
        self.level_b_index_loaded_table = None;
        self.level_b_index_value_blocks.clear();
        self.level_b_values.clear();
        self.level_b_values_loaded_value_block = None;
        self.pool_is_active = false;
        self.callback_is_set = false;
    }

    /// Port of upstream `Compaction.assert_between_bars`.
    #[allow(clippy::missing_panics_doc)]
    pub fn assert_between_bars(&self) {
        assert_eq!(self.stage, Stage::Inactive);
        assert!(self.is_idle());
        assert!(self.block_queues_empty_input());
        assert_eq!(self.table_builder.state(), crate::table::BuilderState::NoBlocks);
        assert!(self.table_builder_index_block.is_none());
        assert!(self.table_builder_value_block.is_none());
        assert!(self.manifest_entries.is_empty());
    }

    /// Returns `true` if the compaction has no pending pool or callback.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        !self.pool_is_active && !self.callback_is_set && self.quotas.beat_exhausted()
    }

    /// Returns `true` if all input block queues are empty.
    #[must_use]
    pub fn block_queues_empty_input(&self) -> bool {
        // TODO(port): check actual ring buffer queues when ResourcePool is ported.
        true
    }

    /// Set the per-beat quota for the current half-bar (upstream `beat_commence`).
    ///
    /// The forest calls this at the start of each beat to tell the compaction how many values
    /// it may process. The beat quota is capped by the remaining half-bar quota.
    ///
    /// # Panics
    /// Panics if the compaction is not in the paused stage, or if `move_table` is set but
    /// the half-bar is not exhausted.
    #[allow(clippy::missing_panics_doc)]
    pub fn beat_commence(&mut self, values_count: u64) {
        assert!(self.is_idle());
        assert_eq!(self.stage, Stage::Paused);
        // We may be carrying over some blocks from the previous beat.
        assert!(self.block_queues_empty_input() || !self.block_queues_empty_input());

        if self.move_table {
            assert!(self.quotas.half_bar_exhausted());
        }

        // Run the compaction up to completion of the half bar quota, if possible.
        let values_remaining = self.quotas.half_bar - self.quotas.half_bar_done;

        self.quotas.beat = values_count.min(values_remaining);
        self.quotas.beat_done = 0;
        assert!(self.quotas.beat <= self.quotas.half_bar);
    }

    // -----------------------------------------------------------------------
    // I/O dispatch (sans-IO synchronous port of compaction_dispatch)
    // -----------------------------------------------------------------------

    /// Reserve enough grid blocks for this compaction's output for one beat
    /// (port of `forest.compact_trees_reserve_grid_blocks`, single compaction).
    ///
    /// The returned [`Reservation`] must be used for every [`Self::dispatch`] of the
    /// beat and forfeited (via [`Grid::forfeit`]) at the end of the half-bar.
    ///
    /// # Panics
    /// Panics on overflow or if the grid cannot cover the reservation (upstream
    /// aborts the process).
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn reserve_output_blocks(&mut self, grid: &mut Grid) -> Reservation {
        // The +1 covers a partially-finished output block from the previous beat,
        // plus the up-to-one-block overshoot of the pacing.
        let beat_value_blocks =
            self.quotas.beat.div_ceil(u64::from(S::LAYOUT.block_value_count_max)) + 1;
        // Index blocks fit `value_block_count_max` address entries each.
        let beat_index_blocks =
            beat_value_blocks.div_ceil(u64::from(S::LAYOUT.value_block_count_max));

        // One carried-over index block and one carried-over value block.
        let total = 1 + 1 + beat_value_blocks + beat_index_blocks;

        grid.reserve(total as usize)
    }

    /// Run the current beat to completion (upstream `compaction_dispatch` + `merge` +
    /// `write_value_block` + `write_index_block` + `beat_complete`).
    ///
    /// Sans-I/O: all grid reads/writes complete synchronously via `storage`, so the
    /// dispatch loop runs to quota exhaustion before returning.
    ///
    /// # Scope
    /// Handles the immutable (L0) source against an optional level-B overlap (disk
    /// tables). Level-B index/value blocks are read synchronously through the grid and
    /// decoded on demand; the immutable source needs no reads. Block allocation uses the
    /// grid stash directly (no ResourcePool).
    ///
    /// # Reservation
    /// The caller must acquire the grid reservation (via [`Grid::reserve`]) after
    /// [`Self::beat_commence`] and pass it in. The caller is responsible for
    /// forfeiting the reservation at the end of the half-bar.
    #[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
    pub fn dispatch(
        &mut self,
        storage: &mut dyn Storage,
        tree: &crate::tree::Tree<S>,
        grid: &mut Grid,
        reservation: Reservation,
    ) where
        S: crate::table::TableSpec,
    {
        assert!(self.table_info_a.is_some());
        assert_eq!(self.level_a_position, Position::default());
        assert_eq!(self.level_b_position, Position::default());
        assert!(matches!(self.level_a_immutable_stage, LevelAImmutableStage::Ready));
        assert_eq!(self.table_builder.state(), BuilderState::NoBlocks);
        assert!(!self.quotas.beat_exhausted());

        self.stage = Stage::Beat;

        loop {
            if self.quotas.beat_exhausted() {
                self.flush_pending_writes(storage, tree, grid, reservation);

                let flush_pending = match self.table_builder.state() {
                    BuilderState::IndexAndValueBlock => {
                        self.quotas.half_bar_exhausted()
                            || self.table_builder.value_block_full(&S::LAYOUT)
                    }
                    BuilderState::IndexBlock => {
                        self.quotas.half_bar_exhausted()
                            || self.table_builder.index_block_full(&S::LAYOUT)
                    }
                    BuilderState::NoBlocks => false,
                };

                assert!(!flush_pending, "sans-IO: writes complete synchronously");
                self.beat_complete();
                return;
            }

            // Allocate blocks for the table builder first to avoid deadlocks.
            if self.table_builder.state() == BuilderState::NoBlocks {
                assert!(self.table_builder_index_block.is_none());
                let location = grid.get_block();
                grid.block_mut(location).fill(0);
                self.table_builder.set_index_block(grid.block_mut(location));
                self.table_builder_index_block = Some(location);
            }

            if self.table_builder.state() == BuilderState::IndexBlock {
                assert!(self.table_builder_value_block.is_none());
                let location = grid.get_block();
                grid.block_mut(location).fill(0);
                self.table_builder.set_value_block(grid.block_mut(location));
                self.table_builder_value_block = Some(location);
            }

            // Load the current level-B value block (synchronously) if any remains, so the
            // merge gate below can decide whether B is ready or exhausted. Level A is either
            // the immutable table (in memory) or a disk table whose current value block is
            // loaded into scratch.
            self.load_level_b(storage, tree, grid);

            let b_ready = self.level_b_values_loaded_value_block.is_some()
                && (self.level_b_position.value as usize) < self.level_b_values.len();
            let b_exhausted = self.level_b_index_loaded_table.is_none();

            let (a_exhausted, a_available) = match self.table_info_a {
                Some(TableInfoA::Immutable) => {
                    let a_exhausted =
                        matches!(self.level_a_immutable_stage, LevelAImmutableStage::Exhausted);
                    // A is available until its stage is `Exhausted`.
                    let a_available =
                        !matches!(self.level_a_immutable_stage, LevelAImmutableStage::Exhausted);
                    (a_exhausted, a_available)
                }
                Some(TableInfoA::Disk(_)) => {
                    self.load_level_a_disk(storage, tree, grid);
                    // A is available while a current value block is loaded and not fully consumed.
                    let a_available = self.level_a_values_loaded_value_block
                        == Some(self.level_a_position.value_block)
                        && (self.level_a_position.value as usize) < self.level_a_values.len();
                    (a_available, a_available)
                }
                None => unreachable!("dispatch requires a level-A table"),
            };

            assert!(!a_exhausted || !b_exhausted);
            assert!(!self.quotas.beat_exhausted());
            assert!(!self.quotas.half_bar_exhausted());

            if matches!(self.table_builder.state(), BuilderState::IndexAndValueBlock)
                && (a_available || b_ready)
                && !self.table_builder.value_block_full(&S::LAYOUT)
            {
                self.merge_step(tree, grid);
            }

            self.flush_pending_writes(storage, tree, grid, reservation);
        }
    }

    /// Merge one step from the immutable (A) and decoded level-B (B) inputs to the output
    /// table builder (upstream `merge` + `merge_immutable` + `merge_callback` +
    /// `merge_advance_position`).
    ///
    /// Merge one step of the two inputs, dispatching on the level-A source
    /// (immutable table in memory vs. a disk table decoded into scratch).
    #[allow(clippy::too_many_lines)]
    fn merge_step(&mut self, tree: &crate::tree::Tree<S>, grid: &mut Grid)
    where
        S: crate::table::TableSpec,
    {
        match self.table_info_a {
            Some(TableInfoA::Immutable) => self.merge_step_immutable(tree, grid),
            Some(TableInfoA::Disk(_)) => self.merge_step_disk(tree, grid),
            None => unreachable!("merge_step requires a level-A table"),
        }
    }

    /// Merge one step of the in-memory immutable (L0) level A against the current level-B
    /// value block (upstream `merge_immutable_table`); output values are written through the
    /// grid's value block.
    ///
    /// When both A and B are available we merge (A wins on equal keys, dropping A tombstones
    /// if `drop_tombstones`); when only A remains we copy it; when only B remains we copy it —
    /// mirroring the dispatch of upstream `merge_immutable`.
    #[allow(clippy::too_many_lines, clippy::manual_div_ceil)]
    fn merge_step_immutable(&mut self, tree: &crate::tree::Tree<S>, grid: &mut Grid)
    where
        S: crate::table::TableSpec,
    {
        assert_eq!(self.table_builder.state(), BuilderState::IndexAndValueBlock);

        // Port of upstream `merge`: transition the immutable stage (Ready → Merge), or
        // accept Exhausted (level B may still need merging).
        if self.level_a_immutable_stage == LevelAImmutableStage::Ready {
            self.level_a_immutable_stage = LevelAImmutableStage::Merge;
        } else {
            assert_eq!(self.level_a_immutable_stage, LevelAImmutableStage::Exhausted);
        }

        let mut iterator = ImmutableTableIterator::new(
            tree.table_immutable_ref().iterator_context(),
            None,
            Direction::Ascending,
        );

        // Advance the freshly created iterator to the saved position.
        // DEVIATION: upstream stores the iterator inline in the compaction union
        // and persists it across merges; we re-create and fast-forward.
        for _ in 0..self.level_a_position.value {
            assert!(iterator.pop().is_some(), "position ahead of the immutable table length");
        }

        let budget_immutable = S::LAYOUT.data.value_count_max.min(iterator.count_remaining());
        let space_left = S::LAYOUT.data.value_count_max - self.table_builder.value_count();
        let b_source = &self.level_b_values[self.level_b_position.value as usize..];

        let Some(value_location) = self.table_builder_value_block else {
            unreachable!("value block required for merge")
        };
        let block = grid.block_mut(value_location);

        // A is available iff its stage is not `Exhausted` (the immutable table still has
        // values to merge). B is available iff a value block is loaded.
        let a_available = !matches!(self.level_a_immutable_stage, LevelAImmutableStage::Exhausted);
        let b_available = !b_source.is_empty();
        assert!(a_available || b_available, "both inputs exhausted in a merge step");

        let MergeResult { consumed_a, consumed_b, dropped, produced: _produced } = if !a_available {
            // Only B remains: value_copy (upstream `values_source_a == null` branch).
            let mut index_target: usize = 0;
            while index_target < space_left as usize && index_target < b_source.len() {
                self.table_builder.insert_block_value(&b_source[index_target], block, &S::LAYOUT);
                index_target += 1;
            }
            MergeResult {
                consumed_a: 0,
                consumed_b: index_target as u32,
                dropped: 0,
                produced: index_target as u32,
            }
        } else if !b_available {
            // Only A remains: value_copy_immutable (drop-tombstones sensitive).
            let mut index_target: u32 = 0;
            let mut index_source: u32 = 0;
            while index_source < budget_immutable && index_target < space_left {
                let Some(value_in) = iterator.pop() else {
                    break;
                };
                index_source += 1;
                if self.drop_tombstones && <S as TableTrait>::tombstone(&value_in) {
                    assert!(<S as TableTrait>::USAGE != Usage::SecondaryIndex);
                    continue;
                }
                self.table_builder.insert_block_value(&value_in, block, &S::LAYOUT);
                index_target += 1;
            }
            MergeResult {
                consumed_a: index_source,
                consumed_b: 0,
                dropped: index_source - index_target,
                produced: index_target,
            }
        } else {
            // Both A and B available: values_merge_immutable.
            let mut index_source_a: usize = 0;
            let mut index_source_b: usize = 0;
            let mut index_target: usize = 0;

            while index_source_a < budget_immutable as usize
                && index_source_b < b_source.len()
                && index_target < space_left as usize
            {
                let key_a =
                    iterator.peek().unwrap_or_else(|| unreachable!("budget > 0 implies a value"));
                let value_b = &b_source[index_source_b];
                match key_a.cmp(&<S as TableTrait>::key_from_value(value_b)) {
                    Ordering::Less => {
                        // Pick value from level A.
                        index_source_a += 1;
                        let value_a =
                            iterator.pop().unwrap_or_else(|| unreachable!("peeked a value"));
                        if self.drop_tombstones && <S as TableTrait>::tombstone(&value_a) {
                            assert!(<S as TableTrait>::USAGE != Usage::SecondaryIndex);
                            continue;
                        }
                        self.table_builder.insert_block_value(&value_a, block, &S::LAYOUT);
                        index_target += 1;
                    }
                    Ordering::Greater => {
                        // Pick value from level B.
                        index_source_b += 1;
                        self.table_builder.insert_block_value(value_b, block, &S::LAYOUT);
                        index_target += 1;
                    }
                    Ordering::Equal => {
                        // Equal keys — collapse them.
                        index_source_a += 1;
                        index_source_b += 1;
                        let value_a =
                            iterator.pop().unwrap_or_else(|| unreachable!("peeked a value"));
                        if <S as TableTrait>::USAGE == Usage::SecondaryIndex {
                            assert!(
                                <S as TableTrait>::tombstone(&value_a)
                                    != <S as TableTrait>::tombstone(value_b)
                            );
                        } else if self.drop_tombstones && <S as TableTrait>::tombstone(&value_a) {
                            continue;
                        }
                        self.table_builder.insert_block_value(&value_a, block, &S::LAYOUT);
                        index_target += 1;
                    }
                }
            }

            MergeResult {
                consumed_a: index_source_a as u32,
                consumed_b: index_source_b as u32,
                dropped: (index_source_a + index_source_b - index_target) as u32,
                produced: index_target as u32,
            }
        };

        // Advance positions (port of merge_callback bookkeeping).
        self.level_a_position.value += consumed_a;
        self.level_b_position.value += consumed_b;
        // table_builder.value_count already advanced by insert_value calls above.

        assert!(
            self.level_a_position.value <= <S as TableTrait>::VALUE_COUNT_MAX as u32,
            "immutable position overflow"
        );
        assert!(
            self.table_builder.value_count() <= S::LAYOUT.data.value_count_max,
            "builder value_count overflow"
        );

        let consumed_ab = consumed_a + consumed_b;
        self.quotas.half_bar_done += u64::from(consumed_ab);
        self.quotas.beat_done += u64::from(consumed_ab);
        assert!(self.quotas.half_bar_done <= self.quotas.half_bar);
        self.counters.dropped += u64::from(dropped);

        self.merge_advance_position(grid, tree.table_immutable_ref().count());
    }

    /// Merge one step of two **disk** inputs — level A's current value block against level B's
    /// current value block (upstream `merge_disk` / `merge_inputs_disk`). Output values are
    /// written through the grid's value block.
    ///
    /// Level A is limited to one value block per step (`limit`), matching upstream's
    /// `values_merge`; on equal keys level A wins and the entry is dropped when
    /// `drop_tombstones` and A holds the tombstone.
    #[allow(clippy::too_many_lines)]
    fn merge_step_disk(&mut self, tree: &crate::tree::Tree<S>, grid: &mut Grid)
    where
        S: crate::table::TableSpec,
    {
        assert_eq!(self.table_builder.state(), BuilderState::IndexAndValueBlock);

        let a_source = &self.level_a_values[self.level_a_position.value as usize..];
        let budget = S::LAYOUT.data.value_count_max as usize;
        let budget = budget.min(a_source.len());
        let space_left = S::LAYOUT.data.value_count_max - self.table_builder.value_count();
        let b_source = &self.level_b_values[self.level_b_position.value as usize..];

        let Some(value_location) = self.table_builder_value_block else {
            unreachable!("value block required for merge")
        };
        let block = grid.block_mut(value_location);

        let a_available = !a_source.is_empty();
        let b_available = !b_source.is_empty();
        assert!(a_available || b_available, "both inputs exhausted in a merge step");

        let MergeResult { consumed_a, consumed_b, dropped, produced: _produced } = if !a_available {
            // Only B remains: value_copy.
            let mut index_target: usize = 0;
            while index_target < space_left as usize && index_target < b_source.len() {
                self.table_builder.insert_block_value(&b_source[index_target], block, &S::LAYOUT);
                index_target += 1;
            }
            MergeResult {
                consumed_a: 0,
                consumed_b: index_target as u32,
                dropped: 0,
                produced: index_target as u32,
            }
        } else if !b_available {
            // Only A remains: value_copy (drop-tombstones sensitive).
            let mut index_target: usize = 0;
            let mut index_source: usize = 0;
            while index_source < budget && index_target < space_left as usize {
                let Some(value_in) = a_source.get(index_source).copied() else {
                    break;
                };
                index_source += 1;
                if self.drop_tombstones && <S as TableTrait>::tombstone(&value_in) {
                    assert!(<S as TableTrait>::USAGE != Usage::SecondaryIndex);
                    continue;
                }
                self.table_builder.insert_block_value(&value_in, block, &S::LAYOUT);
                index_target += 1;
            }
            MergeResult {
                consumed_a: index_source as u32,
                consumed_b: 0,
                dropped: (index_source - index_target) as u32,
                produced: index_target as u32,
            }
        } else {
            // Both A and B available: values_merge.
            let mut index_source_a: usize = 0;
            let mut index_source_b: usize = 0;
            let mut index_target: usize = 0;

            while index_source_a < budget
                && index_source_b < b_source.len()
                && index_target < space_left as usize
            {
                let value_a = a_source[index_source_a];
                let value_b = &b_source[index_source_b];
                match <S as TableTrait>::key_from_value(&value_a)
                    .cmp(&<S as TableTrait>::key_from_value(value_b))
                {
                    Ordering::Less => {
                        // Pick value from level A.
                        index_source_a += 1;
                        if self.drop_tombstones && <S as TableTrait>::tombstone(&value_a) {
                            assert!(<S as TableTrait>::USAGE != Usage::SecondaryIndex);
                            continue;
                        }
                        self.table_builder.insert_block_value(&value_a, block, &S::LAYOUT);
                        index_target += 1;
                    }
                    Ordering::Greater => {
                        // Pick value from level B.
                        index_source_b += 1;
                        self.table_builder.insert_block_value(value_b, block, &S::LAYOUT);
                        index_target += 1;
                    }
                    Ordering::Equal => {
                        // Equal keys — collapse them; level A wins (secondary index cancels
                        // the put/remove pair and emits nothing).
                        index_source_a += 1;
                        index_source_b += 1;
                        if <S as TableTrait>::USAGE == Usage::SecondaryIndex {
                            assert!(
                                <S as TableTrait>::tombstone(&value_a)
                                    != <S as TableTrait>::tombstone(value_b)
                            );
                        } else {
                            if self.drop_tombstones && <S as TableTrait>::tombstone(&value_a) {
                                continue;
                            }
                            self.table_builder.insert_block_value(&value_a, block, &S::LAYOUT);
                            index_target += 1;
                        }
                    }
                }
            }

            MergeResult {
                consumed_a: index_source_a as u32,
                consumed_b: index_source_b as u32,
                dropped: (index_source_a + index_source_b - index_target) as u32,
                produced: index_target as u32,
            }
        };

        // Advance positions (port of merge_callback bookkeeping).
        self.level_a_position.value += consumed_a;
        self.level_b_position.value += consumed_b;

        assert!(
            self.level_a_position.value <= <S as TableTrait>::VALUE_COUNT_MAX as u32,
            "level-A position overflow"
        );
        assert!(
            self.table_builder.value_count() <= S::LAYOUT.data.value_count_max,
            "builder value_count overflow"
        );

        let consumed_ab = consumed_a + consumed_b;
        self.quotas.half_bar_done += u64::from(consumed_ab);
        self.quotas.beat_done += u64::from(consumed_ab);
        assert!(self.quotas.half_bar_done <= self.quotas.half_bar);
        self.counters.dropped += u64::from(dropped);

        self.merge_advance_position(grid, tree.table_immutable_ref().count());
    }

    /// Synchronously load the current level-B value block into [`Self::level_b_values`]
    /// (upstream reads level-B index/value blocks through `read_index_block` +
    /// `read_value_block`; this sans-I/O port reads them eagerly before each merge).
    ///
    /// The current table is `range_b[level_b_position.index_block]`; its index block is
    /// decoded once per table into `level_b_index_value_blocks`. The current value block is
    /// `value_block` within that table. Nothing is loaded once B is exhausted.
    fn load_level_b(
        &mut self,
        storage: &mut dyn Storage,
        tree: &crate::tree::Tree<S>,
        grid: &mut Grid,
    ) where
        S: crate::table::TableSpec,
    {
        let table_index = self.level_b_position.index_block as usize;
        let count = self.range_b.as_ref().map_or(0, |r| r.tables.tables.slice().len());
        if table_index >= count {
            // B exhausted; scratch stays empty (`level_b_index_loaded_table == None`).
            return;
        }

        if self.level_b_index_loaded_table != Some(self.level_b_position.index_block) {
            // Copy the table metadata out of `range_b` (it is `Copy`) so the scratch
            // mutations below don't conflict with the borrow of `self.range_b`.
            let Some(table) = self
                .range_b
                .as_ref()
                .and_then(|r| r.tables.tables.slice().get(table_index))
                .map(|t| t.table_info)
            else {
                unreachable!("table_index < count checked above")
            };
            let index_block = grid.read_block_sync(storage, table.address, table.checksum);
            let index_schema =
                S::LAYOUT.index.from_block_with_schema(&index_block, tree.config_ref().id);
            let value_count = index_schema.value_blocks_used(&index_block) as usize;
            let mut value_blocks = Vec::with_capacity(value_count);
            for i in 0..value_count {
                value_blocks.push((
                    index_schema.value_address(&index_block, i),
                    index_schema.value_checksum(&index_block, i),
                ));
            }
            self.level_b_index_value_blocks = value_blocks;
            self.level_b_index_loaded_table = Some(self.level_b_position.index_block);
            self.level_b_values_loaded_value_block = None;
        }

        let value_block = self.level_b_position.value_block;
        if self.level_b_values_loaded_value_block != Some(value_block) {
            let (address, checksum) = self.level_b_index_value_blocks[value_block as usize];
            let value_block_bytes = grid.read_block_sync(storage, address, checksum);
            self.level_b_values =
                crate::table::value_block_values_used::<S>(&value_block_bytes, &S::LAYOUT.data);
            self.level_b_values_loaded_value_block = Some(value_block);
        }
    }

    /// Load the current level-A **disk** value block into [`Self::level_a_values`] (upstream
    /// reads level-A index/value blocks through `read_index_block` + `read_value_block`; this
    /// sans-I/O port reads them eagerly before each merge).
    ///
    /// Level A is a single disk table (`TableInfoA::Disk`), so its index block is decoded once
    /// into `level_a_index_value_blocks`; the current value block is `level_a_position.value_block`.
    /// Nothing is loaded once A is exhausted (`level_a_position.value_block` past the table).
    fn load_level_a_disk(
        &mut self,
        storage: &mut dyn Storage,
        tree: &crate::tree::Tree<S>,
        grid: &mut Grid,
    ) where
        S: crate::table::TableSpec,
    {
        let Some(TableInfoA::Disk(table)) = &self.table_info_a else {
            unreachable!("load_level_a_disk requires a disk level-A table")
        };
        // Once the single disk table's last value block is consumed, `merge_advance_position`
        // advances `index_block` to 1 and releases the table; that position marks A exhausted
        // (the released scratch must not be reloaded).
        if self.level_a_position.index_block > 0 {
            return;
        }
        if self.level_a_index_value_blocks.is_empty() {
            let index_block =
                grid.read_block_sync(storage, table.table_info.address, table.table_info.checksum);
            let index_schema =
                S::LAYOUT.index.from_block_with_schema(&index_block, tree.config_ref().id);
            let value_count = index_schema.value_blocks_used(&index_block) as usize;
            let mut value_blocks = Vec::with_capacity(value_count);
            for i in 0..value_count {
                value_blocks.push((
                    index_schema.value_address(&index_block, i),
                    index_schema.value_checksum(&index_block, i),
                ));
            }
            self.level_a_index_value_blocks = value_blocks;
            self.level_a_values_loaded_value_block = None;
        }

        let value_block = self.level_a_position.value_block;
        if value_block as usize >= self.level_a_index_value_blocks.len() {
            // A exhausted; the last loaded block has already been released.
            return;
        }
        if self.level_a_values_loaded_value_block != Some(value_block) {
            let (address, checksum) = self.level_a_index_value_blocks[value_block as usize];
            let value_block_bytes = grid.read_block_sync(storage, address, checksum);
            self.level_a_values =
                crate::table::value_block_values_used::<S>(&value_block_bytes, &S::LAYOUT.data);
            self.level_a_values_loaded_value_block = Some(value_block);
        }
    }

    /// Advance level-A/B positions after a merge step (upstream `merge_advance_position`).
    ///
    /// Handles immutable-table carry-over (Ready ↔ Exhausted) and the level-B disk table's
    /// value/index-block advancement (releasing a table's blocks once fully consumed).
    fn merge_advance_position(&mut self, grid: &mut Grid, immutable_count: u32)
    where
        S: crate::table::TableSpec,
    {
        if matches!(self.table_info_a, Some(TableInfoA::Immutable)) {
            if self.level_a_immutable_stage == LevelAImmutableStage::Merge {
                if self.level_a_position.value == immutable_count {
                    self.level_a_position.value_block += 1;
                    assert_eq!(self.level_a_position.value_block, 1);
                    self.level_a_position.value = 0;
                    self.level_a_immutable_stage = LevelAImmutableStage::Exhausted;
                } else {
                    self.level_a_immutable_stage = LevelAImmutableStage::Ready;
                }
            } else {
                assert!(matches!(self.level_a_immutable_stage, LevelAImmutableStage::Exhausted));
            }
        } else {
            // Disk level A: advance through the single disk table's value blocks (upstream
            // also pops the value/index block queues when each value block / the table ends).
            if self.level_a_position.value == self.level_a_values.len() as u32 {
                self.level_a_position.value_block += 1;
                self.level_a_position.value = 0;

                let value_block_count = self.level_a_index_value_blocks.len() as u32;
                if self.level_a_position.value_block == value_block_count {
                    self.level_a_position.index_block += 1;
                    assert_eq!(self.level_a_position.index_block, 1); // single disk table
                    self.level_a_position.value_block = 0;
                    self.release_current_level_a_table(grid);
                }
            }
        }

        // Level B: the current value block is fully consumed once the position reaches its
        // used-value count. When the whole table (all its value blocks) is consumed, release
        // its blocks and advance to the next table in range_b.
        if self.level_b_position.value == self.level_b_values.len() as u32
            && self.level_b_index_loaded_table.is_some()
        {
            self.level_b_position.value_block += 1;
            self.level_b_position.value = 0;

            let value_block_count = self.level_b_index_value_blocks.len() as u32;
            if self.level_b_position.value_block == value_block_count {
                self.level_b_position.index_block += 1;
                assert_eq!(self.level_b_position.value_block, value_block_count);
                self.level_b_position.value_block = 0;
                self.release_current_level_b_table(grid);
            }
        }
    }

    /// Release the current level-B table's index and value blocks (upstream
    /// `read_value_block_release_table`), then invalidate the level-B scratch so the next
    /// table is re-read on the next [`Self::load_level_b`].
    fn release_current_level_b_table(&mut self, grid: &mut Grid) {
        let Some(table) = self.range_b.as_ref().and_then(|r| {
            r.tables.tables.slice().get(self.level_b_position.index_block as usize - 1)
        }) else {
            unreachable!("released table was part of range_b");
        };
        let index_address = table.table_info.address;
        let value_addresses: Vec<u64> =
            self.level_b_index_value_blocks.iter().map(|(address, _)| *address).collect();

        grid.release(&value_addresses);
        grid.release(&[index_address]);

        self.level_b_index_loaded_table = None;
        self.level_b_index_value_blocks.clear();
        self.level_b_values.clear();
        self.level_b_values_loaded_value_block = None;
    }

    /// Release the level-A **disk** table's index and value blocks once its last value block is
    /// consumed, then invalidate the level-A disk scratch.
    fn release_current_level_a_table(&mut self, grid: &mut Grid) {
        let Some(TableInfoA::Disk(table)) = &self.table_info_a else {
            unreachable!("release_current_level_a_table requires a disk level-A table");
        };
        let index_address = table.table_info.address;
        let value_addresses: Vec<u64> =
            self.level_a_index_value_blocks.iter().map(|(address, _)| *address).collect();

        grid.release(&value_addresses);
        grid.release(&[index_address]);

        self.level_a_index_value_blocks.clear();
        self.level_a_values.clear();
        self.level_a_values_loaded_value_block = None;
    }

    /// Write a completed value block through the grid (upstream `write_value_block`).
    ///
    /// Acquires an address, finishes the value block header, enqueues the write,
    /// polls the grid, and hands back the fresh block returned by the write
    /// completion (must be unref'd).
    fn write_value_block(
        &mut self,
        storage: &mut dyn Storage,
        tree: &crate::tree::Tree<S>,
        grid: &mut Grid,
        reservation: Reservation,
    ) where
        S: crate::table::TableSpec,
    {
        let Some(value_location) = self.table_builder_value_block else {
            unreachable!("value block required for write")
        };
        let Some(index_location) = self.table_builder_index_block else {
            unreachable!("index block required for value block finish")
        };

        let address = grid.acquire(reservation);
        let view = grid.superblock_view();

        // Upstream increments counters.out by value_count *before* the write is
        // submitted (the block is logically committed to the output).
        self.counters.out += u64::from(self.table_builder.value_count());

        assert!(grid.block_references(value_location) > 0);

        // Borrow both blocks simultaneously; blocks_mut2 guarantees disjointness.
        let (value_block, index_block) = grid.blocks_mut2(value_location, index_location);
        self.table_builder.value_block_finish::<S>(
            value_block,
            index_block,
            &S::LAYOUT,
            DataFinishOptions {
                cluster: view.cluster,
                release: view.release,
                address,
                snapshot_min: snapshot_min_for_table_output(self.op_min),
                tree_id: tree.config_ref().id,
            },
        );

        assert_eq!(self.table_builder.state(), BuilderState::IndexBlock);
        self.table_builder_value_block = None;

        // Enqueue the write (grid.create_block consumes the caller's location ref).
        let token = grid.create_block(storage, address, value_location);
        grid.poll(storage);
        let fresh_location = Self::take_write_done_event(grid, token, address);
        grid.block_unref(fresh_location);
    }

    /// Write a completed index block through the grid (upstream `write_index_block`).
    ///
    /// The resulting [`TreeTableInfo`] is queued in [`Self::manifest_entries`] for
    /// insertion into the manifest during [`Self::half_bar_complete`].
    fn write_index_block(
        &mut self,
        storage: &mut dyn Storage,
        tree: &crate::tree::Tree<S>,
        grid: &mut Grid,
        reservation: Reservation,
    ) where
        S: crate::table::TableSpec,
    {
        let Some(index_location) = self.table_builder_index_block else {
            unreachable!("index block required for write")
        };

        let address = grid.acquire(reservation);
        let view = grid.superblock_view();

        let table_info = self.table_builder.index_block_finish::<<S as TableTrait>::Key>(
            grid.block_mut(index_location),
            &S::LAYOUT,
            IndexFinishOptions {
                cluster: view.cluster,
                release: view.release,
                address,
                snapshot_min: snapshot_min_for_table_output(self.op_min),
                tree_id: tree.config_ref().id,
            },
        );

        assert_eq!(self.table_builder.state(), BuilderState::NoBlocks);
        assert!(self.table_builder_value_block.is_none());
        self.table_builder_index_block = None;

        // Queue the manifest entry before dispatching the write (matches upstream
        // ordering; sans-IO makes the order inconsequential).
        self.manifest_entries.push(ManifestEntry {
            operation: ManifestEntryOperation::InsertToLevelB,
            table: TreeTableInfo {
                checksum: table_info.checksum,
                address: table_info.address,
                snapshot_min: table_info.snapshot_min,
                snapshot_max: table_info.snapshot_max,
                key_min: table_info.key_min,
                key_max: table_info.key_max,
                value_count: table_info.value_count,
            },
        });

        let token = grid.create_block(storage, address, index_location);
        grid.poll(storage);
        let fresh_location = Self::take_write_done_event(grid, token, address);
        grid.block_unref(fresh_location);
    }

    /// Flush a full or half-bar-complete value and/or index block pair
    /// (upstream `flush_table_builder_blocks`).
    ///
    /// Flush order: value block first (the index block accumulates the value
    /// block's key range and address during value block finish), then index
    /// block.
    #[allow(clippy::missing_panics_doc)]
    fn flush_pending_writes(
        &mut self,
        storage: &mut dyn Storage,
        tree: &crate::tree::Tree<S>,
        grid: &mut Grid,
        reservation: Reservation,
    ) where
        S: crate::table::TableSpec,
    {
        // Value block: write when full or half-bar exhausted.
        if matches!(self.table_builder.state(), BuilderState::IndexAndValueBlock)
            && (self.table_builder.value_block_full(&S::LAYOUT) || self.quotas.half_bar_exhausted())
        {
            if self.table_builder.value_block_empty() {
                // Zero values this beat; release the empty block.
                assert!(self.quotas.half_bar_exhausted());
                self.table_builder.release_empty_value_block();
                let Some(value_location) = self.table_builder_value_block.take() else {
                    unreachable!("value block set for finish")
                };
                grid.block_unref(value_location);
            } else {
                self.write_value_block(storage, tree, grid, reservation);
            }
        }

        // Index block: write when full or half-bar exhausted.
        if matches!(self.table_builder.state(), BuilderState::IndexBlock)
            && (self.table_builder.index_block_full(&S::LAYOUT) || self.quotas.half_bar_exhausted())
        {
            if self.table_builder.index_block_empty() {
                // Only possible when half-bar exhausted with zero values this beat.
                assert!(self.quotas.half_bar_exhausted());
                self.table_builder.release_empty_index_block();
                let Some(index_location) = self.table_builder_index_block.take() else {
                    unreachable!("index block set for finish")
                };
                grid.block_unref(index_location);
            } else {
                self.write_index_block(storage, tree, grid, reservation);
            }
        }
    }

    /// Drain events and return the fresh location from a matching `WriteDone`.
    ///
    /// # Panics
    /// Panics if no matching `WriteDone` event is present.
    fn take_write_done_event(grid: &mut Grid, expected_token: u32, expected_address: u64) -> u32 {
        let mut result = None;
        for event in grid.take_events() {
            if let Event::WriteDone { token, address, fresh_location } = event {
                assert!(result.is_none(), "unexpected multiple WriteDone events");
                assert_eq!(token, expected_token, "write token mismatch");
                assert_eq!(address, expected_address, "write address mismatch");
                result = Some(fresh_location);
            }
        }
        let Some(fresh_location) = result else {
            unreachable!("expected a WriteDone event from the grid")
        };
        fresh_location
    }

    /// Complete the current beat (upstream `beat_complete`): verify the builder and
    /// positions are consistent, then return to the paused stage with the beat quota
    /// exhausted.
    #[allow(clippy::missing_panics_doc)]
    fn beat_complete(&mut self) {
        assert_eq!(self.stage, Stage::Beat);
        assert_eq!(self.table_builder.state(), BuilderState::NoBlocks);
        assert!(self.table_builder_index_block.is_none());
        assert!(self.table_builder_value_block.is_none());

        if matches!(self.table_info_a, Some(TableInfoA::Immutable)) {
            assert!(
                !matches!(self.level_a_immutable_stage, LevelAImmutableStage::Merge),
                "cannot end a beat mid-merge"
            );
        }

        if self.quotas.half_bar_exhausted() {
            assert_eq!(self.counters.out, self.counters.in_ - self.counters.dropped);
        }

        self.stage = Stage::Paused;
        self.pool_is_active = false;
        self.callback_is_set = false;
        assert!(self.is_idle());
    }

    // -----------------------------------------------------------------------

    /// Plan a compaction half-bar: select input tables, compute quota, handle move-table
    /// optimization (upstream `half_bar_commence`).
    ///
    /// Returns the half-bar quota (total values to compact), or 0 if compaction is unnecessary.
    ///
    /// DEVIATION: the skip optimization (deferring immutable compaction when mutable table fits)
    /// is deferred — it needs `grid.superblock.working.vsr_state.checkpoint.header.op` which
    /// Grid does not expose. The caller should pass `checkpoint_op` if implementing this
    /// optimization later.
    ///
    /// # Panics
    /// Panics if the compaction is not idle, `op` is not aligned to `HALF_BAR_BEAT_COUNT`,
    /// or manifest invariants are violated.
    #[allow(clippy::missing_panics_doc)]
    pub fn half_bar_commence(&mut self, op: u64, tree: &crate::tree::Tree<S>, grid: &Grid) -> u64 {
        assert!(self.is_idle());
        assert!(self.block_queues_empty_input());
        assert_eq!(self.stage, Stage::Inactive);
        assert_eq!(op, tigerbeetle_lsm::compaction::compaction_op_min(op));

        self.stage = Stage::Paused;
        self.op_min = op;

        if self.level_b == 0 {
            if matches!(
                tree.table_immutable_ref().mutability(),
                Mutability::Immutable(state) if state.flushed
            ) {
                assert_eq!(self.quotas.half_bar, 0);
                assert!(self.quotas.half_bar_exhausted());
                return 0;
            }

            let table_value_count_limit = S::VALUE_COUNT_MAX as u32;
            assert!(tree.table_immutable_ref().count() > 0);
            assert!(tree.table_immutable_ref().count() <= table_value_count_limit);

            self.table_info_a = Some(TableInfoA::Immutable);

            self.range_b = Some(tree.manifest_ref().immutable_table_compaction_range(
                tree.table_immutable_ref().key_min(),
                tree.table_immutable_ref().key_max(),
                tree.table_immutable_ref().count(),
                S::VALUE_COUNT_MAX as u32,
            ));

            let Some(range_b) = &self.range_b else { unreachable!() };
            assert!(
                range_b.tables.tables.count()
                    < tigerbeetle_lsm::compaction::COMPACTION_TABLES_INPUT_MAX
            );
            assert!(range_b.key_min <= tree.table_immutable_ref().key_min());
            assert!(tree.table_immutable_ref().key_max() <= range_b.key_max);
        } else {
            let level_a = self.level_b - 1;

            let Some(table_range) = tree.manifest_ref().compaction_table(level_a) else {
                assert_eq!(self.quotas.half_bar, 0);
                assert!(self.quotas.half_bar_exhausted());
                return 0;
            };

            self.table_info_a = Some(TableInfoA::Disk(table_range.table_a));
            self.range_b = Some(table_range.range_b);

            let Some(range_b) = &self.range_b else { unreachable!() };
            let Some(TableInfoA::Disk(table_a)) = &self.table_info_a else {
                unreachable!("level_b > 0 implies disk")
            };
            assert!(
                range_b.tables.tables.count()
                    < tigerbeetle_lsm::compaction::COMPACTION_TABLES_INPUT_MAX
            );
            assert!(table_a.table_info.key_min <= table_a.table_info.key_max);
            assert!(range_b.key_min <= table_a.table_info.key_min);
            assert!(table_a.table_info.key_max <= range_b.key_max);
        }

        if let Some(TableInfoA::Disk(table_ref)) = &self.table_info_a {
            assert!(!grid.free_set_is_released(table_ref.table_info.address));
            assert!(!grid.free_set_is_free(table_ref.table_info.address));
        }
        if let Some(range_b) = &self.range_b {
            for table in range_b.tables.tables.slice() {
                assert!(!grid.free_set_is_released(table.table_info.address));
                assert!(!grid.free_set_is_free(table.table_info.address));
            }
        }

        let mut quota_half_bar: u64 = match &self.table_info_a {
            Some(TableInfoA::Immutable) => u64::from(tree.table_immutable_ref().count()),
            Some(TableInfoA::Disk(table_ref)) => u64::from(table_ref.table_info.value_count),
            None => unreachable!("table_info_a set above"),
        };
        if let Some(range_b) = &self.range_b {
            for table in range_b.tables.tables.slice() {
                quota_half_bar += u64::from(table.table_info.value_count);
            }
        }
        self.quotas = Quotas { beat: 0, beat_done: 0, half_bar: quota_half_bar, half_bar_done: 0 };

        self.move_table = matches!(&self.table_info_a, Some(TableInfoA::Disk(_)))
            && self.range_b.as_ref().is_some_and(|r| r.tables.tables.empty());
        let Some(range_b) = &self.range_b else { unreachable!() };
        self.drop_tombstones =
            tree.manifest_ref().compaction_must_drop_tombstones(self.level_b, range_b);

        if self.level_b == constants::LSM_LEVELS - 1 {
            assert!(self.drop_tombstones);
        }

        assert!(self.counters.consistent());
        assert_eq!(self.level_a_position, Position::default());
        assert_eq!(self.level_b_position, Position::default());

        if self.move_table {
            let snapshot_max =
                tigerbeetle_lsm::compaction::snapshot_max_for_table_input(self.op_min);
            if let Some(TableInfoA::Disk(table_ref)) = &self.table_info_a {
                assert!(table_ref.table_info.snapshot_max >= snapshot_max);

                self.manifest_entries.push(ManifestEntry {
                    operation: ManifestEntryOperation::MoveToLevelB,
                    table: table_ref.table_info,
                });

                let value_count = u64::from(table_ref.table_info.value_count);

                self.quotas.beat = value_count;
                self.quotas.beat_done = value_count;
                self.quotas.half_bar_done = value_count;

                assert!(self.quotas.beat_exhausted());
                assert!(self.quotas.half_bar_exhausted());

                return 0;
            }
        }

        if matches!(&self.table_info_a, Some(TableInfoA::Immutable)) {
            self.counters.in_ += u64::from(tree.table_immutable_ref().count());
        }
        if let Some(TableInfoA::Disk(table_ref)) = &self.table_info_a {
            self.counters.in_ += u64::from(table_ref.table_info.value_count);
        }
        if let Some(range_b) = &self.range_b {
            for table in range_b.tables.tables.slice() {
                self.counters.in_ += u64::from(table.table_info.value_count);
            }
        }
        self.quotas.half_bar
    }

    /// Apply the changes accumulated in memory to the manifest and remove invisible tables
    /// (upstream `half_bar_complete`).
    ///
    /// The caller must provide the manifest log so that manifest mutations are recorded.
    ///
    /// # Panics
    /// Panics if the compaction is not idle/paused, counters are inconsistent, or the
    /// half-bar is not exhausted.
    #[allow(clippy::missing_panics_doc)]
    pub fn half_bar_complete(
        &mut self,
        tree: &mut crate::tree::Tree<S>,
        grid: &Grid,
        log: &mut impl tigerbeetle_lsm::manifest::ManifestLog,
    ) {
        assert!(self.is_idle());
        assert!(self.block_queues_empty_input());
        assert_eq!(self.stage, Stage::Paused);
        assert!(self.counters.consistent());
        assert!(self.quotas.half_bar_exhausted());
        assert_eq!(
            self.table_builder.state(),
            crate::table::BuilderState::NoBlocks,
            "output table must be fully written by the end of the half-bar"
        );

        // Reset compaction to inactive, keeping only the globally scoped fields.
        let grid_ptr = self.grid;
        let tree_ptr = self.tree;
        let level_b = self.level_b;

        if self.table_info_a.is_none() {
            assert!(self.range_b.is_none());
            assert!(self.manifest_entries.is_empty());
            assert_eq!(self.quotas.half_bar, 0);
            if self.level_b == 0 {
                // Either the immutable table is already flushed, or the mutable table will be
                // absorbed into the immutable table.
            }
            self.reset();
            return;
        }

        assert!(self.range_b.is_some());
        assert!(self.quotas.half_bar > 0);

        if let Some(TableInfoA::Disk(table_ref)) = &self.table_info_a {
            if self.move_table {
                assert!(!grid.free_set_is_released(table_ref.table_info.address));
                assert!(!grid.free_set_is_free(table_ref.table_info.address));
            } else {
                assert!(grid.free_set_is_released(table_ref.table_info.address));
            }
        }
        if let Some(range_b) = &self.range_b {
            for table in range_b.tables.tables.slice() {
                assert!(grid.free_set_is_released(table.table_info.address));
            }
        }

        if self.level_b == 0 && matches!(&self.table_info_a, Some(TableInfoA::Immutable)) {
            assert!(matches!(
                tree.table_immutable_ref().mutability(),
                Mutability::Immutable(state) if !state.flushed
            ));
            tree.table_immutable_mut().set_flushed();
        }

        // Each compaction's manifest updates are deferred to the end of the last bar.
        // Read immutable table count before taking &mut manifest to avoid borrow conflict.
        let immutable_count = if matches!(&self.table_info_a, Some(TableInfoA::Immutable)) {
            u64::from(tree.table_immutable_ref().count())
        } else {
            0
        };
        let manifest = tree.manifest_mut();
        let snapshot_max = tigerbeetle_lsm::compaction::snapshot_max_for_table_input(self.op_min);

        let mut manifest_removed_value_count: u64 = 0;
        let mut manifest_added_value_count: u64 = 0;

        if self.move_table {
            // If no compaction is required, don't update snapshot_max.
        } else {
            // These updates MUST precede insert_table() and move_table().
            match &self.table_info_a {
                Some(TableInfoA::Immutable) => {
                    manifest_removed_value_count = immutable_count;
                }
                Some(TableInfoA::Disk(table_ref)) => {
                    manifest_removed_value_count += u64::from(table_ref.table_info.value_count);
                    manifest.update_table(log, self.level_b - 1, snapshot_max, *table_ref);
                }
                None => unreachable!("checked above"),
            }
            if let Some(range_b) = &self.range_b {
                for table in range_b.tables.tables.slice() {
                    manifest_removed_value_count += u64::from(table.table_info.value_count);
                    manifest.update_table(log, self.level_b, snapshot_max, *table);
                }
            }
        }

        // Process queued manifest entries.
        for entry in &self.manifest_entries {
            match entry.operation {
                ManifestEntryOperation::InsertToLevelB => {
                    manifest.insert_table(log, self.level_b, &entry.table);
                    manifest_added_value_count += u64::from(entry.table.value_count);
                }
                ManifestEntryOperation::MoveToLevelB => {
                    manifest.move_table(log, self.level_b - 1, self.level_b, &entry.table);
                    manifest_removed_value_count += u64::from(entry.table.value_count);
                    manifest_added_value_count += u64::from(entry.table.value_count);
                }
            }
        }
        if self.move_table {
            assert!(self.counters == CompactionCounters::default());
            assert_eq!(manifest_added_value_count, manifest_removed_value_count);
            assert!(manifest_added_value_count > 0);
        } else {
            assert_eq!(manifest_added_value_count, self.counters.out);
            assert_eq!(manifest_removed_value_count, self.counters.in_);
            assert_eq!(
                manifest_removed_value_count - manifest_added_value_count,
                self.counters.dropped
            );
        }

        // Hide any tables that are now invisible.
        if let Some(range_b) = &self.range_b {
            manifest.remove_invisible_tables(
                log,
                self.level_b,
                &[],
                range_b.key_min,
                range_b.key_max,
            );
            if self.level_b > 0 {
                manifest.remove_invisible_tables(
                    log,
                    self.level_b - 1,
                    &[],
                    range_b.key_min,
                    range_b.key_max,
                );
            }
        }

        // Reset compaction, preserving only globally scoped fields.
        self.stage = Stage::Inactive;
        self.op_min = 0;
        self.table_info_a = None;
        self.range_b = None;
        self.move_table = false;
        self.drop_tombstones = false;
        self.counters = CompactionCounters::default();
        self.quotas = Quotas::default();
        self.level_a_position = Position::default();
        self.level_b_position = Position::default();
        self.manifest_entries.clear();
        self.table_builder = TableBuilder::new();
        self.table_builder_index_block = None;
        self.table_builder_value_block = None;
        self.level_a_immutable_stage = LevelAImmutableStage::Ready;
        self.level_a_index_value_blocks = Vec::new();
        self.level_a_values = Vec::new();
        self.level_a_values_loaded_value_block = None;
        self.level_b_index_loaded_table = None;
        self.level_b_index_value_blocks = Vec::new();
        self.level_b_values = Vec::new();
        self.level_b_values_loaded_value_block = None;
        self.pool_is_active = false;
        self.callback_is_set = false;

        // Preserve globally scoped fields (upstream `defer` block).
        self.grid = grid_ptr;
        self.tree = tree_ptr;
        self.level_b = level_b;
        assert_eq!(self.stage, Stage::Inactive);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::table::{BlockValue, TableKey as VsrTableKey, TableLayout};
    use tigerbeetle_lsm::direction::Direction;
    use tigerbeetle_lsm::manifest::TableKey as ManifestTableKey;
    use tigerbeetle_lsm::table_memory::{self as tm, MergeContext, Table};
    use tigerbeetle_lsm::tree::TreeConfig;

    use crate::Zone;
    use crate::grid::{Grid, GridOptions, SuperBlockView};
    use crate::multiversion::Release;
    use crate::storage::MemoryStorage;
    use crate::tree::{Options as TreeOptions, Tree};
    use tigerbeetle_core::constants::{self, BLOCK_SIZE};
    use tigerbeetle_lsm::manifest::ManifestLog;
    use tigerbeetle_lsm::schema::manifest_node::TableInfo as WireTableInfo;
    use tigerbeetle_lsm::scratch_memory::ScratchMemory;

    fn new_test_grid() -> Grid {
        Grid::new(GridOptions {
            cache_blocks_count: 4096,
            stash_blocks_count: 4,
            read_iops_max: 0,
            write_iops_max: 0,
            free_set_blocks_count: Some(4096),
            free_set_blocks_capacity: None,
        })
    }

    /// Grid for the I/O dispatch tests: enough stash for the output table's index and
    /// value blocks plus a fresh stash block per in-flight write, and ≥1 write iops so
    /// `create_block` reaches the storage (write_iops_max = 0 would queue forever).
    fn new_dispatch_grid() -> (Grid, MemoryStorage) {
        let mut grid = Grid::new(GridOptions {
            cache_blocks_count: 4096,
            stash_blocks_count: 16,
            read_iops_max: 1,
            write_iops_max: 2,
            free_set_blocks_count: Some(4096),
            free_set_blocks_capacity: None,
        });
        grid.attach_superblock_view(SuperBlockView {
            cluster: 0xDEAD_BEEF_u128,
            release: Release { value: 3 },
            storage_size: Zone::Grid.start(),
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        });

        // The output blocks are written to storage; size the grid zone generously.
        let storage_size =
            Zone::Grid.start() + 4096 * (BLOCK_SIZE as u64) + constants::SECTOR_SIZE as u64;
        let storage = MemoryStorage::new(storage_size);
        (grid, storage)
    }

    fn new_test_tree() -> Tree<TestSpec> {
        let config = TreeConfig { id: 1, name: "test" };
        let options = TreeOptions { batch_value_count_limit: 32 };
        Tree::<TestSpec>::new(config, options)
    }

    // -- Test key/value types (shared with tree tests) --

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
    struct TestKey(u64);

    impl ManifestTableKey for TestKey {
        const MAX: Self = Self(u64::MAX);
        const ZERO: Self = Self(0);

        fn to_bytes(self) -> [u8; 32] {
            let mut buf = [0u8; 32];
            buf[..8].copy_from_slice(&self.0.to_le_bytes());
            buf
        }

        fn from_bytes(bytes: &[u8; 32]) -> Self {
            let mut key_bytes = [0u8; 8];
            key_bytes.copy_from_slice(&bytes[..8]);
            Self(u64::from_le_bytes(key_bytes))
        }

        fn to_sort_key_high(self) -> u64 {
            self.0
        }
    }

    impl VsrTableKey for TestKey {
        fn to_le_bytes_padded(self) -> [u8; 32] {
            let mut buf = [0u8; 32];
            buf[..8].copy_from_slice(&self.0.to_le_bytes());
            buf
        }

        fn from_le_bytes_padded(bytes: &[u8; 32]) -> Self {
            let mut key_bytes = [0u8; 8];
            key_bytes.copy_from_slice(&bytes[..8]);
            Self(u64::from_le_bytes(key_bytes))
        }

        const SENTINEL_KEY: Self = Self(u64::MAX);
    }

    impl tigerbeetle_lsm::k_way_merge::TournamentKey for TestKey {
        const SENTINEL_KEY: Self = Self(u64::MAX);
        const MIN_KEY: Self = Self(u64::MIN);
    }

    impl tigerbeetle_core::stdx::radix::RadixKey for TestKey {
        const BITS: u32 = 64;

        fn digit(self, shift: u32, bits: u32) -> usize {
            let mask: u64 = ((1_u128 << bits) - 1) as u64;
            ((self.0 >> shift) & mask) as usize
        }
    }

    impl BlockValue for TestKey {
        fn write_bytes(&self, buf: &mut [u8]) {
            buf[..8].copy_from_slice(&self.0.to_le_bytes());
        }

        fn from_bytes(bytes: &[u8]) -> Self {
            let mut key_bytes = [0u8; 8];
            key_bytes.copy_from_slice(&bytes[..8]);
            Self(u64::from_le_bytes(key_bytes))
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct TestValue {
        key: TestKey,
        data: u64,
    }

    impl BlockValue for TestValue {
        fn write_bytes(&self, buf: &mut [u8]) {
            buf[..8].copy_from_slice(&self.key.0.to_le_bytes());
            buf[8..16].copy_from_slice(&self.data.to_le_bytes());
        }

        fn from_bytes(bytes: &[u8]) -> Self {
            let mut key_bytes = [0u8; 8];
            key_bytes.copy_from_slice(&bytes[..8]);
            let mut data_bytes = [0u8; 8];
            data_bytes.copy_from_slice(&bytes[8..16]);
            Self {
                key: TestKey(u64::from_le_bytes(key_bytes)),
                data: u64::from_le_bytes(data_bytes),
            }
        }
    }

    struct TestSpec;

    impl Table for TestSpec {
        type Key = TestKey;
        type Value = TestValue;
        const VALUE_COUNT_MAX: usize = 128;
        const USAGE: tm::Usage = tm::Usage::General;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            value.key
        }

        fn tombstone(value: &Self::Value) -> bool {
            value.key.0 == u64::MAX && value.data == 0
        }
    }

    impl crate::table::TableSpec for TestSpec {
        type Key = TestKey;
        type Value = TestValue;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            value.key
        }

        const SENTINEL_KEY: Self::Key = TestKey(u64::MAX);

        fn tombstone(value: &Self::Value) -> bool {
            value.key.0 == u64::MAX && value.data == 0
        }

        fn tombstone_from_key(key: Self::Key) -> Self::Value {
            TestValue { key, data: 0 }
        }

        const VALUE_COUNT_MAX: usize = 128;
        const USAGE: crate::table::TableUsage = crate::table::TableUsage::General;
    }

    impl crate::tree::TreeSpec for TestSpec {
        // DEVIATION-safe: compute() derives offsets from HEADER_SIZE like the comptime
        // layout in upstream, so the hand-written tree.rs test layout (which uses the
        // 128-byte upstream header) is avoided here where blocks are actually written.
        const LAYOUT: TableLayout = TableLayout::compute(8, 16, 128);

        fn index_blocks_for_key(
            _: &[u8],
            _: Self::Key,
        ) -> Option<crate::table::IndexBlocks<Self::Key>> {
            None
        }

        fn value_block_search(_: &[u8], _: Self::Key) -> Option<Self::Value> {
            None
        }

        fn tombstone_from_key(key: Self::Key) -> Self::Value {
            TestValue { key, data: 0 }
        }
    }

    // -- CompactionCounters tests --

    #[test]
    fn counters_consistent() {
        let c = CompactionCounters { in_: 10, dropped: 3, out: 7 };
        assert!(c.consistent());
    }

    #[test]
    fn counters_inconsistent() {
        let c = CompactionCounters { in_: 10, dropped: 3, out: 6 };
        assert!(!c.consistent());
    }

    // -- Quotas tests --

    #[test]
    fn quotas_beat_exhausted() {
        let q = Quotas { beat: 5, beat_done: 5, half_bar: 10, half_bar_done: 5 };
        assert!(q.beat_exhausted());
    }

    #[test]
    fn quotas_half_bar_exhausted() {
        let q = Quotas { beat: 5, beat_done: 3, half_bar: 10, half_bar_done: 10 };
        assert!(q.half_bar_exhausted());
    }

    // -- values_copy tests --

    #[test]
    fn values_copy_basic() {
        let source =
            [TestValue { key: TestKey(1), data: 10 }, TestValue { key: TestKey(2), data: 20 }];
        let mut target = [TestValue::default(); 4];
        let n = values_copy(&mut target, &source);
        assert_eq!(n, 2);
        assert_eq!(target[0], source[0]);
        assert_eq!(target[1], source[1]);
    }

    #[test]
    fn values_copy_truncates() {
        let source =
            [TestValue { key: TestKey(1), data: 10 }, TestValue { key: TestKey(2), data: 20 }];
        let mut target = [TestValue::default(); 1];
        let n = values_copy(&mut target, &source);
        assert_eq!(n, 1);
        assert_eq!(target[0], source[0]);
    }

    // -- values_copy_drop_tombstones tests --

    #[test]
    fn copy_drop_tombstones_drops() {
        let tombstone = TestValue { key: TestKey(u64::MAX), data: 0 };
        let source = [
            TestValue { key: TestKey(1), data: 10 },
            tombstone,
            TestValue { key: TestKey(3), data: 30 },
        ];
        let mut target = [TestValue::default(); 3];
        let r = values_copy_drop_tombstones::<TestSpec>(&mut target, &source);
        assert_eq!(r.consumed, 3);
        assert_eq!(r.dropped, 1);
        assert_eq!(r.produced, 2);
        assert_eq!(target[0], source[0]);
        assert_eq!(target[1], source[2]);
    }

    // -- values_merge tests --

    #[test]
    fn merge_interleaves() {
        // Merge only processes values while BOTH sources have data.
        // Remaining values from the unconsumed source are handled by the caller.
        let a = [TestValue { key: TestKey(1), data: 10 }, TestValue { key: TestKey(3), data: 30 }];
        let b = [TestValue { key: TestKey(2), data: 20 }, TestValue { key: TestKey(4), data: 40 }];
        let mut target = [TestValue::default(); 4];
        let r = values_merge::<TestSpec>(&mut target, &a, &b, false);
        // a[0]=1, b[0]=2, a[1]=3 — then a is exhausted, loop exits. b[1]=4 is unconsumed.
        assert_eq!(r.produced, 3);
        assert_eq!(r.consumed_a, 2);
        assert_eq!(r.consumed_b, 1);
        assert_eq!(target[0], a[0]); // key 1 from a
        assert_eq!(target[1], b[0]); // key 2 from b
        assert_eq!(target[2], a[1]); // key 3 from a
    }

    #[test]
    fn merge_a_wins_on_equal_key() {
        let a = [TestValue { key: TestKey(1), data: 100 }];
        let b = [TestValue { key: TestKey(1), data: 200 }];
        let mut target = [TestValue::default(); 4];
        let r = values_merge::<TestSpec>(&mut target, &a, &b, false);
        assert_eq!(r.produced, 1);
        assert_eq!(r.dropped, 1);
        assert_eq!(target[0], a[0]); // A wins
    }

    #[test]
    fn merge_drops_tombstones() {
        let a = [TestValue { key: TestKey(u64::MAX), data: 0 }]; // tombstone
        let b = [TestValue { key: TestKey(1), data: 20 }];
        let mut target = [TestValue::default(); 4];
        let r = values_merge::<TestSpec>(&mut target, &a, &b, true);
        assert_eq!(r.produced, 1);
        assert_eq!(target[0], b[0]);
    }

    // -- Immutable-table merge/copy tests --
    //
    // The `_immutable` variants operate on an `ImmutableTableIterator` (level A's in-memory
    // table) instead of a plain slice; these feed the L0-flush dispatch path.

    /// Builds a single-run ascending `ImmutableTableIterator` over `values`.
    fn immutable_iterator<S: TableTrait>(values: &[S::Value]) -> ImmutableTableIterator<'_, S> {
        let context =
            MergeContext::<S> { streams: [values; tm::SORTED_RUNS_MAX], streams_count: 1 };
        ImmutableTableIterator::new(context, None, Direction::Ascending)
    }

    #[test]
    fn values_copy_immutable_respects_budget() {
        let source = [
            TestValue { key: TestKey(1), data: 10 },
            TestValue { key: TestKey(2), data: 20 },
            TestValue { key: TestKey(3), data: 30 },
        ];
        let mut target = [TestValue::default(); 4];
        let mut iterator = immutable_iterator(&source);
        let copied = values_copy_immutable::<TestSpec>(&mut target, &mut iterator, 2);
        assert_eq!(copied, 2);
        assert_eq!(target[0], source[0]);
        assert_eq!(target[1], source[1]);
        assert_eq!(iterator.count_remaining(), 1);
    }

    #[test]
    fn copy_drop_tombstones_immutable_drops() {
        let tombstone = TestValue { key: TestKey(u64::MAX), data: 0 };
        let source = [
            TestValue { key: TestKey(1), data: 10 },
            tombstone,
            TestValue { key: TestKey(3), data: 30 },
        ];
        let mut target = [TestValue::default(); 3];
        let mut iterator = immutable_iterator(&source);
        let r = values_copy_drop_tombstones_immutable::<TestSpec>(&mut target, &mut iterator, 8);
        assert_eq!(r.consumed, 3);
        assert_eq!(r.dropped, 1);
        assert_eq!(r.produced, 2);
        assert_eq!(target[0], source[0]);
        assert_eq!(target[1], source[2]);
    }

    #[test]
    fn merge_immutable_a_wins_and_interleaves() {
        // Merge only processes values while BOTH sources have data; remaining A values are
        // handled by the caller (the immutable tail).
        let a =
            [TestValue { key: TestKey(1), data: 100 }, TestValue { key: TestKey(3), data: 300 }];
        let b = [TestValue { key: TestKey(2), data: 20 }, TestValue { key: TestKey(4), data: 40 }];
        let mut target = [TestValue::default(); 4];
        let mut iterator_a = immutable_iterator(&a);
        let r = values_merge_immutable::<TestSpec>(&mut target, &mut iterator_a, &b, false, 8);
        assert_eq!(r.produced, 3);
        assert_eq!(r.consumed_a, 2);
        assert_eq!(r.consumed_b, 1);
        assert_eq!(target[0], a[0]); // key 1 from a
        assert_eq!(target[1], b[0]); // key 2 from b
        assert_eq!(target[2], a[1]); // key 3 from a
    }

    #[test]
    fn merge_immutable_equal_key_a_wins() {
        let a = [TestValue { key: TestKey(1), data: 100 }];
        let b = [TestValue { key: TestKey(1), data: 200 }];
        let mut target = [TestValue::default(); 4];
        let mut iterator_a = immutable_iterator(&a);
        let r = values_merge_immutable::<TestSpec>(&mut target, &mut iterator_a, &b, false, 8);
        // Equal keys collapse: both consumed, only A's value emitted.
        assert_eq!(r.produced, 1);
        assert_eq!(r.consumed_a, 1);
        assert_eq!(r.consumed_b, 1);
        assert_eq!(target[0], a[0]);
    }

    #[test]
    fn merge_immutable_equal_key_drops_tombstone() {
        // A tombstone (key MAX) colliding with a real B value collapses into a drop.
        let a = [TestValue { key: TestKey(u64::MAX), data: 0 }]; // tombstone
        let b = [TestValue { key: TestKey(u64::MAX), data: 200 }];
        let mut target = [TestValue::default(); 4];
        let mut iterator_a = immutable_iterator(&a);
        let r = values_merge_immutable::<TestSpec>(&mut target, &mut iterator_a, &b, true, 8);
        assert_eq!(r.produced, 0);
        assert_eq!(r.consumed_a, 1);
        assert_eq!(r.consumed_b, 1);
        assert_eq!(r.dropped, 2);
    }

    // -- Secondary-index merge tests --
    //
    // A secondary index never "drops" tombstones by filtering; instead matching put/remove
    // pairs at the same key cancel each other (upstream asserts the pair is exactly one put and
    // one remove). These branches are unreachable through the General-usage `TestSpec`.

    struct TestSecondarySpec;

    impl Table for TestSecondarySpec {
        type Key = TestKey;
        type Value = TestValue;
        const VALUE_COUNT_MAX: usize = 128;
        const USAGE: tm::Usage = tm::Usage::SecondaryIndex;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            value.key
        }

        fn tombstone(value: &Self::Value) -> bool {
            value.data == u64::MAX
        }
    }

    #[test]
    fn merge_secondary_index_cancels_put_remove_pair() {
        let a = [TestValue { key: TestKey(5), data: 100 }]; // put
        let b = [TestValue { key: TestKey(5), data: u64::MAX }]; // remove
        let mut target = [TestValue::default(); 4];
        let r = values_merge::<TestSecondarySpec>(&mut target, &a, &b, false);
        // The pair is consumed and cancels: neither value is produced.
        assert_eq!(r.produced, 0);
        assert_eq!(r.consumed_a, 1);
        assert_eq!(r.consumed_b, 1);
        assert_eq!(r.dropped, 2);
        assert_eq!(r.produced, r.consumed_a + r.consumed_b - r.dropped);
    }

    #[test]
    fn merge_secondary_index_cancels_remove_put_pair() {
        let a = [TestValue { key: TestKey(5), data: u64::MAX }]; // remove
        let b = [TestValue { key: TestKey(5), data: 100 }]; // put
        let mut target = [TestValue::default(); 4];
        let r = values_merge::<TestSecondarySpec>(&mut target, &a, &b, false);
        assert_eq!(r.produced, 0);
        assert_eq!(r.consumed_a, 1);
        assert_eq!(r.consumed_b, 1);
        assert_eq!(r.dropped, 2);
    }

    #[test]
    fn merge_secondary_index_two_puts_panic() {
        // Two non-tombstone values at the same key violate the cancel-pair invariant.
        let a = [TestValue { key: TestKey(5), data: 100 }];
        let b = [TestValue { key: TestKey(5), data: 200 }];
        let mut target = [TestValue::default(); 4];
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values_merge::<TestSecondarySpec>(&mut target, &a, &b, false);
        }))
        .expect_err("two puts at equal key must panic");
    }

    #[test]
    fn merge_immutable_secondary_index_cancels_put_remove_pair() {
        let a = [TestValue { key: TestKey(5), data: 100 }]; // put
        let b = [TestValue { key: TestKey(5), data: u64::MAX }]; // remove
        let mut target = [TestValue::default(); 4];
        let mut iterator_a = immutable_iterator(&a);
        let r =
            values_merge_immutable::<TestSecondarySpec>(&mut target, &mut iterator_a, &b, false, 8);
        assert_eq!(r.produced, 0);
        assert_eq!(r.consumed_a, 1);
        assert_eq!(r.consumed_b, 1);
        assert_eq!(r.dropped, 2);
    }

    #[test]
    fn copy_drop_tombstones_secondary_index_panics() {
        // Tombstone-dropping copy is only legal for General usage.
        let tombstone = TestValue { key: TestKey(5), data: u64::MAX };
        let source = [TestValue { key: TestKey(1), data: 10 }, tombstone];
        let mut target = [TestValue::default(); 2];
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values_copy_drop_tombstones::<TestSecondarySpec>(&mut target, &source);
        }))
        .expect_err("tombstone drop on a secondary index must panic");
    }

    // -- Compaction lifecycle tests --

    #[test]
    fn compaction_new_and_reset() {
        let mut tree = new_test_tree();
        let mut grid = new_test_grid();

        let mut compaction = Compaction::<TestSpec>::new(
            core::ptr::addr_of_mut!(tree),
            core::ptr::addr_of_mut!(grid),
            1,
        );
        assert_eq!(compaction.stage, Stage::Inactive);
        assert_eq!(compaction.level_b, 1);
        assert!(compaction.is_idle());
        assert!(compaction.manifest_entries.is_empty());

        compaction.reset();
        assert_eq!(compaction.stage, Stage::Inactive);
        assert!(compaction.is_idle());
    }

    #[test]
    fn compaction_assert_between_bars() {
        let mut tree = new_test_tree();
        let mut grid = new_test_grid();

        let compaction = Compaction::<TestSpec>::new(
            core::ptr::addr_of_mut!(tree),
            core::ptr::addr_of_mut!(grid),
            0,
        );
        compaction.assert_between_bars(); // Should not panic
    }

    #[test]
    fn beat_commence_caps_beat_by_remaining_half_bar_quota() {
        let mut tree = new_test_tree();
        let mut grid = new_test_grid();

        let mut compaction = Compaction::<TestSpec>::new(
            core::ptr::addr_of_mut!(tree),
            core::ptr::addr_of_mut!(grid),
            0,
        );
        // Ready for a half-bar: paused stage, a 10-value quota, nothing consumed yet.
        compaction.stage = Stage::Paused;
        compaction.quotas = Quotas { beat: 0, beat_done: 0, half_bar: 10, half_bar_done: 0 };
        assert!(compaction.is_idle());

        // The first beat is capped by the caller's values_count.
        compaction.beat_commence(6);
        assert_eq!((compaction.quotas.beat, compaction.quotas.beat_done), (6, 0));

        // Complete the beat (advance cumulative half-bar progress); the next beat is capped by
        // the *remaining* half-bar quota (10 − 6 = 4) rather than the 6 requested.
        compaction.quotas.beat_done = 6;
        compaction.quotas.half_bar_done = 6;
        assert!(compaction.is_idle());
        compaction.beat_commence(6);
        assert_eq!((compaction.quotas.beat, compaction.quotas.beat_done), (4, 0));

        // Complete the final beat; a zero-value beat is allowed once the half-bar is exhausted.
        compaction.quotas.beat_done = 4;
        compaction.quotas.half_bar_done = 10;
        assert!(compaction.is_idle());
        compaction.beat_commence(0);
        assert_eq!((compaction.quotas.beat, compaction.quotas.beat_done), (0, 0));
    }

    #[test]
    fn beat_commence_move_table_requires_half_bar_exhausted() {
        let mut tree = new_test_tree();
        let mut grid = new_test_grid();

        let mut compaction = Compaction::<TestSpec>::new(
            core::ptr::addr_of_mut!(tree),
            core::ptr::addr_of_mut!(grid),
            1,
        );
        // A move-table compaction consumes its whole quota up front, so commence must observe
        // the half-bar already exhausted; the resulting beat quota is zero.
        compaction.stage = Stage::Paused;
        compaction.move_table = true;
        compaction.quotas = Quotas { beat: 0, beat_done: 0, half_bar: 3, half_bar_done: 3 };
        assert!(compaction.is_idle());

        compaction.beat_commence(3);
        assert_eq!((compaction.quotas.beat, compaction.quotas.beat_done), (0, 0));
    }

    // -- I/O dispatch tests --

    /// Minimal, viable manifest log backing for `half_bar_complete`.
    struct MockLog {
        entries: Vec<WireTableInfo>,
        opened: bool,
    }

    impl MockLog {
        /// A log that is *not* yet opened; `Manifest::open_commence` requires that.
        fn new_unopened() -> Self {
            Self { entries: Vec::new(), opened: false }
        }

        fn open(&mut self) {
            self.opened = true;
        }
    }

    impl ManifestLog for MockLog {
        fn is_opened(&self) -> bool {
            self.opened
        }

        fn append(&mut self, entry: &WireTableInfo) {
            assert!(self.opened);
            self.entries.push(*entry);
        }
    }

    #[test]
    fn dispatch_l0_flush_immutable_drop_tombstones_writes_table() {
        let mut tree = new_test_tree();
        // Seed 5 values: keys 1..5 with one tombstone at key u64::MAX.
        for key in 1..=5_u64 {
            tree.put(&TestValue { key: TestKey(key), data: key * 10 });
        }
        tree.put(&TestValue { key: TestKey(u64::MAX), data: 0 }); // tombstone
        tree.compact(&mut ScratchMemory::<TestValue>::new(128));
        tree.swap_mutable_and_immutable(1, &mut ScratchMemory::<TestValue>::new(128));
        assert_eq!(tree.table_immutable_ref().count(), 6);

        let (mut grid, mut storage) = new_dispatch_grid();

        let mut compaction = Compaction::<TestSpec>::new(
            core::ptr::addr_of_mut!(tree),
            core::ptr::addr_of_mut!(grid),
            0,
        );

        let values_count = u64::from(tree.table_immutable_ref().count());
        let quota_half_bar = compaction.half_bar_commence(8, &tree, &grid);
        assert_eq!(quota_half_bar, values_count);

        // Empty level 0 → drop_tombstones on.
        assert!(compaction.drop_tombstones);
        assert!(compaction.range_b.as_ref().is_some_and(|r| r.tables.tables.empty()));

        compaction.beat_commence(values_count);
        let reservation = compaction.reserve_output_blocks(&mut grid);

        compaction.dispatch(&mut storage, &tree, &mut grid, reservation);

        // Single beat consumed the whole (small) immutable table.
        assert_eq!(compaction.quotas.beat_done, values_count);
        assert_eq!(compaction.quotas.half_bar_done, values_count);
        assert_eq!(compaction.counters.in_, values_count);
        assert_eq!(compaction.counters.dropped, 1); // exactly the tombstone dropped
        assert_eq!(compaction.counters.out, values_count - 1);
        assert!(compaction.counters.consistent());
        assert_eq!(compaction.stage, Stage::Paused);
        assert!(compaction.quotas.beat_exhausted());
        assert!(compaction.quotas.half_bar_exhausted());
        assert_eq!(compaction.table_builder.state(), BuilderState::NoBlocks);

        // One output table queued for level 0.
        assert_eq!(compaction.manifest_entries.len(), 1);
        let entry = &compaction.manifest_entries[0];
        assert_eq!(entry.operation, ManifestEntryOperation::InsertToLevelB);
        assert_eq!(entry.table.value_count, (values_count - 1) as u32);
        // The tombstone key (u64::MAX) is excluded from the output range.
        assert_ne!(entry.table.key_min, TestKey(u64::MAX));
        assert_ne!(entry.table.key_max, TestKey(u64::MAX));

        // The output blocks were written: the address is acquired (not free/released) and
        // the value block is cached.
        let address = entry.table.address;
        assert!(address > 0);
        assert!(!grid.free_set_is_free(address));
        assert!(!grid.free_set_is_released(address));
        assert!(grid.cached_location(address).is_some());

        // half_bar_complete persists the output table & flushes the immutable.
        let mut log = MockLog::new_unopened();
        tree.manifest_mut().open_commence(&log);
        log.open();
        compaction.half_bar_complete(&mut tree, &grid, &mut log);
        assert_eq!(compaction.stage, Stage::Inactive);
        assert!(compaction.manifest_entries.is_empty());
        assert!(!log.entries.is_empty());
        assert_eq!(tree.manifest_ref().levels[0].table_count_visible(), 1);
        assert!(matches!(
            tree.table_immutable_ref().mutability(),
            Mutability::Immutable(state) if state.flushed
        ));

        grid.forfeit(reservation);
    }

    /// Build a single-value-block table for [`TestSpec`] (values must be strictly key-sorted).
    fn build_test_table(
        values: &[TestValue],
        value_address: u64,
        index_address: u64,
    ) -> (Vec<u8>, Vec<u8>, crate::table::TableInfo<TestKey>) {
        let layout = TestSpec::LAYOUT;
        let mut index_block = vec![0u8; BLOCK_SIZE];
        let mut value_block = vec![0u8; BLOCK_SIZE];

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);
        builder.set_value_block(&mut value_block);
        for value in values {
            builder.insert_block_value(value, &mut value_block, &layout);
        }
        builder.value_block_finish::<TestSpec>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: value_address,
                snapshot_min: 1,
                tree_id: 1,
            },
        );
        let info = builder.index_block_finish::<TestKey>(
            &mut index_block,
            &layout,
            IndexFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: index_address,
                snapshot_min: 1,
                tree_id: 1,
            },
        );
        (index_block, value_block, info)
    }

    /// Copy a finished block into the grid cache and return its header checksum.
    fn seed_grid_block(grid: &mut Grid, address: u64, block: &[u8]) -> u128 {
        assert!(!grid.free_set_is_free(address));
        let location = grid.get_block();
        grid.block_mut(location).copy_from_slice(block);
        let checksum = crate::schema::header_from_block(grid.block(location)).checksum;
        grid.cache_upsert(address, location);
        checksum
    }

    /// The L0-flush merge path with a level-0 disk table overlapping the immutable A: reads the
    /// level-B index/value blocks from the grid cache and merges them with the in-memory A.
    #[test]
    fn dispatch_l0_flush_immutable_with_level_b_overlap_merges() {
        let mut tree = new_test_tree();
        let mut log = MockLog::new_unopened();
        tree.manifest_mut().open_commence(&log);
        log.open();

        // Level-B table spans keys 0..8, strictly key-sorted.
        let level_b_values = [
            TestValue { key: TestKey(0), data: 0 },
            TestValue { key: TestKey(2), data: 200 },
            TestValue { key: TestKey(4), data: 400 },
            TestValue { key: TestKey(6), data: 600 },
            TestValue { key: TestKey(8), data: 800 },
        ];

        // Immutable A: keys 1..7 odd (interleave with B, no key overlap).
        for key in [1_u64, 3, 5, 7] {
            tree.put(&TestValue { key: TestKey(key), data: key * 10 });
        }
        tree.compact(&mut ScratchMemory::<TestValue>::new(128));
        tree.swap_mutable_and_immutable(1, &mut ScratchMemory::<TestValue>::new(128));
        assert_eq!(tree.table_immutable_ref().count(), 4);

        // Allocate and seed the level-B table's value and index blocks in the grid cache.
        let (mut grid, mut storage) = new_dispatch_grid();
        let reservation = grid.reserve(2);
        let value_address = grid.acquire(reservation);
        let index_address = grid.acquire(reservation);
        let (index_block, value_block, info) =
            build_test_table(&level_b_values, value_address, index_address);
        let index_checksum = seed_grid_block(&mut grid, index_address, &index_block);
        seed_grid_block(&mut grid, value_address, &value_block);

        let table: TreeTableInfo<TestKey> = TreeTableInfo {
            checksum: index_checksum,
            address: index_address,
            snapshot_min: 1,
            snapshot_max: tigerbeetle_lsm::tree::SNAPSHOT_LATEST,
            key_min: info.key_min,
            key_max: info.key_max,
            value_count: info.value_count,
        };
        tree.manifest_mut().insert_table(&mut log, 0, &table);

        let mut compaction = Compaction::<TestSpec>::new(
            core::ptr::addr_of_mut!(tree),
            core::ptr::addr_of_mut!(grid),
            0,
        );

        let quota_half_bar = compaction.half_bar_commence(8, &tree, &grid);
        // Quota = immutable (4) + level-B table (5).
        assert_eq!(quota_half_bar, 9);
        assert!(compaction.drop_tombstones);
        assert_eq!(compaction.range_b.as_ref().unwrap().tables.tables.count(), 1);

        assert_eq!(compaction.quotas.half_bar, 9);
        compaction.beat_commence(quota_half_bar);
        let reservation = compaction.reserve_output_blocks(&mut grid);

        compaction.dispatch(&mut storage, &tree, &mut grid, reservation);

        assert_eq!(compaction.quotas.beat_done, 9);
        assert_eq!(compaction.quotas.half_bar_done, 9);
        assert_eq!(compaction.counters.in_, 9);
        assert_eq!(compaction.counters.dropped, 0);
        assert_eq!(compaction.counters.out, 9);
        assert!(compaction.counters.consistent());
        assert_eq!(compaction.stage, Stage::Paused);
        assert!(compaction.quotas.beat_exhausted());
        assert!(compaction.quotas.half_bar_exhausted());

        // A single output table merging A and B.
        assert_eq!(compaction.manifest_entries.len(), 1);
        let entry = &compaction.manifest_entries[0];
        assert_eq!(entry.table.value_count, 9);
        assert_eq!(entry.table.key_min, TestKey(0));
        assert_eq!(entry.table.key_max, TestKey(8));

        // The level-B input blocks were released after being fully consumed.
        assert!(grid.free_set_is_released(value_address));
        assert!(grid.free_set_is_released(index_address));

        grid.forfeit(reservation);
    }

    /// The disk level-A merge path (`TableInfoA::Disk`, level_b > 0): seeds a level-0 manifest
    /// to its compaction threshold so `compaction_table(0)` selects the least-overlapping
    /// level-0 table (the lowest-keyed one, since the level-1 table spans every level-0 table)
    /// as disk level A, whose value block is read alongside the overlapping level-1 table
    /// (disk level B) and merged into a single output table.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn dispatch_disk_level_a_merges_with_level_b() {
        // Deterministic value factories.
        fn values_for(start_key: u64, count: u64) -> Vec<TestValue> {
            (0..count)
                .map(|i| TestValue { key: TestKey(start_key + i), data: (start_key + i) * 10 })
                .collect()
        }

        let mut tree = new_test_tree();
        let mut log = MockLog::new_unopened();
        tree.manifest_mut().open_commence(&log);
        log.open();

        let (mut grid, mut storage) = new_dispatch_grid();
        // Reservation for the 5 input tables' blocks (2 per table); forfeited after seeding.
        let reservation = grid.reserve(10);

        // Level 0 tables: T0 keys 1..4, T1 11..14, T2 21..24, T3 31..34. Level 1 spans 2..34 so
        // it overlaps every level-0 table (forcing a merge, and tie-broken to T0 as the lowest).
        let level_0_sets =
            [values_for(1, 4), values_for(11, 4), values_for(21, 4), values_for(31, 4)];
        let level_1_set = vec![
            TestValue { key: TestKey(2), data: 102 },
            TestValue { key: TestKey(3), data: 103 },
            TestValue { key: TestKey(4), data: 104 },
            TestValue { key: TestKey(12), data: 112 },
            TestValue { key: TestKey(22), data: 122 },
            TestValue { key: TestKey(32), data: 132 },
        ];

        // Build + seed the level-0 tables.
        let mut level_0_tables = Vec::new();
        for values in &level_0_sets {
            let value_address = grid.acquire(reservation);
            let index_address = grid.acquire(reservation);
            let (index_block, value_block, info) =
                build_test_table(values, value_address, index_address);
            let index_checksum = seed_grid_block(&mut grid, index_address, &index_block);
            seed_grid_block(&mut grid, value_address, &value_block);
            level_0_tables.push(TreeTableInfo::<TestKey> {
                checksum: index_checksum,
                address: index_address,
                snapshot_min: 1,
                snapshot_max: tigerbeetle_lsm::tree::SNAPSHOT_LATEST,
                key_min: info.key_min,
                key_max: info.key_max,
                value_count: info.value_count,
            });
        }

        // Build + seed the single level-1 table.
        let b_value_address = grid.acquire(reservation);
        let b_index_address = grid.acquire(reservation);
        let (b_index_block, b_value_block, b_info) =
            build_test_table(&level_1_set, b_value_address, b_index_address);
        let b_index_checksum = seed_grid_block(&mut grid, b_index_address, &b_index_block);
        seed_grid_block(&mut grid, b_value_address, &b_value_block);
        let level_1_table = TreeTableInfo::<TestKey> {
            checksum: b_index_checksum,
            address: b_index_address,
            snapshot_min: 1,
            snapshot_max: tigerbeetle_lsm::tree::SNAPSHOT_LATEST,
            key_min: b_info.key_min,
            key_max: b_info.key_max,
            value_count: b_info.value_count,
        };

        // The seeded input blocks are now referenced by the manifest; the reservation that
        // acquired them must be forfeited so the grid's outstanding-reservation counter is
        // back to zero before `dispatch` / the output reservation runs.
        grid.forfeit(reservation);

        for table in &level_0_tables {
            tree.manifest_mut().insert_table(&mut log, 0, table);
        }
        tree.manifest_mut().insert_table(&mut log, 1, &level_1_table);
        assert_eq!(tree.manifest_ref().levels[0].table_count_visible(), 4);

        let mut compaction = Compaction::<TestSpec>::new(
            core::ptr::addr_of_mut!(tree),
            core::ptr::addr_of_mut!(grid),
            1,
        );

        let quota_half_bar = compaction.half_bar_commence(8, &tree, &grid);
        // table_a value_count 4 + range_b value_count 6.
        assert_eq!(quota_half_bar, 10);
        assert!(!compaction.move_table);
        assert!(matches!(compaction.table_info_a, Some(TableInfoA::Disk(_))));

        compaction.beat_commence(quota_half_bar);
        let reservation = compaction.reserve_output_blocks(&mut grid);

        compaction.dispatch(&mut storage, &tree, &mut grid, reservation);

        assert_eq!(compaction.quotas.beat_done, 10);
        assert_eq!(compaction.quotas.half_bar_done, 10);
        assert_eq!(compaction.counters.in_, 10);
        // Keys 2,3,4 are present in both inputs; each pair collapses to one (A wins) → 3 dropped.
        assert_eq!(compaction.counters.dropped, 3);
        assert_eq!(compaction.counters.out, 7);
        assert!(compaction.counters.consistent());
        assert_eq!(compaction.stage, Stage::Paused);
        assert!(compaction.quotas.beat_exhausted());
        assert!(compaction.quotas.half_bar_exhausted());

        // A single output table merging the selected level-0 and level-1 tables (keys 1..32).
        assert_eq!(compaction.manifest_entries.len(), 1);
        let entry = &compaction.manifest_entries[0];
        assert_eq!(entry.table.value_count, 7);
        assert_eq!(entry.table.key_min, TestKey(1));
        assert_eq!(entry.table.key_max, TestKey(32));

        // The level-A and level-B input blocks are released once fully consumed.
        assert!(grid.free_set_is_released(level_0_tables[0].address));
        assert!(grid.free_set_is_released(b_index_address));

        grid.forfeit(reservation);
    }
}
