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
use tigerbeetle_lsm::compaction::{COMPACTION_TABLES_INPUT_MAX, COMPACTION_TABLES_OUTPUT_MAX};
use tigerbeetle_lsm::manifest::{CompactionRange, TableInfoReference, TreeTableInfo};
use tigerbeetle_lsm::table_memory::{
    ImmutableTableIterator, Mutability, Table as TableTrait, Usage,
};

use crate::grid::Grid;
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
/// Phase 3: ResourcePool, I/O dispatch, read/write callbacks.
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

    // Table builder fields — placeholder until TableBuilder is ported:
    // upstream: table_builder: Table.Builder,
    // upstream: table_builder_index_block: ?*ResourcePool.Block,
    // upstream: table_builder_value_block: ?*ResourcePool.Block,
    pub table_builder_value_count: u32,

    pub level_a_immutable_stage: LevelAImmutableStage,

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
            table_builder_value_count: 0,
            level_a_immutable_stage: LevelAImmutableStage::Ready,
            pool_is_active: false,
            callback_is_set: false,
        }
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
        self.table_builder_value_count = 0;
        self.level_a_immutable_stage = LevelAImmutableStage::Ready;
        self.pool_is_active = false;
        self.callback_is_set = false;
    }

    /// Port of upstream `Compaction.assert_between_bars`.
    #[allow(clippy::missing_panics_doc)]
    pub fn assert_between_bars(&self) {
        assert_eq!(self.stage, Stage::Inactive);
        assert!(self.is_idle());
        assert!(self.block_queues_empty_input());
        // upstream also asserts:
        // - table_builder.state == .no_blocks
        // - table_builder_value_block == null
        // - table_builder_index_block == null
        // - manifest_entries.empty()
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
        // TODO(port): assert table_builder state == .no_blocks (Phase 3)

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
        self.table_builder_value_count = 0;
        self.level_a_immutable_stage = LevelAImmutableStage::Ready;
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
    use tigerbeetle_lsm::manifest::TableKey as ManifestTableKey;
    use tigerbeetle_lsm::table_memory::{self as tm, Table};
    use tigerbeetle_lsm::tree::TreeConfig;

    use crate::grid::{Grid, GridOptions};
    use crate::tree::{Options as TreeOptions, Tree};

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

    impl crate::tree::TreeSpec for TestSpec {
        const LAYOUT: TableLayout = TableLayout {
            block_value_count_max: 128,
            value_block_count_max: 1,
            index: crate::schema::TableIndex {
                key_size: 8,
                value_block_count_max: 1,
                size: 184,
                value_checksums_offset: 128,
                value_checksums_size: 32,
                keys_min_offset: 160,
                keys_max_offset: 168,
                keys_size: 8,
                value_addresses_offset: 176,
                value_addresses_size: 8,
                padding_offset: 184,
                padding_size: 3912,
            },
            data: crate::schema::TableValue {
                value_size: 16,
                value_count_max: 128,
                values_offset: 128,
                values_size: 2048,
                padding_offset: 2176,
                padding_size: 1920,
            },
        };

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
}
