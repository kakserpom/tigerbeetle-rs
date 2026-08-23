//! Memory tables for an LSM (log-structured merge) stage.
//! Port of `src/lsm/table_memory.zig`.
//!
//! Each tree maintains two in-memory tables:
//! - Mutable table: accepts all updates (e.g., `Tree.put`).
//! - Immutable table: read-only staging area for the next flush to disk.
//!
//! New puts are appended to the mutable table. At the end of each beat, the new mutable suffix is
//! sorted and deduplicated into a sorted run. At a bar boundary, the mutable table is staged as the
//! next immutable table:
//! - If the previous immutable table was flushed to disk, [`TableMemory::compact`] swaps the
//!   mutable table's backing storage and run tracker into the immutable table.
//! - If the previous immutable table was not flushed, [`TableMemory::absorb`] retains that
//!   immutable run, appends the mutable runs, and re-materializes the combined table as the
//!   immutable table.
//!
//! The immutable table may therefore contain multiple sorted runs. Reads and disk flushes use
//! [`ImmutableTableIterator`] to merge those runs and deduplicate equal keys lazily.
//!
//! Optimizations:
//! 1) Sorted runs:
//!    - Beat compaction sorts only the newly appended mutable suffix.
//!    - The run tracker records contiguous sorted ranges and their origin so that newer mutable
//!      runs win during lazy cross-run deduplication.
//! 2) Deferred disk flush:
//!    - If the immutable table is not sufficiently full, compaction may choose not to flush it
//!      to disk. The next bar then absorbs the new mutable table into the retained immutable
//!      table, avoiding a disk flush for a small table.

// DEVIATION: upstream Zig relies on arbitrary-width integer arithmetic with implicit safe
// truncation; run/stream counts here are bounded well below u32/u8 maxima by construction
// (see SORTED_RUNS_MAX).
#![allow(clippy::cast_possible_truncation)]

use tigerbeetle_core::constants::{LSM_COMPACTION_OPS, VERIFY};
use tigerbeetle_core::stdx::radix::{self, RadixKey};

use crate::binary_search::{Config, Mode, binary_search_values, binary_search_values_range};
use crate::direction::Direction;
use crate::k_way_merge::{TournamentKey, TournamentTree};
use crate::scratch_memory::ScratchMemory;

/// Upstream: `sorted_runs_max = constants.lsm_compaction_ops + 2` — at most: LSM compactions +
/// one sort() call + one immutable run for absorb.
pub const SORTED_RUNS_MAX: usize = LSM_COMPACTION_OPS + 2;

/// Upstream: `Table.usage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Usage {
    General,
    SecondaryIndex,
}

/// Port of upstream's comptime `Table` parameter of `TableMemoryType(Table)`:
/// provides `Key`, `Value`, `value_count_max`, `usage`, `key_from_value`, `tombstone`.
///
/// DEVIATION: `Value: Default` is required because the backing storage is eagerly initialized
/// (upstream allocates `undefined` memory; entries past `count` are never read). `Value: Copy`
/// matches upstream's POD values (needed by the shared scratch buffer abstraction).
pub trait Table {
    type Key: TournamentKey + RadixKey;
    type Value: Copy + Default + core::fmt::Debug + PartialEq;
    /// Upstream: `Table.value_count_max`.
    const VALUE_COUNT_MAX: usize;
    /// Upstream: `Table.usage`.
    const USAGE: Usage;
    /// Upstream: `Table.key_from_value`.
    fn key_from_value(value: &Self::Value) -> Self::Key;
    /// Upstream: `Table.tombstone`.
    fn tombstone(value: &Self::Value) -> bool;
}

/// Where the sorted run originated; affects merge precedence on equal keys.
/// We prefer the immutable ones to keep ordering for deduplication (last key wins).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunOrigin {
    Mutable,
    Immutable,
}

/// A contiguous sorted range in `values[index_min..index_max]`.
#[derive(Clone, Copy, Debug)]
pub struct SortedRun {
    pub index_min: u32, // inclusive
    pub index_max: u32, // exclusive
    /// Where the run originated; affects merge precedence on equal keys.
    pub origin: RunOrigin,
}

impl SortedRun {
    fn is_empty(&self) -> bool {
        self.index_min == self.index_max
    }
}

/// Sorted run slices for the immutable table iterator.
///
/// DEVIATION: streams borrow from the table's values (`[]const Value` upstream).
pub struct MergeContext<'a, T: Table> {
    pub streams: [&'a [T::Value]; SORTED_RUNS_MAX],
    pub streams_count: u32,
}

/// Maintains per-table mutable state that must be snapshotted for "scopes".
/// When a scope is opened (e.g., in tree), we copy [`ValueContext`] so we can
/// roll back both the count and the sorted-run tracker if the scope is discarded.
#[derive(Clone, Copy, Debug)]
pub struct ValueContext {
    pub count: u32,
    pub run_tracker: SortedRunTracker,
}

#[derive(Clone, Copy, Debug)]
pub struct SortedRunTracker {
    /// Invariants:
    /// - Runs are in ascending order.
    /// - Runs have no gaps between them.
    /// - There is at most one run with origin = Immutable.
    ///
    /// DEVIATION: upstream leaves unused slots `undefined`; here they hold a dummy value.
    runs: [SortedRun; SORTED_RUNS_MAX],
    runs_count: u8,
}

impl Default for SortedRunTracker {
    fn default() -> Self {
        Self {
            runs: [SortedRun { index_min: 0, index_max: 0, origin: RunOrigin::Mutable };
                SORTED_RUNS_MAX],
            runs_count: 0,
        }
    }
}

impl SortedRunTracker {
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, run: SortedRun) {
        if run.is_empty() {
            return; // Ignore empty runs.
        }

        self.runs[usize::from(self.runs_count)] = run;
        self.runs_count += 1;
    }

    /// Adds a new run at the front and shifts the other runs by the offset to maintain
    /// the invariant that no run overlaps and they have no gaps.
    fn add_front_and_propagate_offset(&mut self, run: SortedRun) {
        if run.is_empty() {
            return; // Ignore empty runs.
        }
        assert_eq!(run.index_min, 0);
        assert!(usize::from(self.runs_count) < self.runs.len());

        self.runs.copy_within(0..usize::from(self.runs_count), 1);

        self.runs[0] = run;
        self.runs_count += 1;

        // Propagate the new offset to the remaining runs.
        let runs_count = self.count();
        for run_old in &mut self.runs[1..runs_count] {
            run_old.index_min += run.index_max;
            run_old.index_max += run.index_max;
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn count(&self) -> usize {
        usize::from(self.runs_count)
    }

    fn merge_context<'a, T: Table>(&self, values: &'a [T::Value]) -> MergeContext<'a, T> {
        let mut context = MergeContext::<T> { streams: [&[]; SORTED_RUNS_MAX], streams_count: 0 };

        let mut stream_idx: usize = 0;

        // Place the immutable run first so smaller stream_id wins on ties.
        for run in &self.runs[..self.count()] {
            if run.origin != RunOrigin::Immutable {
                continue;
            }
            context.streams[stream_idx] = &values[run.index_min as usize..run.index_max as usize];
            stream_idx += 1;
            break;
        }
        // Now place all the mutable runs.
        for run in &self.runs[..self.count()] {
            if run.origin == RunOrigin::Immutable {
                continue;
            }
            context.streams[stream_idx] = &values[run.index_min as usize..run.index_max as usize];
            stream_idx += 1;
        }
        context.streams_count = stream_idx as u32;
        context
    }

    fn last(&self) -> Option<&SortedRun> {
        if self.count() == 0 {
            return None;
        }
        Some(&self.runs[self.count() - 1])
    }

    fn assert_invariants(&self, table_count: u32) {
        let runs_count = self.count();

        if runs_count == 0 {
            return;
        }

        assert_eq!(self.runs[0].index_min, 0);
        assert_eq!(self.runs[runs_count - 1].index_max, table_count);

        for (a, b) in self.runs[..runs_count - 1].iter().zip(self.runs[1..runs_count].iter()) {
            assert!(a.index_min < b.index_min); // Ordered and we ignore empty runs.
            assert_eq!(a.index_max, b.index_min); // No gaps.
        }

        let immutable_runs =
            self.runs[..runs_count].iter().filter(|r| r.origin == RunOrigin::Immutable).count();
        assert!(immutable_runs == 0 || immutable_runs == 1);
    }
}

/// An empty immutable table has nothing to flush.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImmutableState {
    pub flushed: bool,
    /// Only used for assertions, to verify that we don't absorb the
    /// mutable table immediately prior to checkpoint.
    pub absorbed: bool,
    pub snapshot_min: u64,
}

impl ImmutableState {
    const DEFAULT: Self = Self { flushed: true, absorbed: false, snapshot_min: 0 };
}

#[derive(Clone, Copy, Debug)]
pub enum Mutability {
    Mutable,
    /// An empty table has nothing to flush.
    Immutable(ImmutableState),
}

/// Port of `TableMemoryType(Table)`.
pub struct TableMemory<T: Table> {
    values: Box<[T::Value]>,
    value_context: ValueContext,
    mutability: Mutability,
    // Used by tree.zig logging/flush paths upstream (not yet ported).
    #[allow(dead_code)]
    name: &'static str,
}

impl<T: Table> TableMemory<T> {
    /// Merges values with identical keys (last one wins) and collapses tombstones for
    /// secondary indexes in a streaming fashion.
    fn dedup_values(candidate: T::Value, value: T::Value) -> Option<T::Value> {
        if VERIFY {
            assert_eq!(T::key_from_value(&candidate), T::key_from_value(&value));
        }

        if T::USAGE == Usage::SecondaryIndex {
            // Secondary index optimization: cancel matching put/remove pairs.
            // NB: while this prevents redundant tombstones from getting to disk, we
            // still spend some extra CPU work to sort the entries in memory. Ideally,
            // we annihilate tombstones immediately, before sorting, but that's tricky
            // to do with scopes.
            assert_ne!(T::tombstone(&candidate), T::tombstone(&value));
            // Effect: consume both and produce nothing for this key.
            return None;
        }

        // The last value in a run of duplicates needs to be the one that ends up
        // in target.
        Some(value)
    }

    /// # Panics
    /// Panics if `value_count_limit > T::VALUE_COUNT_MAX` (upstream asserts).
    ///
    /// DEVIATION: upstream takes an allocator and stores `radix_buffer: *ScratchMemory`
    /// (shared between tables via a raw pointer); the scratch buffer is instead passed per
    /// operation ([`Self::sort`], [`Self::sort_suffix`], [`Self::absorb`]), which Rust's
    /// aliasing rules require.
    #[must_use]
    pub fn new(mutability: Mutability, name: &'static str, value_count_limit: u32) -> Self {
        assert!((value_count_limit as usize) <= T::VALUE_COUNT_MAX);

        let mutability = match mutability {
            Mutability::Mutable => Mutability::Mutable,
            Mutability::Immutable(_) => Mutability::Immutable(ImmutableState::DEFAULT),
        };

        Self {
            // TODO(port): upstream allocates value_count_max (not value_count_limit) to ensure
            // that memory table coalescing stays deterministic if the batch limit changes;
            // VALUE_COUNT_MAX is used for the same reason.
            values: vec![T::Value::default(); T::VALUE_COUNT_MAX].into_boxed_slice(),
            value_context: ValueContext { count: 0, run_tracker: SortedRunTracker::new() },
            mutability,
            name,
        }
    }

    #[must_use]
    pub const fn count(&self) -> u32 {
        self.value_context.count
    }

    #[must_use]
    pub fn values_used(&self) -> &[T::Value] {
        &self.values[..self.count() as usize]
    }

    fn values_used_mut(&mut self) -> &mut [T::Value] {
        let count = self.count() as usize;
        &mut self.values[..count]
    }

    /// Appends a `value`. If it is strictly greater than the previous key,
    /// expand the last run by 1; otherwise the suffix will be sorted later.
    ///
    /// # Panics
    /// Panics if the table is immutable or already full (upstream asserts).
    pub fn put(&mut self, value: &T::Value) {
        assert!(matches!(self.mutability, Mutability::Mutable));
        assert!(self.count() < self.values.len() as u32);

        let run_count = self.value_context.run_tracker.count();
        if run_count > 0
            && self.value_context.run_tracker.runs[run_count - 1].index_max == self.count()
        {
            let expand = self.count() == 0
                || T::key_from_value(&self.values[self.count() as usize - 1])
                    < T::key_from_value(value);
            if expand {
                self.value_context.run_tracker.runs[run_count - 1].index_max += 1;
            }
        }

        self.values[self.count() as usize] = *value;
        self.value_context.count += 1;
    }

    /// May return a tombstone.
    #[must_use]
    /// # Panics
    /// Panics if the table has more runs than [`SORTED_RUNS_MAX`] (upstream asserts).
    pub fn get(&self, key: T::Key) -> Option<&T::Value> {
        assert!(self.count() <= self.values.len() as u32);

        let run_count = self.value_context.run_tracker.count();
        assert!(run_count <= SORTED_RUNS_MAX);

        if run_count == 0 {
            return None;
        }

        // Iterate runs backwards i.e. newest first so the most recent version of a key wins.
        for run_info in self.value_context.run_tracker.runs[..run_count].iter().rev() {
            let run_sorted =
                &self.values_used()[run_info.index_min as usize..run_info.index_max as usize];

            // Skip binary search if the key is not in the range.
            if key < T::key_from_value(&run_sorted[0]) {
                continue;
            }
            if key > T::key_from_value(&run_sorted[run_sorted.len() - 1]) {
                continue;
            }

            if let Some(value) = binary_search_values(
                &|v| T::key_from_value(v),
                run_sorted,
                key,
                Config { mode: Mode::UpperBound },
            ) {
                return Some(value);
            }
        }
        None
    }

    fn finalize(&mut self, snapshot_min: u64) {
        assert!(matches!(self.mutability, Mutability::Immutable(_)));

        self.mutability = Mutability::Immutable(ImmutableState {
            flushed: self.count() == 0,
            absorbed: false,
            snapshot_min,
        });
    }

    #[must_use]
    /// # Panics
    /// Panics if the table is not immutable (upstream asserts).
    pub fn iterator_context(&self) -> MergeContext<'_, T> {
        assert!(matches!(self.mutability, Mutability::Immutable(_)));
        self.value_context.run_tracker.merge_context::<T>(self.values_used())
    }

    fn slice_run_for_range(values: &[T::Value], range: KeyRange<T>) -> Option<&[T::Value]> {
        let range_slice =
            binary_search_values_range(&|v| T::key_from_value(v), values, range.min, range.max);
        if range_slice.count == 0 {
            return None;
        }
        Some(&values[range_slice.start as usize..][..range_slice.count as usize])
    }

    #[must_use]
    /// # Panics
    /// Panics if the table is not immutable (upstream asserts).
    pub fn iterator_context_range(&self, range: KeyRange<T>) -> MergeContext<'_, T> {
        assert!(matches!(self.mutability, Mutability::Immutable(_)));
        let mut context = self.iterator_context();

        let mut target_index: usize = 0;
        for source_index in 0..context.streams_count as usize {
            let stream = context.streams[source_index];
            let run_min = T::key_from_value(&stream[0]);
            let run_max = T::key_from_value(&stream[stream.len() - 1]);
            if range.min <= run_max
                && range.max >= run_min
                && let Some(run_slice) = Self::slice_run_for_range(stream, range)
            {
                context.streams[target_index] = run_slice;
                target_index += 1;
            }
        }
        context.streams_count = target_index as u32;

        context
    }

    /// Merges `table_mutable` runs into `self` (the immutable table).
    /// # Panics
    /// Panics unless `self` is immutable and `table_mutable` is mutable, or if the mutable
    /// run tracker violates its invariants (upstream asserts).
    pub fn compact(&mut self, table_mutable: &mut Self, snapshot_min: u64) {
        assert!(matches!(self.mutability, Mutability::Immutable(_)));
        // maybe(absorbed)
        assert!(matches!(table_mutable.mutability, Mutability::Mutable));
        let _ = &table_mutable; // maybe(table_mutable.sorted())

        let mutable_count = table_mutable.count();
        table_mutable.value_context.run_tracker.assert_invariants(mutable_count);

        std::mem::swap(&mut table_mutable.values, &mut self.values);
        std::mem::swap(&mut table_mutable.value_context, &mut self.value_context);

        table_mutable.reset();
        self.finalize(snapshot_min);

        assert_eq!(table_mutable.count(), 0);
    }

    /// Absorbs the current immutable table into the mutable one,
    /// then re-materializes a compact immutable table.
    /// # Panics
    /// Panics unless `self` is immutable and `table_mutable` is mutable, if the combined count
    /// exceeds the backing storage, or on run-tracker invariant violations (upstream asserts).
    pub fn absorb(
        &mut self,
        table_mutable: &mut Self,
        snapshot_min: u64,
        radix_buffer: &mut ScratchMemory<T::Value>,
    ) {
        assert!(matches!(self.mutability, Mutability::Immutable(_)));
        assert!(matches!(table_mutable.mutability, Mutability::Mutable));
        let _ = (&self, &table_mutable); // maybe(sorted) on both

        let values_count_max = self.values.len();
        assert!(self.count() <= values_count_max as u32);
        assert!(table_mutable.count() <= values_count_max as u32);
        assert!(self.count() + table_mutable.count() <= values_count_max as u32);

        if table_mutable.count() == 0 {
            return;
        }

        if !self.sorted() {
            // NOTE: We could also only collapse if it is required.
            let scratch = radix_buffer.acquire(self.count() as usize);
            let target_count = sort_suffix_from_index::<T>(self.values_used_mut(), scratch, 0);

            self.value_context.count = target_count;
            self.value_context.run_tracker.reset();
            self.value_context.run_tracker.add(SortedRun {
                index_min: 0,
                index_max: self.count(),
                origin: RunOrigin::Immutable,
            });

            // scratch's borrow ends here (NLL); release the buffer for the next user.
            radix_buffer.release();
        }
        assert!(self.sorted());
        let values_combined_count = self.count() + table_mutable.count();

        // Because `table_mutable` is likely to be smaller than `self` we:
        // 1. Copy the values from `table_mutable` into `self`.
        // 2. Swap the backing arrays so that `table_mutable` has all the values.
        // 3. Add the new run (from the old immutable table) at the beginning.
        let immutable_count = self.count();
        let combined_usize = values_combined_count as usize;
        self.values[immutable_count as usize..combined_usize]
            .clone_from_slice(table_mutable.values_used());
        std::mem::swap(&mut table_mutable.values, &mut self.values);

        table_mutable.value_context.run_tracker.add_front_and_propagate_offset(SortedRun {
            index_min: 0,
            index_max: immutable_count,
            origin: RunOrigin::Immutable,
        });

        table_mutable.value_context.count = values_combined_count;
        table_mutable.value_context.run_tracker.assert_invariants(values_combined_count);

        if let Mutability::Immutable(state) = &mut self.mutability {
            state.absorbed = true;
        }
        self.compact(table_mutable, snapshot_min);

        // One fully sorted run or all keys are annihilated.
        assert_eq!(table_mutable.value_context.run_tracker.count(), 0);
    }

    /// Fully sorts the table if needed. Produces a single run `[0..count)`.
    /// # Panics
    /// Panics if the table is not mutable (upstream asserts), and on run-tracker invariant
    /// violations.
    pub fn sort(&mut self, radix_buffer: &mut ScratchMemory<T::Value>) {
        assert!(matches!(self.mutability, Mutability::Mutable));

        if !self.sorted() {
            self.mutable_sort_suffix_from_index(0, radix_buffer);
            self.value_context.run_tracker.reset();
            self.value_context.run_tracker.add(SortedRun {
                index_min: 0,
                index_max: self.count(),
                origin: RunOrigin::Mutable,
            });
        }

        self.value_context.run_tracker.assert_invariants(self.count());
    }

    /// When true, `values` is strictly ascending-ordered (no duplicates).
    fn sorted(&self) -> bool {
        // Empty table is considered sorted.
        if self.count() == 0 {
            return true;
        }

        // Only one sorted run can exist if it is sorted.
        if self.value_context.run_tracker.count() != 1 {
            return false;
        }

        let Some(last_run) = self.value_context.run_tracker.last() else {
            unreachable!("run_tracker.count() == 1");
        };
        assert_eq!(last_run.index_min, 0);
        assert!(last_run.index_max <= self.count());

        self.count() == last_run.index_max
    }

    /// Sorts only the unsorted suffix (everything after the last run).
    /// # Panics
    /// Panics if the table is not mutable (upstream asserts), and on run-tracker invariant
    /// violations.
    pub fn sort_suffix(&mut self, radix_buffer: &mut ScratchMemory<T::Value>) {
        assert!(matches!(self.mutability, Mutability::Mutable));

        if self.sorted() {
            self.value_context.run_tracker.assert_invariants(self.count());
            return;
        }

        let sort_suffix_index = match self.value_context.run_tracker.last() {
            Some(last_run) => last_run.index_max,
            None => 0,
        };

        assert!(sort_suffix_index <= self.count());

        if sort_suffix_index == self.count() {
            self.value_context.run_tracker.assert_invariants(self.count());
            return;
        }

        let run = self.mutable_sort_suffix_from_index(sort_suffix_index, radix_buffer);
        assert!(run.index_min <= run.index_max);
        assert_eq!(run.index_max, self.count());
        assert!(sort_suffix_index <= run.index_max);
        self.value_context.run_tracker.add(run);
        self.value_context.run_tracker.assert_invariants(self.count());
    }

    fn mutable_sort_suffix_from_index(
        &mut self,
        index: u32,
        radix_buffer: &mut ScratchMemory<T::Value>,
    ) -> SortedRun {
        assert!(matches!(self.mutability, Mutability::Mutable));
        assert!(
            index == 0
                || self
                    .value_context
                    .run_tracker
                    .last()
                    .is_some_and(|last_run| index == last_run.index_max)
        );
        assert!(index <= self.count());

        let scratch = radix_buffer.acquire(self.count() as usize);
        let target_count = sort_suffix_from_index::<T>(self.values_used_mut(), scratch, index);
        self.value_context.count = target_count;
        radix_buffer.release();

        SortedRun { index_min: index, index_max: target_count, origin: RunOrigin::Mutable }
    }

    pub fn reset(&mut self) {
        let mutability = match self.mutability {
            Mutability::Immutable(_) => Mutability::Immutable(ImmutableState::DEFAULT),
            Mutability::Mutable => Mutability::Mutable,
        };

        self.value_context.run_tracker.reset();
        self.value_context.count = 0;
        self.mutability = mutability;
    }

    /// The smallest key across all runs (upstream asserts immutable).
    #[must_use]
    /// # Panics
    /// Panics if the table is not immutable or has no runs (upstream asserts).
    pub fn key_min(&self) -> T::Key {
        assert!(matches!(self.mutability, Mutability::Immutable(_)));

        let run_count = self.value_context.run_tracker.count();
        assert!(run_count > 0);

        let mut table_min: T::Key = T::Key::SENTINEL_KEY;
        for run_info in self.value_context.run_tracker.runs[..run_count].iter().rev() {
            let run_min = T::key_from_value(&self.values_used()[run_info.index_min as usize]);
            table_min = if run_min < table_min { run_min } else { table_min };
        }
        table_min
    }

    /// The largest key across all runs (upstream asserts immutable).
    #[must_use]
    /// # Panics
    /// Panics if the table is not immutable or has no runs (upstream asserts).
    pub fn key_max(&self) -> T::Key {
        assert!(matches!(self.mutability, Mutability::Immutable(_)));

        let run_count = self.value_context.run_tracker.count();
        assert!(run_count > 0);

        let mut table_max: T::Key = T::Key::MIN_KEY;
        for run_info in self.value_context.run_tracker.runs[..run_count].iter().rev() {
            let run_max = T::key_from_value(&self.values_used()[run_info.index_max as usize - 1]);
            table_max = if run_max > table_max { run_max } else { table_max };
        }
        table_max
    }
}

/// Upstream: `KeyRange`.
///
/// DEVIATION: manual `Clone`/`Copy` impls — a derive would add a `T: Copy` bound, but only
/// `T::Key` needs to be `Copy` here.
#[derive(Debug)]
pub struct KeyRange<T: Table> {
    pub min: T::Key,
    pub max: T::Key,
}

impl<T: Table> Clone for KeyRange<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Table> Copy for KeyRange<T> {}

/// Returns the new length of `values`. Values are deduplicated after sorting, so the
/// returned count may be less than or equal to the original length.
fn sort_suffix_from_index<T: Table>(
    values: &mut [T::Value],
    values_scratch: &mut [T::Value],
    index: u32,
) -> u32 {
    let index_usize = index as usize;
    assert_eq!(values.len(), values_scratch.len());
    assert!(index_usize <= values.len());

    radix::sort(
        &mut values[index_usize..],
        &mut values_scratch[index_usize..],
        |value: &T::Value| T::key_from_value(value),
    );

    // Deduplicate values in streaming fashion.
    let mut dedup_sink = DedupSink::<T>::new();
    for i in index_usize..values.len() {
        let value = values[i]; // Copy before the sink's mutable borrow of the same slice.
        dedup_sink.push(&mut values[index_usize..], value);
    }
    index + dedup_sink.finish(&mut values[index_usize..])
}

/// Streamed equivalent of upstream's `deduplicate` loop over a sorted suffix.
///
/// DEVIATION: upstream stores `values_out: []Value` inside the sink; here the output slice is
/// passed per call, avoiding a self-referential struct.
struct DedupSink<T: Table> {
    target_index: u32,
    /// Holds the current candidate that may merge with the next item.
    candidate: Option<T::Value>,
}

impl<T: Table> DedupSink<T> {
    fn new() -> Self {
        Self { target_index: 0, candidate: None }
    }

    fn push(&mut self, values_out: &mut [T::Value], value: T::Value) {
        let Some(candidate) = self.candidate.take() else {
            // Starting a new run with a pending `value`.
            self.candidate = Some(value);
            return;
        };

        // If we're at the end of the source, there is no next value, so the next value
        // can't be equal.
        if T::key_from_value(&candidate) == T::key_from_value(&value) {
            self.candidate = TableMemory::<T>::dedup_values(candidate, value);
        } else {
            // New key encountered: flush previous winner, start a new run.
            values_out[self.target_index as usize] = candidate;
            self.target_index += 1;
            self.candidate = Some(value);
        }
    }

    fn finish(&mut self, values_out: &mut [T::Value]) -> u32 {
        // Flush the final pending value, if any.
        if let Some(candidate) = self.candidate.take() {
            values_out[self.target_index as usize] = candidate;
            self.target_index += 1;
        }

        // At this point, target_index is the number of deduplicated items written.
        if VERIFY && self.target_index > 0 {
            for (value, value_next) in values_out[..self.target_index as usize - 1]
                .iter()
                .zip(values_out[1..self.target_index as usize].iter())
            {
                assert!(T::key_from_value(value) < T::key_from_value(value_next));
            }
        }

        self.target_index
    }
}

/// Tournament capacity for the iterator: upstream `ceilPowerOfTwo(sorted_runs_max)` = 64.
const TOURNAMENT_NODE_COUNT_MAX: usize = SORTED_RUNS_MAX.next_power_of_two();

/// Merges the immutable table's sorted runs lazily, deduplicating equal keys.
pub struct ImmutableTableIterator<'a, T: Table> {
    direction: Direction,
    ready: Option<T::Value>,
    tournament_tree: Option<TournamentTree<T::Key, TOURNAMENT_NODE_COUNT_MAX>>,
    streams: [&'a [T::Value]; SORTED_RUNS_MAX],
    streams_count: u32,
    candidate: Option<T::Value>,
    end_key: Option<T::Key>,
    end_reached: bool,
    counters: IteratorCounters,
}

#[derive(Clone, Copy, Debug, Default)]
struct IteratorCounters {
    input: u32,   // This is the input count of the immutable table.
    dropped: u32, // Tombstones.
    out: u32,
}

impl<'a, T: Table> ImmutableTableIterator<'a, T> {
    /// DEVIATION: upstream initializes through an out-parameter and stores a self-pointer to
    /// detect moving the iterator after init (`assert_not_moved`); Rust move checking makes that
    /// bug class unrepresentable, so the pointer and the assertion are dropped.
    // DEVIATION-free signature parity: upstream passes the context by value; the slices are
    // copied into the iterator's own storage.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn new(
        merge_context: MergeContext<'a, T>,
        end_key: Option<T::Key>,
        direction: Direction,
    ) -> Self {
        let input_count: u32 = merge_context.streams[..merge_context.streams_count as usize]
            .iter()
            .map(|stream| stream.len() as u32)
            .sum();

        let mut iterator = Self {
            streams: [&[]; SORTED_RUNS_MAX],
            streams_count: merge_context.streams_count,
            tournament_tree: None,
            direction,
            ready: None,
            candidate: None,
            end_key,
            end_reached: false,
            counters: IteratorCounters { input: input_count, dropped: 0, out: 0 },
        };

        iterator.streams[..iterator.streams_count as usize]
            .copy_from_slice(&merge_context.streams[..merge_context.streams_count as usize]);

        iterator.load_tree();
        iterator
    }

    #[must_use]
    pub fn count_max(&self) -> u32 {
        self.counters.input
    }

    #[must_use]
    pub fn count_dropped(&self) -> u32 {
        self.counters.dropped
    }

    #[must_use]
    pub fn count_remaining(&self) -> u32 {
        self.counters.input - (self.counters.out + self.counters.dropped)
    }

    #[must_use]
    pub fn peek(&mut self) -> Option<T::Key> {
        // Early exit to avoid invoking the more expensive `ensure_next`.
        if let Some(value) = &self.ready {
            return Some(T::key_from_value(value));
        }
        self.ensure_next();
        self.ready.as_ref().map(|value| T::key_from_value(value))
    }

    pub fn pop(&mut self) -> Option<T::Value> {
        // Early exit to avoid invoking the more expensive `ensure_next`.
        if let Some(value) = self.ready.take() {
            self.counters.out += 1;
            return Some(value);
        }
        self.ensure_next();
        let value = self.ready.take()?;
        self.counters.out += 1;
        Some(value)
    }

    /// Advances past every value whose key precedes `probe_key` in iteration order.
    /// # Panics
    /// Panics if an internal pop returns nothing while keys remain (upstream asserts).
    pub fn probe(&mut self, probe_key: T::Key) {
        let remaining = self.count_remaining();
        for _ in 0..remaining {
            let Some(key_peek) = self.peek() else { break };
            match self.direction {
                Direction::Ascending if key_peek >= probe_key => break,
                Direction::Descending if key_peek <= probe_key => break,
                _ => {}
            }
            assert!(self.pop().is_some());
        }
    }

    fn load_tree(&mut self) {
        assert!(self.tournament_tree.is_none());

        let mut contestants = [crate::k_way_merge::Node::SENTINEL; TOURNAMENT_NODE_COUNT_MAX];
        for (id_usize, stream) in self.streams[..self.streams_count as usize].iter().enumerate() {
            if stream.is_empty() {
                continue;
            }
            let value = self.direction.slice_peek(stream);
            contestants[id_usize] =
                crate::k_way_merge::Node { key: T::key_from_value(value), id: id_usize as u32 };
        }

        self.tournament_tree =
            Some(TournamentTree::init(self.direction, &mut contestants, self.streams_count as u16));
    }

    fn pop_from_tree(&mut self) -> Option<T::Value> {
        if self.tournament_tree.is_none() {
            self.load_tree();
        }
        let Some(tree) = self.tournament_tree.as_mut() else {
            unreachable!("load_tree() installs the tree");
        };

        if tree.contestants_left() == 0 {
            return None;
        }

        // Pop the current winner's value directly from its stream.
        let win_id = tree.winner().id as usize;
        let stream = self.streams[win_id];
        let (&value, rest) = self.direction.slice_pop(stream);
        self.streams[win_id] = rest;

        // Feed the stream's next key back into the tree (or none if exhausted).
        let next_key: Option<T::Key> = if rest.is_empty() {
            None
        } else {
            Some(T::key_from_value(self.direction.slice_peek(rest)))
        };
        tree.pop_winner(next_key);

        Some(value)
    }

    fn ensure_next(&mut self) {
        if self.ready.is_some() {
            return;
        }
        if self.end_reached {
            return;
        }

        // A bounded-loop implementation similar to `probe` would
        // be preferable, but it currently performs worse, so we keep this version.
        loop {
            let Some(value_next) = self.pop_from_tree() else {
                if let Some(candidate) = self.candidate.take() {
                    self.ready = Some(candidate);
                } else {
                    let consumed = self.counters.out + self.counters.dropped;
                    assert_eq!(self.counters.input, consumed);
                }

                self.end_reached = true;
                return;
            };

            let value_key = T::key_from_value(&value_next);

            if let Some(candidate) = self.candidate {
                let candidate_key = T::key_from_value(&candidate);

                if value_key == candidate_key {
                    self.candidate = TableMemory::<T>::dedup_values(candidate, value_next);
                    if self.candidate.is_none() {
                        self.counters.dropped += 2;
                    } else {
                        self.counters.dropped += 1;
                    }
                    continue;
                }

                self.ready = Some(candidate);

                if !self.within_range(value_key) {
                    self.candidate = None;
                    self.end_reached = true;
                    return;
                }

                self.candidate = Some(value_next);
                return;
            }

            if !self.within_range(value_key) {
                self.end_reached = true;
                return;
            }

            self.candidate = Some(value_next);
        }
    }

    fn within_range(&self, key: T::Key) -> bool {
        let Some(key_end) = self.end_key else {
            return true;
        };

        match self.direction {
            Direction::Ascending => key <= key_end,
            Direction::Descending => key >= key_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Direction, ImmutableState, ImmutableTableIterator, Mutability, Table, TableMemory, Usage,
    };
    use crate::scratch_memory::ScratchMemory;

    /// Upstream: `TestHelper.TestTableType(mode)`'s `Value`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    struct TValue {
        key: u32,
        version: u32,
        tombstone: bool,
    }

    struct GeneralTable;

    impl Table for GeneralTable {
        type Key = u32;
        type Value = TValue;
        const VALUE_COUNT_MAX: usize = 16;
        const USAGE: Usage = Usage::General;
        fn key_from_value(value: &TValue) -> u32 {
            value.key
        }
        fn tombstone(value: &TValue) -> bool {
            value.tombstone
        }
    }

    struct SecondaryTable;

    impl Table for SecondaryTable {
        type Key = u32;
        type Value = TValue;
        const VALUE_COUNT_MAX: usize = 16;
        const USAGE: Usage = Usage::SecondaryIndex;
        fn key_from_value(value: &TValue) -> u32 {
            value.key
        }
        fn tombstone(value: &TValue) -> bool {
            value.tombstone
        }
    }

    fn create_table_immutable() -> TableMemory<GeneralTable> {
        TableMemory::new(Mutability::Immutable(ImmutableState::DEFAULT), "immutable", 16)
    }

    fn create_table_mutable() -> TableMemory<GeneralTable> {
        TableMemory::new(Mutability::Mutable, "mutable", 16)
    }

    fn v(key: u32, version: u32, tombstone: bool) -> TValue {
        TValue { key, version, tombstone }
    }

    /// Upstream test "table_memory: merge and absorb (last wins across streams)".
    ///
    /// DEVIATION: upstream snapshots the merged output via `stdx.Snap`; direct expectations here.
    #[test]
    fn merge_and_absorb_last_wins_across_streams() {
        let mut radix_buffer = ScratchMemory::<TValue>::new(16);

        let mut table_immutable = create_table_immutable();
        let mut table_mutable = create_table_mutable();

        table_mutable.put(&v(2, 0, false));
        table_mutable.put(&v(4, 0, false));
        table_mutable.sort(&mut radix_buffer);

        table_immutable.compact(&mut table_mutable, 0);
        assert_eq!(table_mutable.count(), 0);

        table_mutable.put(&v(2, 1, false));
        table_mutable.put(&v(5, 0, false));
        table_mutable.sort(&mut radix_buffer);

        table_immutable.absorb(&mut table_mutable, 0, &mut radix_buffer);

        assert_eq!(table_mutable.count(), 0);
        assert_eq!(table_immutable.count(), 4);
        assert_eq!(table_immutable.value_context.run_tracker.count(), 2);

        // Even though both runs are retained, reading should prefer the newest version.
        let Some(latest) = table_immutable.get(2) else {
            panic!("key 2 present");
        };
        assert_eq!(latest.version, 1);
        assert!(!latest.tombstone);

        let run_1_context = table_immutable.iterator_context();
        let mut iterator =
            ImmutableTableIterator::<GeneralTable>::new(run_1_context, None, Direction::Ascending);

        let mut merged: Vec<TValue> = Vec::new();
        while let Some(value) = iterator.pop() {
            merged.push(value);
        }

        assert_eq!(merged.len(), 3);
        assert_eq!(merged, vec![v(2, 1, false), v(4, 0, false), v(5, 0, false)]);
    }

    /// Upstream test "table_memory: compact and deduplicate across runs".
    #[test]
    fn compact_and_deduplicate_across_runs() {
        let mut radix_buffer = ScratchMemory::<TValue>::new(16);

        let mut table_immutable = create_table_immutable();
        let mut table_mutable = create_table_mutable();

        table_mutable.put(&v(2, 0, false));
        table_mutable.put(&v(2, 1, false));
        table_mutable.sort_suffix(&mut radix_buffer);

        table_mutable.put(&v(2, 2, false));
        table_mutable.put(&v(2, 3, false));
        table_mutable.sort_suffix(&mut radix_buffer);

        table_immutable.compact(&mut table_mutable, 0);
        assert_eq!(table_mutable.count(), 0);
        assert_eq!(table_immutable.count(), 2);
        assert_eq!(table_immutable.value_context.run_tracker.count(), 2);

        let Some(latest) = table_immutable.get(2) else {
            panic!("key 2 present");
        };
        assert_eq!(latest.version, 3);

        let merge_context = table_immutable.iterator_context();
        let mut iterator =
            ImmutableTableIterator::<GeneralTable>::new(merge_context, None, Direction::Ascending);

        let mut merged: Vec<TValue> = Vec::new();
        while let Some(value) = iterator.pop() {
            merged.push(value);
        }

        assert_eq!(merged.len(), 1);
        assert_eq!(merged, vec![v(2, 3, false)]);
    }

    /// Upstream test "table_memory (secondary): annihilation yields zero after deduplicate".
    #[test]
    fn secondary_annihilation_yields_zero_after_deduplicate() {
        let mut radix_buffer = ScratchMemory::<TValue>::new(16);

        let create_immutable = || {
            TableMemory::<SecondaryTable>::new(
                Mutability::Immutable(ImmutableState::DEFAULT),
                "immutable",
                16,
            )
        };
        let create_mutable =
            || TableMemory::<SecondaryTable>::new(Mutability::Mutable, "mutable", 16);

        let mut table_immutable = create_immutable();
        let mut table_mutable = create_mutable();

        table_mutable.put(&v(2, 0, false));
        table_mutable.put(&v(2, 0, true));
        table_mutable.sort_suffix(&mut radix_buffer);

        table_immutable.compact(&mut table_mutable, 0);
        assert_eq!(table_mutable.count(), 0);
        assert_eq!(table_immutable.count(), 0);
    }

    /// Exercise [`TableMemory::iterator_context_range`] end to end (upstream exercises this via
    /// groove scans; covered here directly until those exist).
    #[test]
    fn iterator_context_range_slices_streams() {
        let mut radix_buffer = ScratchMemory::<TValue>::new(16);

        let mut table_immutable = create_table_immutable();
        let mut table_mutable = create_table_mutable();

        for &(key, version) in &[(1_u32, 0_u32), (3, 0), (5, 0), (7, 0)] {
            table_mutable.put(&v(key, version, false));
        }
        table_mutable.sort(&mut radix_buffer);
        table_immutable.compact(&mut table_mutable, 0);

        let range = super::KeyRange::<GeneralTable> { min: 3, max: 5 };
        let context = table_immutable.iterator_context_range(range);
        let mut iterator =
            ImmutableTableIterator::<GeneralTable>::new(context, None, Direction::Ascending);

        let mut merged: Vec<TValue> = Vec::new();
        while let Some(value) = iterator.pop() {
            merged.push(value);
        }
        assert_eq!(merged, vec![v(3, 0, false), v(5, 0, false)]);

        // Out-of-range queries produce an empty merge context.
        let range = super::KeyRange::<GeneralTable> { min: 100, max: 200 };
        let context = table_immutable.iterator_context_range(range);
        assert_eq!(context.streams_count, 0);
        let mut iterator =
            ImmutableTableIterator::<GeneralTable>::new(context, None, Direction::Ascending);
        assert!(iterator.pop().is_none());
        assert_eq!(iterator.count_remaining(), 0);
    }

    /// `probe` must consume exactly the keys preceding the probe key.
    #[test]
    fn probe_advances_past_preceding_keys() {
        let mut radix_buffer = ScratchMemory::<TValue>::new(16);

        let mut table_immutable = create_table_immutable();
        let mut table_mutable = create_table_mutable();

        for key in [2_u32, 4, 6, 8] {
            table_mutable.put(&v(key, 0, false));
        }
        table_mutable.sort(&mut radix_buffer);
        table_immutable.compact(&mut table_mutable, 0);

        let context = table_immutable.iterator_context();
        let mut iterator =
            ImmutableTableIterator::<GeneralTable>::new(context, None, Direction::Ascending);

        iterator.probe(6);
        assert_eq!(iterator.pop().map(|value| value.key), Some(6));
        assert_eq!(iterator.pop().map(|value| value.key), Some(8));
        assert_eq!(iterator.pop().map(|value| value.key), None);
    }
}
