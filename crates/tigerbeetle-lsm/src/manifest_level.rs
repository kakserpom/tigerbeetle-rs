//! A single level of an LSM tree's manifest: the set of tables (and their key bounds) that
//! live on one compaction level, ordered by ascending `(key_max, snapshot_min)`.
//!
//! Upstream: `src/lsm/manifest_level.zig`.
//!
//! DEVIATION: upstream stores direct pointers into segmented-array memory
//! (`TableInfoReference.table_info: *TableInfo`), guarded against staleness by a generation
//! counter. Safe Rust cannot hold interior mutable pointers into another structure, so the
//! reference carries a *copy* of the table info; mutating operations
//! ([`ManifestLevel::set_snapshot_max`]) re-locate the live entry at its sort-key position
//! and update it in place via identity (address + checksum). The generation counter
//! semantics are kept.
//!
//! DEVIATION: upstream's `LevelKeyRange` methods use `@fieldParentPtr` to reach the owning
//! level; here they are plain [`ManifestLevel`] methods (`key_range_latest_*`).
//!
//! DEVIATION: upstream allocates both parallel segmented arrays from one shared node pool;
//! this port's arrays own their node buffers (see the `segmented_array.zig` port notes).

use core::fmt::Debug;
use core::marker::PhantomData;

use tigerbeetle_core::constants::LSM_GROWTH_FACTOR;
use tigerbeetle_core::stdx::bounded_array::BoundedArray;

use crate::direction::Direction::{self, Ascending, Descending};
use crate::segmented_array::{Cursor, SegmentedArray, SegmentedArrayIterator, SegmentedArraySpec};

/// Upstream: `lsm/tree.zig snapshot_latest`.
pub use crate::tree::SNAPSHOT_LATEST;

/// [`LSM_GROWTH_FACTOR`] in the `usize` domain (the constant is small by definition).
#[allow(clippy::cast_possible_truncation)]
const LSM_GROWTH_FACTOR_USIZE: usize = LSM_GROWTH_FACTOR as usize;

/// The per-table metadata a manifest level operates on (upstream: the `TableInfo`
/// comptime parameter, i.e. `manifest.zig`'s `TreeTableInfo`).
pub trait LevelTableInfo: Copy + Debug + Default {
    type Key: Ord + Copy + Debug;

    fn checksum(&self) -> u128;
    fn address(&self) -> u64;
    fn snapshot_min(&self) -> u64;
    fn snapshot_max(&self) -> u64;
    /// Upstream only ever lowers this field (from `u64::MAX` to a tombstoning bound).
    fn set_snapshot_max(&mut self, snapshot_max: u64);
    fn key_min(&self) -> Self::Key;
    fn key_max(&self) -> Self::Key;
    fn value_count(&self) -> u32;

    /// Whether `snapshot` may observe this table.
    ///
    /// # Panics
    /// Panics if `snapshot > SNAPSHOT_LATEST`, or on inconsistent metadata (upstream asserts).
    fn visible(&self, snapshot: u64) -> bool {
        assert!(self.address() != 0);
        assert!(self.snapshot_min() <= self.snapshot_max());
        assert!(snapshot <= SNAPSHOT_LATEST);

        self.snapshot_min() <= snapshot && snapshot <= self.snapshot_max()
    }

    /// Whether no retained snapshot observes this table (i.e., it is garbage).
    ///
    /// # Panics
    /// Panics if any snapshot is out of range, or if the table is still visible at
    /// `SNAPSHOT_LATEST` yet has a bounded `snapshot_max` (upstream asserts).
    fn invisible(&self, snapshots: &[u64]) -> bool {
        // Return early and do not iterate all snapshots if the table was never deleted:
        if self.visible(SNAPSHOT_LATEST) {
            return false;
        }
        for &snapshot in snapshots {
            assert!(snapshot <= SNAPSHOT_LATEST);
            if self.visible(snapshot) {
                return false;
            }
        }
        assert!(self.snapshot_max() < u64::MAX);
        true
    }

    /// Exact equality, including `snapshot_max` (upstream `TreeTableInfo.equal`).
    fn equal(&self, other: &Self) -> bool
    where
        Self: PartialEq,
    {
        self.checksum() == other.checksum()
            && self.address() == other.address()
            && self.snapshot_min() == other.snapshot_min()
            && self.snapshot_max() == other.snapshot_max()
            && self.key_min() == other.key_min()
            && self.key_max() == other.key_max()
            && self.value_count() == other.value_count()
    }
}

/// Static description of a manifest level instantiation (upstream comptime parameters of
/// `ManifestLevelType`, plus the derived node capacities of its two arrays).
pub trait ManifestLevelSpec: 'static {
    type Key: Ord + Copy + Debug + Default + 'static;

    type TableInfo: LevelTableInfo<Key = Self::Key> + PartialEq + 'static;

    /// Upstream `table_count_max_tree`.
    const TABLE_COUNT_MAX_TREE: u32;

    const KEYS_NODE_CAPACITY: usize;
    const TABLES_NODE_CAPACITY: usize;

    /// Sort key of the Tables array: upstream packs `(key_max, snapshot_min)` into a wide
    /// unsigned integer via `KeyMaxSnapshotMin`.
    type TableSortKey: Ord + Copy + Debug + Default + 'static;
    const TABLE_SORT_KEY_MIN: Self::TableSortKey;
    const TABLE_SORT_KEY_MAX: Self::TableSortKey;
    fn table_sort_key(table_info: &Self::TableInfo) -> Self::TableSortKey;

    const KEY_MIN: Self::Key;
    const KEY_MAX: Self::Key;
}

/// Helper mirroring upstream's packed `(key_max, snapshot_min)` ordering value
/// (`KeyMaxSnapshotMin.Int`): the packed integer orders primarily by `key_max`,
/// secondarily by `snapshot_min`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyMaxSnapshotMin<K> {
    pub snapshot_min: u64,
    pub key_max: K,
}

impl KeyMaxSnapshotMin<u64> {
    /// Packs into the 128-bit ordering integer (upstream `Int.from`; upstream uses a
    /// bit-width-generic Zig integer, this port fixes the u64-key layout).
    #[must_use]
    pub fn to_int(self) -> u128 {
        (u128::from(self.key_max) << 64) | u128::from(self.snapshot_min)
    }

    /// Unpacks from the 128-bit ordering integer.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // high half discarded by construction
    pub fn from_int(value: u128) -> Self {
        Self { snapshot_min: value as u64, key_max: (value >> 64) as u64 }
    }
}

/// A reference to a table within the level, valid while `generation` is unchanged.
///
/// DEVIATION: holds a copy rather than a pointer (see module notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableInfoReference<T> {
    pub table_info: T,
    pub generation: u32,
}

/// Inclusive key range (upstream `ManifestLevel.KeyRange`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyRange<K> {
    /// Inclusive.
    pub key_min: K,
    /// Inclusive.
    pub key_max: K,
}

/// Tables in level B that intersect with a chosen table in level A (upstream `OverlapRange`).
#[derive(Clone, Debug)]
pub struct OverlapRange<T, K> {
    /// The minimum key across both levels.
    pub key_min: K,
    /// The maximum key across both levels.
    pub key_max: K,
    /// References to tables in level B that intersect with the chosen table in level A.
    pub tables: BoundedArray<TableInfoReference<T>, { LSM_GROWTH_FACTOR_USIZE }>,
}

/// Result of [`ManifestLevel::table_with_least_overlap`].
#[derive(Clone, Debug)]
pub struct LeastOverlapTable<T, K> {
    pub table: TableInfoReference<T>,
    pub range: OverlapRange<T, K>,
}

/// Which tables an iterator yields relative to the given snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Visible,
    Invisible,
}

/// Wrapper spec for the parallel Keys array: values are keys themselves.
struct KeysSpec<S: ManifestLevelSpec>(PhantomData<S>);

impl<S: ManifestLevelSpec> SegmentedArraySpec for KeysSpec<S> {
    type Value = S::Key;
    type Key = S::Key;
    const SORTED: bool = true;
    const ELEMENT_COUNT_MAX: u32 = S::TABLE_COUNT_MAX_TREE;
    const NODE_CAPACITY: usize = S::KEYS_NODE_CAPACITY;
    const KEY_MIN: S::Key = S::KEY_MIN;
    const KEY_MAX: S::Key = S::KEY_MAX;
    fn key_from_value(value: &Self::Value) -> Self::Key {
        *value
    }
}

/// Wrapper spec for the parallel Tables array, ordered by `(key_max, snapshot_min)`.
struct TablesSpec<S: ManifestLevelSpec>(PhantomData<S>);

impl<S: ManifestLevelSpec> SegmentedArraySpec for TablesSpec<S> {
    type Value = S::TableInfo;
    type Key = S::TableSortKey;
    const SORTED: bool = true;
    const ELEMENT_COUNT_MAX: u32 = S::TABLE_COUNT_MAX_TREE;
    const NODE_CAPACITY: usize = S::TABLES_NODE_CAPACITY;
    const KEY_MIN: S::TableSortKey = S::TABLE_SORT_KEY_MIN;
    const KEY_MAX: S::TableSortKey = S::TABLE_SORT_KEY_MAX;
    fn key_from_value(value: &Self::Value) -> Self::Key {
        S::table_sort_key(value)
    }
}

pub struct ManifestLevel<S: ManifestLevelSpec> {
    // These two segmented arrays are parallel. That is, the absolute indexes of maximum key
    // and corresponding TableInfo are the same. However, the number of nodes, node index, and
    // relative index into the node differ as the elements per node are different.
    //
    // Ordered by ascending (maximum) key. Keys may repeat due to snapshots.
    keys: SegmentedArray<KeysSpec<S>>,
    tables: SegmentedArray<TablesSpec<S>>,

    /// The range of keys in this level covered by tables visible to `SNAPSHOT_LATEST`
    /// (upstream `key_range_latest: LevelKeyRange`).
    key_range_latest: Option<KeyRange<S::Key>>,

    /// The number of tables visible to `SNAPSHOT_LATEST`. Used to enforce
    /// `table_count_max_tree_for_level()` (upstream TODO: track this in Manifest instead).
    table_count_visible: u32,

    /// The number of values, across all tables visible to `SNAPSHOT_LATEST`.
    value_count_visible: u64,

    /// A monotonically increasing generation number used to detect invalid internal
    /// TableInfo references.
    generation: u32,

    spec: PhantomData<S>,
}

impl<S: ManifestLevelSpec> ManifestLevel<S> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            keys: SegmentedArray::new(),
            tables: SegmentedArray::new(),
            key_range_latest: None,
            table_count_visible: 0,
            value_count_visible: 0,
            generation: 0,
            spec: PhantomData,
        }
    }

    /// Empties the level; the generation advances so outstanding references are invalidated
    /// (upstream `reset`).
    pub fn reset(&mut self) {
        self.keys.reset();
        self.tables.reset();

        self.key_range_latest = None;
        self.table_count_visible = 0;
        self.value_count_visible = 0;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Returns the key range of tables visible to `SNAPSHOT_LATEST` (upstream
    /// `key_range_latest.key_range`).
    pub fn key_range_latest(&self) -> Option<KeyRange<S::Key>> {
        self.key_range_latest
    }

    /// The number of tables visible to `SNAPSHOT_LATEST`.
    pub fn table_count_visible(&self) -> u32 {
        self.table_count_visible
    }

    /// The total number of values across all tables visible to `SNAPSHOT_LATEST`.
    pub fn value_count_visible(&self) -> u64 {
        self.value_count_visible
    }

    /// Inserts the given table into the level.
    ///
    /// # Panics
    /// Panics on invariant violation (upstream asserts under `constants.verify`, which is
    /// always enabled in this port), including inserting a duplicate table.
    pub fn insert_table(&mut self, table: &S::TableInfo) {
        assert!(!self.contains(table));
        assert_eq!(self.keys.len(), self.tables.len());

        let absolute_index_keys = self.keys.insert_element(table.key_max());
        assert!(absolute_index_keys < self.keys.len());

        let absolute_index_tables = self.tables.insert_element(*table);
        assert!(absolute_index_tables < self.tables.len());

        if table.visible(SNAPSHOT_LATEST) {
            self.table_count_visible += 1;
            self.value_count_visible += u64::from(table.value_count());
        }
        self.generation = self.generation.wrapping_add(1);

        self.key_range_latest_include(KeyRange {
            key_min: table.key_min(),
            key_max: table.key_max(),
        });

        assert!(self.contains(table));

        // `keys` may have duplicate entries due to tables with the same key_max, but
        // different snapshots (upstream: maybe(absolute_index_keys != absolute_index_tables)).
        //
        // Verify that both parallel arrays received the element at the same absolute index.
        // DEVIATION: upstream crosses the two indexes here (keys iterator seeded with
        // `absolute_index_tables` and vice versa), which looks like an upstream typo; we
        // check each array against its own insertion index.
        assert_eq!(
            self.keys.element_at_cursor(self.keys.cursor_for_absolute_index(absolute_index_keys)),
            table.key_max()
        );
        assert_eq!(
            self.tables
                .element_at_cursor(self.tables.cursor_for_absolute_index(absolute_index_tables)),
            *table
        );

        assert_eq!(self.keys.len(), self.tables.len());
    }

    /// Set snapshot_max for the given table in the level.
    ///
    /// * Asserts that the reference is not stale and the table exists in the level.
    /// * Asserts that the table currently has snapshot_max of `u64::MAX`.
    ///
    /// # Panics
    /// Panics on stale references or invariant violations (upstream asserts).
    pub fn set_snapshot_max(&mut self, snapshot: u64, table_ref: TableInfoReference<S::TableInfo>) {
        assert_eq!(table_ref.generation, self.generation);
        assert!(self.contains(&table_ref.table_info));
        assert!(snapshot < SNAPSHOT_LATEST);
        assert_eq!(table_ref.table_info.snapshot_max(), u64::MAX);

        let table = table_ref.table_info;
        assert!(table.key_min() <= table.key_max());

        // Locate the live row and update it in place. Upstream mutates through the stored
        // pointer directly; this port re-finds the entry (see module notes).
        let Some((index, _)) = self.find_live(&table) else {
            panic!("set_snapshot_max: table not found at its sort-key position");
        };
        let element = self.tables.element_mut(index);
        assert_eq!(element.snapshot_max(), u64::MAX);
        element.set_snapshot_max(snapshot);

        self.table_count_visible -= 1;
        self.value_count_visible -= u64::from(table.value_count());
        self.key_range_latest_exclude(KeyRange {
            key_min: table.key_min(),
            key_max: table.key_max(),
        });
    }

    /// Removes the given table.
    ///
    /// # Panics
    /// Panics if the table is not present (upstream panics likewise).
    pub fn remove_table(&mut self, table: &S::TableInfo) {
        assert_eq!(self.keys.len(), self.tables.len());
        assert!(table.key_min() <= table.key_max());

        // Use `key_min` for both ends of the iterator; we are looking for a single table.
        let Some(cursor_start) = self.iterator_start(table.key_min(), table.key_min(), Ascending)
        else {
            panic!("remove_table: level is empty");
        };

        let mut i = self.keys.absolute_index_for_cursor(cursor_start);
        let mut tables = self.tables.iterator_from_index(i, Ascending);
        let table_index_absolute = loop {
            let Some(level_table) = tables.next() else {
                panic!("ManifestLevel.remove_table: table not found");
            };

            // DEVIATION: upstream asserts the search argument is not aliased into array
            // memory; copies cannot alias.

            if level_table.equal(table) {
                break i;
            }
            assert_ne!(level_table.checksum(), table.checksum());
            assert_ne!(level_table.address(), table.address());

            i += 1;
        };

        self.generation = self.generation.wrapping_add(1);
        self.keys.remove_elements(table_index_absolute, 1);
        self.tables.remove_elements(table_index_absolute, 1);
        assert_eq!(self.keys.len(), self.tables.len());

        if table.visible(SNAPSHOT_LATEST) {
            self.table_count_visible -= 1;
            self.value_count_visible -= u64::from(table.value_count());

            self.key_range_latest_exclude(KeyRange {
                key_min: table.key_min(),
                key_max: table.key_max(),
            });
        }
    }

    /// Returns true if the given key may be present in the level, false if the key is
    /// guaranteed to not be present.
    ///
    /// Our key range keeps track of tables that are visible to `SNAPSHOT_LATEST`, so it cannot
    /// be relied upon for queries to older snapshots.
    ///
    /// # Panics
    /// Panics if `snapshot >= SNAPSHOT_LATEST` (upstream asserts; persistent snapshots will
    /// lift this).
    #[must_use]
    pub fn key_range_contains(&self, snapshot: u64, key: S::Key) -> bool {
        assert!(snapshot < SNAPSHOT_LATEST);
        self.key_range_latest_contains(key)
    }

    // LevelKeyRange.include/exclude/contains, re-homed onto the level (upstream reaches the
    // parent through @fieldParentPtr).

    fn key_range_latest_include(&mut self, include_range: KeyRange<S::Key>) {
        match &mut self.key_range_latest {
            Some(level_range) => {
                if include_range.key_min < level_range.key_min {
                    level_range.key_min = include_range.key_min;
                }
                if include_range.key_max > level_range.key_max {
                    level_range.key_max = include_range.key_max;
                }
            }
            None => self.key_range_latest = Some(include_range),
        }
        let Some(range) = self.key_range_latest else {
            unreachable!("assigned above");
        };
        assert!(range.key_min <= range.key_max);
        assert!(range.key_min <= include_range.key_min && include_range.key_max <= range.key_max);
    }

    /// Excludes the specified range from the level's key range, i.e. if the specified range
    /// contributes to the level's key_min/key_max, finds a new key_min/key_max.
    ///
    /// This is achieved by querying the tables visible to `SNAPSHOT_LATEST` and updating the
    /// level key_min/key_max to the key_min/key_max of the first table returned by the
    /// iterator. The query is guaranteed to only fetch non-snapshotted tables, since tables
    /// visible to old snapshots that users have retained would have snapshot_max set to less
    /// than `u64::MAX`, and therefore wouldn't be visible to queries with `SNAPSHOT_LATEST`.
    ///
    /// # Panics
    /// Panics if the level's key range is unset (upstream asserts).
    fn key_range_latest_exclude(&mut self, exclude_range: KeyRange<S::Key>) {
        assert!(self.key_range_latest.is_some());
        if self.table_count_visible == 0 {
            self.key_range_latest = None;
            return;
        }

        let snapshots = [SNAPSHOT_LATEST];
        let Some(mut current) = self.key_range_latest else {
            unreachable!("checked above");
        };
        if exclude_range.key_max == current.key_max {
            let mut itr = self.iterator(Visibility::Visible, &snapshots, Descending, None);
            let Some(table) = itr.next() else {
                panic!("at least one visible table remains");
            };
            current.key_max = table.key_max();
        }
        if exclude_range.key_min == current.key_min {
            let mut itr = self.iterator(Visibility::Visible, &snapshots, Ascending, None);
            let Some(table) = itr.next() else {
                panic!("at least one visible table remains");
            };
            current.key_min = table.key_min();
        }
        self.key_range_latest = Some(current);
        assert!(current.key_min <= current.key_max);
    }

    pub fn key_range_latest_contains(&self, key: S::Key) -> bool {
        match self.key_range_latest {
            Some(range) => range.key_min <= key && key <= range.key_max,
            None => false,
        }
    }

    /// Iterates tables matching `visibility` and `snapshots`, optionally restricted to
    /// `key_range`, in `direction` (upstream `iterator`).
    ///
    /// # Panics
    /// Panics if any snapshot exceeds `SNAPSHOT_LATEST` or the range is inverted (upstream
    /// asserts).
    #[must_use]
    pub fn iterator<'a>(
        &'a self,
        visibility: Visibility,
        snapshots: &'a [u64],
        direction: Direction,
        key_range: Option<KeyRange<S::Key>>,
    ) -> ManifestLevelIterator<'a, S> {
        for &snapshot in snapshots {
            assert!(snapshot <= SNAPSHOT_LATEST);
        }

        let inner = if let Some(range) = key_range {
            assert!(range.key_min <= range.key_max);

            if let Some(start) = self.iterator_start(range.key_min, range.key_max, direction) {
                // Translate the keys-array cursor through absolute indexes, as the two
                // parallel arrays pack different element sizes per node.
                let absolute_index = self.keys.absolute_index_for_cursor(start);
                self.tables.iterator_from_index(absolute_index, direction)
            } else {
                // Nothing to iterate because we know for sure that the key range is disjoint
                // with the tables stored in this level (upstream sets `it.done = true`).
                self.tables.exhausted_iterator(direction)
            }
        } else {
            match direction {
                Ascending => self.tables.iterator_from_index(0, Ascending),
                Descending => self.tables.iterator_from_cursor(self.tables.last(), Descending),
            }
        };

        ManifestLevelIterator { inner, visibility, snapshots, direction, key_range }
    }

    /// Returns the keys segmented array cursor at which iteration should be started.
    /// May return null if there is nothing to iterate because we know for sure that the key
    /// range is disjoint with the tables stored in this level.
    ///
    /// However, the cursor returned is not guaranteed to be in range for the query as only
    /// the key_max is stored in the index structures, not the key_min, and only the start
    /// bound for the given direction is checked here.
    fn iterator_start(
        &self,
        key_min: S::Key,
        key_max: S::Key,
        direction: Direction,
    ) -> Option<Cursor> {
        assert!(key_min <= key_max);
        assert_eq!(self.keys.len(), self.tables.len());

        if self.keys.is_empty() {
            return None;
        }

        // Ascending:  Find the first table where table.key_max >= iterator.key_min.
        // Descending: Find the first table where table.key_max >= iterator.key_max.
        let target = self.keys.search(match direction {
            Ascending => key_min,
            Descending => key_max,
        });
        assert!(target.node <= self.keys.node_count());

        if self.keys.absolute_index_for_cursor(target) == self.keys.len() {
            match direction {
                // The key_min of the target range is greater than the key_max of the last
                // table in the level and we are ascending, so this range matches no tables
                // on this level.
                Ascending => None,
                // The key_max of the target range is greater than the key_max of the last
                // table in the level and we are descending, so we need to start iteration
                // at the last table in the level.
                Descending => Some(self.keys.last()),
            }
        } else {
            // Multiple tables in the level may share a key.
            // Scan to the edge so that the iterator will cover them all.
            Some(self.iterator_start_boundary(target, direction))
        }
    }

    /// This function exists because there may be tables in the level with the same
    /// key_max but non-overlapping snapshot visibility.
    ///
    /// Put differently, there may be several tables with different snapshots but the same
    /// `key_max`, and `iterator_start`'s binary search may have landed in the middle of them.
    fn iterator_start_boundary(&self, key_cursor: Cursor, direction: Direction) -> Cursor {
        let mut reverse = self.keys.iterator_from_cursor(key_cursor, direction.reverse());
        assert_eq!(reverse.cursor(), key_cursor);

        // This cursor will always point to a key equal to start_key.
        let mut adjusted = reverse.cursor();
        let Some(start_key) = reverse.next().copied() else {
            unreachable!("non-empty by caller");
        };
        assert_eq!(start_key, self.keys.element_at_cursor(adjusted));

        let mut adjusted_next = reverse.cursor();
        let mut ran_to_end = true;
        while let Some(k) = reverse.next() {
            if start_key != *k {
                ran_to_end = false;
                break;
            }
            adjusted = adjusted_next;
            adjusted_next = reverse.cursor();
        }
        if ran_to_end {
            match direction {
                Ascending => assert_eq!(adjusted, self.keys.first()),
                Descending => assert_eq!(adjusted, self.keys.last()),
            }
        }
        assert_eq!(start_key, self.keys.element_at_cursor(adjusted));

        adjusted
    }

    /// Locates the live row of `table` at its sort-key position, matched by identity
    /// (address + checksum). Returns its absolute index and a copy. Internal helper replacing
    /// upstream's raw-pointer access.
    fn find_live(&self, table: &S::TableInfo) -> Option<(u32, S::TableInfo)> {
        let table_sort_key = S::table_sort_key(table);
        let table_cursor = self.tables.search(table_sort_key);
        let index = self.tables.absolute_index_for_cursor(table_cursor);
        if index == self.tables.len() {
            return None;
        }
        let level_table = self.tables.element_at_cursor(table_cursor);
        if level_table.address() == table.address() && level_table.checksum() == table.checksum() {
            // Upstream: maybe(level_table.snapshot_max != table.snapshot_max).
            Some((index, level_table))
        } else {
            None
        }
    }

    /// Returns a table which matches the given table *except possibly the snapshot_max*.
    #[must_use]
    pub fn find(&self, table: &S::TableInfo) -> Option<TableInfoReference<S::TableInfo>> {
        let (_, found) = self.find_live(table)?;
        Some(TableInfoReference { table_info: found, generation: self.generation })
    }

    /// Returns whether the level contains the *exact* table (including snapshot_max).
    ///
    /// # Panics
    /// Panics if an inconsistent table is found (upstream asserts under `constants.verify`,
    /// which is always enabled in this port).
    #[must_use]
    pub fn contains(&self, table: &S::TableInfo) -> bool {
        // Upstream asserts constants.verify here ("Currently only used for testing");
        // verification is always enabled in this port.
        let Some(found) = self.find(table) else {
            return false;
        };
        let table_exact = table.snapshot_max() == found.table_info.snapshot_max();
        assert_eq!(table_exact, table.equal(&found.table_info));
        table_exact
    }

    /// Given two levels (where A is the level on which this function is invoked and B is the
    /// other level), finds a table in Level A that overlaps with the least number of tables
    /// in Level B.
    ///
    /// Uses a two-pointer sweep over both levels for O(n_a + n_b) overlap counting, then a
    /// single binary search to retrieve the full overlap range for the chosen table.
    ///
    /// * Exits early if it finds a table that doesn't overlap with any tables in the second
    ///   level.
    /// * Ties are resolved by smaller key and then smaller snapshot.
    ///
    /// # Panics
    /// Panics if level A has no visible tables or `max_overlapping_tables` exceeds the growth
    /// factor (upstream asserts).
    #[must_use]
    pub fn table_with_least_overlap(
        &self,
        level_b: &Self,
        snapshot: u64,
        max_overlapping_tables: usize,
    ) -> LeastOverlapTable<S::TableInfo, S::Key> {
        assert!(max_overlapping_tables <= LSM_GROWTH_FACTOR_USIZE);
        assert!(self.table_count_visible > 0);

        let snapshots = [snapshot];

        // Two-pointer sweep to count overlaps in O(n_a + n_b).
        //
        // Both levels are sorted. As we advance through A left-to-right, the B pointers only
        // move forward (monotonic), so total work across all A-tables is O(n_b) per pointer.
        //
        // For each a_table, we maintain two counts into B:
        //
        //   b_lower_count: B-tables with key_max  < a_table.key_min  (left of A)
        //   b_upper_count: B-tables with key_min <= a_table.key_max  (started before A ends)
        //
        //   overlap = b_upper_count - b_lower_count
        //
        // Example: 4 B-tables, a_table overlaps B2 and B3:
        //        Level A:               |= a_table ===|
        //        Level B:  |=B0=| |=B1=| |==B2==| |==B3==|  |=B4=|
        //                  ├───────────┤              :
        //                  b_lower_count = 2          :
        //                  (B0, B1 end before         :
        //                   a_table starts)           :
        //                  ├──────────────────────────┤
        //                  b_upper_count = 4
        //                  (B0, B1, B2, B3 start before a_table ends)
        //                  overlap = 4 - 2 = 2  (B2, B3)

        let mut a_iterator = self.iterator(Visibility::Visible, &snapshots, Ascending, None);

        let mut b_lower_iterator =
            level_b.iterator(Visibility::Visible, &snapshots, Ascending, None);
        let mut b_lower_count: usize = 0;
        let mut b_lower: Option<&S::TableInfo> = b_lower_iterator.next();
        let mut b_upper_iterator =
            level_b.iterator(Visibility::Visible, &snapshots, Ascending, None);
        let mut b_upper_count: usize = 0;
        let mut b_upper: Option<&S::TableInfo> = b_upper_iterator.next();

        let optimal: (&S::TableInfo, usize) = 'optimal: {
            let mut optimal_table: Option<&S::TableInfo> = None;
            let mut optimal_overlap: usize = max_overlapping_tables + 1;
            let mut iterations: usize = 0;

            while let Some(a_table) = a_iterator.next() {
                iterations += 1;

                while let Some(lower) = b_lower {
                    if lower.key_max() >= a_table.key_min() {
                        break;
                    }
                    b_lower_count += 1; // TODO(upstream): move this to continuation.
                    b_lower = b_lower_iterator.next();
                }
                if let Some(lower) = b_lower {
                    assert!(a_table.key_min() <= lower.key_max());
                }

                while let Some(upper) = b_upper {
                    if upper.key_min() > a_table.key_max() {
                        break;
                    }
                    b_upper_count += 1; // TODO(upstream): move this to continuation.
                    b_upper = b_upper_iterator.next();
                }
                if let Some(upper) = b_upper {
                    assert!(a_table.key_max() < upper.key_min());
                }

                let overlap = b_upper_count - b_lower_count;

                if overlap > max_overlapping_tables {
                    continue;
                }

                // Zero overlap is already optimal.
                if optimal_overlap == 0 {
                    break;
                }

                if overlap < optimal_overlap {
                    optimal_table = Some(a_table);
                    optimal_overlap = overlap;
                }
            }
            assert!(iterations > 0);
            assert!(iterations == self.table_count_visible as usize || optimal_overlap == 0);
            assert!(optimal_overlap <= max_overlapping_tables);

            let Some(optimal_table) = optimal_table else {
                unreachable!("the overlap bound guarantees at least one candidate");
            };
            break 'optimal (optimal_table, optimal_overlap);
        };

        // Retrieve the full OverlapRange for the chosen table.
        let Some(range) = level_b.tables_overlapping_with_key_range(
            optimal.0.key_min(),
            optimal.0.key_max(),
            snapshot,
            max_overlapping_tables,
        ) else {
            panic!("the chosen table's overlap fits the budget");
        };
        assert_eq!(range.tables.count(), optimal.1);
        assert!(range.tables.count() <= max_overlapping_tables);

        LeastOverlapTable {
            table: TableInfoReference { table_info: *optimal.0, generation: self.generation },
            range,
        }
    }

    /// Returns the next table in the range, after `key_exclusive` if provided.
    ///
    /// * The table returned is visible to `snapshot`.
    ///
    /// # Panics
    /// Panics if any bound is inverted or a yielded table violates invariants (upstream
    /// asserts).
    #[must_use]
    pub fn next_table(
        &self,
        parameters: NextTableParameters<S::Key>,
    ) -> Option<TableInfoReference<S::TableInfo>> {
        let NextTableParameters { snapshot, key_min, key_max, key_exclusive, direction } =
            parameters;
        let snapshots = [snapshot];

        assert!(key_min <= key_max);

        let Some(key_exclusive) = key_exclusive else {
            let mut it = self.iterator(
                Visibility::Visible,
                &snapshots,
                direction,
                Some(KeyRange { key_min, key_max }),
            );
            return it.next().map(|table_info| TableInfoReference {
                table_info: *table_info,
                generation: self.generation,
            });
        };

        assert!(key_min <= key_exclusive);
        assert!(key_exclusive <= key_max);

        let key_min_exclusive = if direction == Ascending { key_exclusive } else { key_min };
        let key_max_exclusive = if direction == Descending { key_exclusive } else { key_max };
        assert!(key_min_exclusive <= key_max_exclusive);

        let mut it = self.iterator(
            Visibility::Visible,
            &snapshots,
            direction,
            Some(KeyRange { key_min: key_min_exclusive, key_max: key_max_exclusive }),
        );

        while let Some(table) = it.next() {
            assert!(table.visible(snapshot));
            assert!(table.key_min() <= table.key_max());
            assert!(key_min_exclusive <= table.key_max());
            assert!(table.key_min() <= key_max_exclusive);

            // These conditions are required to avoid iterating over the same
            // table twice. This is because the invoker sets key_exclusive to the
            // key_min or key_max of the previous table returned by this function,
            // based on the direction of iteration (ascending/descending).
            // key_exclusive is then set as KeyRange.key_min or KeyRange.key_max for the next
            // ManifestLevel query. This query would return the same table again,
            // so it needs to be skipped.
            let next = match direction {
                Ascending => table.key_min() > key_exclusive,
                Descending => table.key_max() < key_exclusive,
            };
            if next {
                return Some(TableInfoReference {
                    table_info: *table,
                    generation: self.generation,
                });
            }
        }

        None
    }

    /// Returns the smallest visible range of tables in the given level
    /// that overlap with the given range: [key_min, key_max].
    ///
    /// Returns null if the number of tables that intersect with the range intersects more
    /// than `max_overlapping_tables` tables.
    ///
    /// The range keys are guaranteed to encompass all the relevant level A and level B
    /// tables:
    ///   range.key_min = min(a.key_min, b.key_min)
    ///   range.key_max = max(a.key_max, b.key_max)
    ///
    /// This last invariant is critical to ensuring that tombstones are dropped correctly.
    ///
    /// * Assumption: Currently, we only support a maximum of lsm_growth_factor overlapping
    ///   tables. This is because OverlapRange.tables is a BoundedArray of size
    ///   lsm_growth_factor. This works with our current compaction strategy that is
    ///   guaranteed to choose a table that intersects with <= lsm_growth_factor tables in
    ///   the next level.
    ///
    /// # Panics
    /// Panics if `max_overlapping_tables` exceeds the growth factor or on invariant
    /// violations (upstream asserts).
    #[must_use]
    pub fn tables_overlapping_with_key_range(
        &self,
        key_min: S::Key,
        key_max: S::Key,
        snapshot: u64,
        max_overlapping_tables: usize,
    ) -> Option<OverlapRange<S::TableInfo, S::Key>> {
        assert!(max_overlapping_tables <= LSM_GROWTH_FACTOR_USIZE);

        let mut range = OverlapRange { key_min, key_max, tables: BoundedArray::new() };
        let snapshots = [snapshot];
        let mut it = self.iterator(
            Visibility::Visible,
            &snapshots,
            Ascending,
            Some(KeyRange { key_min: range.key_min, key_max: range.key_max }),
        );

        while let Some(table) = it.next() {
            assert!(table.visible(SNAPSHOT_LATEST));
            assert!(table.key_min() <= table.key_max());
            assert!(range.key_min <= table.key_max());
            assert!(table.key_min() <= range.key_max);

            // The first iterated table.key_min/max may overlap range.key_min/max entirely.
            if table.key_min() < range.key_min {
                range.key_min = table.key_min();
            }

            // Thereafter, iterated tables may/may not extend the range in ascending order.
            if table.key_max() > range.key_max {
                range.key_max = table.key_max();
            }
            if range.tables.count() < max_overlapping_tables {
                range
                    .tables
                    .push(TableInfoReference { table_info: *table, generation: self.generation });
            } else {
                return None;
            }
        }
        assert!(range.key_min <= range.key_max);
        assert!(range.key_min <= key_min);
        assert!(range.tables.count() <= max_overlapping_tables);
        assert!(key_max <= range.key_max);

        Some(range)
    }
}

impl<S: ManifestLevelSpec> Default for ManifestLevel<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameters for [`ManifestLevel::next_table`] (upstream anonymous struct argument).
#[derive(Clone, Copy, Debug)]
pub struct NextTableParameters<K> {
    pub snapshot: u64,
    pub key_min: K,
    pub key_max: K,
    pub key_exclusive: Option<K>,
    pub direction: Direction,
}

/// Filtered view over a level's tables (upstream `ManifestLevel.Iterator`).
pub struct ManifestLevelIterator<'a, S: ManifestLevelSpec> {
    inner: SegmentedArrayIterator<'a, TablesSpec<S>>,
    visibility: Visibility,
    snapshots: &'a [u64],
    direction: Direction,
    key_range: Option<KeyRange<S::Key>>,
}

impl<'a, S: ManifestLevelSpec> ManifestLevelIterator<'a, S> {
    /// Advances to the next matching table (upstream `Iterator.next`); like upstream, the
    /// result references memory owned by the level.
    ///
    /// # Panics
    /// Panics if a yielded table has out-of-range snapshots or inconsistent metadata
    /// (upstream asserts), or if iteration continues past exhaustion.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&'a S::TableInfo> {
        while let Some(table) = self.inner.next() {
            // We can't assert !self.inner.is_done() here, as inner.next() may set done
            // before returning.

            // Skip tables that don't match the provided visibility interests
            // (upstream uses a labeled block with `continue`).
            let matches_visibility = match self.visibility {
                Visibility::Invisible => table.invisible(self.snapshots),
                Visibility::Visible => {
                    self.snapshots.iter().any(|&snapshot| table.visible(snapshot))
                }
            };
            if !matches_visibility {
                continue;
            }

            // Filter the table using the key range if provided.
            if let Some(key_range) = self.key_range {
                match self.direction {
                    Ascending => {
                        // Assert that the table is not out of bounds to the left.
                        //
                        // We can assert this as it is exactly the same key comparison when
                        // we binary search in iterator_start(), and since we move in
                        // ascending order this remains true beyond the first iteration.
                        assert!(key_range.key_min <= table.key_max());

                        // Check if the table is out of bounds to the right.
                        if table.key_min() > key_range.key_max {
                            self.inner.stop();
                            return None;
                        }
                    }
                    Descending => {
                        // Check if the table is out of bounds to the right.
                        //
                        // Unlike in the ascending case, it is not guaranteed that
                        // table.key_min is less than or equal to key_range.key_max on the
                        // first iteration as the underlying SegmentedArray search uses
                        // upper_bound regardless of direction.
                        if table.key_min() > key_range.key_max {
                            continue;
                        }

                        // Check if the table is out of bounds to the left.
                        if table.key_max() < key_range.key_min {
                            self.inner.stop();
                            return None;
                        }
                    }
                }
            }

            return Some(table);
        }

        assert!(self.inner.is_done());
        None
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;
    use crate::binary_search::{Config, binary_search_values_upsert_index};
    use crate::segmented_array::node_capacity_for;
    use tigerbeetle_core::stdx::prng::Prng;

    /// Upstream `TestContextType`'s `TableInfo` (`manifest.zig`'s `TreeTableInfo` for the
    /// test table type).
    ///
    /// DEVIATION: upstream asserts `@sizeOf(TestTableInfo) == 56`; with Rust's `#[repr(C)]`
    /// layout the trailing `value_count` padding rounds the size up to 64. These tests never
    /// serialize the struct, so only the exact size changes.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    struct TestTableInfo {
        checksum: u128,
        address: u64,
        snapshot_min: u64,
        snapshot_max: u64,
        key_min: u64,
        key_max: u64,
        value_count: u32,
    }

    const _: () = assert!(size_of::<TestTableInfo>() == 64);

    impl LevelTableInfo for TestTableInfo {
        type Key = u64;

        fn checksum(&self) -> u128 {
            self.checksum
        }
        fn address(&self) -> u64 {
            self.address
        }
        fn snapshot_min(&self) -> u64 {
            self.snapshot_min
        }
        fn snapshot_max(&self) -> u64 {
            self.snapshot_max
        }
        fn set_snapshot_max(&mut self, snapshot_max: u64) {
            self.snapshot_max = snapshot_max;
        }
        fn key_min(&self) -> u64 {
            self.key_min
        }
        fn key_max(&self) -> u64 {
            self.key_max
        }
        fn value_count(&self) -> u32 {
            self.value_count
        }
    }

    fn key_min_from_table(table: &TestTableInfo) -> u64 {
        table.key_min
    }

    macro_rules! test_spec {
        ($name:ident, $node_size:expr, $count_max:expr) => {
            #[derive(Clone, Copy, Debug, Default)]
            struct $name;

            impl ManifestLevelSpec for $name {
                type Key = u64;
                type TableInfo = TestTableInfo;
                const TABLE_COUNT_MAX_TREE: u32 = $count_max;
                const KEYS_NODE_CAPACITY: usize = node_capacity_for($node_size, size_of::<u64>());
                const TABLES_NODE_CAPACITY: usize =
                    node_capacity_for($node_size, size_of::<TestTableInfo>());
                type TableSortKey = u128;
                const TABLE_SORT_KEY_MIN: u128 = 0;
                const TABLE_SORT_KEY_MAX: u128 = u128::MAX;
                fn table_sort_key(table_info: &TestTableInfo) -> u128 {
                    (KeyMaxSnapshotMin {
                        snapshot_min: table_info.snapshot_min,
                        key_max: table_info.key_max,
                    })
                    .to_int()
                }
                const KEY_MIN: u64 = 0;
                const KEY_MAX: u64 = u64::MAX;
            }
        };
    }

    test_spec!(SpecNode256Max33, 256, 33);
    test_spec!(SpecNode256Max34, 256, 34);
    test_spec!(SpecNode256Max1024, 256, 1024);
    test_spec!(SpecNode512Max1024, 512, 1024);
    test_spec!(SpecNode1024Max1024, 1024, 1024);

    /// Any [`ManifestLevelSpec`] over the shared test key/info types.
    trait TestSpec: ManifestLevelSpec<Key = u64, TableInfo = TestTableInfo> {}
    impl<T: ManifestLevelSpec<Key = u64, TableInfo = TestTableInfo>> TestSpec for T {}

    enum Action {
        InsertTables,
        CreateSnapshot,
        DeleteTables,
        DropSnapshot,
    }

    fn weighted_action(prng: &mut Prng, weights: (u32, u32, u32, u32)) -> Action {
        let roll = prng.gen_int_inclusive_u32(weights.0 + weights.1 + weights.2 + weights.3 - 1);
        if roll < weights.0 {
            Action::InsertTables
        } else if roll < weights.0 + weights.1 {
            Action::CreateSnapshot
        } else if roll < weights.0 + weights.1 + weights.2 {
            Action::DeleteTables
        } else {
            Action::DropSnapshot
        }
    }

    /// Port of upstream `TestContextType`.
    ///
    /// DEVIATION: upstream threads one shared PRNG through all five configurations; each
    /// context here owns a freshly seeded PRNG instead (determinism is unaffected).
    struct TestContext<S: TestSpec> {
        prng: Prng,
        level: ManifestLevel<S>,
        snapshot_max: u64,
        snapshots: Vec<u64>,
        snapshot_tables: Vec<Vec<TestTableInfo>>,
        /// Contains only tables with snapshot_max == SNAPSHOT_LATEST.
        reference: Vec<TestTableInfo>,
        inserts: u64,
        removes: u64,
        spec: PhantomData<S>,
    }

    impl<S: TestSpec> TestContext<S> {
        fn new() -> Self {
            Self {
                prng: Prng::from_seed(42),
                level: ManifestLevel::new(),
                snapshot_max: 1,
                snapshots: Vec::new(),
                snapshot_tables: Vec::new(),
                reference: Vec::new(),
                inserts: 0,
                removes: 0,
                spec: PhantomData,
            }
        }

        fn run(&mut self) {
            // Phase 1: mostly inserts.
            #[allow(clippy::cast_possible_truncation)]
            for _ in 0..(S::TABLE_COUNT_MAX_TREE as usize) * 2 {
                match weighted_action(&mut self.prng, (60, 10, 25, 5)) {
                    Action::InsertTables => self.insert_tables(),
                    Action::CreateSnapshot => self.create_snapshot(),
                    Action::DeleteTables => self.delete_tables(),
                    Action::DropSnapshot => self.drop_snapshot(),
                }
            }

            // Phase 2: mostly deletes.
            #[allow(clippy::cast_possible_truncation)]
            for _ in 0..(S::TABLE_COUNT_MAX_TREE as usize) * 2 {
                match weighted_action(&mut self.prng, (35, 5, 50, 10)) {
                    Action::InsertTables => self.insert_tables(),
                    Action::CreateSnapshot => self.create_snapshot(),
                    Action::DeleteTables => self.delete_tables(),
                    Action::DropSnapshot => self.drop_snapshot(),
                }
            }

            self.remove_all();
        }

        fn insert_tables(&mut self) {
            // Both sides fit in u32; upstream computes in u32 as well.
            #[allow(clippy::cast_possible_truncation)]
            let count_free = (S::TABLE_COUNT_MAX_TREE - self.level.keys.len()) as usize;
            if count_free == 0 {
                return;
            }

            let count_max_insert = count_free.min(13);
            let count = self.prng.range_inclusive_usize(1, count_max_insert);

            let mut buffer = Vec::with_capacity(count);

            let mut key = self.prng.gen_int_inclusive_u64(u64::from(S::TABLE_COUNT_MAX_TREE) * 64);
            for _ in 0..count {
                let table = self.random_greater_non_overlapping_table(key);
                key = table.key_max;
                buffer.push(table);
            }

            for table in &buffer {
                self.level.insert_table(table);
            }

            for table in &buffer {
                let index = binary_search_values_upsert_index(
                    &key_min_from_table,
                    &self.reference,
                    table.key_max,
                    Config::default(),
                ) as usize;
                // Can't be equal as the tables may not overlap.
                if index < self.reference.len() {
                    assert!(self.reference[index].key_min > table.key_max);
                }
                self.reference.insert(index, *table);
            }

            self.inserts += count as u64;

            self.verify();
        }

        fn random_greater_non_overlapping_table(&mut self, key: u64) -> TestTableInfo {
            let mut new_key_min = key + self.prng.range_inclusive_usize(1, 31) as u64;
            assert!(new_key_min > key);

            let i = binary_search_values_upsert_index(
                &key_min_from_table,
                &self.reference,
                new_key_min,
                Config::default(),
            ) as usize;

            if i > 0 && new_key_min <= self.reference[i - 1].key_max {
                new_key_min = self.reference[i - 1].key_max + 1;
            }

            let next_key_min = {
                let mut result = u64::MAX;
                for table in &self.reference[i..] {
                    match new_key_min.cmp(&table.key_min) {
                        core::cmp::Ordering::Less => {
                            result = table.key_min;
                            break;
                        }
                        core::cmp::Ordering::Equal => new_key_min = table.key_max + 1,
                        core::cmp::Ordering::Greater => unreachable!(),
                    }
                }
                result
            };

            let max_delta = 32.min(next_key_min - 1 - new_key_min) as u64;
            let new_key_max = new_key_min + self.prng.gen_int_inclusive_u64(max_delta);

            TestTableInfo {
                checksum: self.prng.int_u128(),
                address: self.prng.int_u64(),
                // Upstream relies on the struct default: snapshot_max starts at maxInt(u64).
                snapshot_max: u64::MAX,
                snapshot_min: self.take_snapshot(),
                key_min: new_key_min,
                key_max: new_key_max,
                value_count: self.prng.int_u32(),
            }
        }

        /// See `Manifest.take_snapshot()`.
        fn take_snapshot(&mut self) -> u64 {
            // A snapshot cannot be 0 as this is a reserved value in the superblock.
            assert!(self.snapshot_max > 0);
            // The constant snapshot_latest must compare greater than any issued snapshot.
            // This also ensures that we are not about to overflow the u64 counter.
            assert!(self.snapshot_max < SNAPSHOT_LATEST - 1);

            self.snapshot_max += 1;

            self.snapshot_max
        }

        fn create_snapshot(&mut self) {
            if self.snapshots.len() >= 8 {
                return;
            }

            let snapshot = self.take_snapshot();
            self.snapshots.push(snapshot);
            self.snapshot_tables.push(self.reference.clone());
        }

        fn drop_snapshot(&mut self) {
            if self.snapshots.is_empty() {
                return;
            }

            let index = self.prng.index(self.snapshots.len());

            self.snapshots.swap_remove(index);
            self.snapshot_tables.swap_remove(index);

            let snapshots = self.snapshots.clone();

            // Ensure that iteration with a null key range in both directions is tested.
            let mut tables: Vec<TestTableInfo> = Vec::new();
            if self.prng.boolean() {
                let mut it =
                    self.level.iterator(Visibility::Invisible, &snapshots, Ascending, None);
                while let Some(table) = it.next() {
                    tables.push(*table);
                }
            } else {
                let mut it =
                    self.level.iterator(Visibility::Invisible, &snapshots, Descending, None);
                while let Some(table) = it.next() {
                    tables.push(*table);
                }
                tables.reverse();
            }

            for table in &tables {
                self.level.remove_table(table);
            }
        }

        fn find_exact_reference(
            &self,
            table: &TestTableInfo,
        ) -> Option<TableInfoReference<TestTableInfo>> {
            // Use `key_min` for both ends of the iterator; we are looking for a single table.
            let cursor_start =
                self.level.iterator_start(table.key_min, table.key_min, Ascending)?;
            let absolute_index = self.level.keys.absolute_index_for_cursor(cursor_start);

            let mut it = self.level.tables.iterator_from_index(absolute_index, Ascending);
            while let Some(level_table) = it.next() {
                if level_table.equal(table) {
                    return Some(TableInfoReference {
                        table_info: *level_table,
                        generation: self.level.generation,
                    });
                }
            }
            None
        }

        fn delete_tables(&mut self) {
            let reference_len = self.reference.len();
            if reference_len == 0 {
                return;
            }

            let count_max_delete = reference_len.min(13);
            let count = self.prng.range_inclusive_usize(1, count_max_delete);

            assert!(reference_len <= S::TABLE_COUNT_MAX_TREE as usize);
            let index = self.prng.int_inclusive_usize(reference_len - count);

            let snapshot = self.take_snapshot();

            // Copy out the window so that the level can be borrowed immutably while
            // searching, then mutated (upstream mutates through pointers).
            let mut modified = self.reference[index..index + count].to_vec();

            for table in &mut modified {
                if let Some(found) = self.find_exact_reference(table) {
                    self.level.set_snapshot_max(snapshot, found);
                    table.set_snapshot_max(snapshot);
                }
            }
            self.reference[index..index + count].copy_from_slice(&modified);

            for tables in &mut self.snapshot_tables {
                for table in tables.iter_mut() {
                    for modified_table in &modified {
                        if table.address == modified_table.address {
                            table.set_snapshot_max(snapshot);
                            assert!(table.equal(modified_table));
                        }
                    }
                }
            }

            {
                let to_remove: Vec<TestTableInfo> = modified
                    .iter()
                    .filter(|table| table.invisible(&self.snapshots))
                    .copied()
                    .collect();

                for table in &to_remove {
                    self.level.remove_table(table);
                }
            }

            self.reference.drain(index..index + count);

            self.removes += count as u64;

            self.verify();
        }

        fn remove_all(&mut self) {
            while !self.snapshots.is_empty() {
                self.drop_snapshot();
            }
            while !self.reference.is_empty() {
                self.delete_tables();
            }

            assert_eq!(self.level.keys.len(), 0);
            assert_eq!(self.level.tables.len(), 0);
            assert!(self.inserts > 0);
            assert_eq!(self.inserts, self.removes);

            self.verify();
        }

        fn verify(&mut self) {
            let reference = self.reference.clone();
            self.verify_snapshot(SNAPSHOT_LATEST, &reference);

            for (i, &snapshot) in self.snapshots.clone().iter().enumerate() {
                let tables = self.snapshot_tables[i].clone();
                self.verify_snapshot(snapshot, &tables);
            }
        }

        fn verify_snapshot(&mut self, snapshot: u64, reference: &[TestTableInfo]) {
            let snapshots = [snapshot];
            let full_range = KeyRange { key_min: 0, key_max: u64::MAX };

            {
                let mut it = self.level.iterator(
                    Visibility::Visible,
                    &snapshots,
                    Ascending,
                    Some(full_range),
                );

                for expect in reference {
                    let Some(actual) = it.next() else {
                        panic!("TestUnexpectedResult");
                    };
                    assert_eq!(expect, actual);
                }
                assert!(it.next().is_none());
            }

            {
                let mut it = self.level.iterator(
                    Visibility::Visible,
                    &snapshots,
                    Descending,
                    Some(full_range),
                );

                for expect in reference.iter().rev() {
                    let Some(actual) = it.next() else {
                        panic!("TestUnexpectedResult");
                    };
                    assert_eq!(expect, actual);
                }
                assert!(it.next().is_none());
            }

            if !reference.is_empty() {
                let start = self.prng.int_inclusive_usize(reference_len_minus_one(reference));
                let end =
                    self.prng.range_inclusive_usize(start, reference_len_minus_one(reference));

                let key_min = reference[start].key_min;
                let key_max = reference[end].key_max;

                {
                    let mut it = self.level.iterator(
                        Visibility::Visible,
                        &snapshots,
                        Ascending,
                        Some(KeyRange { key_min, key_max }),
                    );

                    for expect in &reference[start..=end] {
                        let Some(actual) = it.next() else {
                            panic!("TestUnexpectedResult");
                        };
                        assert_eq!(expect, actual);
                    }
                    assert!(it.next().is_none());
                }

                {
                    let mut it = self.level.iterator(
                        Visibility::Visible,
                        &snapshots,
                        Descending,
                        Some(KeyRange { key_min, key_max }),
                    );

                    for expect in reference[start..=end].iter().rev() {
                        let Some(actual) = it.next() else {
                            panic!("TestUnexpectedResult");
                        };
                        assert_eq!(expect, actual);
                    }
                    assert!(it.next().is_none());
                }
            }
        }

        fn run_fuzz_overlap() {
            let a_tables_max = 20;

            let mut prng = Prng::from_seed(42);

            for _ in 0..100 {
                let mut level_a: ManifestLevel<S> = ManifestLevel::new();
                let mut level_b: ManifestLevel<S> = ManifestLevel::new();

                let count_a = prng.range_inclusive_usize(1, a_tables_max);
                // Skew count_b <= count_a * growth_factor so that by the pigeonhole
                // principle at least one A table has overlap <= growth_factor in most
                // iterations. This mirrors the real LSM invariant.
                let count_b = prng.int_inclusive_usize(count_a * LSM_GROWTH_FACTOR_USIZE);
                let max_overlap = LSM_GROWTH_FACTOR_USIZE;

                // Generate non-overlapping tables for level A.
                let mut key = prng.gen_int_inclusive_u64(500);
                for _ in 0..count_a {
                    key += prng.range_inclusive_usize(1, 31) as u64;
                    let key_min = key;
                    key += prng.gen_int_inclusive_u64(32);
                    level_a.insert_table(&random_table(key_min, key, &mut prng));
                }

                // Generate non-overlapping tables for level B.
                let mut key = prng.gen_int_inclusive_u64(500);
                for _ in 0..count_b {
                    key += prng.range_inclusive_usize(1, 31) as u64;
                    let key_min = key;
                    key += prng.gen_int_inclusive_u64(32);
                    level_b.insert_table(&random_table(key_min, key, &mut prng));
                }

                let expected = brute_force_least_overlap::<S>(&level_a, &level_b, max_overlap);

                let result =
                    level_a.table_with_least_overlap(&level_b, SNAPSHOT_LATEST, max_overlap);

                assert_eq!(expected.0.key_min, result.table.table_info.key_min);
                assert_eq!(expected.0.key_max, result.table.table_info.key_max);
                assert_eq!(expected.1.len(), result.range.tables.count());
                for (expected_b, actual_b) in expected.1.iter().zip(result.range.tables.slice()) {
                    assert_eq!(expected_b.key_min, actual_b.table_info.key_min);
                    assert_eq!(expected_b.key_max, actual_b.table_info.key_max);
                }
            }
        }
    }

    fn reference_len_minus_one(reference: &[TestTableInfo]) -> usize {
        reference.len() - 1
    }

    fn random_table(key_min: u64, key_max: u64, prng: &mut Prng) -> TestTableInfo {
        TestTableInfo {
            checksum: prng.int_u128(),
            address: prng.int_u64(),
            // Upstream relies on the struct default: snapshot_max starts at maxInt(u64).
            snapshot_max: u64::MAX,
            snapshot_min: 1,
            key_min,
            key_max,
            value_count: prng.int_u32(),
        }
    }

    /// Returns `(best_table, best_overlapping_b_tables)` by brute force.
    fn brute_force_least_overlap<S: TestSpec>(
        level_a: &ManifestLevel<S>,
        level_b: &ManifestLevel<S>,
        max_overlap: usize,
    ) -> (TestTableInfo, Vec<TestTableInfo>) {
        let snapshots = [SNAPSHOT_LATEST];
        let mut best_table: Option<(TestTableInfo, usize)> = None;
        let mut best_b_tables: Vec<TestTableInfo> = Vec::new();

        let mut it_a = level_a.iterator(Visibility::Visible, &snapshots, Ascending, None);
        'tables: while let Some(table_a) = it_a.next() {
            let mut overlap: usize = 0;
            let overlapping: Vec<TestTableInfo> = {
                let mut result = Vec::new();
                let mut it_b = level_b.iterator(Visibility::Visible, &snapshots, Ascending, None);
                while let Some(table_b) = it_b.next() {
                    if table_b.key_max() >= table_a.key_min()
                        && table_b.key_min() <= table_a.key_max()
                    {
                        overlap += 1;
                        result.push(*table_b);
                    }
                }
                result
            };
            if overlap <= max_overlap && best_table.is_none_or(|(_, best)| overlap < best) {
                best_table = Some((*table_a, overlap));
                best_b_tables = overlapping;
            }
            if best_table.is_some_and(|(_, best)| best == 0) {
                break 'tables;
            }
        }
        let Some((table, _)) = best_table else {
            panic!("a candidate always exists");
        };
        (table, best_b_tables)
    }

    #[test]
    fn manifest_level_node_256_count_33() {
        TestContext::<SpecNode256Max33>::new().run();
    }

    #[test]
    fn manifest_level_node_256_count_34() {
        TestContext::<SpecNode256Max34>::new().run();
    }

    #[test]
    fn manifest_level_node_256_count_1024() {
        TestContext::<SpecNode256Max1024>::new().run();
    }

    #[test]
    fn manifest_level_node_512_count_1024() {
        TestContext::<SpecNode512Max1024>::new().run();
    }

    #[test]
    fn manifest_level_node_1024_count_1024() {
        TestContext::<SpecNode1024Max1024>::new().run();
    }

    #[test]
    fn fuzz_table_with_least_overlap_random_levels() {
        TestContext::<SpecNode256Max1024>::run_fuzz_overlap();
    }
}
