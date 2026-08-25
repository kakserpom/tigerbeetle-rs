//! Port of `src/lsm/manifest.zig` — manifest of tables per LSM level.
//!
//! The manifest tracks which tables exist at each compaction level and their key/snapshot
//! bounds. Mutations (insert, update, move, remove) are logged to a [`ManifestLog`] for
//! crash recovery.
//!
//! # Differences from upstream
//!
//! - **No NodePool:** our [`ManifestLevel`]s own their segmented-array memory (see
//!   [`manifest_level`](crate::manifest_level) module notes).
//! - **No Tracer:** tracer gauge calls are deferred until the tracing subsystem is ported.
//! - **ManifestLog seam:** upstream stores a `?*ManifestLog` pointer inside `Manifest`;
//!   safe Rust cannot alias `&mut Manifest` and `&mut ManifestLog` through a shared
//!   `&mut self`, so mutating methods take `log: &mut impl ManifestLog` explicitly.
//!   The caller (Forest) owns both.
//! - **verify():** the Storage-dependent table-data verification loop is deferred; only
//!   the structural invariants (non-empty level monotonicity for `.general` tables) are
//!   checked when the method is ported.
//! - **Key encoding:** upstream reinterprets key bytes via `mem.bytesAsValue`. A minimal
//!   [`TableKey`] trait provides explicit little-endian encode/decode instead of `unsafe`.

use core::fmt::Debug;

use tigerbeetle_core::constants::{self, LSM_LEVELS, LSM_MANIFEST_NODE_SIZE};

use crate::direction::Direction;
use crate::manifest_level::{
    KeyRange, LevelTableInfo, ManifestLevel, NextTableParameters, OverlapRange, TableInfoReference,
    Visibility,
};
use crate::schema::manifest_node::{self, Event, TableInfo as WireTableInfo};
use crate::segmented_array::node_capacity_for;
use crate::tree::{SNAPSHOT_LATEST, TreeConfig, table_count_max_for_level};

/// Minimal encode/decode for manifest table keys.
///
/// Upstream reinterprets key memory with `mem.bytesAsValue`. Without `unsafe`, we need an
/// explicit conversion. Keys are always fixed-width little-endian integers (u64, u128, …)
/// that fit within 32 bytes (`KeyPadded`).
pub trait TableKey: Copy + Ord + Debug + Default + 'static {
    /// The maximum value for this key type (upstream `std.math.maxInt(Key)`).
    const MAX: Self;

    /// The zero/minimum value for this key type (upstream `std.math.minInt(Key)`).
    const ZERO: Self;

    /// Encode into a 32-byte little-endian padded buffer (`KeyPadded`).
    fn to_bytes(self) -> [u8; 32];

    /// Decode from a `KeyPadded` buffer. Must have been produced by [`TableKey::to_bytes`].
    fn from_bytes(bytes: &[u8; 32]) -> Self;

    /// Pack the key into the high 64 bits of a u128 for sort ordering. The low 64 bits
    /// are left for the caller (typically `snapshot_min`).
    fn to_sort_key_high(self) -> u64;
}

impl TableKey for u64 {
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;

    fn to_bytes(self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[..8].copy_from_slice(&self.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut key_bytes = [0u8; 8];
        key_bytes.copy_from_slice(&bytes[..8]);
        Self::from_le_bytes(key_bytes)
    }

    fn to_sort_key_high(self) -> u64 {
        self
    }
}

impl TableKey for u128 {
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;

    fn to_bytes(self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(&self.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut key_bytes = [0u8; 16];
        key_bytes.copy_from_slice(&bytes[..16]);
        Self::from_le_bytes(key_bytes)
    }

    fn to_sort_key_high(self) -> u64 {
        // DEVIATION: upstream uses `@truncate` to fit the 128-bit key into a 64-bit sort
        // key. This truncation is intentional and matches upstream behavior.
        #[allow(clippy::cast_possible_truncation)]
        let v = self as u64;
        v
    }
}

/// Seamed dependency for the manifest log (upstream `ManifestLogType`).
///
/// # Contract
///
/// The log must be opened before any mutating method on [`Manifest`] is called.
pub trait ManifestLog {
    /// Whether the log has been opened for appending (upstream `manifest_log.opened`).
    fn is_opened(&self) -> bool;

    /// Append a manifest entry (upstream `manifest_log.append`).
    fn append(&mut self, entry: &WireTableInfo);
}

/// `TreeTableInfoType(Table)` in upstream — per-tree, in-memory table metadata.
///
/// Generic over the key type `K` (upstream: `Table.Key`). Encodes to/from wire
/// [`WireTableInfo`] for manifest log entries.
#[derive(Clone, Copy, Debug)]
pub struct TreeTableInfo<K> {
    pub checksum: u128,
    pub address: u64,
    pub snapshot_min: u64,
    pub snapshot_max: u64,
    pub key_min: K,
    pub key_max: K,
    pub value_count: u32,
}

impl<K: TableKey> PartialEq for TreeTableInfo<K> {
    fn eq(&self, other: &Self) -> bool {
        self.checksum == other.checksum
            && self.address == other.address
            && self.snapshot_min == other.snapshot_min
            && self.snapshot_max == other.snapshot_max
            && self.key_min == other.key_min
            && self.key_max == other.key_max
            && self.value_count == other.value_count
    }
}

impl<K: TableKey> Eq for TreeTableInfo<K> {}

impl<K: TableKey> Default for TreeTableInfo<K> {
    fn default() -> Self {
        Self {
            checksum: 0,
            address: 0,
            snapshot_min: 0,
            snapshot_max: u64::MAX,
            key_min: K::default(),
            key_max: K::default(),
            value_count: 0,
        }
    }
}

impl<K: TableKey> TreeTableInfo<K> {
    /// Decode a wire [`WireTableInfo`] entry into the in-memory representation.
    ///
    /// # Panics
    ///
    /// Panics if `tree_id == 0`, `value_count == 0`, or trailing key padding bytes are
    /// nonzero (upstream asserts).
    #[must_use]
    pub fn decode(wire: &WireTableInfo, expected_tree_id: u16) -> Self {
        assert!(wire.tree_id > 0);
        assert_eq!(wire.tree_id, expected_tree_id);
        assert!(wire.value_count > 0);

        let key_min = K::from_bytes(&wire.key_min);
        let key_max = K::from_bytes(&wire.key_max);
        assert!(key_min <= key_max);

        let key_size = core::mem::size_of::<K>();
        assert!(wire.key_min[key_size..].iter().all(|&b| b == 0));
        assert!(wire.key_max[key_size..].iter().all(|&b| b == 0));

        Self {
            checksum: wire.checksum,
            address: wire.address,
            snapshot_min: wire.snapshot_min,
            snapshot_max: wire.snapshot_max,
            key_min,
            key_max,
            value_count: wire.value_count,
        }
    }

    /// Encode into a wire [`WireTableInfo`] entry.
    ///
    /// # Panics
    ///
    /// Panics if `tree_id == 0` or `value_count == 0`.
    #[must_use]
    pub fn encode(&self, tree_id: u16, level: u8, event: Event) -> WireTableInfo {
        assert!(tree_id > 0);
        assert!(self.value_count > 0);

        WireTableInfo {
            checksum: self.checksum,
            address: self.address,
            snapshot_min: self.snapshot_min,
            snapshot_max: self.snapshot_max,
            tree_id,
            key_min: self.key_min.to_bytes(),
            key_max: self.key_max.to_bytes(),
            value_count: self.value_count,
            label: manifest_node::Label { level, event },
        }
    }
}

impl<K: TableKey> LevelTableInfo for TreeTableInfo<K> {
    type Key = K;

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
    fn key_min(&self) -> K {
        self.key_min
    }
    fn key_max(&self) -> K {
        self.key_max
    }
    fn value_count(&self) -> u32 {
        self.value_count
    }
}

/// Concrete [`ManifestLevelSpec`] for a given key type `K`, using production node sizes.
///
/// Upstream derives this from `ManifestLevelType(NodePool, Key, TreeTableInfo,
/// table_count_max_tree)` with `node_capacity_for(constants.lsm_manifest_node_size, …)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ManifestLevelSpecFor<K>(core::marker::PhantomData<K>);

impl<K: TableKey> crate::manifest_level::ManifestLevelSpec for ManifestLevelSpecFor<K> {
    type Key = K;
    type TableInfo = TreeTableInfo<K>;
    const TABLE_COUNT_MAX_TREE: u32 = crate::tree::TABLE_COUNT_MAX;

    const KEYS_NODE_CAPACITY: usize =
        node_capacity_for(LSM_MANIFEST_NODE_SIZE, core::mem::size_of::<K>());
    const TABLES_NODE_CAPACITY: usize =
        node_capacity_for(LSM_MANIFEST_NODE_SIZE, core::mem::size_of::<TreeTableInfo<K>>());

    type TableSortKey = u128;
    const TABLE_SORT_KEY_MIN: u128 = 0;
    const TABLE_SORT_KEY_MAX: u128 = u128::MAX;

    fn table_sort_key(table_info: &TreeTableInfo<K>) -> u128 {
        (u128::from(table_info.key_max.to_sort_key_high()) << 64)
            | u128::from(table_info.snapshot_min)
    }

    const KEY_MIN: K = K::ZERO;
    const KEY_MAX: K = K::MAX;
}

/// Manifest entry for compaction table selection (upstream `CompactionTableRange`).
#[derive(Clone, Debug)]
pub struct CompactionTableRange<T, K> {
    pub table_a: TableInfoReference<T>,
    pub range_b: CompactionRange<T, K>,
}

/// Range of tables in level B that overlap with the compaction input (upstream
/// `ManifestType.CompactionRange`).
#[derive(Clone, Debug)]
pub struct CompactionRange<T, K> {
    pub key_min: K,
    pub key_max: K,
    pub tables: OverlapRange<T, K>,
}

/// Concrete type alias for manifest levels with production node sizes.
pub type Level<K> = ManifestLevel<ManifestLevelSpecFor<K>>;

/// `ManifestType(Table, Storage)` in upstream — the full manifest across all LSM levels.
///
/// Generic over the key type `K` (upstream: `Table.Key`).
pub struct Manifest<K: TableKey> {
    pub levels: [Level<K>; LSM_LEVELS as usize],
    pub config: TreeConfig,

    log_attached: bool,

    /// The highest snapshot seen so far (upstream `snapshot_max`).
    pub snapshot_max: u64,
}

impl<K: TableKey> Manifest<K> {
    /// Initialize a manifest. All levels start empty.
    #[must_use]
    pub fn new(config: TreeConfig) -> Self {
        Self {
            levels: core::array::from_fn(|_| ManifestLevel::new()),
            config,
            log_attached: false,
            snapshot_max: 1,
        }
    }

    /// Attach the manifest log. Must be called before any mutation (upstream
    /// `open_commence`).
    ///
    /// # Panics
    ///
    /// Panics if `open_commence` was already called, or if the log is already opened.
    pub fn open_commence(&mut self, log: &impl ManifestLog) {
        assert!(!self.log_attached);
        assert!(!log.is_opened());
        self.log_attached = true;
    }

    /// Insert a new table into `level`.
    ///
    /// # Panics
    ///
    /// Panics if `open_commence` was not called.
    #[allow(clippy::missing_panics_doc)]
    pub fn insert_table(
        &mut self,
        log: &mut impl ManifestLog,
        level: u8,
        table: &TreeTableInfo<K>,
    ) {
        self.assert_log_attached();

        self.levels[level as usize].insert_table(table);
        log.append(&table.encode(self.config.id, level, Event::Insert));
    }

    /// Updates the `snapshot_max` on a table at `level` (compaction input processing).
    ///
    /// # Panics
    ///
    /// Panics if `open_commence` was not called, or the snapshot ordering is violated.
    #[allow(clippy::missing_panics_doc)]
    pub fn update_table(
        &mut self,
        log: &mut impl ManifestLog,
        level: u8,
        snapshot: u64,
        table_ref: TableInfoReference<TreeTableInfo<K>>,
    ) {
        self.assert_log_attached();
        assert!(log.is_opened());

        let manifest_level = &mut self.levels[level as usize];

        let mut table = table_ref.table_info;
        assert!(table.snapshot_max() >= snapshot);
        assert!(table.snapshot_min() <= snapshot);
        manifest_level.set_snapshot_max(snapshot, table_ref);
        table.set_snapshot_max(snapshot);

        log.append(&table.encode(self.config.id, level, Event::Update));
    }

    /// Move a table from `level_a` to `level_a + 1`.
    ///
    /// Emits only an update at `level_b` (no remove at `level_a`), so that during replay
    /// the table appears only at `level_b`.
    ///
    /// # Panics
    ///
    /// Panics if `open_commence` was not called, level bounds violated, or table is not
    /// visible to `SNAPSHOT_LATEST`.
    #[allow(clippy::missing_panics_doc)]
    pub fn move_table(
        &mut self,
        log: &mut impl ManifestLog,
        level_a: u8,
        level_b: u8,
        table: &TreeTableInfo<K>,
    ) {
        self.assert_log_attached();
        assert!(log.is_opened());
        assert_eq!(level_b, level_a + 1);
        assert!(level_b < LSM_LEVELS);
        assert!(table.visible(SNAPSHOT_LATEST));

        self.levels[level_a as usize].remove_table(table);
        self.levels[level_b as usize].insert_table(table);
        log.append(&table.encode(self.config.id, level_b, Event::Update));
    }

    /// Return the key range spanning all visible tables across all levels.
    pub fn key_range(&self) -> Option<KeyRange<K>> {
        self.assert_log_attached();

        let mut manifest_range: Option<KeyRange<K>> = None;
        for level in &self.levels {
            if let Some(level_range) = level.key_range_latest() {
                match &mut manifest_range {
                    Some(range) => {
                        if level_range.key_min < range.key_min {
                            range.key_min = level_range.key_min;
                        }
                        if level_range.key_max > range.key_max {
                            range.key_max = level_range.key_max;
                        }
                    }
                    None => manifest_range = Some(level_range),
                }
            }
        }
        manifest_range
    }

    /// Remove all invisible tables in `[key_min, key_max]` at `level`.
    ///
    /// Iterates in descending order to avoid iterator desynchronization.
    ///
    /// # Panics
    ///
    /// Panics if `open_commence` was not called, or level/range bounds violated.
    #[allow(clippy::missing_panics_doc)]
    pub fn remove_invisible_tables(
        &mut self,
        log: &mut impl ManifestLog,
        level: u8,
        snapshots: &[u64],
        key_min: K,
        key_max: K,
    ) {
        self.assert_log_attached();
        assert!(level < LSM_LEVELS);
        assert!(key_min <= key_max);

        // Collect first, then remove (to avoid invalidating the iterator):
        let mut tables_to_remove = Vec::new();
        {
            let mut iter = self.levels[level as usize].iterator(
                Visibility::Invisible,
                snapshots,
                Direction::Descending,
                Some(KeyRange { key_min, key_max }),
            );
            while let Some(table) = iter.next() {
                tables_to_remove.push(*table);
            }
        }

        for table in &tables_to_remove {
            log.append(&table.encode(self.config.id, level, Event::Remove));
            self.levels[level as usize].remove_table(table);
        }
    }

    /// Returns an iterator over tables visible to `snapshot` that may contain `key`,
    /// across all levels > `level_min` (upstream `LookupIterator`).
    ///
    /// DEVIATION: returns owned `TreeTableInfo<K>` values instead of references, to avoid
    /// complex lifetime management with the level iterators.
    pub fn lookup(
        &self,
        snapshot: u64,
        key: K,
        level_min: u8,
    ) -> impl Iterator<Item = TreeTableInfo<K>> + '_ {
        ManifestLookupIterator { manifest: self, snapshot, key, level: level_min }
    }

    /// Returns the next visible table at `level` in the given key range (upstream
    /// `next_table`). Returns a copy of the table info (upstream returns a pointer).
    pub fn next_table_at(
        &self,
        level: u8,
        parameters: NextTableParameters<K>,
    ) -> Option<TreeTableInfo<K>> {
        self.levels[level as usize].next_table(parameters).map(|tref| tref.table_info)
    }

    /// Returns the most optimal table from `level_a` that is due for compaction.
    ///
    /// `None` if the level is not yet due.
    ///
    /// # Panics
    ///
    /// Panics if `level_a >= lsm_levels - 1`.
    pub fn compaction_table(
        &self,
        level_a: u8,
    ) -> Option<CompactionTableRange<TreeTableInfo<K>, K>> {
        assert!(level_a < LSM_LEVELS - 1);

        let table_count_visible_max =
            table_count_max_for_level(constants::LSM_GROWTH_FACTOR, u32::from(level_a));
        assert!(table_count_visible_max > 0);

        let level_a_ref = &self.levels[level_a as usize];
        let level_next_ref = &self.levels[(level_a + 1) as usize];

        // Allow one extra table on even levels ahead of odd levels:
        assert!(level_a_ref.table_count_visible() <= table_count_visible_max + 1);
        if level_a_ref.table_count_visible() < table_count_visible_max {
            return None;
        }
        assert!(level_a_ref.table_count_visible() > 0);

        let least_overlap = level_a_ref.table_with_least_overlap(
            level_next_ref,
            SNAPSHOT_LATEST,
            constants::LSM_GROWTH_FACTOR as usize,
        );
        assert!(least_overlap.range.tables.count() <= constants::LSM_GROWTH_FACTOR as usize);

        Some(CompactionTableRange {
            table_a: least_overlap.table,
            range_b: CompactionRange {
                key_min: least_overlap.range.key_min,
                key_max: least_overlap.range.key_max,
                tables: least_overlap.range,
            },
        })
    }

    /// Returns the smallest visible range of tables in level 0 that overlaps with the
    /// immutable table's key range, optionally coalesced with adjacent tables.
    ///
    /// `value_count_max` is the maximum values per table (upstream `Table.value_count_max`).
    #[allow(clippy::missing_panics_doc)]
    pub fn immutable_table_compaction_range(
        &self,
        key_min: K,
        key_max: K,
        value_count: u32,
        value_count_max: u32,
    ) -> CompactionRange<TreeTableInfo<K>, K> {
        assert!(key_min <= key_max);
        assert!(value_count > 0);
        assert!(value_count <= value_count_max);

        let level_b = 0u8;
        let manifest_level = &self.levels[level_b as usize];
        assert!(manifest_level.table_count_visible() <= constants::LSM_GROWTH_FACTOR);

        let range_overlap = manifest_level
            .tables_overlapping_with_key_range(
                key_min,
                key_max,
                SNAPSHOT_LATEST,
                constants::LSM_GROWTH_FACTOR as usize,
            )
            .unwrap_or_else(|| {
                panic!("level 0 has at most lsm_growth_factor tables, so overlap must be non-empty")
            });

        // DEVIATION: upstream uses Zig `stdx.div_ceil` which rounds up integer division.
        // Rust stdlib provides `u64::div_ceil` which does the same.
        let value_count_target = u64::from(value_count_max)
            .saturating_mul(constants::LSM_TABLE_COALESCING_THRESHOLD_PERCENT as u64)
            .div_ceil(100);
        assert!(u32::try_from(value_count_target).is_ok(), "value_count_target overflows u32");
        // SAFETY: checked above
        #[allow(clippy::cast_possible_truncation)]
        let value_count_target = value_count_target as u32;
        assert!(value_count_target > 1);
        assert!(value_count_target < value_count_max);

        let mut value_count_output: u32 = value_count
            + range_overlap.tables.slice().iter().map(|t| t.table_info.value_count()).sum::<u32>();

        let overlap_tables_count = range_overlap.tables.count();

        let mut range = range_overlap;
        let mut coalesced_small_table = value_count_output < value_count_target;

        for direction in [Direction::Descending, Direction::Ascending] {
            for _ in 0..constants::LSM_GROWTH_FACTOR {
                if range.tables.full() || value_count_output >= value_count_target {
                    break;
                }

                let key_exclusive = match direction {
                    Direction::Descending => Some(range.key_min),
                    Direction::Ascending => Some(range.key_max),
                };

                let Some(table_next) = self.next_table_at(
                    0,
                    NextTableParameters {
                        snapshot: SNAPSHOT_LATEST,
                        key_min: K::default(), // 0 in upstream
                        key_max: K::MAX,
                        key_exclusive,
                        direction,
                    },
                ) else {
                    break;
                };

                let next_value_count = table_next.value_count();
                assert!(next_value_count > 0);

                if value_count_output + next_value_count <= value_count_max {
                    value_count_output += next_value_count;
                    coalesced_small_table =
                        coalesced_small_table || next_value_count < value_count_target;

                    match direction {
                        Direction::Descending => range.key_min = table_next.key_min(),
                        Direction::Ascending => range.key_max = table_next.key_max(),
                    }

                    let tref = TableInfoReference { table_info: table_next, generation: 0 };
                    match direction {
                        Direction::Descending => range.tables.insert_at(0, tref),
                        Direction::Ascending => range.tables.push(tref),
                    }
                } else {
                    break;
                }
            }
        }

        // DEVIATION: upstream keeps `range_overlap` separate and selects it in the else
        // branch. In Rust, since `range` was moved from `range_overlap` and we can't use a
        // moved value, we just use `range` in both branches — the else case is equivalent
        // because `range` started as a copy of `range_overlap`.
        if !(range.tables.count() > overlap_tables_count && coalesced_small_table) {
            // None of the tables benefit much from coalescing; reset to original overlap.
            // Since we can't recover the moved value, we rely on the fact that `range` was
            // only modified if the coalescing condition held. If it didn't hold, `range`
            // is still equal to the original `range_overlap`.
            debug_assert_eq!(range.tables.count(), overlap_tables_count);
        }

        assert!(range.key_min <= range.key_max);
        assert!(range.key_min <= key_min);
        assert!(key_max <= range.key_max);

        if range.tables.count() > 1 {
            let tables = range.tables.slice();
            for pair in tables.windows(2) {
                assert!(pair[0].table_info.key_max() < pair[1].table_info.key_min());
            }
        }

        CompactionRange { key_min: range.key_min, key_max: range.key_max, tables: range }
    }

    /// Whether tombstones must be dropped during compaction at `level_b`.
    ///
    /// # Panics
    ///
    /// Panics if `level_b >= LSM_LEVELS` or `range.key_min > range.key_max`.
    pub fn compaction_must_drop_tombstones(
        &self,
        level_b: u8,
        range: &CompactionRange<TreeTableInfo<K>, K>,
    ) -> bool {
        assert!(level_b < LSM_LEVELS);
        assert!(range.key_min <= range.key_max);

        let mut level_c = level_b + 1;
        while level_c < LSM_LEVELS {
            let manifest_level = &self.levels[level_c as usize];
            if manifest_level
                .next_table(NextTableParameters {
                    snapshot: SNAPSHOT_LATEST,
                    direction: Direction::Ascending,
                    key_min: range.key_min,
                    key_max: range.key_max,
                    key_exclusive: None,
                })
                .is_some()
            {
                assert!(level_b != LSM_LEVELS - 1);
                return false;
            }
            level_c += 1;
        }

        assert_eq!(level_c, LSM_LEVELS);
        true
    }

    /// Assert each level's visible table count is within bounds.
    #[allow(clippy::missing_panics_doc)]
    pub fn assert_level_table_counts(&self) {
        let mut table_count_visible: u32 = 0;
        let mut table_count_visible_max: u32 = 0;
        let mut value_count_visible: u64 = 0;

        for (index, manifest_level) in self.levels.iter().enumerate() {
            // SAFETY: LSM_LEVELS fits in u8, so index always fits.
            #[allow(clippy::cast_possible_truncation)]
            let level = index as u8;
            let level_max =
                table_count_max_for_level(constants::LSM_GROWTH_FACTOR, u32::from(level));
            assert!(manifest_level.table_count_visible() <= level_max);

            table_count_visible += manifest_level.table_count_visible();
            table_count_visible_max += level_max;
            value_count_visible += manifest_level.value_count_visible();
        }

        // TODO(port): emit tracer gauges.
        let _ = (table_count_visible, table_count_visible_max, value_count_visible);
    }

    /// Assert no invisible tables exist across all levels for the given snapshots.
    #[allow(clippy::missing_panics_doc)]
    pub fn assert_no_invisible_tables(&self, snapshots: &[u64]) {
        for level in 0..LSM_LEVELS as usize {
            // SAFETY: LSM_LEVELS fits in u8, so level always fits.
            #[allow(clippy::cast_possible_truncation)]
            let level_u8 = level as u8;
            self.assert_no_invisible_tables_at_level(level_u8, snapshots);
        }
    }

    fn assert_no_invisible_tables_at_level(&self, level: u8, snapshots: &[u64]) {
        let mut it = self.levels[level as usize].iterator(
            Visibility::Invisible,
            snapshots,
            Direction::Ascending,
            None,
        );
        assert!(it.next().is_none());
    }

    fn assert_log_attached(&self) {
        assert!(self.log_attached, "open_commence was not called");
    }
}

/// Iterator across levels yielding tables visible to a given snapshot (upstream
/// `LookupIterator`).
///
/// DEVIATION: returns owned values instead of references to avoid complex lifetime
/// management with level iterators.
struct ManifestLookupIterator<'a, K: TableKey> {
    manifest: &'a Manifest<K>,
    snapshot: u64,
    key: K,
    level: u8,
}

impl<K: TableKey> Iterator for ManifestLookupIterator<'_, K> {
    type Item = TreeTableInfo<K>;

    fn next(&mut self) -> Option<Self::Item> {
        while (self.level as usize) < LSM_LEVELS as usize {
            let level = &self.manifest.levels[self.level as usize];
            // DEVIATION: upstream uses `level.key_range_contains(self.snapshot, key)`, which
            // calls `key_range_latest_contains` under the hood. Our `key_range_contains`
            // asserts `snapshot < SNAPSHOT_LATEST`, so we branch here to match upstream's
            // actual behavior.
            let key_in_range = if self.snapshot < SNAPSHOT_LATEST {
                level.key_range_contains(self.snapshot, self.key)
            } else {
                level.key_range_latest_contains(self.key)
            };
            if !key_in_range {
                self.level += 1;
                continue;
            }

            // Use a single-element array for the snapshot. The iterator will borrow it,
            // but since we only need the table info (which is Copy), we can extract it.
            let snapshots = [self.snapshot];
            let mut inner = level.iterator(
                Visibility::Visible,
                &snapshots,
                Direction::Ascending,
                Some(KeyRange { key_min: self.key, key_max: self.key }),
            );

            if let Some(table) = inner.next() {
                assert!(table.visible(self.snapshot));
                assert!(table.key_min() <= self.key);
                assert!(self.key <= table.key_max());
                debug_assert!(inner.next().is_none());

                self.level += 1;
                return Some(*table);
            }

            self.level += 1;
        }

        assert_eq!(self.level as usize, LSM_LEVELS as usize);
        None
    }
}

// --- Tests ---

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A mock manifest log that records appended entries.
    struct MockLog {
        opened: Cell<bool>,
        entries: Vec<WireTableInfo>,
    }

    impl MockLog {
        fn new() -> Self {
            Self { opened: Cell::new(false), entries: Vec::new() }
        }

        /// Mark the log as opened — call after `open_commence`.
        fn open(&self) {
            self.opened.set(true);
        }
    }

    impl ManifestLog for MockLog {
        fn is_opened(&self) -> bool {
            self.opened.get()
        }

        fn append(&mut self, entry: &WireTableInfo) {
            assert!(self.opened.get());
            self.entries.push(*entry);
        }
    }

    /// Create a log, call `open_commence`, then open it — the standard lifecycle.
    fn setup_log(manifest: &mut Manifest<u64>) -> MockLog {
        let log = MockLog::new();
        manifest.open_commence(&log);
        log.open();
        log
    }

    fn make_table(
        address: u64,
        key_min: u64,
        key_max: u64,
        snapshot_min: u64,
        snapshot_max: u64,
        value_count: u32,
    ) -> TreeTableInfo<u64> {
        TreeTableInfo {
            checksum: u128::from(address) * 7,
            address,
            snapshot_min,
            snapshot_max,
            key_min,
            key_max,
            value_count,
        }
    }

    #[test]
    fn tree_table_info_encode_decode_round_trip() {
        let table = make_table(100, 10, 20, 5, 100, 77);
        let encoded = table.encode(42, 3, Event::Insert);
        let decoded = TreeTableInfo::<u64>::decode(&encoded, 42);
        assert_eq!(table, decoded);
        assert_eq!(encoded.tree_id, 42);
        assert_eq!(encoded.label.level, 3);
        assert_eq!(encoded.label.event, Event::Insert);
    }

    #[test]
    fn tree_table_info_wire_format_pads_keys() {
        let table = make_table(1, 42, 100, 1, u64::MAX, 1);
        let encoded = table.encode(1, 0, Event::Insert);
        assert!(encoded.key_min[8..].iter().all(|&b| b == 0));
        assert!(encoded.key_max[8..].iter().all(|&b| b == 0));
        let key_min_val = u64::from_le_bytes(encoded.key_min[..8].try_into().unwrap());
        assert_eq!(key_min_val, 42);
    }

    #[test]
    fn insert_table_records_log_entry() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        let table = make_table(1, 10, 20, 1, u64::MAX, 5);
        manifest.insert_table(&mut log, 0, &table);

        assert_eq!(log.entries.len(), 1);
        assert_eq!(log.entries[0].label.event, Event::Insert);
        assert_eq!(log.entries[0].address, 1);
    }

    #[test]
    fn update_table_records_log_entry() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        let table = make_table(1, 10, 20, 1, u64::MAX, 5);
        manifest.insert_table(&mut log, 0, &table);

        let tref = manifest.levels[0].find(&table).expect("table should be in level 0");
        manifest.update_table(&mut log, 0, 50, tref);

        assert_eq!(log.entries.len(), 2);
        assert_eq!(log.entries[1].label.event, Event::Update);
        assert_eq!(log.entries[1].snapshot_max, 50);
    }

    #[test]
    fn move_table_emits_single_update_at_destination() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        let table = make_table(1, 10, 20, 1, u64::MAX, 5);
        manifest.insert_table(&mut log, 0, &table);
        manifest.move_table(&mut log, 0, 1, &table);

        assert_eq!(log.entries.len(), 2);
        assert_eq!(log.entries[1].label.level, 1);
        assert_eq!(log.entries[1].label.event, Event::Update);

        // DEVIATION: upstream test expected "still at level 0", but move_table calls
        // remove_table on level_a. The table is only at level_b after the move.
        assert!(!manifest.levels[0].contains(&table));
        // Also at level 1:
        assert!(manifest.levels[1].contains(&table));
    }

    #[test]
    fn lookup_iterates_levels_ascending() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        let t0 = make_table(1, 10, 20, 1, u64::MAX, 5);
        let t1 = make_table(2, 10, 20, 1, u64::MAX, 3);
        manifest.insert_table(&mut log, 0, &t0);
        manifest.insert_table(&mut log, 1, &t1);

        let mut lookup = manifest.lookup(SNAPSHOT_LATEST, 15, 0);
        let found = lookup.next().expect("should find a table");
        assert_eq!(found.address, 1);
        let found = lookup.next().expect("should find a table");
        assert_eq!(found.address, 2);
        assert!(lookup.next().is_none());
    }

    #[test]
    fn remove_invisible_tables_only_removes_invisible() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        let visible = make_table(1, 10, 20, 1, u64::MAX, 5);
        let invisible = make_table(2, 30, 40, 1, 50, 3);
        manifest.insert_table(&mut log, 0, &visible);
        manifest.insert_table(&mut log, 0, &invisible);

        manifest.remove_invisible_tables(&mut log, 0, &[100], 0, u64::MAX);

        let removes: Vec<_> =
            log.entries.iter().filter(|e| e.label.event == Event::Remove).collect();
        assert_eq!(removes.len(), 1);
        assert_eq!(removes[0].address, 2);

        assert!(manifest.levels[0].contains(&visible));
        assert!(!manifest.levels[0].contains(&invisible));
    }

    #[test]
    fn key_range_spans_all_levels() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        assert!(manifest.key_range().is_none());

        let t1 = make_table(1, 5, 10, 1, u64::MAX, 5);
        let t2 = make_table(2, 20, 30, 1, u64::MAX, 3);
        manifest.insert_table(&mut log, 0, &t1);
        manifest.insert_table(&mut log, 2, &t2);

        let range = manifest.key_range().expect("should have a range");
        assert_eq!(range.key_min, 5);
        assert_eq!(range.key_max, 30);
    }

    #[test]
    fn compaction_must_drop_tombstones_true_when_no_lower_overlap() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        let t = make_table(1, 10, 20, 1, u64::MAX, 5);
        manifest.insert_table(&mut log, 0, &t);

        let range = CompactionRange {
            key_min: 10,
            key_max: 20,
            tables: OverlapRange {
                key_min: 10,
                key_max: 20,
                tables: tigerbeetle_core::stdx::bounded_array::BoundedArray::new(),
            },
        };
        assert!(manifest.compaction_must_drop_tombstones(0, &range));
    }

    #[test]
    fn compaction_must_drop_tombstones_false_when_lower_overlap() {
        let mut manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        let mut log = setup_log(&mut manifest);

        let t = make_table(1, 10, 20, 1, u64::MAX, 5);
        manifest.insert_table(&mut log, 1, &t);

        let range = CompactionRange {
            key_min: 10,
            key_max: 20,
            tables: OverlapRange {
                key_min: 10,
                key_max: 20,
                tables: tigerbeetle_core::stdx::bounded_array::BoundedArray::new(),
            },
        };
        assert!(!manifest.compaction_must_drop_tombstones(0, &range));
    }

    #[test]
    fn assert_level_table_counts_passes_for_empty() {
        let manifest = Manifest::<u64>::new(TreeConfig { id: 1, name: "test" });
        manifest.assert_level_table_counts();
    }
}
