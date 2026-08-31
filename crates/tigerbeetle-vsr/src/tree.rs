//! An LSM tree.
//! Port of `src/lsm/tree.zig` (`TreeType`).
//!
//! # Differences from upstream
//!
//! - **No Grid stored:** upstream stores `*Grid`; Rust's borrow checker prevents
//!   self-referential borrows. Methods that need Grid take `grid: &mut Grid` as a parameter.
//! - **No ScratchMemory stored:** upstream stores `*ScratchMemory`; here it is passed per-call
//!   to [`Tree::compact`] and [`Tree::swap_mutable_and_immutable`].
//! - **No Tracer:** `grid.trace.start/stop` calls are deferred until the tracing subsystem
//!   is ported.
//! - **`open_complete` takes `checkpoint_op`:** instead of reaching into
//!   `grid.superblock.working.vsr_state.checkpoint.header.op`, the caller passes the op directly
//!   (Grid does not own SuperBlock in this port).
//! - **`open_table` bypasses ManifestLog:** during replay, tables are inserted directly into
//!   manifest levels without logging, matching upstream's `manifest.levels[l].insert_table()`.
//! - **Compaction stub:** [`Compaction`] is a placeholder until `compaction.zig` is ported.
//! - **Two `TableKey` traits:** lsm's `manifest::TableKey` (for Manifest) and vsr's
//!   `table::TableKey` (for block accessors) are distinct. [`TreeSpec`] requires both.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use tigerbeetle_core::constants::{self, VERIFY};
use tigerbeetle_lsm::manifest::{
    Manifest, ManifestLog as ManifestLogTrait, TableKey as ManifestTableKey, TreeTableInfo,
};
use tigerbeetle_lsm::manifest_level::{KeyRange, LevelTableInfo};
use tigerbeetle_lsm::scratch_memory::ScratchMemory;
use tigerbeetle_lsm::table_memory::{self, Mutability, TableMemory};
use tigerbeetle_lsm::tree::{SNAPSHOT_LATEST, ScopeCloseMode, TreeConfig};

use crate::grid::Grid;
use crate::table::{BlockValue, IndexBlocks, TableKey as VsrTableKey, TableLayout};

/// Generic parameters for a tree's table schema (upstream `TreeTable` comptime param).
///
/// Combines schema accessors with block-level operations needed by
/// [`Tree::lookup_from_levels_cache`].
pub trait TreeSpec:
    table_memory::Table<Key: ManifestTableKey + VsrTableKey, Value: BlockValue>
{
    /// Index block schema (upstream `Table.index`).
    const LAYOUT: TableLayout;

    /// Port of `Table.index_blocks_for_key`.
    fn index_blocks_for_key(index_block: &[u8], key: Self::Key) -> Option<IndexBlocks<Self::Key>>;

    /// Port of `Table.value_block_search`.
    fn value_block_search(value_block: &[u8], key: Self::Key) -> Option<Self::Value>;

    /// Port of `Table.tombstone_from_key`.
    fn tombstone_from_key(key: Self::Key) -> Self::Value;
}

/// Upstream `Tree.Options`.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// The (runtime) upper-limit of values created by a single batch.
    pub batch_value_count_limit: u32,
}

/// Snapshot of mutable state for scope open/close.
#[derive(Clone, Copy, Debug)]
pub struct ActiveScope<K: Copy> {
    pub value_context: table_memory::ValueContext,
    pub key_range: Option<KeyRange<K>>,
}

/// Result of a memory/cache lookup (upstream `Tree.LookupMemoryResult`).
///
/// DEVIATION: upstream returns `positive: *const Value` (pointer). This port returns
/// the value by copy since all lookups are synchronous and values are `Copy`.
#[derive(Clone, Copy, Debug)]
pub enum LookupMemoryResult<V: Copy> {
    Negative,
    Positive(V),
    Possible { level: u8 },
}

/// Cached value block search result (upstream `cached_value_block_search` return).
enum CachedValueSearch<V: Copy> {
    NotFound,
    Found(V),
    Tombstone,
    BlockNotInCache,
}

/// Stub for `CompactionType(Tree, Storage)` — replaced when `compaction.zig` is ported.
///
/// Upstream Compaction is ~2388 lines with deep I/O integration. This stub satisfies
/// Tree's structural needs only.
struct Compaction {
    #[allow(dead_code)]
    level_b: u8,
}

impl Compaction {
    fn new(level_b: u8) -> Self {
        assert!((level_b as usize) < constants::LSM_LEVELS as usize);
        Self { level_b }
    }

    #[allow(clippy::unused_self)]
    fn reset(&mut self) {}

    #[allow(clippy::unused_self)]
    fn assert_between_bars(&self) {}
}

/// An LSM tree: `TreeType(TreeTable, Storage)` in upstream.
///
/// Generic over [`TreeSpec`] which provides the table schema and block-level operations.
pub struct Tree<S: TreeSpec> {
    config: TreeConfig,
    #[allow(dead_code)]
    options: Options,

    table_mutable: TableMemory<S>,
    table_immutable: TableMemory<S>,

    manifest: Manifest<S::Key>,

    /// One compaction per level. + 1 for immutable→L0 is cancelled by −1 since the last
    /// level doesn't compact to anything.
    compactions: [Compaction; constants::LSM_LEVELS as usize],

    /// While a compaction is running, this is the op of the last `compact()`.
    /// While no compaction is running, this is the op of the last `compact()` to complete.
    /// When recovering from a checkpoint, `compaction_op` starts at `checkpoint_op`.
    compaction_op: Option<u64>,

    active_scope: Option<ActiveScope<S::Key>>,

    /// The range of keys in this tree at snapshot_latest.
    key_range: Option<KeyRange<S::Key>>,
}

impl<S: TreeSpec> Tree<S> {
    /// Port of upstream `Tree.init`.
    ///
    /// # Panics
    /// Panics if `config.id == 0` or `config.name` is empty (upstream asserts).
    #[must_use]
    pub fn new(config: TreeConfig, options: Options) -> Self {
        assert!(config.id != 0);
        assert!(!config.name.is_empty());

        let value_count_limit =
            u64::from(options.batch_value_count_limit) * (constants::LSM_COMPACTION_OPS as u64);
        assert!(value_count_limit > 0);
        assert!(value_count_limit <= S::VALUE_COUNT_MAX as u64);

        let value_count_limit = u32::try_from(value_count_limit)
            .unwrap_or_else(|_| unreachable!("value_count_limit fits u32"));

        let table_mutable = TableMemory::new(Mutability::Mutable, config.name, value_count_limit);
        let table_immutable = TableMemory::new(
            Mutability::Immutable(table_memory::ImmutableState::default()),
            config.name,
            value_count_limit,
        );
        let manifest = Manifest::new(config);
        let compactions = core::array::from_fn(|i| Compaction::new(i as u8));

        Self {
            config,
            options,
            table_mutable,
            table_immutable,
            manifest,
            compactions,
            compaction_op: None,
            active_scope: None,
            key_range: None,
        }
    }

    /// Port of upstream `Tree.reset`.
    pub fn reset(&mut self) {
        self.table_mutable.reset();
        self.table_immutable.reset();

        for compaction in &mut self.compactions {
            compaction.reset();
        }

        self.compaction_op = None;
        self.active_scope = None;
        self.key_range = None;
    }

    /// Port of upstream `Tree.scope_open`.
    ///
    /// # Panics
    /// Panics if a scope is already active (upstream asserts).
    pub fn scope_open(&mut self) {
        assert!(self.active_scope.is_none());
        self.active_scope = Some(ActiveScope {
            value_context: *self.table_mutable.value_context(),
            key_range: self.key_range,
        });
    }

    /// Port of upstream `Tree.scope_close`.
    ///
    /// # Panics
    /// Panics if no scope is active (upstream asserts).
    pub fn scope_close(&mut self, mode: ScopeCloseMode) {
        let active_scope =
            self.active_scope.unwrap_or_else(|| unreachable!("scope_close without active scope"));
        assert!(active_scope.value_context.count <= self.table_mutable.count());

        if mode == ScopeCloseMode::Discard {
            self.table_mutable.restore_value_context(active_scope.value_context);
            self.key_range = active_scope.key_range;
        }

        self.active_scope = None;
    }

    /// Port of upstream `Tree.put`.
    pub fn put(&mut self, value: &<S as table_memory::Table>::Value) {
        self.table_mutable.put(value);
    }

    /// Port of upstream `Tree.remove`.
    pub fn remove(&mut self, value: &<S as table_memory::Table>::Value) {
        let key = S::key_from_value(value);
        self.table_mutable.put(&S::tombstone_from_key(key));
    }

    /// Port of upstream `Tree.key_range_update`.
    pub fn key_range_update(&mut self, key: S::Key) {
        if let Some(ref mut key_range) = self.key_range {
            if key < key_range.key_min {
                key_range.key_min = key;
            }
            if key > key_range.key_max {
                key_range.key_max = key;
            }
        } else {
            self.key_range = Some(KeyRange { key_min: key, key_max: key });
        }
    }

    /// Port of upstream `Tree.key_range_contains`.
    ///
    /// Returns `true` if the given key may be present in the Tree, `false` if the key is
    /// guaranteed to not be present.
    ///
    /// # Panics
    /// Panics if `snapshot >= SNAPSHOT_LATEST`.
    #[must_use]
    pub fn key_range_contains(&self, snapshot: u64, key: S::Key) -> bool {
        assert!(snapshot < SNAPSHOT_LATEST);
        self.key_range.is_some_and(|kr| kr.key_min <= key && key <= kr.key_max)
    }

    /// Port of upstream `Tree.lookup_from_memory`.
    ///
    /// This function is intended to never be called by regular code. It only exists for
    /// fuzzing, due to the performance overhead it carries. Real code must rely on the
    /// Groove cache for lookups.
    ///
    /// May return a tombstone.
    ///
    /// DEVIATION: `radix_buffer` is passed as a parameter since sort requires it.
    ///
    /// # Panics
    /// Panics if `VERIFY` is `false` (const assertion).
    #[must_use]
    #[allow(clippy::assertions_on_constants)]
    pub fn lookup_from_memory(
        &mut self,
        key: S::Key,
        radix_buffer: &mut ScratchMemory<S::Value>,
    ) -> Option<&<S as table_memory::Table>::Value> {
        assert!(VERIFY);

        self.table_mutable.sort(radix_buffer);
        self.table_mutable.get(key).or_else(|| self.table_immutable.get(key))
    }

    /// Port of upstream `Tree.lookup_from_levels_cache`.
    ///
    /// Attempts to find a key synchronously from the Grid cache. Returns:
    /// - `.negative` if the key does not exist in the Manifest or a tombstone was found.
    /// - `.positive(value)` if the key exists in the Manifest.
    /// - `.possible` if the key may exist but its existence cannot be ascertained without I/O.
    ///
    /// # Panics
    /// Panics if manifest table assertions fail (key range, visibility).
    #[must_use]
    pub fn lookup_from_levels_cache(
        &mut self,
        grid: &mut Grid,
        snapshot: u64,
        key: S::Key,
    ) -> LookupMemoryResult<S::Value> {
        if let Some(value) = self.table_immutable.get(key) {
            return if S::tombstone(value) {
                LookupMemoryResult::Negative
            } else {
                LookupMemoryResult::Positive(*value)
            };
        }

        let mut level = 0u8;
        // Collect manifest lookup results to release the immutable borrow on self.manifest
        // before calling cached_value_block_search (which needs &mut self).
        let tables: Vec<TreeTableInfo<S::Key>> = self.manifest.lookup(snapshot, key, 0).collect();
        for table in tables {
            assert!(table.visible(snapshot));
            assert!(table.key_min <= key);
            assert!(key <= table.key_max);

            let Some(cache_location) =
                grid.read_block_from_cache(table.address, table.checksum, true)
            else {
                return LookupMemoryResult::Possible { level };
            };

            let index_block = grid.block(cache_location);
            let Some(key_blocks) = S::index_blocks_for_key(index_block, key) else {
                level = level.saturating_add(1);
                continue;
            };

            match Self::cached_value_block_search(
                grid,
                key_blocks.value_block_address,
                key_blocks.value_block_checksum,
                key,
            ) {
                CachedValueSearch::NotFound => {
                    level = level.saturating_add(1);
                }
                CachedValueSearch::Found(value) => {
                    assert!(!S::tombstone(&value));
                    assert_eq!(S::key_from_value(&value), key);
                    return LookupMemoryResult::Positive(value);
                }
                CachedValueSearch::Tombstone => return LookupMemoryResult::Negative,
                CachedValueSearch::BlockNotInCache => {
                    return LookupMemoryResult::Possible { level };
                }
            }
        }

        LookupMemoryResult::Negative
    }

    /// Port of upstream `Tree.block_value_count_max`.
    #[must_use]
    pub fn block_value_count_max(&self) -> u32 {
        S::LAYOUT.block_value_count_max
    }

    /// Port of upstream `Tree.open_commence`.
    ///
    /// Attach the manifest log. Must be called before any mutation.
    ///
    /// # Panics
    /// Panics if `open_commence` was already called on the manifest.
    pub fn open_commence(&mut self, log: &impl ManifestLogTrait) {
        assert!(self.compaction_op.is_none());
        assert!(self.key_range.is_none());
        self.manifest.open_commence(log);
    }

    /// Port of upstream `Tree.open_table`.
    ///
    /// Replays a manifest entry during open. Inserts directly into the manifest level
    /// without logging (matching upstream's `manifest.levels[l].insert_table()`).
    ///
    /// # Panics
    /// Panics if `open_commence` was not called, or if the table's tree_id doesn't match.
    pub fn open_table(&mut self, table: &tigerbeetle_lsm::schema::manifest_node::TableInfo) {
        assert!(self.compaction_op.is_none());
        assert!(self.key_range.is_none());

        let tree_table = TreeTableInfo::decode(table, self.config.id);
        let level = table.label.level;
        self.manifest.levels[level as usize].insert_table(&tree_table);
    }

    /// Port of upstream `Tree.open_complete`.
    ///
    /// DEVIATION: takes `checkpoint_op: u64` as a parameter instead of reaching into
    /// `grid.superblock.working.vsr_state.checkpoint.header.op`.
    ///
    /// # Panics
    /// Panics if `open_commence` was not called.
    pub fn open_complete(&mut self, checkpoint_op: u64) {
        assert!(self.compaction_op.is_none());
        assert!(self.key_range.is_none());

        self.compaction_op = Some(checkpoint_op);
        self.key_range = self.manifest.key_range();

        self.manifest.assert_level_table_counts();
        assert!(
            self.compaction_op.is_some_and(|op| op == 0
                || (op + 1) % (constants::LSM_COMPACTION_OPS as u64) == 0),
            "compaction_op must be 0 or at a bar boundary"
        );
    }

    /// Port of upstream `Tree.compact`.
    ///
    /// Spreads sort+deduplication work between beats, to avoid a latency spike at the end of
    /// each bar (or immediately prior to scans).
    ///
    /// DEVIATION: `radix_buffer` is passed as a parameter instead of stored on Tree.
    pub fn compact(&mut self, radix_buffer: &mut ScratchMemory<S::Value>) {
        // TODO(port): grid.trace.start/stop — tracer not yet ported.
        self.table_mutable.sort_suffix(radix_buffer);
    }

    /// Called after the last beat of a full compaction bar, by the compaction instance.
    ///
    /// # Panics
    /// Panics if tables are not in the expected mutability state, or if snapshot_min is invalid.
    ///
    /// DEVIATION: `radix_buffer` is passed as a parameter instead of stored on Tree.
    pub fn swap_mutable_and_immutable(
        &mut self,
        snapshot_min: u64,
        radix_buffer: &mut ScratchMemory<S::Value>,
    ) {
        assert!(matches!(self.table_mutable.mutability(), Mutability::Mutable));
        assert!(matches!(self.table_immutable.mutability(), Mutability::Immutable(_)));
        assert!(snapshot_min > 0);
        assert!(snapshot_min < SNAPSHOT_LATEST);

        // TODO(port): grid.trace.start/stop — tracer not yet ported.

        let immutable_flushed = matches!(
            self.table_immutable.mutability(),
            Mutability::Immutable(state) if state.flushed
        );

        if immutable_flushed {
            self.table_immutable.compact(&mut self.table_mutable, snapshot_min);
        } else {
            assert!(
                self.table_immutable.count() + self.table_mutable.count()
                    <= self.table_immutable.values_capacity() as u32
            );
            self.table_immutable.absorb(&mut self.table_mutable, snapshot_min, radix_buffer);
            assert_eq!(self.table_mutable.count(), 0);
        }

        assert_eq!(self.table_mutable.count(), 0);
        assert!(matches!(self.table_mutable.mutability(), Mutability::Mutable));
        assert!(matches!(self.table_immutable.mutability(), Mutability::Immutable(_)));
    }

    /// Port of upstream `Tree.assert_between_bars`.
    #[allow(clippy::missing_panics_doc)]
    pub fn assert_between_bars(&self) {
        for compaction in &self.compactions {
            compaction.assert_between_bars();
        }

        self.manifest.assert_level_table_counts();

        if VERIFY {
            self.manifest.assert_no_invisible_tables(&[]);
        }
    }

    // -- Accessors for Compaction lifecycle methods --

    /// Immutable table reference (read-only).
    #[must_use]
    pub fn table_immutable_ref(&self) -> &table_memory::TableMemory<S> {
        &self.table_immutable
    }

    /// Mutable table reference (read-only).
    #[must_use]
    pub fn table_mutable_ref(&self) -> &table_memory::TableMemory<S> {
        &self.table_mutable
    }

    /// Immutable table reference (mutable, for `set_flushed`).
    #[must_use]
    pub fn table_immutable_mut(&mut self) -> &mut table_memory::TableMemory<S> {
        &mut self.table_immutable
    }

    /// Manifest reference (read-only).
    #[must_use]
    pub fn manifest_ref(&self) -> &Manifest<S::Key> {
        &self.manifest
    }

    /// Manifest reference (mutable, for `update_table`/`insert_table`/etc.).
    #[must_use]
    pub fn manifest_mut(&mut self) -> &mut Manifest<S::Key> {
        &mut self.manifest
    }

    /// Tree config reference.
    #[must_use]
    pub fn config_ref(&self) -> &TreeConfig {
        &self.config
    }

    /// Returns null if the value is null or a tombstone, otherwise returns the value.
    ///
    /// We use tombstone values internally, but expose them as null to the user.
    /// This distinction enables us to cache a null result as a tombstone in our hash maps.
    #[inline]
    #[must_use]
    pub fn unwrap_tombstone(
        value: Option<&<S as table_memory::Table>::Value>,
    ) -> Option<&<S as table_memory::Table>::Value> {
        value.filter(|v| !S::tombstone(v))
    }

    fn cached_value_block_search(
        grid: &mut Grid,
        address: u64,
        checksum: u128,
        key: S::Key,
    ) -> CachedValueSearch<S::Value> {
        let Some(cache_location) = grid.read_block_from_cache(address, checksum, true) else {
            return CachedValueSearch::BlockNotInCache;
        };

        let value_block = grid.block(cache_location);
        match S::value_block_search(value_block, key) {
            Some(value) => {
                if S::tombstone(&value) {
                    CachedValueSearch::Tombstone
                } else {
                    CachedValueSearch::Found(value)
                }
            }
            None => CachedValueSearch::NotFound,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use tigerbeetle_lsm::table_memory::Usage;

    /// A minimal TreeSpec implementation for testing.
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

    impl table_memory::Table for TestSpec {
        type Key = TestKey;
        type Value = TestValue;

        const VALUE_COUNT_MAX: usize = 128;
        const USAGE: Usage = Usage::General;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            value.key
        }

        fn tombstone(value: &Self::Value) -> bool {
            value.key.0 == u64::MAX && value.data == 0
        }
    }

    impl TreeSpec for TestSpec {
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
            _index_block: &[u8],
            _key: Self::Key,
        ) -> Option<IndexBlocks<Self::Key>> {
            None
        }

        fn value_block_search(_value_block: &[u8], _key: Self::Key) -> Option<Self::Value> {
            None
        }

        fn tombstone_from_key(key: Self::Key) -> Self::Value {
            TestValue { key, data: 0 }
        }
    }

    #[test]
    fn tree_new_and_put() {
        let config = TreeConfig { id: 1, name: "test" };
        let options = Options { batch_value_count_limit: 32 };
        let mut tree = Tree::<TestSpec>::new(config, options);
        assert_eq!(tree.table_mutable.count(), 0);

        let value = TestValue { key: TestKey(42), data: 100 };
        tree.put(&value);
        assert_eq!(tree.table_mutable.count(), 1);
    }

    #[test]
    fn tree_scope_open_close_persist() {
        let config = TreeConfig { id: 1, name: "test" };
        let options = Options { batch_value_count_limit: 32 };
        let mut tree = Tree::<TestSpec>::new(config, options);

        tree.scope_open();
        assert!(tree.active_scope.is_some());

        let value = TestValue { key: TestKey(1), data: 10 };
        tree.put(&value);
        assert_eq!(tree.table_mutable.count(), 1);

        tree.scope_close(ScopeCloseMode::Persist);
        assert!(tree.active_scope.is_none());
        assert_eq!(tree.table_mutable.count(), 1);
    }

    #[test]
    fn tree_scope_open_close_discard() {
        let config = TreeConfig { id: 1, name: "test" };
        let options = Options { batch_value_count_limit: 32 };
        let mut tree = Tree::<TestSpec>::new(config, options);

        tree.scope_open();
        let value = TestValue { key: TestKey(1), data: 10 };
        tree.put(&value);
        assert_eq!(tree.table_mutable.count(), 1);

        tree.scope_close(ScopeCloseMode::Discard);
        assert_eq!(tree.table_mutable.count(), 0);
    }

    #[test]
    fn tree_key_range_update_and_contains() {
        let config = TreeConfig { id: 1, name: "test" };
        let options = Options { batch_value_count_limit: 32 };
        let mut tree = Tree::<TestSpec>::new(config, options);

        assert!(!tree.key_range_contains(0, TestKey(5)));

        tree.key_range_update(TestKey(10));
        tree.key_range_update(TestKey(20));
        tree.key_range_update(TestKey(5));

        assert!(tree.key_range_contains(0, TestKey(5)));
        assert!(tree.key_range_contains(0, TestKey(15)));
        assert!(tree.key_range_contains(0, TestKey(20)));
        assert!(!tree.key_range_contains(0, TestKey(4)));
        assert!(!tree.key_range_contains(0, TestKey(21)));
    }

    #[test]
    fn tree_remove_inserts_tombstone() {
        let config = TreeConfig { id: 1, name: "test" };
        let options = Options { batch_value_count_limit: 32 };
        let mut tree = Tree::<TestSpec>::new(config, options);

        let value = TestValue { key: TestKey(42), data: 100 };
        tree.put(&value);
        assert_eq!(tree.table_mutable.count(), 1);

        tree.remove(&value);
        assert_eq!(tree.table_mutable.count(), 2);
    }

    #[test]
    fn tree_unwrap_tombstone() {
        let value = TestValue { key: TestKey(42), data: 100 };
        let tombstone = TestValue { key: TestKey(u64::MAX), data: 0 };

        assert!(Tree::<TestSpec>::unwrap_tombstone(Some(&value)).is_some());
        assert!(Tree::<TestSpec>::unwrap_tombstone(Some(&tombstone)).is_none());
        assert!(Tree::<TestSpec>::unwrap_tombstone(None).is_none());
    }

    #[test]
    fn tree_block_value_count_max() {
        let config = TreeConfig { id: 1, name: "test" };
        let options = Options { batch_value_count_limit: 32 };
        let tree = Tree::<TestSpec>::new(config, options);
        assert_eq!(tree.block_value_count_max(), 128);
    }
}
