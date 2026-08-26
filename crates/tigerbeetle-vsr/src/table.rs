//! Port of `src/lsm/table.zig` — block layout, builder, and index/value accessors.
//!
//! A table is a set of blocks:
//!
//! * Index block (exactly 1)
//! * Value blocks (at least one, at most `value_block_count_max`) store the actual keys/values.
//!
//! # Differences from upstream
//!
//! - **No raw pointers:** blocks are `&[u8]`/`&mut [u8]` slices instead of
//!   `*align(sector_size) [block_size]u8` (`BlockPtr`/`BlockPtrConst`).
//! - **Trait-based generics:** upstream's `TableType(comptime …)` becomes the
//!   [`TableSpec`] trait; layout, builder, and accessors are generic over it.
//! - **No Tracer:** tracer gauge calls deferred.
//! - **Builder takes blocks by reference:** upstream's `Builder` stores raw pointers;
//!   safe-Rust borrows prevent this, so methods take block slices as parameters.
//! - **No `constants.verify` deep-checks:** verification is conditional in upstream
//!   under `constants.verify`; this port always asserts (matching the test-min config).
//! - **No external byte-cast crates:** upstream is deliberately dependency-free;
//!   key/value byte conversion goes through [`TableKey`] / [`TableSpec`] methods.

use core::fmt::Debug;

use tigerbeetle_core::constants;

use crate::message_header::{self, BlockType, TypedHeader};
use crate::multiversion::Release;
use crate::schema::{self, TableIndex, TableValue};

use tigerbeetle_lsm::binary_search;

/// Checksum storage size in the index block (32 bytes per entry, u128 in first 16).
const CHECKSUM_SIZE: usize = 32;

/// Address storage size in the index block (8 bytes per entry, u64).
const ADDRESS_SIZE: usize = core::mem::size_of::<u64>();

/// Upstream `TableUsage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableUsage {
    /// General purpose table.
    General,
    /// Secondary index: insert/remove pairs can be immediately cancelled.
    SecondaryIndex,
}

/// Generic parameters for a table type (upstream `TableType` comptime parameters).
pub trait TableSpec: 'static {
    /// The key type — must be a fixed-width integer (u64, u128, or a composite).
    type Key: TableKey;

    /// The value type stored in value blocks.
    type Value: BlockValue;

    /// Returns the key for a value. For example, given `object` returns `object.id`.
    fn key_from_value(value: &Self::Value) -> Self::Key;

    /// Must compare greater than all other keys.
    const SENTINEL_KEY: Self::Key;

    /// Returns whether a value is a tombstone value.
    fn tombstone(value: &Self::Value) -> bool;

    /// Returns a tombstone value representation for a key.
    fn tombstone_from_key(key: Self::Key) -> Self::Value;

    /// The maximum number of values per table.
    const VALUE_COUNT_MAX: usize;

    /// The table's intended usage pattern.
    const USAGE: TableUsage;
}

// ---------------------------------------------------------------------------
// Layout computation
// ---------------------------------------------------------------------------

/// Computed layout constants for a table (upstream `Table.layout`).
#[derive(Clone, Copy, Debug)]
pub struct TableLayout {
    /// The maximum number of values in a single value block.
    pub block_value_count_max: u32,
    /// The maximum number of value blocks in the table.
    pub value_block_count_max: u32,
    /// Index block schema.
    pub index: TableIndex,
    /// Value block schema (for a single value block).
    pub data: TableValue,
}

impl TableLayout {
    /// Compute layout for the given key/value sizes and value count maximum.
    ///
    /// # Panics
    ///
    /// Panics if the layout exceeds a single block (upstream asserts the same).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // constants fit u32; upstream uses comptime
    pub const fn compute(key_size: u32, value_size: u32, value_count_max: u32) -> Self {
        assert!(key_size > 0);
        assert!(key_size <= 32);
        assert!(key_size >= 8);
        assert!(value_size > 0);
        assert!(value_count_max > 0);

        let block_body_size = (constants::BLOCK_SIZE - constants::HEADER_SIZE) as u32;
        assert!(block_body_size >= value_size);

        let block_value_count_max = block_body_size / value_size;
        assert!(block_value_count_max > 0);

        let value_blocks = value_count_max.div_ceil(block_value_count_max);
        assert!(value_blocks >= 1);
        assert!(value_blocks <= constants::LSM_TABLE_VALUE_BLOCKS_MAX as u32);

        let index = TableIndex::init(key_size, value_blocks);
        let data = TableValue::init(block_value_count_max, value_size);

        Self { block_value_count_max, value_block_count_max: value_blocks, index, data }
    }

    /// Compute layout from a [`TableSpec`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Key/Value sizes are ≤32; VALUE_COUNT_MAX fits u32
    pub fn compute_for<S: TableSpec>() -> Self {
        Self::compute(
            core::mem::size_of::<S::Key>() as u32,
            core::mem::size_of::<S::Value>() as u32,
            S::VALUE_COUNT_MAX as u32,
        )
    }

    /// The maximum number of blocks (1 index + value blocks).
    #[must_use]
    pub const fn block_count_max(&self) -> u32 {
        1 + self.value_block_count_max
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// State of the [`TableBuilder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuilderState {
    /// No blocks set.
    NoBlocks,
    /// Index block is set, waiting for value block.
    IndexBlock,
    /// Both index and value blocks are set.
    IndexAndValueBlock,
}

/// Options for finishing a value block (upstream `DataFinishOptions`).
#[derive(Clone, Copy, Debug)]
pub struct DataFinishOptions {
    pub cluster: u128,
    pub release: Release,
    pub address: u64,
    pub snapshot_min: u64,
    pub tree_id: u16,
}

/// Options for finishing the index block (upstream `IndexFinishOptions`).
#[derive(Clone, Copy, Debug)]
pub struct IndexFinishOptions {
    pub cluster: u128,
    pub release: Release,
    pub address: u64,
    pub snapshot_min: u64,
    pub tree_id: u16,
}

/// In-memory table metadata returned by [`TableBuilder::index_block_finish`]
/// (upstream `TreeTableInfo`).
#[derive(Clone, Copy, Debug)]
pub struct TableInfo<K> {
    pub checksum: u128,
    pub address: u64,
    pub snapshot_min: u64,
    pub snapshot_max: u64,
    pub key_min: K,
    pub key_max: K,
    pub value_count: u32,
}

/// Builder for constructing index and value blocks (upstream `TableType.Builder`).
///
/// DEVIATION: upstream stores raw block pointers; this port takes block references
/// as method parameters to satisfy safe-Rust aliasing rules.
pub struct TableBuilder {
    key_min: Option<u128>,
    key_max: Option<u128>,
    value_block_count: u32,
    value_count: u32,
    value_count_total: u32,
    state: BuilderState,
}

impl TableBuilder {
    /// Create a new builder in the initial state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            key_min: None,
            key_max: None,
            value_block_count: 0,
            value_count: 0,
            value_count_total: 0,
            state: BuilderState::NoBlocks,
        }
    }

    /// Set the index block. Must be called first (upstream `set_index_block`).
    ///
    /// # Panics
    ///
    /// Panics if called out of order.
    pub fn set_index_block(&mut self, _index_block: &mut [u8]) {
        assert_eq!(self.state, BuilderState::NoBlocks);
        assert_eq!(self.value_block_count, 0);
        assert_eq!(self.value_count, 0);
        assert_eq!(self.value_count_total, 0);
        self.state = BuilderState::IndexBlock;
    }

    /// Set the value block for the current value block. Must be called after
    /// [`set_index_block`](Self::set_index_block) or after finishing a value block.
    ///
    /// # Panics
    ///
    /// Panics if called out of order.
    pub fn set_value_block(&mut self, _value_block: &mut [u8]) {
        assert_eq!(self.state, BuilderState::IndexBlock);
        assert_eq!(self.value_count, 0);
        self.state = BuilderState::IndexAndValueBlock;
    }

    /// Whether the current value block is empty (upstream `value_block_empty`).
    #[must_use]
    pub fn value_block_empty(&self) -> bool {
        self.value_count == 0
    }

    /// Whether the current value block is full (upstream `value_block_full`).
    ///
    /// # Panics
    ///
    /// Panics if no value block is set.
    #[must_use]
    pub fn value_block_full(&self, layout: &TableLayout) -> bool {
        assert_eq!(self.state, BuilderState::IndexAndValueBlock);
        assert!(self.value_count <= layout.data.value_count_max);
        self.value_count == layout.data.value_count_max
    }

    /// Finish the current value block: write the header, update the index block,
    /// and advance the builder state.
    ///
    /// # Panics
    ///
    /// Panics if called out of order, or if the value block is empty.
    #[allow(clippy::cast_possible_truncation)] // block ≤ BLOCK_SIZE ≤ u32::MAX
    pub fn value_block_finish<S: TableSpec>(
        &mut self,
        value_block: &mut [u8],
        index_block: &mut [u8],
        layout: &TableLayout,
        options: DataFinishOptions,
    ) {
        assert_eq!(self.state, BuilderState::IndexAndValueBlock);
        assert!(options.address > 0);
        assert!(self.value_count > 0);

        let header_size = constants::HEADER_SIZE;
        let value_size = core::mem::size_of::<S::Value>();
        let body_size = self.value_count as usize * value_size;
        let total_size = header_size + body_size;
        let block_size = value_block.len();

        // Write the value block header:
        let mut header = message_header::Block::default();
        header.cluster = options.cluster;
        header.address = options.address;
        header.snapshot = options.snapshot_min;
        header.size = total_size as u32;
        header.release = options.release;
        header.block_type_ordinal = BlockType::Value as u8;

        let metadata = schema::TableValueMetadata {
            value_count_max: layout.data.value_count_max,
            value_count: self.value_count,
            value_size: layout.data.value_size,
            tree_id: options.tree_id,
        };
        header.metadata_bytes = metadata.to_wire();

        header.set_checksum_body(&value_block[header_size..total_size]);
        header.set_checksum();

        let wire = header.to_wire();
        value_block[..header_size].copy_from_slice(&wire);

        // Read values from the block for key extraction and verification.
        // Extract into locals before mutating self (borrow checker).
        let used_count = self.value_count as usize;
        let (first_value, last_value) = {
            let values = read_values_from_block::<S::Value>(value_block, &layout.data, used_count);

            // Upstream verify: values are strictly sorted by key.
            if values.len() > 1 {
                for pair in values.windows(2) {
                    assert!(
                        S::key_from_value(&pair[0]) < S::key_from_value(&pair[1]),
                        "value block values not strictly sorted"
                    );
                }
            }

            assert_eq!(used_count, values.len());
            assert_eq!(
                block_size - total_size,
                (layout.data.value_count_max as usize - values.len()) * value_size
                    + layout.data.padding_size as usize
            );

            (values[0], values[values.len() - 1])
        };

        let key_min = S::key_from_value(&first_value);
        let key_max = if used_count == 1 {
            key_min
        } else {
            let key = S::key_from_value(&last_value);
            assert!(key_min < key);
            key
        };

        let current = self.value_block_count as usize;

        // Write to the index block.
        {
            let key_min_bytes = key_min.to_le_bytes_padded();
            let key_max_bytes = key_max.to_le_bytes_padded();
            let key_size = layout.index.key_size as usize;

            let km_offset = layout.index.keys_min_offset as usize + current * key_size;
            index_block[km_offset..km_offset + key_size]
                .copy_from_slice(&key_min_bytes[..key_size]);

            let kx_offset = layout.index.keys_max_offset as usize + current * key_size;
            index_block[kx_offset..kx_offset + key_size]
                .copy_from_slice(&key_max_bytes[..key_size]);

            let addr_offset = layout.index.value_addresses_offset as usize + current * ADDRESS_SIZE;
            index_block[addr_offset..addr_offset + ADDRESS_SIZE]
                .copy_from_slice(&options.address.to_le_bytes());

            let cs_offset = layout.index.value_checksums_offset as usize + current * CHECKSUM_SIZE;
            index_block[cs_offset..cs_offset + CHECKSUM_SIZE].fill(0);
            index_block[cs_offset..cs_offset + 16].copy_from_slice(&header.checksum.to_le_bytes());
        }

        if current == 0 {
            self.key_min = Some(key_min.to_u128());
        }
        self.key_max = Some(key_max.to_u128());

        if current == 0 && used_count == 1 {
            assert_eq!(self.key_min, self.key_max);
        } else {
            assert!(self.key_min < self.key_max);
        }
        assert!(key_max < S::SENTINEL_KEY);

        if current > 0 {
            // Read previous key_max directly from the index block without going through
            // layout.index.key_max() which validates the header (which isn't written yet).
            let key_size = layout.index.key_size as usize;
            let prev_offset = layout.index.keys_max_offset as usize + (current - 1) * key_size;
            let mut prev_key_max_raw = [0u8; 32];
            prev_key_max_raw[..key_size]
                .copy_from_slice(&index_block[prev_offset..prev_offset + key_size]);
            let prev_key_max = S::Key::from_le_bytes_padded(&prev_key_max_raw);
            assert!(prev_key_max < S::key_from_value(&first_value));
        }

        self.value_block_count += 1;
        self.value_count_total += self.value_count;
        self.value_count = 0;
        self.state = BuilderState::IndexBlock;
    }

    /// Whether the index block is empty (upstream `index_block_empty`).
    #[must_use]
    pub fn index_block_empty(&self) -> bool {
        self.value_block_count == 0
    }

    /// Whether the index block is full (upstream `index_block_full`).
    #[must_use]
    pub fn index_block_full(&self, layout: &TableLayout) -> bool {
        self.value_block_count == layout.value_block_count_max
    }

    /// Finish the index block: write the header and return table metadata.
    ///
    /// # Panics
    ///
    /// Panics if called out of order, or if no value blocks have been finished.
    pub fn index_block_finish<K: TableKey>(
        &mut self,
        index_block: &mut [u8],
        layout: &TableLayout,
        options: IndexFinishOptions,
    ) -> TableInfo<K> {
        assert_eq!(self.state, BuilderState::IndexBlock);
        assert!(options.address > 0);
        assert!(self.value_block_empty());
        assert!(self.value_block_count > 0);
        assert_eq!(self.value_count, 0);

        let header_size = constants::HEADER_SIZE;
        let mut header = message_header::Block::default();
        header.cluster = options.cluster;
        header.address = options.address;
        header.snapshot = options.snapshot_min;
        header.size = layout.index.size;
        header.release = options.release;
        header.block_type_ordinal = BlockType::Index as u8;

        let metadata = schema::TableIndexMetadata {
            value_block_count: self.value_block_count,
            value_block_count_max: layout.index.value_block_count_max,
            key_size: layout.index.key_size,
            tree_id: options.tree_id,
        };
        header.metadata_bytes = metadata.to_wire();

        // Write header BEFORE padding — padding() internally calls header_from_block()
        // via value_blocks_used() → metadata(), so the header must be present.
        header.set_checksum_body(&index_block[header_size..layout.index.size as usize]);
        header.set_checksum();

        let wire = header.to_wire();
        index_block[..header_size].copy_from_slice(&wire);

        // Zero padding areas (now that the header is written):
        for (start, end) in layout.index.padding(index_block) {
            index_block[start..end].fill(0);
        }

        let key_size = layout.index.key_size as usize;

        let key_min_bytes = layout.index.key_min(index_block, 0);
        let mut key_min_raw = [0u8; 32];
        key_min_raw[..key_size].copy_from_slice(key_min_bytes);
        let key_min = K::from_le_bytes_padded(&key_min_raw);

        let last_block = self.value_block_count as usize - 1;
        let key_max_bytes = layout.index.key_max(index_block, last_block);
        let mut key_max_raw = [0u8; 32];
        key_max_raw[..key_size].copy_from_slice(key_max_bytes);
        let key_max = K::from_le_bytes_padded(&key_max_raw);

        let info = TableInfo {
            checksum: header.checksum,
            address: options.address,
            snapshot_min: options.snapshot_min,
            snapshot_max: u64::MAX,
            key_min,
            key_max,
            value_count: self.value_count_total,
        };

        *self = Self::new();

        info
    }

    /// Add a value to the current value block.
    ///
    /// # Panics
    ///
    /// Panics if no value block is set, or the block is already full.
    pub fn insert_value<S: TableSpec>(
        &mut self,
        value: &S::Value,
        value_block: &mut [u8],
        layout: &TableLayout,
    ) {
        assert_eq!(self.state, BuilderState::IndexAndValueBlock);
        assert!(!self.value_block_full(layout));

        let value_size = core::mem::size_of::<S::Value>();
        let offset = layout.data.values_offset as usize + self.value_count as usize * value_size;
        let mut buf = [0u8; 32];
        S::Value::write_bytes(value, &mut buf[..value_size]);
        value_block[offset..offset + value_size].copy_from_slice(&buf[..value_size]);
        self.value_count += 1;
    }
}

impl Default for TableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Key trait — encode/decode keys for index block storage
// ---------------------------------------------------------------------------

/// Trait for keys that can be stored in index blocks.
///
/// DEVIATION: upstream uses `mem.bytesAsValue` to reinterpret key bytes. We use
/// explicit encode/decode through this trait instead of `unsafe`.
pub trait TableKey: Copy + Ord + Debug {
    /// Convert to a 32-byte padded little-endian representation.
    fn to_le_bytes_padded(self) -> [u8; 32];

    /// Decode from a 32-byte padded little-endian representation.
    fn from_le_bytes_padded(bytes: &[u8; 32]) -> Self;

    /// Convert to u128 for storage in builder state (key_min/key_max tracking).
    fn to_u128(self) -> u128;

    /// Sentinel key — must compare greater than all other keys.
    const SENTINEL_KEY: Self;
}

impl TableKey for u64 {
    fn to_le_bytes_padded(self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[..8].copy_from_slice(&self.to_le_bytes());
        buf
    }

    fn from_le_bytes_padded(bytes: &[u8; 32]) -> Self {
        let mut key_bytes = [0u8; 8];
        key_bytes.copy_from_slice(&bytes[..8]);
        Self::from_le_bytes(key_bytes)
    }

    fn to_u128(self) -> u128 {
        u128::from(self)
    }

    const SENTINEL_KEY: Self = Self::MAX;
}

impl TableKey for u128 {
    fn to_le_bytes_padded(self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(&self.to_le_bytes());
        buf
    }

    fn from_le_bytes_padded(bytes: &[u8; 32]) -> Self {
        let mut key_bytes = [0u8; 16];
        key_bytes.copy_from_slice(&bytes[..16]);
        Self::from_le_bytes(key_bytes)
    }

    fn to_u128(self) -> u128 {
        self
    }

    const SENTINEL_KEY: Self = Self::MAX;
}

impl TableKey for tigerbeetle_lsm::composite_key::U256 {
    fn to_le_bytes_padded(self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(&self.hi().to_le_bytes());
        buf[16..24].copy_from_slice(&self.low_word().to_le_bytes());
        buf
    }

    fn from_le_bytes_padded(bytes: &[u8; 32]) -> Self {
        let mut hi_bytes = [0u8; 16];
        hi_bytes.copy_from_slice(&bytes[..16]);
        let mut lo_bytes = [0u8; 8];
        lo_bytes.copy_from_slice(&bytes[16..24]);
        Self::from_parts(u128::from_le_bytes(hi_bytes), u64::from_le_bytes(lo_bytes))
    }

    fn to_u128(self) -> u128 {
        self.hi()
    }

    const SENTINEL_KEY: Self = Self::MAX;
}

/// Trait for values stored in value blocks.
///
/// DEVIATION: upstream uses `mem.bytesAsValue`/`mem.bytesAsSlice` pointer casts.
/// This trait provides safe byte-level conversion instead.
///
/// Named `BlockValue` to avoid collision with [`schema::TableValue`] (the layout struct).
pub trait BlockValue: Copy + Debug {
    /// Write the value as little-endian bytes into `buf[0..size_of::<Self>()]`.
    ///
    /// # Panics
    ///
    /// Panics if `buf` is shorter than `size_of::<Self>()`.
    fn write_bytes(&self, buf: &mut [u8]);

    /// Read the value from little-endian bytes.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is shorter than `size_of::<Self>()`.
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl BlockValue for u64 {
    fn write_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[..8]);
        Self::from_le_bytes(buf)
    }
}

impl BlockValue for u128 {
    fn write_bytes(&self, buf: &mut [u8]) {
        buf[..16].copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&bytes[..16]);
        Self::from_le_bytes(buf)
    }
}

// ---------------------------------------------------------------------------
// Index block accessors
// ---------------------------------------------------------------------------

/// Returns the zero-based index of the value block that may contain `key`,
/// or `None` if the key is not contained in the index block's key range.
///
/// May be called on an index block only when the key is in range of the table.
///
/// # Panics
///
/// Panics if the key block data is malformed.
#[must_use]
pub fn index_value_block_for_key<K: TableKey>(
    index_block: &[u8],
    layout: &TableIndex,
    key: K,
) -> Option<u32> {
    let used = layout.value_blocks_used(index_block);
    let key_size = layout.key_size as usize;
    let count = used as usize;

    let keys_max_used = extract_keys::<K>(index_block, layout.keys_max_offset, key_size, count);

    let value_block_index = binary_search::binary_search_keys_upsert_index(
        &keys_max_used,
        key,
        binary_search::Config::default(),
    );
    assert!(value_block_index < used);

    let keys_min_used = extract_keys::<K>(index_block, layout.keys_min_offset, key_size, count);
    let block_key_min = keys_min_used[value_block_index as usize];
    if key < block_key_min { None } else { Some(value_block_index) }
}

/// Returns all data stored in the index block relating to a given key,
/// or `None` if the key is not contained in the index block's key range.
///
/// May be called on an index block only when the key is in range of the table.
#[must_use]
pub fn index_blocks_for_key<K: TableKey>(
    index_block: &[u8],
    layout: &TableIndex,
    key: K,
) -> Option<IndexBlocks<K>> {
    index_value_block_for_key(index_block, layout, key).map(|i| {
        let key_size = layout.key_size as usize;
        let count = layout.value_blocks_used(index_block) as usize;

        let keys_min = extract_keys::<K>(index_block, layout.keys_min_offset, key_size, count);
        let keys_max = extract_keys::<K>(index_block, layout.keys_max_offset, key_size, count);

        IndexBlocks {
            value_block_address: layout.value_address(index_block, i as usize),
            value_block_checksum: layout.value_checksum(index_block, i as usize),
            value_block_key_min: keys_min[i as usize],
            value_block_key_max: keys_max[i as usize],
        }
    })
}

/// Data returned by [`index_blocks_for_key`] (upstream `IndexBlocks`).
#[derive(Clone, Copy, Debug)]
pub struct IndexBlocks<K> {
    pub value_block_address: u64,
    pub value_block_checksum: u128,
    pub value_block_key_min: K,
    pub value_block_key_max: K,
}

/// The block address from a block header.
///
/// # Panics
///
/// Panics if the block header is invalid or the address is zero.
#[must_use]
pub fn block_address(block: &[u8]) -> u64 {
    let header = schema::header_from_block(block);
    assert!(header.address > 0);
    header.address
}

/// Search for a key within a value block using binary search.
///
/// Returns a copy of the value if found, or `None`.
///
/// # Panics
///
/// Panics if the block is invalid.
#[must_use]
pub fn value_block_search<S: TableSpec>(
    value_block: &[u8],
    layout: &TableValue,
    key: S::Key,
) -> Option<S::Value> {
    let used_len = layout.block_values_used_bytes_len(value_block);
    let value_size = core::mem::size_of::<S::Value>();
    assert!(value_size > 0);
    let count = used_len / value_size;
    assert!(count > 0);

    let values = read_values_from_block::<S::Value>(value_block, layout, count);

    binary_search::binary_search_values(
        &|v: &S::Value| S::key_from_value(v),
        &values,
        key,
        binary_search::Config::default(),
    )
    .copied()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract `count` keys of type `K` from an index block at the given offset.
fn extract_keys<K: TableKey>(block: &[u8], offset: u32, key_size: usize, count: usize) -> Vec<K> {
    let offset = offset as usize;
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * key_size;
        let mut raw = [0u8; 32];
        raw[..key_size].copy_from_slice(&block[start..start + key_size]);
        keys.push(K::from_le_bytes_padded(&raw));
    }
    keys
}

/// Read `count` values from a value block using the layout.
fn read_values_from_block<V: BlockValue>(
    block: &[u8],
    layout: &TableValue,
    count: usize,
) -> Vec<V> {
    let value_size = core::mem::size_of::<V>();
    let offset = layout.values_offset as usize;
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * value_size;
        values.push(V::from_bytes(&block[start..start + value_size]));
    }
    values
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A simple table spec for testing: u64 keys, u64 values (identity mapping).
    struct TestTable;

    impl TableSpec for TestTable {
        type Key = u64;
        type Value = u64;

        fn key_from_value(value: &u64) -> u64 {
            *value
        }

        const SENTINEL_KEY: u64 = u64::MAX;

        fn tombstone(value: &u64) -> bool {
            *value == u64::MAX
        }

        fn tombstone_from_key(_key: u64) -> u64 {
            u64::MAX
        }

        const VALUE_COUNT_MAX: usize = 128;
        const USAGE: TableUsage = TableUsage::General;
    }

    #[test]
    fn layout_computation_basic() {
        let layout = TableLayout::compute_for::<TestTable>();
        assert!(layout.value_block_count_max >= 1);
        assert!(layout.block_count_max() >= 2);
        assert_eq!(layout.index.key_size, 8);
        assert_eq!(layout.data.value_size, 8);
    }

    #[test]
    fn layout_computation_32_byte_key_value() {
        let layout = TableLayout::compute(32, 32, 1);
        assert!(layout.value_block_count_max >= 1);
        assert!(layout.block_count_max() >= 2);
        assert_eq!(layout.index.key_size, 32);
        assert_eq!(layout.data.value_size, 32);
    }

    #[test]
    fn table_key_u64_round_trip() {
        let key: u64 = 42;
        let padded = key.to_le_bytes_padded();
        assert_eq!(u64::from_le_bytes_padded(&padded), key);
        assert!(padded[8..].iter().all(|&b| b == 0));
    }

    #[test]
    fn table_key_u128_round_trip() {
        let key: u128 = 0xdead_beef_cafe_babe_1234_5678_9abc_def0;
        let padded = key.to_le_bytes_padded();
        assert_eq!(u128::from_le_bytes_padded(&padded), key);
        assert!(padded[16..].iter().all(|&b| b == 0));
    }

    #[test]
    fn builder_new_is_default() {
        let builder = TableBuilder::new();
        let builder2 = TableBuilder::default();
        assert_eq!(builder.value_block_count, builder2.value_block_count);
        assert_eq!(builder.state, builder2.state);
    }

    #[test]
    fn builder_insert_and_finish_value_block() {
        let layout = TableLayout::compute_for::<TestTable>();
        let mut index_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_block = vec![0u8; constants::BLOCK_SIZE];

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);
        builder.set_value_block(&mut value_block);

        builder.insert_value::<TestTable>(&10u64, &mut value_block, &layout);
        builder.insert_value::<TestTable>(&20u64, &mut value_block, &layout);
        builder.insert_value::<TestTable>(&30u64, &mut value_block, &layout);

        assert!(!builder.value_block_empty());
        assert!(!builder.value_block_full(&layout));

        builder.value_block_finish::<TestTable>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 1,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        assert!(builder.value_block_empty());
        assert_eq!(builder.value_block_count, 1);
        assert_eq!(builder.state, BuilderState::IndexBlock);
    }

    #[test]
    fn builder_finish_index_block() {
        let layout = TableLayout::compute_for::<TestTable>();
        let mut index_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_block = vec![0u8; constants::BLOCK_SIZE];

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);
        builder.set_value_block(&mut value_block);

        builder.insert_value::<TestTable>(&10u64, &mut value_block, &layout);
        builder.insert_value::<TestTable>(&20u64, &mut value_block, &layout);

        builder.value_block_finish::<TestTable>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 1,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        let info = builder.index_block_finish::<u64>(
            &mut index_block,
            &layout,
            IndexFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 2,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        assert_eq!(info.key_min, 10u64);
        assert_eq!(info.key_max, 20u64);
        assert_eq!(info.value_count, 2);
        assert!(info.address > 0);
    }

    #[test]
    fn index_blocks_for_key_finds_value_block() {
        let layout = TableLayout::compute_for::<TestTable>();
        let mut index_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_block = vec![0u8; constants::BLOCK_SIZE];

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);
        builder.set_value_block(&mut value_block);

        builder.insert_value::<TestTable>(&10u64, &mut value_block, &layout);
        builder.insert_value::<TestTable>(&20u64, &mut value_block, &layout);
        builder.insert_value::<TestTable>(&30u64, &mut value_block, &layout);

        builder.value_block_finish::<TestTable>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 1,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        builder.index_block_finish::<u64>(
            &mut index_block,
            &layout,
            IndexFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 2,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        let result = index_blocks_for_key::<u64>(&index_block, &layout.index, 15);
        assert!(result.is_some());
        let blocks = result.unwrap();
        assert_eq!(blocks.value_block_key_min, 10u64);
        assert_eq!(blocks.value_block_key_max, 30u64);
        assert!(blocks.value_block_address > 0);
    }

    #[test]
    fn index_blocks_for_key_returns_none_out_of_range() {
        let layout = TableLayout::compute_for::<TestTable>();
        let mut index_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_block = vec![0u8; constants::BLOCK_SIZE];

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);
        builder.set_value_block(&mut value_block);

        builder.insert_value::<TestTable>(&100u64, &mut value_block, &layout);

        builder.value_block_finish::<TestTable>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 1,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        builder.index_block_finish::<u64>(
            &mut index_block,
            &layout,
            IndexFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 2,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        // Key 50 < key_min (100) — should return None.
        let result = index_blocks_for_key::<u64>(&index_block, &layout.index, 50);
        assert!(result.is_none());
    }

    /// Table spec with enough values to force 2 value blocks:
    /// 8-byte values → block_value_count_max = 468, VALUE_COUNT_MAX = 937 → 2 blocks.
    struct TestTableMultiBlock;

    impl TableSpec for TestTableMultiBlock {
        type Key = u64;
        type Value = u64;

        fn key_from_value(value: &u64) -> u64 {
            *value
        }

        const SENTINEL_KEY: u64 = u64::MAX;

        fn tombstone(value: &u64) -> bool {
            *value == u64::MAX
        }

        fn tombstone_from_key(_key: u64) -> u64 {
            u64::MAX
        }

        const VALUE_COUNT_MAX: usize = 937;
        const USAGE: TableUsage = TableUsage::General;
    }

    #[test]
    fn multiple_value_blocks() {
        let layout = TableLayout::compute_for::<TestTableMultiBlock>();
        assert_eq!(layout.value_block_count_max, 2, "need exactly 2 value blocks");
        let mut index_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_block = vec![0u8; constants::BLOCK_SIZE];

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);

        builder.set_value_block(&mut value_block);
        builder.insert_value::<TestTableMultiBlock>(&10u64, &mut value_block, &layout);
        builder.insert_value::<TestTableMultiBlock>(&20u64, &mut value_block, &layout);
        builder.value_block_finish::<TestTableMultiBlock>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 1,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        builder.set_value_block(&mut value_block);
        builder.insert_value::<TestTableMultiBlock>(&30u64, &mut value_block, &layout);
        builder.insert_value::<TestTableMultiBlock>(&40u64, &mut value_block, &layout);
        builder.value_block_finish::<TestTableMultiBlock>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 2,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        assert_eq!(builder.value_block_count, 2);

        let info = builder.index_block_finish::<u64>(
            &mut index_block,
            &layout,
            IndexFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: 3,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        assert_eq!(info.key_min, 10u64);
        assert_eq!(info.key_max, 40u64);
        assert_eq!(info.value_count, 4);

        let result0 = index_blocks_for_key::<u64>(&index_block, &layout.index, 15);
        assert!(result0.is_some());
        assert_eq!(result0.unwrap().value_block_key_min, 10u64);

        let result1 = index_blocks_for_key::<u64>(&index_block, &layout.index, 35);
        assert!(result1.is_some());
        assert_eq!(result1.unwrap().value_block_key_min, 30u64);
    }
}
