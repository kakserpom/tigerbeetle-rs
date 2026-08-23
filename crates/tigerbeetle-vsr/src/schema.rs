//! Decode grid blocks.
//!
//! Rather than switching between specialized decoders depending on the tree, each schema encodes
//! relevant parameters directly into the block's header. This allows the decoders to not be
//! generic. This is convenient for compaction, but critical for the scrubber and repair queue.
//!
//! Index block body schema:
//! │ [value_block_count_max]u256   │ checksums of value blocks
//! │ [value_block_count_max]Key    │ the minimum/first key in the respective value block
//! │ [value_block_count_max]Key    │ the maximum/last key in the respective value block
//! │ [value_block_count_max]u64    │ addresses of value blocks
//! │ […]u8{0}                     │ padding (to end of block)
//!
//! Value block body schema:
//! │ [≤value_count_max]Value  │ At least one value (no empty tables).
//! │ […]u8{0}                 │ padding (to end of block)
//!
//! ManifestNode block body schema:
//! │ [entry_count]TableInfo      │
//! │ […]u8{0}                    │ padding (to end of block)
//!
//! DEVIATION: upstream lives at `src/lsm/schema.zig`; this port sits in `tigerbeetle-vsr`
//! because it is built on [`crate::message_header`] and the crate graph is
//! `core ← lsm ← vsr` (lsm may not depend on vsr).
//!
//! DEVIATION: upstream reinterprets block bytes as structs (`mem.bytesAsValue`) over
//! sector-aligned pointers. Without `unsafe`, all multi-byte fields are decoded explicitly
//! little-endian; "slices" of u64/u128 values are exposed as element accessors.

#![allow(clippy::cast_possible_truncation)] // offsets/sizes are < 2^32 by construction (BLOCK_SIZE)
#![allow(clippy::similar_names)]
// upstream names: value_block_count vs value_block_count_max
// Upstream panics on invalid block schemas (scrubber/repair callers); tests assert this.
#![allow(clippy::missing_panics_doc)]

use crate::BlockReference;
use crate::message_header::{self, BlockType, TypedHeader};
use tigerbeetle_core::constants;
use tigerbeetle_core::stdx;

/// Size of a grid block (upstream alias for `constants.block_size`).
const BLOCK_SIZE: usize = constants::BLOCK_SIZE;
const HEADER_SIZE: usize = constants::HEADER_SIZE;

/// Block body capacity (upstream `block_body_size`).
const BLOCK_BODY_SIZE: usize = BLOCK_SIZE - HEADER_SIZE;

const ADDRESS_SIZE: usize = core::mem::size_of::<u64>();
/// Upstream stores checksums on disk padded to u256 (`schema.Checksum`).
const CHECKSUM_SIZE: usize = 32;

/// Width of the per-node metadata area inside `Header.Block.metadata_bytes`.
pub const METADATA_SIZE: usize = 96;

fn le_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes.try_into().map_err(|_| "len").unwrap_or([0; 2]))
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().map_err(|_| "len").unwrap_or([0; 4]))
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().map_err(|_| "len").unwrap_or([0; 8]))
}

fn le_u128(bytes: &[u8]) -> u128 {
    u128::from_le_bytes(bytes.try_into().map_err(|_| "len").unwrap_or([0; 16]))
}

/// Upstream `header_from_block`: parses and validates a block's fixed header.
///
/// # Panics
/// Panics if the header fails any structural invariant (upstream asserts the same).
#[must_use]
pub fn header_from_block(block: &[u8]) -> message_header::Block {
    assert_eq!(block.len(), BLOCK_SIZE);
    let header_bytes: [u8; HEADER_SIZE] =
        block[..HEADER_SIZE].try_into().map_err(|_| "len").unwrap_or([0; HEADER_SIZE]);

    // `from_wire` already rejects wrong-command frames (upstream asserts command == .block):
    let Some(header) = message_header::Block::from_wire(&header_bytes) else {
        panic!("invalid block header");
    };

    assert!(header.address > 0);
    assert!(header.size >= HEADER_SIZE as u32); // Every block has a header.
    assert!(header.size > HEADER_SIZE as u32); // Every block has a non-empty body.
    assert!(header.size <= block.len() as u32);
    let block_type = BlockType::from_ordinal(header.block_type_ordinal);
    assert!(block_type.is_some(), "invalid block type");
    assert_ne!(block_type, Some(BlockType::Reserved));
    assert!(header.release.value > 0);
    header
}

/// Decoded metadata area shared by all node types (96 bytes inside `Header.Block`).
mod metadata_wire {
    use super::{METADATA_SIZE, le_u16, le_u32, le_u64, le_u128};

    pub(super) fn get_u32(bytes: &[u8; METADATA_SIZE], offset: usize) -> u32 {
        le_u32(&bytes[offset..offset + 4])
    }

    pub(super) fn put_u32(dst: &mut [u8; METADATA_SIZE], offset: usize, value: u32) {
        dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    pub(super) fn get_u16(bytes: &[u8; METADATA_SIZE], offset: usize) -> u16 {
        le_u16(&bytes[offset..offset + 2])
    }

    pub(super) fn put_u16(dst: &mut [u8; METADATA_SIZE], offset: usize, value: u16) {
        dst[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    pub(super) fn get_u64(bytes: &[u8; METADATA_SIZE], offset: usize) -> u64 {
        le_u64(&bytes[offset..offset + 8])
    }

    pub(super) fn put_u64(dst: &mut [u8; METADATA_SIZE], offset: usize, value: u64) {
        dst[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    pub(super) fn get_u128(bytes: &[u8; METADATA_SIZE], offset: usize) -> u128 {
        le_u128(&bytes[offset..offset + 16])
    }

    pub(super) fn put_u128(dst: &mut [u8; METADATA_SIZE], offset: usize, value: u128) {
        dst[offset..offset + 16].copy_from_slice(&value.to_le_bytes());
    }

    /// Asserts that `bytes[field_offset..][..count]` (reserved area) is zeroed.
    pub(super) fn assert_reserved_zeroed(
        bytes: &[u8; METADATA_SIZE],
        field_offset: usize,
        count: usize,
    ) {
        let reserved = &bytes[field_offset..field_offset + count];
        assert!(super::stdx::zeroed(reserved));
    }
}

use metadata_wire::{
    assert_reserved_zeroed, get_u16, get_u32, get_u64, get_u128, put_u16, put_u32, put_u64,
    put_u128,
};

fn le_u32_at(bytes: &[u8], offset: usize) -> u32 {
    let mut buf = [0_u8; 4];
    buf.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(buf)
}

/// Decoded `schema.TableIndex.Metadata`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableIndexMetadata {
    pub value_block_count: u32,
    pub value_block_count_max: u32,
    pub key_size: u32,
    pub tree_id: u16,
}

impl TableIndexMetadata {
    /// Layout: value_block_count(4) value_block_count_max(4) key_size(4) tree_id(2)
    /// reserved[82].
    const OFFSET_VALUE_BLOCK_COUNT: usize = 0;
    const OFFSET_VALUE_BLOCK_COUNT_MAX: usize = 4;
    const OFFSET_KEY_SIZE: usize = 8;
    const OFFSET_TREE_ID: usize = 12;
    const OFFSET_RESERVED: usize = 14;
    const RESERVED_LEN: usize = METADATA_SIZE - Self::OFFSET_RESERVED;

    /// # Panics
    /// Panics if the reserved area is nonzero (upstream asserts the same).
    #[must_use]
    pub fn from_wire(bytes: &[u8; METADATA_SIZE]) -> Self {
        assert_reserved_zeroed(bytes, Self::OFFSET_RESERVED, Self::RESERVED_LEN);
        Self {
            value_block_count: get_u32(bytes, Self::OFFSET_VALUE_BLOCK_COUNT),
            value_block_count_max: get_u32(bytes, Self::OFFSET_VALUE_BLOCK_COUNT_MAX),
            key_size: get_u32(bytes, Self::OFFSET_KEY_SIZE),
            tree_id: get_u16(bytes, Self::OFFSET_TREE_ID),
        }
    }

    /// Serializes into the 96-byte metadata area (reserved zeroed).
    #[must_use]
    pub fn to_wire(self) -> [u8; METADATA_SIZE] {
        let mut bytes = [0_u8; METADATA_SIZE];
        put_u32(&mut bytes, Self::OFFSET_VALUE_BLOCK_COUNT, self.value_block_count);
        put_u32(&mut bytes, Self::OFFSET_VALUE_BLOCK_COUNT_MAX, self.value_block_count_max);
        put_u32(&mut bytes, Self::OFFSET_KEY_SIZE, self.key_size);
        put_u16(&mut bytes, Self::OFFSET_TREE_ID, self.tree_id);
        bytes
    }
}

/// Layout descriptor for index blocks (upstream `TableIndex`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableIndex {
    pub key_size: u32,
    pub value_block_count_max: u32,

    pub size: u32,
    pub value_checksums_offset: u32,
    pub value_checksums_size: u32,
    pub keys_min_offset: u32,
    pub keys_max_offset: u32,
    pub keys_size: u32,
    pub value_addresses_offset: u32,
    pub value_addresses_size: u32,
    pub padding_offset: u32,
    pub padding_size: u32,
}

impl TableIndex {
    /// # Panics
    /// Panics if parameters are zero/out-of-range or the layout exceeds a block
    /// (upstream asserts the same).
    #[must_use]
    pub fn init(key_size: u32, value_block_count_max: u32) -> Self {
        assert!(key_size > 0);
        assert!(value_block_count_max > 0);
        assert!(value_block_count_max <= constants::LSM_TABLE_VALUE_BLOCKS_MAX as u32);

        let value_checksums_offset = HEADER_SIZE as u32;
        let value_checksums_size = value_block_count_max * CHECKSUM_SIZE as u32;

        let keys_size = value_block_count_max * key_size;
        let keys_min_offset = value_checksums_offset + value_checksums_size;
        let keys_max_offset = keys_min_offset + keys_size;

        let value_addresses_offset = keys_max_offset + keys_size;
        let value_addresses_size = value_block_count_max * ADDRESS_SIZE as u32;

        let padding_offset = value_addresses_offset + value_addresses_size;
        assert!(padding_offset <= BLOCK_SIZE as u32);
        let padding_size = BLOCK_SIZE as u32 - padding_offset;

        // `keys_size * 2` for counting both key_min and key_max:
        let size =
            HEADER_SIZE as u32 + value_checksums_size + (keys_size * 2) + value_addresses_size;
        assert!(size <= BLOCK_SIZE as u32);

        Self {
            key_size,
            value_block_count_max,
            size,
            value_checksums_offset,
            value_checksums_size,
            keys_min_offset,
            keys_max_offset,
            keys_size,
            value_addresses_offset,
            value_addresses_size,
            padding_offset,
            padding_size,
        }
    }

    /// Parses + validates the metadata area (upstream `metadata()`); also cross-checks the
    /// encoded size against this schema.
    ///
    /// # Panics
    /// Panics if the block is not a valid index block.
    #[must_use]
    pub fn metadata(index_block: &[u8]) -> TableIndexMetadata {
        let header = header_from_block(index_block);
        let result = TableIndexMetadata::from_wire(&header.metadata_bytes);

        assert!(result.value_block_count <= result.value_block_count_max);
        assert_eq!(
            header.size,
            HEADER_SIZE as u32
                + result.value_block_count_max
                    * (CHECKSUM_SIZE as u32 + ADDRESS_SIZE as u32 + result.key_size * 2)
        );
        result
    }

    /// Upstream `block_metadata` — metadata that additionally matches this schema.
    ///
    /// # Panics
    /// Panics if the block's schema does not match.
    #[must_use]
    pub fn block_metadata(&self, index_block: &[u8]) -> TableIndexMetadata {
        let result = Self::metadata(index_block);
        assert_eq!(result.key_size, self.key_size);
        assert_eq!(result.value_block_count_max, self.value_block_count_max);
        result
    }

    /// Upstream `from_block_without_schema`.
    ///
    /// # Panics
    /// Panics if the block violates the schema encoded in its own header.
    #[must_use]
    pub fn from_block_without_schema(index_block: &[u8]) -> TableIndex {
        let header_metadata = Self::metadata(index_block);
        let index =
            TableIndex::init(header_metadata.key_size, header_metadata.value_block_count_max);

        for (start, end) in index.padding(index_block) {
            assert!(stdx::zeroed(&index_block[start..end]));
        }

        index
    }

    /// Upstream `from_block_with_schema`.
    ///
    /// # Panics
    /// Panics if the block's schema differs from `tree_id`/this schema.
    #[must_use]
    pub fn from_block_with_schema(&self, index_block: &[u8], tree_id: u16) -> TableIndex {
        let header_metadata = Self::metadata(index_block);
        assert_eq!(header_metadata.tree_id, tree_id);
        assert_eq!(header_metadata.key_size, self.key_size);
        assert_eq!(header_metadata.value_block_count_max, self.value_block_count_max);
        TableIndex::from_block_without_schema(index_block)
    }

    /// Number of value blocks referenced by this index block.
    ///
    /// # Panics
    /// Panics if the count is zero or exceeds the schema maximum.
    #[must_use]
    pub fn value_blocks_used(&self, index_block: &[u8]) -> u32 {
        let header_metadata = self.block_metadata(index_block);
        assert!(header_metadata.value_block_count > 0);
        assert!(header_metadata.value_block_count <= self.value_block_count_max);
        header_metadata.value_block_count
    }

    /// Upstream `value_addresses_used()[i]` (element-wise; see module DEVIATION).
    #[must_use]
    pub fn value_address(&self, index_block: &[u8], i: usize) -> u64 {
        assert!(i < self.value_blocks_used(index_block) as usize);
        let offset = self.value_addresses_offset as usize + i * ADDRESS_SIZE;
        le_u64_at(index_block, offset)
    }

    /// Upstream `value_checksums_used()[i]` — the unpadded u128 checksum value.
    #[must_use]
    pub fn value_checksum(&self, index_block: &[u8], i: usize) -> u128 {
        assert!(i < self.value_blocks_used(index_block) as usize);
        let base = self.value_checksums_offset as usize + i * CHECKSUM_SIZE;
        le_u128_at(index_block, base)
    }

    /// Upstream `keys_min` region entry `i` (zero-copy byte slice of `key_size`).
    #[must_use]
    pub fn key_min<'a>(&self, index_block: &'a [u8], i: usize) -> &'a [u8] {
        assert!(i < self.value_blocks_used(index_block) as usize);
        let offset = self.keys_min_offset as usize + i * self.key_size as usize;
        &index_block[offset..offset + self.key_size as usize]
    }

    /// Upstream `keys_max` region entry `i`.
    #[must_use]
    pub fn key_max<'a>(&self, index_block: &'a [u8], i: usize) -> &'a [u8] {
        assert!(i < self.value_blocks_used(index_block) as usize);
        let offset = self.keys_max_offset as usize + i * self.key_size as usize;
        &index_block[offset..offset + self.key_size as usize]
    }

    /// Upstream `padding()`: regions that must be zero, given the used-block count.
    /// Returns `(start, end)` pairs.
    ///
    /// # Panics
    /// Panics if no blocks are used (upstream asserts used > 0 first).
    #[must_use]
    pub fn padding(&self, index_block: &[u8]) -> [(usize, usize); 4] {
        let used = self.value_blocks_used(index_block);

        let value_checksums_skip = used as usize * CHECKSUM_SIZE;
        let keys_min_skip = used as usize * self.key_size as usize;
        let keys_max_skip = used as usize * self.key_size as usize;
        let value_addresses_skip = used as usize * ADDRESS_SIZE;

        [
            (
                self.value_checksums_offset as usize + value_checksums_skip,
                self.value_checksums_offset as usize + self.value_checksums_size as usize,
            ),
            (
                self.keys_min_offset as usize + keys_min_skip,
                self.keys_min_offset as usize + self.keys_size as usize,
            ),
            (
                self.keys_max_offset as usize + keys_max_skip,
                self.keys_max_offset as usize + self.keys_size as usize,
            ),
            (
                self.value_addresses_offset as usize + value_addresses_skip,
                self.value_addresses_offset as usize + self.value_addresses_size as usize,
            ),
        ]
    }
}

fn le_u64_at(bytes: &[u8], offset: usize) -> u64 {
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(buf)
}

fn le_u128_at(bytes: &[u8], offset: usize) -> u128 {
    let mut buf = [0_u8; 16];
    buf.copy_from_slice(&bytes[offset..offset + 16]);
    u128::from_le_bytes(buf)
}

/// Decoded `schema.TableValue.Metadata`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableValueMetadata {
    pub value_count_max: u32,
    pub value_count: u32,
    pub value_size: u32,
    pub tree_id: u16,
}

impl TableValueMetadata {
    /// Layout: value_count_max(4) value_count(4) value_size(4) tree_id(2) reserved[82].
    const OFFSET_VALUE_COUNT_MAX: usize = 0;
    const OFFSET_VALUE_COUNT: usize = 4;
    const OFFSET_VALUE_SIZE: usize = 8;
    const OFFSET_TREE_ID: usize = 12;
    const OFFSET_RESERVED: usize = 14;
    const RESERVED_LEN: usize = METADATA_SIZE - Self::OFFSET_RESERVED;

    /// # Panics
    /// Panics if the reserved area is nonzero.
    #[must_use]
    pub fn from_wire(bytes: &[u8; METADATA_SIZE]) -> Self {
        assert_reserved_zeroed(bytes, Self::OFFSET_RESERVED, Self::RESERVED_LEN);
        Self {
            value_count_max: get_u32(bytes, Self::OFFSET_VALUE_COUNT_MAX),
            value_count: get_u32(bytes, Self::OFFSET_VALUE_COUNT),
            value_size: get_u32(bytes, Self::OFFSET_VALUE_SIZE),
            tree_id: get_u16(bytes, Self::OFFSET_TREE_ID),
        }
    }

    /// Serializes into the 96-byte metadata area (reserved zeroed).
    #[must_use]
    pub fn to_wire(self) -> [u8; METADATA_SIZE] {
        let mut bytes = [0_u8; METADATA_SIZE];
        put_u32(&mut bytes, Self::OFFSET_VALUE_COUNT_MAX, self.value_count_max);
        put_u32(&mut bytes, Self::OFFSET_VALUE_COUNT, self.value_count);
        put_u32(&mut bytes, Self::OFFSET_VALUE_SIZE, self.value_size);
        put_u16(&mut bytes, Self::OFFSET_TREE_ID, self.tree_id);
        bytes
    }
}

/// Layout descriptor for value blocks (upstream `TableValue`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableValue {
    /// `@sizeOf(Table.Value)` — the fixed size of one value.
    pub value_size: u32,
    /// The maximum number of values in a value block.
    pub value_count_max: u32,

    pub values_offset: u32,
    pub values_size: u32,

    pub padding_offset: u32,
    pub padding_size: u32,
}

impl TableValue {
    /// # Panics
    /// Panics if parameters are zero or `value_size` is not a power of two.
    #[must_use]
    pub fn init(value_count_max: u32, value_size: u32) -> Self {
        assert!(value_count_max > 0);
        assert!(value_size > 0);
        assert!(value_size.is_power_of_two());

        let values_offset = HEADER_SIZE as u32;
        let values_size = value_count_max * value_size;

        let padding_offset = values_offset + values_size;
        let padding_size = BLOCK_SIZE as u32 - padding_offset;

        Self {
            value_size,
            value_count_max,
            values_offset,
            values_size,
            padding_offset,
            padding_size,
        }
    }

    /// Parses + validates the metadata area (upstream `metadata()`).
    ///
    /// # Panics
    /// Panics if the block is not a valid non-empty value block.
    #[must_use]
    pub fn metadata(value_block: &[u8]) -> TableValueMetadata {
        let header = header_from_block(value_block);
        let result = TableValueMetadata::from_wire(&header.metadata_bytes);

        assert!(result.value_size > 0);
        assert!(result.value_count > 0);
        assert!(result.value_count <= result.value_count_max);
        assert!(result.tree_id > 0);
        assert_eq!(
            header.size,
            (HEADER_SIZE + result.value_size as usize * result.value_count as usize) as u32
        );
        result
    }

    /// Upstream `block_metadata` — metadata that additionally matches this schema.
    ///
    /// # Panics
    /// Panics if the block's schema does not match.
    #[must_use]
    pub fn block_metadata(&self, value_block: &[u8]) -> TableValueMetadata {
        let result = Self::metadata(value_block);
        assert_eq!(result.value_size, self.value_size);
        assert_eq!(result.value_count_max, self.value_count_max);
        result
    }

    /// Upstream `from` — reconstructs the schema from a value block's own metadata.
    ///
    /// # Panics
    /// Panics if the block is invalid.
    #[must_use]
    pub fn from_block(value_block: &[u8]) -> TableValue {
        let header_metadata = Self::metadata(value_block);
        Self::init(header_metadata.value_count_max, header_metadata.value_size)
    }

    /// Upstream `assert_matching_block_schema`.
    ///
    /// # Panics
    /// Panics if the block's schema does not match this schema/`tree_id`.
    pub fn assert_matching_block_schema(&self, value_block: &[u8], tree_id: u16) {
        // Upstream validates the block header first (address/snapshot invariants):
        _ = header_from_block(value_block);

        let header_metadata = Self::metadata(value_block);
        assert_eq!(header_metadata.tree_id, tree_id);
        assert_eq!(header_metadata.value_size, self.value_size);
        assert_eq!(header_metadata.value_count_max, self.value_count_max);
    }

    /// Upstream `block_values_used_bytes()` length: how many bytes of values are used.
    ///
    /// # Panics
    /// Panics if the block is empty or oversized.
    #[must_use]
    pub fn block_values_used_bytes_len(&self, value_block: &[u8]) -> usize {
        let header = header_from_block(value_block);

        let used_values = self.block_metadata(value_block).value_count;
        assert!(used_values > 0);
        assert!(used_values <= self.value_count_max);

        let used_bytes = used_values as usize * self.value_size as usize;
        assert_eq!(HEADER_SIZE + used_bytes, header.size as usize);
        assert!(header.size as usize <= self.padding_offset as usize); // Maximum padding_offset.
        used_bytes
    }
}

/// A TrailerNode is either a `BlockType.free_set` or `BlockType.client_sessions`
/// (upstream `TrailerNode`).
///
/// DEVIATION: a module rather than an inherent impl — Rust forbids nested type definitions
/// inside `impl` blocks.
#[allow(non_snake_case)]
// mirrors upstream type name
// DEVIATION: upstream nests these under one file scope; a glob keeps the port diff-able.
#[allow(clippy::wildcard_imports)]
pub mod TrailerNode {
    use super::*;

    /// Decoded `schema.TrailerNode.Metadata`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Metadata {
        pub previous_trailer_block_checksum: u128,
        pub previous_trailer_block_address: u64,
    }

    impl Metadata {
        /// Layout: prev_checksum(16) prev_checksum_padding(16) prev_address(8) reserved[56].
        const OFFSET_PREV_CHECKSUM: usize = 0;
        const OFFSET_PREV_CHECKSUM_PADDING: usize = 16;
        const OFFSET_PREV_ADDRESS: usize = 32;
        const OFFSET_RESERVED: usize = 40;
        const RESERVED_LEN: usize = METADATA_SIZE - Self::OFFSET_RESERVED;

        /// # Panics
        /// Panics if padding/reserved areas are nonzero.
        #[must_use]
        pub fn from_wire(bytes: &[u8; METADATA_SIZE]) -> Self {
            // The checksum padding word must be zero:
            assert_eq!(get_u128(bytes, Self::OFFSET_PREV_CHECKSUM_PADDING), 0, "checksum padding");
            assert_reserved_zeroed(bytes, Self::OFFSET_RESERVED, Self::RESERVED_LEN);
            Self {
                previous_trailer_block_checksum: get_u128(bytes, Self::OFFSET_PREV_CHECKSUM),
                previous_trailer_block_address: get_u64(bytes, Self::OFFSET_PREV_ADDRESS),
            }
        }

        /// Serializes into the 96-byte metadata area (padding/reserved zeroed).
        #[must_use]
        pub fn to_wire(self) -> [u8; METADATA_SIZE] {
            let mut bytes = [0_u8; METADATA_SIZE];
            put_u128(&mut bytes, Self::OFFSET_PREV_CHECKSUM, self.previous_trailer_block_checksum);
            put_u64(&mut bytes, Self::OFFSET_PREV_ADDRESS, self.previous_trailer_block_address);
            bytes
        }
    }

    /// Upstream `TrailerNode.metadata()`.
    ///
    /// # Panics
    /// Panics if the block is not a valid trailer block.
    #[must_use]
    pub fn metadata(trailer_block: &[u8]) -> Metadata {
        let header = header_from_block(trailer_block);
        let block_type = BlockType::from_ordinal(header.block_type_ordinal);
        assert!(
            block_type == Some(BlockType::FreeSet) || block_type == Some(BlockType::ClientSessions)
        );
        assert_eq!(header.snapshot, 0);

        let metadata = Metadata::from_wire(&header.metadata_bytes);

        if metadata.previous_trailer_block_address == 0 {
            assert_eq!(metadata.previous_trailer_block_checksum, 0);
        }

        assert!(header.size > HEADER_SIZE as u32);

        match block_type {
            Some(BlockType::FreeSet) => {
                assert_eq!(
                    (header.size - HEADER_SIZE as u32) % core::mem::size_of::<u64>() as u32,
                    0
                );
            }
            Some(BlockType::ClientSessions) => {
                let item_size =
                    (crate::checkpoint_trailer::TrailerType::ClientSessions.item_size()) as u32;
                assert_eq!((header.size - HEADER_SIZE as u32) % item_size, 0);
            }
            _ => panic!("unreachable"),
        }

        metadata
    }

    /// Upstream `TrailerNode.assert_valid_header`.
    ///
    /// # Panics
    /// Panics unless the trailer block header is valid.
    pub fn assert_valid_header(trailer_block: &[u8]) {
        _ = metadata(trailer_block);
    }

    /// Upstream `TrailerNode.previous`.
    #[must_use]
    pub fn previous(trailer_block: &[u8]) -> Option<BlockReference> {
        let metadata = metadata(trailer_block);

        if metadata.previous_trailer_block_address == 0 {
            assert_eq!(metadata.previous_trailer_block_checksum, 0);
            None
        } else {
            Some(BlockReference {
                checksum: metadata.previous_trailer_block_checksum,
                address: metadata.previous_trailer_block_address,
            })
        }
    }

    /// Returns the body up to the size specified in the header (zero-copy byte slice).
    #[must_use]
    pub fn body(trailer_block: &[u8]) -> &[u8] {
        let header = header_from_block(trailer_block);
        &trailer_block[HEADER_SIZE..header.size as usize]
    }
}

/// A Manifest block's body is an array of [`ManifestNode::TableInfo`] entries
/// (upstream `ManifestNode`).
///
/// DEVIATION: a module rather than an inherent impl — see [`TrailerNode`].
#[allow(non_snake_case)]
// mirrors upstream type name
// DEVIATION: upstream nests these under one file scope; a glob keeps the port diff-able.
#[allow(clippy::wildcard_imports)]
pub mod ManifestNode {
    use super::*;

    /// Wire size of one [`TableInfo`] entry.
    /// DEVIATION: upstream derives this from `@sizeOf(TableInfo)` (extern struct with
    /// trailing padding); our safe-Rust `TableInfo` carries no padding fields, so the size is
    /// fixed by the on-disk layout instead.
    pub const ENTRY_SIZE: usize = 128;

    pub const ENTRY_COUNT_MAX: usize = BLOCK_BODY_SIZE / ENTRY_SIZE;

    /// Bit 7 of the label is reserved to indicate whether the event is an insert or remove
    /// (upstream asserts levels fit u6):
    const _: () = assert!(constants::LSM_LEVELS <= 0b0011_1111 + 1);

    /// Decoded `schema.ManifestNode.Metadata`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Metadata {
        pub previous_manifest_block_checksum: u128,
        pub previous_manifest_block_address: u64,
        pub entry_count: u32,
    }

    impl Metadata {
        /// Layout: prev_checksum(16) prev_padding(16) prev_address(8) entry_count(4)
        /// reserved[52].
        const OFFSET_PREV_CHECKSUM: usize = 0;
        const OFFSET_PREV_CHECKSUM_PADDING: usize = 16;
        const OFFSET_PREV_ADDRESS: usize = 32;
        const OFFSET_ENTRY_COUNT: usize = 40;
        const OFFSET_RESERVED: usize = 44;
        const RESERVED_LEN: usize = METADATA_SIZE - Self::OFFSET_RESERVED;

        /// # Panics
        /// Panics if padding/reserved areas are nonzero.
        #[must_use]
        pub fn from_wire(bytes: &[u8; METADATA_SIZE]) -> Self {
            assert_eq!(get_u128(bytes, Self::OFFSET_PREV_CHECKSUM_PADDING), 0);
            assert_reserved_zeroed(bytes, Self::OFFSET_RESERVED, Self::RESERVED_LEN);
            Self {
                previous_manifest_block_checksum: get_u128(bytes, Self::OFFSET_PREV_CHECKSUM),
                previous_manifest_block_address: get_u64(bytes, Self::OFFSET_PREV_ADDRESS),
                entry_count: get_u32(bytes, Self::OFFSET_ENTRY_COUNT),
            }
        }

        /// Serializes into the 96-byte metadata area (padding/reserved zeroed).
        #[must_use]
        pub fn to_wire(self) -> [u8; METADATA_SIZE] {
            let mut bytes = [0_u8; METADATA_SIZE];
            put_u128(&mut bytes, Self::OFFSET_PREV_CHECKSUM, self.previous_manifest_block_checksum);
            put_u64(&mut bytes, Self::OFFSET_PREV_ADDRESS, self.previous_manifest_block_address);
            put_u32(&mut bytes, Self::OFFSET_ENTRY_COUNT, self.entry_count);
            bytes
        }
    }

    /// Upstream `Event` (2 bits inside `Label`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Event {
        Reserved = 0,
        Insert = 1,
        Update = 2,
        Remove = 3,
    }

    /// Upstream `Label` — packed u8: level in bits 0..6, event in bits 6..8.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Label {
        pub level: u8,
        pub event: Event,
    }

    impl Label {
        #[must_use]
        pub fn to_u8(self) -> u8 {
            assert!(self.level <= 0b0011_1111);
            #[allow(clippy::identity_op)] // mirrors upstream packed-struct field placement
            {
                (self.level & 0b0011_1111)
                    | ((self.event as u8).checked_shl(6).unwrap_or(u8::MAX & 0b1100_0000))
            }
        }

        /// # Panics
        /// Panics if the event ordinal is invalid.
        #[must_use]
        pub fn from_u8(bits: u8) -> Self {
            let level = bits & 0b0011_1111;
            let event_ordinal = bits >> 6;
            let event = match event_ordinal {
                0 => Event::Reserved,
                1 => Event::Insert,
                2 => Event::Update,
                3 => Event::Remove,
                _ => panic!("invalid label event"),
            };
            Self { level, event }
        }
    }

    /// See manifest.zig's TreeTableInfoType declaration for field documentation
    /// (upstream `TableInfo`, 128 bytes on disk).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct TableInfo {
        /// All keys must fit within 32 bytes (upstream `KeyPadded`).
        pub key_min: [u8; 32],
        pub key_max: [u8; 32],
        pub checksum: u128,
        pub address: u64,
        pub snapshot_min: u64,
        pub snapshot_max: u64,
        pub value_count: u32,
        pub tree_id: u16,
        pub label: Label,
    }

    impl TableInfo {
        /// Wire offsets within the 128-byte entry (checksums padded to u256).
        const OFFSET_KEY_MIN: usize = 0;
        const OFFSET_KEY_MAX: usize = 32;
        const OFFSET_CHECKSUM: usize = 64;
        const OFFSET_CHECKSUM_PADDING: usize = 80;
        const OFFSET_ADDRESS: usize = 96;
        const OFFSET_SNAPSHOT_MIN: usize = 104;
        const OFFSET_SNAPSHOT_MAX: usize = 112;
        const OFFSET_VALUE_COUNT: usize = 120;
        const OFFSET_TREE_ID: usize = 124;
        const OFFSET_LABEL: usize = 126;
        const OFFSET_RESERVED: usize = 127;
        pub const SIZE: usize = 128;

        /// # Panics
        /// Panics if padding/reserved areas are nonzero.
        #[must_use]
        pub fn from_wire(bytes: &[u8; Self::SIZE]) -> Self {
            assert_eq!(le_u128_at(bytes, Self::OFFSET_CHECKSUM_PADDING), 0);
            assert_eq!(bytes[Self::OFFSET_RESERVED], 0);

            let mut key_min = [0_u8; 32];
            key_min.copy_from_slice(&bytes[Self::OFFSET_KEY_MIN..Self::OFFSET_KEY_MAX]);
            let mut key_max = [0_u8; 32];
            key_max.copy_from_slice(&bytes[Self::OFFSET_KEY_MAX..Self::OFFSET_CHECKSUM]);

            Self {
                key_min,
                key_max,
                checksum: le_u128_at(bytes, Self::OFFSET_CHECKSUM),
                address: le_u64_at(bytes, Self::OFFSET_ADDRESS),
                snapshot_min: le_u64_at(bytes, Self::OFFSET_SNAPSHOT_MIN),
                snapshot_max: le_u64_at(bytes, Self::OFFSET_SNAPSHOT_MAX),
                value_count: le_u32_at(bytes, Self::OFFSET_VALUE_COUNT),
                tree_id: u16::from_le_bytes(
                    bytes[Self::OFFSET_TREE_ID..Self::OFFSET_TREE_ID + 2]
                        .try_into()
                        .map_err(|_| "len")
                        .unwrap_or([0; 2]),
                ),
                label: Label::from_u8(bytes[Self::OFFSET_LABEL]),
            }
        }

        /// Serializes into the 128-byte on-disk entry (padding/reserved zeroed).
        #[must_use]
        pub fn to_wire(self) -> [u8; Self::SIZE] {
            let mut bytes = [0_u8; Self::SIZE];
            bytes[Self::OFFSET_KEY_MIN..Self::OFFSET_KEY_MAX].copy_from_slice(&self.key_min);
            bytes[Self::OFFSET_KEY_MAX..Self::OFFSET_CHECKSUM].copy_from_slice(&self.key_max);
            bytes[Self::OFFSET_CHECKSUM..Self::OFFSET_CHECKSUM + 16]
                .copy_from_slice(&self.checksum.to_le_bytes());
            bytes[Self::OFFSET_ADDRESS..Self::OFFSET_ADDRESS + 8]
                .copy_from_slice(&self.address.to_le_bytes());
            bytes[Self::OFFSET_SNAPSHOT_MIN..Self::OFFSET_SNAPSHOT_MIN + 8]
                .copy_from_slice(&self.snapshot_min.to_le_bytes());
            bytes[Self::OFFSET_SNAPSHOT_MAX..Self::OFFSET_SNAPSHOT_MAX + 8]
                .copy_from_slice(&self.snapshot_max.to_le_bytes());
            bytes[Self::OFFSET_VALUE_COUNT..Self::OFFSET_VALUE_COUNT + 4]
                .copy_from_slice(&self.value_count.to_le_bytes());
            bytes[Self::OFFSET_TREE_ID..Self::OFFSET_TREE_ID + 2]
                .copy_from_slice(&self.tree_id.to_le_bytes());
            bytes[Self::OFFSET_LABEL] = self.label.to_u8();
            bytes
        }
    }

    /// Upstream `ManifestNode.metadata()`.
    ///
    /// # Panics
    /// Panics if the block is not a valid manifest block.
    #[must_use]
    pub fn metadata(manifest_block: &[u8]) -> Metadata {
        let header = header_from_block(manifest_block);
        let metadata = Metadata::from_wire(&header.metadata_bytes);

        assert!(metadata.entry_count > 0);
        assert!(metadata.entry_count as usize <= ENTRY_COUNT_MAX);
        assert_eq!(
            metadata.entry_count as usize,
            (header.size as usize - HEADER_SIZE) / ENTRY_SIZE
        );

        if metadata.previous_manifest_block_address == 0 {
            assert_eq!(metadata.previous_manifest_block_checksum, 0);
        }

        metadata
    }

    /// Note that the returned block reference is no longer part of the manifest if
    /// `manifest_block` is the oldest block in the superblock's CheckpointState.
    ///
    /// # Panics
    /// Panics if the block is invalid (validation only, upstream `_ = from(block)`).
    #[must_use]
    pub fn previous(manifest_block: &[u8]) -> Option<BlockReference> {
        let metadata = metadata(manifest_block);
        if metadata.previous_manifest_block_address == 0 {
            assert_eq!(metadata.previous_manifest_block_checksum, 0);
            None
        } else {
            Some(BlockReference {
                checksum: metadata.previous_manifest_block_checksum,
                address: metadata.previous_manifest_block_address,
            })
        }
    }

    /// Upstream `size` — encoded size of a manifest block with this entry count.
    ///
    /// # Panics
    /// Panics if `entry_count` is zero or exceeds the maximum.
    #[must_use]
    pub fn size(entry_count: u32) -> u32 {
        assert!(entry_count > 0);
        assert!(entry_count as usize <= ENTRY_COUNT_MAX);
        (HEADER_SIZE + entry_count as usize * ENTRY_SIZE) as u32
    }

    /// Upstream `tables_const`: decodes all table entries (copied; see module DEVIATION).
    ///
    /// # Panics
    /// Panics if the block is invalid.
    #[must_use]
    pub fn tables(manifest_block: &[u8]) -> Vec<TableInfo> {
        let metadata = metadata(manifest_block);
        let body =
            &manifest_block[HEADER_SIZE..HEADER_SIZE + metadata.entry_count as usize * ENTRY_SIZE];

        (0..metadata.entry_count as usize)
            .map(|i| {
                let mut entry = [0_u8; TableInfo::SIZE];
                entry.copy_from_slice(&body[i * ENTRY_SIZE..(i + 1) * ENTRY_SIZE]);
                TableInfo::from_wire(&entry)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message_header::{self, SIZE as HEADER_WIRE_SIZE, TypedHeader};
    use ManifestNode::TableInfo;
    use TrailerNode::Metadata as TrailerMetadata;

    const BLOCK: usize = BLOCK_SIZE;
    /// Offset of the metadata area inside the block header frame.
    const METADATA_OFFSET: usize = 128;

    /// Builds a minimal valid block header with `Command::Block` and writes it into `block`.
    fn put_block_header(block: &mut [u8], size: u32, address: u64, block_type: BlockType) {
        let mut header = message_header::Block::default();
        header.cluster = 1;
        header.address = address;
        // Trailer blocks always carry snapshot == 0 (upstream TrailerNode asserts this):
        header.snapshot = if matches!(block_type, BlockType::FreeSet | BlockType::ClientSessions) {
            0
        } else {
            7
        };
        header.block_type_ordinal = block_type as u8;
        header.release.value = 1;
        header.size = size;
        let wire = TypedHeader::to_wire(&header);
        block[..HEADER_WIRE_SIZE].copy_from_slice(&wire);
    }

    fn write_metadata(block: &mut [u8], metadata_wire: &[u8; METADATA_SIZE]) {
        block[METADATA_OFFSET..METADATA_OFFSET + METADATA_SIZE].copy_from_slice(metadata_wire);
    }

    #[test]
    fn table_index_metadata_round_trip() {
        let metadata = TableIndexMetadata {
            value_block_count: 3,
            value_block_count_max: 10,
            key_size: 16,
            tree_id: 2,
        };
        let wire = metadata.to_wire();
        assert_eq!(wire.len(), METADATA_SIZE);
        // Reserved area zeroed:
        assert!(stdx::zeroed(&wire[14..]));
        assert_eq!(TableIndexMetadata::from_wire(&wire), metadata);
    }

    #[test]
    fn table_index_layout_matches_formula() {
        for &key_size in &[1_u32, 8, 16, 24, 32] {
            for &value_block_count_max in &[1_u32, 4] {
                let index = TableIndex::init(key_size, value_block_count_max);

                assert_eq!(index.value_checksums_offset as usize, HEADER_SIZE);
                assert_eq!(
                    index.value_checksums_size as usize,
                    value_block_count_max as usize * CHECKSUM_SIZE
                );
                assert_eq!(
                    index.keys_min_offset,
                    index.value_checksums_offset + index.value_checksums_size
                );
                assert_eq!(index.keys_max_offset, index.keys_min_offset + index.keys_size);
                assert_eq!(index.keys_size, index.key_size * index.value_block_count_max);
                assert_eq!(index.value_addresses_offset, index.keys_max_offset + index.keys_size);
                assert_eq!(
                    index.value_addresses_size as usize,
                    value_block_count_max as usize * ADDRESS_SIZE
                );
                assert_eq!(
                    index.padding_offset,
                    index.value_addresses_offset + index.value_addresses_size
                );

                // Upstream size formula:
                let expect_size = HEADER_SIZE as u32
                    + index.value_checksums_size
                    + index.keys_size * 2
                    + index.value_addresses_size;
                assert_eq!(index.size, expect_size);
                assert!(index.size <= BLOCK_SIZE as u32);
                assert_eq!(index.padding_size as usize, BLOCK_SIZE - index.padding_offset as usize);
            }
        }
        // A layout must fit the block: for 32-byte keys, the largest fitting count is
        // floor((BLOCK_SIZE - HEADER_SIZE) / (CHECKSUM_SIZE + ADDRESS_SIZE + 2*key_size)) = 36.
        // (Upstream's `value_block_count_max <= lsm_table_value_blocks_max` is a looser sanity
        // bound; actual callers pass counts that fit.)
        let index = TableIndex::init(32, 36);
        assert!(index.size <= BLOCK_SIZE as u32);
    }

    /// Builds an index block body per schema and returns (block, schema).
    fn build_index_block(
        used: u32,
        tree_id: u16,
        key_size: u32,
        value_block_count_max: u32,
    ) -> (Vec<u8>, TableIndex) {
        let schema = TableIndex::init(key_size, value_block_count_max);
        let mut block = vec![0_u8; BLOCK];

        let metadata = TableIndexMetadata {
            value_block_count: used,
            value_block_count_max,
            key_size,
            tree_id,
        };
        put_block_header(&mut block, schema.size, 5, BlockType::Index);
        write_metadata(&mut block, &metadata.to_wire());

        for i in 0..used as usize {
            let base_checksum = schema.value_checksums_offset as usize + i * CHECKSUM_SIZE;
            block[base_checksum..base_checksum + 16]
                .copy_from_slice(&(100 + i as u128).to_le_bytes());

            let key = (i as u64).to_le_bytes();
            for (j, byte) in key.iter().enumerate() {
                let min_at = schema.keys_min_offset as usize + i * key_size as usize + j;
                let max_at = schema.keys_max_offset as usize + i * key_size as usize + j;
                block[min_at] = *byte;
                block[max_at] = byte.wrapping_add(1);
            }

            let addr_base = schema.value_addresses_offset as usize + i * ADDRESS_SIZE;
            block[addr_base..addr_base + ADDRESS_SIZE]
                .copy_from_slice(&(200 + i as u64).to_le_bytes());
        }

        (block, schema)
    }

    #[test]
    fn table_index_block_round_trip() {
        let (block, schema_written) = build_index_block(3, 42, 16, 8);

        let schema_read = TableIndex::from_block_without_schema(&block);
        assert_eq!(schema_read, schema_written);

        // With-schema path validates tree_id:
        let _ = schema_written.from_block_with_schema(&block, 42);

        assert_eq!(schema_written.value_blocks_used(&block), 3);
        assert_eq!(schema_written.value_checksum(&block, 0), 100);
        assert_eq!(schema_written.value_checksum(&block, 2), 102);
        assert_eq!(schema_written.value_address(&block, 0), 200);
        assert_eq!(schema_written.value_address(&block, 2), 202);
        // Key size is 16 in this fixture; entries hold `i.to_le_bytes()` with each
        // key_max byte incremented:
        assert_eq!(schema_written.key_min(&block, 1), &{
            let mut k = [0_u8; 16];
            k[..8].copy_from_slice(&1_u64.to_le_bytes());
            k
        });
        assert_eq!(
            schema_written.key_max(&block, 1),
            &[2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        );

        // Padding regions are zeroed in this synthetic block:
        for (start, end) in schema_written.padding(&block) {
            assert!(stdx::zeroed(&block[start..end]), "padding at {start}..{end}");
        }
    }

    #[test]
    fn table_value_layout_and_metadata() {
        let schema = TableValue::init(4, 128);
        assert_eq!(schema.values_offset as usize, HEADER_SIZE);
        assert_eq!(schema.values_size as usize, 4 * 128);
        assert_eq!(schema.padding_size as usize, BLOCK_SIZE - HEADER_SIZE - 4 * 128);

        let mut block = vec![0_u8; BLOCK];
        let metadata =
            TableValueMetadata { value_count_max: 4, value_count: 3, value_size: 128, tree_id: 9 };
        put_block_header(&mut block, (HEADER_SIZE + 3 * 128) as u32, 11, BlockType::Value);
        write_metadata(&mut block, &metadata.to_wire());

        let read_schema = TableValue::from_block(&block);
        assert_eq!(read_schema, schema);
        schema.assert_matching_block_schema(&block, 9);

        assert_eq!(TableValueMetadata::from_wire(&metadata.to_wire()), metadata);
        assert_eq!(read_schema.block_values_used_bytes_len(&block), 3 * 128);
    }

    #[test]
    fn trailer_node_metadata_and_previous() {
        let mut block = vec![0_u8; BLOCK];

        let metadata = TrailerMetadata {
            previous_trailer_block_checksum: 0xABCD,
            previous_trailer_block_address: 77,
        };
        put_block_header(&mut block, (HEADER_SIZE + 64) as u32, 3, BlockType::FreeSet);
        write_metadata(&mut block, &metadata.to_wire());

        TrailerNode::assert_valid_header(&block);
        assert_eq!(TrailerNode::metadata(&block), metadata);
        assert_eq!(
            TrailerNode::previous(&block),
            Some(BlockReference { checksum: 0xABCD, address: 77 })
        );
        assert_eq!(TrailerNode::body(&block).len(), 64);

        // Root trailer has no previous block. Client-sessions trailer bodies are whole
        // entries of `@sizeOf(Header) + @sizeOf(u64)` bytes (upstream TrailerType.item_size):
        let mut root = vec![0_u8; BLOCK];
        put_block_header(
            &mut root,
            (HEADER_SIZE + crate::checkpoint_trailer::TrailerType::ClientSessions.item_size())
                as u32,
            4,
            BlockType::ClientSessions,
        );
        write_metadata(
            &mut root,
            &TrailerMetadata {
                previous_trailer_block_checksum: 0,
                previous_trailer_block_address: 0,
            }
            .to_wire(),
        );
        assert_eq!(TrailerNode::previous(&root), None);
    }

    #[test]
    fn label_bit_packing() {
        use ManifestNode::{Event, Label};
        for level in [0_u8, 1, 31, 63] {
            for event in [Event::Insert, Event::Update, Event::Remove] {
                let label = Label { level, event };
                assert_eq!(Label::from_u8(label.to_u8()), label);
            }
        }
        // Bit layout: level low, event high.
        assert_eq!(Label { level: 5, event: Event::Update }.to_u8(), 0b1000_0101);
    }

    #[test]
    fn manifest_node_table_info_round_trip() {
        let info = TableInfo {
            key_min: core::array::from_fn(|i| i as u8),
            key_max: core::array::from_fn(|i| (255 - i) as u8),
            checksum: u128::MAX - 1,
            address: 123_456_789,
            snapshot_min: 10,
            snapshot_max: 20,
            value_count: 999,
            tree_id: 3,
            label: ManifestNode::Label { level: 63, event: ManifestNode::Event::Remove },
        };
        let wire = info.to_wire();
        assert_eq!(wire.len(), TableInfo::SIZE);
        assert_eq!(TableInfo::from_wire(&wire), info);
    }

    #[test]
    fn manifest_node_constants_and_metadata() {
        // At least one TableInfo entry must fit in a block body:
        const _: () = assert!(ManifestNode::ENTRY_COUNT_MAX > 0);

        let mut block = vec![0_u8; BLOCK];
        let entry_count = 2_u32;
        let metadata = ManifestNode::Metadata {
            previous_manifest_block_checksum: 55,
            previous_manifest_block_address: 66,
            entry_count,
        };
        put_block_header(&mut block, ManifestNode::size(entry_count), 12, BlockType::Manifest);
        write_metadata(&mut block, &metadata.to_wire());

        // Fill two entries:
        for i in 0..entry_count as usize {
            let offset = HEADER_SIZE + i * ManifestNode::ENTRY_SIZE;
            let info = TableInfo {
                key_min: [i as u8 + 1; 32],
                key_max: [i as u8 + 2; 32],
                checksum: 300 + i as u128,
                address: 400 + i as u64,
                snapshot_min: i as u64,
                snapshot_max: 500 + i as u64,
                value_count: 600 + i as u32,
                tree_id: i as u16 + 1,
                label: ManifestNode::Label {
                    level: i as u8,
                    event: if i == 0 {
                        ManifestNode::Event::Insert
                    } else {
                        ManifestNode::Event::Remove
                    },
                },
            };
            block[offset..offset + ManifestNode::ENTRY_SIZE].copy_from_slice(&info.to_wire());
        }

        assert_eq!(ManifestNode::metadata(&block), metadata);
        assert_eq!(
            ManifestNode::previous(&block),
            Some(BlockReference { checksum: 55, address: 66 })
        );

        let tables = ManifestNode::tables(&block);
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].address, 400);
        assert_eq!(tables[1].address, 401);
        assert_eq!(tables[1].label.event, ManifestNode::Event::Remove);
        assert_eq!(tables[0].key_min, [1_u8; 32]);
    }

    #[test]
    #[should_panic(expected = "invalid block type")]
    fn header_from_block_rejects_bad_type() {
        let mut block = vec![0_u8; BLOCK];
        put_block_header(&mut block, (HEADER_SIZE + 8) as u32, 9, BlockType::FreeSet);
        // Patch in an out-of-range block-type ordinal:
        block[240] = 9;
        let _ = header_from_block(&block);
    }
}
