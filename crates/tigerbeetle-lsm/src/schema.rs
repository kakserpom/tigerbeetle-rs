//! On-disk entry schemas for LSM blocks.
//!
//! Upstream: `src/lsm/schema.zig` — this port carries the pieces that do not require the
//! VSR block-header framing ([`crate`] consumers decode whole blocks only through
//! `tigerbeetle-vsr`, which re-exports these types).
//!
//! DEVIATION: upstream lives at `src/lsm/schema.zig` and may import `../vsr.zig`; here the
//! entry-level wire codecs sit in `tigerbeetle-lsm` while block-framed accessors
//! (`ManifestNode.metadata()`/`tables()`/…) remain in `tigerbeetle-vsr/src/schema.rs`
//! because the crate graph is `core ← lsm ← vsr`.
//!
//! DEVIATION: upstream reinterprets block bytes as structs (`mem.bytesAsValue`). Without
//! `unsafe`, all multi-byte fields are decoded explicitly little-endian.

#![allow(clippy::similar_names)] // snapshot_min/snapshot_max mirror upstream names

/// Width of the per-node metadata area inside a grid block's header frame
/// (upstream `Header.Block.metadata_bytes` length).
pub const METADATA_SIZE: usize = 96;

/// DEVIATION: upstream reads fields straight out of packed memory; these decode explicitly
/// little-endian at `offset`, panicking loudly on a short buffer.
fn le_u32_at(bytes: &[u8], offset: usize) -> u32 {
    let mut field = [0_u8; 4];
    field.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(field)
}

fn le_u64_at(bytes: &[u8], offset: usize) -> u64 {
    let mut field = [0_u8; 8];
    field.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(field)
}

fn le_u128_at(bytes: &[u8], offset: usize) -> u128 {
    let mut field = [0_u8; 16];
    field.copy_from_slice(&bytes[offset..offset + 16]);
    u128::from_le_bytes(field)
}

/// Asserts `bytes[offset..offset+len]` are zero (upstream `stdx.zeroed` on reserved areas).
///
/// # Panics
/// Panics if any byte in the area is nonzero.
pub fn assert_area_zeroed(bytes: &[u8], offset: usize, len: usize) {
    assert!(
        bytes[offset..offset + len].iter().all(|&byte| byte == 0),
        "reserved area must be zeroed"
    );
}

/// A Manifest block's body is an array of [`manifest_node::TableInfo`] entries
/// (upstream `ManifestNode`).
// mirrors upstream type name
#[allow(non_snake_case)]
#[allow(clippy::wildcard_imports)]
pub mod manifest_node {
    use super::{METADATA_SIZE, assert_area_zeroed, le_u32_at, le_u64_at, le_u128_at};

    /// Wire size of one [`TableInfo`] entry.
    /// DEVIATION: upstream derives this from `@sizeOf(TableInfo)` (extern struct with
    /// trailing padding); our safe-Rust `TableInfo` carries no padding fields, so the size is
    /// fixed by the on-disk layout instead.
    pub const ENTRY_SIZE: usize = 128;

    pub const ENTRY_COUNT_MAX: usize = (tigerbeetle_core::constants::BLOCK_SIZE
        - tigerbeetle_core::constants::HEADER_SIZE)
        / ENTRY_SIZE;

    /// Bit 7 of the label is reserved to indicate whether the event is an insert or remove
    /// (upstream asserts levels fit u6):
    const _: () = assert!(tigerbeetle_core::constants::LSM_LEVELS <= 0b0011_1111 + 1);

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
            assert_eq!(le_u128_at(bytes, Self::OFFSET_PREV_CHECKSUM_PADDING), 0);
            assert_area_zeroed(bytes, Self::OFFSET_RESERVED, Self::RESERVED_LEN);
            Self {
                previous_manifest_block_checksum: le_u128_at(bytes, Self::OFFSET_PREV_CHECKSUM),
                previous_manifest_block_address: le_u64_at(bytes, Self::OFFSET_PREV_ADDRESS),
                entry_count: le_u32_at(bytes, Self::OFFSET_ENTRY_COUNT),
            }
        }

        /// Serializes into the 96-byte metadata area (padding/reserved zeroed).
        #[must_use]
        pub fn to_wire(self) -> [u8; METADATA_SIZE] {
            let mut bytes = [0_u8; METADATA_SIZE];
            bytes[Self::OFFSET_PREV_CHECKSUM..Self::OFFSET_PREV_CHECKSUM + 16]
                .copy_from_slice(&self.previous_manifest_block_checksum.to_le_bytes());
            bytes[Self::OFFSET_PREV_ADDRESS..Self::OFFSET_PREV_ADDRESS + 8]
                .copy_from_slice(&self.previous_manifest_block_address.to_le_bytes());
            bytes[Self::OFFSET_ENTRY_COUNT..Self::OFFSET_ENTRY_COUNT + 4]
                .copy_from_slice(&self.entry_count.to_le_bytes());
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
        /// Encode label to a packed u8.
        ///
        /// # Panics
        ///
        /// Panics if `level > 63`.
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
                checksum: le_u128_at(bytes, Self::OFFSET_CHECKSUM),
                address: le_u64_at(bytes, Self::OFFSET_ADDRESS),
                snapshot_min: le_u64_at(bytes, Self::OFFSET_SNAPSHOT_MIN),
                snapshot_max: le_u64_at(bytes, Self::OFFSET_SNAPSHOT_MAX),
                key_min,
                key_max,
                value_count: le_u32_at(bytes, Self::OFFSET_VALUE_COUNT),
                tree_id: u16::from_le_bytes([
                    bytes[Self::OFFSET_TREE_ID],
                    bytes[Self::OFFSET_TREE_ID + 1],
                ]),
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
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::METADATA_SIZE;
    use super::manifest_node::{self, Event, TableInfo};

    /// Fully-populated `TableInfo` with distinct bytes in every field to catch any
    /// offset/ordering slip in the wire codecs.
    fn test_table_info() -> TableInfo {
        let key_min = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let key_max = [
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];
        TableInfo {
            key_min,
            key_max,
            checksum: 0xdead_beef_cafe_babe_feed_f00d_0123_4567,
            address: 0x0010_0020_0030_0040,
            snapshot_min: 0x1122_3344_5566_7788,
            snapshot_max: 0x9988_7766_5544_3322,
            value_count: 1234,
            tree_id: 0xBEEF,
            label: manifest_node::Label { level: 63, event: Event::Update },
        }
    }

    #[test]
    fn metadata_round_trip() {
        let metadata = manifest_node::Metadata {
            previous_manifest_block_checksum: 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
            previous_manifest_block_address: 0xdead_beef_cafe_babe,
            entry_count: 4096,
        };

        let wire = metadata.to_wire();
        assert_eq!(wire.len(), METADATA_SIZE);
        let decoded = manifest_node::Metadata::from_wire(&wire);
        assert_eq!(decoded, metadata);
    }

    #[test]
    fn metadata_wire_layout() {
        let metadata = manifest_node::Metadata {
            previous_manifest_block_checksum: 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
            previous_manifest_block_address: 0xdead_beef_cafe_babe,
            entry_count: 0x0102_0304,
        };
        let wire = metadata.to_wire();

        // prev_checksum(16) at offset 0, prev_padding(16) zeroed at 16,
        // prev_address(8) at 32, entry_count(4) at 40, reserved[52] zeroed at 44.
        assert_eq!(&wire[0..16], &metadata.previous_manifest_block_checksum.to_le_bytes());
        assert_eq!(&wire[16..32], &[0_u8; 16]);
        assert_eq!(&wire[32..40], &metadata.previous_manifest_block_address.to_le_bytes());
        assert_eq!(&wire[40..44], &metadata.entry_count.to_le_bytes());
        assert_eq!(&wire[44..METADATA_SIZE], &[0_u8; METADATA_SIZE - 44]);
    }

    #[test]
    #[should_panic(expected = "reserved area must be zeroed")]
    fn metadata_from_wire_panics_on_nonzero_reserved() {
        let mut wire = [0_u8; METADATA_SIZE];
        wire[50] = 1; // inside reserved
        let _ = manifest_node::Metadata::from_wire(&wire);
    }

    #[test]
    #[should_panic(expected = "left == right")]
    fn metadata_from_wire_panics_on_nonzero_padding() {
        let mut wire = [0_u8; METADATA_SIZE];
        wire[20] = 1; // inside prev_checksum padding
        let _ = manifest_node::Metadata::from_wire(&wire);
    }

    #[test]
    fn label_packing_all_events_and_levels() {
        for level in [0_u8, 1, 31, 62, 63] {
            for event in [Event::Reserved, Event::Insert, Event::Update, Event::Remove] {
                let label = manifest_node::Label { level, event };
                let packed = label.to_u8();
                assert_eq!(packed & 0b0011_1111, level, "low 6 bits hold the level");
                assert_eq!(packed >> 6, event as u8, "high 2 bits hold the event");
                assert_eq!(manifest_node::Label::from_u8(packed), label);
            }
        }
    }

    #[test]
    #[should_panic(expected = "self.level <= 0b0011_1111")]
    fn label_to_u8_rejects_level_above_63() {
        let _ = manifest_node::Label { level: 64, event: Event::Insert }.to_u8();
    }

    #[test]
    fn table_info_round_trip() {
        let info = test_table_info();
        let wire = info.to_wire();
        assert_eq!(wire.len(), TableInfo::SIZE);
        assert_eq!(TableInfo::SIZE, manifest_node::ENTRY_SIZE);
        assert_eq!(TableInfo::from_wire(&wire), info);
    }

    #[test]
    fn table_info_wire_layout() {
        let wire = test_table_info().to_wire();

        // key_min(32) at 0, key_max(32) at 32, checksum(16) at 64 (+padding 16 zeroed at 80),
        // address(8) at 96, snapshot_min(8) at 104, snapshot_max(8) at 112,
        // value_count(4) at 120, tree_id(2) at 124, label(1) at 126, reserved(1) at 127.
        assert_eq!(&wire[0..32], &test_table_info().key_min);
        assert_eq!(&wire[32..64], &test_table_info().key_max);
        assert_eq!(&wire[64..80], &0xdead_beef_cafe_babe_feed_f00d_0123_4567u128.to_le_bytes());
        assert_eq!(&wire[80..96], &[0_u8; 16]);
        assert_eq!(&wire[96..104], &0x0010_0020_0030_0040u64.to_le_bytes());
        assert_eq!(&wire[104..112], &0x1122_3344_5566_7788u64.to_le_bytes());
        assert_eq!(&wire[112..120], &0x9988_7766_5544_3322u64.to_le_bytes());
        assert_eq!(&wire[120..124], &1234_u32.to_le_bytes());
        assert_eq!(&wire[124..126], &0xBEEF_u16.to_le_bytes());
        assert_eq!(wire[126], manifest_node::Label { level: 63, event: Event::Update }.to_u8());
        assert_eq!(wire[127], 0);
    }

    #[test]
    #[should_panic(expected = "left == right")]
    fn table_info_from_wire_panics_on_nonzero_checksum_padding() {
        let mut wire = test_table_info().to_wire();
        wire[90] = 1; // checksum padding
        let _ = TableInfo::from_wire(&wire);
    }

    #[test]
    #[should_panic(expected = "left == right")]
    fn table_info_from_wire_panics_on_nonzero_reserved() {
        let mut wire = test_table_info().to_wire();
        wire[127] = 1; // reserved
        let _ = TableInfo::from_wire(&wire);
    }
}
