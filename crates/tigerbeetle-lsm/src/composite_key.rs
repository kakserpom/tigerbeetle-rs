//! Combines a field (the key prefix) with a timestamp (the primary key).
//! - To keep alignment, it supports either `u64` or `u128` prefixes (which can be truncated
//!   to smaller types to fit the correct field data type).
//! - "Deleted" values are denoted by a tombstone bit in the timestamp.
//! - It also supports composite keys without a prefix (`Field == ()`), which is useful for
//!   indexing flags that are only checked with "exists".
//!
//! Upstream: `src/lsm/composite_key.zig`.
//!
//! DEVIATION: Zig instantiates this per comptime `Field` type (`CompositeKeyType(F)`); this
//! port declares one [`CompositeKey`] implementation struct per supported prefix
//! ([`CompositeKeyUnit`], [`CompositeKey64`], [`CompositeKey128`]), unified by the
//! [`CompositeKey`] trait. The upstream reflection helpers (`is_composite_key`, comptime
//! negative checks against unrelated structs) collapse into "does the type implement
//! [`CompositeKey`]".
//!
//! DEVIATION: upstream's `CompositeKeyType(u128)` key is a native `u256`; Rust has none, so
//! [`CompositeKey128`] uses the private [`U256`] pair below (arithmetic-only; never
//! serialized as-is).

use core::fmt::Debug;

/// Upstream `tombstone_bit`: the most significant bit of the timestamp.
pub const TOMBSTONE_BIT: u64 = 1_u64 << (64 - 1);

/// Operations shared by every composite-key instantiation (upstream members of
/// `CompositeKeyType(Field)`).
pub trait CompositeKey: Copy + Debug + PartialEq {
    /// The key prefix type (upstream `Field`).
    type Field: Copy + Debug;
    /// The combined unsigned key integer (upstream `Key`).
    type Key: Copy + Ord + Debug;

    /// Upstream `sentinel_key`.
    fn sentinel_key() -> Self::Key;
    /// Upstream `key_from_value`.
    fn key_from_value(&self) -> Self::Key;
    /// Upstream `key_prefix`.
    fn key_prefix(key: Self::Key) -> Self::Field;
    /// Upstream `tombstone`.
    fn tombstone(&self) -> bool;
    /// Upstream `tombstone_from_key`.
    ///
    /// # Panics
    /// Panics if `key`'s low word already has the tombstone bit set (upstream assert).
    fn tombstone_from_key(key: Self::Key) -> Self;
}

/// The `void`-field instantiation (`CompositeKeyType(void)`): just the timestamp, sized like
/// a `u64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CompositeKeyUnit {
    /// Present for shape parity with upstream; zero-sized.
    pub field: (),
    /// The most significant bit must be unset as it is used to indicate a tombstone.
    pub timestamp: u64,
}

const _: () = assert!(core::mem::size_of::<CompositeKeyUnit>() == core::mem::size_of::<u64>());
const _: () = assert!(core::mem::align_of::<CompositeKeyUnit>() == core::mem::align_of::<u64>());

impl CompositeKey for CompositeKeyUnit {
    type Field = ();
    type Key = u64;

    fn sentinel_key() -> u64 {
        Self { field: (), timestamp: u64::MAX }.key_from_value()
    }

    fn key_from_value(&self) -> u64 {
        self.timestamp & !TOMBSTONE_BIT
    }

    fn key_prefix(_key: u64) {}

    fn tombstone(&self) -> bool {
        (self.timestamp & TOMBSTONE_BIT) != 0
    }

    fn tombstone_from_key(key: u64) -> Self {
        let timestamp = key;
        assert_eq!(timestamp & TOMBSTONE_BIT, 0);

        Self { field: (), timestamp: timestamp | TOMBSTONE_BIT }
    }
}

/// The `u64`-prefix instantiation (`CompositeKeyType(u64)`), sized like a `u128`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CompositeKey64 {
    pub field: u64,
    /// The most significant bit must be unset as it is used to indicate a tombstone.
    pub timestamp: u64,
}

const _: () = assert!(core::mem::size_of::<CompositeKey64>() == core::mem::size_of::<u128>());
const _: () =
    assert!(core::mem::align_of::<CompositeKey64>() == core::mem::align_of::<CompositeKeyUnit>());

impl CompositeKey for CompositeKey64 {
    type Field = u64;
    type Key = u128;

    fn sentinel_key() -> u128 {
        Self { field: u64::MAX, timestamp: u64::MAX }.key_from_value()
    }

    fn key_from_value(&self) -> u128 {
        ((u128::from(self.field)) << 64) | u128::from(self.timestamp & !TOMBSTONE_BIT)
    }

    fn key_prefix(key: u128) -> u64 {
        (key >> 64) as u64
    }

    fn tombstone(&self) -> bool {
        (self.timestamp & TOMBSTONE_BIT) != 0
    }

    fn tombstone_from_key(key: u128) -> Self {
        #[allow(clippy::cast_possible_truncation)] // low word by construction
        let timestamp = key as u64;
        assert_eq!(timestamp & TOMBSTONE_BIT, 0);

        Self { field: Self::key_prefix(key), timestamp: timestamp | TOMBSTONE_BIT }
    }
}

/// A minimal 256-bit unsigned integer: `(hi: u128, lo: u64)` with derived big-endian
/// ordering. Stands in for Zig's native `u256` (see the module-level deviation); its memory
/// layout is private and not wire/disk compatible.
/// Public only because it names [`CompositeKey128`]'s associated `Key`; treat as internal.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct U256 {
    hi: u128,
    lo: u64,
}

impl U256 {
    pub const MAX: Self = Self { hi: u128::MAX, lo: u64::MAX };

    #[must_use]
    pub const fn from_parts(hi: u128, lo: u64) -> Self {
        Self { hi, lo }
    }

    #[must_use]
    pub const fn hi(self) -> u128 {
        self.hi
    }

    #[must_use]
    pub const fn low_word(self) -> u64 {
        self.lo
    }
}

const _: () = assert!(core::mem::size_of::<U256>() == 32);

impl super::k_way_merge::TournamentKey for U256 {
    const SENTINEL_KEY: Self = Self::MAX;
    const MIN_KEY: Self = Self { hi: 0, lo: 0 };
}

impl tigerbeetle_core::stdx::radix::RadixKey for U256 {
    const BITS: u32 = 256;

    #[allow(clippy::cast_possible_truncation)] // mask fits usize; same pattern as u128 impl
    fn digit(self, shift: u32, bits: u32) -> usize {
        // For 256-bit radix, we need to handle cross-word extraction.
        // Convert to bytes and extract the digit manually.
        assert!(shift + bits <= 256);
        if bits == 0 {
            return 0;
        }
        // Extract the u64-sized digit at the given bit position.
        let byte_offset = (shift / 8) as usize;
        let bit_offset = shift % 8;
        let mut buf = [0u8; 40]; // enough for any aligned u64 extraction
        buf[..16].copy_from_slice(&self.hi.to_le_bytes());
        buf[16..24].copy_from_slice(&self.lo.to_le_bytes());
        // We need `bits` bits starting at `shift`.
        // Extract a u64 from the byte stream at byte_offset, then mask.
        let mut word = [0u8; 8];
        let end = byte_offset + 8;
        if end <= buf.len() {
            word.copy_from_slice(&buf[byte_offset..end]);
        } else {
            let available = buf.len() - byte_offset;
            word[..available].copy_from_slice(&buf[byte_offset..]);
        }
        let value = u64::from_le_bytes(word);
        let mask = if bits >= 64 { u64::MAX } else { (1_u64 << bits) - 1 };
        ((value >> bit_offset) & mask) as usize
    }
}

impl super::manifest::TableKey for U256 {
    const MAX: Self = Self::MAX;
    const ZERO: Self = Self { hi: 0, lo: 0 };

    fn to_bytes(self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(&self.hi.to_le_bytes());
        buf[16..24].copy_from_slice(&self.lo.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut hi_bytes = [0u8; 16];
        hi_bytes.copy_from_slice(&bytes[..16]);
        let mut lo_bytes = [0u8; 8];
        lo_bytes.copy_from_slice(&bytes[16..24]);
        Self::from_parts(u128::from_le_bytes(hi_bytes), u64::from_le_bytes(lo_bytes))
    }

    fn to_sort_key_high(self) -> u64 {
        (self.hi >> 64) as u64
    }
}

/// The `u128`-prefix instantiation (`CompositeKeyType(u128)`), sized like a `u256`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CompositeKey128 {
    pub field: u128,
    /// The most significant bit must be unset as it is used to indicate a tombstone.
    pub timestamp: u64,
    pub padding: u64,
}

const _: () = assert!(core::mem::size_of::<CompositeKey128>() == 2 * core::mem::size_of::<u128>());
const _: () = assert!(core::mem::align_of::<CompositeKey128>() == core::mem::align_of::<u128>());

impl CompositeKey for CompositeKey128 {
    type Field = u128;
    type Key = U256;

    fn sentinel_key() -> U256 {
        Self { field: u128::MAX, timestamp: u64::MAX, padding: 0 }.key_from_value()
    }

    fn key_from_value(&self) -> U256 {
        debug_assert_eq!(self.padding, 0);
        U256::from_parts(self.field, self.timestamp & !TOMBSTONE_BIT)
    }

    fn key_prefix(key: U256) -> u128 {
        key.hi()
    }

    fn tombstone(&self) -> bool {
        debug_assert_eq!(self.padding, 0);
        (self.timestamp & TOMBSTONE_BIT) != 0
    }

    fn tombstone_from_key(key: U256) -> Self {
        let timestamp = key.low_word();
        assert_eq!(timestamp & TOMBSTONE_BIT, 0);

        Self { field: Self::key_prefix(key), timestamp: timestamp | TOMBSTONE_BIT, padding: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompositeKey, CompositeKey64, CompositeKey128, CompositeKeyUnit, TOMBSTONE_BIT};

    #[test]
    fn composite_key_u64_and_u128_field_ordering() {
        // Port of upstream test "composite_key - u64 and u128", Prefix = u128 arm and u64 arm.
        for _ in [false, true] {
            // Prefix = u128:
            {
                let a = CompositeKey128 { field: 1, timestamp: 100, padding: 0 };
                let b = CompositeKey128 { field: 1, timestamp: 101, padding: 0 };
                assert!(a.key_from_value() < b.key_from_value());
            }

            {
                let a = CompositeKey128 { field: 1, timestamp: 100, padding: 0 };
                let b = CompositeKey128 { field: 2, timestamp: 99, padding: 0 };
                assert!(a.key_from_value() < b.key_from_value());
            }

            {
                let a = CompositeKey128 { field: 1, timestamp: 0x64 | TOMBSTONE_BIT, padding: 0 };
                let b = CompositeKey128 { field: 1, timestamp: 100, padding: 0 };
                assert_eq!(a.key_from_value(), b.key_from_value());
            }

            {
                let value = CompositeKey128 { field: 1, timestamp: 100, padding: 0 };
                assert!(!value.tombstone());
            }

            {
                let value = CompositeKey128 { field: 1, timestamp: 100, padding: 0 };
                let tombstone = CompositeKey128::tombstone_from_key(value.key_from_value());
                assert!(tombstone.tombstone());
                assert_eq!(tombstone.timestamp, 0x64 | TOMBSTONE_BIT);
            }

            // Prefix = u64:
            {
                let a = CompositeKey64 { field: 1, timestamp: 100 };
                let b = CompositeKey64 { field: 1, timestamp: 101 };
                assert!(a.key_from_value() < b.key_from_value());
            }

            {
                let a = CompositeKey64 { field: 1, timestamp: 100 };
                let b = CompositeKey64 { field: 2, timestamp: 99 };
                assert!(a.key_from_value() < b.key_from_value());
            }

            {
                let a = CompositeKey64 { field: 1, timestamp: 0x64 | TOMBSTONE_BIT };
                let b = CompositeKey64 { field: 1, timestamp: 100 };
                assert_eq!(a.key_from_value(), b.key_from_value());
            }

            {
                let value = CompositeKey64 { field: 1, timestamp: 100 };
                assert!(!value.tombstone());
            }

            {
                let value = CompositeKey64 { field: 1, timestamp: 100 };
                let tombstone = CompositeKey64::tombstone_from_key(value.key_from_value());
                assert!(tombstone.tombstone());
                assert_eq!(tombstone.timestamp, 0x64 | TOMBSTONE_BIT);
            }
        }
    }

    #[test]
    fn composite_key_void() {
        // Port of upstream test "composite_key - void".
        {
            let a = CompositeKeyUnit { field: (), timestamp: 100 };
            let b = CompositeKeyUnit { field: (), timestamp: 101 };
            assert!(a.key_from_value() < b.key_from_value());
        }

        {
            let a = CompositeKeyUnit { field: (), timestamp: 0x64 | TOMBSTONE_BIT };
            let b = CompositeKeyUnit { field: (), timestamp: 100 };
            assert_eq!(a.key_from_value(), b.key_from_value());
        }

        {
            let value = CompositeKeyUnit { field: (), timestamp: 100 };
            assert!(!value.tombstone());
        }

        {
            let value = CompositeKeyUnit { field: (), timestamp: 100 };
            let tombstone = CompositeKeyUnit::tombstone_from_key(value.key_from_value());
            assert!(tombstone.tombstone());
            assert_eq!(tombstone.timestamp, 0x64 | TOMBSTONE_BIT);
        }
    }

    #[test]
    fn sentinel_keys_match_max_fields_and_timestamps() {
        assert_eq!(CompositeKeyUnit::sentinel_key(), !TOMBSTONE_BIT);
        assert_eq!(
            CompositeKey64::sentinel_key(),
            (u128::from(u64::MAX) << 64) | u128::from(!TOMBSTONE_BIT)
        );
        assert_eq!(
            CompositeKey128::sentinel_key(),
            super::U256 { hi: u128::MAX, lo: !TOMBSTONE_BIT }
        );
    }

    #[test]
    fn key_prefix_round_trips() {
        let value = CompositeKey64 { field: 0x1234_5678_9abc_def0, timestamp: 7 };
        let key = value.key_from_value();
        assert_eq!(<CompositeKey64 as CompositeKey>::key_prefix(key), value.field);

        let value = CompositeKey128 {
            #[allow(clippy::cast_possible_truncation)]
            field: u128::from_be_bytes(core::array::from_fn(|i: usize| (i + 1) as u8)),
            timestamp: 9,
            padding: 0,
        };
        let key = value.key_from_value();
        assert_eq!(<CompositeKey128 as CompositeKey>::key_prefix(key), value.field);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn tombstone_from_key_rejects_pre_tombstoned_low_word() {
        // Craft a raw key whose low word carries the tombstone bit (key_from_value would
        // have stripped it).
        let key: u128 = (u128::from(1_u64) << 64) | u128::from(0x64 | TOMBSTONE_BIT);
        let _ = CompositeKey64::tombstone_from_key(key);
    }
}
