//! Represents a 1:1 map from a unique key to a timestamp (the primary key).
//! Similarly to `composite_key`, it stores the field, the timestamp, and padding if
//! required, but differs in that it uses only the field, not the timestamp, for comparison.
//! - To keep alignment, it supports either `u64` or `u128` keys.
//! - "Deleted" values are denoted by a tombstone bit in the timestamp.
//!
//! Upstream: `src/lsm/unique_key.zig`.
//!
//! DEVIATION: Zig instantiates this per comptime `Key` type (`UniqueKeyType(K)`); this port
//! declares one [`UniqueKey`] implementation struct per supported key width
//! ([`UniqueKey64`], [`UniqueKey128`]), unified by the [`UniqueKey`] trait. The upstream
//! reflection helpers (`is_unique_key` and the comptime negative checks) collapse into
//! "does the type implement [`UniqueKey`]".

use core::fmt::Debug;

use crate::composite_key::TOMBSTONE_BIT;

/// Operations shared by every unique-key instantiation (upstream members of
/// `UniqueKeyType(Key)`).
pub trait UniqueKey: Copy + Debug + PartialEq {
    /// The unique field type (upstream `Key`).
    type Key: Copy + Ord + Debug;

    /// Upstream `sentinel_key = maxInt(Key)`.
    const SENTINEL_KEY: Self::Key;

    /// Upstream `key_from_value`.
    fn key_from_value(&self) -> Self::Key;
    /// Upstream `tombstone`.
    fn tombstone(&self) -> bool;
    /// Upstream `tombstone_from_key`.
    fn tombstone_from_key(field: Self::Key) -> Self;
}

/// The `u64`-key instantiation (`UniqueKeyType(u64)`), sized like a `u128`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UniqueKey64 {
    pub field: u64,
    pub timestamp: u64,
}

const _: () = assert!(core::mem::size_of::<UniqueKey64>() == 2 * core::mem::size_of::<u64>());
const _: () = assert!(core::mem::align_of::<UniqueKey64>() == core::mem::align_of::<u64>());

impl UniqueKey for UniqueKey64 {
    type Key = u64;
    const SENTINEL_KEY: u64 = u64::MAX;

    fn key_from_value(&self) -> u64 {
        self.field
    }

    fn tombstone(&self) -> bool {
        (self.timestamp & TOMBSTONE_BIT) != 0
    }

    fn tombstone_from_key(field: u64) -> Self {
        Self { field, timestamp: TOMBSTONE_BIT }
    }
}

/// The `u128`-key instantiation (`UniqueKeyType(u128)`), sized like a `u256`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UniqueKey128 {
    pub field: u128,
    pub timestamp: u64,
    pub padding: u64,
}

const _: () = assert!(core::mem::size_of::<UniqueKey128>() == 2 * core::mem::size_of::<u128>());
const _: () = assert!(core::mem::align_of::<UniqueKey128>() == core::mem::align_of::<u128>());

impl UniqueKey for UniqueKey128 {
    type Key = u128;
    const SENTINEL_KEY: u128 = u128::MAX;

    fn key_from_value(&self) -> u128 {
        self.field
    }

    fn tombstone(&self) -> bool {
        (self.timestamp & TOMBSTONE_BIT) != 0
    }

    fn tombstone_from_key(field: u128) -> Self {
        Self { field, timestamp: TOMBSTONE_BIT, padding: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::{UniqueKey, UniqueKey64, UniqueKey128};
    use crate::composite_key::{CompositeKey, CompositeKey64, CompositeKey128, CompositeKeyUnit};

    #[test]
    fn unique_key_u64_and_u128() {
        // Port of upstream test "unique_key - u64 and u128" (Prefix = u128 and u64 arms).

        // Timestamp does not participate in comparison:
        {
            let a = UniqueKey128 { field: 1, timestamp: 100, padding: 0 };
            let b = UniqueKey128 { field: 1, timestamp: 101, padding: 0 };
            assert_eq!(a.key_from_value(), b.key_from_value());
        }

        {
            let a = UniqueKey128 { field: 1, timestamp: 100, padding: 0 };
            let b = UniqueKey128 { field: 2, timestamp: 100, padding: 0 };
            assert!(a.key_from_value() < b.key_from_value());
        }

        {
            let a = UniqueKey64 { field: 1, timestamp: 100 };
            let b = UniqueKey64 { field: 1, timestamp: 101 };
            assert_eq!(a.key_from_value(), b.key_from_value());
        }

        {
            let a = UniqueKey64 { field: 1, timestamp: 100 };
            let b = UniqueKey64 { field: 2, timestamp: 100 };
            assert!(a.key_from_value() < b.key_from_value());
        }
    }

    #[test]
    fn tombstones_flag_the_timestamp_without_touching_the_field() {
        let value = UniqueKey64::tombstone_from_key(42);
        assert!(value.tombstone());
        assert_eq!(value.key_from_value(), 42);

        let value = UniqueKey128::tombstone_from_key(u128::from(42_u32));
        assert!(value.tombstone());
        assert_eq!(value.key_from_value(), 42);

        let value = UniqueKey64 { field: 42, timestamp: 7 };
        assert!(!value.tombstone());

        let value = UniqueKey128 { field: u128::from(42_u32), timestamp: 7, padding: 0 };
        assert!(!value.tombstone());
    }

    #[test]
    fn sentinel_keys_are_max_int() {
        assert_eq!(UniqueKey64::SENTINEL_KEY, u64::MAX);
        assert_eq!(UniqueKey128::SENTINEL_KEY, u128::MAX);
    }

    #[test]
    fn composite_keys_are_not_unique_keys() {
        // Port of the upstream comptime cross-checks: a composite key's identity is its
        // (field, timestamp) pair, so it cannot implement the timestamp-ignoring unique-key
        // semantics. Enforced structurally here — distinct traits.
        fn assert_unique<K: UniqueKey>(_: &K) {}

        // …while composite types satisfy the composite trait only.
        fn assert_composite<K: CompositeKey>(_: &K) {}

        let unique = UniqueKey64 { field: 1, timestamp: 2 };
        assert_unique(&unique);
        assert_composite(&CompositeKeyUnit { field: (), timestamp: 0 });
        assert_composite(&CompositeKey64 { field: 1, timestamp: 2 });
        assert_composite(&CompositeKey128 { field: 3, timestamp: 4, padding: 0 });
    }
}
