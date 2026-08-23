//! Upstream: `src/lsm/timestamp_range.zig`.

/// Inclusive range of timestamps, as used by LSM manifests and grooves to version values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestampRange {
    /// Inclusive.
    pub min: u64,
    /// Inclusive.
    pub max: u64,
}

impl TimestampRange {
    /// The minimum timestamp allowed (inclusive).
    pub const TIMESTAMP_MIN: u64 = 1;

    /// The maximum timestamp allowed (inclusive).
    ///
    /// It is `u63::MAX` because the most significant bit of the `u64` timestamp is used as
    /// the tombstone flag.
    pub const TIMESTAMP_MAX: u64 = i64::MAX as u64;

    #[must_use]
    pub const fn all() -> Self {
        Self { min: Self::TIMESTAMP_MIN, max: Self::TIMESTAMP_MAX }
    }

    #[must_use]
    pub const fn gte(initial: u64) -> Self {
        Self { min: initial, max: Self::TIMESTAMP_MAX }
    }

    #[must_use]
    pub const fn lte(last: u64) -> Self {
        Self { min: Self::TIMESTAMP_MIN, max: last }
    }

    #[must_use]
    pub const fn valid(timestamp: u64) -> bool {
        timestamp >= Self::TIMESTAMP_MIN && timestamp <= Self::TIMESTAMP_MAX
    }
}

#[cfg(test)]
mod tests {
    use super::TimestampRange;

    #[test]
    fn bounds_match_upstream() {
        assert_eq!(TimestampRange::TIMESTAMP_MIN, 1);
        assert_eq!(TimestampRange::TIMESTAMP_MAX, u64::MAX >> 1);
        assert_eq!(TimestampRange::all(), TimestampRange { min: 1, max: i64::MAX as u64 });
    }

    #[test]
    fn gte_and_lte_pin_one_bound() {
        assert_eq!(
            TimestampRange::gte(7),
            TimestampRange { min: 7, max: TimestampRange::TIMESTAMP_MAX }
        );
        assert_eq!(
            TimestampRange::lte(7),
            TimestampRange { min: TimestampRange::TIMESTAMP_MIN, max: 7 }
        );
    }

    #[test]
    fn valid_rejects_zero_and_tombstone_bit() {
        assert!(!TimestampRange::valid(0));
        assert!(TimestampRange::valid(1));
        assert!(TimestampRange::valid(TimestampRange::TIMESTAMP_MAX));
        // Tombstone flag set.
        assert!(!TimestampRange::valid(TimestampRange::TIMESTAMP_MAX + 1));
    }
}
