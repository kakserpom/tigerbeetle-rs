//! Port of pieces of `src/stdx/` as they become needed.
//! Upstream: `src/stdx/stdx.zig`, `src/stdx/time_units.zig`.
//!
//! TODO(port): `src/stdx/time_units.zig` `InstantUnix` (needs civil-calendar date conversion),
//! `Duration::parse_flag_value` (needs the Flags/CLI layer).

pub mod bitset;
pub mod bounded_array;
pub mod prng;
pub mod radix;
pub mod ring_buffer;
pub mod stack;

/// Upstream: `src/stdx/stdx.zig` `pub const radix_sort = @import("radix.zig").sort`.
pub use radix::sort as radix_sort;

/// Upstream: `src/stdx/stdx.zig` KiB/MiB/GiB/TiB.
pub const KIB: usize = 1 << 10;
pub const MIB: usize = 1 << 20;
pub const GIB: usize = 1 << 30;
pub const TIB: usize = 1 << 40;

// Upstream uses std.time.* constants.
pub const NS_PER_US: u64 = 1_000;
pub const NS_PER_MS: u64 = 1_000_000;
pub const NS_PER_S: u64 = 1_000_000_000;
pub const NS_PER_MINUTE: u64 = 60 * NS_PER_S;
pub const NS_PER_HOUR: u64 = 60 * NS_PER_MINUTE;
pub const NS_PER_DAY: u64 = 24 * NS_PER_HOUR;

/// A moment in monotonic time not anchored to any particular epoch.
///
/// The absolute value of `ns` is meaningless, but it is possible to compute `Duration` between
/// two `Instant`s sourced from the same clock.
///
/// Port of `stdx.Instant` (`src/stdx/time_units.zig`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant {
    pub ns: u64,
}

impl Instant {
    #[must_use]
    pub const fn add(self, duration: Duration) -> Self {
        Self { ns: self.ns + duration.ns }
    }

    /// # Panics
    /// Panics if `now < self`, i.e. if time went backwards (upstream asserts the same).
    #[must_use]
    pub fn elapsed(self, now: Self) -> Duration {
        assert!(now.ns >= self.ns);
        Duration { ns: now.ns - self.ns }
    }
}

/// Non-negative time difference between two `Instant`s.
///
/// Port of `stdx.Duration` (`src/stdx/time_units.zig`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration {
    pub ns: u64,
}

impl Duration {
    #[must_use]
    pub const fn us(amount_us: u64) -> Self {
        Self { ns: amount_us * NS_PER_US }
    }

    #[must_use]
    pub const fn ms(amount_ms: u64) -> Self {
        Self { ns: amount_ms * NS_PER_MS }
    }

    #[must_use]
    pub const fn seconds(amount_seconds: u64) -> Self {
        Self { ns: amount_seconds * NS_PER_S }
    }

    #[must_use]
    pub const fn minutes(amount_minutes: u64) -> Self {
        Self { ns: amount_minutes * NS_PER_MINUTE }
    }

    /// Duration in microseconds (μs), one millionth of a second.
    #[must_use]
    pub const fn to_us(self) -> u64 {
        self.ns / NS_PER_US
    }

    /// Duration in milliseconds (ms), one thousandth of a second.
    #[must_use]
    pub const fn to_ms(self) -> u64 {
        self.ns / NS_PER_MS
    }

    #[must_use]
    pub const fn to_ns(self) -> u64 {
        self.ns
    }

    #[must_use]
    pub const fn min(lhs: Self, rhs: Self) -> Self {
        Self { ns: umin(lhs.ns, rhs.ns) }
    }

    #[must_use]
    pub const fn max(lhs: Self, rhs: Self) -> Self {
        Self { ns: umax(lhs.ns, rhs.ns) }
    }

    /// # Panics
    /// Panics if `clamp_min > clamp_max` (upstream asserts the same).
    #[must_use]
    pub const fn clamp(self, clamp_min: Self, clamp_max: Self) -> Self {
        assert!(clamp_min.ns <= clamp_max.ns);
        Self { ns: uclamp(self.ns, clamp_min.ns, clamp_max.ns) }
    }
}

const fn umin(a: u64, b: u64) -> u64 {
    if a < b { a } else { b }
}

const fn umax(a: u64, b: u64) -> u64 {
    if a > b { a } else { b }
}

const fn uclamp(v: u64, lo: u64, hi: u64) -> u64 {
    umin(umax(v, lo), hi)
}

/// Port of `stdx.div_ceil()`: division, rounding up.
#[must_use]
pub const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Port of `std.mem.alignForward`: round `value` up to the next multiple of `alignment`.
#[must_use]
pub const fn align_forward(value: usize, alignment: usize) -> usize {
    div_ceil(value, alignment) * alignment
}

/// Port of `stdx.zeroed` (byte-array case): whether every byte is zero.
///
/// TODO(port): `src/stdx.zig` `zeroed()` is generic over any type; add typed variants as needed.
#[must_use]
pub fn zeroed(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream: test "Instant/Duration".
    #[test]
    fn instant_duration() {
        let instant_1 = Instant { ns: 100 * NS_PER_DAY };
        let instant_2 = Instant { ns: 100 * NS_PER_DAY + NS_PER_S };
        assert_eq!(instant_1.elapsed(instant_1).ns, 0);
        assert_eq!(instant_1.elapsed(instant_2).ns, NS_PER_S);

        let duration = instant_1.elapsed(instant_2);
        assert_eq!(duration.ns, 1_000_000_000);
        assert_eq!(duration.to_us(), 1_000_000);
        assert_eq!(duration.to_ms(), 1_000);

        assert_eq!(Duration::ms(1).ns, NS_PER_MS);
        assert_eq!(Duration::seconds(1).ns, NS_PER_S);
        assert_eq!(Duration::minutes(1).ns, NS_PER_MINUTE);
    }

    #[test]
    fn duration_min_max_clamp() {
        let a = Duration::seconds(1);
        let b = Duration::ms(1500);
        assert_eq!(Duration::min(a, b), a);
        assert_eq!(Duration::max(a, b), b);
        assert_eq!(b.clamp(a, Duration::seconds(2)), b);
        assert_eq!(a.clamp(Duration::ms(1100), Duration::seconds(2)), Duration::ms(1100));
    }

    #[test]
    fn div_ceil_rounds_up() {
        assert_eq!(div_ceil(10, 5), 2);
        assert_eq!(div_ceil(11, 5), 3);
        assert_eq!(div_ceil(0, 5), 0);
    }

    #[test]
    fn align_forward_rounds_up_to_multiple() {
        assert_eq!(align_forward(1, 4096), 4096);
        assert_eq!(align_forward(4096, 4096), 4096);
        assert_eq!(align_forward(4097, 4096), 8192);
    }
}
