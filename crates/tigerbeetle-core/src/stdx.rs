//! Port of pieces of `src/stdx/` as they become needed.
//! Upstream: `src/stdx/stdx.zig`, `src/stdx/time_units.zig`.
//!
//! TODO(port): `src/stdx/time_units.zig` `InstantUnix` (needs civil-calendar date conversion),
//! `Duration::parse_flag_value` (needs the Flags/CLI layer).

pub mod bitset;
pub mod bounded_array;
pub mod hash;
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

/// Port of `stdx.fastrange()`: fast alternative to modulo reduction
/// (note: it is *not* the same as modulo).
///
/// See <https://github.com/lemire/fastrange/> and
/// <https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/>.
#[must_use]
pub const fn fastrange(word: u64, p: u64) -> u64 {
    // DEVIATION: `u128::from()` is not yet callable in const fns on this toolchain; casts
    // are equivalent for unsigned widening.
    #[allow(clippy::cast_possible_truncation)]
    let product = (word as u128).wrapping_mul(p as u128);
    (product >> 64) as u64
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

/// Options for [`parse_int`] (upstream `stdx.parse_int` comptime options).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseIntOptions {
    pub base: u32,
    pub allow_leading_zero: bool,
    pub allow_separators: bool,
}

impl Default for ParseIntOptions {
    fn default() -> Self {
        Self { base: 10, allow_leading_zero: false, allow_separators: false }
    }
}

/// Errors from [`parse_int`]; upstream returns the same names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseIntError {
    /// The value overflows the destination integer type.
    Overflow,
    /// A character is not a digit in the given base (including `-`) or a stray `_`.
    InvalidCharacter,
    /// A multi-digit number starts with `0` (disallowed by default).
    LeadingZero,
}

/// Strict-by-default integer parsing, port of `stdx.parse_int` (`src/stdx/stdx.zig:933`).
///
/// Only bases 10 and 16 are supported (upstream asserts the same). `_` separators are accepted
/// only with `allow_separators`, and a multi-digit leading zero is rejected unless
/// `allow_leading_zero` is set, mirroring upstream exactly.
///
/// DEVIATION: upstream type-generates one function per `T` via Zig comptime. This macro emits a
/// same-named function per destination type; an empty string is `InvalidCharacter`, matching
/// `std.fmt.parseInt("", ...)`.
macro_rules! parse_int_impl {
    ($name:ident, $t:ty) => {
        /// Parses `text` as a `$t`, matching upstream `stdx.parse_int`.
        ///
        /// # Errors
        ///
        /// Returns [`ParseIntError::Overflow`] on overflow, [`ParseIntError::InvalidCharacter`]
        /// when a character is not a digit in `options.base` (or separators are disabled), and
        /// [`ParseIntError::LeadingZero`] for a multi-digit number starting with `0` when
        /// `allow_leading_zero` is false.
        #[allow(clippy::cast_possible_truncation)]
        pub fn $name(text: &str, options: ParseIntOptions) -> Result<$t, ParseIntError> {
            assert!(options.base == 10 || options.base == 16);
            if !options.allow_leading_zero && text.len() > 1 && text.starts_with('0') {
                return Err(ParseIntError::LeadingZero);
            }

            let mut accumulator: u128 = 0;
            let mut saw_digit = false;
            for c in text.chars() {
                if c == '_' {
                    if !options.allow_separators {
                        return Err(ParseIntError::InvalidCharacter);
                    }
                    continue;
                }
                let digit = c.to_digit(options.base).ok_or(ParseIntError::InvalidCharacter)?;
                saw_digit = true;
                accumulator = accumulator
                    .checked_mul(u128::from(options.base))
                    .and_then(|value| value.checked_add(u128::from(digit)))
                    .ok_or(ParseIntError::Overflow)?;
            }
            if !saw_digit {
                return Err(ParseIntError::InvalidCharacter);
            }
            <$t>::try_from(accumulator).map_err(|_| ParseIntError::Overflow)
        }
    };
}

parse_int_impl!(parse_int_u8, u8);
parse_int_impl!(parse_int_u16, u16);
parse_int_impl!(parse_int_u32, u32);
parse_int_impl!(parse_int_u64, u64);
parse_int_impl!(parse_int_u128, u128);

/// Measurement-unit of a [`ByteSize`] (upstream `ByteSize.Unit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteSizeUnit {
    Bytes,
    Kib,
    Mib,
    Gib,
    Tib,
}

impl ByteSizeUnit {
    #[must_use]
    pub const fn multiplier(self) -> u64 {
        match self {
            Self::Bytes => 1,
            Self::Kib => KIB as u64,
            Self::Mib => MIB as u64,
            Self::Gib => GIB as u64,
            Self::Tib => TIB as u64,
        }
    }

    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Bytes => "",
            Self::Kib => "KiB",
            Self::Mib => "MiB",
            Self::Gib => "GiB",
            Self::Tib => "TiB",
        }
    }
}

/// A size with its user-specified unit preserved, for CLI arguments.
///
/// Port of `stdx.ByteSize` (`src/stdx/stdx.zig:1032`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteSize {
    pub value: u64,
    pub unit: ByteSizeUnit,
}

/// Parse error for [`ByteSize::parse_flag_value`], carrying the upstream static diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseByteSizeError {
    /// "value exceeds 64-bit unsigned integer:"
    OverflowValue,
    /// "expected a size, but found:"
    InvalidCharacter,
    /// "leading zero disallowed:"
    LeadingZero,
    /// "invalid unit in size, needed KiB, MiB, GiB or TiB:"
    InvalidUnit,
    /// "size in bytes exceeds 64-bit unsigned integer:"
    OverflowBytes,
}

impl ByteSize {
    /// Port of `ByteSize.parse_flag_value` (`src/stdx/stdx.zig:1044`).
    ///
    /// Accepts `"512"`, `"1GiB"`, `"10kib"`, `"1_0KiB"`, ...; the numeric part uses base 10 and
    /// does not allow leading zeros or a `u64` overflow; the unit must be empty or a
    /// case-insensitive IEC size.
    ///
    /// # Panics
    /// Panics on an empty string, matching upstream's assertion.
    ///
    /// # Errors
    /// Returns [`ParseByteSizeError`] with the upstream static diagnostic for an invalid numeric
    /// part or unit.
    pub fn parse_flag_value(text: &str) -> Result<Self, ParseByteSizeError> {
        assert!(!text.is_empty());

        let split_index = text
            .char_indices()
            .find(|(_, c)| !c.is_ascii_digit() && *c != '_')
            .map_or(text.len(), |(index, _)| index);

        let string_amount = &text[..split_index];
        let string_unit = &text[split_index..];

        let amount = parse_int_u64(
            string_amount,
            ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: true },
        )
        .map_err(|err| match err {
            ParseIntError::Overflow => ParseByteSizeError::OverflowValue,
            ParseIntError::InvalidCharacter => ParseByteSizeError::InvalidCharacter,
            ParseIntError::LeadingZero => ParseByteSizeError::LeadingZero,
        })?;

        let unit = if string_unit.is_empty() {
            ByteSizeUnit::Bytes
        } else {
            let match_unit = |tag: ByteSizeUnit, tag_name: &str| {
                tag_name.eq_ignore_ascii_case(string_unit).then_some(tag)
            };
            match_unit(ByteSizeUnit::Kib, "kib")
                .or_else(|| match_unit(ByteSizeUnit::Mib, "mib"))
                .or_else(|| match_unit(ByteSizeUnit::Gib, "gib"))
                .or_else(|| match_unit(ByteSizeUnit::Tib, "tib"))
                .ok_or(ParseByteSizeError::InvalidUnit)?
        };

        // Upstream validates that `amount * unit` does not overflow (`std.math.mul`).
        amount.checked_mul(unit.multiplier()).ok_or(ParseByteSizeError::OverflowBytes)?;

        Ok(Self { value: amount, unit })
    }

    /// Total size in bytes (upstream `ByteSize.bytes`).
    #[must_use]
    pub fn bytes(self) -> u64 {
        self.value * self.unit.multiplier()
    }

    /// The unit suffix as the user specified it (upstream `ByteSize.suffix`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        self.unit.suffix()
    }
}

/// Format a size as its exact IEC magnitude (upstream `stdx.fmt_int_size_bin_exact`).
///
/// Upstream computes this at comptime over constants and asserts the value is a multiple of the
/// chosen unit; here it is a runtime helper for CLI diagnostics.
#[must_use]
pub fn fmt_int_size_bin_exact(value: u64) -> String {
    if value == 0 {
        return "0B".to_owned();
    }
    let mut magnitude = 0_u8;
    let mut value_unit = value;
    while value_unit.is_multiple_of(1024) {
        value_unit /= 1024;
        magnitude += 1;
    }

    let mut result = value_unit.to_string();
    if magnitude == 0 {
        result.push('B');
    } else {
        const MAGNITUDES_IEC: &[u8] = b"KMGTPEZY";
        result.push(char::from(MAGNITUDES_IEC[(magnitude - 1) as usize]));
        result.push_str("iB");
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

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
    fn fastrange_maps_into_range() {
        // Port of upstream expectations: uniform-ish spread over [0, p).
        assert_eq!(fastrange(0, 100), 0);
        assert_eq!(fastrange(u64::MAX, u64::MAX), u64::MAX - 1);
        // (2^64-1 * p) >> 64 == p - 1 for any p.
        for p in [1_u64, 2, 3, 7, 1024, u64::from(u32::MAX), u64::MAX] {
            assert_eq!(fastrange(u64::MAX, p), p.wrapping_sub(1));
            assert!(fastrange(1 << 32, p) < p);
        }
    }

    #[test]
    fn align_forward_rounds_up_to_multiple() {
        assert_eq!(align_forward(1, 4096), 4096);
        assert_eq!(align_forward(4096, 4096), 4096);
        assert_eq!(align_forward(4097, 4096), 8192);
    }

    /// Upstream: test `parse_int` — separators/leading zeros rejected by default.
    #[test]
    fn parse_int_strict_by_default() {
        assert_eq!(parse_int_u8("0", ParseIntOptions::default()), Ok(0));
        assert_eq!(parse_int_u8("255", ParseIntOptions::default()), Ok(255));
        assert_eq!(parse_int_u8("256", ParseIntOptions::default()), Err(ParseIntError::Overflow));
        assert_eq!(
            parse_int_u8("1_0", ParseIntOptions::default()),
            Err(ParseIntError::InvalidCharacter)
        );
        assert_eq!(
            parse_int_u8("000", ParseIntOptions::default()),
            Err(ParseIntError::LeadingZero)
        );
        assert_eq!(
            parse_int_u64(
                "1_0",
                ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: true }
            ),
            Ok(10)
        );
        assert_eq!(
            parse_int_u16(
                "0ff",
                ParseIntOptions { base: 16, allow_leading_zero: true, allow_separators: false }
            ),
            Ok(0xFF)
        );
        assert_eq!(
            parse_int_u16(
                "1f",
                ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: false }
            ),
            Err(ParseIntError::InvalidCharacter)
        );
        assert_eq!(
            parse_int_u64("", ParseIntOptions::default()),
            Err(ParseIntError::InvalidCharacter)
        );
    }

    /// Upstream: `ByteSize.parse_flag_value` test vectors.
    #[test]
    fn byte_size_parse_flag_value() {
        let ok = |text: &str, value: u64, unit: ByteSizeUnit| {
            let size = ByteSize::parse_flag_value(text).unwrap();
            assert_eq!(size.value, value, "{text}");
            assert_eq!(size.unit, unit, "{text}");
            assert_eq!(size.suffix(), unit.suffix(), "{text}");
        };
        ok("0", 0, ByteSizeUnit::Bytes);
        ok("1", 1, ByteSizeUnit::Bytes);
        ok("140737488355328", 140_737_488_355_328, ByteSizeUnit::Bytes);
        ok("128TiB", 128, ByteSizeUnit::Tib);
        ok("1TiB", 1, ByteSizeUnit::Tib);
        ok("10tib", 10, ByteSizeUnit::Tib);
        ok("1GiB", 1, ByteSizeUnit::Gib);
        ok("10gib", 10, ByteSizeUnit::Gib);
        ok("1MiB", 1, ByteSizeUnit::Mib);
        ok("10mib", 10, ByteSizeUnit::Mib);
        ok("1KiB", 1, ByteSizeUnit::Kib);
        ok("10kib", 10, ByteSizeUnit::Kib);
        ok("1_0kib", 10, ByteSizeUnit::Kib);

        let err = |text: &str, expected: ParseByteSizeError| {
            assert_eq!(ByteSize::parse_flag_value(text), Err(expected), "{text}");
        };
        err("18446744073709551616", ParseByteSizeError::OverflowValue);
        err("MiB", ParseByteSizeError::InvalidCharacter);
        err("_MiB", ParseByteSizeError::InvalidCharacter);
        err("10bananas", ParseByteSizeError::InvalidUnit);
        err("10GB", ParseByteSizeError::InvalidUnit);
        err("18446744073709551GiB", ParseByteSizeError::OverflowBytes);
        err("0009GiB", ParseByteSizeError::LeadingZero);

        assert_eq!(ByteSize { value: 8, unit: ByteSizeUnit::Kib }.bytes(), 8 * KIB as u64);
        assert_eq!(ByteSize { value: 1, unit: ByteSizeUnit::Tib }.bytes(), 1 << 40);
    }

    /// Upstream: test `fmt_int_size_bin_exact`.
    #[test]
    fn fmt_int_size_bin_exact_formats_exact_magnitudes() {
        assert_eq!(fmt_int_size_bin_exact(0), "0B");
        assert_eq!(fmt_int_size_bin_exact(128), "128B");
        assert_eq!(fmt_int_size_bin_exact(8 * 1024), "8KiB");
        assert_eq!(fmt_int_size_bin_exact(1025 * 1024), "1025KiB");
        assert_eq!(fmt_int_size_bin_exact(12345 * 1024), "12345KiB");
        assert_eq!(fmt_int_size_bin_exact(42 * 1024 * 1024), "42MiB");
        assert_eq!(fmt_int_size_bin_exact(u64::MAX - 1023), "18014398509481983KiB");
    }
}
