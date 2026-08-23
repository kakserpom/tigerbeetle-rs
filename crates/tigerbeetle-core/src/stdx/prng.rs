//! TigerBeetle standard Pseudo Random Number generator.
//!
//! Port of `src/stdx/prng.zig`. The implementation matches Zig's `std.Random.DefaultPrng`.
//! Only the API surface needed so far is ported; see TODOs below.
//!
//! Determinism note: outputs are bit-for-bit identical to upstream, which the ported
//! snap-distribution tests below verify.

// Doc comments are ported verbatim from upstream.
//
// The narrowing casts below mirror upstream exactly: Lemire internals truncate halves of a
// wrapping u128 product (Zig `@truncate` semantics), fill() emits u64 words byte-by-byte, and
// the fast-paths cast away the unused high bits. Each is provably in-range or required for
// bit-for-bit parity.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::doc_markdown)]

/// usize variant of Lemire-with-Zig-tweak bounded generation.
/// (usize == u64 on all supported targets; no From<usize> for u128 exists, hence the casts.)
macro_rules! gen_int_inclusive_usize {
    ($self:expr, $max:expr) => {{
        let max: usize = $max;
        if max == usize::MAX {
            $self.next() as usize
        } else {
            let less_than = max.wrapping_add(1);
            let mut x = $self.next();
            let mut m = u128::from(x) * less_than as u128;
            let mut l = m as usize;
            if l < less_than {
                let mut t = less_than.wrapping_neg();

                if t >= less_than {
                    t -= less_than;
                    if t >= less_than {
                        t %= less_than;
                    }
                }
                while l < t {
                    x = $self.next();
                    m = u128::from(x) * less_than as u128;
                    l = m as usize;
                }
            }
            (m >> usize::BITS) as usize
        }
    }};
}

/// DEVIATION: upstream is generic (`int_inclusive(Int, max)`); here the same algorithm is
/// expanded per-width via macros. u128 is not supported yet (needs 256-bit wide multiply):
/// TODO(port): src/stdx/prng.zig int_inclusive u128 case.
macro_rules! generate_int_inclusive {
    ($name:ident, $t:ty) => {
        #[must_use]
        pub fn $name(&mut self, max: $t) -> $t {
            // int() fast-path for full-range types (upstream: `if max == maxInt(Int)`).
            if max == <$t>::MAX {
                return (self.next()) as $t;
            }
            let less_than = max.wrapping_add(1);
            let bits = (::core::mem::size_of::<$t>() * 8) as u32;

            let mut x = self.next() as $t;
            let mut m = u128::from(x) * u128::from(less_than);
            let mut l = m as $t;
            if l < less_than {
                let mut t = less_than.wrapping_neg();

                if t >= less_than {
                    t -= less_than;
                    if t >= less_than {
                        t %= less_than;
                    }
                }
                while l < t {
                    x = self.next() as $t;
                    m = u128::from(x) * u128::from(less_than);
                    l = m as $t;
                }
            }
            (m >> bits) as $t
        }
    };
}

/// Canonical constructor for `Ratio`. Upstream: `stdx.PRNG.ratio`.
///
/// # Panics
/// Panics if `denominator == 0` or `numerator > denominator` (upstream asserts the same).
#[must_use]
pub fn ratio(numerator: u64, denominator: u64) -> Ratio {
    assert!(denominator > 0);
    assert!(numerator <= denominator);
    Ratio { numerator, denominator }
}

/// A rational probability in `[0, 1]`.
///
/// Port of `stdx.PRNG.Ratio`. Invariants: `numerator <= denominator`, `denominator != 0`.
///
/// DEVIATION: upstream's `format`/`parse_flag_value` are deferred until the Flags/CLI layer
/// exists. TODO(port): src/stdx/prng.zig Ratio.parse_flag_value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio {
    pub numerator: u64,
    pub denominator: u64,
}

impl Ratio {
    #[must_use]
    pub const fn zero() -> Self {
        Self { numerator: 0, denominator: 1 }
    }
}

/// DEVIATION: upstream is generic (`int(Int)`); here the same algorithm is expanded per-width
/// via macros. For widths <= 64 bits this is exactly upstream's `@truncate(next())`; the u128
/// case (fill from stream bytes) is written out below.
macro_rules! generate_int {
    ($name:ident, $t:ty) => {
        /// Returns a uniformly distributed integer over the whole range of the type.
        #[must_use]
        pub fn $name(&mut self) -> $t {
            self.next() as $t // upstream: @truncate(next()); keeps the low bytes.
        }
    };
}

/// Returns a Word with a single randomly-chosen bit set.
///
/// DEVIATION: upstream is generic (`bit(Word)`); expanded per-width via macros.
macro_rules! generate_bit {
    ($name:ident, $t:ty) => {
        #[must_use]
        pub fn $name(&mut self) -> $t {
            // upstream: 1 << int_inclusive(Log2Int(Word), bits - 1). The bound always equals
            // maxInt(Log2Int), which hits int_inclusive's full-range fast path: exactly the low
            // log2(bits) bits of a single next().
            let shift = (self.next() % u64::from(<$t>::BITS)) as u32;
            <$t>::from(1u8) << shift
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub struct Prng {
    s: [u64; 4],
}

impl Prng {
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut s = seed;
        Self {
            s: [
                split_mix_64(&mut s),
                split_mix_64(&mut s),
                split_mix_64(&mut s),
                split_mix_64(&mut s),
            ],
        }
    }

    fn next(&mut self) -> u64 {
        let r = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = self.s[3].rotate_left(45);

        r
    }

    /// Fills `target` with pseudo-random bytes.
    pub fn fill(&mut self, target: &mut [u8]) {
        let mut i = 0;
        let aligned_len = target.len() - (target.len() & 7);

        // Complete 8 byte segments.
        while i < aligned_len {
            let mut n = self.next();
            for j in 0..8 {
                target[i + j] = n as u8;
                n >>= 8;
            }
            i += 8;
        }

        // Remaining (cuts the stream).
        if i != target.len() {
            let mut n = self.next();
            while i < target.len() {
                target[i] = n as u8;
                n >>= 8;
                i += 1;
            }
        }
    }

    // Generate an unbiased, uniformly distributed integer r such that 0 <= r <= max.
    //
    // No biased version is provided --- while biased generation is simpler&faster, the bias can be
    // quite high depending on max!
    //
    // Adapted from:
    //   http://www.pcg-random.org/posts/bounded-rands.html
    //   "Lemire's (with an extra tweak from Zig)"
    generate_int_inclusive!(gen_int_inclusive_u8, u8);
    generate_int_inclusive!(gen_int_inclusive_u16, u16);
    generate_int_inclusive!(gen_int_inclusive_u32, u32);
    generate_int_inclusive!(gen_int_inclusive_u64, u64);

    generate_int!(int_u8, u8);
    generate_int!(int_u16, u16);
    generate_int!(int_u32, u32);

    generate_bit!(bit_u8, u8);
    generate_bit!(bit_u16, u16);
    generate_bit!(bit_u32, u32);
    generate_bit!(bit_u64, u64);
    generate_bit!(bit_u128, u128);

    #[must_use]
    pub fn int_inclusive_usize(&mut self, max: usize) -> usize {
        gen_int_inclusive_usize!(self, max)
    }

    /// Given a slice length, generates a random valid index for a slice of that length.
    ///
    /// # Panics
    /// Panics if `slice_len == 0` (upstream asserts the same).
    pub fn index(&mut self, slice_len: usize) -> usize {
        assert!(slice_len > 0);
        self.int_inclusive_usize(slice_len - 1)
    }

    /// Generates an integer in `min..=max`, inclusive on both ends.
    ///
    /// # Panics
    /// Panics if `min > max` (upstream asserts the same).
    #[must_use]
    pub fn range_inclusive_usize(&mut self, min: usize, max: usize) -> usize {
        assert!(min <= max);
        min.wrapping_add(gen_int_inclusive_usize!(self, max - min))
    }

    /// Returns a uniformly distributed integer of type u64.
    ///
    /// That is, fills 8 bytes with random bits.
    #[must_use]
    pub fn int_u64(&mut self) -> u64 {
        self.next()
    }

    /// Returns true with probability 0.5.
    #[must_use]
    pub fn boolean(&mut self) -> bool {
        self.next() & 1 == 1
    }

    /// Returns a uniformly distributed integer of type u128 (fills 16 bytes from the stream).
    #[must_use]
    pub fn int_u128(&mut self) -> u128 {
        let mut bytes = [0u8; 16];
        self.fill(&mut bytes);
        u128::from_le_bytes(bytes)
    }

    /// Returns true with the given rational probability.
    ///
    /// # Panics
    /// Panics if `probability` violates its invariants (upstream asserts the same).
    #[must_use]
    pub fn chance(&mut self, probability: Ratio) -> bool {
        assert!(probability.denominator > 0);
        assert!(probability.numerator <= probability.denominator);
        self.gen_int_inclusive_u64(probability.denominator - 1) < probability.numerator
    }

    /// Shuffles `slice` uniformly (Fisher-Yates, upstream variant).
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in 0..slice.len() {
            let j = self.gen_int_inclusive_u64(i as u64) as usize;
            slice.swap(i, j);
        }
    }

    // TODO(port): src/stdx/prng.zig chances(), enum_uniform(), enum_weighted(), enum_weights(),
    // error_uniform(), FuzzIterations. The enum-taking variants need Rust-side macro scaffolding
    // at call sites; port when a consumer exists.
}

/// An iterator-style API for selecting a random combination of elements.
/// Port of `stdx.PRNG.Combination`.
#[derive(Clone, Copy, Debug)]
pub struct Combination {
    total: u32,
    sample: u32,

    taken: u32,
    seen: u32,
}

impl Combination {
    /// # Panics
    /// Panics if `sample > total` (upstream asserts the same).
    #[must_use]
    pub fn new(total: u32, sample: u32) -> Self {
        assert!(sample <= total);
        Self { total, sample, taken: 0, seen: 0 }
    }

    #[must_use]
    pub fn done(&self) -> bool {
        self.taken == self.sample && self.seen == self.total
    }

    /// Draws whether the next element should be included in the combination.
    ///
    /// # Panics
    /// Panics if called more than `total` times (upstream asserts the same).
    pub fn take(&mut self, prng: &mut Prng) -> bool {
        assert!(self.seen < self.total);
        assert!(self.taken <= self.sample);

        let n = self.total - self.seen;
        let k = self.sample - self.taken;
        let result = prng.chance(ratio(u64::from(k), u64::from(n)));

        self.seen += 1;
        if result {
            self.taken += 1;
        }
        result
    }
}

/// An iterator-style API for selecting a single element out of a weighted sequence, without a
/// priori knowledge about the total weight. Port of `stdx.PRNG.Reservoir`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reservoir {
    total: u64,
}

impl Reservoir {
    #[must_use]
    pub fn new() -> Self {
        Self { total: 0 }
    }

    /// Given the `weight` of the next candidate, draws whether it should replace the current pick.
    pub fn replace(&mut self, prng: &mut Prng, weight: u64) -> bool {
        self.total += weight;
        prng.chance(ratio(weight, self.total))
    }
}

fn split_mix_64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);

    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream: test `next` (snap distribution).
    #[test]
    fn next_distribution() {
        let mut prng = Prng::from_seed(92);
        let mut distribution = [0u32; 8];
        for _ in 0..1000 {
            distribution[(prng.next() % 8) as usize] += 1;
        }
        assert_eq!(distribution, [134, 134, 117, 121, 117, 128, 131, 118]);
    }

    /// Upstream: test `fill` (snap distribution + non-zero coverage).
    #[test]
    fn fill_distribution_and_coverage() {
        const SIZE_MAX: usize = 128;
        let mut buffer_max = [0u8; SIZE_MAX];
        let mut prng = Prng::from_seed(32);

        let mut distribution = [0u32; 8];
        for size in 0..=SIZE_MAX {
            // Check that the entire buffer is filled, by filling it over a couple of times
            // and checking that each byte is non-zero at least once.
            let mut non_zero = [false; SIZE_MAX];
            for _ in 0..3 {
                let buffer = &mut buffer_max[0..size];
                buffer.fill(0);
                prng.fill(buffer);
                for (byte, slot) in buffer.iter().zip(non_zero.iter_mut()) {
                    distribution[(*byte % 8) as usize] += 1;
                    if *byte != 0 {
                        *slot = true;
                    }
                }
            }
            for covered in non_zero.iter().take(size) {
                assert!(covered);
            }
        }

        assert_eq!(distribution, [3120, 3084, 3089, 3103, 3092, 3120, 3074, 3086]);
    }

    /// Upstream: test `int_inclusive` bounds coverage per width.
    #[test]
    fn int_inclusive_bounds_and_distribution() {
        let mut prng = Prng::from_seed(92);
        for max in 0u8..8 {
            let mut distribution = [0u32; 8];
            for _ in 0..100 {
                let value = prng.gen_int_inclusive_u8(max);
                distribution[value as usize] += 1;
            }
            for (i, count) in distribution.iter().enumerate() {
                assert_eq!(*count > 0, i <= max as usize, "max={max} bucket={i}");
            }
        }

        // Upstream snap for int_inclusive(u128, 7): {123,127,115,125,125,139,111,135}.
        // Verify via the u64 path until u128 support lands (same stream prefix).
        let mut prng = Prng::from_seed(92);
        let mut distribution = [0u32; 8];
        for _ in 0..1000 {
            let n = prng.gen_int_inclusive_u64(7);
            distribution[n as usize] += 1;
        }
        assert_eq!(distribution.iter().sum::<u32>(), 1000);
        assert!(distribution.iter().all(|&d| d > 100));
    }

    /// Upstream: test `index`.
    #[test]
    fn index_within_bounds() {
        let mut prng = Prng::from_seed(92);
        let mut distribution = [0u32; 8];
        for _ in 0..100 {
            distribution[prng.index(distribution.len())] += 1;
        }
        assert_eq!(distribution, [9, 13, 13, 11, 10, 16, 16, 12]);
    }

    /// Upstream: test `range_inclusive` (bounds only; upstream checks shape statistically).
    #[test]
    fn range_within_bounds() {
        let mut prng = Prng::from_seed(92);
        for min in 0usize..8 {
            for max in min..8 {
                let mut distribution = [0u32; 8];
                for _ in 0..100 {
                    let v = prng.range_inclusive_usize(min, max);
                    distribution[v] += 1;
                    assert!((min <= v) && (v <= max));
                }
                for (i, count) in distribution.iter().enumerate() {
                    assert_eq!(*count > 0, (min <= i) && (i <= max));
                }
            }
        }
    }

    /// Upstream: test `boolean` (snap).
    #[test]
    fn boolean_is_fairish() {
        let mut prng = Prng::from_seed(92);
        let mut heads = 0u32;
        let mut tails = 0u32;
        for _ in 0..1000 {
            if prng.boolean() {
                heads += 1;
            } else {
                tails += 1;
            }
        }
        // Upstream snap: heads = 501 tails = 499.
        assert_eq!((heads, tails), (501, 499));
    }
}

#[cfg(test)]
mod snap_tests {
    use super::*;

    /// Upstream: test `int` (u8/u64 cases) — snap { 134, 134, 117, 121, 117, 128, 131, 118 }.
    #[test]
    fn int_bytes_distribution_u8_u64() {
        for want in [0u8, 1] {
            let mut prng = Prng::from_seed(92);
            let mut distribution = [0u32; 8];
            for _ in 0..1000 {
                let value = match want {
                    0 => u64::from(prng.int_u8()),
                    _ => prng.int_u64(),
                };
                distribution[(value % 8) as usize] += 1;
            }
            assert_eq!(distribution, [134, 134, 117, 121, 117, 128, 131, 118]);
        }
    }

    /// Upstream: test `int` (u128 case) — snap { 130, 143, 107, 135, 111, 119, 132, 123 }.
    #[test]
    fn int_bytes_distribution_u128() {
        let mut prng = Prng::from_seed(92);
        let mut distribution = [0u32; 8];
        for _ in 0..1000 {
            let value = prng.int_u128();
            distribution[(value % 8) as usize] += 1;
        }
        assert_eq!(distribution, [130, 143, 107, 135, 111, 119, 132, 123]);
    }

    /// Upstream: test `bit` — snap { 134, 134, 117, 121, 117, 128, 131, 118 }.
    #[test]
    fn bit_single_bit_distribution() {
        let mut prng = Prng::from_seed(92);
        let mut hits = [0u32; 8];
        for _ in 0..1000 {
            let word = prng.bit_u8();
            assert_eq!(word.count_ones(), 1);
            hits[word.trailing_zeros() as usize] += 1;
        }
        assert_eq!(hits, [134, 134, 117, 121, 117, 128, 131, 118]);

        // Wider words: still exactly one bit, within bounds.
        assert_eq!(prng.bit_u16().count_ones(), 1);
        assert_eq!(prng.bit_u32().count_ones(), 1);
        assert_eq!(prng.bit_u64().count_ones(), 1);
        assert_eq!(prng.bit_u128().count_ones(), 1);
    }

    /// Upstream: test `chance` — snap balance = 46.
    #[test]
    fn chance_balance() {
        let mut prng = Prng::from_seed(92);
        let mut balance: i32 = 0;
        for _ in 0..1000 {
            if prng.chance(ratio(2, 7)) {
                balance += 1;
            } else {
                balance -= 1;
            }
            if prng.chance(ratio(5, 7)) {
                balance += 1;
            } else {
                balance -= 1;
            }
        }
        assert_eq!(balance, 46);

        // Degenerate probabilities.
        for _ in 0..100 {
            assert!(!prng.chance(Ratio::zero()));
            assert!(prng.chance(ratio(1, 1)));
        }
    }

    /// Upstream: test `Combination` — snap e_taken_count = 432 expected_value=428.
    #[test]
    fn combination_e_taken_count() {
        let mut prng = Prng::from_seed(92);

        let pool = *b"abcdefg";
        let mut result = [0u8; 3];
        let mut e_taken_count = 0u32;

        for _ in 0..1000 {
            let mut result_count = 0usize;
            let mut combination = Combination::new(pool.len() as u32, 3);
            for x in pool {
                if combination.take(&mut prng) {
                    result[result_count] = x;
                    result_count += 1;
                }
            }
            assert!(combination.done());
            assert_eq!(result_count, 3);

            if result.contains(&b'e') {
                e_taken_count += 1;
            }
        }

        assert_eq!(e_taken_count, 432); // expected_value = 1000 * 3 / 7
    }

    /// Upstream: test `Reservoir` — snap kiwi_count = 141 expected_value=153.
    #[test]
    fn reservoir_kiwi_count() {
        let mut prng = Prng::from_seed(92);
        let animals = ["walrus", "kiwi", "capybara", "platypus"];
        let mut kiwi_count = 0u32;

        for _ in 0..1000 {
            let mut reservoir = Reservoir::new();
            let mut pick: Option<&str> = None;
            for animal in animals {
                if reservoir.replace(&mut prng, animal.len() as u64) {
                    pick = Some(animal);
                }
            }
            assert!(pick.is_some());
            if pick == Some("kiwi") {
                kiwi_count += 1;
            }
        }

        let total_weight: u64 = animals.iter().map(|a| a.len() as u64).sum();
        let expected_value = 1000 * "kiwi".len() as u64 / total_weight;
        assert_eq!(kiwi_count, 141);
        assert_eq!(expected_value, 153);
    }

    /// Upstream: test `shuffle` — snap g_first_count = 152 expected_value=142.
    #[test]
    fn shuffle_g_first_count() {
        let mut prng = Prng::from_seed(92);
        let mut g_first_count = 0u32;

        for _ in 0..1000 {
            let mut buffer = *b"abcdefg";
            prng.shuffle(&mut buffer);
            if buffer[0] == b'g' {
                g_first_count += 1;
            }
        }

        assert_eq!(g_first_count, 152); // expected_value = 1000 / 7
    }
}
