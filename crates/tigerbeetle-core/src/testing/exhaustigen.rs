//! An utility for exhaustive generation of arbitrary data.
//! Port of `src/testing/exhaustigen.zig`. See:
//! <https://matklad.github.io/2021/11/07/generate-all-the-things.html>
//!
//! On each iteration of a `while (!g.done())` loop, [`Gen`] generates a sequence of numbers.
//! Internally, it remembers this sequence together with the bounds the user requested:
//!
//! ```text
//! value:  3 1 4 4
//! bound:  5 4 4 4
//! ```
//!
//! To advance to the next iteration, Gen finds the smallest sequence of values which is larger
//! than the current one, but still satisfies all the bounds. "Smallest" means that Gen tries to
//! increment the rightmost number.
//!
//! In the above example, the last two "4"s already match the bound, so we can't increment them.
//! However, we can increment the second number, "1", to get 3 2 4 4. This isn't the smallest
//! sequence though, 3 2 0 0 is smaller. So, after incrementing the rightmost number possible,
//! we zero the rest.

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    value: u32,
    bound: u32,
}

/// Upstream: `Gen`.
#[derive(Clone, Debug, Default)]
pub struct Gen {
    started: bool,
    v: [Entry; 32],
    p: usize,
    p_max: usize,
}

impl Gen {
    /// Advances to the next combination, returning `true` once all combinations were generated.
    pub fn done(&mut self) -> bool {
        if !self.started {
            self.started = true;
            return false;
        }
        let mut i = self.p_max;
        while i > 0 {
            i -= 1;
            if self.v[i].value < self.v[i].bound {
                self.v[i].value += 1;
                self.p_max = i + 1;
                self.p = 0;
                return false;
            }
        }
        true
    }

    // DEVIATION: upstream names this `gen`, a reserved keyword in Rust 2024.
    fn next_value(&mut self, bound: u32) -> u32 {
        assert!(self.p < self.v.len());
        if self.p == self.p_max {
            self.v[self.p] = Entry { value: 0, bound: 0 };
            self.p_max += 1;
        }
        self.p += 1;
        self.v[self.p - 1].bound = bound;
        self.v[self.p - 1].value
    }

    /// A uniformly enumerated integer in `[0, bound]` across iterations.
    pub fn int_inclusive(&mut self, bound: u32) -> u32 {
        self.next_value(bound)
    }

    /// A uniformly enumerated integer in `[min, max]` across iterations.
    ///
    /// # Panics
    /// Panics if `min > max` (upstream asserts).
    pub fn range_inclusive(&mut self, min: u32, max: u32) -> u32 {
        assert!(min <= max);
        min + self.int_inclusive(max - min)
    }

    /// Enumerates all permutations of `slice` across iterations (Fisher-Yates shape).
    // DEVIATION: upstream is generic over any unsigned length; here indices pass through the
    // u32-only generator, and slice lengths in exhaustive tests are tiny.
    #[allow(clippy::cast_possible_truncation)]
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in 0..slice.len() {
            let j = self.int_inclusive(i as u32) as usize;
            slice.swap(i, j);
        }
    }

    /// An index into `slice_len`, enumerating `[0, slice_len - 1]`.
    ///
    /// # Panics
    /// Panics if `slice_len == 0` (upstream asserts).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn index(&mut self, slice_len: usize) -> usize {
        assert!(slice_len > 0);
        self.int_inclusive(slice_len as u32 - 1) as usize
    }

    /// Enumerates the values of `values` (upstream: `std.enums.values`).
    pub fn enum_value<E: Copy>(&mut self, values: &[E]) -> E {
        values[self.index(values.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::Gen;

    #[test]
    fn generate_all_permutations() {
        let mut g = Gen::default();
        let mut permutation_count: u32 = 0;

        // This loop imperatively generates all permutations of "abcd":
        while !g.done() {
            let mut pool_buffer = *b"abcd";
            let mut permutation = [0_u8; 4];

            for (i, slot) in permutation.iter_mut().enumerate() {
                let pool_len = pool_buffer.len() - i;

                // Pick an enumerated index into the pool, append it to the permutation,
                // and swap-remove it from the pool.
                let pool_index = g.index(pool_len);
                *slot = pool_buffer[pool_index];
                pool_buffer[pool_index] = pool_buffer[pool_len - 1];
            }
            permutation_count += 1;
        }

        // Verify that we indeed generated n! permutations.
        let mut factorial: usize = 1;
        for n in 1..5 {
            factorial *= n;
        }

        assert_eq!(permutation_count as usize, factorial);
    }

    #[test]
    fn shuffle_enumerates_all_permutations() {
        let mut n_factorial: u32 = 1;
        // DEVIATION: upstream uses a comptime-sized `[n]u8` per iteration; here one buffer is
        // sliced to length n instead.
        for n in 0_u32..5 {
            let mut g = Gen::default();
            let mut count: u32 = 0;
            while !g.done() {
                let mut array = [0_u8; 4];
                g.shuffle(&mut array[..n as usize]);
                count += 1;
            }
            assert_eq!(count, n_factorial);
            n_factorial *= n + 1;
        }
    }
}
