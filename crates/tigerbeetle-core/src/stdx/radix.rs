//! Stable, non-allocating, out-of-place LSD radix sort over unsigned integer keys.
//!
//! Sorts `values` in ascending order by `key_from_value`, using `values_scratch`
//! as an equally sized, disjoint swap buffer. Keys must be an unsigned integer.
//! The sorted result is in the original buffer `values`.
//!
//! The implementation builds per-pass histograms, skips trivial passes (all items
//! in one bucket), and uses a fixed digit width (8 or 11 bits based on `Value` size)
//! to reduce the number of passes. Buffers are swapped after each non-trivial pass;
//! if the number of such passes is odd, results are copied back so `values` holds
//! the output on return.
//!
//! DEVIATIONS vs upstream (`src/stdx/radix.zig`):
//! - Keys cover u8..u128 via [`RadixKey`] (upstream: any unsigned `Int`, incl. u256).
//! - `Value: Clone` is required because safe Rust cannot bitwise-move elements between
//!   two live slices without `unsafe`; every ported table value is a small POD deriving
//!   `Clone`, matching upstream's memcpy semantics.

#![allow(clippy::cast_possible_truncation)] // digit masks are < 2048; lengths < 2^32 (asserted)
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::checked_conversions)] // `<= u32::MAX as usize` mirrors upstream's comptime assert

/// An unsigned integer usable as a radix-sort key (upstream: any unsigned `Int`).
pub trait RadixKey: Copy + Ord + core::fmt::Debug {
    /// Width in bits (upstream: `@bitSizeOf(Key)`).
    const BITS: u32;

    /// Extracts the `bits`-wide digit starting at `shift`
    /// (upstream: `(key >> pass_bit_offset) & radix_mask`).
    fn digit(self, shift: u32, bits: u32) -> usize;
}

macro_rules! impl_radix_key {
    ($($t:ty),*) => {$(
        impl RadixKey for $t {
            const BITS: u32 = <$t>::BITS;

            #[inline]
            fn digit(self, shift: u32, bits: u32) -> usize {
                let mask: $t = ((1_u128 << bits) - 1) as $t;
                ((self >> shift) & mask) as usize
            }
        }
    )*};
}

impl_radix_key!(u8, u16, u32, u64, u128);

/// Stable, ascending radix sort for unsigned integers. The sorted result will be in `values`.
///
/// # Panics
/// Panics if `values` and `scratch` overlap, differ in length, or exceed `u32::MAX` elements
/// (upstream asserts the same).
pub fn sort<Key, Value, K>(values: &mut [Value], scratch: &mut [Value], key_from_value: K)
where
    Key: RadixKey,
    Value: Clone,
    K: Fn(&Value) -> Key,
{
    assert!(disjoint_slices(values, scratch));
    assert_eq!(values.len(), scratch.len());
    assert!(values.len() <= u32::MAX as usize);

    if values.is_empty() {
        return;
    }
    if values.len() <= 32 {
        // Upstream delegates small inputs to `std.sort.insertion` (stable):
        insertion_sort(values, &key_from_value);
        return;
    }
    radix_sort(values, scratch, key_from_value);
}

fn radix_sort<Key, Value, K>(values: &mut [Value], scratch: &mut [Value], key_from_value: K)
where
    Key: RadixKey,
    Value: Clone,
    K: Fn(&Value) -> Key,
{
    // Per-instantiation constants (upstream comptime values):
    // Heuristic: use more bits for larger value sizes to reduce the number of passes.
    let heuristic_bits: u32 = if core::mem::size_of::<Value>() >= 128 { 11 } else { 8 };
    let radix_bits = Key::BITS.min(heuristic_bits);
    let radix_passes = Key::BITS.div_ceil(radix_bits) as usize;
    let radix_partitions = 1_usize << radix_bits;

    let count = values.len() as u32;

    // Upstream comptime assertion:
    //   const { assert(@sizeOf(Histograms) <= 200 * stdx.KiB); }
    assert!(radix_passes * radix_partitions * core::mem::size_of::<u32>() <= 200 * super::KIB);

    // Create histograms per radix pass in a single iteration over `values`.
    let mut histograms = vec![vec![0_u32; radix_partitions]; radix_passes];
    for value in &*values {
        let key = key_from_value(value);
        for (pass, histogram) in histograms.iter_mut().enumerate() {
            let shift = (pass as u32) * radix_bits;
            histogram[key.digit(shift, radix_bits)] += 1;
        }
    }

    let mut target_offsets = vec![0_u32; radix_partitions];
    let mut source_is_values = true;

    for (pass, histogram) in histograms.iter().enumerate() {
        // Determine if a pass is trivial if exactly one partition has all `count` elements.
        let pass_trivial = histogram.contains(&count);

        if !pass_trivial {
            // Build prefix sums.
            let mut next_offset = 0_u32;
            for (partition_id, &partition_count) in histogram.iter().enumerate() {
                target_offsets[partition_id] = next_offset;
                next_offset += partition_count;
            }

            // Partitioning pass; buffers swap roles after each non-trivial pass.
            let shift = (pass as u32) * radix_bits;
            if source_is_values {
                scatter(values, scratch, &mut target_offsets, shift, radix_bits, &key_from_value);
            } else {
                scatter(scratch, values, &mut target_offsets, shift, radix_bits, &key_from_value);
            }
            source_is_values = !source_is_values;
        }
    }

    // Copy the values back into the input buffer `values`.
    if !source_is_values {
        values.clone_from_slice(scratch);
    }
}

/// One partitioning pass (upstream inline loop): moves every element from `source`
/// to its bucket position in `target`, consuming and advancing `offsets`.
fn scatter<Key, Value, K>(
    source: &[Value],
    target: &mut [Value],
    offsets: &mut [u32],
    shift: u32,
    radix_bits: u32,
    key_from_value: &K,
) where
    Key: RadixKey,
    Value: Clone,
    K: Fn(&Value) -> Key,
{
    for value in source {
        let key = key_from_value(value);
        let partition_id = key.digit(shift, radix_bits);

        let index = offsets[partition_id] as usize;
        target[index] = value.clone();
        offsets[partition_id] += 1;
    }
}

/// Stable insertion sort by `key_from_value` (upstream: `std.sort.insertion`).
fn insertion_sort<Key, Value, K>(values: &mut [Value], key_from_value: &K)
where
    Key: Ord,
    K: Fn(&Value) -> Key,
{
    for i in 1..values.len() {
        let mut j = i;
        while j > 0 && key_from_value(&values[j]) < key_from_value(&values[j - 1]) {
            values.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Upstream: `stdx.disjoint_slices` — pointer-range overlap check (no dereference).
fn disjoint_slices<T>(a: &[T], b: &[T]) -> bool {
    let a_start = a.as_ptr() as usize;
    let a_end = a_start.wrapping_add(core::mem::size_of_val(a));
    let b_start = b.as_ptr() as usize;
    let b_end = b_start.wrapping_add(core::mem::size_of_val(b));
    a_end <= b_start || b_end <= a_start
}

// Test fixtures cast indices to key/value fields; sizes are bounded well below 2^32.
#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::{RadixKey, disjoint_slices, insertion_sort, radix_sort};
    use crate::stdx::prng::{Prng, ratio};

    /// Upstream `TestValueType`: `y` ensures values are distinct for stability checks.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct TestValueType<KeyT, const VALUE_LENGTH: usize> {
        pub x: KeyT,
        pub y: u32,

        /// Ensures that values are distinct for the purpose of checking stability.
        padding: [u8; VALUE_LENGTH],
    }

    impl<KeyT: Default, const VALUE_LENGTH: usize> Default for TestValueType<KeyT, VALUE_LENGTH> {
        fn default() -> Self {
            Self { x: KeyT::default(), y: 0, padding: [0; VALUE_LENGTH] }
        }
    }

    impl<KeyT, const VALUE_LENGTH: usize> TestValueType<KeyT, VALUE_LENGTH> {
        pub fn new(x: KeyT, y: u32) -> Self {
            Self { x, y, padding: [0; VALUE_LENGTH] }
        }

        pub fn key_from_value(value: &Self) -> KeyT
        where
            KeyT: Copy,
        {
            value.x
        }

        pub fn set_y(&mut self, y: u32) {
            self.y = y;
        }
    }

    /// Verifies that the order is `ascending` and `stable` (shared by several tests).
    fn assert_ascending_stable<KeyT: Copy + Ord, const N: usize>(
        values: &[TestValueType<KeyT, N>],
    ) {
        for pair in values.windows(2) {
            match pair[0].x.cmp(&pair[1].x) {
                core::cmp::Ordering::Equal => assert!(pair[0].y < pair[1].y),
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Greater => panic!("not ascending"),
            }
        }
    }

    #[test]
    fn radix_sort_smoke() {
        type Value = TestValueType<u8, 0>;

        let mut values = [
            Value::new(3, 0),
            Value::new(2, 0),
            Value::new(3, 1),
            Value::new(1, 0),
            Value::new(5, 0),
        ];
        let values_expected = [
            Value::new(1, 0),
            Value::new(2, 0),
            Value::new(3, 0),
            Value::new(3, 1),
            Value::new(5, 0),
        ];
        let mut values_scratch = [Value::default(); 5];
        radix_sort(&mut values, &mut values_scratch, Value::key_from_value);
        assert_eq!(values_expected, values);
    }

    // Ascending order + stability against an (x, y) baseline, with a large Value
    // payload to exercise the 11-bit radix path (since size_of(Value) >= 128).
    #[test]
    fn radix_sort_ascending_and_stable_on_many_duplicates() {
        type Key = u32;
        type Value = TestValueType<Key, 128>; // >=128 so radix_bits heuristic picks 11

        let n: usize = 2048;

        let mut values: Vec<Value> = vec![Value::default(); n];
        let mut scratch: Vec<Value> = vec![Value::default(); n];

        // Many duplicates; y = original index (used to check stability).
        for (i, v) in values.iter_mut().enumerate() {
            *v = Value::new((i % 257) as Key, i as u32);
        }

        radix_sort(&mut values, &mut scratch, Value::key_from_value);

        assert_ascending_stable::<Key, 128>(&values);
    }

    // All keys equal → every pass is "trivial". Sort should be a no-op on values,
    // keep relative order (stability).
    #[test]
    fn radix_sort_all_equal_keys_preserve_relative_order_stability() {
        type Key = u64;
        type Value = TestValueType<Key, 8>;

        let n: usize = 1024;

        let mut values: Vec<Value> = vec![Value::default(); n];
        let mut scratch: Vec<Value> = vec![Value::default(); n];

        // Fill scratch with a sentinel to detect writes.
        for s in &mut scratch {
            *s = Value::new(u64::MAX, 0xDEAD_BEEF);
        }

        // All keys identical; y = original index to check stability.
        for (i, v) in values.iter_mut().enumerate() {
            *v = Value::new(42, i as u32);
        }

        radix_sort(&mut values, &mut scratch, Value::key_from_value);

        assert_ascending_stable::<Key, 8>(&values);
    }

    /// Per-width bounded key generation for the fuzz test
    /// (upstream: `prng.int_inclusive(Key, min(maxInt(Key), count * 2 - 1))`).
    trait FuzzKey: RadixKey + Default {
        fn generate_capped(prng: &mut Prng, cap: usize) -> Self;
    }

    macro_rules! impl_fuzz_key {
        ($($t:ty, $method:ident);*) => {$(
            impl FuzzKey for $t {
                fn generate_capped(prng: &mut Prng, cap: usize) -> Self {
                    prng.$method(cap.min(<$t>::MAX as usize) as $t)
                }
            }
        )*};
    }

    impl_fuzz_key!(u8, gen_int_inclusive_u8; u16, gen_int_inclusive_u16; u32, gen_int_inclusive_u32; u64, gen_int_inclusive_u64);

    // DEVIATION: upstream fuzzes over {u3, u256} keys; we cover {u8} (fewer bits than one
    // digit) and {u64 with 130-byte values} (largest histogram + multiple passes + bit
    // heuristic), since Prng.int_inclusive supports up to u64.
    #[test]
    fn fuzz_radix_sort_stable() {
        fuzz_radix_sort_for::<u8, 0>(); // Smaller than radix bits.
        fuzz_radix_sort_for::<u64, 130>(); // Largest histogram, requires multiple passes.
    }

    fn fuzz_radix_sort_for<Key: FuzzKey, const N: usize>() {
        let mut prng = Prng::from_seed(92);

        // Explores uneven and even passes to test copy back.
        let values_max = 1_usize << 18;
        let mut values: Vec<TestValueType<Key, N>> = vec![TestValueType::default(); values_max];
        let mut values_scratch: Vec<TestValueType<Key, N>> =
            vec![TestValueType::default(); values_max];

        for _ in 0..64 {
            let values_count = prng.range_inclusive_usize(2, values_max);
            let values = &mut values[..values_count];
            let values_scratch = &mut values_scratch[..values_count];

            // Set up `values`.
            for value in values.iter_mut() {
                *value =
                    TestValueType::new(Key::generate_capped(&mut prng, values_count * 2 - 1), 0);
            }

            // Sort algorithms often optimize the case of already-sorted
            // (or already-reverse-sorted) sub-arrays.
            let partitions_count = prng.range_inclusive_usize(1, values_count.max(64) - 1);

            // The `partition_reverse_probability` is a subset of the partitions sorted by
            // `partition_sort_percent`.
            let partition_sort_probability = ratio(u64::from(prng.gen_int_inclusive_u8(100)), 100);
            let partition_reverse_probability =
                ratio(u64::from(prng.gen_int_inclusive_u8(100)), 100);

            let mut partitions_remaining: usize = partitions_count;
            let mut partition_offset: usize = 0;
            while partition_offset < values_count {
                let partition_size = if partitions_remaining == 1 {
                    values_count - partition_offset
                } else {
                    prng.range_inclusive_usize(1, values_count - partition_offset)
                };

                if prng.chance(partition_sort_probability) {
                    let partition = &mut values[partition_offset..][..partition_size];
                    if prng.chance(partition_reverse_probability) {
                        partition.sort_unstable_by_key(|v| core::cmp::Reverse(v.x));
                    } else {
                        partition.sort_unstable_by_key(|v| v.x);
                    }
                }

                partitions_remaining -= 1;
                partition_offset += partition_size;
            }

            for (i, value) in values.iter_mut().enumerate() {
                value.set_y(i as u32);
            }

            super::sort(values, values_scratch, TestValueType::key_from_value);

            // Verify that the order is `ascending` and `stable`.
            assert_ascending_stable::<Key, N>(values);
        }
    }

    #[test]
    fn insertion_sort_matches_reference() {
        let mut values: [u64; 7] = [5, 1, 4, 1, 5, 9, 2];
        insertion_sort(&mut values, &|v: &u64| *v);
        assert_eq!(values, [1, 1, 2, 4, 5, 5, 9]);
    }

    #[test]
    fn disjoint_slice_detection() {
        let a = [0_u8; 4];
        let b = [0_u8; 4];
        assert!(disjoint_slices(&a, &b));
        assert!(!disjoint_slices(&a[..2], &a[1..]));
    }
}
