//! Binary searches over sorted values.
//! Port of `src/lsm/binary_search.zig`.
//!
//! DEVIATION: upstream's `Config.prefetch` issues `@prefetch` hints via pointer
//! arithmetic; stable safe Rust has no prefetch intrinsic, so the hint is dropped
//! (pure performance annotation, no semantic effect).

#![allow(clippy::cast_possible_truncation)] // slice lengths are <= u32::MAX by contract
#![allow(clippy::missing_panics_doc)] // upstream asserts invariants; documented on the core fn

/// Upstream: `Config.mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    LowerBound,
    UpperBound,
}

/// Upstream: `Config`. The `prefetch` flag is dropped — see module docs.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub mode: Mode,
}

impl Default for Config {
    fn default() -> Self {
        Self { mode: Mode::LowerBound }
    }
}

/// Returns either the index of the value equal to `key`,
/// or if there is no such value then the index where `key` would be inserted.
///
/// In other words, return `i` such that both:
/// * key <= key_from_value(values[i]) or i == values.len
/// * key_from_value(values[i-1]) <= key or i == 0
///
/// If `values` contains duplicated matches, then returns
/// the first index when `config.mode == Mode::LowerBound`,
/// or the last index when `config.mode == Mode::UpperBound`.
/// This invariant can be expressed as:
/// * key_from_value(values[i-1]) < key or i == 0 when `mode == LowerBound`.
/// * key < key_from_value(values[i+1]) or i == values.len when `mode == UpperBound`.
///
/// Expects `values` to be sorted by key.
/// Doesn't perform the extra key comparison to determine if the match is exact.
///
/// # Panics
/// Panics if internal invariants are violated (upstream asserts under `constants.verify`,
/// which is always on in this port).
pub fn binary_search_values_upsert_index<Key, Value, K>(
    key_from_value: &K,
    values: &[Value],
    key: Key,
    config: Config,
) -> u32
where
    Key: Ord + Copy + core::fmt::Debug,
    K: Fn(&Value) -> Key,
{
    if values.is_empty() {
        return 0;
    }

    let mode = config.mode;
    let mut offset: usize = 0;
    let mut length: usize = values.len();
    while length > 1 {
        assert!(
            offset == 0
                || match mode {
                    Mode::LowerBound => key_from_value(&values[offset - 1]) < key,
                    Mode::UpperBound => key_from_value(&values[offset - 1]) <= key,
                }
        );
        assert!(
            offset + length == values.len()
                || match mode {
                    Mode::LowerBound => key <= key_from_value(&values[offset + length]),
                    Mode::UpperBound => key < key_from_value(&values[offset + length]),
                }
        );

        let half = length / 2;
        let mid = offset + half;

        // For exact matches, takes the first half if `mode == LowerBound`,
        // or the second half if `mode == UpperBound`.
        let take_upper_half = match mode {
            Mode::LowerBound => key_from_value(&values[mid]) < key,
            Mode::UpperBound => key_from_value(&values[mid]) <= key,
        };

        if take_upper_half {
            offset = mid;
        }

        length -= half;
    }

    assert_eq!(length, 1);

    assert!(
        offset == 0
            || match mode {
                Mode::LowerBound => key_from_value(&values[offset - 1]) < key,
                Mode::UpperBound => key_from_value(&values[offset - 1]) <= key,
            }
    );
    assert!(
        offset + length == values.len()
            || match mode {
                Mode::LowerBound => key <= key_from_value(&values[offset + length]),
                Mode::UpperBound => key < key_from_value(&values[offset + length]),
            }
    );

    offset += u32::from(key_from_value(&values[offset]) < key) as usize;

    assert!(
        offset == 0
            || match mode {
                Mode::LowerBound => key_from_value(&values[offset - 1]) < key,
                Mode::UpperBound => key_from_value(&values[offset - 1]) <= key,
            }
    );
    assert!(
        offset >= values.len() - 1
            || match mode {
                Mode::LowerBound => key <= key_from_value(&values[offset + 1]),
                Mode::UpperBound => key < key_from_value(&values[offset + 1]),
            }
    );
    assert!(offset == values.len() || key <= key_from_value(&values[offset]));

    // values.len() <= u32::MAX by construction (upstream: @intCast).
    u32::try_from(offset).unwrap_or(u32::MAX)
}

/// Upstream: `binary_search_keys_upsert_index`.
pub fn binary_search_keys_upsert_index<Key: Ord + Copy + core::fmt::Debug>(
    keys: &[Key],
    key: Key,
    config: Config,
) -> u32 {
    binary_search_values_upsert_index(&|k: &Key| *k, keys, key, config)
}

/// Upstream: `BinarySearchResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinarySearchResult {
    pub index: u32,
    pub exact: bool,
}

/// Returns the value equal to `key`, if any, according to `config.mode` tie-breaking.
pub fn binary_search_values<'a, Key, Value, K>(
    key_from_value: &K,
    values: &'a [Value],
    key: Key,
    config: Config,
) -> Option<&'a Value>
where
    Key: Ord + Copy + core::fmt::Debug,
    K: Fn(&Value) -> Key,
{
    let index = binary_search_values_upsert_index(key_from_value, values, key, config);
    let exact = (index as usize) < values.len() && key_from_value(&values[index as usize]) == key;

    if exact {
        let value = &values[index as usize];
        assert_eq!(key, key_from_value(value));
        Some(value)
    } else {
        // TODO(port): figure out how to fuzz this without causing asymptotic
        // slowdown in all fuzzers.
        None
    }
}

/// Upstream: `binary_search_keys`.
#[must_use]
pub fn binary_search_keys<Key: Ord + Copy + core::fmt::Debug>(
    keys: &[Key],
    key: Key,
    config: Config,
) -> BinarySearchResult {
    let index = binary_search_keys_upsert_index(keys, key, config);
    BinarySearchResult {
        index,
        exact: (index as usize) < keys.len() && keys[index as usize] == key,
    }
}

/// Upstream: `BinarySearchRangeUpsertIndexes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinarySearchRangeUpsertIndexes {
    pub start: u32,
    pub end: u32,
}

/// Same semantics of `binary_search_values_upsert_index`:
/// Returns either the indexes of the values equal to `key_min` and `key_max`,
/// or the indexes where they would be inserted.
///
/// Expects `values` to be sorted by key.
/// If `values` contains duplicated matches, then returns
/// the first index for `key_min` and the last index for `key_max`.
///
/// Doesn't perform the extra key comparison to determine if the match is exact.
pub fn binary_search_values_range_upsert_indexes<Key, Value, K>(
    key_from_value: &K,
    values: &[Value],
    key_min: Key,
    key_max: Key,
) -> BinarySearchRangeUpsertIndexes
where
    Key: Ord + Copy + core::fmt::Debug,
    K: Fn(&Value) -> Key,
{
    assert!(key_min <= key_max);

    let start = binary_search_values_upsert_index(
        key_from_value,
        values,
        key_min,
        Config { mode: Mode::LowerBound },
    );

    if start == values.len() as u32 {
        return BinarySearchRangeUpsertIndexes { start, end: start };
    }

    let end = binary_search_values_upsert_index(
        key_from_value,
        &values[start as usize..],
        key_max,
        Config { mode: Mode::UpperBound },
    );

    BinarySearchRangeUpsertIndexes { start, end: start + end }
}

/// Upstream: `binary_search_keys_range_upsert_indexes`.
pub fn binary_search_keys_range_upsert_indexes<Key: Ord + Copy + core::fmt::Debug>(
    keys: &[Key],
    key_min: Key,
    key_max: Key,
) -> BinarySearchRangeUpsertIndexes {
    binary_search_values_range_upsert_indexes(&|k: &Key| *k, keys, key_min, key_max)
}

/// Upstream: `BinarySearchRange`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinarySearchRange {
    pub start: u32,
    pub count: u32,
}

/// Returns the index of the first value greater than or equal to `key_min` and
/// the count of elements until the last value less than or equal to `key_max`.
///
/// Expects `values` to be sorted by key.
/// The result is always safe for slicing using the `values[start..][..count]` idiom,
/// even when no elements are matched.
pub fn binary_search_values_range<Key, Value, K>(
    key_from_value: &K,
    values: &[Value],
    key_min: Key,
    key_max: Key,
) -> BinarySearchRange
where
    Key: Ord + Copy + core::fmt::Debug,
    K: Fn(&Value) -> Key,
{
    let upsert_indexes =
        binary_search_values_range_upsert_indexes(key_from_value, values, key_min, key_max);

    if upsert_indexes.start == values.len() as u32 {
        return BinarySearchRange { start: upsert_indexes.start.saturating_sub(1), count: 0 };
    }

    let inclusive = u32::from(
        (upsert_indexes.end as usize) < values.len()
            && key_max == key_from_value(&values[upsert_indexes.end as usize]),
    );
    BinarySearchRange {
        start: upsert_indexes.start,
        count: upsert_indexes.end - upsert_indexes.start + inclusive,
    }
}

/// Upstream: `binary_search_keys_range`.
pub fn binary_search_keys_range<Key: Ord + Copy + core::fmt::Debug>(
    keys: &[Key],
    key_min: Key,
    key_max: Key,
) -> BinarySearchRange {
    binary_search_values_range(&|k: &Key| *k, keys, key_min, key_max)
}

#[cfg(test)]
mod tests {
    use super::{
        BinarySearchRange, BinarySearchResult, Config, Mode, binary_search_keys,
        binary_search_keys_range,
    };
    use tigerbeetle_core::stdx::prng::Prng;
    use tigerbeetle_core::testing::fuzz::random_int_exponential;

    /// Upstream reference scan building the expected result for a single-key search.
    fn expect_for(keys: &[u32], target_key: u32, mode: Mode) -> BinarySearchResult {
        let mut expect = BinarySearchResult { index: 0, exact: false };
        for (i, &key) in keys.iter().enumerate() {
            match key.cmp(&target_key) {
                core::cmp::Ordering::Less => expect.index = i as u32 + 1,
                core::cmp::Ordering::Equal => {
                    expect.index = i as u32;
                    expect.exact = true;
                    if mode == Mode::LowerBound {
                        break;
                    }
                }
                core::cmp::Ordering::Greater => break,
            }
        }
        expect
    }

    fn exhaustive_search(keys_count: u32, mode: Mode) {
        let mut keys = vec![0_u32; keys_count as usize];
        for (i, key) in keys.iter_mut().enumerate() {
            *key = 7 * i as u32 + 3;
        }

        for target_key in 0..keys_count + 13 {
            let expect = expect_for(&keys, target_key, mode);
            let actual = binary_search_keys(&keys, target_key, Config { mode });
            assert_eq!(expect, actual, "target {target_key}");
        }
    }

    fn explicit_search<const N: usize, const M: usize>(
        keys: &[u32],
        target_keys: &[u32; M],
        expected_results: [BinarySearchResult; M],
        mode: Mode,
    ) {
        let _ = N;
        assert_eq!(target_keys.len(), expected_results.len());

        for (i, &target_key) in target_keys.iter().enumerate() {
            let expect = expected_results[i];
            let actual = binary_search_keys(keys, target_key, Config { mode });
            assert_eq!(expect, actual, "target {target_key}");
        }
    }

    fn random_sequence(prng: &mut Prng, iter: usize) -> Vec<u32> {
        let keys_count = (1_000_000_usize).min(random_int_exponential(prng, iter));

        let mut keys: Vec<u32> =
            (0..keys_count).map(|_| random_int_exponential(prng, 100_u32)).collect();
        keys.sort_unstable();
        keys
    }

    fn random_search(prng: &mut Prng, iter: usize, mode: Mode) {
        let keys = random_sequence(prng, iter);

        let target_key = random_int_exponential(prng, 100_u32);

        let expect = expect_for(&keys, target_key, mode);

        let actual = binary_search_keys(&keys, target_key, Config { mode });

        assert_eq!(expect, actual);
    }

    fn explicit_range_search(
        sequence: &[u32],
        key_min: u32,
        key_max: u32,
        expected: BinarySearchRange,
    ) {
        let actual = binary_search_keys_range(sequence, key_min, key_max);

        assert_eq!(expected.start, actual.start);
        assert_eq!(expected.count, actual.count);

        // Make sure that the index is valid for slicing using the [start..][..count] idiom:
        let expected_slice = &sequence[expected.start as usize..][..expected.count as usize];
        let actual_slice = &sequence[actual.start as usize..][..actual.count as usize];
        assert_eq!(expected_slice, actual_slice);
    }

    enum Target {
        Min,
        Max,
    }

    fn random_range_search(prng: &mut Prng, iter: usize) {
        let keys = random_sequence(prng, iter);

        let target_range = {
            // Cover many combinations of key_min, key_max:
            let mut key_min = if !keys.is_empty() && prng.boolean() {
                prng.range_inclusive_usize(keys[0] as usize, keys[keys.len() - 1] as usize) as u32
            } else {
                random_int_exponential(prng, 100_u32)
            };

            let mut key_max = if !keys.is_empty() && prng.boolean() {
                prng.range_inclusive_usize(keys[0] as usize, keys[keys.len() - 1] as usize) as u32
            } else if prng.boolean() {
                key_min
            } else {
                random_int_exponential(prng, 100_u32)
            };

            if key_max < key_min {
                core::mem::swap(&mut key_min, &mut key_max);
            }
            assert!(key_min <= key_max);

            (key_min, key_max)
        };
        let (key_min, key_max) = target_range;

        let mut expect = BinarySearchRange { start: 0, count: 0 };
        let mut key_target = Target::Min;
        for &key in &keys {
            if matches!(key_target, Target::Min) {
                match key.cmp(&key_min) {
                    core::cmp::Ordering::Less => {
                        if expect.start < keys.len().saturating_sub(1) as u32 {
                            expect.start += 1;
                        }
                    }
                    core::cmp::Ordering::Greater | core::cmp::Ordering::Equal => {
                        key_target = Target::Max;
                    }
                }
            }

            if matches!(key_target, Target::Max) {
                match key.cmp(&key_max) {
                    core::cmp::Ordering::Less | core::cmp::Ordering::Equal => expect.count += 1,
                    core::cmp::Ordering::Greater => break,
                }
            }
        }

        let actual = binary_search_keys_range(&keys, key_min, key_max);

        assert_eq!(expect.start, actual.start);
        assert_eq!(expect.count, actual.count);
    }

    #[test]
    fn binary_search_exhaustive() {
        for mode in [Mode::LowerBound, Mode::UpperBound] {
            for keys_count in 1_u32..300 {
                exhaustive_search(keys_count, mode);
            }
        }
    }

    #[test]
    fn binary_search_explicit() {
        use Mode::{LowerBound, UpperBound};
        for mode in [LowerBound, UpperBound] {
            explicit_search::<0, 1>(
                &[],
                &[0],
                [BinarySearchResult { index: 0, exact: false }],
                mode,
            );

            explicit_search::<0, 1>(
                &[4; 10],
                &[4],
                [BinarySearchResult { index: if mode == LowerBound { 0 } else { 9 }, exact: true }],
                mode,
            );

            explicit_search::<0, 3>(
                &[1],
                &[0, 1, 2],
                [
                    BinarySearchResult { index: 0, exact: false },
                    BinarySearchResult { index: 0, exact: true },
                    BinarySearchResult { index: 1, exact: false },
                ],
                mode,
            );

            explicit_search::<0, 5>(
                &[1, 3],
                &[0, 1, 2, 3, 4],
                [
                    BinarySearchResult { index: 0, exact: false },
                    BinarySearchResult { index: 0, exact: true },
                    BinarySearchResult { index: 1, exact: false },
                    BinarySearchResult { index: 1, exact: true },
                    BinarySearchResult { index: 2, exact: false },
                ],
                mode,
            );

            explicit_search::<0, 14>(
                &[1, 3, 5, 8, 9, 11],
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [
                    BinarySearchResult { index: 0, exact: false },
                    BinarySearchResult { index: 0, exact: true },
                    BinarySearchResult { index: 1, exact: false },
                    BinarySearchResult { index: 1, exact: true },
                    BinarySearchResult { index: 2, exact: false },
                    BinarySearchResult { index: 2, exact: true },
                    BinarySearchResult { index: 3, exact: false },
                    BinarySearchResult { index: 3, exact: false },
                    BinarySearchResult { index: 3, exact: true },
                    BinarySearchResult { index: 4, exact: true },
                    BinarySearchResult { index: 5, exact: false },
                    BinarySearchResult { index: 5, exact: true },
                    BinarySearchResult { index: 6, exact: false },
                    BinarySearchResult { index: 6, exact: false },
                ],
                mode,
            );
        }
    }

    #[test]
    fn binary_search_duplicates() {
        use Mode::{LowerBound, UpperBound};
        explicit_search::<0, 7>(
            &[0, 0, 3, 3, 3, 5, 5, 5, 5],
            &[0, 1, 2, 3, 4, 5, 6],
            [
                BinarySearchResult { index: 0, exact: true },
                BinarySearchResult { index: 2, exact: false },
                BinarySearchResult { index: 2, exact: false },
                BinarySearchResult { index: 2, exact: true },
                BinarySearchResult { index: 5, exact: false },
                BinarySearchResult { index: 5, exact: true },
                BinarySearchResult { index: 9, exact: false },
            ],
            LowerBound,
        );
        explicit_search::<0, 7>(
            &[0, 0, 3, 3, 3, 5, 5, 5, 5],
            &[0, 1, 2, 3, 4, 5, 6],
            [
                BinarySearchResult { index: 1, exact: true },
                BinarySearchResult { index: 2, exact: false },
                BinarySearchResult { index: 2, exact: false },
                BinarySearchResult { index: 4, exact: true },
                BinarySearchResult { index: 5, exact: false },
                BinarySearchResult { index: 8, exact: true },
                BinarySearchResult { index: 9, exact: false },
            ],
            UpperBound,
        );
    }

    #[test]
    fn binary_search_random() {
        let mut prng = Prng::from_seed(92);
        for mode in [Mode::LowerBound, Mode::UpperBound] {
            for i in 0..2048_usize {
                random_search(&mut prng, i, mode);
            }
        }
    }

    #[test]
    fn binary_search_explicit_range() {
        // Exact interval:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            3,
            1000,
            BinarySearchRange { start: 0, count: 9 },
        );

        // Larger interval:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            2,
            1001,
            BinarySearchRange { start: 0, count: 9 },
        );

        // Inclusive key_min and exclusive key_max:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            3,
            9,
            BinarySearchRange { start: 0, count: 2 },
        );

        // Exclusive key_min and inclusive key_max:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            5,
            10,
            BinarySearchRange { start: 2, count: 1 },
        );

        // Exclusive interval:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            5,
            14,
            BinarySearchRange { start: 2, count: 1 },
        );

        // Inclusive interval:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            15,
            100,
            BinarySearchRange { start: 3, count: 5 },
        );

        // Inclusive interval with duplicates:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 100, 100, 1000],
            15,
            100,
            BinarySearchRange { start: 3, count: 7 },
        );

        // Where key_min == key_max:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            10,
            10,
            BinarySearchRange { start: 2, count: 1 },
        );

        // Interval smaller than the first element:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            1,
            2,
            BinarySearchRange { start: 0, count: 0 },
        );

        // Interval greater than the last element:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            1_001,
            10_000,
            BinarySearchRange { start: 8, count: 0 },
        );

        // Nonexistent interval in the middle:
        explicit_range_search(
            &[3, 4, 10, 15, 20, 25, 30, 100, 1000],
            31,
            99,
            BinarySearchRange { start: 7, count: 0 },
        );

        // Empty slice:
        explicit_range_search(&[], 1, 2, BinarySearchRange { start: 0, count: 0 });
    }

    #[test]
    fn binary_search_duplicated_range() {
        explicit_range_search(
            &[1, 3, 3, 3, 5, 5, 5, 7],
            3,
            5,
            BinarySearchRange { start: 1, count: 6 },
        );
        explicit_range_search(&[1, 1, 1, 3, 5, 7], 1, 1, BinarySearchRange { start: 0, count: 3 });
    }

    #[test]
    fn binary_search_random_range() {
        let mut prng = Prng::from_seed(93);
        for i in 0..2048_usize {
            random_range_search(&mut prng, i);
        }
    }
}
