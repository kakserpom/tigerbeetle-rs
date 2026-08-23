//! LSM Tree is a sorted array with a monocle and a top hat.
//!
//! We want to iterate it in both directions:
//! - For CDC, you want to learn about all new objects with timestamp>threshold.
//! - For paginated timelines, you want to learn about past objects with timestamp<threshold.
//!
//! Sadly, this can't be implemented via just a single branch near the end of the system, we need
//! to change a whole bunch of `<` to `>` throughout the stack.
//!
//! [`Direction`] encapsulates the logic of "if ascending use < if descending use >". The mnemonic
//! is that usual comparison is horizontal along a number line, but Direction-aware is vertical.
//!
//! In other words, `key_min` and `key_max` track natural ordering, while `key_lower` and
//! `key_upper` are direction-aware.

use crate::binary_search::{Config, Mode, binary_search_values_upsert_index};

/// Upstream: `src/direction.zig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Ascending = 0,
    Descending = 1,
}

impl Direction {
    #[must_use]
    pub fn reverse(self) -> Self {
        match self {
            Self::Ascending => Self::Descending,
            Self::Descending => Self::Ascending,
        }
    }

    /// Upstream: `d.cmp(a, .@"<", b)`.
    #[must_use]
    pub fn cmp_lt<T: PartialOrd>(self, a: &T, b: &T) -> bool {
        match self {
            Self::Ascending => a < b,
            Self::Descending => a > b,
        }
    }

    /// Upstream: `d.cmp(a, .@"<=", b)`.
    #[must_use]
    pub fn cmp_le<T: PartialOrd>(self, a: &T, b: &T) -> bool {
        match self {
            Self::Ascending => a <= b,
            Self::Descending => a >= b,
        }
    }

    /// The direction-aware minimum of `a` and `b`.
    #[must_use]
    pub fn lower<T: Copy + PartialOrd>(self, a: T, b: T) -> T {
        if self.cmp_lt(&a, &b) { a } else { b }
    }

    /// The direction-aware maximum of `a` and `b`.
    #[must_use]
    pub fn upper<T: Copy + PartialOrd>(self, a: T, b: T) -> T {
        if self.cmp_lt(&a, &b) { b } else { a }
    }

    /// Peeks at the first element in iteration order.
    ///
    /// # Panics
    /// Panics if `slice` is empty (upstream asserts).
    #[must_use]
    pub fn slice_peek<T>(self, slice: &[T]) -> &T {
        assert!(!slice.is_empty());
        match self {
            Self::Ascending => &slice[0],
            Self::Descending => &slice[slice.len() - 1],
        }
    }

    /// Pops the first element in iteration order, returning it with the remaining slice.
    ///
    /// # Panics
    /// Panics if `slice` is empty (upstream asserts).
    #[must_use]
    pub fn slice_pop<T>(self, slice: &[T]) -> (&T, &[T]) {
        assert!(!slice.is_empty());
        match self {
            Self::Ascending => (&slice[0], &slice[1..]),
            Self::Descending => (&slice[slice.len() - 1], &slice[..slice.len() - 1]),
        }
    }

    /// Returns the subslice of `slice` whose keys are on the "near" side of `key` in iteration
    /// order: everything from the first key `>= key` (ascending) or through the last key
    /// `<= key` (descending). An empty result is represented by an empty slice.
    ///
    /// Upstream: `Direction.slice_lower_bound` (`src/direction.zig:79`). The comptime
    /// `key_from_value` fn pointer becomes a closure parameter.
    #[must_use]
    pub fn slice_lower_bound<'a, Key, Value, K>(
        self,
        key_from_value: &K,
        slice: &'a [Value],
        key: Key,
    ) -> &'a [Value]
    where
        Key: Ord + Copy + core::fmt::Debug,
        K: Fn(&Value) -> Key,
    {
        match self {
            Self::Ascending => {
                let start = binary_search_values_upsert_index(
                    key_from_value,
                    slice,
                    key,
                    Config { mode: Mode::LowerBound },
                );

                if start as usize == slice.len() { &[] } else { &slice[start as usize..] }
            }
            Self::Descending => {
                let end = {
                    let index = binary_search_values_upsert_index(
                        key_from_value,
                        slice,
                        key,
                        Config { mode: Mode::UpperBound },
                    );

                    let index_usize = index as usize;
                    index
                        + u32::from(
                            index_usize < slice.len() && key_from_value(&slice[index_usize]) <= key,
                        )
                };

                if end == 0 { &[] } else { &slice[..end as usize] }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Direction::*;

    #[test]
    fn reverse_swaps_directions() {
        assert_eq!(Ascending.reverse(), Descending);
        assert_eq!(Descending.reverse(), Ascending);
    }

    #[test]
    fn direction_aware_comparisons() {
        assert!(Ascending.cmp_lt(&1, &2));
        assert!(Descending.cmp_lt(&2, &1));
        assert!(Ascending.cmp_le(&2, &2));
        assert!(Descending.cmp_le(&2, &2));
        assert!(!Ascending.cmp_lt(&2, &1));

        assert_eq!(Ascending.lower(1, 2), 1);
        assert_eq!(Ascending.upper(1, 2), 2);
        assert_eq!(Descending.lower(1, 2), 2);
        assert_eq!(Descending.upper(1, 2), 1);
    }

    #[test]
    fn slice_peek_and_pop_follow_iteration_order() {
        let values = [10, 20, 30];

        assert_eq!(*Ascending.slice_peek(&values), 10);
        assert_eq!(*Descending.slice_peek(&values), 30);

        let (popped, rest) = Ascending.slice_pop(&values);
        assert_eq!(*popped, 10);
        assert_eq!(rest, &[20, 30]);

        let (popped, rest) = Descending.slice_pop(&values);
        assert_eq!(*popped, 30);
        assert_eq!(rest, &[10, 20]);
    }

    #[test]
    fn works_with_generic_value_types() {
        // Stored ascending; descending iteration peeks/pops from the tail.
        let values = [(1_u32, 'a'), (3, 'c')];
        let peeked = Descending.slice_peek(&values);
        assert_eq!(peeked.0, 3);
        let (popped, rest) = Ascending.slice_pop(&values);
        assert_eq!(popped.0, 1);
        assert_eq!(rest.len(), 1);
    }

    #[test]
    fn slice_lower_bound_ascending_and_descending() {
        let key = |value: &(u64, char)| value.0;
        // Sorted ascending by key, with a duplicated key.
        let values = [(1_u64, 'a'), (3, 'b'), (3, 'c'), (5, 'd')];

        // Ascending: everything at or after the first `key`.
        assert_eq!(
            Ascending.slice_lower_bound(&key, &values, 3),
            &[(3_u64, 'b'), (3, 'c'), (5, 'd')]
        );
        assert_eq!(Ascending.slice_lower_bound(&key, &values, 0), &values[..]);
        assert_eq!(Ascending.slice_lower_bound(&key, &values, 6), &[]);
        // Lower bound: the duplicate run starts at its first occurrence.
        assert_eq!(Ascending.slice_lower_bound(&key, &values, 4), &[(5_u64, 'd')]);

        // Descending: everything up to and including the last `key` in storage order.
        assert_eq!(
            Descending.slice_lower_bound(&key, &values, 3),
            &[(1_u64, 'a'), (3, 'b'), (3, 'c')]
        );
        assert_eq!(Descending.slice_lower_bound(&key, &values, 0), &[]);
        assert_eq!(Descending.slice_lower_bound(&key, &values, 9), &values[..]);
        // Upper bound + inclusive step: the duplicate run ends at its last occurrence.
        assert_eq!(Descending.slice_lower_bound(&key, &values, 2), &[(1_u64, 'a')]);

        // Empty input stays empty.
        let empty: [(u64, char); 0] = [];
        assert_eq!(Ascending.slice_lower_bound(&key, &empty, 1), &[]);
        assert_eq!(Descending.slice_lower_bound(&key, &empty, 1), &[]);
    }
}
