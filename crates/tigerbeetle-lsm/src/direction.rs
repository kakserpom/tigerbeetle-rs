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

    // TODO(port): src/direction.zig:79 `slice_lower_bound` — needs binary_search.zig
    // (`binary_search_values_upsert_index`), ported next.
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
}
