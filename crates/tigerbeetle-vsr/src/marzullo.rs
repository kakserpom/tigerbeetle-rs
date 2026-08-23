//! Marzullo's algorithm, invented by Keith Marzullo for his Ph.D. dissertation in 1984, is an
//! agreement algorithm used to select sources for estimating accurate time from a number of noisy
//! time sources. NTP uses a modified form of this called the Intersection algorithm, which returns
//! a larger interval for further statistical sampling. However, here we want the smallest interval.
//!
//! Port of `src/vsr/marzullo.zig`.

/// Port of `Marzullo`.
#[derive(Clone, Copy, Debug)]
pub struct Marzullo;

/// The smallest interval consistent with the largest number of sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    /// The lower bound on the minimum clock offset.
    pub lower_bound: i64,

    /// The upper bound on the maximum clock offset.
    pub upper_bound: i64,

    /// The number of "truechimers" consistent with the largest number of sources.
    pub sources_true: u8,

    /// The number of "falsetickers" falling outside this interval.
    /// Where `sources_false` plus `sources_true` always equals the total number of sources.
    pub sources_false: u8,
}

/// Either the lower or upper end of a bound, fed as input to the Marzullo algorithm to compute the
/// smallest interval across all tuples.
///
/// For example, given a clock offset to a remote replica of 3s, a round trip time of 1s, and
/// a maximum tolerance between clocks of 100ms on either side, we might create two tuples, the
/// lower bound having an offset of 2.4s and the upper bound having an offset of 3.6s,
/// to represent the error introduced by the round trip time and by the clocks themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuple {
    /// An identifier, the index of the clock source in the list of clock sources:
    pub source: u8,
    pub offset: i64,
    pub bound: Bound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Lower,
    Upper,
}

impl Marzullo {
    /// Returns the smallest interval consistent with the largest number of sources.
    ///
    /// # Panics
    /// Panics if the input violates internal invariants: an odd-length slice, or a sort that does
    /// not satisfy the comparator's total order (upstream asserts the same).
    #[must_use]
    pub fn smallest_interval(tuples: &mut [Tuple]) -> Interval {
        // There are two bounds (lower and upper) per source clock offset sample.
        assert!(tuples.len().is_multiple_of(2));
        assert!(tuples.len() <= usize::from(u8::MAX) * 2);
        #[allow(clippy::cast_possible_truncation)] // bounded by the assertion above
        let sources = (tuples.len() / 2) as u8;

        if sources == 0 {
            return Interval { lower_bound: 0, upper_bound: 0, sources_true: 0, sources_false: 0 };
        }

        // DEVIATION: upstream uses insertion sort "for safety"; `sort_unstable_by` is also
        // in-place and allocation-free. `less_than` fully specifies the order (offset, then
        // bound, then source), so any correct sort yields the identical sequence — determinism
        // is unaffected by the algorithm choice.
        tuples.sort_unstable_by(|a, b| less_than(*a, *b));

        // Here is a description of the algorithm:
        // <https://en.wikipedia.org/wiki/Marzullo%27s_algorithm#Method>
        let mut best: i64 = 0;
        let mut count: i64 = 0;
        let mut previous: Option<Tuple> = None;
        // Zero-initialized placeholder; both branches below assign it before it is read
        // (with at least one source, the first tuple raises count above best).
        let mut interval =
            Interval { lower_bound: 0, upper_bound: 0, sources_true: 0, sources_false: 0 };

        for (i, tuple) in tuples.iter().enumerate() {
            // Verify that our sort implementation is correct:
            if let Some(p) = previous {
                assert!(p.offset <= tuple.offset);
                if p.offset == tuple.offset {
                    if p.bound == tuple.bound {
                        assert!(p.source < tuple.source);
                    } else {
                        assert!(p.bound == Bound::Lower && tuple.bound == Bound::Upper);
                    }
                }
            }
            previous = Some(*tuple);

            // Update the current number of overlapping intervals:
            match tuple.bound {
                Bound::Lower => count += 1,
                Bound::Upper => count -= 1,
            }
            // The last upper bound tuple will have a count of one less than the lower bound.
            // Therefore, we should never see count >= best for the last tuple:
            if count > best {
                best = count;
                interval.lower_bound = tuple.offset;
                interval.upper_bound = tuples[i + 1].offset;
            } else if count == best && tuples[i + 1].bound == Bound::Upper {
                // This is a tie for best overlap. Both intervals have the same number of sources.
                // We want to choose the smaller of the two intervals:
                let alternative = tuples[i + 1].offset - tuple.offset;
                if alternative < interval.upper_bound - interval.lower_bound {
                    interval.lower_bound = tuple.offset;
                    interval.upper_bound = tuples[i + 1].offset;
                }
            }
        }
        assert_eq!(previous.map(|p| p.bound), Some(Bound::Upper));

        // The number of false sources (ones which do not overlap the optimal interval) is the
        // number of sources minus the value of `best`:
        assert!(best >= 0);
        assert!(best <= i64::from(sources));
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss // bounded by the assertion above
        )]
        let best_u8 = best as u8;
        interval.sources_true = best_u8;
        interval.sources_false = sources - interval.sources_true;
        assert_eq!(
            u16::from(interval.sources_true) + u16::from(interval.sources_false),
            u16::from(sources)
        );

        interval
    }
}

/// Sorts the list of tuples by clock offset. If two tuples with the same offset but opposite
/// bounds exist, indicating that one interval ends just as another begins, then a method of
/// deciding which comes first is necessary. Such an occurrence can be considered an overlap
/// with no duration, which can be found by the algorithm by sorting the lower bound before the
/// upper bound. Alternatively, if such pathological overlaps are considered objectionable then
/// they can be avoided by sorting the upper bound before the lower bound.
fn less_than(a: Tuple, b: Tuple) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    if a.offset < b.offset {
        return Ordering::Less;
    }
    if b.offset < a.offset {
        return Ordering::Greater;
    }
    if a.bound == Bound::Lower && b.bound == Bound::Upper {
        return Ordering::Less;
    }
    if b.bound == Bound::Lower && a.bound == Bound::Upper {
        return Ordering::Greater;
    }
    // Use the source index to break the tie and ensure the sort is fully specified and stable
    // so that different sort algorithms sort the same way:
    a.source.cmp(&b.source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_smallest_interval(bounds: &[i64], want: Interval) {
        let mut tuples = Vec::with_capacity(bounds.len());
        for (i, &offset) in bounds.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)] // test inputs have < 256 sources
            let source = (i / 2) as u8;
            let bound = if i % 2 == 0 { Bound::Lower } else { Bound::Upper };
            tuples.push(Tuple { source, offset, bound });
        }

        let got = Marzullo::smallest_interval(&mut tuples);
        assert_eq!(got, want);
    }

    /// Upstream: test "marzullo".
    #[test]
    fn marzullo() {
        test_smallest_interval(
            &[11, 13, 10, 12, 8, 12],
            Interval { lower_bound: 11, upper_bound: 12, sources_true: 3, sources_false: 0 },
        );

        test_smallest_interval(
            &[8, 12, 11, 13, 14, 15],
            Interval { lower_bound: 11, upper_bound: 12, sources_true: 2, sources_false: 1 },
        );

        test_smallest_interval(
            &[-10, 10, -1, 1, 0, 0],
            Interval { lower_bound: 0, upper_bound: 0, sources_true: 3, sources_false: 0 },
        );

        // The upper bound of the first interval overlaps inclusively with the lower of the last.
        test_smallest_interval(
            &[8, 12, 10, 11, 8, 10],
            Interval { lower_bound: 10, upper_bound: 10, sources_true: 3, sources_false: 0 },
        );

        // The first smallest interval is selected. The alternative with equal overlap is 10..12.
        // However, while this shares the same number of sources, it is not the smallest interval.
        test_smallest_interval(
            &[8, 12, 10, 12, 8, 9],
            Interval { lower_bound: 8, upper_bound: 9, sources_true: 2, sources_false: 1 },
        );

        // The last smallest interval is selected. The alternative with equal overlap is 7..9.
        // However, while this shares the same number of sources, it is not the smallest interval.
        test_smallest_interval(
            &[7, 9, 7, 12, 10, 11],
            Interval { lower_bound: 10, upper_bound: 11, sources_true: 2, sources_false: 1 },
        );

        // The same idea as the previous test, but with negative offsets.
        test_smallest_interval(
            &[-9, -7, -12, -7, -11, -10],
            Interval { lower_bound: -11, upper_bound: -10, sources_true: 2, sources_false: 1 },
        );

        // A cluster of one with no remote sources.
        test_smallest_interval(
            &[],
            Interval { lower_bound: 0, upper_bound: 0, sources_true: 0, sources_false: 0 },
        );

        // A cluster of two with one remote source.
        test_smallest_interval(
            &[1, 3],
            Interval { lower_bound: 1, upper_bound: 3, sources_true: 1, sources_false: 0 },
        );

        // A cluster of three with agreement.
        test_smallest_interval(
            &[1, 3, 2, 2],
            Interval { lower_bound: 2, upper_bound: 2, sources_true: 2, sources_false: 0 },
        );

        // A cluster of three with no agreement, still returns the smallest interval.
        test_smallest_interval(
            &[1, 3, 4, 5],
            Interval { lower_bound: 4, upper_bound: 5, sources_true: 1, sources_false: 1 },
        );
    }
}
