//! ZigZag merge join.
//! Resources:
//! <https://github.com/objectify/objectify/wiki/Concepts#indexes>.
//! <https://youtu.be/AgaL6NGpkB8?t=26m10s>
//!
//! Upstream: `src/lsm/zig_zag_merge.zig`.
//!
//! DEVIATION: Zig instantiates `ZigZagMergeIteratorType(…)` per context/key/value/stream
//! closures; this port is a generic struct over the [`ZigZagMergeSpec`] trait. The
//! `error{Pending}` error becomes the [`Pending`] marker in `Result`s.

use core::fmt::Debug;

use crate::direction::Direction;

/// The stream was consumed and must be refilled before calling `stream_peek` again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pending;

/// Width of the pending-stream bitset: `streams_max: u8` upstream bounds it at 256.
const PENDING_BITSET_WIDTH: usize = 256;

/// Static description of a zig-zag merge (upstream comptime parameters).
pub trait ZigZagMergeSpec: 'static {
    type Context;
    type Key: Copy + Ord + Debug;
    type Value: Copy + PartialEq + Debug;

    /// Upper bound on stream count (`streams_max`); also the pending-bitset width.
    const STREAMS_MAX: usize;
    /// `std.math.maxInt(Key)`.
    const KEY_MAX: Self::Key;
    /// Zero, used as the initial ascending candidate.
    const KEY_MIN: Self::Key;

    fn key_from_value(value: &Self::Value) -> Self::Key;

    /// Peek the next key in the stream identified by `stream_index`.
    ///
    /// Returns `Err(Pending)` if the stream was consumed and must be refilled before calling
    /// peek again; `Ok(None)` if the stream was fully consumed and reached the end.
    ///
    /// # Errors
    /// Returns [`Pending`] when the stream must be refilled before peeking again.
    fn stream_peek(
        context: &mut Self::Context,
        stream_index: u32,
    ) -> Result<Option<Self::Key>, Pending>;

    /// Consumes the current value and moves the stream identified by `stream_index`.
    /// Pop is always called after `peek`; the stream must be neither empty nor pending.
    fn stream_pop(context: &mut Self::Context, stream_index: u32) -> Self::Value;

    /// Probes the stream identified by `stream_index`, causing it to move to the next value
    /// such that `value.key >= probe_key` (ascending) or `value.key <= probe_key`
    /// (descending).
    ///
    /// Should not be called when the current key already matches the probe.
    /// The stream may become empty or pending _after_ probing.
    fn stream_probe(context: &mut Self::Context, stream_index: u32, probe_key: Self::Key);
}

pub struct ZigZagMergeIterator<'a, S: ZigZagMergeSpec> {
    context: &'a mut S::Context,
    streams_count: u32,
    direction: Direction,
    key_popped: Option<S::Key>,
    key_peeked: Option<S::Key>,
}

impl<'a, S: ZigZagMergeSpec> ZigZagMergeIterator<'a, S> {
    /// At least two scans are required for zig-zag merge.
    ///
    /// # Panics
    /// Panics unless `1 < streams_count <= S::STREAMS_MAX` (upstream asserts).
    #[must_use]
    pub fn new(context: &'a mut S::Context, streams_count: u32, direction: Direction) -> Self {
        assert!(streams_count as usize <= S::STREAMS_MAX);
        assert!(streams_count > 1);

        Self { context, streams_count, direction, key_popped: None, key_peeked: None }
    }

    /// Resets the iterator when the underlying streams are moved.
    /// It's not necessary for ZigZagMerge, but it follows the same API for all
    /// MergeIterators.
    pub fn reset(&mut self) {}

    /// Pops the value at the next common key across all streams.
    ///
    /// # Errors
    /// Returns [`Pending`] when the underlying streams need refilling; retry the call.
    ///
    /// # Panics
    /// Panics on duplicate keys or cross-stream value mismatches (upstream asserts).
    pub fn pop(&mut self) -> Result<Option<S::Value>, Pending> {
        let Some(key) = self.peek_key()? else {
            return Ok(None);
        };

        if let Some(previous) = self.key_popped {
            // Duplicate values are not expected.
            assert!(self.direction.cmp_lt(&previous, &key));
        }
        self.key_popped = Some(key);

        let value = S::stream_pop(self.context, 0);
        assert_eq!(S::key_from_value(&value), key);
        for stream_index in 1..self.streams_count {
            let value_other = S::stream_pop(self.context, stream_index);
            assert_eq!(S::key_from_value(&value_other), key);

            // Differently from K-way merge, there's no precedence between streams
            // in Zig-Zag merge. It's assumed that all streams will produce the same
            // value during a key intersection.
            assert_eq!(value, value_other);
        }

        Ok(Some(value))
    }

    /// Zig zig-zag join algorithm: finding the next _common_ key.
    /// Algorithm is conflict driven --- if any two streams disagree
    /// on the next key, one of the streams can be advanced (probed).
    /// In particular, if any stream is empty, there are no common keys.
    /// Conversely, the algorithm finishes when there is no disagreement:
    /// - some streams are pending (need IO to fetch next key from disk),
    /// - _all_ other streams agree on the key.
    ///
    /// The schedule to interrogate the streams is arbitrary. We use
    /// simple round-robin: going in circles, resetting the "tour" every
    /// time a conflict is detected, until we complete a full circle
    /// without a reset. The schedule ensures that any pending stream is
    /// probed with our best guess for optimal IO.
    fn peek_key(&mut self) -> Result<Option<S::Key>, Pending> {
        assert!(self.streams_count > 1);
        assert!(self.streams_count as usize <= S::STREAMS_MAX);

        // NB: We could start with `self.key_peeked`, but starting
        // from zero tightens assertions on the underlying streams.
        let mut candidate = match self.direction {
            Direction::Ascending => S::KEY_MIN,
            Direction::Descending => S::KEY_MAX,
        };

        // Upstream uses a stack bitset of `streams_max` bits; `streams_max: u8` there
        // bounds the stream count at 256, so a fixed-width stack bitset suffices here too
        // (an associated const cannot size an array).
        assert!(u8::try_from(self.streams_count).is_ok());
        let mut pending = [false; PENDING_BITSET_WIDTH];
        let mut pending_count: u32 = 0;

        let mut tour_total: u32 = 0;
        let mut tour_equal: u32 = 0;
        let mut tour_pending: u32 = 0;

        // TODO(upstream): Find a way to add a safety counter here.
        let mut tour_index: u32 = 0;
        while tour_total < self.streams_count {
            assert_eq!(tour_total, tour_equal + tour_pending);

            let index = tour_index as usize;

            // Optimization: don't re-probe already pending streams
            // until the very end, when the final candidate is known.
            if pending[index] {
                tour_total += 1;
                tour_pending += 1;
            } else {
                match self.gallop_key(tour_index, candidate) {
                    // The stream needs I/O to refill: record it as pending and keep touring
                    // the remaining streams so the candidate is pinned before we give up.
                    // (Upstream `catch |err| switch { error.Pending => { ...; continue; } }`.)
                    Err(Pending) => {
                        assert!(!pending[index]);
                        pending[index] = true;
                        pending_count += 1;
                        tour_total += 1;
                        tour_pending += 1;
                    }
                    // An empty stream short-circuits the entire thing.
                    Ok(None) => return Ok(None),
                    Ok(Some(key)) => {
                        if self.direction.cmp_lt(&candidate, &key) {
                            // The stream is strictly ahead, restart a tour with a new
                            // candidate.
                            candidate = key;
                            tour_total = 1;
                            tour_equal = 1;
                            tour_pending = 0;
                        } else {
                            assert_eq!(candidate, key);
                            tour_total += 1;
                            tour_equal += 1;
                        }
                    }
                }
            }

            // The upstream loop advances its round-robin index on every iteration
            // (the loop's continue expression), including the `continue` paths above.
            tour_index = (tour_index + 1) % self.streams_count;
        }
        assert_eq!(tour_total, tour_equal + tour_pending);
        assert_eq!(tour_total, self.streams_count);
        assert_eq!(tour_pending, pending_count);

        if tour_pending == self.streams_count {
            return Err(Pending);
        }

        // Completing the optimization, probe pending streams one last time.
        // We minimize probe & peek virtual function calls, keeping IO optimal.
        for stream_index in 0..self.streams_count {
            let index = stream_index as usize;
            if pending[index] {
                pending[index] = false;
                pending_count -= 1;
                S::stream_probe(self.context, stream_index, candidate);
                assert!(matches!(S::stream_peek(self.context, stream_index), Err(Pending)));
            } else {
                let Ok(Some(key)) = S::stream_peek(self.context, stream_index) else {
                    unreachable!("settled stream must hold the candidate key")
                };
                assert_eq!(key, candidate);
            }
        }
        assert_eq!(pending_count, 0);

        if tour_pending > 0 {
            return Err(Pending);
        }

        if let Some(key_peeked) = self.key_peeked {
            assert!(self.direction.cmp_le(&key_peeked, &candidate));
        }
        self.key_peeked = Some(candidate);
        Ok(Some(candidate))
    }

    fn gallop_key(
        &mut self,
        stream_index: u32,
        candidate: S::Key,
    ) -> Result<Option<S::Key>, Pending> {
        assert!(stream_index < self.streams_count);

        let Some(mut key) = S::stream_peek(self.context, stream_index)? else {
            return Ok(None);
        };
        if let Some(key_peeked) = self.key_peeked {
            assert!(self.direction.cmp_le(&key_peeked, &key));
        }

        if self.direction.cmp_lt(&key, &candidate) {
            S::stream_probe(self.context, stream_index, candidate);
            let Some(next) = S::stream_peek(self.context, stream_index)? else {
                return Ok(None);
            };
            key = next;

            assert!(self.direction.cmp_le(&candidate, &key));
            if let Some(key_peeked) = self.key_peeked {
                assert!(self.direction.cmp_le(&key_peeked, &key));
            }
        }

        Ok(Some(key))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)] // test streams never pend

    use tigerbeetle_core::stdx::prng::Prng;

    use super::{Direction, Pending, ZigZagMergeIterator, ZigZagMergeSpec};

    /// Port of upstream `TestContextType(streams_max)`; consumes each stream from either
    /// end depending on the scan direction.
    struct TestStream {
        values: Vec<u128>,
        start: usize,
        len: usize,
    }

    impl TestStream {
        fn new(values: &[u128]) -> Self {
            Self { values: values.to_vec(), start: 0, len: values.len() }
        }
    }

    struct TestContext<const STREAMS_MAX: usize> {
        streams: Vec<TestStream>,
        direction: Direction,
    }

    impl<const STREAMS_MAX: usize> ZigZagMergeSpec for TestContext<STREAMS_MAX> {
        type Context = Self;
        // Using `u128` simplifies the fuzzer, avoiding undesirable matches
        // and duplicate elements when generating random values.
        type Key = u128;
        type Value = u128;

        const STREAMS_MAX: usize = STREAMS_MAX;
        const KEY_MAX: u128 = u128::MAX;
        const KEY_MIN: u128 = 0;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            *value
        }

        fn stream_peek(
            context: &mut Self,
            stream_index: u32,
        ) -> Result<Option<Self::Key>, Pending> {
            let stream = &context.streams[stream_index as usize];
            if stream.len == 0 {
                return Ok(None);
            }
            let key = match context.direction {
                Direction::Ascending => stream.values[stream.start],
                Direction::Descending => stream.values[stream.start + stream.len - 1],
            };
            Ok(Some(key))
        }

        fn stream_pop(context: &mut Self, stream_index: u32) -> Self::Value {
            let stream = &mut context.streams[stream_index as usize];
            match context.direction {
                Direction::Ascending => {
                    let value = stream.values[stream.start];
                    stream.start += 1;
                    stream.len -= 1;
                    value
                }
                Direction::Descending => {
                    stream.len -= 1;
                    stream.values[stream.start + stream.len]
                }
            }
        }

        fn stream_probe(context: &mut Self, stream_index: u32, probe_key: Self::Key) {
            loop {
                let peeked = match Self::stream_peek(context, stream_index) {
                    Ok(peeked) => peeked,
                    Err(Pending) => unreachable!("test streams never pend"),
                };
                let Some(key) = peeked else {
                    break;
                };

                if match context.direction {
                    Direction::Ascending => key >= probe_key,
                    Direction::Descending => key <= probe_key,
                } {
                    break;
                }

                let value = Self::stream_pop(context, stream_index);
                assert_eq!(key, Self::key_from_value(&value));
            }
        }
    }

    impl<const STREAMS_MAX: usize> TestContext<STREAMS_MAX> {
        fn merge(streams: &[&[u128]], expect: &[u128]) {
            for direction in [Direction::Ascending, Direction::Descending] {
                let mut context = Self {
                    streams: streams.iter().map(|stream| TestStream::new(stream)).collect(),
                    direction,
                };

                let mut actual = Vec::new();
                let streams_count = u32::try_from(streams.len()).expect("few streams");
                let mut it =
                    ZigZagMergeIterator::<Self>::new(&mut context, streams_count, direction);
                while let Some(value) = it.pop().expect("test streams never pend") {
                    actual.push(value);
                }

                if direction == Direction::Descending {
                    actual.reverse();
                }
                assert_eq!(actual, expect);
            }
        }

        fn fuzz(prng: &mut Prng, stream_key_count_max: usize) {
            const INTERSECTION_LEN_MIN: usize = 5;
            for streams_count in 2..=STREAMS_MAX {
                let mut streams: Vec<Vec<u128>> = Vec::new();
                let mut stream_len_min = stream_key_count_max;
                for _ in 0..streams_count {
                    let len =
                        prng.range_inclusive_usize(INTERSECTION_LEN_MIN, stream_key_count_max);
                    stream_len_min = stream_len_min.min(len);
                    streams.push(vec![0; len]);
                }

                let intersection_len =
                    prng.range_inclusive_usize(INTERSECTION_LEN_MIN, stream_len_min);
                assert!(INTERSECTION_LEN_MIN <= intersection_len);
                assert!(intersection_len <= stream_len_min);
                let mut intersection = vec![0u128; intersection_len];

                Self::fuzz_make_intersection(prng, &mut streams, &mut intersection);

                let mut refs: Vec<&[u128]> = streams.iter().map(Vec::as_slice).collect();

                // Positive space.
                Self::merge(&refs, &intersection);

                // Negative space: disjoint stream.
                let dummy: [u128; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
                let last = refs.len() - 1;
                let saved = refs[last];
                refs[last] = &dummy;
                Self::merge(&refs, &[]);

                // Negative space: empty stream.
                let empty: [u128; 0] = [];
                refs[last] = &empty;
                Self::merge(&refs, &[]);
                refs[last] = saved;
            }
        }

        /// DEVIATION: upstream seeds the intersection/stream fillers through
        /// `prng.fill(mem.sliceAsBytes(…))`; per-element `int_u128` draws are used here
        /// instead (same distribution shape, different byte layout).
        fn fuzz_make_intersection(
            prng: &mut Prng,
            streams: &mut [Vec<u128>],
            intersection: &mut [u128],
        ) {
            // Starting with the values we want to be the intersection:
            for value in intersection.iter_mut() {
                *value = prng.int_u128();
            }
            intersection.sort_unstable();

            // Then injecting the intersection into each stream and filling the rest with
            // random values:
            for stream in streams {
                assert!(intersection.len() <= stream.len());
                stream[..intersection.len()].copy_from_slice(intersection);
                for value in &mut stream[intersection.len()..] {
                    *value = prng.int_u128();
                }
                stream.sort_unstable();
            }
        }
    }

    #[test]
    fn zig_zag_merge_unit() {
        type Context = TestContext<10>;

        // Equal streams:
        Context::merge(&[&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]], &[1, 2, 3, 4, 5]);

        // Disjoint streams:
        Context::merge(&[&[1, 3, 5, 7, 9], &[2, 4, 6, 8, 10]], &[]);

        // Equal and disjoint streams:
        Context::merge(
            &[&[1, 3, 5, 7, 9], &[1, 3, 5, 7, 9], &[2, 4, 6, 8, 10], &[2, 4, 6, 8, 10]],
            &[],
        );

        // Intersection with an empty stream:
        Context::merge(&[&[2, 4, 6, 8, 10], &[2, 4, 6, 8, 10], &[]], &[]);

        // Partial intersection:
        Context::merge(
            &[&[1, 2, 3, 4, 5], &[2, 3, 4, 5, 6], &[3, 4, 5, 6, 7], &[4, 5, 6, 7, 8]],
            &[4, 5],
        );

        // Intersection with streams of different sizes:
        // {1, 2, 3, ..., 1000}.
        let thousands: Vec<u128> = (1..=1000).collect();
        // {10, 20, 30, ..., 1000}.
        let hundreds: Vec<u128> = (1..=100).map(|i| 10 * i).collect();
        // {1, 10, 100, 1000, ..., 10 ^ 10}.
        let powers: Vec<u128> = (0..10).map(|i| 10u128.pow(i)).collect();
        Context::merge(&[&thousands, &hundreds, &powers], &[10, 100, 1000]);

        // Sparse matching values: {1, ..., 100} ∩ {100, ..., 199} = {100}.
        let low: Vec<u128> = (1..=100).collect();
        let high: Vec<u128> = (0..100).map(|i| i + 100).collect();
        Context::merge(&[&low, &high], &[100]);

        // Sparse matching values: {100, ..., 199} ∩ {1, ..., 100} = {100}.
        Context::merge(&[&high, &low], &[100]);
    }

    #[test]
    fn zig_zag_merge_fuzz() {
        let mut prng = Prng::from_seed(42);
        TestContext::<32>::fuzz(&mut prng, 256);
    }

    // -- Pending (async refill) tests --
    //
    // Upstream's production contexts (e.g. the compaction `levels_b`) return `error.Pending`
    // from `stream_peek` when the underlying stream needs I/O to refill. The iterator must
    // then keep touring the *settled* streams to pin the candidate, probe the pending streams
    // toward it, and only then surface `Pending` so the caller can issue the I/O.

    /// Context whose streams pend against a caller-controlled mask; probing is recorded so
    /// tests can observe the candidate used to advance pending streams.
    struct PendShared {
        streams: Vec<Vec<u128>>,
        cursor: Vec<usize>,
        pend: Vec<bool>,
        probe_log: Vec<(u32, u128)>,
    }

    /// DEVIATION (test-only): state lives behind `Rc<RefCell>` so the test can flip the
    /// pending mask between `pop` calls while the iterator borrows the context.
    struct TestPendContext<const STREAMS_MAX: usize> {
        shared: std::rc::Rc<std::cell::RefCell<PendShared>>,
    }

    impl<const STREAMS_MAX: usize> TestPendContext<STREAMS_MAX> {
        fn new(streams: Vec<Vec<u128>>) -> Self {
            let count = streams.len();
            Self {
                shared: std::rc::Rc::new(std::cell::RefCell::new(PendShared {
                    streams,
                    cursor: vec![0; count],
                    pend: vec![false; count],
                    probe_log: Vec::new(),
                })),
            }
        }
    }

    impl<const STREAMS_MAX: usize> ZigZagMergeSpec for TestPendContext<STREAMS_MAX> {
        type Context = Self;
        type Key = u128;
        type Value = u128;

        const STREAMS_MAX: usize = STREAMS_MAX;
        const KEY_MAX: u128 = u128::MAX;
        const KEY_MIN: u128 = 0;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            *value
        }

        fn stream_peek(
            context: &mut Self,
            stream_index: u32,
        ) -> Result<Option<Self::Key>, Pending> {
            let index = stream_index as usize;
            let shared = context.shared.borrow();
            if shared.pend[index] {
                return Err(Pending);
            }
            let stream = &shared.streams[index];
            if shared.cursor[index] >= stream.len() {
                return Ok(None);
            }
            Ok(Some(stream[shared.cursor[index]]))
        }

        fn stream_pop(context: &mut Self, stream_index: u32) -> Self::Value {
            let index = stream_index as usize;
            let mut shared = context.shared.borrow_mut();
            let value = shared.streams[index][shared.cursor[index]];
            shared.cursor[index] += 1;
            value
        }

        fn stream_probe(context: &mut Self, stream_index: u32, probe_key: Self::Key) {
            let index = stream_index as usize;
            let mut shared = context.shared.borrow_mut();
            shared.probe_log.push((stream_index, probe_key));
            let advance = shared.streams[index]
                .iter()
                .skip(shared.cursor[index])
                .take_while(|&&key| key < probe_key)
                .count();
            shared.cursor[index] += advance;
        }
    }

    #[test]
    fn zig_zag_merge_pending_surfaces_after_pinning_candidate() {
        type Context = TestPendContext<32>;

        let mut context = Context::new(vec![vec![2, 4, 6], vec![4, 5, 6, 7, 8], vec![4, 6, 8]]);
        let shared = std::rc::Rc::clone(&context.shared);
        let mut it = ZigZagMergeIterator::<Context>::new(&mut context, 3, Direction::Ascending);

        // Stream 0 needs I/O to refill.
        shared.borrow_mut().pend[0] = true;
        assert_eq!(it.pop(), Err(Pending));
        // The settled streams (1, 2) share key 4: the pending stream must be probed toward it,
        // and stream 0's cursor must have advanced past 2.
        assert_eq!(shared.borrow().probe_log, vec![(0, 4)]);
        assert_eq!(shared.borrow().cursor[0], 1);

        // I/O completes; the merge resumes and yields the full intersection {4, 6}.
        shared.borrow_mut().pend[0] = false;
        let mut actual = Vec::new();
        while let Ok(Some(value)) = it.pop() {
            actual.push(value);
        }
        assert_eq!(actual, vec![4, 6]);
    }

    #[test]
    fn zig_zag_merge_pending_all_streams_short_circuits() {
        type Context = TestPendContext<32>;

        let mut context = Context::new(vec![vec![1, 2], vec![1, 2], vec![1, 2]]);
        let shared = std::rc::Rc::clone(&context.shared);
        shared.borrow_mut().pend.fill(true);
        let mut it = ZigZagMergeIterator::<Context>::new(&mut context, 3, Direction::Ascending);

        // With every stream pending there is no candidate to pin, so we surface `Pending`
        // immediately without probing.
        assert_eq!(it.pop(), Err(Pending));
        assert!(shared.borrow().probe_log.is_empty());
    }
}
