//! K-way merge via a loser tree algorithm (Knuth Volume 3 p. 253).
//! Port of `src/lsm/k_way_merge.zig`.
//!
//! Merges k sorted streams using a tournament (loser) tree.
//! The current global winner lives in `win_key`/`win_id`. Internal nodes store
//! the losers of the last comparisons along the root-to-leaf paths in a
//! struct-of-arrays layout (`loser_keys`/`loser_ids`).
//!
//! ```text
//!     0 (winner)
//!
//!     1
//!    / \
//!   2   3
//!  / \ / \
//! 4  5 6  7
//! -------------
//! K input streams
//! ```
//!
//! The internal nodes are organized in a flat Eytzinger layout.
//! That is the tree above is stored as [1][2][3][4][5][6][7].
//! Empty streams are represented with a sentinel node that always loses against real nodes.

// DEVIATION: upstream Zig relies on arbitrary-width integer arithmetic with implicit safe
// truncation; every cast allowed here is bounded by construction (indices below
// NODE_COUNT_MAX, lengths below u16/u32 maxima at these call sites).
#![allow(clippy::cast_possible_truncation)]

use crate::direction::Direction;

/// Upstream: `error{Pending}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pending;

impl core::fmt::Display for Pending {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("pending")
    }
}

impl std::error::Error for Pending {}

/// The key type usable by [`TournamentTree`] and [`KWayMergeIterator`].
///
/// DEVIATION: upstream accepts any unsigned integer type (`sentinel_key = maxInt(Key)`);
/// here the sentinel values are provided through this trait.
pub trait TournamentKey: Copy + Ord + core::fmt::Debug {
    /// Upstream: `std.math.maxInt(Key)`.
    const SENTINEL_KEY: Self;
    /// Upstream: `std.math.minInt(Key)` (used by `table_memory.key_max`).
    const MIN_KEY: Self;
}

macro_rules! impl_tournament_key {
    ($t:ty) => {
        impl TournamentKey for $t {
            const SENTINEL_KEY: Self = <$t>::MAX;
            const MIN_KEY: Self = <$t>::MIN;
        }
    };
}

impl_tournament_key!(u8);
impl_tournament_key!(u16);
impl_tournament_key!(u32);
impl_tournament_key!(u64);
impl_tournament_key!(u128);

/// DEVIATION: upstream calls `stdx.branchless_select`; written here as an ordinary conditional,
/// which LLVM lowers to a `cmov`, matching upstream's intent.
#[inline]
#[must_use]
fn select<T>(a_wins: bool, a: T, b: T) -> T {
    if a_wins { a } else { b }
}

/// A single contestant stream head.
///
/// DEVIATION: upstream derives `Node.sentinel` from the tree's comptime key type; here the
/// sentinel lives on the generic impl via [`TournamentKey::SENTINEL_KEY`].
#[derive(Clone, Copy, Debug)]
pub struct Node<Key> {
    pub key: Key,
    pub id: u32,
}

impl<Key: TournamentKey> Node<Key> {
    pub const ID_SENTINEL: u32 = u32::MAX;
    pub const SENTINEL: Self = Self { key: Key::SENTINEL_KEY, id: Self::ID_SENTINEL };
}

/// Port of `TournamentTreeType(Key, contestants_max)`.
///
/// DEVIATION: upstream derives `node_count_max = ceilPowerOfTwoAssert(contestants_max)` at
/// compile time and sizes its arrays with it. Rust rejects const-generic arithmetic
/// (`CONTESTANTS_MAX.next_power_of_two()`) as an array length, so the tree is parameterized by
/// the power-of-two node count directly. Any stream count up to [`Self::NODE_COUNT_MAX`] is
/// still accepted (upstream's bound `contestant_count <= contestants_max` is subsumed, since
/// `contestants_max <= node_count_max` always holds); use [`ceil_power_of_two`] at call sites.
pub struct TournamentTree<Key: TournamentKey, const NODE_COUNT_MAX: usize> {
    loser_keys: [Key; NODE_COUNT_MAX],
    loser_ids: [u32; NODE_COUNT_MAX],
    win_key: Key,
    win_id: u32,
    contestants_left: u16,
    height: u8,
    direction: Direction,
}

/// Upstream equivalent: `std.math.ceilPowerOfTwo(usize, n)`; mirrors the tree parameter.
#[must_use]
pub const fn ceil_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

impl<Key: TournamentKey, const NODE_COUNT_MAX: usize> TournamentTree<Key, NODE_COUNT_MAX> {
    /// Asserts the contestant invariants and builds the loser tree.
    ///
    /// # Panics
    /// Panics if `contestant_count > NODE_COUNT_MAX`, if non-sentinel ids are not strictly
    /// ascending, or if any slot past `contestant_count` is not the sentinel (upstream asserts).
    pub fn init(
        direction: Direction,
        contestants: &mut [Node<Key>; NODE_COUNT_MAX],
        contestant_count: u16,
    ) -> Self {
        assert!((contestant_count as usize) <= NODE_COUNT_MAX);

        let mut contestant_previous: Option<Node<Key>> = None;
        let mut contestants_left: u16 = 0;
        for contestant in contestants.iter().take(usize::from(contestant_count)) {
            if contestant.id == Node::<Key>::ID_SENTINEL {
                // Stream is empty to begin with.
            } else {
                contestants_left += 1;
                if let Some(previous) = contestant_previous {
                    assert!(previous.id < contestant.id);
                }
                contestant_previous = Some(*contestant);
            }
        }
        for contestant in contestants.iter().skip(usize::from(contestant_count)) {
            assert_eq!(contestant.id, Node::<Key>::ID_SENTINEL);
        }

        let mut tree = Self {
            win_key: Key::SENTINEL_KEY,
            win_id: Node::<Key>::ID_SENTINEL,
            loser_keys: [Key::SENTINEL_KEY; NODE_COUNT_MAX],
            loser_ids: [Node::<Key>::ID_SENTINEL; NODE_COUNT_MAX],
            direction,
            contestants_left,
            height: 0,
        };

        if contestants_left == 0 {
            return tree;
        }

        // Compute effective tree size: only as large as needed for contestant_count.
        // (Upstream asserts power-of-two-ness via ceilPowerOfTwoAssert.)
        let node_count: usize = usize::from(contestant_count).next_power_of_two();
        tree.height = node_count.trailing_zeros() as u8;

        for level in 0..usize::from(tree.height) {
            let level_min: usize = (node_count >> (level + 1)) - 1;
            let level_max: usize = (node_count >> level) - 1;

            // Upstream zips `level_min..level_max` (loser indices) with `0..` (competitor
            // indices), pairing leaves two at a time and promoting winners one slot up.
            for (competitor_index, loser_index) in (level_min..level_max).enumerate() {
                let a = contestants[competitor_index * 2];
                let b = contestants[competitor_index * 2 + 1];
                let a_wins = beats(a.key, a.id, b.key, b.id, direction);

                contestants[competitor_index] =
                    Node { key: select(a_wins, a.key, b.key), id: select(a_wins, a.id, b.id) };
                // We select the loser here thus a, b are swapped.
                tree.loser_keys[loser_index] = select(a_wins, b.key, a.key);
                tree.loser_ids[loser_index] = select(a_wins, b.id, a.id);
            }
        }

        tree.win_key = contestants[0].key;
        tree.win_id = contestants[0].id;

        tree
    }

    /// Replaces the current winner with `entrant` (`None` retires its stream) and replays the
    /// root path.
    ///
    /// # Panics
    /// Panics if the winner id is out of range for the effective tree height, or if the tree
    /// empties while `contestants_left` is nonzero (upstream asserts).
    ///
    /// DEVIATION: upstream dispatches on height at compile time (`inline 0...height_max`) and on
    /// direction via `inline else`; here both are plain runtime values.
    pub fn pop_winner(&mut self, entrant: Option<Key>) {
        let direction = self.direction;

        let node_count: usize = 1_usize << self.height;
        let winner_id = self.win_id;

        assert!(winner_id < node_count as u32);
        if entrant.is_none() {
            self.contestants_left -= 1;
        }

        let mut new_key: Key = entrant.unwrap_or(Key::SENTINEL_KEY);
        let mut new_id: u32 = if entrant.is_some() { winner_id } else { Node::<Key>::ID_SENTINEL };

        let mut idx: usize = (node_count - 1) + winner_id as usize;
        for _ in 0..self.height {
            idx = (idx - 1) >> 1;

            let opp_key = self.loser_keys[idx];
            let opp_id = self.loser_ids[idx];
            let new_wins = beats(new_key, new_id, opp_key, opp_id, direction);

            self.loser_keys[idx] = select(new_wins, opp_key, new_key);
            self.loser_ids[idx] = select(new_wins, opp_id, new_id);
            new_key = select(new_wins, new_key, opp_key);
            new_id = select(new_wins, new_id, opp_id);
        }

        self.win_key = new_key;
        self.win_id = new_id;

        if self.win_id == Node::<Key>::ID_SENTINEL {
            assert_eq!(self.contestants_left, 0);
        }
    }

    #[must_use]
    pub const fn contestants_left(&self) -> u16 {
        self.contestants_left
    }

    #[must_use]
    pub const fn winner(&self) -> Node<Key> {
        Node { key: self.win_key, id: self.win_id }
    }
}

/// Returns true if (a_key, a_id) wins over (b_key, b_id).
/// Sentinels (`ID_SENTINEL`) always lose. Equal keys broken by id for stability.
/// In ascending mode, `SENTINEL_KEY` (maxInt) naturally loses on `<` so no
/// explicit sentinel checks are needed. In descending mode, maxInt would
/// incorrectly "win" on `>`, so explicit sentinel checks are required.
#[must_use]
fn beats<Key: TournamentKey>(
    a_key: Key,
    a_id: u32,
    b_key: Key,
    b_id: u32,
    direction: Direction,
) -> bool {
    let id_lt = u8::from(a_id < b_id);
    let keys_eq = u8::from(a_key == b_key);
    let eq_and_id_wins = keys_eq & id_lt;

    match direction {
        Direction::Ascending => {
            let key_lt = u8::from(a_key < b_key);
            (key_lt | eq_and_id_wins) == 1
        }
        Direction::Descending => {
            let key_gt = u8::from(a_key > b_key);
            let b_is_sentinel = u8::from(b_id == Node::<Key>::ID_SENTINEL);
            let a_is_sentinel = u8::from(a_id == Node::<Key>::ID_SENTINEL);
            let key_wins = key_gt | eq_and_id_wins;
            (b_is_sentinel | ((1 - a_is_sentinel) & key_wins)) == 1
        }
    }
}

/// The per-stream callbacks of `KWayMergeIteratorType`.
///
/// DEVIATION: upstream passes `stream_peek`/`stream_pop`/`key_from_value` as comptime function
/// pointers into `KWayMergeIteratorType(Context, Key, Value, options, ...)`; here they form one
/// trait so the iterator stays generic over any context type.
pub trait MergeStream {
    type Key: TournamentKey;
    type Value;

    /// Upstream: `fn stream_peek(context: *Context, stream_index: u32) Pending!?Key`.
    ///
    /// # Errors
    /// Returns [`Pending`] when the underlying stream is not ready (upstream: `error.Pending`).
    ///
    /// # Panics
    /// Panics if `stream_index` has no stream (upstream asserts implicitly).
    fn stream_peek(&mut self, stream_index: u32) -> Result<Option<Self::Key>, Pending>;

    /// Upstream: `fn stream_pop(context: *Context, stream_index: u32) Value`.
    fn stream_pop(&mut self, stream_index: u32) -> Self::Value;

    /// Upstream: comptime `key_from_value: fn (*const Value) callconv(.@"inline") Key`.
    #[must_use]
    fn value_key(value: &Self::Value) -> Self::Key;
}

/// Port of `KWayMergeIteratorType`; `DEDUPLICATE` mirrors `options.deduplicate`.
///
/// DEVIATION: upstream computes the tournament capacity from `options.streams_max` at compile
/// time; here [`NODE_COUNT_MAX`] is an explicit power-of-two parameter (see
/// [`TournamentTree`]), asserted against `STREAMS_MAX`.
pub struct KWayMergeIterator<
    S: MergeStream,
    const STREAMS_MAX: usize,
    const NODE_COUNT_MAX: usize,
    const DEDUPLICATE: bool,
> {
    context: S,
    streams_count: u16,
    direction: Direction,
    key_popped: Option<S::Key>,
    tree: Option<TournamentTree<S::Key, NODE_COUNT_MAX>>,
}

impl<S: MergeStream, const STREAMS_MAX: usize, const NODE_COUNT_MAX: usize, const DEDUPLICATE: bool>
    KWayMergeIterator<S, STREAMS_MAX, NODE_COUNT_MAX, DEDUPLICATE>
{
    /// Upstream asserts `streams_max >= 1` and `streams_max <= 1024` at compile time.
    ///
    /// # Panics
    /// Panics if `STREAMS_MAX` is outside `1..=1024`, if `NODE_COUNT_MAX` is not a
    /// power-of-two at least `STREAMS_MAX`, or if `streams_count > STREAMS_MAX`.
    pub fn init(context: S, streams_count: u16, direction: Direction) -> Self {
        const {
            assert!(STREAMS_MAX >= 1 && STREAMS_MAX <= 1024);
        };
        assert!(NODE_COUNT_MAX.is_power_of_two() && NODE_COUNT_MAX >= STREAMS_MAX);
        assert!(usize::from(streams_count) <= STREAMS_MAX);

        Self { context, key_popped: None, direction, streams_count, tree: None }
    }

    /// Clears the tree while preserving context, direction, stream count, and `key_popped`.
    pub fn reset(&mut self) {
        self.tree = None;
    }

    /// Loads all stream heads into a fresh [`TournamentTree`].
    fn load(&mut self) -> Result<(), Pending> {
        assert!(self.tree.is_none());

        let mut contestants = [Node::<S::Key>::SENTINEL; NODE_COUNT_MAX];
        for (id_usize, slot) in
            contestants.iter_mut().take(usize::from(self.streams_count)).enumerate()
        {
            let id = id_usize as u32;
            if let Some(key) = self.context.stream_peek(id)? {
                *slot = Node { key, id };
            }
        }

        self.tree =
            Some(TournamentTree::init(self.direction, &mut contestants, self.streams_count));
        Ok(())
    }

    /// Returns the next merged value, or `None` once every stream is exhausted.
    ///
    /// # Errors
    /// Returns [`Pending`] when a stream peek is not ready; the caller may simply retry.
    pub fn pop(&mut self) -> Result<Option<S::Value>, Pending> {
        if self.tree.is_none() {
            self.load()?;
        }
        // load() always installs a tree; the fallback keeps this panic-free.
        let Some(tree) = self.tree.as_mut() else { return Ok(None) };

        while tree.contestants_left > 0 {
            let key = self.context.stream_peek(tree.win_id)?;
            tree.pop_winner(key);
            if tree.contestants_left == 0 {
                return Ok(None);
            }
            let value = self.context.stream_pop(tree.win_id);
            if DEDUPLICATE {
                let key_next = S::value_key(&value);
                if self.key_popped.is_some_and(|key_prev| key_prev == key_next) {
                    continue;
                }
                self.key_popped = Some(key_next);
            }
            return Ok(Some(value));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::{KWayMergeIterator, MergeStream, Node, Pending, TournamentKey, TournamentTree};
    use crate::direction::Direction;
    use tigerbeetle_core::stdx::prng::{Prng, ratio};
    use tigerbeetle_core::testing::exhaustigen::Gen;

    use std::error::Error;

    fn value(key: u32, version: u32) -> Value {
        Value { key, version }
    }

    /// Upstream: `TestContextType(streams_max)` — a fixed set of sorted value streams.
    ///
    /// DEVIATION: streams are `Vec`s instead of slices of a shared arena.
    struct TestContext<const STREAMS_MAX: usize> {
        streams: Vec<Vec<Value>>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Value {
        key: u32,
        version: u32,
    }

    impl<const STREAMS_MAX: usize> TestContext<STREAMS_MAX> {
        fn new(streams: Vec<Vec<Value>>) -> Self {
            Self { streams }
        }
    }

    impl Value {
        fn less_than(direction: Direction, a: Self, b: Self) -> bool {
            let order = a.key.cmp(&b.key);
            let order = match direction {
                Direction::Ascending => order,
                Direction::Descending => order.reverse(),
            };
            match order {
                std::cmp::Ordering::Less => true,
                std::cmp::Ordering::Equal => a.version < b.version,
                std::cmp::Ordering::Greater => false,
            }
        }
    }

    impl<const STREAMS_MAX: usize> MergeStream for TestContext<STREAMS_MAX> {
        type Key = u32;
        type Value = Value;

        fn stream_peek(&mut self, stream_index: u32) -> Result<Option<u32>, Pending> {
            let stream = &self.streams[stream_index as usize];
            if stream.is_empty() {
                return Ok(None);
            }
            Ok(Some(stream[0].key))
        }

        fn stream_pop(&mut self, stream_index: u32) -> Value {
            let stream = &mut self.streams[stream_index as usize];
            let value = stream[0];
            stream.remove(0);
            value
        }

        fn value_key(value: &Self::Value) -> u32 {
            value.key
        }
    }

    /// Reference implementation: flatten, sort, deduplicate consecutive equal keys.
    fn merge_naive(streams: &[&[u32]], direction: Direction) -> Vec<Value> {
        let mut result: Vec<Value> = streams
            .iter()
            .enumerate()
            .flat_map(|(stream_index, keys)| {
                keys.iter().map(move |&key| Value { key, version: stream_index as u32 })
            })
            .collect();

        result.sort_by(|a, b| {
            if Value::less_than(direction, *a, *b) {
                std::cmp::Ordering::Less
            } else if Value::less_than(direction, *b, *a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        let mut deduplicated = Vec::with_capacity(result.len());
        let mut previous_key: Option<u32> = None;
        for value in result {
            if previous_key.is_some_and(|p| p == value.key) {
                continue;
            }
            previous_key = Some(value.key);
            deduplicated.push(value);
        }
        deduplicated
    }

    fn merge<const STREAMS_MAX: usize, const NODE_COUNT_MAX: usize>(
        direction: Direction,
        streams_keys: &[&[u32]],
        expect: Option<&[Value]>,
    ) -> Result<(), Box<dyn Error>> {
        let mut actual: Vec<Value> = Vec::new();

        let streams: Vec<Vec<Value>> = streams_keys
            .iter()
            .enumerate()
            .map(|(i, stream_keys)| {
                stream_keys.iter().map(|&key| Value { key, version: i as u32 }).collect()
            })
            .collect();

        let expect_naive = merge_naive(streams_keys, direction);

        if let Some(expect_explicit) = expect {
            assert_eq!(expect_naive, expect_explicit);
        }

        let context = TestContext::<STREAMS_MAX>::new(streams);
        let mut kway = KWayMergeIterator::<
            TestContext<STREAMS_MAX>,
            STREAMS_MAX,
            NODE_COUNT_MAX,
            true,
        >::init(context, streams_keys.len() as u16, direction);

        while let Some(value) = kway.pop()? {
            actual.push(value);
        }

        assert_eq!(expect_naive, actual);
        Ok(())
    }

    #[test]
    fn k_way_merge_unit() -> Result<(), Box<dyn Error>> {
        // Empty stream.
        merge::<1, 1>(Direction::Ascending, &[], Some(&[]))?;

        merge::<1, 1>(
            Direction::Ascending,
            &[&[0, 3, 4, 8]],
            Some(&[value(0, 0), value(3, 0), value(4, 0), value(8, 0)]),
        )?;
        merge::<1, 1>(
            Direction::Descending,
            &[&[8, 4, 3, 0]],
            Some(&[value(8, 0), value(4, 0), value(3, 0), value(0, 0)]),
        )?;
        merge::<3, 4>(
            Direction::Ascending,
            &[&[0, 3, 4, 8, 11], &[2, 11, 12, 13, 15], &[1, 2, 11]],
            Some(&[
                value(0, 0),
                value(1, 2),
                value(2, 1),
                value(3, 0),
                value(4, 0),
                value(8, 0),
                value(11, 0),
                value(12, 1),
                value(13, 1),
                value(15, 1),
            ]),
        )?;
        merge::<3, 4>(
            Direction::Descending,
            &[&[11, 8, 4, 3, 0], &[15, 13, 12, 11, 2], &[11, 2, 1]],
            Some(&[
                value(15, 1),
                value(13, 1),
                value(12, 1),
                value(11, 0),
                value(8, 0),
                value(4, 0),
                value(3, 0),
                value(2, 1),
                value(1, 2),
                value(0, 0),
            ]),
        )?;

        merge::<32, 32>(
            Direction::Ascending,
            &[&[0, 3, 4, 8]],
            Some(&[value(0, 0), value(3, 0), value(4, 0), value(8, 0)]),
        )?;

        merge::<32, 32>(
            Direction::Descending,
            &[&[11, 8, 4, 3, 0], &[15, 13, 12, 11, 2], &[11, 2, 1]],
            Some(&[
                value(15, 1),
                value(13, 1),
                value(12, 1),
                value(11, 0),
                value(8, 0),
                value(4, 0),
                value(3, 0),
                value(2, 1),
                value(1, 2),
                value(0, 0),
            ]),
        )?;

        Ok(())
    }

    /// Upstream test "k_way_merge: exhaustigen": N=3 streams of up to M=2 keys each.
    #[test]
    fn k_way_merge_exhaustigen() -> Result<(), Box<dyn Error>> {
        const N: usize = 3;
        const M: u32 = 2;

        let mut g = Gen::default();
        while !g.done() {
            let direction = g.enum_value(&[Direction::Ascending, Direction::Descending]);

            let mut streams: Vec<Vec<u32>> = Vec::with_capacity(N);
            for _ in 0..N {
                let key_count = g.int_inclusive(M) as usize;
                let mut keys: Vec<u32> = (0..key_count).map(|_| g.int_inclusive(M)).collect();
                keys.sort_unstable();
                if direction == Direction::Descending {
                    keys.reverse();
                }
                streams.push(keys);
            }
            let streams_refs: Vec<&[u32]> = streams.iter().map(Vec::as_slice).collect();

            merge::<3, 4>(direction, &streams_refs, None)?;
        }
        Ok(())
    }

    /// Upstream: `FuzzTestContextType` — injects spurious `Pending` from peek.
    ///
    /// DEVIATION: upstream shares a single PRNG pointer between the context and the test
    /// driver; here the context owns a copy of the (Copy) PRNG, so the two streams of draws
    /// advance independently. Test-only randomness, sequences are not part of the spec.
    struct FuzzTestContext<const STREAMS_MAX: usize> {
        prng: Prng,
        inner: TestContext<STREAMS_MAX>,
    }

    impl<const STREAMS_MAX: usize> MergeStream for FuzzTestContext<STREAMS_MAX> {
        type Key = u32;
        type Value = Value;

        fn stream_peek(&mut self, stream_index: u32) -> Result<Option<u32>, Pending> {
            if self.prng.chance(ratio(5, 100)) {
                return Err(Pending);
            }
            self.inner.stream_peek(stream_index)
        }

        fn stream_pop(&mut self, stream_index: u32) -> Value {
            self.inner.stream_pop(stream_index)
        }

        fn value_key(value: &Self::Value) -> u32 {
            value.key
        }
    }

    /// DEVIATION: replaces `prng.enum_weighted(Declarations, .{ .pop = 98, .reset = 2 })`
    /// (reflection over declarations is not portable); the weights are preserved.
    fn fuzz_merge(
        direction: Direction,
        streams_keys: &[&[u32]],
        expect: &[Value],
        prng: &mut Prng,
    ) {
        let mut actual: Vec<Value> = Vec::new();

        let streams: Vec<Vec<Value>> = streams_keys
            .iter()
            .enumerate()
            .map(|(i, stream_keys)| {
                stream_keys.iter().map(|&key| Value { key, version: i as u32 }).collect()
            })
            .collect();

        let context = FuzzTestContext::<32> { prng: *prng, inner: TestContext::<32>::new(streams) };
        let mut kway = KWayMergeIterator::<FuzzTestContext<32>, 32, 32, true>::init(
            context,
            streams_keys.len() as u16,
            direction,
        );

        let mut values_popped: usize = 0;
        while values_popped < expect.len() {
            let roll = prng.gen_int_inclusive_u8(99);
            if roll < 98 {
                match kway.pop() {
                    Err(Pending) => {}
                    Ok(None) => break,
                    Ok(Some(value)) => {
                        actual.push(value);
                        values_popped += 1;
                    }
                }
            } else {
                kway.reset();
            }
        }

        assert_eq!(expect, actual);
    }

    fn fuzz_stream_len(prng: &mut Prng, stream_key_count_max: u32) -> u32 {
        // DEVIATION: replaces enum_weighted(.zero=5, .max=5, .random=90).
        let roll = prng.gen_int_inclusive_u8(99);
        if roll < 5 {
            0
        } else if roll < 10 {
            stream_key_count_max
        } else {
            prng.gen_int_inclusive_u32(stream_key_count_max)
        }
    }

    fn fuzz_stream_keys(prng: &mut Prng, stream: &mut [u32]) {
        let key_max = 512 + prng.gen_int_inclusive_u32(1023 - 512);
        // DEVIATION: replaces enum_weighted(.all_same=5, .random=95) and `prng.fill`
        // over `sliceAsBytes`; random bytes are drawn per element instead.
        if prng.chance(ratio(5, 100)) {
            let key = prng.int_u64() as u32;
            stream.fill(key);
        } else {
            let mut bytes = vec![0_u8; std::mem::size_of_val(&stream[..])];
            prng.fill(&mut bytes);
            for (chunk, key) in bytes.as_chunks::<4>().0.iter().zip(stream.iter_mut()) {
                *key = u32::from_le_bytes(*chunk);
            }
        }
        for key in stream.iter_mut() {
            *key %= key_max;
        }
        stream.sort_unstable();
    }

    #[test]
    fn k_way_merge_fuzz() {
        let mut prng = Prng::from_seed(42);
        let stream_key_count_max: u32 = 1024;

        for k in 0..32_usize {
            let mut streams: Vec<Vec<u32>> = Vec::with_capacity(k);
            for _ in 0..k {
                let len = fuzz_stream_len(&mut prng, stream_key_count_max) as usize;
                let mut keys = vec![0_u32; len];
                fuzz_stream_keys(&mut prng, &mut keys);
                streams.push(keys);
            }
            let streams_refs: Vec<&[u32]> = streams.iter().map(Vec::as_slice).collect();

            let mut expect = merge_naive(&streams_refs, Direction::Ascending);

            fuzz_merge(Direction::Ascending, &streams_refs, &expect, &mut prng);

            for stream in &mut streams {
                stream.reverse();
            }
            expect.reverse();

            let streams_refs: Vec<&[u32]> = streams.iter().map(Vec::as_slice).collect();
            fuzz_merge(Direction::Descending, &streams_refs, &expect, &mut prng);
        }
    }

    /// Direct exercise of [`TournamentTree`] against a brute-force reference.
    #[test]
    fn tournament_tree_matches_brute_force() {
        // Streams of distinct ids with overlapping keys; drain the tree one entrant at a time
        // and confirm it always yields the global minimum (ascending) among live contestants.
        let keys_per_stream: [&[u32]; 7] =
            [&[1, 5, 9], &[1, 2, 9], &[7], &[], &[0], &[5, 5, 6], &[3, 8, 10]];

        let mut heads: Vec<Vec<(u32, u32)>> = keys_per_stream
            .iter()
            .map(|keys| keys.iter().copied().enumerate().map(|(i, k)| (k, i as u32)).collect())
            .collect();

        let mut contestants = [Node::<u32>::SENTINEL; 8];
        for (id, head) in heads.iter().enumerate() {
            if let Some(&(key, _)) = head.first() {
                contestants[id] = Node { key, id: id as u32 };
            }
        }

        let mut tree = TournamentTree::<u32, 8>::init(
            Direction::Ascending,
            &mut contestants,
            heads.len() as u16,
        );

        let mut merged: Vec<u32> = Vec::new();
        while tree.contestants_left > 0 {
            let winner_id = tree.winner().id as usize;
            let &(key, _) = heads[winner_id].first().unwrap_or(&(u32::MAX, 0));
            tree.pop_winner(if heads[winner_id].is_empty() { None } else { Some(key) });
            if tree.contestants_left == 0 {
                break;
            }
            // Pop from the NEW winner's stream (see KWayMergeIterator::pop ordering).
            let new_winner_id = tree.winner().id as usize;
            if !heads[new_winner_id].is_empty() {
                let (k, _) = heads[new_winner_id].remove(0);
                merged.push(k);
                let entrant = heads[new_winner_id].first().map(|&(k, _)| k);
                tree.pop_winner(entrant);
            }
        }

        // The interleaving above must produce globally non-decreasing keys.
        let mut previous: Option<u32> = None;
        for key in merged {
            if let Some(p) = previous {
                assert!(p <= key);
            }
            previous = Some(key);
        }
    }

    #[test]
    fn tournament_key_sentinels_match_upstream() {
        assert_eq!(<u8 as TournamentKey>::SENTINEL_KEY, u8::MAX);
        assert_eq!(<u16 as TournamentKey>::SENTINEL_KEY, u16::MAX);
        assert_eq!(<u32 as TournamentKey>::SENTINEL_KEY, u32::MAX);
        assert_eq!(<u64 as TournamentKey>::SENTINEL_KEY, u64::MAX);
        assert_eq!(<u128 as TournamentKey>::SENTINEL_KEY, u128::MAX);
        assert_eq!(Node::<u32>::ID_SENTINEL, u32::MAX);
    }
}
