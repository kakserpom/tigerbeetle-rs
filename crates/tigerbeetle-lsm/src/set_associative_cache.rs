//! A set-associative cache with CLOCK Nth-Chance eviction: each key hashes to a set of
//! consecutive ways, and a per-way reference count decides which way to evict.
//!
//! Upstream: `src/lsm/set_associative_cache.zig`.
//!
//! DEVIATION: upstream checks layout invariants at compile time (`@compileError`) and
//! specializes storage types per layout (`Tag`, `Count`, `Clock`, SIMD `@Vector`s); this
//! port runs the same assertions at construction and uses runtime-width packed arrays plus
//! a scalar tag search building the identical ways bitmask.
//!
//! DEVIATION: upstream stores `metrics` behind a heap pointer so `get()`/`get_index()` can
//! take `*const Self` while mutating counters; safe Rust requires interior mutability for
//! that, so they take `&mut self` here and the allocation is dropped.
//!
//! DEVIATION: upstream aligns `values` to `Layout.value_alignment`; owned Vec storage cannot
//! express that without unsafe. Removed slots hold stale bytes instead of an `undefined`
//! poison value — never read, because occupancy is tracked by `counts`.

#![allow(clippy::cast_possible_truncation)]

use tigerbeetle_core::stdx::{div_ceil, fastrange};

/// Upstream `Layout`. Defaults match upstream's field defaults; specs override them.
pub trait Layout {
    const WAYS: u64 = 16;
    const TAG_BITS: u64 = 8;
    const CLOCK_BITS: u64 = 2;
    const CACHE_LINE_SIZE: u64 = 64;
}

/// Static description of a cache instantiation (upstream comptime parameters).
pub trait SetAssociativeCacheSpec: Layout + 'static {
    type Key: Copy + PartialEq + core::fmt::Debug;
    type Value: Copy + Default + core::fmt::Debug;

    fn key_from_value(value: &Self::Value) -> Self::Key;
    fn hash(key: Self::Key) -> u64;
}

/// A short, partial hash of a Key, corresponding to a Value.
/// Because the tag is small, collisions are possible:
/// `tag(v₁) = tag(v₂)` does not imply `v₁ = v₂`.
/// However, most of the time, where the tag differs, a full key comparison can be avoided.
/// Since tags are 16-32x smaller than keys, they can also be kept hot in cache.
///
/// Upstream sizes this type to exactly `TAG_BITS` bits; we always store `u16` and mask on
/// write (see the struct-level deviation note).
type Tag = u16;

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    hits: u64,
    misses: u64,
    value_count: u64,
}

/// Whether [`SetAssociativeCache::upsert`] updated an existing entry or inserted a new one
/// (upstream `UpdateOrInsert`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOrInsert {
    Update,
    Insert,
}

/// Result of [`SetAssociativeCache::upsert`] (upstream anonymous return struct).
#[derive(Clone, Copy, Debug)]
pub struct Upserted<V> {
    pub index: usize,
    pub updated: UpdateOrInsert,
    pub evicted: Option<V>,
}

/// Upstream comptime `clock_hand_bits = math.log2_int(u64, layout.ways)` and its asserts.
const fn clock_hand_bits(ways: u64) -> u32 {
    assert!(ways.is_power_of_two());
    let bits = ways.trailing_zeros();
    assert!(bits.is_power_of_two());
    assert!((1_u64 << bits) == ways);
    bits
}

/// Each Key is associated with a set of n consecutive ways (or slots) that may contain the
/// Value.
pub struct SetAssociativeCache<S: SetAssociativeCacheSpec> {
    name: &'static str,
    sets: u64,

    metrics: Metrics,

    tags: Vec<Tag>,

    /// When the corresponding Count is zero, the Value is absent.
    values: Vec<S::Value>,

    /// Each value has a Count, which tracks the number of recent reads.
    ///
    /// * A Count is incremented when the value is accessed by `get`.
    /// * A Count is decremented when a cache write to the value's Set misses.
    /// * The value is evicted when its Count reaches zero.
    counts: PackedUnsignedIntegerArray,

    /// Each set has a Clock: a counter that cycles between each of the set's ways
    /// (i.e. slots).
    ///
    /// On cache write, entries are checked for occupancy (or eviction) beginning from the
    /// clock's position, wrapping around.
    ///
    /// The algorithm implemented is "CLOCK Nth-Chance" — each way has more than one bit,
    /// to give ways more than one chance before eviction.
    ///
    /// * A similar algorithm called "RRIParoo" is described in
    ///   "Kangaroo: Caching Billions of Tiny Objects on Flash".
    /// * For more general information on CLOCK algorithms, see:
    ///   <https://en.wikipedia.org/wiki/Page_replacement_algorithm>.
    clocks: PackedUnsignedIntegerArray,
}

impl<S: SetAssociativeCacheSpec> SetAssociativeCache<S> {
    /// We don't require `value_count_max` in [`Self::new`] to be a power of 2, but we do
    /// require it to be a multiple of `value_count_max_multiple`. The calculation below
    /// follows from a multiple which will satisfy all asserts.
    #[must_use]
    pub fn value_count_max_multiple() -> u64 {
        let value_size = core::mem::size_of::<S::Value>() as u64;
        core::cmp::max(
            // `values`:
            (core::cmp::max(value_size, S::CACHE_LINE_SIZE)
                / core::cmp::min(value_size, S::CACHE_LINE_SIZE))
                * S::WAYS,
            S::CACHE_LINE_SIZE * 8 / S::CLOCK_BITS, // `counts`
        )
    }

    /// Upstream comptime layout validation (`SetAssociativeCacheType` body), which runs here
    /// once per instantiation.
    fn validate_layout() {
        assert!(core::mem::size_of::<S::Key>().is_power_of_two());
        assert!(core::mem::size_of::<S::Value>().is_power_of_two());

        assert!(
            S::WAYS == 2 || S::WAYS == 4 || S::WAYS == 16,
            "ways must be 2, 4 or 16 for optimal CLOCK hand size."
        );
        assert!(S::TAG_BITS == 8 || S::TAG_BITS == 16, "tag_bits must be 8 or 16.");
        assert!(
            S::CLOCK_BITS == 1 || S::CLOCK_BITS == 2 || S::CLOCK_BITS == 4,
            "clock_bits must be 1, 2 or 4."
        );

        assert!(S::WAYS.is_power_of_two());
        assert!(S::TAG_BITS.is_power_of_two());
        assert!(S::CLOCK_BITS.is_power_of_two());
        assert!(S::CACHE_LINE_SIZE.is_power_of_two());

        let key_size = core::mem::size_of::<S::Key>() as u64;
        let value_size = core::mem::size_of::<S::Value>() as u64;
        assert!(key_size <= value_size);
        assert!(key_size < S::CACHE_LINE_SIZE);
        assert!(S::CACHE_LINE_SIZE.is_multiple_of(key_size));

        if S::CACHE_LINE_SIZE > value_size {
            assert!(S::CACHE_LINE_SIZE.is_multiple_of(value_size));
        } else {
            assert!(value_size.is_multiple_of(S::CACHE_LINE_SIZE));
        }

        let clock_hand_bits = clock_hand_bits(S::WAYS);

        let tags_per_line = S::CACHE_LINE_SIZE * 8 / (S::WAYS * S::TAG_BITS);
        assert!(tags_per_line > 0);

        let clocks_per_line = S::CACHE_LINE_SIZE * 8 / (S::WAYS * S::CLOCK_BITS);
        assert!(clocks_per_line > 0);

        let clock_hands_per_line = S::CACHE_LINE_SIZE * 8 / u64::from(clock_hand_bits);
        assert!(clock_hands_per_line > 0);
    }

    /// Upstream `init`.
    ///
    /// # Panics
    /// Panics if `value_count_max` violates the layout multiples, or if the static layout
    /// parameters are invalid (upstream asserts / `@compileError`s).
    #[must_use]
    pub fn new(value_count_max: u64, name: &'static str) -> Self {
        Self::validate_layout();

        let sets = value_count_max / S::WAYS;

        assert!(value_count_max > 0);
        assert!(value_count_max >= S::WAYS);
        assert!(value_count_max.is_multiple_of(S::WAYS));

        let values_size_max = value_count_max * core::mem::size_of::<S::Value>() as u64;
        assert!(values_size_max >= S::CACHE_LINE_SIZE);
        assert!(values_size_max.is_multiple_of(S::CACHE_LINE_SIZE));

        let counts_size = value_count_max * S::CLOCK_BITS / 8;
        assert!(counts_size >= S::CACHE_LINE_SIZE);
        assert!(counts_size.is_multiple_of(S::CACHE_LINE_SIZE));

        // Each clock hand is guaranteed (by construction) to not span multiple cache lines.
        // But in order to shrink the lower-bound cache size, we do not require that `clocks`
        // itself is a multiple of the cache line size.
        //
        // TODO(port): upstream soft-asserts (stdx.maybe) the clocks size bounds; we have no
        // warning-only assertion yet.
        #[allow(clippy::let_and_return)]
        let clocks_size = sets * u64::from(clock_hand_bits(S::WAYS)) / 8;

        assert!(value_count_max.is_multiple_of(Self::value_count_max_multiple()));

        let mut cache = Self {
            name,
            sets,
            metrics: Metrics::default(),
            tags: vec![0; value_count_max as usize],
            values: vec![S::Value::default(); value_count_max as usize],
            counts: PackedUnsignedIntegerArray::new(S::CLOCK_BITS as u32, counts_size),
            clocks: PackedUnsignedIntegerArray::new(clock_hand_bits(S::WAYS), clocks_size),
        };
        cache.reset();
        cache
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    #[must_use]
    pub const fn sets(&self) -> u64 {
        self.sets
    }

    #[must_use]
    pub const fn hits(&self) -> u64 {
        self.metrics.hits
    }

    #[must_use]
    pub const fn misses(&self) -> u64 {
        self.metrics.misses
    }

    #[must_use]
    pub const fn value_count(&self) -> u64 {
        self.metrics.value_count
    }

    pub fn reset(&mut self) {
        self.tags.fill(0);
        self.counts.reset();
        self.clocks.reset();
        self.metrics = Metrics::default();
    }

    /// Returns the index of the value for `key`, bumping its reference count on hit.
    pub fn get_index(&mut self, key: S::Key) -> Option<usize> {
        let set = self.associate(key);
        if let Some(way) = self.search(set, key) {
            self.metrics.hits += 1;
            let index = set.offset + u64::from(way);
            // Upstream adds within the fixed-width Count type (`count +| 1`), which
            // saturates at CLOCK_BITS; clamp to the packed slot mask for the same effect.
            let count = core::cmp::min(self.counts.get(index) + 1, self.counts.mask());
            self.counts.set(index, count);
            Some(index as usize)
        } else {
            self.metrics.misses += 1;
            None
        }
    }

    /// Returns the value for `key`, bumping its reference count on hit.
    #[must_use]
    pub fn get(&mut self, key: S::Key) -> Option<&S::Value> {
        let index = self.get_index(key)?;
        Some(&self.values[index])
    }

    /// Iterates over occupied slots (count != 0) as `(index, value)` pairs without
    /// touching reference counts. Crate-visible for fuzz verification, where upstream
    /// reads `cache.values`/`cache.counts` directly.
    #[cfg(test)]
    pub(crate) fn occupied_slots(&self) -> impl Iterator<Item = (usize, &S::Value)> + '_ {
        self.values
            .iter()
            .enumerate()
            // usize -> u64 is a lossless widening on all supported targets.
            .filter(|(index, _)| self.counts.get(u64::try_from(*index).unwrap_or(u64::MAX)) != 0)
    }

    /// Remove a key from the set associative cache if present.
    /// Returns the removed value, if any.
    pub fn remove(&mut self, key: S::Key) -> Option<S::Value> {
        let set = self.associate(key);
        let way = self.search(set, key)?;

        let removed = self.values[set.offset as usize + way as usize];
        self.counts.set(set.offset + u64::from(way), 0);
        self.metrics.value_count -= 1;

        Some(removed)
    }

    /// Hint that the key is less likely to be accessed in the future, without actually
    /// removing it from the cache.
    pub fn demote(&mut self, key: S::Key) {
        let set = self.associate(key);
        let Some(way) = self.search(set, key) else {
            return;
        };

        self.counts.set(set.offset + u64::from(way), 1);
    }

    /// If the key is present in the set, returns the way. Otherwise returns null.
    fn search(&self, set: AssociatedSet, key: S::Key) -> Option<u16> {
        let ways = search_tags(set.tags_slice(self), set.tag, S::WAYS);
        if ways == 0 {
            return None;
        }

        // Iterate over all ways to help the OOO execution.
        for way in 0..S::WAYS {
            let way = way as u16;
            if (ways >> way) & 1 == 1
                && self.counts.get(set.offset + u64::from(way)) > 0
                && S::key_from_value(&self.values[set.offset as usize + way as usize]) == key
            {
                return Some(way);
            }
        }
        None
    }

    /// Upsert a value, evicting an older entry if needed. The evicted value, if an update or
    /// insert was performed, and the index at which the value was inserted are returned.
    ///
    /// # Panics
    /// Panics if the CLOCK hand fails to terminate within the theoretical iteration bound
    /// (upstream `unreachable`).
    pub fn upsert(&mut self, value: &S::Value) -> Upserted<S::Value> {
        let key = S::key_from_value(value);
        let set = self.associate(key);
        if let Some(way) = self.search(set, key) {
            // Overwrite the old entry for this key.
            self.counts.set(set.offset + u64::from(way), 1);
            let evicted = self.values[set.offset as usize + way as usize];
            self.values[set.offset as usize + way as usize] = *value;
            return Upserted {
                index: set.offset as usize + way as usize,
                updated: UpdateOrInsert::Update,
                evicted: Some(evicted),
            };
        }

        let clock_index = set.offset / S::WAYS;

        // Upstream keeps the clock hand in a type whose max int is `WAYS - 1` so that the
        // increment wraps naturally; our u64 arithmetic mod `WAYS` is equivalent.

        // The maximum number of iterations happens when every slot in the set has the maximum
        // count. In this case, the loop will iterate until all counts have been decremented
        // to 1. Then in the next iteration it will decrement a count to 0 and break.
        let count_max = (1_u64 << S::CLOCK_BITS) - 1;
        let clock_iterations_max = S::WAYS * (count_max - 1);

        let mut evicted: Option<S::Value> = None;
        let mut safety_count: u64 = 0;
        let mut way = self.clocks.get(clock_index);
        while safety_count <= clock_iterations_max {
            let mut count = self.counts.get(set.offset + way);
            if count == 0 {
                break; // Way is already free.
            }

            count -= 1;
            self.counts.set(set.offset + way, count);
            if count == 0 {
                // Way has become free.
                evicted = Some(self.values[(set.offset + way) as usize]);
                break;
            }

            safety_count += 1;
            way = (way + 1) % S::WAYS;
        }
        assert!(safety_count <= clock_iterations_max, "clock did not terminate");
        assert_eq!(self.counts.get(set.offset + way), 0);

        self.tags[set.offset as usize + way as usize] = set.tag;
        self.values[set.offset as usize + way as usize] = *value;
        self.counts.set(set.offset + way, 1);
        self.clocks.set(clock_index, (way + 1) % S::WAYS);
        if evicted.is_none() {
            self.metrics.value_count += 1;
        }

        Upserted {
            index: set.offset as usize + way as usize,
            updated: UpdateOrInsert::Insert,
            evicted,
        }
    }

    /// Maps `key` to its set: the partial hash tag plus the base offset of the set's ways.
    /// Upstream's `Set` also carries raw pointers into the arrays; this port resolves
    /// elements through the cache instead.
    fn associate(&self, key: S::Key) -> AssociatedSet {
        let entropy = S::hash(key);

        let tag_mask = (1_u64 << S::TAG_BITS) - 1;
        let tag: Tag = (entropy & tag_mask) as Tag;
        let index = fastrange(entropy, self.sets);
        let offset = index * S::WAYS;

        AssociatedSet { tag, offset }
    }
}

/// Where each set bit represents the index of a way that has the same tag.
///
/// Upstream compares a `@Vector(WAYS, Tag)` and bitcasts the bool vector; this loop builds
/// the same bitmask scalar-wise.
fn search_tags(tags: &[Tag], tag: Tag, ways: u64) -> u16 {
    let mut bits: u16 = 0;
    for (i, t) in tags.iter().enumerate().take(ways as usize) {
        if *t == tag {
            bits |= 1 << i;
        }
    }
    bits
}

/// Hash-derived location within the cache (upstream `Set`).
#[derive(Clone, Copy)]
struct AssociatedSet {
    tag: Tag,
    offset: u64,
}

impl AssociatedSet {
    fn tags_slice<'a, S: SetAssociativeCacheSpec>(
        &self,
        cache: &'a SetAssociativeCache<S>,
    ) -> &'a [Tag] {
        &cache.tags[self.offset as usize..][..S::WAYS as usize]
    }
}

/// A little simpler than PackedIntArray in the std lib, restricted to little endian 64-bit
/// words, and using words exactly without padding
/// (upstream generic `PackedUnsignedIntegerArrayType(UInt)`).
///
/// Upstream specializes the slot width per type parameter; here it is a runtime field.
struct PackedUnsignedIntegerArray {
    bits_per_uint: u32,
    words: Vec<u64>,
}

impl PackedUnsignedIntegerArray {
    /// Allocates enough words to hold integers packing to `size_bytes` bytes, zero-filled.
    ///
    /// # Panics
    /// Panics unless `bits_per_uint` is 1, 2 or 4 (upstream comptime asserts).
    fn new(bits_per_uint: u32, size_bytes: u64) -> Self {
        assert!(bits_per_uint > 0);
        assert!(bits_per_uint < 8);
        assert!(bits_per_uint.is_power_of_two());

        Self { bits_per_uint, words: vec![0; div_ceil(size_bytes as usize, 8)] }
    }

    fn reset(&mut self) {
        self.words.fill(0);
    }

    const fn uints_per_word(&self) -> u64 {
        64 / (self.bits_per_uint as u64)
    }

    const fn mask(&self) -> u64 {
        (1_u64 << self.bits_per_uint) - 1
    }

    /// Returns the unsigned integer at `index`.
    #[must_use]
    fn get(&self, index: u64) -> u64 {
        // Masking the right-shifted word by exactly one slot-width is the equivalent of
        // upstream's truncate-to-UInt.
        self.word(index) >> self.bits_index(index) & self.mask()
    }

    /// Sets the unsigned integer at `index` to `value`.
    ///
    /// # Panics
    /// Panics if `value` does not fit in `bits_per_uint` bits.
    fn set(&mut self, index: u64, value: u64) {
        assert!(value <= self.mask(), "value {} does not fit width {}", value, self.bits_per_uint);
        let bits_index = self.bits_index(index);
        let slot_mask = self.mask() << bits_index;
        let word = self.word_mut(index);
        *word &= !slot_mask;
        *word |= value << bits_index;
    }

    fn word(&self, index: u64) -> u64 {
        self.words[(index / self.uints_per_word()) as usize]
    }

    fn word_mut(&mut self, index: u64) -> &mut u64 {
        let word_index = (index / self.uints_per_word()) as usize;
        &mut self.words[word_index]
    }

    /// Bit offset of `index`'s slot within its word.
    ///
    /// If bits_per_uint=2, then it's normal for the maximum return value to be 62, even
    /// where a word allows bit indexes up to 63 (inclusive). This is because 62 is the bit
    /// index of the highest 2-bit UInt (e.g. bit index + bit length == 64).
    const fn bits_index(&self, index: u64) -> u32 {
        (index % self.uints_per_word()) as u32 * self.bits_per_uint
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::{
        Layout, PackedUnsignedIntegerArray, SetAssociativeCache, SetAssociativeCacheSpec, Tag,
        clock_hand_bits, search_tags,
    };
    use tigerbeetle_core::stdx::prng::Prng;

    macro_rules! sac_spec {
        ($name:ident, $hash:expr) => {
            #[derive(Clone, Copy, Debug, Default)]
            struct $name;

            impl Layout for $name {}

            impl SetAssociativeCacheSpec for $name {
                type Key = u64;
                type Value = u64;

                fn key_from_value(value: &u64) -> u64 {
                    *value
                }

                fn hash(key: u64) -> u64 {
                    $hash(key)
                }
            }
        };
    }

    sac_spec!(IdentityHashSpec, |key| key);

    sac_spec!(BrokenHashSpec, |_| 0_u64); // Intentionally broken to simulate hash collision.

    /// Port of upstream `set_associative_cache_test(...).run()` (default layout).
    fn run_cache_test<S: SetAssociativeCacheSpec<Key = u64, Value = u64>>() {
        // TODO(upstream): Add a nice calculator method to help solve the minimum
        // value_count_max required.
        let mut sac = SetAssociativeCache::<S>::new(16 * 16 * 8, "test");
        let ways = S::WAYS;

        for tag in &sac.tags {
            assert_eq!(*tag, 0);
        }
        for word in sac.counts.words.iter().chain(sac.clocks.words.iter()) {
            assert_eq!(*word, 0);
        }
        assert_eq!(sac.metrics.value_count, 0);

        // Fill up the first set entirely.
        {
            for i in 0..ways {
                assert_eq!(sac.clocks.get(0), i);

                let key = i * sac.sets();
                _ = sac.upsert(&key);
                assert_eq!(sac.counts.get(i), 1);
                assert_eq!(*sac.get(key).expect("present"), key);
                assert_eq!(sac.counts.get(i), 2);
            }
            assert_eq!(sac.clocks.get(0), 0);
            assert_eq!(sac.metrics.value_count, ways);
        }

        // Insert another element into the first set, causing key 0 to be evicted.
        {
            let key = ways * sac.sets();
            _ = sac.upsert(&key);
            assert_eq!(sac.counts.get(0), 1);
            assert_eq!(*sac.get(key).expect("present"), key);
            assert_eq!(sac.counts.get(0), 2);

            assert!(sac.get(0).is_none());

            for i in 1..ways {
                assert_eq!(sac.counts.get(i), 1);
            }
            assert_eq!(sac.metrics.value_count, ways);
        }

        // Ensure removal works.
        {
            let key = 5 * sac.sets();
            assert_eq!(*sac.get(key).expect("present"), key);
            assert_eq!(sac.counts.get(5), 2);

            let removed = sac.remove(key);
            assert_eq!(removed, Some(key));
            assert!(sac.get(key).is_none());
            assert_eq!(sac.counts.get(5), 0);
            assert_eq!(sac.metrics.value_count, ways - 1);
        }

        sac.reset();

        for tag in &sac.tags {
            assert_eq!(*tag, 0);
        }
        for word in sac.counts.words.iter().chain(sac.clocks.words.iter()) {
            assert_eq!(*word, 0);
        }
        assert_eq!(sac.metrics.value_count, 0);

        // Fill up the first set entirely, maxing out the count for each slot.
        {
            let count_max = (1_u64 << S::CLOCK_BITS) - 1;
            for i in 0..ways {
                assert_eq!(sac.clocks.get(0), i);

                let key = i * sac.sets();
                _ = sac.upsert(&key);
                assert_eq!(sac.counts.get(i), 1);
                for j in 2..=count_max {
                    assert_eq!(*sac.get(key).expect("present"), key);
                    assert_eq!(sac.counts.get(i), j);
                }
                assert_eq!(*sac.get(key).expect("present"), key);
                assert_eq!(sac.counts.get(i), count_max);
            }
            assert_eq!(sac.clocks.get(0), 0);
            assert_eq!(sac.metrics.value_count, ways);
        }

        // Insert another element into the first set, causing key 0 to be evicted despite
        // its maxed-out count.
        {
            let key = ways * sac.sets();
            _ = sac.upsert(&key);
            assert_eq!(sac.counts.get(0), 1);
            assert_eq!(*sac.get(key).expect("present"), key);
            assert_eq!(sac.counts.get(0), 2);

            assert!(sac.get(0).is_none());

            for i in 1..ways {
                assert_eq!(sac.counts.get(i), 1);
            }
            assert_eq!(sac.metrics.value_count, ways);
        }
    }

    #[test]
    fn set_associative_cache_eviction() {
        run_cache_test::<IdentityHashSpec>();
    }

    #[test]
    fn set_associative_cache_hash_collision() {
        run_cache_test::<BrokenHashSpec>();
    }

    /// Port of upstream test "PackedUnsignedIntegerArray: unit".
    #[test]
    fn packed_unsigned_integer_array_unit() {
        let mut p = PackedUnsignedIntegerArray::new(2, 8 * 8);
        p.words[1] = 0b1011_0010;

        assert_eq!(p.get(32), 0b10);
        assert_eq!(p.get(32 + 1), 0b00);
        assert_eq!(p.get(32 + 2), 0b11);
        assert_eq!(p.get(32 + 3), 0b10);

        p.set(0, 0b01);
        assert_eq!(p.words[0], 0b0000_0001);
        assert_eq!(p.get(0), 0b01);
        p.set(1, 0b10);
        assert_eq!(p.words[0], 0b0000_1001);
        assert_eq!(p.get(1), 0b10);
        p.set(2, 0b11);
        assert_eq!(p.words[0], 0b0011_1001);
        assert_eq!(p.get(2), 0b11);
        p.set(3, 0b11);
        assert_eq!(p.words[0], 0b1111_1001);
        assert_eq!(p.get(3), 0b11);
        p.set(3, 0b01);
        assert_eq!(p.words[0], 0b0111_1001);
        assert_eq!(p.get(3), 0b01);
        p.set(3, 0b00);
        assert_eq!(p.words[0], 0b0011_1001);
        assert_eq!(p.get(3), 0b00);

        p.set(4, 0b11);
        assert_eq!(
            p.words[0],
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0011_0011_1001
        );
        p.set(31, 0b11);
        assert_eq!(
            p.words[0],
            0b1100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0011_0011_1001
        );
    }

    /// Port of upstream `ContextType` + test "PackedUnsignedIntegerArray: fuzz"
    /// (`UInt ∈ {u1, u2, u4}`, one shared PRNG across widths).
    fn packed_fuzz(prng: &mut Prng, bits: u32) {
        let len = 1024_usize;
        let mut array = PackedUnsignedIntegerArray::new(bits, len as u64 * u64::from(bits) / 8);
        let mut reference = vec![0_u64; len];

        let value_max = (1_u64 << bits) - 1;
        for _ in 0..10_000 {
            let index = prng.index(len);
            let value = prng.gen_int_inclusive_u64(value_max);

            array.set(index as u64, value);
            reference[index] = value;

            for (i, expected) in reference.iter().enumerate() {
                assert_eq!(*expected, array.get(i as u64));
            }
        }
    }

    #[test]
    fn packed_unsigned_integer_array_fuzz() {
        let seed = 42;
        let mut prng = Prng::from_seed(seed);

        packed_fuzz(&mut prng, 1);
        packed_fuzz(&mut prng, 2);
        packed_fuzz(&mut prng, 4);
    }

    /// Port of upstream `search_tags_test(...).run()` over all `{ways} × {tag_bits}` layouts,
    /// including the brute-force `reference.search_tags`.
    #[test]
    fn set_associative_cache_search_tags() {
        const SEED: u64 = 42;
        let mut prng = Prng::from_seed(SEED);

        for ways in [2_u64, 4, 16] {
            for tag_bits in [8_u64, 16] {
                let tag_max = (1_u64 << tag_bits) - 1;

                for _ in 0..10_000 {
                    let mut tags: Vec<Tag> = vec![0; ways as usize];

                    // Upstream fills raw bytes of a `[ways]Tag` array; our Tag is always
                    // u16, so draw within the layout's tag range instead.
                    for t in &mut tags {
                        *t = prng.gen_int_inclusive_u64(tag_max) as Tag;
                    }

                    let tag = prng.gen_int_inclusive_u64(tag_max) as Tag;

                    let mut indexes: Vec<usize> = (0..ways as usize).collect();
                    prng.shuffle(&mut indexes);

                    let matches_count_min = prng.gen_int_inclusive_u64(ways);
                    for &index in &indexes[..matches_count_min as usize] {
                        tags[index] = tag;
                    }

                    // Brute-force reference (upstream `reference.search_tags`).
                    let mut expected: u16 = 0;
                    let mut count = 0_usize;
                    for (i, t) in tags.iter().enumerate() {
                        if *t == tag {
                            expected |= 1 << i;
                            count += 1;
                        }
                    }
                    assert_eq!(u32::from(expected).count_ones() as usize, count);

                    let actual = search_tags(&tags, tag, ways);
                    assert_eq!(expected, actual);
                }
            }
        }
    }

    #[test]
    fn clock_hand_bits_matches_ways() {
        assert_eq!(clock_hand_bits(2), 1);
        assert_eq!(clock_hand_bits(4), 2);
        assert_eq!(clock_hand_bits(16), 4);
    }
}
