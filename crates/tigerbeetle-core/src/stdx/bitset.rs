//! A dynamically allocated bitset.
//!
//! Port of the `std.bit_set.DynamicBitSetUnmanaged` API surface used by
//! `src/vsr/free_set.zig` (initEmpty/initFull, count/capacity, set/unset/isSet/setValue,
//! iterators over set or unset bits, forward or reverse).
//!
//! DEVIATION: upstream uses Zig std's bitset; this is a from-scratch safe-Rust implementation
//! with identical observable behavior (words of 64 bits, little-endian bit order within words).

/// Number of bits per word (upstream `MaskInt`).
pub const WORD_BITS: usize = u64::BITS as usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
    /// The number of addressable bits; the last word may hold padding bits (always zero).
    bit_length: usize,
}

/// Which bits an iterator yields (upstream `iterator(.{ .kind = ... })`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitKind {
    Set,
    Unset,
}

/// Iteration direction (upstream `iterator(.{ .direction = ... })`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Reverse,
}

impl BitSet {
    /// All bits cleared (upstream `initEmpty`).
    #[must_use]
    pub fn new_empty(bit_length: usize) -> Self {
        Self { words: vec![0_u64; bit_length.div_ceil(WORD_BITS)], bit_length }
    }

    /// All bits set, including padding bits in the final word (upstream `initFull`).
    #[must_use]
    pub fn new_full(bit_length: usize) -> Self {
        let mut result = Self { words: vec![u64::MAX; bit_length.div_ceil(WORD_BITS)], bit_length };
        if let Some(last) = result.words.last_mut() {
            *last = mask_for(bit_length);
        }
        result
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.bit_length
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bit_length == 0
    }

    /// The number of bits in this bit set (upstream `capacity()` returns `bit_length`).
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.bit_length
    }

    #[must_use]
    pub fn empty(&self) -> bool {
        self.count() == 0
    }

    #[must_use]
    pub fn full(&self) -> bool {
        self.count() == self.bit_length
    }

    /// The number of set bits. Padding bits are never set.
    #[must_use]
    pub fn count(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// # Panics
    /// Panics if `index >= len`.
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.len());
        self.words[index / WORD_BITS] & (1_u64 << (index % WORD_BITS)) != 0
    }

    /// # Panics
    /// Panics if `index >= len`.
    pub fn set(&mut self, index: usize) {
        assert!(index < self.len());
        self.words[index / WORD_BITS] |= 1_u64 << (index % WORD_BITS);
    }

    /// # Panics
    /// Panics if `index >= len`.
    pub fn unset(&mut self, index: usize) {
        assert!(index < self.len());
        self.words[index / WORD_BITS] &= !(1_u64 << (index % WORD_BITS));
    }

    /// # Panics
    /// Panics if `index >= len`.
    pub fn toggle(&mut self, index: usize) {
        assert!(index < self.len());
        self.words[index / WORD_BITS] ^= 1_u64 << (index % WORD_BITS);
    }

    /// # Panics
    /// Panics if `index >= len`.
    pub fn set_value(&mut self, index: usize, value: bool) {
        if value {
            self.set(index);
        } else {
            self.unset(index);
        }
    }

    /// The backing words covering exactly `len()` bits (upstream `bit_set_masks()`).
    /// Padding bits in the final word are always zero.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Mutable counterpart of [`Self::words`].
    /// Callers must keep padding bits zeroed to preserve [`Self::count`] invariants.
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    /// Flip every addressable bit; padding bits stay zero (upstream `toggleAll`).
    pub fn toggle_all(&mut self) {
        for word in &mut self.words {
            *word = !*word;
        }
        // Re-zero padding bits in the final word:
        if let Some(last) = self.words.last_mut() {
            *last &= mask_for(self.bit_length);
        }
    }

    /// Iterate bits by kind and direction (upstream `iterator(.{ .kind, .direction })`).
    #[must_use]
    pub fn iter(&self, kind: BitKind, direction: Direction) -> BitIter<'_> {
        BitIter {
            bitset: self,
            kind,
            forward: matches!(direction, Direction::Forward),
            position: match direction {
                Direction::Forward => 0,
                Direction::Reverse => self.len(),
            },
        }
    }

    /// The index of the first set bit, or `None` (upstream `findFirstSet`).
    #[must_use]
    pub fn find_first_set(&self) -> Option<usize> {
        self.iter(BitKind::Set, Direction::Forward).next()
    }

    /// The index of the last set bit, or `None` (upstream `findLastSet`).
    #[must_use]
    pub fn find_last_set(&self) -> Option<usize> {
        self.iter(BitKind::Set, Direction::Reverse).next()
    }
}

/// Mask covering exactly `bit_length` bits within the final word (padding bits zeroed).
fn mask_for(bit_length: usize) -> u64 {
    let remainder = bit_length % WORD_BITS;
    if remainder == 0 { u64::MAX } else { (1_u64 << remainder) - 1 }
}

/// Iterator over set or unset bits, forward or reverse.
#[derive(Clone, Debug)]
pub struct BitIter<'a> {
    bitset: &'a BitSet,
    kind: BitKind,
    forward: bool,
    position: usize,
}

impl Iterator for BitIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            let index = if self.forward {
                if self.position >= self.bitset.len() {
                    return None;
                }
                let index = self.position;
                self.position += 1;
                index
            } else if self.position == 0 {
                return None;
            } else {
                self.position -= 1;
                self.position
            };
            if self.bitset.get(index) == (self.kind == BitKind::Set) {
                return Some(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdx::prng::Prng;

    /// Fuzz against a `Vec<bool>` reference model, mirroring the pattern of upstream's
    /// free-set bit-set tests (`p < prng.int_inclusive(usize, 100)` model fills).
    #[test]
    fn bit_set_fuzz_against_model() {
        let mut prng = Prng::from_seed(0);

        for bit_length in [1_usize, 63, 64, 65, 100, 128, 4096] {
            let mut bitset = BitSet::new_empty(bit_length);
            let mut model = vec![false; bit_length];

            for _ in 0..1000 {
                let index = prng.range_inclusive_usize(0, bit_length - 1);
                let p = prng.range_inclusive_usize(0, 99);
                bitset.set_value(index, p < 50);
                model[index] = p < 50;

                // Spot-check reads agree:
                assert_eq!(bitset.get(index), model[index]);
            }

            assert_eq!(bitset.count(), model.iter().filter(|bit| **bit).count());

            // Full iteration agrees with the model, both kinds and directions:
            let set_bits: Vec<usize> = bitset.iter(BitKind::Set, Direction::Forward).collect();
            let model_set: Vec<usize> =
                model.iter().enumerate().filter(|(_, bit)| **bit).map(|(index, _)| index).collect();
            assert_eq!(set_bits, model_set);

            let unset_bits: Vec<usize> = bitset.iter(BitKind::Unset, Direction::Forward).collect();
            let model_unset: Vec<usize> = model
                .iter()
                .enumerate()
                .filter(|(_, bit)| !**bit)
                .map(|(index, _)| index)
                .collect();
            assert_eq!(unset_bits, model_unset);

            let set_reverse: Vec<usize> = bitset.iter(BitKind::Set, Direction::Reverse).collect();
            let mut model_reverse = model_set.clone();
            model_reverse.reverse();
            assert_eq!(set_reverse, model_reverse);

            // find_first_set/find_last_set:
            assert_eq!(bitset.find_first_set(), model_set.first().copied());
            assert_eq!(bitset.find_last_set(), model_set.last().copied());
        }
    }

    #[test]
    fn bit_set_full_and_empty_extremes() {
        let full = BitSet::new_full(130);
        assert_eq!(full.len(), 130);
        assert_eq!(full.capacity(), 130); // upstream capacity() == bit_length
        assert!(full.full());
        assert_eq!(full.count(), 130);
        // Padding bits are not addressable and never yielded as unset either:
        assert!(!full.iter(BitKind::Unset, Direction::Forward).any(|_| true));

        let empty = BitSet::new_empty(64);
        assert!(empty.empty());
        assert_eq!(empty.find_first_set(), None);
        assert_eq!(empty.find_last_set(), None);

        let mut single = BitSet::new_empty(1);
        single.set(0);
        assert_eq!(single.find_first_set(), Some(0));
        assert_eq!(single.find_last_set(), Some(0));
        single.toggle(0);
        assert_eq!(single.find_first_set(), None);
    }
}
