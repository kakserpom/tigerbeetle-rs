//! Encode or decode a bitset using Daniel Lemire's EWAH codec.
//! ("Histogram-Aware Sorting for Enhanced Word-Aligned Compression in Bitmap Indexes")
//!
//! EWAH uses only two types of words, where the first type is a 64-bit verbatim ("literal") word.
//! The second type of word is a marker word:
//! * The first bit indicates which uniform word will follow.
//! * The next 31 bits are used to store the number of uniform words.
//! * The last 32 bits are used to store the number of literal words following the uniform words.
//!
//! EWAH bitmaps begin with a marker word. A 'marker' looks like (assuming a 64-bit word):
//! `[uniform_bit:u1][uniform_word_count:u31(LE)][literal_word_count:u32(LE)]`
//! and is immediately followed by `literal_word_count` 64-bit literals.
//! When decoding a marker, the uniform words precede the literal words.
//!
//! This encoding requires that the architecture is little-endian with 64-bit words.
//!
//! Port of `src/ewah.zig`.
//!
//! DEVIATION: upstream is generic over word widths (`ewah(u8)` … `ewah(usize)`); every consumer
//! in the codebase (`FreeSet.Word`, checkpoint trailers) uses `u64`, so this port implements the
//! codec for `u64` only.

#![allow(clippy::doc_markdown)] // doc comments are ported verbatim from upstream

/// Upstream `MarkerUniformCount` (u31), widened to u32 for storage.
pub const MARKER_UNIFORM_WORD_COUNT_MAX: u32 = (1 << 31) - 1;

/// Upstream `MarkerLiteralCount` (u32).
pub const MARKER_LITERAL_WORD_COUNT_MAX: u32 = u32::MAX;

const WORD_BYTES: usize = core::mem::size_of::<u64>();

/// Upstream `Marker` packed into a 64-bit little-endian word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Marker {
    /// Whether the uniform word is all 0s or all 1s.
    uniform_bit: u8,
    /// 31-bit number of uniform words following the marker.
    uniform_word_count: u32,
    /// 32-bit number of literal words following the uniform words.
    literal_word_count: u32,
}

impl Marker {
    fn pack(self) -> u64 {
        assert!(self.uniform_bit <= 1);
        debug_marker_bounds(&self);
        u64::from(self.uniform_bit)
            | (u64::from(self.uniform_word_count) << 1)
            | (u64::from(self.literal_word_count) << 32)
    }

    fn unpack(word: u64) -> Self {
        Self {
            uniform_bit: (word & 1) as u8,
            uniform_word_count: ((word >> 1) & 0x7FFF_FFFF) as u32,
            literal_word_count: (word >> 32) as u32,
        }
    }
}

fn debug_marker_bounds(marker: &Marker) {
    assert!(marker.uniform_word_count <= MARKER_UNIFORM_WORD_COUNT_MAX);
}

fn read_word(source: &[u8], index: usize) -> u64 {
    let start = index * WORD_BYTES;
    let bytes: [u8; WORD_BYTES] = source[start..start + WORD_BYTES]
        .try_into()
        .unwrap_or_else(|_| unreachable!("caller guarantees % WORD_BYTES == 0"));
    u64::from_le_bytes(bytes)
}

fn write_word(target: &mut [u8], index: usize, word: u64) {
    let start = index * WORD_BYTES;
    target[start..start + WORD_BYTES].copy_from_slice(&word.to_le_bytes());
}

/// Incremental decoder over chunked input (upstream `Decoder`).
#[derive(Debug)]
pub struct Decoder<'a> {
    /// The number of bytes of the source buffer (the encoded data) that still need to be
    /// processed.
    source_size_remaining: usize,
    target_words: &'a mut [u64],
    target_index: usize,
    source_literal_words: usize,
}

impl<'a> Decoder<'a> {
    /// Upstream `decode_chunks`.
    #[must_use]
    pub fn new(target_words: &'a mut [u64], source_size: usize) -> Self {
        Self {
            source_size_remaining: source_size,
            target_words,
            target_index: 0,
            source_literal_words: 0,
        }
    }

    /// Returns the number of *words* written to `target_words` by this invocation.
    ///
    /// # Panics
    /// Panics if `source_chunk.len()` is not a multiple of the word size, or if the decoded
    /// output exceeds the target length (upstream asserts/memset-faults analogously).
    pub fn decode_chunk(&mut self, source_chunk: &[u8]) -> usize {
        assert_eq!(source_chunk.len() % WORD_BYTES, 0);

        self.source_size_remaining = self
            .source_size_remaining
            .checked_sub(source_chunk.len())
            .unwrap_or_else(|| panic!("decoded more source bytes than announced"));

        let source_words_len = source_chunk.len() / WORD_BYTES;

        let mut source_index: usize = 0;
        let mut target_index: usize = self.target_index;

        if self.source_literal_words > 0 {
            let literal_word_count_chunk =
                std::cmp::min(self.source_literal_words, source_words_len);

            for i in 0..literal_word_count_chunk {
                let word = read_word(source_chunk, source_index + i);
                self.target_words[target_index + i] = word;
            }
            source_index += literal_word_count_chunk;
            target_index += literal_word_count_chunk;
            self.source_literal_words -= literal_word_count_chunk;
        }

        while source_index < source_words_len {
            assert_eq!(self.source_literal_words, 0);

            let marker = Marker::unpack(read_word(source_chunk, source_index));
            source_index += 1;

            let uniform_value = if marker.uniform_bit == 1 { u64::MAX } else { 0 };
            let uniform_end = target_index + marker.uniform_word_count as usize;
            assert!(uniform_end <= self.target_words.len(), "decoded output overflows");
            self.target_words[target_index..uniform_end].fill(uniform_value);
            target_index = uniform_end;

            let literal_word_count_chunk =
                std::cmp::min(marker.literal_word_count as usize, source_words_len - source_index);
            for i in 0..literal_word_count_chunk {
                let word = read_word(source_chunk, source_index + i);
                self.target_words[target_index + i] = word;
            }
            source_index += literal_word_count_chunk;
            target_index += literal_word_count_chunk;
            self.source_literal_words =
                marker.literal_word_count as usize - literal_word_count_chunk;
        }
        assert!(source_index <= source_words_len);
        assert!(target_index <= self.target_words.len());

        let written = target_index - self.target_index;
        self.target_index = target_index;
        written
    }

    /// # Panics
    /// Panics if more words were decoded than the target holds (upstream asserts the same).
    #[must_use]
    pub fn done(&self) -> bool {
        assert!(self.target_index <= self.target_words.len());

        if self.source_size_remaining == 0 {
            assert_eq!(self.source_literal_words, 0);
            true
        } else {
            false
        }
    }
}

/// Incremental encoder over chunked output (upstream `Encoder`).
#[derive(Debug)]
pub struct Encoder<'a> {
    source_words: &'a [u64],
    source_index: usize,
    /// The number of literals left over from the previous [`Self::encode_chunk`] call that still
    /// need to be copied.
    literal_word_count: usize,

    trailing_zero_runs_count: usize,
}

impl<'a> Encoder<'a> {
    /// Upstream `encode_chunks`.
    #[must_use]
    pub const fn new(source_words: &'a [u64]) -> Self {
        Self { source_words, source_index: 0, literal_word_count: 0, trailing_zero_runs_count: 0 }
    }

    /// Returns the number of bytes written to `target_chunk` by this invocation.
    ///
    /// # Panics
    /// Panics if `target_chunk.len()` is not a multiple of the word size.
    pub fn encode_chunk(&mut self, target_chunk: &mut [u8]) -> usize {
        assert_eq!(target_chunk.len() % WORD_BYTES, 0);
        assert!(self.source_index <= self.source_words.len());
        assert!(self.literal_word_count <= self.source_words.len());

        let target_words_len = target_chunk.len() / WORD_BYTES;
        target_chunk.fill(0);

        let mut target_index: usize = 0;
        let mut source_index: usize = self.source_index;

        if self.literal_word_count > 0 {
            let literal_word_count_chunk = std::cmp::min(self.literal_word_count, target_words_len);

            for i in 0..literal_word_count_chunk {
                write_word(target_chunk, target_index + i, self.source_words[source_index + i]);
            }

            source_index += literal_word_count_chunk;
            target_index += literal_word_count_chunk;
            self.literal_word_count -= literal_word_count_chunk;
        }

        while source_index < self.source_words.len() && target_index < target_words_len {
            assert_eq!(self.literal_word_count, 0);

            let word = self.source_words[source_index];

            let uniform_word_count: usize = if is_literal(word) {
                0
            } else {
                // Measure run length.
                let uniform_max = std::cmp::min(
                    self.source_words.len() - source_index,
                    MARKER_UNIFORM_WORD_COUNT_MAX as usize,
                );
                let mut count = uniform_max;
                for (i, w) in
                    self.source_words[source_index..source_index + uniform_max].iter().enumerate()
                {
                    if *w != word {
                        count = i;
                        break;
                    }
                }
                count
            };
            source_index += uniform_word_count;
            // For consistent encoding, set the run/uniform bit to 0 when there is no run.
            let uniform_bit: u8 = if uniform_word_count == 0 { 0 } else { (word & 1) as u8 };

            // Count sequential literals that immediately follow the run.
            let literals_max = std::cmp::min(
                self.source_words.len() - source_index,
                MARKER_LITERAL_WORD_COUNT_MAX as usize,
            );
            let mut literal_word_count = literals_max;
            for (i, w) in
                self.source_words[source_index..source_index + literals_max].iter().enumerate()
            {
                if !is_literal(*w) {
                    literal_word_count = i;
                    break;
                }
            }

            let marker = Marker {
                uniform_bit,
                uniform_word_count: u32::try_from(uniform_word_count)
                    .unwrap_or_else(|_| unreachable!("bounded by marker max")),
                literal_word_count: u32::try_from(literal_word_count)
                    .unwrap_or_else(|_| unreachable!("bounded by marker max")),
            };
            write_word(target_chunk, target_index, marker.pack());
            target_index += 1;

            let literal_word_count_chunk =
                std::cmp::min(literal_word_count, target_words_len - target_index);
            for i in 0..literal_word_count_chunk {
                write_word(target_chunk, target_index + i, self.source_words[source_index + i]);
            }
            source_index += literal_word_count_chunk;
            target_index += literal_word_count_chunk;

            self.literal_word_count = literal_word_count - literal_word_count_chunk;

            if uniform_bit == 0 && literal_word_count == 0 {
                assert!(uniform_word_count > 0);
                self.trailing_zero_runs_count += 1;
            } else {
                self.trailing_zero_runs_count = 0;
            }
        }
        assert!(source_index <= self.source_words.len());

        self.source_index = source_index;
        target_index * WORD_BYTES
    }

    #[must_use]
    pub const fn done(&self) -> bool {
        self.source_index == self.source_words.len()
    }

    #[must_use]
    pub const fn trailing_zero_runs_count(&self) -> usize {
        self.trailing_zero_runs_count
    }
}

// (This is a helper for testing only.)
/// Decodes the compressed bitset in `source` into `target_words`.
/// Returns the number of *words* written to `target_words`.
///
/// # Panics
/// Panics if the encoding is invalid or overflows the target.
#[must_use]
pub fn decode_all(source: &[u8], target_words: &mut [u64]) -> usize {
    assert_eq!(source.len() % WORD_BYTES, 0);

    let mut decoder = Decoder::new(target_words, source.len());
    let written = decoder.decode_chunk(source);
    assert!(decoder.done());
    written
}

// (This is a helper for testing only.)
/// Returns the number of bytes written to `target`.
///
/// # Panics
/// Panics unless `target.len() == encode_size_max(source_words.len())`.
#[must_use]
pub fn encode_all(source_words: &[u64], target: &mut [u8]) -> usize {
    assert_eq!(target.len(), encode_size_max(source_words.len()));

    let mut encoder = Encoder::new(source_words);
    let written = encoder.encode_chunk(target);
    assert!(encoder.done());

    written
}

/// Returns the maximum number of bytes required to encode `word_count` words.
/// Assumes (pessimistically) that every word will be encoded as a literal.
#[must_use]
pub fn encode_size_max(word_count: usize) -> usize {
    let marker_count = word_count.div_ceil(MARKER_LITERAL_WORD_COUNT_MAX as usize);
    marker_count * WORD_BYTES + word_count * WORD_BYTES
}

const fn is_literal(word: u64) -> bool {
    word != 0 && word != u64::MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdx::prng::Prng;

    /// Port of upstream `ewah_fuzz.generate_bits`: modify `data` such that it has exactly
    /// `bits_set_total` randomly-chosen bits set, with the remaining bits unset.
    fn generate_bits(prng: &mut Prng, data: &mut [u8], bits_set_total: usize) {
        let bits_total = data.len() * 8;
        assert!(bits_set_total <= bits_total);

        // Start off full or empty to save some work.
        let init_empty = bits_set_total < bits_total / 2;
        if init_empty {
            data.fill(0);
        } else {
            data.fill(u8::MAX);
        }

        let mut bits_set = if init_empty { 0 } else { bits_total };
        while bits_set != bits_set_total {
            let bit = prng.range_inclusive_usize(0, bits_total - 1);
            let word = bit / 8;
            let mask = 1_u8 << (bit % 8);

            if init_empty {
                if data[word] & mask != 0 {
                    continue;
                }
                data[word] |= mask;
                bits_set += 1;
            } else {
                if data[word] & mask == 0 {
                    continue;
                }
                data[word] &= !mask;
                bits_set -= 1;
            }
        }
    }

    /// Port of upstream `ewah_fuzz.ContextType.test_encode_decode`: encode in chunks of
    /// `encode_chunk_words_count`, decode in chunks of `decode_chunk_words_count`, and compare.
    fn test_encode_decode(
        decoded_expect: &[u64],
        encoded_scratch: &mut [u8],
        decoded_actual: &mut [u64],
        encode_chunk_words_count: usize,
        decode_chunk_words_count: usize,
    ) -> usize {
        assert!(!decoded_expect.is_empty());

        let mut encoder = Encoder::new(decoded_expect);
        let mut encoded_size: usize = 0;
        while !encoder.done() {
            let chunk_words_count = std::cmp::min(
                (encoded_scratch.len() - encoded_size) / WORD_BYTES,
                encode_chunk_words_count,
            );

            let chunk_end = encoded_size + chunk_words_count * WORD_BYTES;
            let chunk = &mut encoded_scratch[encoded_size..chunk_end];

            encoded_size += encoder.encode_chunk(chunk);
        }

        let mut decoder = Decoder::new(decoded_actual, encoded_size);
        let mut decoded_actual_size: usize = 0;
        let mut decoder_input_offset: usize = 0;
        while decoder_input_offset < encoded_size {
            let chunk_size = std::cmp::min(
                encoded_size - decoder_input_offset,
                decode_chunk_words_count * WORD_BYTES,
            );
            assert!(chunk_size.is_multiple_of(WORD_BYTES));

            let chunk = &encoded_scratch[decoder_input_offset..decoder_input_offset + chunk_size];

            decoded_actual_size += decoder.decode_chunk(chunk);
            decoder_input_offset += chunk_size;
        }
        assert!(decoder.done());

        assert_eq!(decoded_expect.len(), decoded_actual_size);
        assert_eq!(decoded_expect, &decoded_actual[..decoded_actual_size]);
        encoded_size
    }

    /// Port of upstream test "ewah encode→decode cycle" (Word=u64 only — see module DEVIATION).
    #[test]
    fn ewah_encode_decode_cycle() {
        const DECODED_LEN: usize = 1024;

        let mut prng = Prng::from_seed(0);

        let mut decoded_expect_bytes = vec![0_u8; DECODED_LEN * WORD_BYTES];
        let mut encoded_actual = vec![0_u8; encode_size_max(DECODED_LEN)];
        let mut decoded_actual = vec![0_u64; DECODED_LEN];

        // Patterns: all-zero, all-ones, and a random mix with an exact set-bit count.
        for pattern in 0..3 {
            match pattern {
                0 => decoded_expect_bytes.fill(0),
                1 => decoded_expect_bytes.fill(u8::MAX),
                _ => generate_bits(&mut prng, &mut decoded_expect_bytes, DECODED_LEN * 64 / 3),
            }
            let decoded_expect: Vec<u64> = decoded_expect_bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| u64::from_le_bytes(*chunk))
                .collect();

            for chunk_count in [1_usize, 2, 4, 5, 8, 16, 17, 32] {
                let encode_chunk_words_count = DECODED_LEN / chunk_count;
                let decode_chunk_words_count = DECODED_LEN / chunk_count;

                let encoded_size = test_encode_decode(
                    &decoded_expect,
                    &mut encoded_actual,
                    &mut decoded_actual,
                    std::cmp::max(encode_chunk_words_count, 1),
                    std::cmp::max(decode_chunk_words_count, 1),
                );
                assert!(encoded_size <= encoded_actual.len());
            }
        }
    }

    /// Port of upstream test "ewah Word=u8" core assertions, adapted to u64:
    /// encoding the empty bitset yields a single marker with no uniform/literal words.
    #[test]
    fn ewah_empty_bitset_encoding() {
        let source = [0_u64; 4];
        let mut target = vec![0_u8; encode_size_max(source.len())];

        let size = encode_all(&source, &mut target);
        // One marker word:
        assert_eq!(WORD_BYTES, size);

        let mut target_decoded = vec![7_u64; source.len()];
        let words = decode_all(&target[..size], &mut target_decoded);
        assert_eq!(source.len(), words);
        assert_eq!(&source[..], &target_decoded[..]);
    }

    #[test]
    fn ewah_marker_pack_unpack_round_trip() {
        let marker = Marker {
            uniform_bit: 1,
            uniform_word_count: MARKER_UNIFORM_WORD_COUNT_MAX,
            literal_word_count: 123_456,
        };
        assert_eq!(marker, Marker::unpack(marker.pack()));

        let zero = Marker { uniform_bit: 0, uniform_word_count: 0, literal_word_count: 0 };
        assert_eq!(0, zero.pack());
    }
}
