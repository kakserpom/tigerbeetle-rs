//! Multi-batch encoding of client requests into a single prepare message.
//!
//! The trailer is an array of [`TrailerItem`]s (one per batch), followed by a
//! [`Postamble`] holding the total number of batches. Batches are laid out back-to-back at
//! the start of the message body; the trailer grows from the end of the buffer towards the
//! payload while encoding, and is compacted next to the payload by [`MultiBatchEncoder::finish`].
//!
//! Upstream: `src/vsr/multi_batch.zig`.
//!
//! DEVIATIONS:
//! - Zig relies on host little-endian pointer casts; this port encodes/decodes explicitly
//!   with `to_le_bytes`/`from_le_bytes`, so buffers need not be aligned (upstream requires
//!   cache-line-aligned bodies and drops that requirement here accordingly);
//! - the encoder holds `Option<&mut [u8]>` instead of an optional slice that becomes
//!   `undefined` after `finish()`.

#![allow(clippy::cast_possible_truncation)] // sizes are bounded by construction

/// The number of batches in the message body (`Postamble`).
const POSTAMBLE_SIZE: usize = 2;
/// One batch's element count in the trailer (`TrailerItem`).
const TRAILER_ITEM_SIZE: usize = 2;

/// `maxInt(u16)` is reserved for padding.
pub const BATCH_COUNT_MAX: u16 = u16::MAX - 1;

/// The maximum number of batches that can be encoded, assuming the worst case single-element
/// batches with the minimum size (maybe empty).
///
/// # Panics
/// Panics unless `batch_size_limit > size_of::<Postamble>()` (upstream asserts).
#[must_use]
pub const fn multi_batch_count_max(batch_size_min: u32, batch_size_limit: u32) -> u16 {
    assert!(batch_size_limit > POSTAMBLE_SIZE as u32);

    let count = (batch_size_limit as usize - POSTAMBLE_SIZE)
        / (batch_size_min as usize + TRAILER_ITEM_SIZE);
    // `core::cmp::min` is not const-stable on this toolchain.
    if count < BATCH_COUNT_MAX as usize { count as u16 } else { BATCH_COUNT_MAX }
}

/// Total space occupied by the trailer: one [`TrailerItem`] per batch plus the
/// [`Postamble`], padded up to a multiple of `element_size` so that the payload stays
/// element-aligned.
///
/// # Panics
/// Panics unless `0 < batch_count <= BATCH_COUNT_MAX` and `element_size` is zero or a power
/// of two (upstream asserts).
#[must_use]
pub const fn trailer_total_size(element_size: u32, batch_count: u16) -> u32 {
    assert!(batch_count > 0);
    assert!(batch_count <= BATCH_COUNT_MAX);
    // Supports zero-sized elements, or any power of two, including 2^0.
    assert!(element_size == 0 || element_size.is_power_of_two());

    let trailer_unpadded_size: u32 =
        (batch_count as u32) * (TRAILER_ITEM_SIZE as u32) + (POSTAMBLE_SIZE as u32);
    if element_size == 0 {
        return trailer_unpadded_size;
    }

    trailer_unpadded_size.div_ceil(element_size) * element_size
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// Decodes a multi-batch message body into its batches.
pub struct MultiBatchDecoder<'a> {
    /// The message payload, excluding the trailer.
    payload: &'a [u8],
    /// The batching metadata (`TrailerItem`s), excluding the postamble; exactly
    /// `batch_count * 2` bytes.
    trailer_items: &'a [u8],

    payload_index: usize,
    batch_index: usize,

    element_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiBatchInvalid;

impl<'a> MultiBatchDecoder<'a> {
    /// Parses the message body used, including the trailer.
    ///
    /// # Errors
    /// Returns [`MultiBatchInvalid`] if the encoding is malformed (counts inconsistent,
    /// padding not filled with sentinels, sizes mismatched, …).
    ///
    /// # Panics
    /// Panics unless `element_size` is zero or a power of two (upstream asserts).
    pub fn new(body: &'a [u8], element_size: u32) -> Result<Self, MultiBatchInvalid> {
        // A closure would capture `parsed` for its whole lifetime, blocking reads between
        // calls, so this is a plain helper instead.
        fn take_suffix(
            body_len: usize,
            parsed: &mut usize,
            count: usize,
        ) -> Result<core::ops::Range<usize>, MultiBatchInvalid> {
            if body_len - *parsed < count {
                return Err(MultiBatchInvalid);
            }
            *parsed += count;
            Ok(body_len - *parsed..body_len - (*parsed - count))
        }

        // Supports zero-sized elements, or any power of two, including 2^0.
        assert!(element_size == 0 || element_size.is_power_of_two());

        // `parsed` counts bytes consumed from the end of the body.
        let mut parsed: usize = 0;
        let postamble_range = take_suffix(body.len(), &mut parsed, POSTAMBLE_SIZE)?;
        let batch_count = read_u16(&body[postamble_range]);
        if batch_count == 0 {
            return Err(MultiBatchInvalid);
        }
        if batch_count > BATCH_COUNT_MAX {
            return Err(MultiBatchInvalid);
        }

        let trailer_size = trailer_total_size(element_size, batch_count) as usize;

        let trailer_items =
            &body[take_suffix(body.len(), &mut parsed, batch_count as usize * TRAILER_ITEM_SIZE)?];
        // The trailer size is a multiple of the element size.
        // Unused elements are filled with `maxInt` for padding.
        let items_padding_len = trailer_size - parsed;
        let trailer_items_padding = &body[take_suffix(body.len(), &mut parsed, items_padding_len)?];
        if !trailer_items_padding.iter().all(|&byte| byte == u8::MAX) {
            return Err(MultiBatchInvalid);
        }

        let events_count_total: u32 = {
            let mut count: u32 = 0;
            for item_index in 0..trailer_items.len() / TRAILER_ITEM_SIZE {
                let item = &trailer_items[item_index * TRAILER_ITEM_SIZE..][..TRAILER_ITEM_SIZE];
                let element_count = u32::from(read_u16(item));
                count = count.checked_add(element_count).ok_or(MultiBatchInvalid)?;
            }
            count
        };
        if element_size == 0 && events_count_total != 0 {
            return Err(MultiBatchInvalid);
        }
        let payload_size: usize =
            events_count_total.checked_mul(element_size).ok_or(MultiBatchInvalid)? as usize;

        // For byte-aligned elements, padding may be required between the payload and the
        // trailer.
        let trailer_padding_size = payload_size % TRAILER_ITEM_SIZE;
        assert!(trailer_padding_size < TRAILER_ITEM_SIZE);
        assert!(trailer_padding_size == 0 || element_size == 1);
        let trailer_padding = &body[take_suffix(body.len(), &mut parsed, trailer_padding_size)?];
        if !trailer_padding.iter().all(|&byte| byte == u8::MAX) {
            return Err(MultiBatchInvalid);
        }

        if payload_size != body.len() - parsed {
            return Err(MultiBatchInvalid);
        }
        assert_eq!(payload_size, body.len() - parsed);

        Ok(Self {
            payload: &body[..payload_size],
            trailer_items,
            payload_index: 0,
            batch_index: 0,
            element_size,
        })
    }

    pub fn reset(&mut self) {
        self.payload_index = 0;
        self.batch_index = 0;
    }

    /// # Panics
    /// Panics if the trailer holds more than [`BATCH_COUNT_MAX`] items (upstream asserts).
    #[must_use]
    pub fn batch_count(&self) -> u16 {
        assert!(self.trailer_items.len() / TRAILER_ITEM_SIZE <= BATCH_COUNT_MAX as usize);
        (self.trailer_items.len() / TRAILER_ITEM_SIZE) as u16
    }

    /// Returns the next batch, or `None` when exhausted.
    ///
    /// # Panics
    /// Panics if the trailer is empty (upstream asserts).
    pub fn pop(&mut self) -> Option<&'a [u8]> {
        assert!(!self.trailer_items.is_empty());

        if self.batch_index == self.trailer_items.len() / TRAILER_ITEM_SIZE {
            assert_eq!(self.payload_index, self.payload.len());
            return None;
        }
        let batch_item: &[u8] = self.peek();
        self.batch_index += 1;
        self.payload_index += batch_item.len();
        assert!(self.batch_index <= self.trailer_items.len() / TRAILER_ITEM_SIZE);
        assert!(self.payload_index <= self.payload.len());
        Some(batch_item)
    }

    /// # Panics
    /// Panics if the trailer is empty or all batches were popped (upstream asserts).
    #[must_use]
    pub fn peek(&self) -> &'a [u8] {
        assert!(!self.trailer_items.is_empty());
        assert!(self.batch_index < self.trailer_items.len() / TRAILER_ITEM_SIZE);
        assert!(self.payload_index <= self.payload.len());

        // Batch metadata is written from the end of the message, so the last
        // element corresponds to the first batch.
        let item_index = self.trailer_items.len() / TRAILER_ITEM_SIZE - self.batch_index - 1;
        let element_count =
            read_u16(&self.trailer_items[item_index * TRAILER_ITEM_SIZE..][..TRAILER_ITEM_SIZE]);
        if element_count == 0 {
            return &[];
        }
        assert!(self.payload_index < self.payload.len());

        let batch_size = usize::from(element_count) * self.element_size as usize;
        assert!(self.payload_index + batch_size <= self.payload.len());

        let slice: &[u8] = &self.payload[self.payload_index..][..batch_size];
        assert!(!slice.is_empty());
        assert!(slice.len().is_multiple_of(self.element_size as usize));
        slice
    }
}

/// Encodes batches into a multi-batch message body.
pub struct MultiBatchEncoder<'a> {
    buffer: Option<&'a mut [u8]>,
    batch_count: usize,
    buffer_index: usize,
    element_size: u32,
}

impl<'a> MultiBatchEncoder<'a> {
    /// # Panics
    /// Panics if the buffer cannot hold at least one batch (upstream asserts).
    pub fn new(buffer: &'a mut [u8], element_size: u32) -> Self {
        // Supports zero-sized elements, or any power of two, including 2^0.
        assert!(element_size == 0 || element_size.is_power_of_two());

        // The buffer must be large enough for at least one batch.
        let trailer_size_min = trailer_total_size(element_size, 1) as usize;
        assert!(buffer.len() >= trailer_size_min);

        // The end of the buffer must be aligned with the trailer.
        // If it isn't, reduce the buffer to maintain alignment.
        let aligned_len = buffer.len() - buffer.len() % TRAILER_ITEM_SIZE;

        Self {
            buffer: Some(&mut buffer[..aligned_len]),
            batch_count: 0,
            buffer_index: 0,
            element_size,
        }
    }

    /// Resets the encoder to reuse its buffer.
    ///
    /// # Panics
    /// Panics if the encoder was already finished (upstream asserts).
    pub fn reset(&mut self) {
        assert!(self.buffer.is_some());
        self.batch_count = 0;
        self.buffer_index = 0;
    }

    /// Batches added so far (`encoder.batch_count`, upstream public field).
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Bytes of payload written so far (`encoder.buffer_index`, upstream public field).
    #[must_use]
    pub fn buffer_index(&self) -> usize {
        self.buffer_index
    }

    /// Length of the aligned working buffer (`encoder.buffer.?.len`, upstream public field).
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        match &self.buffer {
            Some(buffer) => buffer.len(),
            None => 0,
        }
    }

    /// Returns a writable slice aligned and sized appropriately for the current operation.
    /// May return `None` if there isn't enough space in the buffer to add a new element
    /// to the trailer.
    /// The returned slice may have a length of zero if the remaining buffer
    /// isn't large enough to hold at least one element of the current operation.
    ///
    /// # Panics
    /// Panics if the encoder was already finished (upstream asserts).
    pub fn writable(&mut self) -> Option<&mut [u8]> {
        if self.batch_count == BATCH_COUNT_MAX as usize {
            return None;
        }
        assert!(self.batch_count < BATCH_COUNT_MAX as usize);

        assert!(self.element_size > 0 || self.buffer_index == 0);
        assert!(
            self.element_size == 0 || self.buffer_index.is_multiple_of(self.element_size as usize)
        );

        // Takes into account extra trailer bytes that will need to be included.
        let trailer_size =
            trailer_total_size(self.element_size, (self.batch_count + 1) as u16) as usize;

        let buffer: &mut [u8] = match self.buffer.as_deref_mut() {
            Some(buffer) => buffer,
            None => unreachable!("encoder used after finish()"),
        };
        if buffer.len() < self.buffer_index + trailer_size {
            // Insufficient space for one more batch.
            return None;
        }

        if self.element_size == 0 {
            // No writable buffer for zero-size elements, as they only add to the trailer.
            return Some(&mut []);
        }

        // Get an aligned slice.
        let slice_end = buffer.len() - trailer_size;
        let slice: &mut [u8] = &mut buffer[self.buffer_index..slice_end];
        let size = slice.len() - slice.len() % self.element_size as usize;
        Some(&mut slice[..size])
    }

    /// Records how many bytes were written in the slice previously acquired by
    /// [`writable`](Self::writable).
    ///
    /// # Panics
    /// Panics if `bytes_written` is not a whole number of elements, or if the encoder was
    /// already finished (upstream asserts).
    pub fn add(&mut self, bytes_written: usize) {
        assert!(self.batch_count < BATCH_COUNT_MAX as usize);

        let written_element_count: u16 = if self.element_size == 0 {
            assert_eq!(self.buffer_index, 0);
            assert_eq!(bytes_written, 0);
            0
        } else {
            assert!(
                bytes_written.is_multiple_of(self.element_size as usize),
                "partial element writes are invalid"
            );
            let written = bytes_written / self.element_size as usize;
            assert!(u16::try_from(written).is_ok(), "element count fits u16");
            written as u16
        };

        self.batch_count += 1;
        self.buffer_index += bytes_written;

        let buffer: &mut [u8] = match self.buffer.as_deref_mut() {
            Some(buffer) => buffer,
            None => unreachable!("encoder used after finish()"),
        };
        assert!(self.buffer_index < buffer.len());

        let trailer_size = trailer_total_size(self.element_size, self.batch_count as u16) as usize;
        assert!(self.buffer_index + trailer_size <= buffer.len());

        let trailer_items_end = buffer.len() - POSTAMBLE_SIZE;
        let trailer_items_start = buffer.len() - trailer_size;
        let trailer_items_len = (trailer_items_end - trailer_items_start) / TRAILER_ITEM_SIZE;
        assert!(trailer_items_len >= self.batch_count);

        // Batch metadata is stacked from the end of the message, so the first element
        // of the array corresponds to the last batch added.
        let item_range = {
            let item_index = trailer_items_len - self.batch_count;
            let start = trailer_items_start + item_index * TRAILER_ITEM_SIZE;
            start..start + TRAILER_ITEM_SIZE
        };
        buffer[item_range].copy_from_slice(&written_element_count.to_le_bytes());
    }

    /// Finalizes the batch by writing the trailer with proper encoding.
    /// Returns the total number of bytes written (payload + trailer).
    /// At least one batch must be inserted, and the encoder should not be used after
    /// being finished.
    ///
    /// # Panics
    /// Panics unless at least one batch was added, or if the encoder was already finished
    /// (upstream asserts).
    pub fn finish(&mut self) -> usize {
        assert!(self.batch_count > 0);
        assert!(self.batch_count <= BATCH_COUNT_MAX as usize);

        let buffer_opt = self.buffer.take();
        let buffer: &mut [u8] = match buffer_opt {
            Some(buffer) => buffer,
            None => unreachable!("encoder used after finish()"),
        };
        assert!(buffer.len() > self.buffer_index);
        assert!(self.element_size > 0 || self.buffer_index == 0);

        let trailer_size = trailer_total_size(self.element_size, self.batch_count as u16) as usize;

        // For byte-aligned elements, padding may be required between the payload and the
        // trailer.
        let padding = self.buffer_index % TRAILER_ITEM_SIZE;
        assert!(padding < TRAILER_ITEM_SIZE);
        assert!(padding == 0 || self.element_size == 1);
        assert!(buffer.len() >= self.buffer_index + padding + trailer_size);
        // Filling the padding with sentinels.
        buffer[self.buffer_index..][..padding].fill(u8::MAX);

        // While batches are being encoded, the trailer is written at the end of the buffer.
        // Once all batches are encoded, the trailer needs to be moved closer to the last
        // element written.
        let source_start = buffer.len() - trailer_size;
        let target_start = self.buffer_index + padding;
        debug_assert!(source_start >= target_start);
        buffer.copy_within(source_start.., target_start);

        let trailer_items = &mut buffer[target_start..][..trailer_size - POSTAMBLE_SIZE];
        // Filling in the extra alignment bytes with sentinels.
        let unused_items = trailer_items.len() / TRAILER_ITEM_SIZE - self.batch_count;
        trailer_items[..unused_items * TRAILER_ITEM_SIZE].fill_with(|| u8::MAX);

        let postamble_range =
            target_start + trailer_size - POSTAMBLE_SIZE..target_start + trailer_size;
        buffer[postamble_range].copy_from_slice(&(self.batch_count as u16).to_le_bytes());

        let bytes_written = self.buffer_index + padding + trailer_size;
        assert!(self.element_size > 0 || bytes_written == trailer_size);
        assert!(self.element_size == 0 || bytes_written.is_multiple_of(self.element_size as usize));

        #[allow(clippy::assertions_on_constants)] // mirrors upstream `if (constants.verify)`
        if tigerbeetle_core::constants::VERIFY {
            assert!(
                MultiBatchDecoder::new(&buffer[..bytes_written], self.element_size).is_ok(),
                "encoder output must roundtrip"
            );
        }

        bytes_written
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::{
        BATCH_COUNT_MAX, MultiBatchDecoder, MultiBatchEncoder, multi_batch_count_max, read_u16,
        trailer_total_size,
    };
    use tigerbeetle_core::constants::HEADER_SIZE;
    use tigerbeetle_core::stdx::prng::{Prng, ratio};

    // Upstream test buffers are allocated cache-line aligned because the Zig codec casts
    // pointers; the Rust codec is byte-wise, so plain buffers suffice.

    fn fill_pattern_u16(buffer: &mut [u8], value: u16) {
        for index in 0..buffer.len() / 2 {
            buffer[index * 2..][..2].copy_from_slice(&value.to_le_bytes());
        }
    }

    fn assert_all_u16_equal(bytes: &[u8], value: u16) {
        assert!(bytes.len().is_multiple_of(2));
        for index in 0..bytes.len() / 2 {
            assert_eq!(read_u16(&bytes[index * 2..]), value);
        }
    }

    /// Port of upstream `TestRunner.run`.
    fn run(
        prng: &mut Prng,
        element_size: u32,
        buffer: &mut [u8],
        batch_count: u16,
        batch_elements: Option<u16>,
    ) -> usize {
        let mut expected: Vec<u16> = Vec::new();

        let trailer_size = trailer_total_size(element_size, batch_count) as usize;

        // Cleaning the buffer first, so it can assert the bytes.
        prng.fill(buffer);

        let mut encoder = MultiBatchEncoder::new(buffer, element_size);
        for index in 0..batch_count {
            let bytes_available = encoder.buffer_len() - encoder.buffer_index() - trailer_size;

            let elements_count: u16 = if let Some(batch_elements) = batch_elements {
                batch_elements
            } else if index == batch_count - 1 && prng.chance(ratio(30, 100)) {
                u16::try_from(bytes_available / element_size as usize).expect("fits u16")
            } else if prng.chance(ratio(30, 100)) {
                0
            } else {
                u16::try_from(prng.int_inclusive_usize(bytes_available) / element_size as usize)
                    .expect("fits u16")
            };

            let slice = encoder.writable().expect("planned batch fits");
            let bytes_written = elements_count as usize * element_size as usize;
            assert!(slice.len() >= bytes_written);
            fill_pattern_u16(&mut slice[..bytes_written], index);
            encoder.add(bytes_written);

            expected.push(elements_count);
        }
        let bytes_written = encoder.finish();
        assert_eq!(encoder.batch_count(), batch_count as usize);

        let mut decoder =
            MultiBatchDecoder::new(&buffer[..bytes_written], element_size).expect("roundtrip");
        assert_eq!(decoder.batch_count(), batch_count);
        let mut batch_read_index: usize = 0;
        while let Some(batch) = decoder.pop() {
            let event_count = batch.len() / element_size as usize;
            assert_eq!(expected[batch_read_index] as usize, event_count);
            assert_all_u16_equal(batch, batch_read_index as u16);
            batch_read_index += 1;
        }
        assert_eq!(batch_count as usize, batch_read_index);

        bytes_written
    }

    // The maximum number of batches, all with zero elements.
    #[test]
    fn batch_maximum_batches_with_no_elements() {
        let mut prng = Prng::from_seed(42);

        let batch_count = BATCH_COUNT_MAX;
        let element_size = 128;
        let buffer_size = trailer_total_size(element_size, batch_count) as usize;

        let mut buffer = vec![0u8; buffer_size];

        let written_bytes = run(&mut prng, element_size, &mut buffer, batch_count, Some(0));
        assert_eq!(buffer_size, written_bytes);
    }

    // The maximum number of batches, when each one has one single element.
    #[test]
    fn batch_maximum_batches_with_a_single_element() {
        let mut prng = Prng::from_seed(42);

        let element_size = 128;
        let buffer_size = (1 << 20) - HEADER_SIZE; // 1MiB message.
        let batch_count_max: u16 = multi_batch_count_max(element_size, buffer_size as u32);

        let mut buffer = vec![0u8; buffer_size];

        let written_bytes = run(&mut prng, element_size, &mut buffer, batch_count_max, Some(1));

        let written_bytes_expected: usize =
            batch_count_max as usize * element_size as usize + batch_count_max as usize * 2 + 2;
        assert!(written_bytes_expected <= buffer_size);
        assert_eq!(written_bytes_expected, written_bytes);
    }

    // The maximum number of elements on a single batch.
    #[test]
    fn batch_maximum_elements_on_a_single_batch() {
        let mut prng = Prng::from_seed(42);

        let element_size = 128;
        let buffer_size = (1 << 20) - HEADER_SIZE; // 1MiB message.
        let batch_size_max = 8189; // maximum number of elements in a single-batch request.
        assert_eq!(batch_size_max, (buffer_size - element_size as usize) / element_size as usize);

        let mut buffer = vec![0u8; buffer_size];

        let written_bytes =
            run(&mut prng, element_size, &mut buffer, 1, Some(batch_size_max as u16));
        assert_eq!(buffer_size, written_bytes);
    }

    #[test]
    fn batch_invalid_format() {
        let mut prng = Prng::from_seed(42);

        let element_size = 128;
        let buffer_size = (1 << 20) - HEADER_SIZE; // 1MiB message.
        let mut buffer = vec![0u8; buffer_size];

        let batch_count = 10;
        let trailer_size = trailer_total_size(element_size, batch_count) as usize;

        let mut encoder = MultiBatchEncoder::new(&mut buffer, element_size);
        let mut event_total_count: usize = 0;
        for _ in 0..batch_count {
            let event_count: u16 = prng.int_inclusive_usize(100) as u16;
            let batch_size: usize = element_size as usize * event_count as usize;
            let writable = encoder.writable().expect("space reserved");
            assert!(writable.len() >= batch_size);
            encoder.add(batch_size);
            event_total_count += event_count as usize;
        }
        let bytes_written = encoder.finish();

        assert_eq!(batch_count, 10);
        assert_eq!(bytes_written, element_size as usize * event_total_count + trailer_size);

        assert!(MultiBatchDecoder::new(&buffer[..bytes_written], element_size).is_ok());

        assert!(matches!(
            MultiBatchDecoder::new(&buffer[..bytes_written - element_size as usize], element_size),
            Err(super::MultiBatchInvalid)
        ));
        assert!(matches!(
            MultiBatchDecoder::new(&buffer[element_size as usize..bytes_written], element_size),
            Err(super::MultiBatchInvalid)
        ));
        assert!(matches!(
            MultiBatchDecoder::new(&buffer[..bytes_written], element_size * 2),
            Err(super::MultiBatchInvalid)
        ));
        assert!(matches!(
            MultiBatchDecoder::new(&buffer[..bytes_written], element_size / 2),
            Err(super::MultiBatchInvalid)
        ));

        // Corrupt the postamble's batch count directly.
        let postamble_at = bytes_written - 2;
        buffer[postamble_at..bytes_written].copy_from_slice(&(batch_count + 1).to_le_bytes());
        assert!(matches!(
            MultiBatchDecoder::new(&buffer[..bytes_written], element_size),
            Err(super::MultiBatchInvalid)
        ));
        buffer[postamble_at..bytes_written].copy_from_slice(&(batch_count - 1).to_le_bytes());
        assert!(matches!(
            MultiBatchDecoder::new(&buffer[..bytes_written], element_size),
            Err(super::MultiBatchInvalid)
        ));
    }
}
