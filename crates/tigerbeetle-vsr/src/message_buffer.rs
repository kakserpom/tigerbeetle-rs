//! MessageBuffer is the interface between a MessageBus and a Replica for passing batches of
//! messages while minimizing copies. It handles message framing, but doesn't do IO directly.
//!
//! It is a producer-consumer ring buffer of bytes, with two twists:
//! - consumer can skip over or "suspend" certain slices, to return to them later.
//! - producer validates bytes against a checksum, and this validation is sticky: a message is
//!   validated once, even if it is skipped many times.
//!
//! Invariant: suspend_size ≤ process_size ≤ advance_size ≤ receive_size
//!
//! Port of `src/message_buffer.zig`.
//!
//! DEVIATION: upstream's `consume_message()` can hand back the backing `Message` itself when the
//! receive window holds exactly one message (zero-copy fast path) because messages come from a
//! pool. This port always copies the consumed message out of the scratch buffer (see
//! [`crate::message`] DEVIATION); the framing state machine below is otherwise identical.

use crate::message::Message;
use crate::message_header::{Header, SIZE as HEADER_SIZE};
use tigerbeetle_core::constants;

/// `SIZE` (256) narrowed to the wire's u32 size field; the cast cannot truncate.
const HEADER_SIZE_U32: u32 = 256;

const _: () = assert!(crate::message_header::SIZE == 256);

pub struct MessageBuffer {
    /// The buffer passed to the kernel for reading into.
    message: Message,

    /// Suspended bytes, always a number of full messages.
    suspend_size: u32,
    /// Processed (consumed or suspended) bytes, always a number of full messages.
    process_size: u32,
    /// Bytes covered by a valid checksum, a number of full messages and maybe a header.
    advance_size: u32,
    /// The amount of bytes received from the kernel.
    receive_size: u32,

    /// An error occurred, and the MessageBus should terminate connection.
    /// Can be set by replica to indicate semantic errors, such as wrong cluster.
    invalid: Option<InvalidReason>,

    iterator_state: IteratorState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidReason {
    HeaderChecksum,
    HeaderSize,
    // Upstream also has HeaderCluster/Misdirected; those are set by the replica, not here:
    BodyChecksum,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IteratorState {
    Idle,
    AfterPeek,
    AfterConsumeSuspend,
}

impl MessageBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            message: Message::new(),
            suspend_size: 0,
            process_size: 0,
            advance_size: 0,
            receive_size: 0,
            invalid: None,
            iterator_state: IteratorState::Idle,
        }
    }

    fn invariants(&self) {
        assert!(self.suspend_size <= self.process_size);
        assert!(self.process_size <= self.advance_size);
        assert!(self.advance_size <= self.receive_size);
        if self.invalid.is_some() {
            assert_eq!(self.suspend_size, 0);
            assert_eq!(self.process_size, 0);
            assert_eq!(self.advance_size, 0);
            assert_eq!(self.receive_size, 0);
            assert_eq!(self.iterator_state, IteratorState::Idle);
        }
    }

    /// Pass this to the kernel to read into.
    ///
    /// # Panics
    /// Panics if the buffer was invalidated or an iteration is in progress.
    pub fn recv_slice(&mut self) -> &mut [u8] {
        assert!(self.receive_size < constants::MESSAGE_SIZE_MAX);
        assert_eq!(self.iterator_state, IteratorState::Idle);
        assert!(self.invalid.is_none());
        &mut self.message.buffer_mut()[self.receive_size as usize..]
    }

    /// When the kernel returns, informs the buffer about the read size.
    ///
    /// # Panics
    /// Panics on protocol violations (upstream asserts the same conditions).
    pub fn recv_advance(&mut self, size: u32) {
        assert_eq!(self.iterator_state, IteratorState::Idle);
        assert_eq!(self.process_size, 0);
        assert!(size > 0);
        assert!(size <= constants::MESSAGE_SIZE_MAX);

        self.receive_size += size;
        assert!(self.receive_size <= constants::MESSAGE_SIZE_MAX);
        self.advance();
    }

    /// # Panics
    /// Panics if the buffer is already invalid (upstream asserts the same).
    pub fn invalidate(&mut self, reason: InvalidReason) {
        assert!(self.invalid.is_none());
        self.suspend_size = 0;
        self.process_size = 0;
        self.advance_size = 0;
        self.receive_size = 0;
        self.iterator_state = IteratorState::Idle;
        self.invalid = Some(reason);
        self.invariants();
    }

    #[must_use]
    pub const fn invalid(&self) -> Option<InvalidReason> {
        match self.invalid {
            Some(reason) => Some(reason),
            None => None,
        }
    }

    /// Advances the parsing state machine.
    /// Idempotent, but eagerly called whenever receive_size or process_size change.
    fn advance(&mut self) {
        if self.invalid.is_none() {
            self.advance_header();
        }
        if self.invalid.is_none() {
            self.advance_body();
        }
        self.invariants();
    }

    fn advance_header(&mut self) {
        assert!(self.invalid.is_none());
        assert!(self.advance_size <= self.receive_size);
        if self.advance_size >= self.process_size + HEADER_SIZE_U32 {
            return; // Header is already known to be valid.
        }
        assert_eq!(self.advance_size, self.process_size);
        if self.receive_size - self.process_size < HEADER_SIZE_U32 {
            return; // Header not received yet.
        }

        let start = self.process_size as usize;
        let mut header_bytes = [0_u8; HEADER_SIZE];
        header_bytes.copy_from_slice(&self.message.buffer()[start..start + HEADER_SIZE]);

        // Checksum first, before the command byte is interpreted (upstream order):
        let mut checksum_bytes = [0_u8; 16];
        checksum_bytes.copy_from_slice(&header_bytes[0..16]);
        let stored_checksum = u128::from_le_bytes(checksum_bytes);
        if stored_checksum != crate::checksum(&header_bytes[16..]) {
            self.invalidate(InvalidReason::HeaderChecksum);
            return;
        }

        // Check that command is valid without materializing an invalid command value.
        // An unknown command byte means a peer speaking a different protocol version —
        // upstream calls vsr.fatal(.unknown_vsr_command), crashing for safety. Mirror that.
        let Some(header) = Header::from_wire(&header_bytes) else {
            let command_raw = header_bytes[114];
            let protocol = u16::from_le_bytes([header_bytes[112], header_bytes[113]]);
            let mut release_bytes = [0_u8; 4];
            release_bytes.copy_from_slice(&header_bytes[108..112]);
            let release = u32::from_le_bytes(release_bytes);
            panic!(
                "unknown VSR command, crashing for safety \
                 (command={command_raw} protocol={protocol} replica={} release={release})",
                header_bytes[115],
            );
        };

        if header.size < HEADER_SIZE_U32 || header.size > constants::MESSAGE_SIZE_MAX {
            self.invalidate(InvalidReason::HeaderSize);
            return;
        }
        assert!(HEADER_SIZE_U32 <= header.size && header.size <= constants::MESSAGE_SIZE_MAX);

        self.advance_size += HEADER_SIZE_U32;
    }

    fn advance_body(&mut self) {
        assert!(self.invalid.is_none());
        if self.advance_size < self.process_size + HEADER_SIZE_U32 {
            return; // Header not received yet.
        }

        let header = self.copy_header();

        if self.receive_size - self.process_size < header.size {
            return; // Body not received yet.
        }

        if self.advance_size >= self.process_size + header.size {
            return; // Body is already known to be valid.
        }

        assert_eq!(self.advance_size - self.process_size, HEADER_SIZE_U32);
        let start = self.process_size as usize + HEADER_SIZE;
        let end = self.process_size as usize + header.size as usize;
        let body = &self.message.buffer()[start..end];
        if !header.valid_checksum_body(body) {
            self.invalidate(InvalidReason::BodyChecksum);
            return;
        }
        self.advance_size += header.size - HEADER_SIZE_U32;
    }

    /// Peek at the header for the incoming message. Necessitates a copy to guarantee alignment.
    ///
    /// # Panics
    /// Panics if fewer than a header has been received, or if the command byte is unknown
    /// (the latter is unreachable here because `advance_header` validated it first).
    fn copy_header(&self) -> Header {
        assert!(self.receive_size - self.process_size >= HEADER_SIZE_U32);
        let start = self.process_size as usize;
        let mut header_bytes = [0_u8; HEADER_SIZE];
        header_bytes.copy_from_slice(&self.message.buffer()[start..start + HEADER_SIZE]);
        let Some(header) = Header::from_wire(&header_bytes) else {
            panic!("command validated by advance_header");
        };
        header
    }

    #[must_use]
    pub fn has_message(&self) -> bool {
        let valid_unprocessed = self.advance_size - self.process_size;
        if valid_unprocessed >= HEADER_SIZE_U32 {
            let header = self.copy_header();
            if valid_unprocessed >= header.size {
                return true;
            }
        }
        false
    }

    /// MessageBuffer is also an iterator which must be driven to completion.
    /// A call to next_header must be immediately followed by a call to consume_message
    /// or suspend_message.
    ///
    /// # Panics
    /// Panics if an iteration is still in progress (`AfterPeek`).
    pub fn next_header(&mut self) -> Option<Header> {
        match self.iterator_state {
            IteratorState::Idle | IteratorState::AfterConsumeSuspend => {}
            IteratorState::AfterPeek => unreachable!(),
        }

        let valid_unprocessed = self.advance_size - self.process_size;
        if valid_unprocessed >= HEADER_SIZE_U32 {
            assert!(self.invalid.is_none());
            let header = self.copy_header();
            if valid_unprocessed >= header.size {
                self.iterator_state = IteratorState::AfterPeek;
                return Some(header);
            }
        }

        // Move from this:
        // |  bytes  |     hole     |     bytes     |    hole    |
        //           ^suspend_size  ^process_size   ^receive_size
        //
        // To this:
        // |           bytes             |         hole          |
        // ^ suspend_size,process_size   ^ receive_size
        assert!(self.suspend_size <= self.process_size);
        assert!(self.process_size <= self.receive_size);

        if self.suspend_size < self.process_size {
            let buffer = self.message.buffer_mut();
            buffer.copy_within(
                self.process_size as usize..self.receive_size as usize,
                self.suspend_size as usize,
            );
        }
        self.receive_size -= self.process_size - self.suspend_size;
        self.advance_size -= self.process_size - self.suspend_size;
        self.suspend_size = 0;
        self.process_size = 0;
        self.iterator_state = IteratorState::Idle;

        // The purpose of tracking advance_size across iterations is to "cache" checksum
        // validation. As a sanity check, assert that advance-after-back-shift is indeed a no-op.
        let advance_size_idempotent = self.advance_size;
        self.advance();
        assert_eq!(self.advance_size, advance_size_idempotent);

        None
    }

    /// Copy the current message out of the receive buffer.
    ///
    /// # Panics
    /// Panics unless immediately preceded by [`Self::next_header()`] returning `Some`.
    pub fn consume_message(&mut self, header: &Header) -> Message {
        assert_eq!(self.iterator_state, IteratorState::AfterPeek);
        assert!(self.advance_size - self.process_size >= header.size);
        assert!(self.invalid.is_none());

        let mut message = Message::new();
        let start = self.process_size as usize;
        let size = header.size as usize;
        message.buffer_mut()[..size].copy_from_slice(&self.message.buffer()[start..start + size]);
        self.process_size += header.size;
        assert!(self.process_size <= self.receive_size);
        self.advance();

        self.iterator_state = IteratorState::AfterConsumeSuspend;
        let frame: &[u8; HEADER_SIZE] =
            message.buffer()[..HEADER_SIZE].try_into().unwrap_or(&[0_u8; HEADER_SIZE]);
        assert_eq!(Header::from_wire(frame).map(|h| h.checksum), Some(header.checksum));
        message
    }

    /// Keep the current message in the buffer, skipping over it for now.
    ///
    /// # Panics
    /// Panics unless immediately preceded by [`Self::next_header()`] returning `Some`.
    pub fn suspend_message(&mut self, header: &Header) {
        assert_eq!(self.iterator_state, IteratorState::AfterPeek);
        assert!(self.advance_size - self.process_size >= header.size);
        assert!(self.invalid.is_none());
        assert!(header.size <= constants::MESSAGE_SIZE_MAX);
        assert!(self.suspend_size <= self.process_size);

        if self.suspend_size < self.process_size {
            // Move from this:
            // |  bytes  |    hole     |    message    |     bytes    |
            //           ^suspend_size ^process_size                  ^receive_size
            //
            // To this:
            // | bytes |    message    |     hole      |     bytes    |
            //                         ^suspend_size   ^process_size  ^receive_size
            let buffer = self.message.buffer_mut();
            let dst = self.suspend_size as usize;
            let src_start = self.process_size as usize;
            let src_end = src_start + header.size as usize;
            buffer.copy_within(src_start..src_end, dst);
        }

        self.suspend_size += header.size;
        self.process_size += header.size;
        self.advance();

        self.iterator_state = IteratorState::AfterConsumeSuspend;
    }
}

impl Default for MessageBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message_header::{Prepare, TypedHeader};
    use crate::multiversion::Release;
    use tigerbeetle_core::constants::MESSAGE_SIZE_MAX;
    use tigerbeetle_core::stdx::prng::Prng;

    use super::HEADER_SIZE_U32;

    const MESSAGE_SIZE_MAX_USIZE: usize = MESSAGE_SIZE_MAX as usize;

    /// Port of upstream test "MessageBuffer fuzz".
    ///
    /// Generate a byte buffer with a bunch of prepares side-by-side.
    /// Optionally corrupt a single bit in the buffer.
    /// Feed the buffer in chunks of varying length to the MessageBuffer, verify that all messages
    /// are received unless a fault is detected.
    ///
    /// DEVIATION: scaled down (8 iterations, ≤12 messages, sizes biased toward the header-only
    /// minimum) because this port's checksum is software AES — bit-for-bit identical to upstream
    /// but much slower than vaesenc in unoptimized test builds. The state-machine paths exercised
    /// are the same.
    #[test]
    // Mirrors the upstream fuzz test's single long test function:
    #[allow(clippy::too_many_lines)]
    fn message_buffer_fuzz() {
        const MESSAGES_MAX: usize = 12;
        // Upstream chances: min=10 max=10 random=80. Bias toward small messages for speed:
        const CHANCE_MIN: usize = 60;
        const CHANCE_MAX: usize = 65; // (min..max) window is [60, 65)

        let mut prng = Prng::from_seed(0);

        let mut buffer = vec![0_u8; 3 * MESSAGE_SIZE_MAX_USIZE];

        for _ in 0..8 {
            let fault = prng.boolean();
            let mut total_size: usize = 0;
            let mut headers: Vec<Header> = Vec::new();
            for _ in 0..MESSAGES_MAX {
                let roll = prng.range_inclusive_usize(0, 99);
                let message_size: u32 = if roll < CHANCE_MIN {
                    HEADER_SIZE_U32
                } else if roll < CHANCE_MAX {
                    MESSAGE_SIZE_MAX
                } else {
                    let size = prng.range_inclusive_usize(HEADER_SIZE, MESSAGE_SIZE_MAX_USIZE - 1);
                    u32::try_from(size).unwrap_or(MESSAGE_SIZE_MAX)
                };

                if total_size + message_size as usize > buffer.len() {
                    break;
                }

                let mut header = Prepare {
                    cluster: 1,
                    view: 1,
                    release: Release::MINIMUM,
                    parent: prng.int_u128(),
                    request_checksum: prng.int_u128(),
                    checkpoint_id: prng.int_u128(),
                    client: 1,
                    commit: 10,
                    timestamp: 999,
                    request: 1,
                    op: 1,
                    size: message_size,
                    ..Prepare::default()
                };
                let body_start = total_size + HEADER_SIZE;
                let body_end = total_size + message_size as usize;
                let body = &mut buffer[body_start..body_end];
                prng.fill(body);
                header.set_checksum_body(body);
                header.set_checksum();
                buffer[total_size..body_start].copy_from_slice(&header.to_wire());
                total_size += message_size as usize;

                let frame = header.to_wire();
                let Some(base) = Header::from_wire(&frame) else {
                    panic!("constructed prepare must parse");
                };
                headers.push(base);
            }

            if fault {
                let byte_index = prng.index(total_size);
                let bit_index = prng.int_inclusive_usize(7);
                buffer[byte_index] ^= 1_u8 << bit_index;
            }

            let mut message_buffer = MessageBuffer::new();

            let mut recv_size: usize = 0;
            while !headers.is_empty() {
                if message_buffer.receive_size < constants::MESSAGE_SIZE_MAX
                    && recv_size < total_size
                {
                    let recv_slice = message_buffer.recv_slice();
                    let chunk_limit = std::cmp::min(recv_slice.len(), total_size - recv_size);
                    let chunk_size = prng.range_inclusive_usize(1, chunk_limit);
                    recv_slice[..chunk_size]
                        .copy_from_slice(&buffer[recv_size..recv_size + chunk_size]);
                    let chunk_size_u32 = u32::try_from(chunk_size).unwrap_or(u32::MAX);
                    message_buffer.recv_advance(chunk_size_u32);
                    recv_size += chunk_size;
                }

                let mut header_index: usize = 0;
                while let Some(header) = message_buffer.next_header() {
                    message_buffer.invariants();
                    if prng.boolean() {
                        let message = message_buffer.consume_message(&header);
                        let mut frame = [0_u8; HEADER_SIZE];
                        frame.copy_from_slice(&message.buffer()[..HEADER_SIZE]);
                        assert_eq!(
                            Header::from_wire(&frame).as_ref(),
                            Some(&headers[header_index])
                        );
                        headers.remove(header_index);
                    } else {
                        message_buffer.suspend_message(&header);
                        header_index += 1;
                    }
                }
                assert_eq!(message_buffer.iterator_state, IteratorState::Idle);
                if let Some(_reason) = message_buffer.invalid() {
                    assert!(fault, "invalid without faults");
                    break;
                }
            }
            if fault {
                assert!(message_buffer.invalid().is_some());
            } else {
                assert!(message_buffer.invalid().is_none());
                assert!(headers.is_empty());
            }
        }
    }

    /// A header whose command byte is outside the `Command` enum, but whose frame checksum is
    /// valid (as if sent by a peer speaking a newer protocol), must crash-for-safety,
    /// mirroring upstream's `vsr.fatal(.unknown_vsr_command)`.
    #[test]
    #[should_panic(expected = "unknown VSR command, crashing for safety")]
    fn unknown_command_crashes_for_safety() {
        let mut message_buffer = MessageBuffer::new();

        let mut header = Prepare::default();
        header.set_checksum();
        let mut frame = header.to_wire();
        // Forge an unknown command byte, then repair the checksum so that the frame passes
        // checksum validation and reaches the command interpretation:
        frame[114] = 0xEE;
        let repaired = crate::checksum(&frame[16..]);
        frame[0..16].copy_from_slice(&repaired.to_le_bytes());

        let recv_slice = message_buffer.recv_slice();
        recv_slice[..frame.len()].copy_from_slice(&frame);
        message_buffer.recv_advance(HEADER_SIZE_U32);
        unreachable!("recv_advance must panic on the unknown command");
    }
}
