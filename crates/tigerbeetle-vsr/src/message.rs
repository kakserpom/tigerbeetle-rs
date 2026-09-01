//! A VSR protocol message and its fixed-size backing storage.
//!
//! Port of the `MessagePool.Message` buffer half of `src/message_pool.zig`.
//!
//! DEVIATION: upstream messages are reference-counted slots inside a preallocated
//! [`crate::message_pool`]-style pool (`ref()`/`unref()`/intrusive free list), so that a replica
//! can share one buffer across queues without copying and never allocates at runtime. Until the
//! replica exists to exercise sharing, this port carries an owned heap buffer instead:
//! ownership moves instead of reference counting, and pooling is deferred. The public surface
//! (`header::<T>()`, `body_used()`, frame read/write) is shaped to slot a pooled backing in
//! later without touching callers.

use tigerbeetle_core::constants;

use crate::message_header::{SIZE as HEADER_SIZE, TypedHeader};

/// Upstream `constants.message_size_max`, as a `usize` for slicing.
pub const MESSAGE_SIZE_MAX: usize = constants::MESSAGE_SIZE_MAX as usize;

#[derive(Debug)]
/// A single message: a `message_size_max`-byte buffer whose leading bytes are the 256-byte
/// header frame (upstream `Message.buffer`; upstream stores the header *inside* this buffer).
#[derive(Clone)]
pub struct Message {
    buffer: Box<[u8]>,
}

impl Message {
    /// An all-zero message (upstream hands out uninitialized memory; zeroing is the safe-Rust
    /// equivalent — headers are always fully overwritten before use).
    #[must_use]
    pub fn new() -> Self {
        Self { buffer: vec![0_u8; MESSAGE_SIZE_MAX].into_boxed_slice() }
    }

    /// The full backing storage.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    #[must_use]
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    /// The raw 256-byte header frame.
    ///
    /// # Panics
    /// Panics if the buffer is shorter than a header (cannot happen for constructed messages).
    #[must_use]
    pub fn frame(&self) -> &[u8] {
        &self.buffer[..HEADER_SIZE]
    }

    #[must_use]
    pub fn frame_mut(&mut self) -> &mut [u8] {
        &mut self.buffer[..HEADER_SIZE]
    }

    /// Decode the typed header for this message's command.
    /// Returns `None` if the frame does not parse or belongs to another command.
    #[must_use]
    pub fn header<T: TypedHeader>(&self) -> Option<T> {
        let frame: &[u8; HEADER_SIZE] = self.frame().try_into().ok()?;
        T::from_wire(frame)
    }

    /// Encode `header` into the frame prefix of this message.
    pub fn set_header<T: TypedHeader>(&mut self, header: &T) {
        let wire = header.to_wire();
        self.frame_mut().copy_from_slice(&wire);
    }

    /// The total message size from the raw size field (`@offsetOf(Header, "size")` == 96),
    /// readable even before the frame is known to be valid.
    #[must_use]
    pub fn size_raw(&self) -> u32 {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(&self.buffer[96..100]);
        u32::from_le_bytes(bytes)
    }

    /// Copy `bytes` into the body region (following the header frame).
    ///
    /// The caller is responsible for setting `header.size` and the body/header
    /// checksums (upstream: `set_checksum_body` + `set_checksum`).
    ///
    /// # Panics
    /// Panics if the body does not fit in the fixed-size buffer.
    pub fn set_body(&mut self, bytes: &[u8]) {
        let start = HEADER_SIZE;
        let end = start + bytes.len();
        assert!(end <= self.buffer.len());
        self.buffer_mut()[start..end].copy_from_slice(bytes);
    }

    /// The body covered by the size field (upstream `body_used`).
    ///
    /// # Panics
    /// Panics if the size field is smaller than a bare header.
    #[must_use]
    pub fn body_used(&self) -> &[u8] {
        let size = self.size_raw() as usize;
        assert!(size >= HEADER_SIZE);
        &self.buffer[HEADER_SIZE..size]
    }
}

impl Default for Message {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{MESSAGE_SIZE_MAX, Message};
    use crate::command::Command;
    use crate::message_header::{Ping, Prepare, SIZE, SIZE_U32, TypedHeader};

    #[test]
    fn new_message_is_zeroed_and_correctly_sized() {
        let message = Message::new();
        assert_eq!(message.buffer().len(), MESSAGE_SIZE_MAX);
        assert_eq!(message.frame().len(), SIZE);
        assert!(message.buffer().iter().all(|&byte| byte == 0));

        let default = Message::default();
        assert_eq!(default.size_raw(), 0);
    }

    /// `size_raw` must read the `Header.size` field at its wire offset 96, independently of
    /// command validity. Upstream: `@offsetOf(vsr.Header, "size") == 96`.
    #[test]
    fn size_raw_reads_size_field_at_offset_96() {
        let mut message = Message::new();

        let mut ping = Ping::default();
        ping.set_size(1000);
        message.set_header(&ping);
        assert_eq!(message.size_raw(), 1000);
        assert_eq!(message.buffer()[96..100], 1000u32.to_le_bytes());
        assert_eq!(message.frame()[96..100], 1000u32.to_le_bytes());
    }

    #[test]
    fn header_round_trip_and_command_rejection() {
        let mut message = Message::new();

        let mut ping = Ping::default();
        ping.set_size(SIZE_U32 + 3);
        ping.checkpoint_op = 42;
        ping.set_checksum_body(&[1, 2, 3]);
        ping.set_checksum();

        message.set_header(&ping);
        assert_eq!(message.header::<Ping>(), Some(ping));

        // A different command's header type must reject the frame.
        assert_eq!(message.header::<Prepare>(), None);

        // The base-frame view is reachable and consistent.
        let frame = message.header::<Ping>().unwrap().frame();
        assert_eq!(frame.command, Command::Ping);
        assert_eq!(frame.size, SIZE_U32 + 3);

        // Checksums computed through the typed header survive the buffer round trip.
        let decoded = message.header::<Ping>().unwrap();
        assert!(decoded.valid_checksum());
        assert!(decoded.valid_checksum_body(&[1, 2, 3]));
        assert!(!decoded.valid_checksum_body(&[1, 2, 4]));
    }

    #[test]
    fn body_used_matches_size_field() {
        let mut message = Message::new();

        let mut ping = Ping::default();
        ping.set_size(SIZE_U32 + 3);
        message.set_header(&ping);
        message.set_body(&[1, 2, 3]);
        assert_eq!(message.body_used(), &[1, 2, 3]);

        // An empty body: size == bare header.
        let mut ping = Ping::default();
        ping.set_size(SIZE_U32);
        message.set_header(&ping);
        message.set_body(&[]);
        assert_eq!(message.body_used(), &[] as &[u8]);
    }

    #[test]
    #[should_panic(expected = "size >= HEADER_SIZE")]
    fn body_used_panics_when_size_below_bare_header() {
        let mut message = Message::new();
        let mut ping = Ping::default();
        ping.set_size(SIZE_U32 - 1);
        message.set_header(&ping);
        let _ = message.body_used();
    }

    #[test]
    #[should_panic(expected = "end <= self.buffer.len()")]
    fn set_body_panics_on_overflow() {
        let mut message = Message::new();
        let too_big = vec![0_u8; MESSAGE_SIZE_MAX];
        message.set_body(&too_big);
    }
}
