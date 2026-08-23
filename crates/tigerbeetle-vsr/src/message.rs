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
