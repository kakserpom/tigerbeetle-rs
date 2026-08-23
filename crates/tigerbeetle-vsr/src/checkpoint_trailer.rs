//! Checkpoint trailer layout primitives.
//!
//! Port of `src/vsr/checkpoint_trailer.zig` (layout-only subset).
//!
//! TODO(port): src/vsr/checkpoint_trailer.zig:57 — the grid-coupled `CheckpointTrailerType`
//! state machine (`open`/`checkpoint` read-write chains over `Grid`) requires the
//! grid/schema/storage ports first. This module carries the pure layout definitions so that
//! free-set sizing and future superblock code can share them.
//!
//! DEVIATION: upstream computes `chunk_size_max` from `@sizeOf(vsr.Header)`; we reuse
//! [`tigerbeetle_core::constants::HEADER_SIZE`] which is compile-time-pinned to the ported
//! header size.

// Upstream uses `@intCast` freely in this file; every cast below is bounded
// (chunks are < BLOCK_SIZE, block counts are < u32::MAX by grid construction).
#![allow(clippy::cast_possible_truncation)]

use tigerbeetle_core::constants;

/// Body of the block which holds encoded trailer data.
/// All chunks except for possibly the last one are full.
pub const CHUNK_SIZE_MAX: usize = constants::BLOCK_SIZE - constants::HEADER_SIZE;

/// Describes a slice of encoded trailer that goes into nth block on disk (upstream `Chunk`).
pub struct Chunk;

impl Chunk {
    /// Returns the size of the chunk stored in `block_index` of a `block_count`-block trailer.
    ///
    /// # Panics
    /// Panics if `block_count` is zero or inconsistent with `trailer_size`, or if
    /// `block_index >= block_count`.
    #[must_use]
    pub fn size(block_index: u32, block_count: u32, trailer_size: u64) -> u32 {
        assert!(block_count > 0);
        assert_eq!(u64::from(block_count), trailer_size.div_ceil(CHUNK_SIZE_MAX as u64));
        assert!(block_index < block_count);

        let last_block = block_index == block_count - 1;
        if last_block {
            let chunk = trailer_size - u64::from(block_count - 1) * CHUNK_SIZE_MAX as u64;
            assert!(u32::try_from(chunk).is_ok());
            chunk as u32
        } else {
            CHUNK_SIZE_MAX as u32
        }
    }
}

/// The number of blocks needed to store a trailer of the given encoded size.
#[must_use]
pub fn block_count_for_trailer_size(trailer_size: u64) -> u32 {
    trailer_size.div_ceil(CHUNK_SIZE_MAX as u64) as u32
}

/// Which kind of persistent state a trailer stores (upstream `TrailerType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrailerType {
    /// EWAH-encoded bitset of acquired blocks ([`crate::tigerbeetle_lsm_free_set_word_size`]).
    FreeSet,
    /// Reply headers + session numbers.
    ClientSessions,
}

impl TrailerType {
    /// Upstream `TrailerType.block_type`.
    #[must_use]
    pub const fn block_type(self) -> crate::message_header::BlockType {
        match self {
            Self::FreeSet => crate::message_header::BlockType::FreeSet,
            Self::ClientSessions => crate::message_header::BlockType::ClientSessions,
        }
    }

    /// Upstream `TrailerType.item_size`: every trailer's encoded length must be a whole number
    /// of items.
    ///
    /// DEVIATION: upstream derives these from `@sizeOf(FreeSet.Word)` and
    /// `@sizeOf(vsr.Header) + @sizeOf(u64)`; both are fixed by the wire format.
    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            // FreeSet.Word is u64:
            Self::FreeSet => core::mem::size_of::<u64>(),
            // Reply header + session number:
            Self::ClientSessions => constants::HEADER_SIZE + core::mem::size_of::<u64>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_count_boundaries() {
        assert_eq!(block_count_for_trailer_size(0), 0);
        assert_eq!(block_count_for_trailer_size(1), 1);
        assert_eq!(block_count_for_trailer_size(CHUNK_SIZE_MAX as u64), 1);
        assert_eq!(block_count_for_trailer_size(CHUNK_SIZE_MAX as u64 + 1), 2);
        assert_eq!(block_count_for_trailer_size(2 * CHUNK_SIZE_MAX as u64), 2);
    }

    #[test]
    fn chunk_size_layout() {
        // Single-block trailer: one partial chunk.
        assert_eq!(Chunk::size(0, 1, 1), 1);
        assert_eq!(Chunk::size(0, 1, CHUNK_SIZE_MAX as u64), CHUNK_SIZE_MAX as u32);

        // Two-block trailer: full chunk then remainder.
        assert_eq!(Chunk::size(0, 2, CHUNK_SIZE_MAX as u64 + 1), CHUNK_SIZE_MAX as u32);
        assert_eq!(Chunk::size(1, 2, CHUNK_SIZE_MAX as u64 + 1), 1);

        // Three-block trailer with a full middle block and a multi-byte tail.
        let trailer_size = 3 * CHUNK_SIZE_MAX as u64 / 2;
        let block_count = block_count_for_trailer_size(trailer_size);
        assert_eq!(block_count, 2);
        assert_eq!(Chunk::size(0, block_count, trailer_size), CHUNK_SIZE_MAX as u32);
        assert_eq!(
            u64::from(Chunk::size(1, block_count, trailer_size)),
            trailer_size - CHUNK_SIZE_MAX as u64
        );

        // Chunks tile the trailer exactly.
        for &block_count in &[1_u32, 2, 3, 7] {
            let trailer_size = u64::from(block_count) * (CHUNK_SIZE_MAX as u64) - 1;
            if trailer_size == 0 {
                continue;
            }
            let block_count = block_count_for_trailer_size(trailer_size);
            let total: u64 = (0..block_count)
                .map(|i| u64::from(Chunk::size(i, block_count, trailer_size)))
                .sum();
            assert_eq!(total, trailer_size);
        }
    }

    #[test]
    fn trailer_types() {
        use crate::message_header::BlockType;

        assert_eq!(TrailerType::FreeSet.block_type(), BlockType::FreeSet);
        assert_eq!(TrailerType::ClientSessions.block_type(), BlockType::ClientSessions);

        assert_eq!(TrailerType::FreeSet.item_size(), 8);
        assert_eq!(
            TrailerType::ClientSessions.item_size(),
            constants::HEADER_SIZE + core::mem::size_of::<u64>()
        );
    }
}
