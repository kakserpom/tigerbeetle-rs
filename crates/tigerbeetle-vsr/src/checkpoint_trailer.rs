//! Checkpoint trailer layout primitives and the free-set/client-sessions checkpoint
//! trailers themselves.
//!
//! Port of `src/vsr/checkpoint_trailer.zig`.
//!
//! The pure layout definitions (`Chunk`, `block_count_for_trailer_size`, `TrailerType`)
//! mirror upstream 1:1. [`CheckpointTrailer`] is a plain data holder: upstream chains its
//! read/write state machine through callbacks into the Grid, which owns it; here
//! [`crate::grid::Grid`] owns two trailers (free-set blocks-acquired/blocks-released) and
//! drives their state machines from [`crate::grid::Grid::poll`] by correlating IO tokens.
//!
//! DEVIATION: upstream computes `chunk_size_max` from `@sizeOf(vsr.Header)`; we reuse
//! [`tigerbeetle_core::constants::HEADER_SIZE`] which is compile-time-pinned to the ported
//! header size.
//!
//! DEVIATION: upstream holds `BlockPtr`s (pointers into grid memory) plus derived
//! `block_bodies` slices; safe Rust forbids aliasing pointers, so blocks are identified by
//! their grid *location* and chunk bodies are re-derived from locations on demand
//! (`encode_chunks`/`decode_chunks` become Grid-side helpers).

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

/// Sentinel location for "no block held" (pre-`open` state; upstream leaves the
/// `BlockPtr`s `undefined` and gates use via `grid != null` instead).
pub(crate) const NO_LOCATION: u32 = u32::MAX;

/// Upstream `CheckpointTrailer.callback` tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Callback {
    #[default]
    None,
    Open,
    Checkpoint,
}

/// Persistent representation of the free set (or client sessions) between checkpoints
/// (upstream `CheckpointTrailerType`, stripped of its callback plumbing — see the module
/// docs). All fields mirror upstream's SoA layout.
#[derive(Debug)]
pub(crate) struct CheckpointTrailer {
    /// Upstream keeps `grid: ?*Grid` to gate use before `open`; this flag is our
    /// equivalent, set by [`crate::grid::Grid::open`] and only cleared on cancel/reset
    /// (not yet ported).
    pub attached: bool,
    pub trailer_type: TrailerType,

    /// Grid locations holding the trailer chunks; always `block_count_max` entries.
    pub locations: Vec<u32>,

    // SoA representation of block references holding the trailer itself. After the set is
    // read from disk and decoded, these blocks are manually marked as acquired.
    pub addresses: Vec<u64>,
    pub checksums: Vec<u128>,
    /// The block currently being read or written: counts down from `block_count()` to 0
    /// during open, up from 0 to `block_count()` during checkpoint.
    pub block_index: u32,

    /// Size of the encoded set in bytes (excludes block headers).
    pub size: u64,
    /// Trailer bytes read/written during disk IO; cross-checks that no bytes were lost.
    pub size_transferred: u64,
    /// Checksum covering the entire encoded trailer (excludes block headers).
    pub checksum: u128,

    pub callback: Callback,

    /// Outstanding grid IO tokens routed via poll
    /// (DEVIATION: upstream embeds callbacks in `Grid.Read`/`Grid.Write`).
    ///
    /// Open reads chain one at a time; checkpoint writes are all issued in a single
    /// synchronous pass (upstream loops `create_block()` inside `checkpoint()`), so
    /// writes carry their trailer block index alongside the token.
    pub outstanding_read: Option<u32>,
    pub outstanding_writes: Vec<(u32, u32)>,
}

impl CheckpointTrailer {
    /// Upstream `init`: allocate per-chunk arrays for a trailer of at most `buffer_size`
    /// encoded bytes.
    #[must_use]
    pub(crate) fn init(trailer_type: TrailerType, buffer_size: usize) -> Self {
        let block_count_max = block_count_for_trailer_size(buffer_size as u64) as usize;
        Self {
            attached: false,
            trailer_type,
            locations: vec![NO_LOCATION; block_count_max],
            addresses: vec![0; block_count_max],
            checksums: vec![0; block_count_max],
            block_index: 0,
            size: 0,
            size_transferred: 0,
            checksum: 0,
            callback: Callback::None,
            outstanding_read: None,
            outstanding_writes: Vec::new(),
        }
    }

    /// Upstream `block_count`.
    ///
    /// # Panics
    /// Panics unless the trailer is attached to a grid (upstream asserts `grid != null`).
    #[must_use]
    pub(crate) fn block_count(&self) -> u32 {
        assert!(self.attached);
        block_count_for_trailer_size(self.size)
    }

    /// Number of blocks the trailer's buffers can hold (upstream derives it from
    /// `blocks.len()`, which is sized for `encode_size_max` at init).
    #[must_use]
    pub(crate) fn block_count_max(&self) -> u32 {
        self.locations.len() as u32
    }

    /// Upstream `checkpoint_reference`: the superblock-header summary of this trailer.
    ///
    /// # Panics
    /// Panics if the trailer has unflushed IO or is mid-operation (upstream asserts the
    /// same), or if the recorded sizes/checksums disagree.
    #[must_use]
    pub(crate) fn checkpoint_reference(&self) -> crate::superblock::TrailerReference {
        assert_eq!(self.size, self.size_transferred);
        assert_eq!(self.callback, Callback::None);

        let reference = if self.size == 0 {
            crate::superblock::TrailerReference {
                checksum: tigerbeetle_core::checksum::checksum(&[]),
                last_block_address: 0,
                last_block_checksum: 0,
                trailer_size: 0,
            }
        } else {
            crate::superblock::TrailerReference {
                checksum: self.checksum,
                last_block_address: self.addresses[self.block_count() as usize - 1],
                last_block_checksum: self.checksums[self.block_count() as usize - 1],
                trailer_size: self.size,
            }
        };
        assert_eq!(reference.empty(), self.size == 0);
        reference
    }
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
