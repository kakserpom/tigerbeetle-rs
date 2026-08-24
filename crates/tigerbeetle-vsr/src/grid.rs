//! The Grid provides access to on-disk blocks (blobs of [`BLOCK_SIZE`] bytes).
//! Each block is identified by an "address" (`u64`, beginning at 1).
//!
//! Port of `src/vsr/grid.zig` — slice 1 covers the in-memory block-management layer
//! (block storage, reference counting, stash, block cache, offset math and read
//! validation). The remaining upstream surface lands with the IO/lifecycle slices:
//!
//! TODO(port): `src/vsr/grid.zig` — `open`/`checkpoint`/`mark_checkpoint_not_durable`/
//! `checkpoint_durable`/`cancel` (need the full SuperBlock + CheckpointTrailer),
//! `read_block*`/`write_block`/`create_block`/`repair_block` (IO queues + events),
//! `read_block_from_write_queues`/`read_block_from_cache`,
//! `fulfill_block`/`blocks_missing` repair flows, `reserve`/`forfeit`/`acquire`/
//! `release`/`writing`, `verify_table`/`assert_coherent`/`madv_dont_dump`.
//!
//! DEVIATION: upstream hands out `*align(sector_size) [block_size]u8` pointers into a
//! single allocation; safe Rust forbids aliasing pointers, so blocks are identified by
//! their index ("location") within [`Grid::blocks`] instead. Pointer-identity assertions
//! become location comparisons with identical semantics.

#![allow(clippy::cast_possible_truncation)] // block counts and sizes fit u32 like upstream

use std::collections::HashSet;

use tigerbeetle_core::constants::{BLOCK_SIZE, SECTOR_SIZE};
use tigerbeetle_lsm::set_associative_cache::{
    Layout, SetAssociativeCache, SetAssociativeCacheSpec,
};

use crate::command::Command;
use crate::message_header::{self, TypedHeader};
use crate::schema;

/// Upstream `set_associative_cache_ways = 16` and its cache-line override.
struct GridCacheAddress;

impl Layout for GridCacheAddress {
    // Upstream: `.cache_line_size = 16` — allows running with a much smaller grid cache
    // (256MiB vs 1GiB) instead of being completely optimal.
    const CACHE_LINE_SIZE: u64 = 16;
}

impl SetAssociativeCacheSpec for GridCacheAddress {
    type Key = u64;
    type Value = u64;

    fn key_from_value(value: &u64) -> u64 {
        *value
    }

    fn hash(key: u64) -> u64 {
        assert!(key > 0);
        tigerbeetle_core::stdx::hash::hash_inline_u64(key)
    }
}

type Cache = SetAssociativeCache<GridCacheAddress>;

/// Byte offset of a grid address within the `Grid` zone.
///
/// # Panics
/// Asserts `address > 0` (addresses begin at 1).
#[must_use]
pub fn block_offset(address: u64) -> u64 {
    assert!(address > 0);
    (address - 1) * BLOCK_SIZE as u64
}

/// Smallest sector-multiple that fits `size` bytes.
fn sector_ceil(size: usize) -> usize {
    size.div_ceil(SECTOR_SIZE) * SECTOR_SIZE
}

/// Although we distinguish between the reasons why the block is invalid, upstream only uses
/// this info for logging, not logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadBlockResult {
    /// The block is valid.
    Valid,
    /// Checksum of block header is invalid.
    InvalidChecksum,
    /// Checksum of block body is invalid.
    InvalidChecksumBody,
    /// The block header is valid, but its `header.command` is not `block`
    /// (this is possible due to misdirected IO).
    UnexpectedCommand,
    /// The block is valid, but it is not the block we expected.
    UnexpectedChecksum,
    /// The block is valid, and it is the expected block, but the last sector's padding is
    /// corrupt, so it will be repaired just to be safe.
    InvalidPadding,
}

/// Construction options for [`Grid`] (upstream's anonymous init-options struct).
#[derive(Clone, Copy, Debug)]
pub struct GridOptions {
    pub cache_blocks_count: usize,
    pub stash_blocks_count: usize,
    pub read_iops_max: usize,
}

/// The `(address, checksum)` pair a read expects to find
/// (upstream anonymous `expect` parameter of `read_block_validate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpectBlock {
    pub address: u64,
    pub checksum: u128,
}

/// Validates a freshly-read block against the expected address/checksum
/// (upstream `read_block_validate`; the `Valid` variant carries no pointer here —
/// the caller keeps the buffer it passed in).
///
/// # Panics
/// Asserts internal invariants of a checksum-valid block header (size bounds), mirroring
/// upstream's asserts.
#[must_use]
pub fn read_block_validate(block: &[u8], expect: ExpectBlock) -> ReadBlockResult {
    let frame: &[u8; message_header::SIZE] = block[..message_header::SIZE]
        .try_into()
        .unwrap_or_else(|_| unreachable!("grid blocks are at least HEADER_SIZE bytes"));
    let Some(base) = message_header::Header::from_wire(frame) else {
        unreachable!("generic header frames always parse");
    };

    if !base.valid_checksum() {
        return ReadBlockResult::InvalidChecksum;
    }

    // Upstream reinterprets the first bytes as `vsr.Header.Block` without checking the
    // command byte; only after the checksum does it compare the command explicitly:
    if base.command != Command::Block {
        return ReadBlockResult::UnexpectedCommand;
    }

    let Some(header) = message_header::Block::from_wire(frame) else {
        unreachable!("command was verified as Block");
    };

    assert!(header.size >= message_header::SIZE as u32);
    assert!(header.size <= BLOCK_SIZE as u32);

    if !base.valid_checksum_body(&block[message_header::SIZE..header.size as usize]) {
        return ReadBlockResult::InvalidChecksumBody;
    }

    if header.checksum != expect.checksum {
        return ReadBlockResult::UnexpectedChecksum;
    }

    if !tigerbeetle_core::stdx::zeroed(
        &block[header.size as usize..sector_ceil(header.size as usize)],
    ) {
        return ReadBlockResult::InvalidPadding;
    }

    assert_eq!(header.address, expect.address);
    ReadBlockResult::Valid
}

/// A stack of stash locations with O(1) membership. Order mirrors upstream's
/// `AutoArrayHashMapUnmanaged`: `pop()` removes the most recently pushed entry and
/// `remove()` swaps the last element into the hole (`swapRemove`).
#[derive(Debug, Default)]
struct StashStack {
    order: Vec<u32>,
    members: HashSet<u32>,
}

impl StashStack {
    fn push(&mut self, location: u32) {
        assert!(self.members.insert(location), "location {location} already stashed");
        self.order.push(location);
    }

    fn pop(&mut self) -> Option<u32> {
        let location = self.order.pop()?;
        assert!(self.members.remove(&location));
        Some(location)
    }

    fn remove(&mut self, location: u32) -> bool {
        if self.members.remove(&location) {
            let index = self
                .order
                .iter()
                .position(|&candidate| candidate == location)
                .unwrap_or_else(|| unreachable!("membership set and order are in sync"));
            self.order.swap_remove(index);
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.order.len()
    }
}

/// In-memory half of the Grid: block buffers, references, stash and cache
/// (upstream `GridType` fields minus IO queues/superblock hooks).
pub struct Grid {
    /// Block contents. Entries correspond to `blocks_references` entries.
    blocks: Vec<Vec<u8>>,
    /// Per-block outstanding-reference counts (saturates at `u8::MAX` like upstream).
    blocks_references: Vec<u8>,

    cache: Cache,
    /// The block cached at cache way-set slot `i` lives in `blocks[cache_locations[i]]`.
    ///
    /// Invariants:
    /// - `cache_locations[i] < blocks.len()`
    /// - `cache_locations[i] != cache_locations[j] iff i != j`
    cache_locations: Vec<u32>,

    /// Free stash locations (no outstanding references), LIFO.
    stash_free: StashStack,
    /// Stash locations handed out via [`Grid::get_block`]/[`Grid::block_ref`].
    stash_used: StashStack,

    /// How many more stash references may be taken.
    ///
    /// Invariants: `stash_available <= stash_blocks_count`,
    /// `stash_available <= stash_free.len()`.
    stash_available: u32,

    /// Blocks reserved for read IOPs (upstream `read_iop_blocks`); consumed by the IO
    /// slice (TODO(port): src/vsr/grid.zig `read_block_with`).
    #[allow(dead_code)] // reserved at init like upstream; used once read paths land
    read_iop_blocks: Vec<Option<u32>>,
}

impl Grid {
    /// # Panics
    /// Asserts `stash_blocks_count > 0` and capacity arithmetic, mirroring upstream init.
    #[must_use]
    pub fn new(options: GridOptions) -> Self {
        assert!(options.stash_blocks_count > 0);

        let blocks_count = options.cache_blocks_count + options.stash_blocks_count;
        let mut blocks = Vec::with_capacity(blocks_count);
        for _ in 0..blocks_count {
            blocks.push(vec![0u8; BLOCK_SIZE]);
        }

        let mut blocks_references = vec![0u8; blocks_count];
        let mut cache_locations = Vec::with_capacity(options.cache_blocks_count);

        let mut stash_free = StashStack::default();
        let mut stash_used = StashStack::default();

        for location in 0..u32::try_from(blocks_count).unwrap_or_else(|_| {
            unreachable!("block count fits u32 like upstream's location indices")
        }) {
            let location_usize = location as usize;
            if location_usize < options.cache_blocks_count {
                cache_locations.push(location);
            } else {
                stash_free.push(location);
            }
        }
        assert_eq!(stash_free.len(), options.stash_blocks_count);

        // Reserve one stash block per potential read IOP (upstream init loop):
        let mut read_iop_blocks = Vec::new();
        for _ in 0..options.read_iops_max {
            let location =
                stash_free.pop().unwrap_or_else(|| unreachable!("stash sized for reads"));
            blocks_references[location as usize] += 1;
            stash_used.push(location);
            read_iop_blocks.push(Some(location));
        }

        let grid = Self {
            blocks,
            blocks_references,
            cache: Cache::new(options.cache_blocks_count as u64, "grid"),
            cache_locations,
            stash_free,
            stash_used,
            stash_available: u32::try_from(options.stash_blocks_count)
                .unwrap_or_else(|_| unreachable!("stash count fits u32"))
                - u32::try_from(options.read_iops_max)
                    .unwrap_or_else(|_| unreachable!("read iops fit u32")),
            read_iop_blocks,
        };
        grid.assert_invariants();
        grid
    }

    fn assert_invariants(&self) {
        assert!(self.stash_available as usize <= self.stash_free.len());
        let mut seen = HashSet::new();
        for &location in &self.cache_locations {
            assert!(location < self.blocks.len() as u32);
            assert!(seen.insert(location));
        }
    }

    /// Return a block from the stash which had no outstanding references.
    ///
    /// # Panics
    /// Panics when the stash has no free blocks (upstream `@panic`).
    pub fn get_block(&mut self) -> u32 {
        let location = self.stash_free.pop().unwrap_or_else(|| panic!("stash has no free blocks"));

        assert_eq!(self.blocks_references[location as usize], 0);
        self.blocks_references[location as usize] += 1;
        self.stash_available -= 1;
        self.stash_used.push(location);

        location
    }

    /// Take an additional reference to a block (upstream `block_ref`; upstream also
    /// validates the block's header checksum, which we keep).
    ///
    /// # Panics
    /// Asserts the block holds a valid header and the location is not in `stash_free`.
    pub fn block_ref(&mut self, location: u32) {
        let header = schema::header_from_block(&self.blocks[location as usize]);
        assert!(header.valid_checksum());

        assert!(!self.stash_free.members.contains(&location));

        if self.blocks_references[location as usize] == 0 {
            // The only way to reference a zero-reference block is if we got it from the
            // cache, not the stash.
            assert!(!self.stash_used.members.contains(&location));
        }

        self.blocks_references[location as usize] += 1;
        self.stash_available -= 1;

        if !self.stash_used.members.contains(&location) {
            self.stash_used.push(location);
        }
    }

    /// Release a reference to a block.
    ///
    /// # Panics
    /// Asserts the reference count is positive.
    pub fn block_unref(&mut self, location: u32) {
        assert!(!self.stash_free.members.contains(&location));

        assert!(self.blocks_references[location as usize] > 0);
        self.blocks_references[location as usize] -= 1;
        self.stash_available += 1;

        if self.blocks_references[location as usize] == 0 && self.stash_used.remove(location) {
            self.stash_free.push(location);
        }

        assert!(self.stash_available as usize <= self.stash_free.len());
    }

    /// Outstanding references to the given block.
    #[must_use]
    pub fn block_references(&self, location: u32) -> u8 {
        self.blocks_references[location as usize]
    }

    /// Insert the address into the cache, and swap the evicted block into the stash
    /// (upstream `cache_upsert`).
    ///
    /// # Panics
    /// Asserts the saved block is referenced, carries `address` in its header, and is
    /// currently a used stash block.
    pub fn cache_upsert(&mut self, block_address: u64, block_save_location: u32) {
        assert!(block_address != 0);

        // The location/block that is being moved from stash to cache:
        assert!(self.blocks_references[block_save_location as usize] > 0);

        {
            let save_header = schema::header_from_block(&self.blocks[block_save_location as usize]);
            assert_eq!(save_header.address, block_address);
        }

        let upserted = self.cache.upsert(&block_address);
        let cache_index = upserted.index;
        assert!(cache_index < self.cache_locations.len());

        // The location/block being moved from cache to stash:
        let block_drop_location = self.cache_locations[cache_index];
        assert_ne!(block_drop_location, block_save_location);

        let block_drop_removed = self.stash_used.remove(block_save_location);
        assert!(block_drop_removed);

        if self.blocks_references[block_drop_location as usize] == 0 {
            self.stash_free.push(block_drop_location);
        } else {
            self.stash_used.push(block_drop_location);
        }
        self.cache_locations[cache_index] = block_save_location;

        if self.blocks_references[block_drop_location as usize] == 0 {
            // This block content won't be used again. We could overwrite the entire thing,
            // but that would be more expensive.
            self.blocks[block_drop_location as usize][..message_header::SIZE].fill(0);
        }
    }

    /// The block currently cached for `address`, if any (its location).
    #[must_use]
    pub fn cached_location(&mut self, address: u64) -> Option<u32> {
        let cache_index = self.cache.get_index(address)?;
        Some(self.cache_locations[cache_index])
    }

    /// Number of cache slots backing this grid (test/introspection helper).
    #[must_use]
    pub fn cache_slots(&self) -> usize {
        self.cache_locations.len()
    }

    /// Total number of blocks (cache + stash) held by this grid.
    #[must_use]
    pub fn blocks_count(&self) -> usize {
        self.blocks.len()
    }

    /// Contents of a block (read-only view for tests and upcoming slices).
    #[must_use]
    pub fn block(&self, location: u32) -> &[u8] {
        &self.blocks[location as usize]
    }

    /// Mutable contents of a block.
    pub fn block_mut(&mut self, location: u32) -> &mut [u8] {
        &mut self.blocks[location as usize]
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default // typed headers keep reserved fields private
    )]

    use super::{
        BLOCK_SIZE, ExpectBlock, Grid, GridOptions, ReadBlockResult, block_offset,
        read_block_validate,
    };
    use crate::message_header::{self, TypedHeader};
    use crate::multiversion::Release;
    use tigerbeetle_core::constants::SECTOR_SIZE;

    const CACHE_BLOCKS_COUNT: usize = 64;
    const STASH_BLOCKS_COUNT: usize = 12;
    const READ_IOPS_MAX: usize = 2;

    fn new_grid() -> Grid {
        Grid::new(GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: STASH_BLOCKS_COUNT,
            read_iops_max: READ_IOPS_MAX,
        })
    }

    fn new_tiny_grid() -> Grid {
        Grid::new(GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: 4,
            read_iops_max: 0,
        })
    }

    /// Writes a checksum-valid block header into `buffer` and returns the block's checksum.
    fn write_block_header(buffer: &mut [u8], address: u64, body_len: usize) -> u128 {
        let mut header = message_header::Block::default();
        header.size = (message_header::SIZE + body_len) as u32;
        header.release = Release { value: 1 };
        header.address = address;
        header.block_type_ordinal = message_header::BlockType::FreeSet as u8;

        let body = vec![0xAB_u8; body_len];

        header.checksum_body = header.calculate_checksum_body(&body);
        header.set_checksum();
        buffer[..message_header::SIZE].copy_from_slice(&header.to_wire());
        buffer[message_header::SIZE..message_header::SIZE + body.len()].copy_from_slice(&body);
        header.checksum
    }

    #[test]
    fn init_layout_matches_options() {
        let grid = new_grid();
        assert_eq!(grid.blocks_count(), CACHE_BLOCKS_COUNT + STASH_BLOCKS_COUNT);
        // read_iop reservations consume READ_IOPS_MAX free stash blocks:
        assert_eq!(grid.stash_available as usize, STASH_BLOCKS_COUNT - READ_IOPS_MAX);
        assert_eq!(grid.cache_slots(), CACHE_BLOCKS_COUNT);
    }

    #[test]
    fn get_block_is_lifo_and_counts_references() {
        let mut grid = new_grid();

        let first = grid.get_block();
        let second = grid.get_block();
        assert_ne!(first, second);
        assert_eq!(grid.block_references(first), 1);
        assert_eq!(grid.block_references(second), 1);

        // LIFO reuse:
        grid.block_unref(first);
        grid.block_unref(second);
        assert_eq!(grid.get_block(), second);
        assert_eq!(grid.get_block(), first);
    }

    #[test]
    fn ref_and_unref_round_trip_with_shared_references() {
        let mut grid = new_grid();

        let location = grid.get_block();
        write_block_header(grid.block_mut(location), 7, 16);

        grid.block_ref(location);
        assert_eq!(grid.block_references(location), 2);

        grid.block_unref(location);
        assert_eq!(grid.block_references(location), 1);
        grid.block_unref(location);
        assert_eq!(grid.block_references(location), 0);
    }

    #[should_panic(expected = "stash has no free blocks")]
    #[test]
    fn get_block_panics_when_stash_is_exhausted() {
        let mut grid = new_tiny_grid();
        for _ in 0..=4 {
            grid.get_block();
        }
    }

    #[should_panic(expected = "invalid block header")]
    #[test]
    fn block_ref_rejects_invalid_header() {
        let mut grid = new_grid();
        let location = grid.get_block();
        // Garbage contents (all-zero header has an invalid checksum):
        grid.block_ref(location);
    }

    /// Publishes a fresh block for `address` into the cache and releases our reference,
    /// returning the block's location.
    fn publish(grid: &mut Grid, address: u64) -> u32 {
        let location = grid.get_block();
        write_block_header(grid.block_mut(location), address, 8);
        grid.cache_upsert(address, location);
        grid.block_unref(location);
        location
    }

    #[test]
    fn cache_upsert_keeps_inserted_address_and_distinct_locations() {
        let mut grid = new_grid();

        let mut published: Vec<(u64, u32)> = Vec::new();
        // More inserts than cache slots: eviction is guaranteed by pigeonhole, but which
        // keys are evicted depends on the hash, so assert general invariants instead.
        for address in 1..=(CACHE_BLOCKS_COUNT as u64 + CACHE_BLOCKS_COUNT as u64) {
            let location = publish(&mut grid, address);
            assert_eq!(grid.cached_location(address), Some(location));

            // Cache locations stay distinct:
            let mut seen = std::collections::HashSet::new();
            for slot in 0..grid.cache_slots() {
                let cached = grid.cache_locations[slot];
                assert!(seen.insert(cached));
            }
            published.push((address, location));
        }

        // At least one early address was evicted along the way:
        let evicted_early = published
            .iter()
            .take(CACHE_BLOCKS_COUNT)
            .filter(|&&(address, _)| grid.cached_location(address).is_none())
            .count();
        assert!(evicted_early > 0);
    }

    #[test]
    fn cache_upsert_zeroes_first_evicted_unreferenced_header() {
        let mut grid = new_grid();

        let mut cached_locations: Vec<(u64, u32)> = Vec::new();
        let mut evictee: Option<(u64, u32)> = None;
        for address in 1..=(CACHE_BLOCKS_COUNT as u64 * 2) {
            publish(&mut grid, address);

            // Find the first previously-published address that just got evicted:
            for &(old_address, old_location) in &cached_locations {
                if grid.cached_location(old_address).is_none() {
                    evictee = Some((old_address, old_location));
                    break;
                }
            }
            if let Some((_, old_location)) = evictee {
                // Its header must be zeroed (contents may remain, as upstream clears only
                // the fixed-header region):
                let bytes = grid.block(old_location);
                assert!(
                    bytes[..message_header::SIZE].iter().all(|&byte| byte == 0),
                    "evicted unreferenced block header must be zeroed"
                );
                return;
            }
            cached_locations.push((
                address,
                grid.cached_location(address).unwrap_or_else(|| unreachable!("just published")),
            ));
        }
        panic!("no eviction observed despite exceeding cache capacity");
    }

    #[test]
    fn block_offset_math() {
        assert_eq!(block_offset(1), 0);
        assert_eq!(block_offset(2), BLOCK_SIZE as u64);
        assert_eq!(block_offset(10), 9 * BLOCK_SIZE as u64);
    }

    #[test]
    fn read_block_validate_accepts_a_clean_block() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);
        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum }),
            ReadBlockResult::Valid
        );
    }

    #[test]
    fn read_block_validate_detects_corrupt_frame() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        write_block_header(&mut buffer, 3, 100);
        buffer[message_header::SIZE - 1] ^= 0xFF;

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum: 42 }),
            ReadBlockResult::InvalidChecksum
        );
    }

    #[test]
    fn read_block_validate_detects_corrupt_body() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);
        // Corrupting the body leaves the (frame-only) header checksum intact:
        buffer[message_header::SIZE] ^= 0xFF;

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum }),
            ReadBlockResult::InvalidChecksumBody
        );
    }

    #[test]
    fn read_block_validate_flags_other_commands_as_misdirected_io() {
        // A fully valid reply landed where a block was expected:
        let mut reply = message_header::Reply::default();
        reply.size = (message_header::SIZE + 8) as u32;
        reply.release = Release { value: 1 };
        reply.client = 1;
        reply.op = 1;
        reply.commit = 1;
        reply.timestamp = 1;
        reply.set_checksum();

        let mut buffer = vec![0u8; BLOCK_SIZE];
        buffer[..message_header::SIZE].copy_from_slice(&reply.to_wire());

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum: reply.checksum }),
            ReadBlockResult::UnexpectedCommand
        );
    }

    #[test]
    fn read_block_validate_flags_wrong_block() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum: checksum + 1 }),
            ReadBlockResult::UnexpectedChecksum
        );
    }

    #[test]
    fn read_block_validate_flags_dirty_padding() {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let checksum = write_block_header(&mut buffer, 3, 100);
        let padded_end = (message_header::SIZE + 100).div_ceil(SECTOR_SIZE) * SECTOR_SIZE;
        buffer[padded_end - 1] = 0x1F;

        assert_eq!(
            read_block_validate(&buffer, ExpectBlock { address: 3, checksum }),
            ReadBlockResult::InvalidPadding
        );
    }
}
