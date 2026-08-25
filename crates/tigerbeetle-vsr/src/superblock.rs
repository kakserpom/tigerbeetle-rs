//! Superblock sizing constants (upstream: `src/vsr/superblock.zig` file-level constants).
//!
//! The `SuperBlockHeader`/`VSRState`/`CheckpointState` structs land in a follow-up port; the
//! sizes here are pinned to upstream's `@sizeOf` results and asserted against them.

use tigerbeetle_core::constants::{
    CLIENT_REPLIES_SIZE, CLIENTS_MAX, HEADER_SIZE, JOURNAL_SIZE, PIPELINE_PREPARE_QUEUE_MAX,
    SECTOR_SIZE, SUPERBLOCK_COPIES, VIEW_HEADERS_MAX,
};
use tigerbeetle_core::stdx::align_forward;

/// Port of `superblock.SuperBlockVersion` (src/vsr/superblock.zig:45).
///
/// Upstream selects `0` only for development builds whose own injected release is exactly
/// the minimum, and `2` otherwise — which is every real build (tests use 65535.0.0,
/// production uses an actual release). Verified byte-for-byte against upstream's
/// `SuperBlock.format()` output (see `format_matches_upstream_zig_golden`).
///
/// DEVIATION: build-time release injection isn't ported yet (`ConfigProcess` lacks
/// `release`), so we hard-code the value every upstream build takes in practice.
/// TODO(port): src/config.zig BuildOptions — derive from `CONFIG.process.release`.
pub const SUPERBLOCK_VERSION: u16 = 2;

/// Leave enough padding after every superblock copy so that it is feasible, in the future, to
/// modify the `pipeline_prepare_queue_max` of an existing cluster (up to a maximum of
/// clients_max). (That is, this space is reserved for potential `view_headers`).
pub const VIEW_HEADERS_RESERVED_SIZE: usize =
    SECTOR_SIZE - (VIEW_HEADERS_MAX as usize * HEADER_SIZE) % SECTOR_SIZE;

/// Size of `SuperBlockHeader`: fixed fields below `view_headers_all`, then the view headers at
/// offset `SECTOR_SIZE`, then [`VIEW_HEADERS_RESERVED_SIZE`] of trailing padding.
///
/// Upstream asserts `@sizeOf(SuperBlockHeader) == 8192` indirectly via
/// `% sector_size == 0` and `/ sector_size >= 2`.
pub const SUPERBLOCK_HEADER_SIZE: usize =
    SECTOR_SIZE + VIEW_HEADERS_MAX as usize * HEADER_SIZE + VIEW_HEADERS_RESERVED_SIZE;

const _: () = assert!(SUPERBLOCK_HEADER_SIZE == 8192);

/// Padding added after every superblock copy.
pub const SUPERBLOCK_COPY_PADDING: usize = align_forward(
    (CLIENTS_MAX as usize - PIPELINE_PREPARE_QUEUE_MAX as usize) * HEADER_SIZE,
    SECTOR_SIZE,
);

/// The size of an individual superblock header copy, including padding.
pub const SUPERBLOCK_COPY_SIZE: usize = SUPERBLOCK_HEADER_SIZE + SUPERBLOCK_COPY_PADDING;

/// The size of the entire superblock storage zone.
pub const SUPERBLOCK_ZONE_SIZE: usize = SUPERBLOCK_COPY_SIZE * SUPERBLOCK_COPIES;

/// The size of the data file that has an empty grid (upstream: `data_file_size_min`).
pub const DATA_FILE_SIZE_MIN: usize =
    SUPERBLOCK_ZONE_SIZE + JOURNAL_SIZE + CLIENT_REPLIES_SIZE + super::zone_size_grid_padding();

const _: () = assert!(SUPERBLOCK_COPY_PADDING.is_multiple_of(SECTOR_SIZE));
const _: () = assert!(SUPERBLOCK_COPY_SIZE.is_multiple_of(SECTOR_SIZE));

#[cfg(test)]
mod tests {
    use super::{
        DATA_FILE_SIZE_MIN, SUPERBLOCK_COPY_PADDING, SUPERBLOCK_COPY_SIZE, SUPERBLOCK_HEADER_SIZE,
        SUPERBLOCK_ZONE_SIZE, VIEW_HEADERS_RESERVED_SIZE,
    };
    use crate::{Zone, zone_size_grid_padding};
    use tigerbeetle_core::constants::{
        BLOCK_SIZE, CLIENT_REPLIES_SIZE, CLIENTS_MAX, HEADER_SIZE, JOURNAL_SIZE,
        JOURNAL_SIZE_HEADERS, JOURNAL_SIZE_PREPARES, PIPELINE_PREPARE_QUEUE_MAX, SECTOR_SIZE,
        SUPERBLOCK_COPIES as COPIES, VIEW_HEADERS_MAX,
    };

    /// The header is at least two sectors and a whole number of sectors
    /// (upstream: SuperBlockHeader comptime asserts).
    #[test]
    fn superblock_header_sizing() {
        assert_eq!(
            SUPERBLOCK_HEADER_SIZE,
            SECTOR_SIZE + VIEW_HEADERS_MAX as usize * HEADER_SIZE + VIEW_HEADERS_RESERVED_SIZE
        );
        assert_eq!(SUPERBLOCK_HEADER_SIZE % SECTOR_SIZE, 0);
        assert_eq!(SUPERBLOCK_HEADER_SIZE / SECTOR_SIZE, 2);
    }

    /// Copy padding reserves room for view_headers growth up to clients_max
    /// (upstream: superblock_copy_padding / _copy_size / _zone_size).
    #[test]
    fn superblock_copy_and_zone_sizing() {
        assert_eq!(
            SUPERBLOCK_COPY_PADDING,
            tigerbeetle_core::stdx::align_forward(
                (CLIENTS_MAX as usize - PIPELINE_PREPARE_QUEUE_MAX as usize) * HEADER_SIZE,
                SECTOR_SIZE,
            )
        );
        assert_eq!(SUPERBLOCK_COPY_PADDING % SECTOR_SIZE, 0);
        assert_eq!(SUPERBLOCK_COPY_SIZE, SUPERBLOCK_HEADER_SIZE + SUPERBLOCK_COPY_PADDING);
        assert_eq!(SUPERBLOCK_COPY_SIZE % SECTOR_SIZE, 0);
        assert_eq!(SUPERBLOCK_ZONE_SIZE, SUPERBLOCK_COPY_SIZE * COPIES);

        // data_file_size_min = all fixed zones + grid padding:
        let expected_grid_padding = {
            let unaligned = SUPERBLOCK_ZONE_SIZE
                + JOURNAL_SIZE_HEADERS
                + JOURNAL_SIZE_PREPARES
                + CLIENT_REPLIES_SIZE;
            tigerbeetle_core::stdx::align_forward(unaligned, BLOCK_SIZE) - unaligned
        };
        assert_eq!(expected_grid_padding, zone_size_grid_padding());
        assert_eq!(
            DATA_FILE_SIZE_MIN,
            SUPERBLOCK_ZONE_SIZE + JOURNAL_SIZE + CLIENT_REPLIES_SIZE + expected_grid_padding
        );
        assert_eq!(DATA_FILE_SIZE_MIN % SECTOR_SIZE, 0);
        assert_eq!(Zone::Superblock.size(), Some(SUPERBLOCK_ZONE_SIZE as u64));
    }
}

// ---------------------------------------------------------------------------
// SuperBlockHeader and its VSRState/CheckpointState (upstream: superblock.zig
// lines 49–583). DEVIATION: upstream is an `extern struct` reinterpreted via
// pointer casts; this port keeps plain structs with explicit little-endian
// wire codecs, mirroring `message_header.rs` / `schema.rs`.
//
// TODO(port): remaining SuperBlockHeader methods that need Storage (open/checkpoint
// machinery) and ClientSessions codec.
// ---------------------------------------------------------------------------

use crate::Zone;
use crate::message_header;
use crate::message_header::TypedHeader as _;
use crate::multiversion::Release;
use crate::storage::{Completion, ReadRequest, Storage, WriteRequest};
use crate::superblock_quorums;
use crate::{Members, ViewChangeCommand, ViewChangeHeadersSlice, member_index};
use tigerbeetle_core::constants::{
    BLOCK_SIZE, MEMBERS_MAX, REPLICAS_MAX, STANDBYS_MAX, STORAGE_SIZE_LIMIT_MAX,
};

pub use tigerbeetle_core::constants::CHECKPOINT_STATE_SIZE;

/// Port of `vsr.checksum(&.{})`: the checksum of an empty body.
use message_header::checksum_body_empty;

/// Port of `vsr.ClientSessions.encode_size` (src/vsr/client_sessions.zig:80).
pub use crate::client_sessions::ClientSessions;

pub const CLIENT_SESSIONS_ENCODE_SIZE: usize = ClientSessions::ENCODE_SIZE;

const _: () = assert!(CLIENT_SESSIONS_ENCODE_SIZE <= BLOCK_SIZE - message_header::SIZE);

/// Which free-set bitset a [`TrailerReference`] refers to (upstream:
/// `vsr.FreeSet.BitsetKind`; TODO(port): move into the FreeSet port).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitsetKind {
    BlocksAcquired,
    BlocksReleased,
}

/// Port of `superblock.ManifestReferences`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ManifestReferences {
    /// The chronologically first manifest block in the chain.
    pub oldest_checksum: u128,
    pub oldest_address: u64,
    /// The chronologically last manifest block in the chain.
    pub newest_checksum: u128,
    pub newest_address: u64,
    /// The number of manifest blocks in the chain.
    pub block_count: u32,
}

impl ManifestReferences {
    /// # Panics
    /// Panics if a zero `block_count` accompanies nonzero addresses/checksums, or vice versa
    /// (upstream asserts).
    #[must_use]
    pub fn empty(&self) -> bool {
        if self.block_count == 0 {
            assert_eq!(self.oldest_address, 0);
            assert_eq!(self.oldest_checksum, 0);
            assert_eq!(self.newest_address, 0);
            assert_eq!(self.newest_checksum, 0);
            true
        } else {
            assert!(self.oldest_address != 0);
            assert!(self.newest_address != 0);
            false
        }
    }
}

/// Port of `superblock.TrailerReference`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrailerReference {
    /// Checksum over the entire encoded trailer.
    pub checksum: u128,
    pub last_block_address: u64,
    pub last_block_checksum: u128,
    pub trailer_size: u64,
}

impl TrailerReference {
    /// # Panics
    /// Panics if a zero `trailer_size` accompanies nonzero fields, or a nonzero `trailer_size`
    /// accompanies a zero address (upstream asserts).
    #[must_use]
    pub fn empty(&self) -> bool {
        if self.trailer_size == 0 {
            assert_eq!(self.checksum, checksum_body_empty());
            assert_eq!(self.last_block_address, 0);
            assert_eq!(self.last_block_checksum, 0);
            true
        } else {
            assert!(self.last_block_address > 0);
            false
        }
    }
}

/// Port of `SuperBlockHeader.CheckpointState`: the deterministic per-checkpoint state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointState {
    /// The last prepare of the checkpoint committed to the state machine.
    /// At startup, replay the log hereafter.
    pub header: message_header::Prepare,

    pub free_set_blocks_acquired_last_block_checksum: u128,
    pub free_set_blocks_acquired_last_block_checksum_padding: u128,

    pub free_set_blocks_released_last_block_checksum: u128,
    pub free_set_blocks_released_last_block_checksum_padding: u128,

    pub client_sessions_last_block_checksum: u128,
    pub client_sessions_last_block_checksum_padding: u128,
    pub manifest_oldest_checksum: u128,
    pub manifest_oldest_checksum_padding: u128,
    pub manifest_newest_checksum: u128,
    pub manifest_newest_checksum_padding: u128,
    pub snapshots_block_checksum: u128,
    pub snapshots_block_checksum_padding: u128,

    /// Checksum covering the entire encoded free set. Strictly speaking it is redundant:
    /// free_set_last_block_checksum indirectly covers the same data. It is still useful
    /// to protect from encoding-decoding bugs as a defense in depth.
    pub free_set_blocks_acquired_checksum: u128,
    pub free_set_blocks_released_checksum: u128,

    /// Checksum covering the entire client sessions, as defense-in-depth.
    pub client_sessions_checksum: u128,

    /// The checkpoint_id() of the checkpoint which last updated our commit_min.
    /// Following state sync, this is set to the last checkpoint that we skipped.
    pub parent_checkpoint_id: u128,
    /// The parent_checkpoint_id of the parent checkpoint.
    /// TODO We might be able to remove this when
    /// https://github.com/tigerbeetle/tigerbeetle/issues/1378 is fixed.
    pub grandparent_checkpoint_id: u128,

    pub free_set_blocks_acquired_last_block_address: u64,
    pub free_set_blocks_released_last_block_address: u64,

    pub client_sessions_last_block_address: u64,
    pub manifest_oldest_address: u64,
    pub manifest_newest_address: u64,
    pub snapshots_block_address: u64,

    // Logical storage size in bytes.
    //
    // If storage_size is less than the data file size, then the grid blocks beyond
    // storage_size were used previously, but have since been freed.
    //
    // If storage_size is more than the data file size, then the data file might have been
    // truncated/corrupted.
    pub storage_size: u64,

    // Size of the encoded trailers in bytes.
    // It is equal to the sum of sizes of individual trailer blocks and is used for assertions.
    pub free_set_blocks_acquired_size: u64,
    pub free_set_blocks_released_size: u64,

    pub client_sessions_size: u64,

    /// The number of manifest blocks in the manifest log.
    pub manifest_block_count: u32,

    /// All prepares between `CheckpointState.commit_min` (i.e. `op_checkpoint`) and
    /// `trigger_for_checkpoint(checkpoint_after(commit_min))` must be executed by this release.
    /// (Prepares with `operation=upgrade` are the exception – upgrades in the last
    /// `lsm_compaction_ops` before a checkpoint trigger may be replayed by a different release.)
    pub release: Release,
}

impl CheckpointState {
    const OFFSET_HEADER: usize = 0;
    const OFFSET_ACQUIRED_LB_CHECKSUM: usize = 256;
    const OFFSET_ACQUIRED_LB_CHECKSUM_PADDING: usize = 272;
    const OFFSET_RELEASED_LB_CHECKSUM: usize = 288;
    const OFFSET_RELEASED_LB_CHECKSUM_PADDING: usize = 304;
    const OFFSET_CLIENT_SESSIONS_LB_CHECKSUM: usize = 320;
    const OFFSET_CLIENT_SESSIONS_LB_CHECKSUM_PADDING: usize = 336;
    const OFFSET_MANIFEST_OLDEST_CHECKSUM: usize = 352;
    const OFFSET_MANIFEST_OLDEST_CHECKSUM_PADDING: usize = 368;
    const OFFSET_MANIFEST_NEWEST_CHECKSUM: usize = 384;
    const OFFSET_MANIFEST_NEWEST_CHECKSUM_PADDING: usize = 400;
    const OFFSET_SNAPSHOTS_BLOCK_CHECKSUM: usize = 416;
    const OFFSET_SNAPSHOTS_BLOCK_CHECKSUM_PADDING: usize = 432;
    const OFFSET_FREE_SET_BLOCKS_ACQUIRED_CHECKSUM: usize = 448;
    const OFFSET_FREE_SET_BLOCKS_RELEASED_CHECKSUM: usize = 464;
    const OFFSET_CLIENT_SESSIONS_CHECKSUM: usize = 480;
    const OFFSET_PARENT_CHECKPOINT_ID: usize = 496;
    const OFFSET_GRANDPARENT_CHECKPOINT_ID: usize = 512;
    const OFFSET_ACQUIRED_LB_ADDRESS: usize = 528;
    const OFFSET_RELEASED_LB_ADDRESS: usize = 536;
    const OFFSET_CLIENT_SESSIONS_LB_ADDRESS: usize = 544;
    const OFFSET_MANIFEST_OLDEST_ADDRESS: usize = 552;
    const OFFSET_MANIFEST_NEWEST_ADDRESS: usize = 560;
    const OFFSET_SNAPSHOTS_BLOCK_ADDRESS: usize = 568;
    const OFFSET_STORAGE_SIZE: usize = 576;
    const OFFSET_ACQUIRED_SIZE: usize = 584;
    const OFFSET_RELEASED_SIZE: usize = 592;
    const OFFSET_CLIENT_SESSIONS_SIZE: usize = 600;
    const OFFSET_MANIFEST_BLOCK_COUNT: usize = 608;
    const OFFSET_RELEASE: usize = 612;
    const OFFSET_RESERVED: usize = 616;

    /// # Panics
    /// Panics if the reserved area is nonzero or a trailing checksum padding is nonzero.
    #[must_use]
    pub fn from_wire(bytes: &[u8; CHECKPOINT_STATE_SIZE]) -> Self {
        let get_u128 = |offset: usize| {
            u128::from_le_bytes(
                bytes[offset..offset + 16]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            )
        };
        let get_u64 = |offset: usize| {
            u64::from_le_bytes(
                bytes[offset..offset + 8]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            )
        };

        assert!(tigerbeetle_core::stdx::zeroed(&bytes[Self::OFFSET_RESERVED..]));
        for offset in [
            Self::OFFSET_ACQUIRED_LB_CHECKSUM_PADDING,
            Self::OFFSET_RELEASED_LB_CHECKSUM_PADDING,
            Self::OFFSET_CLIENT_SESSIONS_LB_CHECKSUM_PADDING,
            Self::OFFSET_MANIFEST_OLDEST_CHECKSUM_PADDING,
            Self::OFFSET_MANIFEST_NEWEST_CHECKSUM_PADDING,
            Self::OFFSET_SNAPSHOTS_BLOCK_CHECKSUM_PADDING,
        ] {
            assert_eq!(get_u128(offset), 0, "checksum padding != 0 @{offset}");
        }

        // DEVIATION: upstream treats the superblock as raw memory, so a corrupt copy may hold
        // garbage where a valid `Header.Prepare` should be. Such bytes cannot re-encode to
        // what was on disk, so they shift `SuperBlockHeader::calculate_checksum` and the copy
        // is rejected by checksum validation — we substitute a zeroed header instead of
        // rejecting the decode outright.
        Self {
            header: message_header::Prepare::from_wire(
                &bytes[Self::OFFSET_HEADER..Self::OFFSET_HEADER + message_header::SIZE]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            )
            .unwrap_or_else(header_prepare_zeroed),
            free_set_blocks_acquired_last_block_checksum: get_u128(
                Self::OFFSET_ACQUIRED_LB_CHECKSUM,
            ),
            free_set_blocks_acquired_last_block_checksum_padding: get_u128(
                Self::OFFSET_ACQUIRED_LB_CHECKSUM_PADDING,
            ),
            free_set_blocks_released_last_block_checksum: get_u128(
                Self::OFFSET_RELEASED_LB_CHECKSUM,
            ),
            free_set_blocks_released_last_block_checksum_padding: get_u128(
                Self::OFFSET_RELEASED_LB_CHECKSUM_PADDING,
            ),
            client_sessions_last_block_checksum: get_u128(Self::OFFSET_CLIENT_SESSIONS_LB_CHECKSUM),
            client_sessions_last_block_checksum_padding: get_u128(
                Self::OFFSET_CLIENT_SESSIONS_LB_CHECKSUM_PADDING,
            ),
            manifest_oldest_checksum: get_u128(Self::OFFSET_MANIFEST_OLDEST_CHECKSUM),
            manifest_oldest_checksum_padding: get_u128(
                Self::OFFSET_MANIFEST_OLDEST_CHECKSUM_PADDING,
            ),
            manifest_newest_checksum: get_u128(Self::OFFSET_MANIFEST_NEWEST_CHECKSUM),
            manifest_newest_checksum_padding: get_u128(
                Self::OFFSET_MANIFEST_NEWEST_CHECKSUM_PADDING,
            ),
            snapshots_block_checksum: get_u128(Self::OFFSET_SNAPSHOTS_BLOCK_CHECKSUM),
            snapshots_block_checksum_padding: get_u128(
                Self::OFFSET_SNAPSHOTS_BLOCK_CHECKSUM_PADDING,
            ),
            free_set_blocks_acquired_checksum: get_u128(
                Self::OFFSET_FREE_SET_BLOCKS_ACQUIRED_CHECKSUM,
            ),
            free_set_blocks_released_checksum: get_u128(
                Self::OFFSET_FREE_SET_BLOCKS_RELEASED_CHECKSUM,
            ),
            client_sessions_checksum: get_u128(Self::OFFSET_CLIENT_SESSIONS_CHECKSUM),
            parent_checkpoint_id: get_u128(Self::OFFSET_PARENT_CHECKPOINT_ID),
            grandparent_checkpoint_id: get_u128(Self::OFFSET_GRANDPARENT_CHECKPOINT_ID),
            free_set_blocks_acquired_last_block_address: get_u64(Self::OFFSET_ACQUIRED_LB_ADDRESS),
            free_set_blocks_released_last_block_address: get_u64(Self::OFFSET_RELEASED_LB_ADDRESS),
            client_sessions_last_block_address: get_u64(Self::OFFSET_CLIENT_SESSIONS_LB_ADDRESS),
            manifest_oldest_address: get_u64(Self::OFFSET_MANIFEST_OLDEST_ADDRESS),
            manifest_newest_address: get_u64(Self::OFFSET_MANIFEST_NEWEST_ADDRESS),
            snapshots_block_address: get_u64(Self::OFFSET_SNAPSHOTS_BLOCK_ADDRESS),
            storage_size: get_u64(Self::OFFSET_STORAGE_SIZE),
            free_set_blocks_acquired_size: get_u64(Self::OFFSET_ACQUIRED_SIZE),
            free_set_blocks_released_size: get_u64(Self::OFFSET_RELEASED_SIZE),
            client_sessions_size: get_u64(Self::OFFSET_CLIENT_SESSIONS_SIZE),
            manifest_block_count: u32::from_le_bytes(
                bytes[Self::OFFSET_MANIFEST_BLOCK_COUNT..Self::OFFSET_MANIFEST_BLOCK_COUNT + 4]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            release: Release {
                value: u32::from_le_bytes(
                    bytes[Self::OFFSET_RELEASE..Self::OFFSET_RELEASE + 4]
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("slice length checked")),
                ),
            },
        }
    }

    /// Serializes into the 1024-byte on-disk representation (reserved zeroed).
    ///
    /// The length is inherent to the mechanical field-by-field codec (upstream relies on the
    /// extern-struct layout instead).
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn to_wire(&self) -> [u8; CHECKPOINT_STATE_SIZE] {
        let mut bytes = [0u8; CHECKPOINT_STATE_SIZE];
        let put_u128 = |bytes: &mut [u8], offset: usize, value: u128| {
            bytes[offset..offset + 16].copy_from_slice(&value.to_le_bytes());
        };
        let put_u64 = |bytes: &mut [u8], offset: usize, value: u64| {
            bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        };

        bytes[Self::OFFSET_HEADER..Self::OFFSET_HEADER + message_header::SIZE]
            .copy_from_slice(&self.header.to_wire());

        put_u128(
            &mut bytes,
            Self::OFFSET_ACQUIRED_LB_CHECKSUM,
            self.free_set_blocks_acquired_last_block_checksum,
        );
        put_u128(
            &mut bytes,
            Self::OFFSET_ACQUIRED_LB_CHECKSUM_PADDING,
            self.free_set_blocks_acquired_last_block_checksum_padding,
        );
        put_u128(
            &mut bytes,
            Self::OFFSET_RELEASED_LB_CHECKSUM,
            self.free_set_blocks_released_last_block_checksum,
        );
        put_u128(
            &mut bytes,
            Self::OFFSET_RELEASED_LB_CHECKSUM_PADDING,
            self.free_set_blocks_released_last_block_checksum_padding,
        );
        put_u128(
            &mut bytes,
            Self::OFFSET_CLIENT_SESSIONS_LB_CHECKSUM,
            self.client_sessions_last_block_checksum,
        );
        put_u128(
            &mut bytes,
            Self::OFFSET_CLIENT_SESSIONS_LB_CHECKSUM_PADDING,
            self.client_sessions_last_block_checksum_padding,
        );
        put_u128(&mut bytes, Self::OFFSET_MANIFEST_OLDEST_CHECKSUM, self.manifest_oldest_checksum);
        put_u128(
            &mut bytes,
            Self::OFFSET_MANIFEST_OLDEST_CHECKSUM_PADDING,
            self.manifest_oldest_checksum_padding,
        );
        put_u128(&mut bytes, Self::OFFSET_MANIFEST_NEWEST_CHECKSUM, self.manifest_newest_checksum);
        put_u128(
            &mut bytes,
            Self::OFFSET_MANIFEST_NEWEST_CHECKSUM_PADDING,
            self.manifest_newest_checksum_padding,
        );
        put_u128(&mut bytes, Self::OFFSET_SNAPSHOTS_BLOCK_CHECKSUM, self.snapshots_block_checksum);
        put_u128(
            &mut bytes,
            Self::OFFSET_SNAPSHOTS_BLOCK_CHECKSUM_PADDING,
            self.snapshots_block_checksum_padding,
        );
        put_u128(
            &mut bytes,
            Self::OFFSET_FREE_SET_BLOCKS_ACQUIRED_CHECKSUM,
            self.free_set_blocks_acquired_checksum,
        );
        put_u128(
            &mut bytes,
            Self::OFFSET_FREE_SET_BLOCKS_RELEASED_CHECKSUM,
            self.free_set_blocks_released_checksum,
        );
        put_u128(&mut bytes, Self::OFFSET_CLIENT_SESSIONS_CHECKSUM, self.client_sessions_checksum);
        put_u128(&mut bytes, Self::OFFSET_PARENT_CHECKPOINT_ID, self.parent_checkpoint_id);
        put_u128(
            &mut bytes,
            Self::OFFSET_GRANDPARENT_CHECKPOINT_ID,
            self.grandparent_checkpoint_id,
        );

        put_u64(
            &mut bytes,
            Self::OFFSET_ACQUIRED_LB_ADDRESS,
            self.free_set_blocks_acquired_last_block_address,
        );
        put_u64(
            &mut bytes,
            Self::OFFSET_RELEASED_LB_ADDRESS,
            self.free_set_blocks_released_last_block_address,
        );
        put_u64(
            &mut bytes,
            Self::OFFSET_CLIENT_SESSIONS_LB_ADDRESS,
            self.client_sessions_last_block_address,
        );
        put_u64(&mut bytes, Self::OFFSET_MANIFEST_OLDEST_ADDRESS, self.manifest_oldest_address);
        put_u64(&mut bytes, Self::OFFSET_MANIFEST_NEWEST_ADDRESS, self.manifest_newest_address);
        put_u64(&mut bytes, Self::OFFSET_SNAPSHOTS_BLOCK_ADDRESS, self.snapshots_block_address);
        put_u64(&mut bytes, Self::OFFSET_STORAGE_SIZE, self.storage_size);
        put_u64(&mut bytes, Self::OFFSET_ACQUIRED_SIZE, self.free_set_blocks_acquired_size);
        put_u64(&mut bytes, Self::OFFSET_RELEASED_SIZE, self.free_set_blocks_released_size);
        put_u64(&mut bytes, Self::OFFSET_CLIENT_SESSIONS_SIZE, self.client_sessions_size);

        bytes[Self::OFFSET_MANIFEST_BLOCK_COUNT..Self::OFFSET_MANIFEST_BLOCK_COUNT + 4]
            .copy_from_slice(&self.manifest_block_count.to_le_bytes());
        bytes[Self::OFFSET_RELEASE..Self::OFFSET_RELEASE + 4]
            .copy_from_slice(&self.release.value.to_le_bytes());

        bytes
    }
}

/// Port of `SuperBlockHeader.VSRState`: state stored on stable storage for the
/// Viewstamped Replication consensus protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VSRState {
    pub checkpoint: CheckpointState,

    /// Globally unique identifier of the replica, must be non-zero.
    pub replica_id: u128,

    pub members: Members,

    /// The highest operation up to which we may commit.
    pub commit_max: u64,

    /// See `sync_op_max` (upstream: `sync_op_min`).
    pub sync_op_min: u64,

    /// When zero, all of the grid blocks and replies are synced.
    /// (When zero, `sync_op_min` is also zero.)
    ///
    /// When nonzero, we must repair grid-blocks/client-replies that would have been written
    /// during the commits between `sync_op_min` and `sync_op_max` (inclusive).
    /// (Those grid-blocks and client-replies were not written normally because we "skipped"
    /// past them via state sync.)
    pub sync_op_max: u64,

    /// This field was used by the old state sync protocol, but is now unused and is always set
    /// to zero.
    /// TODO: rename to reserved and assert that it is zero, once it is actually set to zero
    /// in all superblocks (in the next release).
    pub sync_view: u32,

    /// The last view in which the replica's status was normal.
    pub log_view: u32,

    /// The view number of the replica.
    pub view: u32,

    /// Number of replicas (determines sizes of the quorums), part of VSR configuration.
    pub replica_count: u8,
}

#[derive(Clone, Copy)]
pub struct VSRStateRootOptions {
    pub cluster: u128,
    pub replica_id: u128,
    pub members: Members,
    pub replica_count: u8,
    pub release: Release,
    pub view: u32,
}

impl VSRState {
    pub const SIZE: usize = 2048;

    const OFFSET_CHECKPOINT: usize = 0;
    const OFFSET_REPLICA_ID: usize = 1024;
    const OFFSET_MEMBERS: usize = 1040;
    const OFFSET_COMMIT_MAX: usize = 1232;
    const OFFSET_SYNC_OP_MIN: usize = 1240;
    const OFFSET_SYNC_OP_MAX: usize = 1248;
    const OFFSET_SYNC_VIEW: usize = 1256;
    const OFFSET_LOG_VIEW: usize = 1260;
    const OFFSET_VIEW: usize = 1264;
    const OFFSET_REPLICA_COUNT: usize = 1268;
    const OFFSET_RESERVED: usize = 1269;
    const RESERVED_LEN: usize = Self::SIZE - Self::OFFSET_RESERVED;

    /// Port of `VSRState.root`.
    #[must_use]
    pub fn root(options: VSRStateRootOptions) -> Self {
        Self {
            checkpoint: CheckpointState {
                header: message_header::Prepare::root(options.cluster),
                parent_checkpoint_id: 0,
                grandparent_checkpoint_id: 0,
                free_set_blocks_acquired_checksum: checksum_body_empty(),
                free_set_blocks_released_checksum: checksum_body_empty(),
                free_set_blocks_acquired_last_block_checksum: 0,
                free_set_blocks_released_last_block_checksum: 0,
                free_set_blocks_acquired_last_block_address: 0,
                free_set_blocks_released_last_block_address: 0,
                free_set_blocks_acquired_size: 0,
                free_set_blocks_released_size: 0,
                client_sessions_checksum: checksum_body_empty(),
                client_sessions_last_block_checksum: 0,
                client_sessions_last_block_address: 0,
                client_sessions_size: 0,
                manifest_oldest_checksum: 0,
                manifest_oldest_address: 0,
                manifest_newest_checksum: 0,
                manifest_newest_address: 0,
                manifest_block_count: 0,
                snapshots_block_checksum: 0,
                snapshots_block_address: 0,
                storage_size: DATA_FILE_SIZE_MIN as u64,
                release: options.release,
                free_set_blocks_acquired_last_block_checksum_padding: 0,
                free_set_blocks_released_last_block_checksum_padding: 0,
                client_sessions_last_block_checksum_padding: 0,
                manifest_oldest_checksum_padding: 0,
                manifest_newest_checksum_padding: 0,
                snapshots_block_checksum_padding: 0,
            },
            replica_id: options.replica_id,
            members: options.members,
            replica_count: options.replica_count,
            commit_max: 0,
            sync_op_min: 0,
            sync_op_max: 0,
            sync_view: 0,
            log_view: 0,
            view: options.view,
        }
    }

    /// # Panics
    /// Panics if any field combination is impossible or reserved/padding data is nonzero
    /// (upstream asserts the same).
    pub fn assert_internally_consistent(&self) {
        assert!(self.commit_max >= self.checkpoint.header.op);
        assert!(self.sync_op_max >= self.sync_op_min);
        assert!(self.view >= self.log_view);
        assert!(self.replica_count > 0);
        assert!(usize::from(self.replica_count) <= REPLICAS_MAX);
        assert!(member_index(&self.members, self.replica_id).is_some());

        // These fields are unused at the moment:
        assert_eq!(self.checkpoint.snapshots_block_checksum, 0);
        assert_eq!(self.checkpoint.snapshots_block_address, 0);

        assert_eq!(self.checkpoint.manifest_oldest_checksum_padding, 0);
        assert_eq!(self.checkpoint.manifest_newest_checksum_padding, 0);
        assert_eq!(self.checkpoint.snapshots_block_checksum_padding, 0);
        assert_eq!(self.checkpoint.free_set_blocks_acquired_last_block_checksum_padding, 0);
        assert_eq!(self.checkpoint.free_set_blocks_released_last_block_checksum_padding, 0);

        assert_eq!(self.checkpoint.client_sessions_last_block_checksum_padding, 0);
        assert!(self.checkpoint.storage_size >= DATA_FILE_SIZE_MIN as u64);

        if self.checkpoint.free_set_blocks_acquired_last_block_address == 0 {
            assert_eq!(self.checkpoint.free_set_blocks_acquired_size, 0);
            assert_eq!(self.checkpoint.free_set_blocks_acquired_checksum, checksum_body_empty());
            assert_eq!(self.checkpoint.free_set_blocks_acquired_last_block_checksum, 0);
        } else {
            assert!(self.checkpoint.free_set_blocks_acquired_size > 0);
        }

        if self.checkpoint.free_set_blocks_released_last_block_address == 0 {
            assert_eq!(self.checkpoint.free_set_blocks_released_size, 0);
            assert_eq!(self.checkpoint.free_set_blocks_released_checksum, checksum_body_empty());
            assert_eq!(self.checkpoint.free_set_blocks_released_last_block_checksum, 0);
        } else {
            assert!(self.checkpoint.free_set_blocks_released_size > 0);
        }

        if self.checkpoint.client_sessions_last_block_address == 0 {
            assert_eq!(self.checkpoint.client_sessions_last_block_checksum, 0);
            assert_eq!(self.checkpoint.client_sessions_size, 0);
            assert_eq!(self.checkpoint.client_sessions_checksum, checksum_body_empty());
        } else {
            assert_eq!(self.checkpoint.client_sessions_size, CLIENT_SESSIONS_ENCODE_SIZE as u64);
        }

        if self.checkpoint.manifest_block_count == 0 {
            assert_eq!(self.checkpoint.manifest_oldest_address, 0);
            assert_eq!(self.checkpoint.manifest_newest_address, 0);
            assert_eq!(self.checkpoint.manifest_oldest_checksum, 0);
            assert_eq!(self.checkpoint.manifest_newest_checksum, 0);
        } else {
            assert!(self.checkpoint.manifest_oldest_address != 0);
            assert!(self.checkpoint.manifest_newest_address != 0);

            assert_eq!(
                self.checkpoint.manifest_block_count == 1,
                self.checkpoint.manifest_oldest_address == self.checkpoint.manifest_newest_address
            );

            assert_eq!(
                self.checkpoint.manifest_block_count == 1,
                self.checkpoint.manifest_oldest_checksum
                    == self.checkpoint.manifest_newest_checksum
            );
        }
    }

    /// Port of `VSRState.monotonic`.
    ///
    /// # Panics
    /// Panics if either state is internally inconsistent or an unrelated field changed
    /// (upstream asserts).
    #[must_use]
    pub fn monotonic(old: &VSRState, new: &VSRState) -> bool {
        old.assert_internally_consistent();
        new.assert_internally_consistent();
        if old.checkpoint.header.op == new.checkpoint.header.op {
            if old.checkpoint.header.checksum == 0 && old.checkpoint.header.op == 0 {
                // "old" is the root VSRState.
                assert_eq!(old.commit_max, 0);
                assert_eq!(old.sync_op_min, 0);
                assert_eq!(old.sync_op_max, 0);
                assert_eq!(old.log_view, 0);
                assert_eq!(old.view, 0);
            } else {
                assert_eq!(old.checkpoint.to_wire(), new.checkpoint.to_wire());
            }
        } else {
            assert_ne!(old.checkpoint.header.checksum, new.checkpoint.header.checksum);
            assert_ne!(old.checkpoint.parent_checkpoint_id, new.checkpoint.parent_checkpoint_id);
        }
        assert_eq!(old.replica_id, new.replica_id);
        assert_eq!(old.replica_count, new.replica_count);
        assert_eq!(old.members, new.members);

        if old.checkpoint.header.op > new.checkpoint.header.op {
            return false;
        }
        if old.view > new.view {
            return false;
        }
        if old.log_view > new.log_view {
            return false;
        }
        if old.commit_max > new.commit_max {
            return false;
        }

        true
    }

    /// Port of `VSRState.would_be_updated_by`.
    ///
    /// # Panics
    /// Panics if `monotonic(old, new)` panics.
    #[must_use]
    pub fn would_be_updated_by(old: &VSRState, new: &VSRState) -> bool {
        assert!(Self::monotonic(old, new));

        old.to_wire() != new.to_wire()
    }

    /// Compaction is one bar ahead of superblock's commit_min.
    /// The commits from the bar following commit_min were in the mutable table, and
    /// thus not preserved in the checkpoint.
    /// But the corresponding `compact()` updates were preserved, and must not be repeated
    /// to ensure deterministic storage.
    ///
    /// # Panics
    /// Panics if `checkpoint.trigger_for_checkpoint()` returns `None` despite a nonzero
    /// checkpoint op (impossible by construction).
    #[must_use]
    pub fn op_compacted(&self, op: u64) -> bool {
        // If commit_min is 0, we have never checkpointed, so no compactions are checkpointed.
        self.checkpoint.header.op > 0
            && op
                <= crate::checkpoint::trigger_for_checkpoint(self.checkpoint.header.op)
                    .unwrap_or_else(|| unreachable!("nonzero checkpoint ops always have a trigger"))
    }

    /// # Panics
    /// Panics if the trailing reserved area is nonzero.
    #[must_use]
    pub fn from_wire(bytes: &[u8; Self::SIZE]) -> Self {
        let mut checkpoint_bytes = [0u8; CHECKPOINT_STATE_SIZE];
        checkpoint_bytes.copy_from_slice(
            &bytes[Self::OFFSET_CHECKPOINT..Self::OFFSET_CHECKPOINT + CHECKPOINT_STATE_SIZE],
        );

        assert!(tigerbeetle_core::stdx::zeroed(
            &bytes[Self::OFFSET_RESERVED..Self::OFFSET_RESERVED + Self::RESERVED_LEN]
        ));

        let get_u128 = |offset: usize| {
            u128::from_le_bytes(
                bytes[offset..offset + 16]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            )
        };

        let mut members = [0u128; tigerbeetle_core::constants::MEMBERS_MAX];
        for (i, member) in members.iter_mut().enumerate() {
            *member = get_u128(Self::OFFSET_MEMBERS + i * 16);
        }

        Self {
            checkpoint: CheckpointState::from_wire(&checkpoint_bytes),
            replica_id: get_u128(Self::OFFSET_REPLICA_ID),
            members,
            commit_max: u64::from_le_bytes(
                bytes[Self::OFFSET_COMMIT_MAX..Self::OFFSET_COMMIT_MAX + 8]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            sync_op_min: u64::from_le_bytes(
                bytes[Self::OFFSET_SYNC_OP_MIN..Self::OFFSET_SYNC_OP_MIN + 8]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            sync_op_max: u64::from_le_bytes(
                bytes[Self::OFFSET_SYNC_OP_MAX..Self::OFFSET_SYNC_OP_MAX + 8]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            sync_view: u32::from_le_bytes(
                bytes[Self::OFFSET_SYNC_VIEW..Self::OFFSET_SYNC_VIEW + 4]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            log_view: u32::from_le_bytes(
                bytes[Self::OFFSET_LOG_VIEW..Self::OFFSET_LOG_VIEW + 4]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            view: u32::from_le_bytes(
                bytes[Self::OFFSET_VIEW..Self::OFFSET_VIEW + 4]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            replica_count: bytes[Self::OFFSET_REPLICA_COUNT],
        }
    }

    /// Serializes into the 2048-byte on-disk representation (reserved zeroed).
    #[must_use]
    pub fn to_wire(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[Self::OFFSET_CHECKPOINT..Self::OFFSET_CHECKPOINT + CHECKPOINT_STATE_SIZE]
            .copy_from_slice(&self.checkpoint.to_wire());
        bytes[Self::OFFSET_REPLICA_ID..Self::OFFSET_REPLICA_ID + 16]
            .copy_from_slice(&self.replica_id.to_le_bytes());
        for (i, member) in self.members.iter().enumerate() {
            bytes[Self::OFFSET_MEMBERS + i * 16..Self::OFFSET_MEMBERS + i * 16 + 16]
                .copy_from_slice(&member.to_le_bytes());
        }
        bytes[Self::OFFSET_COMMIT_MAX..Self::OFFSET_COMMIT_MAX + 8]
            .copy_from_slice(&self.commit_max.to_le_bytes());
        bytes[Self::OFFSET_SYNC_OP_MIN..Self::OFFSET_SYNC_OP_MIN + 8]
            .copy_from_slice(&self.sync_op_min.to_le_bytes());
        bytes[Self::OFFSET_SYNC_OP_MAX..Self::OFFSET_SYNC_OP_MAX + 8]
            .copy_from_slice(&self.sync_op_max.to_le_bytes());
        bytes[Self::OFFSET_SYNC_VIEW..Self::OFFSET_SYNC_VIEW + 4]
            .copy_from_slice(&self.sync_view.to_le_bytes());
        bytes[Self::OFFSET_LOG_VIEW..Self::OFFSET_LOG_VIEW + 4]
            .copy_from_slice(&self.log_view.to_le_bytes());
        bytes[Self::OFFSET_VIEW..Self::OFFSET_VIEW + 4].copy_from_slice(&self.view.to_le_bytes());
        bytes[Self::OFFSET_REPLICA_COUNT] = self.replica_count;

        bytes
    }
}

/// Port of `superblock.SuperBlockHeader`.
///
/// Layout (little-endian on disk): fixed fields below `view_headers_all`, then
/// `view_headers_all` at offset [`SECTOR_SIZE`], then
/// [`VIEW_HEADERS_RESERVED_SIZE`] trailing padding. Total: [`SUPERBLOCK_HEADER_SIZE`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuperBlockHeader {
    pub checksum: u128,
    pub checksum_padding: u128,

    /// Protects against misdirected reads at startup.
    /// For example, if multiple reads are all misdirected to a single copy of the superblock.
    /// Excluded from the checksum calculation to ensure that all copies have the same checksum.
    /// This simplifies writing and comparing multiple copies.
    /// TODO: u8 should be enough here, we use u16 only for alignment.
    pub copy: u16,

    /// The version of the superblock format in use, reserved for major breaking changes.
    pub version: u16,

    /// The release that the data file was originally formatted by.
    /// (Upgrades do not update this field.)
    pub release_format: Release,

    /// A monotonically increasing counter to locate the latest superblock at startup.
    pub sequence: u64,

    /// Protects against writing to or reading from the wrong data file.
    pub cluster: u128,

    /// The checksum of the previous superblock to hash chain across sequence numbers.
    pub parent: u128,
    pub parent_padding: u128,

    /// State stored on stable storage for the Viewstamped Replication consensus protocol.
    pub vsr_state: VSRState,

    /// Reserved for future minor features (e.g. changing a compression algorithm).
    pub flags: u64,

    /// The number of headers in view_headers_all.
    pub view_headers_count: u32,

    /// View/JV header suffix. Headers are ordered from high-to-low op.
    /// Unoccupied headers (after view_headers_count) are zeroed.
    ///
    /// When `vsr_state.log_view < vsr_state.view`, the headers are for a JV.
    /// When `vsr_state.log_view = vsr_state.view`, the headers are for a View.
    ///
    /// DEVIATION: kept as raw wire bytes because an *unoccupied* slot is all-zero, which is
    /// not a decodable `Prepare` (its command would be `reserved`). Decode through
    /// [`message_header::TypedHeader::from_wire`] when reading occupied slots.
    pub view_headers_all: [[u8; message_header::SIZE]; VIEW_HEADERS_MAX as usize],
}

pub struct SuperBlockHeaderOptions {
    pub version: u16,
    pub release_format: Release,
    pub sequence: u64,
    pub cluster: u128,
    pub parent: u128,
    pub vsr_state: VSRState,
    pub flags: u64,
    pub view_headers_count: u32,
}

impl SuperBlockHeader {
    const OFFSET_CHECKSUM: usize = 0;
    const OFFSET_CHECKSUM_PADDING: usize = 16;
    const OFFSET_COPY: usize = 32;
    const OFFSET_VERSION: usize = 34;
    const OFFSET_RELEASE_FORMAT: usize = 36;
    const OFFSET_SEQUENCE: usize = 40;
    const OFFSET_CLUSTER: usize = 48;
    const OFFSET_PARENT: usize = 64;
    const OFFSET_PARENT_PADDING: usize = 80;
    const OFFSET_VSR_STATE: usize = 96;
    const OFFSET_FLAGS: usize = 2144;
    const OFFSET_VIEW_HEADERS_COUNT: usize = 2152;
    const OFFSET_RESERVED: usize = 2156;
    const RESERVED_LEN: usize = SECTOR_SIZE - Self::OFFSET_RESERVED;
    const OFFSET_VIEW_HEADERS_ALL: usize = SECTOR_SIZE;
    const OFFSET_VIEW_HEADERS_RESERVED: usize =
        Self::OFFSET_VIEW_HEADERS_ALL + VIEW_HEADERS_MAX as usize * message_header::SIZE;

    /// Bytes excluded from the header checksum: { checksum, checksum_padding, copy }.
    const CHECKSUM_IGNORE_SIZE: usize = 16 + 16 + 2;

    /// # Panics
    /// Panics if `view_headers_count > VIEW_HEADERS_MAX` or the trailing reserved area is
    /// nonzero.
    #[must_use]
    pub fn from_wire(bytes: &[u8; SUPERBLOCK_HEADER_SIZE]) -> Option<Self> {
        let get_u128 = |offset: usize| {
            u128::from_le_bytes(
                bytes[offset..offset + 16]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            )
        };

        assert!(tigerbeetle_core::stdx::zeroed(
            &bytes[Self::OFFSET_RESERVED..Self::OFFSET_RESERVED + Self::RESERVED_LEN]
        ));
        assert!(tigerbeetle_core::stdx::zeroed(&bytes[Self::OFFSET_VIEW_HEADERS_RESERVED..]));

        let mut view_headers_all = [[0u8; message_header::SIZE]; VIEW_HEADERS_MAX as usize];
        for (i, header) in view_headers_all.iter_mut().enumerate() {
            let offset = Self::OFFSET_VIEW_HEADERS_ALL + i * message_header::SIZE;
            header.copy_from_slice(&bytes[offset..offset + message_header::SIZE]);
        }

        Some(Self {
            checksum: get_u128(Self::OFFSET_CHECKSUM),
            checksum_padding: get_u128(Self::OFFSET_CHECKSUM_PADDING),
            copy: u16::from_le_bytes(
                bytes[Self::OFFSET_COPY..Self::OFFSET_COPY + 2]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            version: u16::from_le_bytes(
                bytes[Self::OFFSET_VERSION..Self::OFFSET_VERSION + 2]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            release_format: Release {
                value: u32::from_le_bytes(
                    bytes[Self::OFFSET_RELEASE_FORMAT..Self::OFFSET_RELEASE_FORMAT + 4]
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("slice length checked")),
                ),
            },
            sequence: u64::from_le_bytes(
                bytes[Self::OFFSET_SEQUENCE..Self::OFFSET_SEQUENCE + 8]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            cluster: get_u128(Self::OFFSET_CLUSTER),
            parent: get_u128(Self::OFFSET_PARENT),
            parent_padding: get_u128(Self::OFFSET_PARENT_PADDING),
            vsr_state: {
                let mut vsr_bytes = [0u8; VSRState::SIZE];
                vsr_bytes.copy_from_slice(
                    &bytes[Self::OFFSET_VSR_STATE..Self::OFFSET_VSR_STATE + VSRState::SIZE],
                );
                VSRState::from_wire(&vsr_bytes)
            },
            flags: u64::from_le_bytes(
                bytes[Self::OFFSET_FLAGS..Self::OFFSET_FLAGS + 8]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            view_headers_count: u32::from_le_bytes(
                bytes[Self::OFFSET_VIEW_HEADERS_COUNT..Self::OFFSET_VIEW_HEADERS_COUNT + 4]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            ),
            view_headers_all,
        })
    }

    /// Serializes into the on-disk representation (reserved areas zeroed).
    ///
    /// # Panics
    /// Panics if `view_headers_count > VIEW_HEADERS_MAX` or an unoccupied view header is not
    /// zeroed (upstream relies on the caller maintaining this).
    #[must_use]
    pub fn to_wire(&self) -> [u8; SUPERBLOCK_HEADER_SIZE] {
        assert!(self.view_headers_count <= VIEW_HEADERS_MAX);
        // Unoccupied view headers must be zeroed:
        for header in self.view_headers_all.iter().skip(self.view_headers_count as usize) {
            assert_eq!(
                *header,
                [0u8; message_header::SIZE],
                "unoccupied view header is not zeroed"
            );
        }

        let mut bytes = [0u8; SUPERBLOCK_HEADER_SIZE];
        bytes[Self::OFFSET_CHECKSUM..Self::OFFSET_CHECKSUM + 16]
            .copy_from_slice(&self.checksum.to_le_bytes());
        bytes[Self::OFFSET_CHECKSUM_PADDING..Self::OFFSET_CHECKSUM_PADDING + 16]
            .copy_from_slice(&self.checksum_padding.to_le_bytes());
        bytes[Self::OFFSET_COPY..Self::OFFSET_COPY + 2].copy_from_slice(&self.copy.to_le_bytes());
        bytes[Self::OFFSET_VERSION..Self::OFFSET_VERSION + 2]
            .copy_from_slice(&self.version.to_le_bytes());
        bytes[Self::OFFSET_RELEASE_FORMAT..Self::OFFSET_RELEASE_FORMAT + 4]
            .copy_from_slice(&self.release_format.value.to_le_bytes());
        bytes[Self::OFFSET_SEQUENCE..Self::OFFSET_SEQUENCE + 8]
            .copy_from_slice(&self.sequence.to_le_bytes());
        bytes[Self::OFFSET_CLUSTER..Self::OFFSET_CLUSTER + 16]
            .copy_from_slice(&self.cluster.to_le_bytes());
        bytes[Self::OFFSET_PARENT..Self::OFFSET_PARENT + 16]
            .copy_from_slice(&self.parent.to_le_bytes());
        bytes[Self::OFFSET_PARENT_PADDING..Self::OFFSET_PARENT_PADDING + 16]
            .copy_from_slice(&self.parent_padding.to_le_bytes());

        bytes[Self::OFFSET_VSR_STATE..Self::OFFSET_VSR_STATE + VSRState::SIZE]
            .copy_from_slice(&self.vsr_state.to_wire());

        bytes[Self::OFFSET_FLAGS..Self::OFFSET_FLAGS + 8]
            .copy_from_slice(&self.flags.to_le_bytes());
        bytes[Self::OFFSET_VIEW_HEADERS_COUNT..Self::OFFSET_VIEW_HEADERS_COUNT + 4]
            .copy_from_slice(&self.view_headers_count.to_le_bytes());

        for (i, header) in self.view_headers_all.iter().enumerate() {
            let offset = Self::OFFSET_VIEW_HEADERS_ALL + i * message_header::SIZE;
            bytes[offset..offset + message_header::SIZE].copy_from_slice(header);
        }

        bytes
    }

    /// Port of `SuperBlockHeader.calculate_checksum`.
    #[must_use]
    pub fn calculate_checksum(&self) -> u128 {
        tigerbeetle_core::checksum::checksum(&self.to_wire()[Self::CHECKSUM_IGNORE_SIZE..])
    }

    /// Port of `SuperBlockHeader.set_checksum`.
    ///
    /// # Panics
    /// Panics unless the working header is a pristine copy-0 staging header with no reserved
    /// data set (upstream asserts the same).
    pub fn set_checksum(&mut self) {
        // `copy` is not covered by the checksum, but for our staging/working superblock headers
        // it should always be zero.
        assert!(usize::from(self.copy) < SUPERBLOCK_COPIES);
        assert_eq!(self.copy, 0);

        assert_eq!(self.version, SUPERBLOCK_VERSION);
        assert!(self.release_format.value > 0);
        assert_eq!(self.flags, 0);

        assert_eq!(self.checksum_padding, 0);
        assert_eq!(self.parent_padding, 0);

        self.checksum = self.calculate_checksum();
    }

    /// Port of `SuperBlockHeader.valid_checksum`.
    #[must_use]
    pub fn valid_checksum(&self) -> bool {
        self.checksum == self.calculate_checksum() && self.checksum_padding == 0
    }

    /// Port of `SuperBlockHeader.checkpoint_id`.
    #[must_use]
    pub fn checkpoint_id(&self) -> u128 {
        tigerbeetle_core::checksum::checksum(&self.vsr_state.checkpoint.to_wire())
    }

    /// Port of `SuperBlockHeader.parent_checkpoint_id`.
    #[must_use]
    pub fn parent_checkpoint_id(&self) -> u128 {
        self.vsr_state.checkpoint.parent_checkpoint_id
    }

    /// Does not consider { checksum, copy } when comparing equality.
    ///
    /// # Panics
    /// Panics if either header has nonzero reserved/padding data (upstream asserts).
    #[must_use]
    pub fn equal(&self, other: &Self) -> bool {
        assert_eq!(self.release_format.value, other.release_format.value);

        assert_eq!(self.checksum_padding, 0);
        assert_eq!(other.checksum_padding, 0);
        assert_eq!(self.parent_padding, 0);
        assert_eq!(other.parent_padding, 0);

        // Reserved areas are implicitly zero in this port's wire codec; upstream asserts them.

        if self.version != other.version {
            return false;
        }
        if self.cluster != other.cluster {
            return false;
        }
        if self.sequence != other.sequence {
            return false;
        }
        if self.parent != other.parent {
            return false;
        }
        if self.vsr_state != other.vsr_state {
            return false;
        }
        if self.view_headers_count != other.view_headers_count {
            return false;
        }
        if self.view_headers_all != other.view_headers_all {
            return false;
        }

        true
    }

    /// Port of `SuperBlockHeader.view_headers`.
    ///
    /// DEVIATION: upstream stores decoded `Header.Prepare`s and returns a slice over them;
    /// this port keeps `view_headers_all` as raw wire bytes (so that garbage beyond
    /// `view_headers_count` still round-trips byte-exactly for checksum validation), so the
    /// caller provides decode storage which the returned slice borrows.
    ///
    /// # Panics
    /// Panics if an occupied slot does not decode as a Prepare header, or if the decoded
    /// headers fail [`ViewChangeHeadersSlice::verify`] (upstream asserts in both cases).
    pub fn view_headers<'s>(
        &self,
        decoded: &'s mut [message_header::Prepare; VIEW_HEADERS_MAX as usize],
    ) -> ViewChangeHeadersSlice<'s> {
        for (slot, wire) in decoded
            .iter_mut()
            .zip(&self.view_headers_all)
            .take(usize::try_from(self.view_headers_count).unwrap_or(usize::MAX))
        {
            *slot = message_header::Prepare::from_wire(wire)
                .unwrap_or_else(|| unreachable!("occupied view header must decode as Prepare"));
        }

        let command = if self.vsr_state.log_view < self.vsr_state.view {
            ViewChangeCommand::JoinView
        } else {
            ViewChangeCommand::View
        };
        ViewChangeHeadersSlice::init(
            command,
            &decoded[..usize::try_from(self.view_headers_count).unwrap_or(usize::MAX)],
        )
    }

    /// Port of `SuperBlockHeader.manifest_references`.
    #[must_use]
    pub fn manifest_references(&self) -> ManifestReferences {
        let checkpoint_state = &self.vsr_state.checkpoint;
        ManifestReferences {
            oldest_address: checkpoint_state.manifest_oldest_address,
            oldest_checksum: checkpoint_state.manifest_oldest_checksum,
            newest_address: checkpoint_state.manifest_newest_address,
            newest_checksum: checkpoint_state.manifest_newest_checksum,
            block_count: checkpoint_state.manifest_block_count,
        }
    }

    /// Port of `SuperBlockHeader.free_set_reference`.
    #[must_use]
    pub fn free_set_reference(&self, bitset: BitsetKind) -> TrailerReference {
        match bitset {
            BitsetKind::BlocksAcquired => TrailerReference {
                checksum: self.vsr_state.checkpoint.free_set_blocks_acquired_checksum,
                last_block_address: self
                    .vsr_state
                    .checkpoint
                    .free_set_blocks_acquired_last_block_address,
                last_block_checksum: self
                    .vsr_state
                    .checkpoint
                    .free_set_blocks_acquired_last_block_checksum,
                trailer_size: self.vsr_state.checkpoint.free_set_blocks_acquired_size,
            },
            BitsetKind::BlocksReleased => TrailerReference {
                checksum: self.vsr_state.checkpoint.free_set_blocks_released_checksum,
                last_block_address: self
                    .vsr_state
                    .checkpoint
                    .free_set_blocks_released_last_block_address,
                last_block_checksum: self
                    .vsr_state
                    .checkpoint
                    .free_set_blocks_released_last_block_checksum,
                trailer_size: self.vsr_state.checkpoint.free_set_blocks_released_size,
            },
        }
    }
}

#[cfg(test)]
mod header_tests {
    #![allow(clippy::unwrap_used)]

    use super::{
        BitsetKind, CheckpointState, SUPERBLOCK_VERSION, SuperBlockHeader, VSRState,
        VSRStateRootOptions,
    };
    use crate::message_header;
    use crate::message_header::TypedHeader as _;
    use crate::multiversion::Release;
    use tigerbeetle_core::constants::{MEMBERS_MAX, VIEW_HEADERS_MAX};

    fn root_vsr_state(cluster: u128) -> VSRState {
        let mut members = [0u128; MEMBERS_MAX];
        members[0] = 11;
        members[1] = 22;
        members[2] = 33;
        VSRState::root(VSRStateRootOptions {
            cluster,
            replica_id: 11,
            members,
            replica_count: 3,
            release: Release::MINIMUM,
            view: 0,
        })
    }

    /// Port of upstream test "SuperBlockHeader".
    #[test]
    fn superblock_header_checksum_excludes_copy() {
        // Upstream zeroes the whole extern struct; our struct has no reserved bytes to zero,
        // so start from a minimal valid staging header instead.
        let vsr_state = root_vsr_state(7);
        let mut a = SuperBlockHeader {
            checksum: 0,
            checksum_padding: 0,
            copy: 0,
            version: SUPERBLOCK_VERSION,
            release_format: Release::MINIMUM,
            sequence: 0,
            cluster: 7,
            parent: 0,
            parent_padding: 0,
            vsr_state,
            flags: 0,
            view_headers_count: 0,
            view_headers_all: zeroed_view_headers(),
        };
        assert_eq!(a.copy, 0);
        a.set_checksum();
        assert!(a.valid_checksum());

        a.copy += 1;
        assert!(a.valid_checksum(), "copy is excluded from the checksum");

        a.version += 1;
        assert!(!a.valid_checksum());
    }

    #[test]
    fn superblock_header_wire_round_trip() {
        let mut header = SuperBlockHeader {
            checksum: 0,
            checksum_padding: 0,
            copy: 0,
            version: SUPERBLOCK_VERSION,
            release_format: Release::MINIMUM,
            sequence: 42,
            cluster: 9_001,
            parent: 123,
            parent_padding: 0,
            vsr_state: root_vsr_state(9_001),
            flags: 0,
            view_headers_count: 0,
            view_headers_all: zeroed_view_headers(),
        };
        header.set_checksum();

        let decoded = SuperBlockHeader::from_wire(&header.to_wire()).unwrap_or_else(|| {
            unreachable!("a freshly encoded header must decode");
        });
        assert_eq!(decoded.checksum, header.checksum);
        assert!(decoded.valid_checksum());
        assert!(header.equal(&decoded));

        // checkpoint_id covers exactly the encoded checkpoint state:
        assert_eq!(
            decoded.checkpoint_id(),
            tigerbeetle_core::checksum::checksum(&decoded.vsr_state.checkpoint.to_wire())
        );
        assert_eq!(decoded.parent_checkpoint_id(), 0);

        // Free-set references of a fresh root point at nothing.
        for bitset in [BitsetKind::BlocksAcquired, BitsetKind::BlocksReleased] {
            assert!(decoded.free_set_reference(bitset).empty());
        }
        assert!(decoded.manifest_references().empty());
    }

    #[test]
    fn vsr_state_root_is_internally_consistent_and_monotonic_base() {
        let root = root_vsr_state(5);
        root.assert_internally_consistent();
        assert_eq!(root.commit_max, 0);
        assert_eq!(root.sync_op_min, 0);
        assert_eq!(root.view, 0);
        assert_eq!(root.log_view, 0);
        assert_eq!(root.checkpoint.header.op, 0);
        assert_eq!(
            usize::try_from(root.checkpoint.storage_size).unwrap_or(usize::MAX),
            super::DATA_FILE_SIZE_MIN
        );

        // The root state is the monotonic base of every successor.
        let mut next = root;
        next.commit_max = 10;
        next.view = 1;
        assert!(VSRState::monotonic(&root, &next));
        assert!(VSRState::would_be_updated_by(&root, &next));
        assert!(!VSRState::monotonic(&next, &root), "regressions are not monotonic");
    }

    #[test]
    fn vsr_state_op_compacted_tracks_trigger() {
        let mut state = root_vsr_state(5);
        assert!(!state.op_compacted(1), "nothing is compacted before a checkpoint");

        // First checkpoint op (vsr_checkpoint_ops - 1):
        let checkpoint = crate::checkpoint::checkpoint_after(0);
        state.checkpoint.header.op = checkpoint;
        let trigger = crate::checkpoint::trigger_for_checkpoint(checkpoint)
            .unwrap_or_else(|| unreachable!("nonzero checkpoints have triggers"));
        // Everything through the trigger counts as compacted; beyond it does not.
        assert!(state.op_compacted(checkpoint));
        assert!(state.op_compacted(trigger));
        assert!(!state.op_compacted(trigger + 1));
    }

    #[test]
    fn checkpoint_state_wire_round_trip() {
        let state = root_vsr_state(6).checkpoint;

        let wire = state.to_wire();
        assert_eq!(wire.len(), 1024);
        let decoded = CheckpointState::from_wire(&wire);
        assert_eq!(decoded, state);
    }

    fn zeroed_view_headers() -> [[u8; message_header::SIZE]; VIEW_HEADERS_MAX as usize] {
        // Unoccupied slots are all-zero on disk (not a decodable Prepare).
        [[0u8; message_header::SIZE]; VIEW_HEADERS_MAX as usize]
    }

    #[test]
    fn superblock_header_view_headers() {
        use crate::command::Command;
        use crate::headers;
        use crate::{ViewChangeCommand, ViewRange};
        use tigerbeetle_core::constants::VSR_OPERATIONS_RESERVED;

        // Two consecutive valid prepares (ops 9, 8) followed by a JV blank gap (op 7).
        // `parent` points at the previous prepare's checksum, so build low-to-high:
        let mut op8 = message_header::Prepare {
            client: 6,
            request: 7,
            command: Command::Prepare,
            release: Release::MINIMUM,
            operation: crate::Operation(VSR_OPERATIONS_RESERVED + 7),
            op: 8,
            view: 10,
            timestamp: 10,
            ..Default::default()
        };
        op8.set_checksum();

        let mut op9 = message_header::Prepare {
            client: 6,
            request: 7,
            command: Command::Prepare,
            release: Release::MINIMUM,
            operation: crate::Operation(VSR_OPERATIONS_RESERVED + 8),
            op: 9,
            view: 10,
            timestamp: 11,
            parent: op8.checksum,
            ..Default::default()
        };
        op9.set_checksum();

        let blank = headers::jv_blank(7);

        let mut header = SuperBlockHeader {
            checksum: 0,
            checksum_padding: 0,
            copy: 0,
            version: SUPERBLOCK_VERSION,
            release_format: Release::MINIMUM,
            sequence: 42,
            cluster: 9_001,
            parent: 123,
            parent_padding: 0,
            vsr_state: root_vsr_state(9_001),
            flags: 0,
            view_headers_count: 3,
            view_headers_all: zeroed_view_headers(),
        };
        header.view_headers_all[0] = op9.to_wire();
        header.view_headers_all[1] = op8.to_wire();
        header.view_headers_all[2] = blank.to_wire();
        header.vsr_state.log_view = 12;
        header.vsr_state.view = 13; // log_view < view → JV headers
        header.set_checksum();

        let mut decoded_storage = [message_header::Prepare::default(); VIEW_HEADERS_MAX as usize];
        let view_change_headers = header.view_headers(&mut decoded_storage);
        assert_eq!(view_change_headers.command, ViewChangeCommand::JoinView);
        assert_eq!(view_change_headers.slice.len(), 3);

        // The decoded slice behaves like upstream's: ops at or above the newest valid header
        // pin to their own view; ops below the oldest valid header widen to {0, oldest.view}.
        // Here the oldest valid header is op 8 (view 10):
        assert_eq!(view_change_headers.view_for_op(9, 13), ViewRange { min: 10, max: 10 });
        assert_eq!(view_change_headers.view_for_op(8, 13), ViewRange { min: 10, max: 10 });
        assert_eq!(view_change_headers.view_for_op(7, 13), ViewRange { min: 0, max: 10 });
        assert_eq!(view_change_headers.view_for_op(4, 13), ViewRange { min: 0, max: 10 });
        assert_eq!(view_change_headers.view_for_op(14, 13), ViewRange { min: 13, max: 13 });

        // With view == log_view the same bytes are interpreted as View headers:
        header.vsr_state.log_view = 13;
        header.set_checksum();
        let view_change_headers = header.view_headers(&mut decoded_storage);
        assert_eq!(view_change_headers.command, ViewChangeCommand::View);
    }
}

/// Port of `superblock.SuperBlock.Caller`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caller {
    Format,
    Open,
    Checkpoint,
    ViewChange,
}

impl Caller {
    /// Port of `Caller.transitions`: the set of callers that may be queued directly behind this
    /// one.
    ///
    /// Beyond formatting and opening of the superblock, which are mutually exclusive of all
    /// other operations, only the following queue combinations are allowed:
    ///
    /// from state → to states
    #[must_use]
    pub fn transitions(self) -> &'static [Self] {
        match self {
            Self::Format | Self::Open => &[],
            Self::Checkpoint => &[Self::ViewChange],
            Self::ViewChange => &[Self::Checkpoint],
        }
    }

    /// Port of `Caller.updates_view_headers`.
    ///
    /// # Panics
    /// Panics for [`Caller::Open`] (upstream: `unreachable`).
    #[must_use]
    pub fn updates_view_headers(self) -> bool {
        match self {
            Self::Format | Self::Checkpoint | Self::ViewChange => true,
            Self::Open => unreachable!("open does not update view headers"),
        }
    }
}

/// Events drained via [`SuperBlock::take_events`] (upstream: context callback invocations).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    FormatDone,
    OpenDone,
    CheckpointDone,
    ViewChangeDone,
}

/// Upstream: `SuperBlock.FormatOptions`.
#[derive(Clone, Copy, Debug)]
pub struct FormatOptions {
    pub cluster: u128,
    pub release: Release,
    pub replica: u8,
    pub replica_count: u8,
    /// Set to `None` during initial cluster formatting.
    /// Set to the target view when constructing a new data file for a reformatted replica.
    pub view: Option<u32>,
}

/// Upstream: `SuperBlock.UpdateCheckpoint`.
///
/// DEVIATION: upstream holds pointers (`view_attributes.headers`,
/// `client_sessions_reference` is by-value already); here every payload is owned.
#[derive(Clone, Debug)]
pub struct UpdateCheckpoint {
    /// Must update the commit_min and commit_min_checksum.
    pub header: message_header::Prepare,
    pub view_attributes: Option<ViewAttributes>,
    pub commit_max: u64,
    pub sync_op_min: u64,
    pub sync_op_max: u64,
    pub manifest_references: ManifestReferences,
    pub free_set_references: FreeSetReferences,
    pub client_sessions_reference: TrailerReference,
    pub storage_size: u64,
    pub release: Release,
}

/// Upstream: the anonymous `view_attributes` struct of [`UpdateCheckpoint`].
#[derive(Clone, Debug)]
pub struct ViewAttributes {
    pub log_view: u32,
    pub view: u32,
    pub headers: crate::ViewChangeHeadersArray,
}

/// Upstream: the anonymous `free_set_references` struct of [`UpdateCheckpoint`].
#[derive(Clone, Copy, Debug)]
pub struct FreeSetReferences {
    pub blocks_acquired: TrailerReference,
    pub blocks_released: TrailerReference,
}

/// Upstream: `SuperBlock.UpdateViewChange`.
///
/// The replica calls view_change():
///
/// - to persist its view/log_view — it cannot advertise either value until it is certain
///   they will never backtrack.
/// - to update checkpoint during sync
///
/// The update must advance view/log_view (monotonically increasing) or checkpoint.
// TODO(port): upstream notes the current naming is confusing and needs changing: during sync,
// this function doesn't necessarily advance the view (superblock.zig `view_change`).
#[derive(Clone, Debug)]
pub struct UpdateViewChange {
    pub commit_max: u64,
    pub log_view: u32,
    pub view: u32,
    pub headers: crate::ViewChangeHeadersArray,
    pub sync_checkpoint: Option<SyncCheckpoint>,
}

/// Upstream: the anonymous `sync_checkpoint` struct of [`UpdateViewChange`].
#[derive(Clone, Debug)]
pub struct SyncCheckpoint {
    pub checkpoint: CheckpointState,
    pub sync_op_min: u64,
    pub sync_op_max: u64,
}

/// Per-operation state (upstream: `superblock.Context`, with our storage completions taking
/// the role of the embedded `Storage.Read`/`Storage.Write` slots).
struct Context {
    caller: Caller,

    /// Write/read progress within the operation (upstream: `context.copy`).
    copy: Option<u8>,
    /// Set while a sequence of superblock reads is in flight (upstream: `read_threshold`).
    read_threshold: Option<superblock_quorums::Threshold>,

    /// Format/checkpoint/view_change: the VSR state to install into the staging header.
    vsr_state: Option<VSRState>,
    /// Format/checkpoint/view_change: the View/JV headers to install.
    view_headers: Option<crate::ViewChangeHeadersArray>,

    /// Open only: the repair plan derived from the working quorum.
    repairs: Option<superblock_quorums::RepairIterator>,
}

/// Port of `superblock.SuperBlock` (the runtime half — format and open).
///
/// DEVIATION: upstream stores a `*Storage` pointer; here the owner passes the storage to each
/// I/O-driving call (`format`/`open`/`poll`), matching this repo's convention in
/// `client_replies`. A future `Replica` will hand us the same storage on every poll.
pub struct SuperBlock {
    /// The current superblock header.
    working: SuperBlockHeader,
    /// The next superblock header (a work-in-progress copy of `working`).
    staging: SuperBlockHeader,
    /// The superblock copies being read from storage.
    reading: [SuperBlockHeader; SUPERBLOCK_COPIES],

    /// Whether the superblock has been opened. An open superblock may not be formatted.
    opened: bool,

    /// Runtime limit on the size of the datafile.
    storage_size_limit: u64,

    /// The active operation, if any (upstream: `queue_head`).
    queue_head: Option<Context>,
    /// An operation queued behind the active one (upstream: `queue_tail`). Only
    /// checkpoint/view_change may be queued, and only behind each other.
    queue_tail: Option<Context>,

    /// Set after format(); finalized by open(). Used for logging.
    replica_index: Option<u8>,

    events: std::collections::VecDeque<Event>,
}

/// An all-zero header (upstream leaves working/staging/reading `undefined` until an operation
/// installs them; Rust needs *something*, and all-zeros round-trips the wire codec).
fn header_zeroed() -> SuperBlockHeader {
    SuperBlockHeader {
        checksum: 0,
        checksum_padding: 0,
        copy: 0,
        version: 0,
        release_format: Release::ZERO,
        sequence: 0,
        cluster: 0,
        parent: 0,
        parent_padding: 0,
        vsr_state: VSRState {
            checkpoint: checkpoint_state_zeroed(),
            replica_id: 0,
            members: [0; MEMBERS_MAX],
            commit_max: 0,
            sync_op_min: 0,
            sync_op_max: 0,
            sync_view: 0,
            log_view: 0,
            view: 0,
            replica_count: 0,
        },
        flags: 0,
        view_headers_count: 0,
        view_headers_all: [[0; message_header::SIZE]; VIEW_HEADERS_MAX as usize],
    }
}

/// An all-zero `CheckpointState` (upstream: field-by-field zeroing in `format()`).
fn checkpoint_state_zeroed() -> CheckpointState {
    CheckpointState {
        header: header_prepare_zeroed(),
        free_set_blocks_acquired_last_block_checksum: 0,
        free_set_blocks_acquired_last_block_checksum_padding: 0,
        free_set_blocks_released_last_block_checksum: 0,
        free_set_blocks_released_last_block_checksum_padding: 0,
        client_sessions_last_block_checksum: 0,
        client_sessions_last_block_checksum_padding: 0,
        manifest_oldest_checksum: 0,
        manifest_oldest_checksum_padding: 0,
        manifest_newest_checksum: 0,
        manifest_newest_checksum_padding: 0,
        snapshots_block_checksum: 0,
        snapshots_block_checksum_padding: 0,
        free_set_blocks_acquired_checksum: 0,
        free_set_blocks_released_checksum: 0,
        client_sessions_checksum: 0,
        parent_checkpoint_id: 0,
        grandparent_checkpoint_id: 0,
        free_set_blocks_acquired_last_block_address: 0,
        free_set_blocks_released_last_block_address: 0,
        client_sessions_last_block_address: 0,
        manifest_oldest_address: 0,
        manifest_newest_address: 0,
        snapshots_block_address: 0,
        storage_size: 0,
        free_set_blocks_acquired_size: 0,
        free_set_blocks_released_size: 0,
        client_sessions_size: 0,
        manifest_block_count: 0,
        release: Release::ZERO,
    }
}

/// An all-zero `vsr.Header.Prepare` (upstream: `mem.zeroes(vsr.Header.Prepare)`). Note that
/// `Prepare::default()` is *not* zeroed: it fills in size/protocol/command.
fn header_prepare_zeroed() -> message_header::Prepare {
    use crate::command::Command;

    message_header::Prepare {
        size: 0,
        protocol: 0,
        command: Command::Reserved,
        ..message_header::Prepare::default()
    }
}

/// Decodes a copy read from disk, tolerating garbage in the areas that valid headers keep
/// zeroed. Upstream reads raw bytes into the struct and lets the checksum reject corruption;
/// here, sanitizing first keeps `from_wire`'s reserved-area assertions intact while producing
/// the same outcome (garbage shifts the checksum input, so such copies fail validation).
fn reading_decode(bytes: &[u8; SUPERBLOCK_HEADER_SIZE]) -> SuperBlockHeader {
    let mut sanitized = *bytes;
    sanitized[SuperBlockHeader::OFFSET_RESERVED
        ..SuperBlockHeader::OFFSET_RESERVED + SuperBlockHeader::RESERVED_LEN]
        .fill(0);
    sanitized[SuperBlockHeader::OFFSET_VIEW_HEADERS_RESERVED..].fill(0);
    SuperBlockHeader::from_wire(&sanitized)
        .unwrap_or_else(|| unreachable!("sanitized bytes satisfy from_wire's checks"))
}

impl SuperBlock {
    /// Port of `SuperBlock.init`.
    ///
    /// # Panics
    /// Panics unless `storage_size_limit` is sector-aligned and within
    /// `[data_file_size_min, storage_size_limit_max]` (upstream asserts).
    #[must_use]
    pub fn new(storage_size_limit: u64) -> Self {
        assert!(storage_size_limit >= DATA_FILE_SIZE_MIN as u64);
        assert!(storage_size_limit <= STORAGE_SIZE_LIMIT_MAX as u64);
        assert_eq!(storage_size_limit % SECTOR_SIZE as u64, 0);

        Self {
            working: header_zeroed(),
            staging: header_zeroed(),
            reading: core::array::from_fn(|_| header_zeroed()),
            opened: false,
            storage_size_limit,
            queue_head: None,
            queue_tail: None,
            replica_index: None,
            events: std::collections::VecDeque::new(),
        }
    }

    #[must_use]
    pub fn opened(&self) -> bool {
        self.opened
    }

    #[must_use]
    pub fn working(&self) -> &SuperBlockHeader {
        &self.working
    }

    #[must_use]
    pub fn staging(&self) -> &SuperBlockHeader {
        &self.staging
    }

    #[must_use]
    pub fn replica_index(&self) -> Option<u8> {
        self.replica_index
    }

    /// Port of `SuperBlock.grid_size_limit`.
    #[must_use]
    pub fn grid_size_limit(&self) -> u64 {
        self.storage_size_limit - DATA_FILE_SIZE_MIN as u64
    }

    /// Port of `SuperBlock.format`.
    ///
    /// # Panics
    /// Panics if the superblock was already opened/formatted or an operation is active, and
    /// on invalid options (upstream asserts).
    pub fn format(&mut self, storage: &mut dyn Storage, options: FormatOptions) {
        assert!(!self.opened);
        assert!(self.replica_index.is_none());
        assert!(self.queue_head.is_none(), "another superblock operation is active");

        assert!(options.release.value > 0);
        assert!(options.replica_count > 0);
        assert!(options.replica_count as usize <= REPLICAS_MAX);
        assert!((options.replica as usize) < options.replica_count as usize + STANDBYS_MAX);
        if let Some(view) = options.view {
            assert!(view > 1);
            assert!(options.replica < options.replica_count);
        }

        let members = crate::root_members(options.cluster);
        let replica_id = members[usize::from(options.replica)];

        self.replica_index = member_index(&members, replica_id);

        // This working copy provides the parent checksum, and will not be written to disk.
        // We therefore use zero values to make this parent checksum as stable as possible.
        self.working = SuperBlockHeader {
            checksum: 0,
            checksum_padding: 0,
            copy: 0,
            version: SUPERBLOCK_VERSION,
            sequence: 0,
            release_format: options.release,
            cluster: options.cluster,
            parent: 0,
            parent_padding: 0,
            vsr_state: VSRState {
                checkpoint: checkpoint_state_zeroed(),
                replica_id,
                members,
                commit_max: 0,
                sync_op_min: 0,
                sync_op_max: 0,
                sync_view: 0,
                log_view: 0,
                view: 0,
                replica_count: options.replica_count,
            },
            flags: 0,
            view_headers_count: 0,
            view_headers_all: [[0; message_header::SIZE]; VIEW_HEADERS_MAX as usize],
        };

        self.working.set_checksum();

        self.acquire(
            storage,
            Context {
                caller: Caller::Format,
                copy: None,
                read_threshold: None,
                vsr_state: Some(VSRState::root(VSRStateRootOptions {
                    cluster: options.cluster,
                    release: options.release,
                    replica_id,
                    members,
                    replica_count: options.replica_count,
                    view: options.view.unwrap_or(0),
                })),
                view_headers: Some(crate::ViewChangeHeadersArray::root(options.cluster)),
                repairs: None,
            },
        );
    }

    /// Port of `SuperBlock.open`.
    ///
    /// # Panics
    /// Panics if the superblock was already opened or another operation is active, and (via
    /// `poll`) when no valid quorum can be restored from disk.
    pub fn open(&mut self, storage: &mut dyn Storage) {
        assert!(!self.opened);
        assert!(self.queue_head.is_none(), "another superblock operation is active");

        self.acquire(
            storage,
            Context {
                caller: Caller::Open,
                copy: None,
                read_threshold: None,
                vsr_state: None,
                view_headers: None,
                repairs: None,
            },
        );
    }

    /// Port of `SuperBlock.checkpoint`.
    ///
    /// Must update the commit_min and commit_min_checksum.
    ///
    /// # Panics
    /// Panics unless the superblock is open and the update advances the checkpoint validly,
    /// and when another operation is active that may not be queued behind (upstream asserts).
    pub fn checkpoint(&mut self, storage: &mut dyn Storage, update: &UpdateCheckpoint) {
        assert!(self.opened);
        assert!(update.header.op <= update.commit_max);
        assert!(update.header.op > self.staging.vsr_state.checkpoint.header.op);
        assert_ne!(
            update.header.checksum, self.staging.vsr_state.checkpoint.header.checksum,
            "the checkpoint header must update the commit_min_checksum"
        );
        assert!(update.sync_op_min <= update.sync_op_max);
        assert!(
            update.release.value >= self.staging.vsr_state.checkpoint.release.value,
            "release downgrade: new={} staging={}",
            update.release.value,
            self.staging.vsr_state.checkpoint.release.value
        );

        assert!(update.storage_size <= self.storage_size_limit);
        assert!(update.storage_size >= DATA_FILE_SIZE_MIN as u64);
        assert_eq!(
            update.storage_size == DATA_FILE_SIZE_MIN as u64,
            update.free_set_references.blocks_acquired.empty()
                && update.free_set_references.blocks_released.empty(),
        );

        // NOTE: Upstream reads `staging.vsr_state` through a copy (`vsr_state_staging`) to dodge
        // a Zig 0.11.0 miscompilation; in Rust plain reads are fine, but we keep the same shape.
        let vsr_state_staging = self.staging.vsr_state;

        let mut vsr_state = self.staging.vsr_state;
        vsr_state.checkpoint = CheckpointState {
            header: update.header,

            parent_checkpoint_id: self.staging.checkpoint_id(),
            grandparent_checkpoint_id: vsr_state_staging.checkpoint.parent_checkpoint_id,

            free_set_blocks_acquired_checksum: update.free_set_references.blocks_acquired.checksum,
            free_set_blocks_released_checksum: update.free_set_references.blocks_released.checksum,

            free_set_blocks_acquired_size: update.free_set_references.blocks_acquired.trailer_size,
            free_set_blocks_released_size: update.free_set_references.blocks_released.trailer_size,

            free_set_blocks_acquired_last_block_checksum: update
                .free_set_references
                .blocks_acquired
                .last_block_checksum,
            free_set_blocks_released_last_block_checksum: update
                .free_set_references
                .blocks_released
                .last_block_checksum,

            free_set_blocks_acquired_last_block_address: update
                .free_set_references
                .blocks_acquired
                .last_block_address,
            free_set_blocks_released_last_block_address: update
                .free_set_references
                .blocks_released
                .last_block_address,

            client_sessions_checksum: update.client_sessions_reference.checksum,
            client_sessions_last_block_checksum: update
                .client_sessions_reference
                .last_block_checksum,
            client_sessions_last_block_address: update.client_sessions_reference.last_block_address,
            client_sessions_size: update.client_sessions_reference.trailer_size,

            manifest_oldest_checksum: update.manifest_references.oldest_checksum,
            manifest_oldest_address: update.manifest_references.oldest_address,
            manifest_newest_checksum: update.manifest_references.newest_checksum,
            manifest_newest_address: update.manifest_references.newest_address,
            manifest_block_count: update.manifest_references.block_count,

            storage_size: update.storage_size,
            snapshots_block_checksum: vsr_state_staging.checkpoint.snapshots_block_checksum,
            snapshots_block_address: vsr_state_staging.checkpoint.snapshots_block_address,
            release: update.release,

            ..checkpoint_state_zeroed()
        };
        vsr_state.commit_max = update.commit_max;
        vsr_state.sync_op_min = update.sync_op_min;
        vsr_state.sync_op_max = update.sync_op_max;
        vsr_state.sync_view = 0;
        if let Some(view_attributes) = &update.view_attributes {
            assert!(view_attributes.log_view <= view_attributes.view);
            view_attributes.headers.verify();
            vsr_state.log_view = view_attributes.log_view;
            vsr_state.view = view_attributes.view;
        }

        assert!(VSRState::would_be_updated_by(&self.staging.vsr_state, &vsr_state));

        let view_headers = if let Some(view_attributes) = &update.view_attributes {
            view_attributes.headers.clone()
        } else {
            let mut decoded = [message_header::Prepare::default(); VIEW_HEADERS_MAX as usize];
            let slice = self.staging.view_headers(&mut decoded);
            crate::ViewChangeHeadersArray::init(slice.command, slice.slice)
        };

        self.acquire(
            storage,
            Context {
                caller: Caller::Checkpoint,
                copy: None,
                read_threshold: None,
                vsr_state: Some(vsr_state),
                view_headers: Some(view_headers),
                repairs: None,
            },
        );
    }

    /// Port of `SuperBlock.view_change`.
    ///
    /// # Panics
    /// Panics unless the superblock is open and the update advances view/log_view (or installs
    /// a sync checkpoint) validly, and when another operation is active that may not be queued
    /// behind (upstream asserts).
    pub fn view_change(&mut self, storage: &mut dyn Storage, update: UpdateViewChange) {
        assert!(self.opened);
        assert!(self.staging.vsr_state.commit_max <= update.commit_max);
        assert!(self.staging.vsr_state.view <= update.view);
        assert!(self.staging.vsr_state.log_view <= update.log_view);
        assert!(
            self.staging.vsr_state.log_view < update.log_view
                || self.staging.vsr_state.view < update.view
                || update.sync_checkpoint.is_some()
        );
        assert!(
            (update.headers.command == crate::ViewChangeCommand::View
                && update.log_view == update.view)
                || (update.headers.command == crate::ViewChangeCommand::JoinView
                    && update.log_view < update.view)
        );
        assert!(
            self.staging.vsr_state.checkpoint.header.op <= update.headers.array.slice()[0].op,
            "the checkpoint must not be ahead of the view change headers"
        );

        update.headers.verify();
        assert!(update.view >= update.log_view);

        let mut vsr_state = self.staging.vsr_state;
        vsr_state.commit_max = update.commit_max;
        vsr_state.log_view = update.log_view;
        vsr_state.view = update.view;
        if let Some(sync_checkpoint) = &update.sync_checkpoint {
            assert!(
                self.staging.vsr_state.checkpoint.header.op < sync_checkpoint.checkpoint.header.op
            );

            let checkpoint_next =
                crate::checkpoint::checkpoint_after(self.staging.vsr_state.checkpoint.header.op);
            let checkpoint_next_next = crate::checkpoint::checkpoint_after(checkpoint_next);

            if sync_checkpoint.checkpoint.header.op == checkpoint_next {
                assert_eq!(
                    sync_checkpoint.checkpoint.parent_checkpoint_id,
                    self.staging.checkpoint_id(),
                    "sync checkpoint parent mismatch"
                );
            } else if sync_checkpoint.checkpoint.header.op == checkpoint_next_next {
                assert_eq!(
                    sync_checkpoint.checkpoint.grandparent_checkpoint_id,
                    self.staging.checkpoint_id(),
                    "sync checkpoint grandparent mismatch"
                );
            }

            vsr_state.checkpoint = sync_checkpoint.checkpoint;
            vsr_state.sync_op_min = sync_checkpoint.sync_op_min;
            vsr_state.sync_op_max = sync_checkpoint.sync_op_max;
        }
        assert!(VSRState::would_be_updated_by(&self.staging.vsr_state, &vsr_state));

        self.acquire(
            storage,
            Context {
                caller: Caller::ViewChange,
                copy: None,
                read_threshold: None,
                vsr_state: Some(vsr_state),
                view_headers: Some(update.headers),
                repairs: None,
            },
        );
    }

    /// Port of `SuperBlock.updating`: whether an operation with this caller is queued or
    /// running.
    ///
    /// # Panics
    /// Panics unless the superblock is open (upstream asserts).
    #[must_use]
    pub fn updating(&self, caller: Caller) -> bool {
        assert!(self.opened);

        if let Some(head) = &self.queue_head
            && head.caller == caller
        {
            return true;
        }

        if let Some(tail) = &self.queue_tail {
            return tail.caller == caller;
        }

        false
    }

    /// Drives storage completions, emitting [`Event`]s (upstream: callback invocations).
    ///
    /// The superblock issues at most one I/O at a time (see `read_working`), so every
    /// completion correlates with the active operation's current step.
    ///
    /// # Panics
    /// Panics on completions without an active operation, foreign zones, quorum failures
    /// (fork / not found / quorum lost / …), incompatible versions, failed post-write
    /// verification, and oversized data files (all mirroring upstream panics/fatals).
    pub fn poll(&mut self, storage: &mut dyn Storage) {
        while let Some(completion) = storage.next_completion() {
            assert_eq!(completion.zone(), Zone::Superblock);
            match completion {
                Completion::Read(request) => self.poll_read(storage, request),
                Completion::Write(request) => self.poll_write(storage, request),
            }
        }
    }

    /// Drains accumulated events (upstream: callbacks invoked synchronously).
    pub fn take_events(&mut self) -> Vec<Event> {
        self.events.drain(..).collect()
    }

    fn op_mut(&mut self) -> &mut Context {
        self.queue_head.as_mut().unwrap_or_else(|| unreachable!("operation is active"))
    }

    /// Port of `SuperBlock.acquire`: starts the operation immediately if none is running,
    /// otherwise queues it behind the active one.
    fn acquire(&mut self, storage: &mut dyn Storage, context: Context) {
        if let Some(head) = &self.queue_head {
            let head_caller = head.caller;
            // All operations are mutually exclusive with themselves.
            assert_ne!(
                head_caller, context.caller,
                "all operations are mutually exclusive with themselves"
            );
            assert!(
                head_caller.transitions().contains(&context.caller),
                "{head_caller:?} may not be followed by {:?}",
                context.caller
            );
            assert!(self.queue_tail.is_none());

            self.queue_tail = Some(context);
        } else {
            assert!(self.queue_tail.is_none());

            let caller = context.caller;
            self.queue_head = Some(context);

            match caller {
                Caller::Open => self.read_working(storage, superblock_quorums::Threshold::Open),
                _ => self.write_staging(storage),
            }
        }
    }

    /// Port of `SuperBlock.release`: completes the head operation, fires its event, and starts
    /// the queued operation (if any).
    fn release(&mut self, storage: &mut dyn Storage) {
        let op = self.queue_head.take().unwrap_or_else(|| unreachable!("operation is active"));
        let queued = self.queue_tail.take();

        match op.caller {
            Caller::Format => {}
            Caller::Open => {
                assert!(!self.opened);
                self.opened = true;
                let replica_index = member_index(
                    &self.working.vsr_state.members,
                    self.working.vsr_state.replica_id,
                )
                .unwrap_or_else(|| unreachable!("working vsr_state carries a known member"));
                self.replica_index = Some(replica_index);
            }
            Caller::Checkpoint | Caller::ViewChange => {
                // read_working_done() installed the staged header into working via a verified
                // quorum; both must carry exactly the state this operation committed to.
                let vsr_state =
                    op.vsr_state.as_ref().unwrap_or_else(|| unreachable!("writer carries state"));
                assert_eq!(&self.staging.vsr_state, vsr_state);
                assert_eq!(&self.working.vsr_state, vsr_state);
            }
        }

        // The next operation in the queue may start now (if any).
        if let Some(tail) = queued {
            self.acquire(storage, tail);
        }

        let event = match op.caller {
            Caller::Format => Event::FormatDone,
            Caller::Open => Event::OpenDone,
            Caller::Checkpoint => Event::CheckpointDone,
            Caller::ViewChange => Event::ViewChangeDone,
        };
        self.events.push_back(event);
    }

    /// Port of `SuperBlock.write_staging`.
    fn write_staging(&mut self, storage: &mut dyn Storage) {
        let caller = self.op_mut().caller;
        assert_ne!(caller, Caller::Open);
        if caller != Caller::Format {
            assert!(self.opened);
        }
        assert!(self.queue_tail.is_none());
        assert!(caller.updates_view_headers());

        // Snapshot the context inputs first (upstream reads context.* freely; here the
        // mutable context borrow must end before mutating working/staging).
        let (vsr_state, view_headers_wires) = {
            let op = self.op_mut();
            assert!(op.copy.is_none());

            let vsr_state =
                op.vsr_state.as_ref().unwrap_or_else(|| unreachable!("writer carries vsr_state"));
            vsr_state.assert_internally_consistent();

            let headers = op
                .view_headers
                .as_ref()
                .unwrap_or_else(|| unreachable!("format/checkpoint/view_change update headers"));
            let view_headers_wires: Vec<[u8; message_header::SIZE]> =
                headers.array.slice().iter().map(message_header::TypedHeader::to_wire).collect();

            (*vsr_state, view_headers_wires)
        };

        self.staging = self.working.clone();
        self.staging.sequence += 1;
        self.staging.parent = self.staging.checksum;
        self.staging.vsr_state = vsr_state;

        let count = view_headers_wires.len();
        self.staging.view_headers_count =
            u32::try_from(count).unwrap_or_else(|_| unreachable!("view_headers_max fits u32"));
        for (slot, wire) in self.staging.view_headers_all.iter_mut().zip(view_headers_wires.iter())
        {
            *slot = *wire;
        }
        for slot in self.staging.view_headers_all.iter_mut().skip(count) {
            *slot = [0; message_header::SIZE];
        }

        self.op_mut().copy = Some(0);
        self.staging.set_checksum();
        self.write_header(storage);
    }

    /// Port of `SuperBlock.write_header`.
    fn write_header(&mut self, storage: &mut dyn Storage) {
        let caller = self.op_mut().caller;

        // We update the working superblock for a checkpoint/format/view_change:
        // open() does not update the working superblock, since it only writes to repair.
        if caller == Caller::Open {
            assert_eq!(self.staging.sequence, self.working.sequence);
        } else {
            assert_eq!(self.staging.sequence, self.working.sequence + 1);
            assert_eq!(self.staging.parent, self.working.checksum);
        }

        // The superblock cluster and replica should never change once formatted:
        assert_eq!(self.staging.cluster, self.working.cluster);
        assert_eq!(self.staging.vsr_state.replica_id, self.working.vsr_state.replica_id);

        let storage_size = self.staging.vsr_state.checkpoint.storage_size;
        assert!(storage_size >= DATA_FILE_SIZE_MIN as u64);
        assert!(storage_size <= STORAGE_SIZE_LIMIT_MAX as u64);

        let copy = self.op_mut().copy.unwrap_or_else(|| unreachable!("copy is set"));
        assert!((copy as usize) < SUPERBLOCK_COPIES);
        self.staging.copy = u16::from(copy);
        // Updating the copy number should not affect the checksum, which was previously set:
        assert!(self.staging.valid_checksum());

        let buffer = self.staging.to_wire().to_vec();
        let offset = SUPERBLOCK_COPY_SIZE as u64 * u64::from(copy);
        assert_bounds(offset, buffer.len());

        storage.write_sectors(WriteRequest {
            zone: Zone::Superblock,
            offset_in_zone: offset,
            buffer,
        });
    }

    fn poll_write(&mut self, storage: &mut dyn Storage, request: WriteRequest) {
        let caller = self.op_mut().caller;
        let copy = self.op_mut().copy.unwrap_or_else(|| unreachable!("write was in flight"));
        assert_eq!(
            request.offset_in_zone,
            SUPERBLOCK_COPY_SIZE as u64 * u64::from(copy),
            "unexpected write completion"
        );
        drop(request.buffer);

        if caller == Caller::Open {
            self.op_mut().copy = None;
            self.repair(storage);
            return;
        }

        if (copy as usize) + 1 == SUPERBLOCK_COPIES {
            self.op_mut().copy = None;
            self.read_working(storage, superblock_quorums::Threshold::Verify);
        } else {
            self.op_mut().copy = Some(copy + 1);
            self.write_header(storage);
        }
    }

    /// Port of `SuperBlock.read_working`.
    fn read_working(
        &mut self,
        storage: &mut dyn Storage,
        threshold: superblock_quorums::Threshold,
    ) {
        {
            let op = self.op_mut();
            assert!(op.copy.is_none());
            assert!(op.read_threshold.is_none());

            // We do not submit reads in parallel, as while this would shave off 1ms, it would
            // also increase the risk that a single fault applies to more reads due to temporal
            // locality. See "An Analysis of Data Corruption in the Storage Stack".
            op.copy = Some(0);
            op.read_threshold = Some(threshold);
        }
        self.reading = core::array::from_fn(|_| header_zeroed());
        self.read_header(storage);
    }

    /// Port of `SuperBlock.read_header`.
    fn read_header(&mut self, storage: &mut dyn Storage) {
        let copy = self.op_mut().copy.unwrap_or_else(|| unreachable!("copy is set"));
        assert!((copy as usize) < SUPERBLOCK_COPIES);
        assert!(self.op_mut().read_threshold.is_some());

        let offset = SUPERBLOCK_COPY_SIZE as u64 * u64::from(copy);
        assert_bounds(offset, SUPERBLOCK_HEADER_SIZE);

        storage.read_sectors(ReadRequest {
            zone: Zone::Superblock,
            offset_in_zone: offset,
            buffer: crate::storage::zeroed_buffer(SUPERBLOCK_HEADER_SIZE),
        });
    }

    fn poll_read(&mut self, storage: &mut dyn Storage, request: ReadRequest) {
        let copy = self.op_mut().copy.unwrap_or_else(|| unreachable!("read was in flight"));
        assert_eq!(
            request.offset_in_zone,
            SUPERBLOCK_COPY_SIZE as u64 * u64::from(copy),
            "unexpected read completion"
        );

        let bytes: [u8; SUPERBLOCK_HEADER_SIZE] = request
            .buffer
            .try_into()
            .unwrap_or_else(|_| unreachable!("read buffers are SUPERBLOCK_HEADER_SIZE bytes"));
        self.reading[usize::from(copy)] = reading_decode(&bytes);

        if usize::from(copy) + 1 != SUPERBLOCK_COPIES {
            self.op_mut().copy = Some(copy + 1);
            self.read_header(storage);
            return;
        }

        let threshold = self
            .op_mut()
            .read_threshold
            .take()
            .unwrap_or_else(|| unreachable!("threshold is set while reading"));
        self.op_mut().copy = None;

        self.read_working_done(storage, threshold);
    }

    /// Port of the second half of `read_header_callback`: quorum decision, installation,
    /// and dispatch to repair/release.
    fn read_working_done(
        &mut self,
        storage: &mut dyn Storage,
        threshold: superblock_quorums::Threshold,
    ) {
        let caller = self.op_mut().caller;
        // True when this read pass verifies completed repairs (upstream: `context.repairs`).
        let verifying_repairs = caller == Caller::Open
            && self
                .queue_head
                .as_ref()
                .unwrap_or_else(|| unreachable!("operation is active"))
                .repairs
                .is_some();

        // DEVIATION: upstream keeps Quorums as a SuperBlock field; the lifetime tying the
        // quorum to the copies slice forces us to extract everything needed while the
        // decision is alive and drop it before mutating working/staging.
        let (working, planned_repairs) = {
            let mut quorums = superblock_quorums::SuperBlockQuorums::default();
            let quorum = match quorums.working(&self.reading, threshold) {
                Ok(quorum) => quorum,
                Err(err) => quorum_error_panic(err),
            };
            assert!(quorum.valid());
            assert!(
                quorum.copies_count()
                    >= usize::from(superblock_quorums::Threshold::count::<SUPERBLOCK_COPIES>(
                        threshold
                    ))
            ); // `copy` may be corrupt.

            let working = quorum.header().clone();

            assert_eq!(
                working.version, SUPERBLOCK_VERSION,
                "cannot read superblock with incompatible version"
            );

            if threshold == superblock_quorums::Threshold::Verify {
                assert_eq!(
                    working.checksum, self.staging.checksum,
                    "superblock failed verification after writing"
                );
                assert!(working.equal(&self.staging));
            }

            if caller == Caller::Format {
                assert_eq!(working.sequence, 1);
                assert_eq!(
                    working.vsr_state.checkpoint.header.checksum,
                    message_header::Prepare::root(working.cluster).checksum
                );
                assert_eq!(working.vsr_state.checkpoint.free_set_blocks_acquired_size, 0);
                assert_eq!(working.vsr_state.checkpoint.free_set_blocks_released_size, 0);
                assert_eq!(working.vsr_state.checkpoint.client_sessions_size, 0);
                assert_eq!(working.vsr_state.checkpoint.storage_size, DATA_FILE_SIZE_MIN as u64);
                assert_eq!(working.vsr_state.checkpoint.header.op, 0);
                assert_eq!(working.vsr_state.commit_max, 0);
                assert_eq!(working.vsr_state.log_view, 0);
                // On reformat view≠0.
                assert_eq!(working.view_headers_count, 1);

                assert!(working.vsr_state.replica_count as usize <= REPLICAS_MAX);
                assert!(
                    member_index(&working.vsr_state.members, working.vsr_state.replica_id)
                        .is_some()
                );
            }

            let planned_repairs = if caller == Caller::Open && !verifying_repairs {
                Some(quorum.repairs())
            } else {
                None
            };
            (working, planned_repairs)
        };

        self.working = working;
        self.staging = self.working.clone();

        // Reset the copies, which may be nonzero due to corruption.
        self.working.copy = 0;
        self.staging.copy = 0;

        assert!(
            self.working.vsr_state.checkpoint.storage_size <= self.storage_size_limit,
            "data file too large size={} > limit={}, restart the replica increasing \
             '--limit-storage'",
            self.working.vsr_state.checkpoint.storage_size,
            self.storage_size_limit
        );

        if caller == Caller::Open {
            if verifying_repairs {
                // We just verified that the repair completed.
                assert_eq!(threshold, superblock_quorums::Threshold::Verify);
                self.release(storage);
            } else {
                assert_eq!(threshold, superblock_quorums::Threshold::Open);
                {
                    let op = self.op_mut();
                    op.repairs =
                        Some(planned_repairs.unwrap_or_else(|| unreachable!("planned above")));
                    op.copy = None;
                }
                self.repair(storage);
            }
        } else {
            self.release(storage);
        }
    }

    /// Port of `SuperBlock.repair`.
    fn repair(&mut self, storage: &mut dyn Storage) {
        assert_eq!(self.op_mut().caller, Caller::Open);
        assert!(self.op_mut().copy.is_none());

        let repair_copy = self
            .op_mut()
            .repairs
            .as_mut()
            .unwrap_or_else(|| unreachable!("repair requires a plan"))
            .next_slot();

        if let Some(repair_copy) = repair_copy {
            self.op_mut().copy = Some(repair_copy);

            self.staging = self.working.clone();
            self.write_header(storage);
        } else {
            self.release(storage);
        }
    }
}

/// Mirrors upstream's per-error panics in `read_header_callback`.
fn quorum_error_panic(error: superblock_quorums::QuorumError) -> ! {
    match error {
        superblock_quorums::QuorumError::Fork => panic!("superblock forked"),
        superblock_quorums::QuorumError::NotFound => panic!("superblock not found"),
        superblock_quorums::QuorumError::QuorumLost => panic!("superblock quorum lost"),
        superblock_quorums::QuorumError::ParentNotConnected => {
            panic!("superblock parent not connected")
        }
        superblock_quorums::QuorumError::ParentSkipped => panic!("superblock parent superseded"),
        superblock_quorums::QuorumError::VSRStateNotMonotonic => {
            panic!("superblock vsr state not monotonic")
        }
    }
}

/// Port of `SuperBlock.assert_bounds`.
fn assert_bounds(offset: u64, size: usize) {
    assert!(offset + size as u64 <= SUPERBLOCK_ZONE_SIZE as u64);
}

#[cfg(test)]
mod superblock_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        Caller, DATA_FILE_SIZE_MIN, Event, FormatOptions, FreeSetReferences, ManifestReferences,
        SUPERBLOCK_COPIES, SUPERBLOCK_COPY_SIZE, SUPERBLOCK_HEADER_SIZE, SUPERBLOCK_ZONE_SIZE,
        SuperBlock, SyncCheckpoint, TrailerReference, UpdateCheckpoint, UpdateViewChange,
    };
    use crate::Zone;
    use crate::message_header::{self, TypedHeader as _};
    use crate::multiversion::Release;
    use crate::storage::{
        Completion, MemoryStorage, ReadRequest, Storage, WriteRequest, zeroed_buffer,
    };
    use tigerbeetle_core::constants::VIEW_HEADERS_MAX;

    const CLUSTER: u128 = 0x00A1_B2C3;

    fn format_options(replica_count: u8) -> FormatOptions {
        FormatOptions {
            cluster: CLUSTER,
            release: Release::MINIMUM,
            replica: 0,
            replica_count,
            view: None,
        }
    }

    /// Drives the storage until no completions remain, then drains events.
    fn poll(sb: &mut SuperBlock, storage: &mut MemoryStorage) -> Vec<Event> {
        sb.poll(storage);
        sb.take_events()
    }

    fn formatted(storage: &mut MemoryStorage, options: FormatOptions) -> SuperBlock {
        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.format(storage, options);
        assert_eq!(poll(&mut sb, storage), vec![Event::FormatDone]);
        assert!(!sb.opened(), "format() must not set opened");
        assert_eq!(sb.replica_index(), Some(0));
        sb
    }

    /// A superblock that has been formatted *and* opened (the fuzzer's steady state).
    fn opened(storage: &mut MemoryStorage, options: FormatOptions) -> SuperBlock {
        let mut sb = formatted(storage, options);
        sb.open(storage);
        assert_eq!(poll(&mut sb, storage), vec![Event::OpenDone]);
        assert!(sb.opened());
        sb
    }

    /// Reads a raw superblock copy straight from disk.
    fn raw_read(storage: &mut MemoryStorage, copy: usize) -> [u8; SUPERBLOCK_HEADER_SIZE] {
        storage.read_sectors(ReadRequest {
            zone: Zone::Superblock,
            offset_in_zone: SUPERBLOCK_COPY_SIZE as u64 * copy as u64,
            buffer: zeroed_buffer(SUPERBLOCK_HEADER_SIZE),
        });
        match storage.next_completion() {
            Some(Completion::Read(request)) => request.buffer.try_into().ok().unwrap(),
            _ => unreachable!("one read in, one completion out"),
        }
    }

    fn raw_write(storage: &mut MemoryStorage, copy: usize, bytes: &[u8; SUPERBLOCK_HEADER_SIZE]) {
        storage.write_sectors(WriteRequest {
            zone: Zone::Superblock,
            offset_in_zone: SUPERBLOCK_COPY_SIZE as u64 * copy as u64,
            buffer: bytes.to_vec(),
        });
        assert!(storage.next_completion().is_some());
    }

    /// Marks every sector of `copy` as a latent sector error (reads zero-fill, so the copy
    /// fails checksum validation — an effective "corrupt copy" for these tests).
    fn corrupt_copy(storage: &mut MemoryStorage, copy: usize) {
        let base = Zone::Superblock.offset(SUPERBLOCK_COPY_SIZE as u64 * copy as u64);
        let sectors = SUPERBLOCK_HEADER_SIZE / tigerbeetle_core::constants::SECTOR_SIZE;
        for i in 0..sectors {
            storage
                .faulty_sectors
                .insert((base / tigerbeetle_core::constants::SECTOR_SIZE as u64) + i as u64);
        }
    }

    /// Port of upstream's format/open smoke flow (simulator-style).
    #[test]
    fn format_then_open_round_trip() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);

        let sb = formatted(&mut storage, format_options(1));

        // Sequential I/O: exactly one write per copy, then a full verification read pass.
        assert_eq!(storage.write_ops, SUPERBLOCK_COPIES as u64);
        assert_eq!(storage.read_ops, SUPERBLOCK_COPIES as u64);

        assert_eq!(sb.working().sequence, 1);
        assert_eq!(sb.working().cluster, CLUSTER);
        assert_eq!(sb.working().view_headers_count, 1);
        assert_eq!(sb.working().copy, 0);

        // Every on-disk copy is valid, self-consistent, and equals the working header.
        for copy in 0..SUPERBLOCK_COPIES {
            let header = super::reading_decode(&raw_read(&mut storage, copy));
            assert!(header.valid_checksum(), "copy {copy} invalid");
            assert_eq!(header.copy as usize, copy);
            assert!(header.equal(sb.working()));
        }
        let (reads_before_open, writes_before_open) = (storage.read_ops, storage.write_ops);

        // Reopen from disk with a fresh instance.
        let mut reopened = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        reopened.open(&mut storage);
        assert_eq!(poll(&mut reopened, &mut storage), vec![Event::OpenDone]);

        assert!(reopened.opened());
        assert_eq!(reopened.replica_index(), Some(0));
        assert_eq!(reopened.working().sequence, 1);
        assert_eq!(reopened.working().checksum, sb.working().checksum);

        // A clean open repairs nothing: one read per copy, zero writes.
        assert_eq!(
            storage.read_ops - reads_before_open,
            SUPERBLOCK_COPIES as u64,
            "open must read every copy"
        );
        assert_eq!(storage.write_ops - writes_before_open, 0, "clean open must not repair");
    }

    #[test]
    fn open_repairs_faulty_copies() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let sb = formatted(&mut storage, format_options(1));
        let working_checksum = sb.working().checksum;
        drop(sb);

        // Two faults: still within the open threshold (2/4).
        corrupt_copy(&mut storage, 1);
        corrupt_copy(&mut storage, 2);

        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.open(&mut storage);
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::OpenDone]);

        assert!(sb.opened());
        assert_eq!(sb.working().sequence, 1);
        assert_eq!(sb.working().checksum, working_checksum);

        // The rewrite cleared the simulated sector faults; read them back.
        storage.faulty_sectors.clear();

        // The faulty copies were rewritten and now match the working header.
        for copy in [1usize, 2] {
            let header = super::reading_decode(&raw_read(&mut storage, copy));
            assert!(header.valid_checksum(), "repaired copy {copy} invalid");
            assert_eq!(header.copy as usize, copy);
            assert!(header.equal(sb.working()));
        }
    }

    #[test]
    #[should_panic(expected = "superblock not found")]
    fn open_panics_on_fresh_storage() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.open(&mut storage);
        poll(&mut sb, &mut storage);
    }

    #[test]
    #[should_panic(expected = "superblock quorum lost")]
    fn open_panics_when_quorum_lost() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let sb = formatted(&mut storage, format_options(1));
        drop(sb);

        // Three faults exceed the open threshold (needs 2/4 valid copies).
        corrupt_copy(&mut storage, 0);
        corrupt_copy(&mut storage, 1);
        corrupt_copy(&mut storage, 2);

        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.open(&mut storage);
        poll(&mut sb, &mut storage);
    }

    /// Two valid quorums with the same cluster/replica/sequence but different checksums:
    /// a fork, which we refuse to resolve.
    #[test]
    #[should_panic(expected = "superblock forked")]
    fn open_panics_on_fork() {
        // Same cluster, same replica slot, but different replica_count → same sequence,
        // same replica_id, different vsr_state → different checksum.
        let mut merged = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut other = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let _a = formatted(&mut merged, format_options(1));
        let _b = formatted(&mut other, format_options(3));

        // Splice copies 2..4 from B into A's image.
        for copy in [2usize, 3] {
            let bytes = raw_read(&mut other, copy);
            raw_write(&mut merged, copy, &bytes);
        }

        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.open(&mut merged);
        poll(&mut sb, &mut merged);
    }

    #[test]
    #[should_panic(expected = "another superblock operation is active")]
    fn operations_are_mutually_exclusive() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.format(&mut storage, format_options(1));
        // The format operation is still in flight (no poll yet): a second caller is refused.
        sb.open(&mut storage);
    }

    /// Mirrors upstream test "SuperBlockHeader" (kept close to its source here because it
    /// exercises set_checksum/valid_checksum through the runtime path too).
    #[test]
    fn header_checksum_ignores_copy_field() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let sb = formatted(&mut storage, format_options(1));

        let mut wire = sb.working().to_wire();
        let mut decoded = super::reading_decode(&wire);
        assert!(decoded.valid_checksum());

        let copies =
            u16::try_from(SUPERBLOCK_COPIES).unwrap_or_else(|_| unreachable!("copies fit"));
        decoded.copy = (decoded.copy + 1) % copies;
        wire[..].copy_from_slice(&decoded.to_wire());
        let redone = super::reading_decode(&wire);
        assert!(redone.valid_checksum(), "copy field must not affect the checksum");
    }

    /// The quorums module drives repair ordering; assert the plan covers all missing slots
    /// (upstream RepairIterator semantics, exercised through the runtime's use of it).
    #[test]
    fn repair_plan_covers_every_missing_slot() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let sb = formatted(&mut storage, format_options(1));
        drop(sb);

        corrupt_copy(&mut storage, 3);

        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.open(&mut storage);
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::OpenDone]);

        storage.faulty_sectors.clear();
        let repaired = super::reading_decode(&raw_read(&mut storage, 3));
        assert!(repaired.valid_checksum());
        assert_eq!(repaired.checksum, sb.working().checksum);
    }

    /// Reads a full on-disk superblock copy (header + padding) straight from disk.
    fn raw_read_full_copy(storage: &mut MemoryStorage, copy: usize) -> Vec<u8> {
        storage.read_sectors(ReadRequest {
            zone: Zone::Superblock,
            offset_in_zone: SUPERBLOCK_COPY_SIZE as u64 * copy as u64,
            buffer: zeroed_buffer(SUPERBLOCK_COPY_SIZE),
        });
        match storage.next_completion() {
            Some(Completion::Read(request)) => request.buffer,
            _ => unreachable!("one read in, one completion out"),
        }
    }

    /// Golden values produced by running upstream's real `SuperBlock.format()` + `open()`
    /// (Zig 0.14.1, test_min config) via `reference/tigerbeetle/src/tbcross_format.zig`.
    ///
    /// Pins the formatted superblock zone byte-for-byte (through region checksums): header
    /// layout, VSRState/checkpoint encoding, root prepare header, checksum chaining, and the
    /// per-copy `checksum` field. Upstream parameters: fixtures.cluster=0, replica=0,
    /// replica_count=6, release minimum, view=null.
    #[test]
    fn format_matches_upstream_zig_golden() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        // Upstream's test Storage allocates through a Zig allocator, which poisons fresh
        // memory with 0xAA in safe builds; never-written superblock padding therefore reads
        // as 0xAA there (the convention replica_format.zig formalizes for checksum
        // comparisons). Replicate it so regions cover the padding too.
        storage.poison_image();
        let sb = formatted(
            &mut storage,
            FormatOptions {
                cluster: 0,
                release: Release::MINIMUM,
                replica: 0,
                replica_count: 6,
                view: None,
            },
        );

        assert_eq!(sb.working().sequence, 1);
        assert_eq!(sb.working().cluster, 0);
        assert_eq!(sb.working().vsr_state.replica_id, crate::root_members(0)[0]);
        assert_eq!(sb.working().checksum, 0xe741_49b8_992b_5101_1bc1_5348_f1b5_dd77);
        assert_eq!(
            tigerbeetle_core::checksum::checksum(&sb.working().vsr_state.to_wire()),
            0x3037_99b6_add2_362c_af2c_353a_f793_f683
        );
        assert_eq!(
            sb.working().vsr_state.checkpoint.header.checksum,
            0x5146_b8d0_e1f6_9ca2_e686_7c42_bb82_63b7
        );

        // Every stored copy (including padding) matches upstream's disk image.
        for (copy, expected) in [
            0x1973_db40_4ead_e7a9_998d_eaf1_33e8_2f3e,
            0xfa6f_a111_10d4_18cd_dc1f_8b4e_e19c_fa65,
            0x8f05_6af4_9265_3ab8_4ff5_192e_c8d8_ad95,
            0xcac7_a8c1_cfac_d517_cdbb_35b7_0aba_2548,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                tigerbeetle_core::checksum::checksum(&raw_read_full_copy(&mut storage, copy)),
                expected,
                "copy {copy} diverges from upstream"
            );
        }

        // And the whole zone as one region:
        let mut whole = Vec::with_capacity(SUPERBLOCK_ZONE_SIZE);
        for copy in 0..SUPERBLOCK_COPIES {
            whole.extend_from_slice(&raw_read_full_copy(&mut storage, copy));
        }
        assert_eq!(
            tigerbeetle_core::checksum::checksum(&whole),
            0x4a8b_265e_05af_09a5_1bae_99d0_b631_d601
        );
    }

    /// Cross-validation of a full checkpoint transition against real upstream output
    /// (`reference/tigerbeetle/src/tbcross_format.zig`, checkpoint phase): format → open →
    /// checkpoint with the same synthetic update, then compare the resulting disk image.
    #[test]
    fn checkpoint_matches_upstream_zig_golden() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        // See `format_matches_upstream_zig_golden` for why the image is poisoned first.
        storage.poison_image();

        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.format(
            &mut storage,
            FormatOptions {
                cluster: 0,
                release: Release::MINIMUM,
                replica: 0,
                replica_count: 6,
                view: None,
            },
        );
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::FormatDone]);
        sb.open(&mut storage);
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::OpenDone]);

        // The synthetic update must match the harness byte-for-byte; upstream prints the new
        // commit_min header's checksum in its checkpoint log line.
        let op = crate::checkpoint::checkpoint_after(0);
        assert_eq!(op, 19);
        let header = prepare_at_op(op, 0);
        assert_eq!(header.checksum, 0x58bb_86d5_5b14_a17c_17a4_86f8_f486_4cb4);

        sb.checkpoint(&mut storage, &checkpoint_update(header));
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::CheckpointDone]);

        assert_eq!(sb.working().sequence, 2);
        assert_eq!(sb.working().vsr_state.checkpoint.header.op, op);
        assert_eq!(sb.working().checksum, 0xc720_ae02_eddf_2d37_328c_7a69_e3aa_930b);
        assert_eq!(
            tigerbeetle_core::checksum::checksum(&sb.working().vsr_state.to_wire()),
            0xd6a4_150f_4475_b794_aaef_27fd_af01_7914
        );

        for (copy, expected) in [
            0xc844_2b67_2e8f_a9b0_8168_0ee6_eb0c_50c1,
            0x3496_2a61_c9ac_c425_3c74_ff72_cb22_931a,
            0x0523_9950_ac9a_b112_1486_569c_61a6_9253,
            0x6426_74d4_98ee_a627_d011_b675_6236_d4e7,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                tigerbeetle_core::checksum::checksum(&raw_read_full_copy(&mut storage, copy)),
                expected,
                "copy {copy} diverges from upstream"
            );
        }

        let mut whole = Vec::with_capacity(SUPERBLOCK_ZONE_SIZE);
        for copy in 0..SUPERBLOCK_COPIES {
            whole.extend_from_slice(&raw_read_full_copy(&mut storage, copy));
        }
        assert_eq!(
            tigerbeetle_core::checksum::checksum(&whole),
            0x7708_a02a_79a9_2bff_e79a_1cf4_ff0b_fa8b
        );
    }

    /// Cross-validation of a sync `view_change` against real upstream output
    /// (`tbcross_format.zig`, sync phase): replays format → open → checkpoint → view_change
    /// with a `sync_checkpoint` replacing the whole checkpoint state (op 19 → 39).
    #[test]
    fn view_change_sync_checkpoint_matches_upstream_zig_golden() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        // See `format_matches_upstream_zig_golden` for why the image is poisoned first.
        storage.poison_image();

        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.format(
            &mut storage,
            FormatOptions {
                cluster: 0,
                release: Release::MINIMUM,
                replica: 0,
                replica_count: 6,
                view: None,
            },
        );
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::FormatDone]);
        sb.open(&mut storage);
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::OpenDone]);

        // Phase 2 (matches `checkpoint_matches_upstream_zig_golden`): commit_min → op 19.
        let op_checkpoint = crate::checkpoint::checkpoint_after(0);
        sb.checkpoint(&mut storage, &checkpoint_update(prepare_at_op(op_checkpoint, 0)));
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::CheckpointDone]);

        // Phase 3: sync view_change jumps the checkpoint to op 39 and advances the view.
        let op_sync = crate::checkpoint::checkpoint_after(op_checkpoint);
        assert_eq!(op_sync, 39);

        let mut sync_checkpoint_state = super::checkpoint_state_zeroed();
        sync_checkpoint_state.header = prepare_at_op(op_sync, 0);
        sync_checkpoint_state.parent_checkpoint_id = sb.staging().checkpoint_id();
        sync_checkpoint_state.free_set_blocks_acquired_checksum =
            message_header::checksum_body_empty();
        sync_checkpoint_state.free_set_blocks_released_checksum =
            message_header::checksum_body_empty();
        sync_checkpoint_state.client_sessions_checksum = message_header::checksum_body_empty();
        sync_checkpoint_state.storage_size = DATA_FILE_SIZE_MIN as u64;
        sync_checkpoint_state.release = Release::MINIMUM;

        let head = prepare_at_op(op_sync + 1, 1);
        sb.view_change(
            &mut storage,
            UpdateViewChange {
                commit_max: op_sync,
                log_view: 1,
                view: 1,
                headers: crate::ViewChangeHeadersArray::init(
                    crate::ViewChangeCommand::View,
                    &[head],
                ),
                sync_checkpoint: Some(SyncCheckpoint {
                    checkpoint: sync_checkpoint_state,
                    sync_op_min: 0,
                    sync_op_max: op_sync,
                }),
            },
        );
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::ViewChangeDone]);

        assert_eq!(sb.working().sequence, 3);
        assert_eq!(sb.working().checksum, 0x3352_08db_27bd_d33e_6dc3_ffc6_5f47_77cb);
        assert_eq!(
            tigerbeetle_core::checksum::checksum(&sb.working().vsr_state.to_wire()),
            0x015d_bc2f_6ad6_eea3_db9c_fc2d_50ca_2c73
        );

        for (copy, expected) in [
            0x9aeb_f596_f641_853f_5273_c14d_e340_efa5,
            0xa26a_a7b9_92ce_04ee_3aa4_e035_0a0f_4899,
            0x5f2b_22b0_1ce0_bf7d_8ae7_b174_6bee_42b7,
            0x561f_de1a_dfbf_613a_0438_2112_9259_ead0,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                tigerbeetle_core::checksum::checksum(&raw_read_full_copy(&mut storage, copy)),
                expected,
                "copy {copy} diverges from upstream"
            );
        }

        let mut whole = Vec::with_capacity(SUPERBLOCK_ZONE_SIZE);
        for copy in 0..SUPERBLOCK_COPIES {
            whole.extend_from_slice(&raw_read_full_copy(&mut storage, copy));
        }
        assert_eq!(
            tigerbeetle_core::checksum::checksum(&whole),
            0xbdbb_b309_cd1a_15b7_827b_50b6_112a_368a
        );
    }

    // ---------------------------------------------------------------------------------------------
    // checkpoint / view_change (deterministic ports of `superblock_fuzz.zig` scenarios)
    // ---------------------------------------------------------------------------------------------

    /// A valid prepare at (`op`, `view`) with a non-reserved operation and a real release
    /// (the root operation itself is pinned to op=0, so the fuzzer uses a normal op here).
    fn prepare_at_op(op: u64, view: u32) -> message_header::Prepare {
        let mut header = message_header::Prepare {
            release: Release::MINIMUM,
            operation: crate::Operation(tigerbeetle_core::constants::VSR_OPERATIONS_RESERVED),
            client: 1,
            request: 1,
            timestamp: 1,
            ..message_header::Prepare::default()
        };
        header.op = op;
        header.view = view;
        header.set_checksum_body(&[]);
        header.set_checksum();
        header
    }

    /// Upstream: an empty trailer reference (`TrailerReference.empty() == true`).
    fn empty_reference() -> TrailerReference {
        TrailerReference {
            checksum: message_header::checksum_body_empty(),
            last_block_address: 0,
            last_block_checksum: 0,
            trailer_size: 0,
        }
    }

    /// The minimal valid checkpoint update: advances commit_min to `header.op` and keeps
    /// everything else (storage size, free set, sessions, manifest, release) unchanged —
    /// exactly what a fresh replica's first checkpoint looks like in the fuzzer.
    fn checkpoint_update(header: message_header::Prepare) -> UpdateCheckpoint {
        UpdateCheckpoint {
            header,
            view_attributes: None,
            commit_max: header.op,
            sync_op_min: 0,
            sync_op_max: 0,
            manifest_references: ManifestReferences {
                oldest_checksum: 0,
                oldest_address: 0,
                newest_checksum: 0,
                newest_address: 0,
                block_count: 0,
            },
            free_set_references: FreeSetReferences {
                blocks_acquired: empty_reference(),
                blocks_released: empty_reference(),
            },
            client_sessions_reference: empty_reference(),
            storage_size: DATA_FILE_SIZE_MIN as u64,
            release: Release::MINIMUM,
        }
    }

    /// Port of the fuzzer's format → open → checkpoint flow: the checkpoint installs
    /// commit_min durably across all copies and survives reopen.
    #[test]
    fn checkpoint_advances_commit_min_and_persists() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        let op = crate::checkpoint::checkpoint_after(0);
        let parent_checkpoint_id = sb.working().checkpoint_id();
        sb.checkpoint(&mut storage, &checkpoint_update(prepare_at_op(op, 0)));

        // While in flight, updating() reports only the running operation.
        assert!(sb.updating(Caller::Checkpoint));
        assert!(!sb.updating(Caller::ViewChange));
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::CheckpointDone]);
        assert!(!sb.updating(Caller::Checkpoint));

        let working = sb.working();
        assert_eq!(working.sequence, 2);
        assert_eq!(working.vsr_state.commit_max, op);
        assert_eq!(working.vsr_state.sync_op_min, 0);
        assert_eq!(working.vsr_state.sync_op_max, 0);
        assert_eq!(working.vsr_state.checkpoint.header.op, op);
        assert_eq!(working.vsr_state.checkpoint.header.checksum, prepare_at_op(op, 0).checksum);
        // The new checkpoint chains onto the previously installed one.
        assert_eq!(working.vsr_state.checkpoint.parent_checkpoint_id, parent_checkpoint_id);
        // The grandparent is whatever the previous checkpoint pointed at (zero here).
        assert_eq!(working.vsr_state.checkpoint.grandparent_checkpoint_id, 0);

        // Every on-disk copy carries the staged state; verification passed on completion.
        for copy in 0..SUPERBLOCK_COPIES {
            let header = super::reading_decode(&raw_read(&mut storage, copy));
            assert!(header.valid_checksum(), "copy {copy} invalid after checkpoint");
            assert_eq!(header.sequence, 2);
            assert!(header.equal(sb.working()));
        }

        let working_checksum = sb.working().checksum;
        drop(sb);

        // Reopen from disk: the checkpoint survives.
        let mut reopened = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        reopened.open(&mut storage);
        assert_eq!(poll(&mut reopened, &mut storage), vec![Event::OpenDone]);
        assert_eq!(reopened.working().checksum, working_checksum);
        assert_eq!(reopened.working().vsr_state.checkpoint.header.op, op);
        assert_eq!(reopened.working().vsr_state.commit_max, op);
    }

    /// Port of the fuzzer's format → open → view_change flow (.view headers): the view is
    /// persisted and survives reopen.
    #[test]
    fn view_change_persists_view() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        let head = prepare_at_op(1, 1);
        let headers = crate::ViewChangeHeadersArray::init(crate::ViewChangeCommand::View, &[head]);
        sb.view_change(
            &mut storage,
            UpdateViewChange {
                commit_max: 0,
                log_view: 1,
                view: 1,
                headers,
                sync_checkpoint: None,
            },
        );

        assert!(sb.updating(Caller::ViewChange));
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::ViewChangeDone]);
        assert!(!sb.updating(Caller::ViewChange));

        let working = sb.working();
        assert_eq!(working.sequence, 2);
        assert_eq!(working.vsr_state.log_view, 1);
        assert_eq!(working.vsr_state.view, 1);
        // Checkpoint state is untouched by a plain view change.
        assert_eq!(working.vsr_state.checkpoint.header.op, 0);

        for copy in 0..SUPERBLOCK_COPIES {
            let header = super::reading_decode(&raw_read(&mut storage, copy));
            assert!(header.valid_checksum());
            assert!(header.equal(sb.working()));
        }

        let working_checksum = sb.working().checksum;
        drop(sb);

        let mut reopened = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        reopened.open(&mut storage);
        assert_eq!(poll(&mut reopened, &mut storage), vec![Event::OpenDone]);
        assert_eq!(reopened.working().checksum, working_checksum);
        assert_eq!(reopened.working().vsr_state.view, 1);
    }

    /// The .join_view variant: log_view < view, JV headers installed verbatim.
    #[test]
    fn view_change_join_view_persists_log_view_and_view() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        let head = prepare_at_op(1, 0);
        let headers =
            crate::ViewChangeHeadersArray::init(crate::ViewChangeCommand::JoinView, &[head]);
        sb.view_change(
            &mut storage,
            UpdateViewChange {
                commit_max: 0,
                log_view: 0,
                view: 1,
                headers,
                sync_checkpoint: None,
            },
        );
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::ViewChangeDone]);

        assert_eq!(sb.working().vsr_state.log_view, 0);
        assert_eq!(sb.working().vsr_state.view, 1);

        // Reopened, log_view < view means the stored headers read back as JoinView.
        let working_checksum = sb.working().checksum;
        drop(sb);

        let mut reopened = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        reopened.open(&mut storage);
        assert_eq!(poll(&mut reopened, &mut storage), vec![Event::OpenDone]);
        assert_eq!(reopened.working().checksum, working_checksum);
        let mut decoded = [message_header::Prepare::default(); VIEW_HEADERS_MAX as usize];
        let stored_headers = reopened.working().view_headers(&mut decoded);
        assert_eq!(stored_headers.command, crate::ViewChangeCommand::JoinView);
        assert_eq!(stored_headers.slice[0].op, 1);
    }

    /// checkpoint ↔ view_change are allowed to queue behind each other (one deep), and run
    /// strictly in queue order. Ports the fuzzer's interleaved scenario.
    #[test]
    fn checkpoint_and_view_change_may_queue_in_either_order() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        let op1 = crate::checkpoint::checkpoint_after(0);
        // checkpoint starts; view_change queues behind it.
        sb.checkpoint(&mut storage, &checkpoint_update(prepare_at_op(op1, 0)));
        let vc_head = prepare_at_op(op1 + 1, 1);
        sb.view_change(
            &mut storage,
            UpdateViewChange {
                commit_max: op1,
                log_view: 1,
                view: 1,
                headers: crate::ViewChangeHeadersArray::init(
                    crate::ViewChangeCommand::View,
                    &[vc_head],
                ),
                sync_checkpoint: None,
            },
        );
        assert!(sb.updating(Caller::Checkpoint));
        assert!(sb.updating(Caller::ViewChange));

        assert_eq!(poll(&mut sb, &mut storage), vec![Event::CheckpointDone, Event::ViewChangeDone]);
        assert!(!sb.updating(Caller::Checkpoint));
        assert!(!sb.updating(Caller::ViewChange));
        assert_eq!(sb.working().sequence, 3);
        assert_eq!(sb.working().vsr_state.checkpoint.header.op, op1);
        assert_eq!(sb.working().vsr_state.view, 1);

        // Reverse order: view_change runs while checkpoint queues behind it.
        let op2 = crate::checkpoint::checkpoint_after(op1);
        let head_second = prepare_at_op(op2 + 1, 2);
        sb.view_change(
            &mut storage,
            UpdateViewChange {
                commit_max: op2,
                log_view: 2,
                view: 2,
                headers: crate::ViewChangeHeadersArray::init(
                    crate::ViewChangeCommand::View,
                    &[head_second],
                ),
                sync_checkpoint: None,
            },
        );
        sb.checkpoint(&mut storage, &checkpoint_update(prepare_at_op(op2, 2)));
        assert!(sb.updating(Caller::ViewChange));
        assert!(sb.updating(Caller::Checkpoint));

        assert_eq!(poll(&mut sb, &mut storage), vec![Event::ViewChangeDone, Event::CheckpointDone]);
        assert_eq!(sb.working().sequence, 5);
        assert_eq!(sb.working().vsr_state.checkpoint.header.op, op2);
        assert_eq!(sb.working().vsr_state.view, 2);
    }

    /// An operation may not queue behind itself (upstream:
    /// `assert(head.caller != context.caller)`).
    #[test]
    #[should_panic(expected = "mutually exclusive with themselves")]
    fn queue_rejects_same_caller_twice() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        let op1 = crate::checkpoint::checkpoint_after(0);
        sb.checkpoint(&mut storage, &checkpoint_update(prepare_at_op(op1, 0)));
        let op2 = crate::checkpoint::checkpoint_after(op1);
        sb.checkpoint(&mut storage, &checkpoint_update(prepare_at_op(op2, 0)));
    }

    /// A `view_change` carrying a `sync_checkpoint` replaces the whole checkpoint state: the
    /// fuzzer's sync scenario. The checkpoint jumps to the next checkpoint op and installs
    /// sync_op_min/max.
    #[test]
    fn view_change_sync_checkpoint_installs_state() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        // The synced checkpoint is exactly one checkpoint interval ahead of the current one.
        let checkpoint_next = crate::checkpoint::checkpoint_after(0);
        let mut sync_checkpoint_state = super::checkpoint_state_zeroed();
        sync_checkpoint_state.header = prepare_at_op(checkpoint_next, 0);
        sync_checkpoint_state.parent_checkpoint_id = sb.working().checkpoint_id();
        sync_checkpoint_state.free_set_blocks_acquired_checksum =
            message_header::checksum_body_empty();
        sync_checkpoint_state.free_set_blocks_released_checksum =
            message_header::checksum_body_empty();
        sync_checkpoint_state.client_sessions_checksum = message_header::checksum_body_empty();
        sync_checkpoint_state.storage_size = DATA_FILE_SIZE_MIN as u64;
        sync_checkpoint_state.release = Release::MINIMUM;

        let head = prepare_at_op(checkpoint_next + 1, 1);
        sb.view_change(
            &mut storage,
            UpdateViewChange {
                commit_max: checkpoint_next,
                log_view: 1,
                view: 1,
                headers: crate::ViewChangeHeadersArray::init(
                    crate::ViewChangeCommand::View,
                    &[head],
                ),
                sync_checkpoint: Some(SyncCheckpoint {
                    checkpoint: sync_checkpoint_state,
                    sync_op_min: 0,
                    sync_op_max: checkpoint_next,
                }),
            },
        );
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::ViewChangeDone]);

        let working = sb.working();
        assert_eq!(working.sequence, 2);
        assert_eq!(working.vsr_state.checkpoint.header.op, checkpoint_next);
        assert_eq!(working.vsr_state.sync_op_min, 0);
        assert_eq!(working.vsr_state.sync_op_max, checkpoint_next);
        assert_eq!(working.vsr_state.commit_max, checkpoint_next);
        assert_eq!(working.vsr_state.log_view, 1);
        assert_eq!(working.vsr_state.view, 1);

        for copy in 0..SUPERBLOCK_COPIES {
            let header = super::reading_decode(&raw_read(&mut storage, copy));
            assert!(header.valid_checksum());
            assert!(header.equal(sb.working()));
        }

        let working_checksum = sb.working().checksum;
        drop(sb);

        let mut reopened = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        reopened.open(&mut storage);
        assert_eq!(poll(&mut reopened, &mut storage), vec![Event::OpenDone]);
        assert_eq!(reopened.working().checksum, working_checksum);
        assert_eq!(reopened.working().vsr_state.checkpoint.header.op, checkpoint_next);
    }

    /// A sync checkpoint at the next checkpoint op must chain onto *this* superblock's
    /// checkpoint (upstream asserts `parent_checkpoint_id == staging.checkpoint_id()`).
    #[test]
    #[should_panic(expected = "sync checkpoint parent mismatch")]
    fn view_change_sync_checkpoint_rejects_wrong_parent() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        let checkpoint_next = crate::checkpoint::checkpoint_after(0);
        let mut sync_checkpoint_state = super::checkpoint_state_zeroed();
        sync_checkpoint_state.header = prepare_at_op(checkpoint_next, 0);
        sync_checkpoint_state.parent_checkpoint_id = 0xdead_beef; // wrong
        sync_checkpoint_state.free_set_blocks_acquired_checksum =
            message_header::checksum_body_empty();
        sync_checkpoint_state.free_set_blocks_released_checksum =
            message_header::checksum_body_empty();
        sync_checkpoint_state.client_sessions_checksum = message_header::checksum_body_empty();
        sync_checkpoint_state.storage_size = DATA_FILE_SIZE_MIN as u64;
        sync_checkpoint_state.release = Release::MINIMUM;

        let head = prepare_at_op(checkpoint_next + 1, 1);
        sb.view_change(
            &mut storage,
            UpdateViewChange {
                commit_max: checkpoint_next,
                log_view: 1,
                view: 1,
                headers: crate::ViewChangeHeadersArray::init(
                    crate::ViewChangeCommand::View,
                    &[head],
                ),
                sync_checkpoint: Some(SyncCheckpoint {
                    checkpoint: sync_checkpoint_state,
                    sync_op_min: 0,
                    sync_op_max: checkpoint_next,
                }),
            },
        );
    }

    /// A corrupted copy after a checkpoint is repaired by the next open.
    #[test]
    fn open_after_checkpoint_repairs_corrupted_copy() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let mut sb = opened(&mut storage, format_options(1));

        let op = crate::checkpoint::checkpoint_after(0);
        sb.checkpoint(&mut storage, &checkpoint_update(prepare_at_op(op, 0)));
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::CheckpointDone]);

        let working_checksum = sb.working().checksum;
        drop(sb);

        corrupt_copy(&mut storage, 1);

        let mut sb = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.open(&mut storage);
        assert_eq!(poll(&mut sb, &mut storage), vec![Event::OpenDone]);

        assert_eq!(sb.working().sequence, 2);
        assert_eq!(sb.working().checksum, working_checksum);
        assert_eq!(sb.working().vsr_state.checkpoint.header.op, op);

        storage.faulty_sectors.clear();
        let header = super::reading_decode(&raw_read(&mut storage, 1));
        assert!(header.valid_checksum(), "repaired copy invalid");
        assert!(header.equal(sb.working()));
    }
}
