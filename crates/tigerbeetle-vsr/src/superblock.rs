//! Superblock sizing constants (upstream: `src/vsr/superblock.zig` file-level constants).
//!
//! The `SuperBlockHeader`/`VSRState`/`CheckpointState` structs land in a follow-up port; the
//! sizes here are pinned to upstream's `@sizeOf` results and asserted against them.

use tigerbeetle_core::constants::{
    CLIENT_REPLIES_SIZE, CLIENTS_MAX, HEADER_SIZE, JOURNAL_SIZE, PIPELINE_PREPARE_QUEUE_MAX,
    SECTOR_SIZE, SUPERBLOCK_COPIES, VIEW_HEADERS_MAX,
};
use tigerbeetle_core::stdx::align_forward;

/// Port of `superblock.SuperBlockVersion`.
///
/// DEVIATION: upstream selects `0` for development builds (release == minimum) and `2` for
/// production releases, based on build-injected config (`config.process.release`) which this
/// port does not model yet (`crates/tigerbeetle-core/src/config.rs`). We are a development
/// build, so the version is 0.
pub const SUPERBLOCK_VERSION: u16 = 0;

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
// TODO(port): src/vsr/superblock.zig SuperBlockHeader.view_headers() — needs
// Headers.ViewChangeSlice.
// ---------------------------------------------------------------------------

use crate::message_header;
use crate::message_header::TypedHeader as _;
use crate::multiversion::Release;
use crate::{Members, member_index};
use tigerbeetle_core::constants::{BLOCK_SIZE, REPLICAS_MAX};

pub use tigerbeetle_core::constants::CHECKPOINT_STATE_SIZE;

/// Port of `vsr.checksum(&.{})`: the checksum of an empty body.
use message_header::checksum_body_empty;

/// Port of `vsr.ClientSessions.encode_size`
/// (TODO(port): src/vsr/client_sessions.zig:80 — the ClientSessions codec itself).
///
/// Layout: vsr headers for all clients, then one u64 session per client; the leading
/// alignment is trivially satisfied.
pub const CLIENT_SESSIONS_ENCODE_SIZE: usize =
    (message_header::SIZE + size_of::<u64>()) * CLIENTS_MAX as usize;

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

        Self {
            header: message_header::Prepare::from_wire(
                &bytes[Self::OFFSET_HEADER..Self::OFFSET_HEADER + message_header::SIZE]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            )
            .unwrap_or_else(|| unreachable!("stored prepare header must be valid")),
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
}
