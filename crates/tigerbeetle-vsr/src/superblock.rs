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
