//! Compaction helpers: pure functions with no I/O dependencies.
//!
//! Port of standalone functions from `src/lsm/compaction.zig` (lines 2310–2388).
//! The main [`Compaction`](crate) struct lives in `tigerbeetle-vsr/src/compaction.rs`.

#![allow(clippy::cast_possible_truncation)] // constants fit their target types

use tigerbeetle_core::constants;

/// The upper-bound count of input tables to a single tree's compaction.
///
/// - +1 from level A.
/// - +lsm_growth_factor from level B.
pub const COMPACTION_TABLES_INPUT_MAX: usize = 1 + constants::LSM_GROWTH_FACTOR as usize;

/// The upper-bound count of output tables from a single tree's compaction.
/// In the "worst" case, no keys are overwritten/merged, and no tombstones are dropped.
pub const COMPACTION_TABLES_OUTPUT_MAX: usize = COMPACTION_TABLES_INPUT_MAX;

/// The minimum number of blocks required for a single beat of a single compaction.
///
/// Compaction needs to carry over the output index block and all input blocks to the next beat:
/// One index and one value block for the output table, one index block for level A, two index
/// blocks for level B (to allow prefetching), and `LSM_COMPACTION_QUEUE_READ_MAX` value blocks
/// for the two input tables.
pub const COMPACTION_BLOCK_COUNT_BEAT_MIN: u32 =
    (1 + 1) + (1 + 2) + constants::LSM_COMPACTION_QUEUE_READ_MAX as u32;

/// Number of beats in a half-bar.
pub const HALF_BAR_BEAT_COUNT: usize = constants::LSM_COMPACTION_OPS / 2;

/// `snapshot_max` for the input tables of a compaction with the given `op_min`.
///
/// After compaction finishes, input tables are given this `snapshot_max` so they become
/// invisible to subsequent read transactions.
#[must_use]
pub fn snapshot_max_for_table_input(op_min: u64) -> u64 {
    snapshot_min_for_table_output(op_min) - 1
}

/// `snapshot_min` for the output tables of a compaction with the given `op_min`.
///
/// # Panics
/// Panics if `op_min` is zero or not aligned to `HALF_BAR_BEAT_COUNT`.
#[must_use]
pub fn snapshot_min_for_table_output(op_min: u64) -> u64 {
    assert!(op_min > 0);
    assert_eq!(op_min as usize % HALF_BAR_BEAT_COUNT, 0);
    op_min + HALF_BAR_BEAT_COUNT as u64
}

/// Returns the first op of the compaction (`Compaction.op_min`) for a given op/beat.
///
/// After this compaction finishes:
/// - `op_min + half_bar_beat_count - 1` will be the input tables' `snapshot_max`.
/// - `op_min + half_bar_beat_count` will be the output tables' `snapshot_min`.
///
/// Each half-bar has a separate `op_min` (for deriving the output `snapshot_min`) instead of
/// each full bar because this allows the output tables of the first half-bar's compaction to
/// be prefetched against earlier — hopefully while they are still warm in the cache from
/// being written.
///
/// # Panics
/// Panics if `op < HALF_BAR_BEAT_COUNT`.
#[must_use]
pub fn compaction_op_min(op: u64) -> u64 {
    assert!(op >= HALF_BAR_BEAT_COUNT as u64);
    op - op % HALF_BAR_BEAT_COUNT as u64
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn compaction_op_min_asserts() {
        // HALF_BAR_BEAT_COUNT = 2 (test-min config: LSM_COMPACTION_OPS = 4)
        assert_eq!(compaction_op_min(2), 2);
        assert_eq!(compaction_op_min(3), 2);
        assert_eq!(compaction_op_min(4), 4);
        assert_eq!(compaction_op_min(5), 4);
        assert_eq!(compaction_op_min(6), 6);
        assert_eq!(compaction_op_min(7), 6);
    }

    #[test]
    fn snapshot_min_max_round_trip() {
        let op_min = 4u64;
        let snapshot_min = snapshot_min_for_table_output(op_min);
        let snapshot_max = snapshot_max_for_table_input(op_min);
        assert_eq!(snapshot_min, 6);
        assert_eq!(snapshot_max, 5);
        assert_eq!(snapshot_max + 1, snapshot_min);
    }
}
