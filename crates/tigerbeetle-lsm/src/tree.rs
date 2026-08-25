//! An LSM tree: constants and configuration shared by the tree, its manifest, and the
//! forest that owns them.
//!
//! Upstream: `src/lsm/tree.zig` (the standalone items only; `TreeType` itself depends on
//! the grid/Storage layer and is ported later).
//!
//! DEVIATION: upstream `TreeConfig.name` is a runtime `[]const u8`; here it is
//! `&'static str`, as every upstream caller passes a static literal.

/// Upstream `ScopeCloseMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScopeCloseMode {
    Persist,
    Discard,
}

/// We reserve maxInt(u64) to indicate that a table has not been deleted.
/// Tables that have not been deleted have snapshot_max of maxInt(u64).
/// Since we ensure and assert that a query snapshot never exactly matches
/// the snapshot_min/snapshot_max of a table, we must use maxInt(u64) - 1
/// to query all non-deleted tables.
pub const SNAPSHOT_LATEST: u64 = u64::MAX - 1;

/// The maximum number of tables for a single tree.
// DEVIATION: `u32::from` is not yet callable in constants; the widening cast is exact.
pub const TABLE_COUNT_MAX: u32 = table_count_max_for_tree(
    tigerbeetle_core::constants::LSM_GROWTH_FACTOR,
    tigerbeetle_core::constants::LSM_LEVELS as u32,
);

/// Upstream `TreeConfig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeConfig {
    /// Unique (stable) identifier, across all trees in the forest.
    pub id: u16,
    /// Human-readable tree name for logging.
    pub name: &'static str,
}

/// The total number of tables that can be supported by the tree across so many levels.
///
/// # Panics
/// Panics if the parameters are outside the supported bounds (upstream asserts), or on
/// overflow of the per-level table count (upstream `math.pow` asserts likewise).
#[must_use]
pub const fn table_count_max_for_tree(growth_factor: u32, levels_count: u32) -> u32 {
    assert!(growth_factor >= 4);
    assert!(growth_factor <= 16); // Limit excessive write amplification.
    assert!(levels_count >= 2);
    assert!(levels_count <= 10); // Limit excessive read amplification.
    assert!(levels_count <= tigerbeetle_core::constants::LSM_LEVELS as u32);

    let mut count: u32 = 0;
    let mut level: u32 = 0;
    while level < levels_count {
        match table_count_max_for_level(growth_factor, level).checked_add(count) {
            Some(sum) => count = sum,
            None => panic!("table_count_max_for_level overflow"),
        }
        level += 1;
    }
    count
}

/// The total number of tables that can be supported by the level alone.
///
/// # Panics
/// Panics if `level` is out of range (upstream asserts) or on overflow (upstream
/// `math.pow` asserts likewise).
#[must_use]
pub const fn table_count_max_for_level(growth_factor: u32, level: u32) -> u32 {
    assert!(level < tigerbeetle_core::constants::LSM_LEVELS as u32);

    match growth_factor.checked_pow(level + 1) {
        Some(count) => count,
        None => panic!("growth_factor^(level+1) overflows u32"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SNAPSHOT_LATEST, TABLE_COUNT_MAX, table_count_max_for_level, table_count_max_for_tree,
    };

    #[test]
    fn table_count_max_for_level_and_tree() {
        // Port of upstream test "table_count_max_for_level/tree".
        assert_eq!(table_count_max_for_level(8, 0), 8);
        assert_eq!(table_count_max_for_level(8, 1), 64);
        assert_eq!(table_count_max_for_level(8, 2), 512);
        assert_eq!(table_count_max_for_level(8, 3), 4096);
        assert_eq!(table_count_max_for_level(8, 4), 32768);
        assert_eq!(table_count_max_for_level(8, 5), 262_144);
        assert_eq!(table_count_max_for_level(8, 6), 2_097_152);

        assert_eq!(table_count_max_for_tree(8, 2), 8 + 64);
        assert_eq!(table_count_max_for_tree(8, 3), 72 + 512);
        assert_eq!(table_count_max_for_tree(8, 4), 584 + 4_096);
        assert_eq!(table_count_max_for_tree(8, 5), 4_680 + 32_768);
        assert_eq!(table_count_max_for_tree(8, 6), 37448 + 262_144);
        assert_eq!(table_count_max_for_tree(8, 7), 299_592 + 2_097_152);
    }

    #[test]
    fn constants_match_upstream_defaults() {
        assert_eq!(SNAPSHOT_LATEST, u64::MAX - 1);
        // table_count_max derives from the active CONFIG, like upstream
        // (`lsm_growth_factor`: 4 under test_min → 21_844, 8 in production → 2_396_744).
        let expected: u32 = (0..u32::from(tigerbeetle_core::constants::LSM_LEVELS))
            .map(|level| tigerbeetle_core::constants::LSM_GROWTH_FACTOR.pow(level + 1))
            .sum();
        assert_eq!(TABLE_COUNT_MAX, expected);
    }
}
