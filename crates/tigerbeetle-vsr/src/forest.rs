// =============================================================================
// Forest — owns all LSM grooves for a replica
// =============================================================================
//
// Ported from `src/lsm/forest.zig`. The Zig version uses comptime codegen to
// auto-generate a `Grooves` struct from a config. In Rust, we define the fields
// explicitly since there are only three grooves (accounts + transfers +
// transfers_pending).

use crate::grid::{Grid, GridOpenReferences, GridOptions, SuperBlockView};
use crate::groove::{
    AccountGroove, AccountGrooveScratch, TransferGroove, TransferGrooveScratch,
    TransferPendingGroove, TransferPendingGrooveScratch,
};
use crate::manifest_log::{ManifestLog, Pace};
use crate::storage::Storage;
use tigerbeetle_core::constants::{CONFIG, LSM_GROWTH_FACTOR, LSM_LEVELS};
use tigerbeetle_lsm::manifest::ManifestLog as ManifestLogTrait;
use tigerbeetle_lsm::tree::table_count_max_for_tree;

/// Number of trees in the forest: 9 (account) + 14 (transfer) + 2 (pending).
/// Used to size the manifest log's compaction pace (upstream `tree_infos.len`).
pub const TREE_COUNT: u32 = 25;

/// The manifest log's compaction pace, sized for the forest's tree count (upstream
/// `forest.manifest_log_compaction_pace`, forest.zig:195).
fn forest_pace() -> Pace {
    // Upstream: `constants.lsm_manifest_compact_extra_blocks`; fits u32 like the comptime value.
    #[allow(clippy::cast_possible_truncation)]
    let compact_extra_blocks = CONFIG.cluster.lsm_manifest_compact_extra_blocks as u32;
    Pace::init(
        TREE_COUNT,
        table_count_max_for_tree(LSM_GROWTH_FACTOR, u32::from(LSM_LEVELS)),
        compact_extra_blocks,
    )
}

/// In-flight open lifecycle, tracked so `Forest::open` completes exactly once.
struct ForestProgress {
    /// The op the grooves resume from after this open (upstream passes the checkpoint op
    /// into `open_complete`).
    checkpoint_op: u64,
}

/// Forest owns all LSM grooves for a single replica.
///
/// The `manifest_log` and `grid` are owned here (upstream `forest.grid` /
/// `forest.manifest_log`); the replica owns the [`Storage`] and drives us via [`Forest::poll`].
///
/// Upstream: `src/lsm/forest.zig:31` (`ForestType`).
pub struct Forest {
    pub accounts: AccountGroove,
    pub transfers: TransferGroove,
    pub transfers_pending: TransferPendingGroove,
    /// Radix-sort scratch buffers for the account groove's trees.
    pub accounts_scratch: AccountGrooveScratch,
    /// Radix-sort scratch buffers for the transfer groove's trees.
    pub transfers_scratch: TransferGrooveScratch,
    /// Radix-sort scratch buffers for the pending-transfer groove's trees.
    pub transfers_pending_scratch: TransferPendingGrooveScratch,
    /// Durable manifest of every tree's tables (upstream `forest.manifest_log`).
    pub manifest_log: ManifestLog,
    /// Block cache + free set shared by all trees and the manifest log
    /// (upstream `forest.grid`).
    pub grid: Grid,
    /// Non-`None` while an open is in flight.
    progress: Option<ForestProgress>,
}

impl Forest {
    /// Construct a fresh [`Forest`] from a superblock snapshot, sizing the grid and the
    /// manifest log's compaction pace for the configured table count.
    ///
    /// Mirrors upstream `Forest.init`: the grid is started with the superblock view attached.
    /// The manifest log's ring-buffer blocks are allocated once the manifest finishes opening
    /// (its `open` asserts the ring buffer is empty, forest.zig/manifest_log.zig).
    #[must_use]
    pub fn init(
        superblock: SuperBlockView,
        grid_options: GridOptions,
        batch_value_count_limit: u32,
    ) -> Self {
        let pace = forest_pace();

        let mut forest = Self {
            accounts: AccountGroove::new(batch_value_count_limit),
            transfers: TransferGroove::new(batch_value_count_limit),
            transfers_pending: TransferPendingGroove::new(batch_value_count_limit),
            accounts_scratch: AccountGrooveScratch::default(),
            transfers_scratch: TransferGrooveScratch::default(),
            transfers_pending_scratch: TransferPendingGrooveScratch::default(),
            manifest_log: ManifestLog::new(superblock, pace, None),
            // DEVIATION: upstream owns a superblock; here the view is a Copy snapshot so we
            // reuse it for both the manifest log and the grid's attached view.
            grid: Grid::new(grid_options),
            progress: None,
        };
        forest.grid.attach_superblock_view(superblock);
        forest
    }

    /// Begin opening the forest: attach the manifest log to every groove, open the grid and
    /// the manifest log, then drive to completion with [`Forest::poll`].
    ///
    /// Port of upstream `forest.open` (forest.zig:369). The `open_event` manifest-entry
    /// routing (replaying each table into its owning tree via `tree.open_table`) is deferred.
    ///
    /// # Panics
    /// Panics if an open is already in flight.
    pub fn open(
        &mut self,
        references: GridOpenReferences,
        storage: &mut dyn Storage,
        checkpoint_op: u64,
    ) {
        assert!(self.progress.is_none());

        self.accounts.open_commence(&mut self.manifest_log);
        self.transfers.open_commence(&mut self.manifest_log);
        self.transfers_pending.open_commence(&mut self.manifest_log);

        self.grid.open(storage, references);
        // TODO(port): src/lsm/forest.zig:381 manifest_log_open_event — route replayed tables
        // by tree_id into the owning tree's `open_table`.
        self.manifest_log.open(|_| {}, || {}, &mut self.grid, storage);

        self.progress = Some(ForestProgress { checkpoint_op });
    }

    /// Drive the grid and manifest log toward their pending completion. Once the manifest
    /// log finishes opening, each groove's `open_complete` is called exactly once.
    pub fn poll(&mut self, storage: &mut dyn Storage) {
        self.grid.poll(storage);
        self.manifest_log.poll(&mut self.grid, storage);

        if self.manifest_log.is_opened() && self.progress.is_some() {
            // Upstream `manifest_log_open_callback` (forest.zig:402): the manifest log finished
            // opening, so resume each groove at `checkpoint_op`.
            let ForestProgress { checkpoint_op } = self
                .progress
                .take()
                .unwrap_or_else(|| unreachable!("progress must be Some to complete an open"));
            // The ring buffer must be allocated after the manifest finishes opening (its
            // `open` asserts it is empty). No appends happen this slice, but pre-allocating
            // here keeps the manifest ready for them.
            self.manifest_log.init_blocks(&mut self.grid);
            self.accounts.open_complete(checkpoint_op);
            self.transfers.open_complete(checkpoint_op);
            self.transfers_pending.open_complete(checkpoint_op);
        }
    }

    /// Compact every groove's mutable tables for the given op, sorting them with
    /// their per-tree radix-sort scratch and compacting the objects caches on the
    /// last beat of the bar.
    ///
    /// Port of upstream `forest.compact` (forest.zig:417), which drives each
    /// groove's `compact(op, &radix_buffer)`.
    pub fn compact(&mut self, op: u64) {
        self.accounts.compact(op, &mut self.accounts_scratch);
        self.transfers.compact(op, &mut self.transfers_scratch);
        self.transfers_pending.compact(op, &mut self.transfers_pending_scratch);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::Zone;
    use crate::multiversion::Release;
    use crate::storage::MemoryStorage;
    use crate::superblock::{DATA_FILE_SIZE_MIN, TrailerReference};
    use tigerbeetle_core::checksum::checksum;
    use tigerbeetle_core::constants::{self, BLOCK_SIZE};
    use tigerbeetle_core::types::Account;

    const CACHE_BLOCKS_COUNT: usize = 256;
    const READ_IOPS_MAX: usize = 2;
    const WRITE_IOPS_MAX: usize = 2;
    const FREE_SET_BLOCKS: usize = 128;

    fn grid_options() -> GridOptions {
        // An unopened free set (like upstream before `Grid.open`), so `Forest::open` can load
        // it from (empty) checkpoint trailers — the `new_unopened_grid` test pattern.
        //
        // The manifest log pre-allocates its whole ring buffer from the grid's stash via
        // `init_blocks`, and the grid's own open/read path takes stash blocks for trailer
        // reads, so size the stash for the ring buffer plus headroom.
        let pace = forest_pace();
        GridOptions {
            cache_blocks_count: CACHE_BLOCKS_COUNT,
            stash_blocks_count: pace.blocks_count() as usize + READ_IOPS_MAX + 128,
            read_iops_max: READ_IOPS_MAX,
            write_iops_max: WRITE_IOPS_MAX,
            free_set_blocks_count: None,
            free_set_blocks_capacity: Some(FREE_SET_BLOCKS),
        }
    }

    fn empty_reference() -> TrailerReference {
        TrailerReference {
            checksum: checksum(&[]),
            last_block_address: 0,
            last_block_checksum: 0,
            trailer_size: 0,
        }
    }

    fn empty_references() -> GridOpenReferences {
        GridOpenReferences {
            blocks_acquired: empty_reference(),
            blocks_released: empty_reference(),
        }
    }

    fn test_superblock() -> SuperBlockView {
        SuperBlockView {
            cluster: 0xAB,
            release: Release { value: 1 },
            storage_size: DATA_FILE_SIZE_MIN as u64,
            manifest_block_count: 0,
            manifest_oldest_address: 0,
            manifest_oldest_checksum: 0,
            op_compacted: false,
        }
    }

    fn storage() -> MemoryStorage {
        MemoryStorage::new(Zone::Grid.start() + 64 * BLOCK_SIZE as u64)
    }

    /// `Forest::compact` drives each groove's `compact` with its scratch buffers,
    /// sorting + deduping the account object mutable table's suffix.
    ///
    /// Mirrors `tree::tests::tree_compact_sorts_and_dedups_the_mutable_suffix` at
    /// the forest level. The account object tree sorts by `timestamp` (its tree
    /// key), so out-of-order timestamps are put and the duplicate resolves to the
    /// latest put.
    #[test]
    fn forest_compact_sorts_and_dedups_groove_trees() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);

        forest.accounts.objects.put(&Account { id: 5, timestamp: 5, ..Account::default() });
        forest.accounts.objects.put(&Account { id: 3, timestamp: 3, ..Account::default() });
        forest.accounts.objects.put(&Account { id: 9, timestamp: 9, ..Account::default() });
        forest.accounts.objects.put(&Account { id: 9, timestamp: 9, ..Account::default() });

        forest.compact(0);

        let values = forest.accounts.objects.table_mutable_ref().values_used();
        assert_eq!(
            values.iter().map(|account| account.timestamp).collect::<Vec<_>>(),
            vec![3, 5, 9]
        );
    }

    /// Compacting across a full bar (including the last-beat objects-cache compact)
    /// runs without panicking for all three grooves.
    #[test]
    fn forest_compact_full_bar() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        for op in 0..(constants::LSM_COMPACTION_OPS as u64) {
            forest.compact(op);
        }
    }

    /// `Forest::open` attaches the manifest log to every groove and, once the manifest log
    /// (with empty checkpoint references) finishes opening, calls each groove's
    /// `open_complete(checkpoint_op)` — the trees are then opened. Poll until complete.
    #[test]
    fn forest_open_completes_groove_open_seam() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();

        forest.open(empty_references(), &mut storage, 0);

        // No polling yet — the open must not have completed synchronously.
        assert!(!forest.accounts.objects.is_opened());

        for _ in 0..1000 {
            forest.poll(&mut storage);
            if forest.accounts.objects.is_opened() {
                break;
            }
        }

        assert!(forest.accounts.objects.is_opened());
        assert!(forest.manifest_log.is_opened());
        assert!(forest.progress.is_none(), "open must complete exactly once");
    }
}
