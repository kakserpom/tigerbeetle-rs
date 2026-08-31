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

/// In-flight lifecycle, tracked so `Forest::open` and `Forest::checkpoint` complete exactly once.
enum ForestProgress {
    /// Forest is opening: `open_complete(checkpoint_op)` fires on each groove once the
    /// manifest log finishes reading.
    Open {
        /// The op the grooves resume from after this open (upstream passes the checkpoint op
        /// into `open_complete`).
        checkpoint_op: u64,
    },
    /// Forest is checkpointing: manifest log flush in progress; user callback fires when done.
    Checkpoint {
        /// Set by the manifest log's flush-completion callback. `Forest::poll` reads this to
        /// know the flush is done (the completion callback itself runs while `manifest_log` is
        /// mutably borrowed, so it cannot touch `self.progress`).
        done: std::rc::Rc<std::cell::Cell<bool>>,
        /// User callback invoked once the manifest log flush completes.
        callback: Box<dyn FnMut()>,
    },
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

        self.progress = Some(ForestProgress::Open { checkpoint_op });
    }

    /// Drive the grid and manifest log toward their pending completion.
    ///
    /// * **Open:** once the manifest log finishes reading, each groove's `open_complete` is
    ///   called exactly once and the ring buffer is allocated.
    /// * **Checkpoint:** once the manifest log flush completes (the `done` flag is set by the
    ///   manifest-log flush callback), the user callback fires.
    pub fn poll(&mut self, storage: &mut dyn Storage) {
        self.grid.poll(storage);
        self.manifest_log.poll(&mut self.grid, storage);

        // Checkpoint completion: the manifest log's flush-completion callback marks
        // `done`; only then fire the user callback. (For the zero-block case the flag is
        // already set before the first `poll`, handled by the explicit `poll` in
        // `Forest::checkpoint`.)
        if matches!(
            &self.progress,
            Some(ForestProgress::Checkpoint { done, .. }) if done.get()
        ) {
            let Some(ForestProgress::Checkpoint { mut callback, .. }) = self.progress.take() else {
                unreachable!("progress was just matched");
            };
            callback();
            return;
        }

        // Open completion: once the manifest log finishes opening, resume each groove.
        if self.manifest_log.is_opened()
            && let Some(ForestProgress::Open { checkpoint_op }) = self.progress.take()
        {
            // Upstream `manifest_log_open_callback` (forest.zig:402): the manifest log
            // finished opening, so resume each groove at `checkpoint_op`.
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

    /// Compact the manifest log, re-appending live entries from old log blocks.
    ///
    /// Call on the last beat of each half-bar and the last beat of each bar
    /// (upstream `forest.compact` → `manifest_log.compact`, forest.zig:467-468).
    /// The callback fires once the compact completes (all reads processed,
    /// entries re-appended into the ring buffer). The accumulated entries are
    /// flushed durably at the next [`checkpoint`](Self::checkpoint).
    ///
    /// # Panics
    ///
    /// Panics if a lifecycle (open or checkpoint) is already in flight.
    pub fn compact_manifest_log<C>(&mut self, callback: C, op: u64, storage: &mut dyn Storage)
    where
        C: FnMut() + 'static,
    {
        assert!(
            self.progress.is_none(),
            "cannot compact manifest log while open/checkpoint in flight"
        );
        self.manifest_log.compact(callback, op, &mut self.grid, storage);
    }

    /// Flush all pending manifest log blocks durably to storage.
    ///
    /// Port of upstream `forest.checkpoint` (forest.zig:798). The user callback fires once
    /// the manifest log's flush completes (all `WriteDone` events processed by `poll`).
    ///
    /// After the callback fires, [`checkpoint_references`](ManifestLog::checkpoint_references)
    /// returns the manifest block checksums/addresses to persist into the superblock
    /// checkpoint trailer.
    ///
    /// For zero-block manifests (no entries appended since last checkpoint/open), the
    /// callback fires synchronously before `checkpoint` returns. For non-zero blocks,
    /// the callback fires on a subsequent [`poll`] call.
    ///
    /// # Panics
    ///
    /// Panics if a lifecycle (open or checkpoint) is already in flight.
    pub fn checkpoint<C>(&mut self, callback: C, storage: &mut dyn Storage)
    where
        C: FnMut() + 'static,
    {
        assert!(self.progress.is_none(), "cannot checkpoint while open/checkpoint in flight");
        // Upstream: assert grooves between_bars, assert mutable tables empty,
        // assert immutable tables not absorbed (forest.zig:815-828). Deferred —
        // needs per-tree compaction state tracking.

        let done = std::rc::Rc::new(std::cell::Cell::new(false));
        self.progress =
            Some(ForestProgress::Checkpoint { done: done.clone(), callback: Box::new(callback) });

        // The manifest log checkpoint closes any open block, asserts all blocks
        // are closed, then flushes (writes) them. The completion callback marks
        // the `done` flag; it fires either synchronously (zero blocks, here) or
        // asynchronously (inside `poll` → `manifest_log.poll` →
        // on_grid_event WriteDone). `Forest::poll` checks `done` to decide when
        // to fire the user callback.
        self.manifest_log.checkpoint(
            {
                let done = done.clone();
                move || done.set(true)
            },
            &mut self.grid,
            storage,
        );

        // For zero blocks, the completion callback fires synchronously inside
        // `manifest_log.checkpoint` above; for non-zero blocks it fires on a
        // later `poll`. Drive `poll` once to process either.
        self.poll(storage);
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
    use tigerbeetle_lsm::schema::manifest_node::{self, Event, Label};

    const CACHE_BLOCKS_COUNT: usize = 256;
    const READ_IOPS_MAX: usize = 2;
    const WRITE_IOPS_MAX: usize = 2;
    // Sized for the manifest log's persistent ring-buffer reservation plus the transient
    // reservation the manifest log makes per compaction (`half_bar_append_blocks_max`), so
    // the forest `open → append → checkpoint → compact` tests fit under the free-set limit.
    const FREE_SET_BLOCKS: usize = 1024;

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
            stash_blocks_count: 2 * pace.blocks_count() as usize + READ_IOPS_MAX + 256,
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

    /// Open the forest to completion, mirroring `forest_open_completes_groove_open_seam`.
    fn open_forest(forest: &mut Forest, storage: &mut MemoryStorage) {
        forest.open(empty_references(), storage, 0);
        let mut done = false;
        for _ in 0..1000 {
            forest.poll(storage);
            if forest.accounts.objects.is_opened() && forest.progress.is_none() {
                done = true;
                break;
            }
        }
        assert!(done, "open must complete");
    }

    /// Construct a valid manifest-node `TableInfo` for appending. Mirrors
    /// `open_tree`'s manifest entry construction (groove.rs), minus the key bytes.
    fn manifest_info(address: u64, tree_id: u16) -> manifest_node::TableInfo {
        manifest_node::TableInfo {
            key_min: [0_u8; 32],
            key_max: [255_u8; 32],
            checksum: checksum(&address.to_le_bytes()),
            address,
            snapshot_min: 1,
            snapshot_max: u64::MAX,
            value_count: 16,
            tree_id,
            label: Label { level: 0, event: Event::Insert },
        }
    }

    /// Append one manifest entry to the physical manifest log via its grid-bearing path.
    fn append_table(forest: &mut Forest, address: u64, tree_id: u16) {
        let info = manifest_info(address, tree_id);
        forest.manifest_log.append(&info, &mut forest.grid);
    }

    #[test]
    fn forest_checkpoint_flushes_manifest_entries() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest, &mut storage);

        // Append entries that open an initial manifest block.
        append_table(&mut forest, 0x1000, 1);
        append_table(&mut forest, 0x1001, 2);

        let checkpointed = std::rc::Rc::new(std::cell::Cell::new(false));
        forest.checkpoint(
            {
                let checkpointed = checkpointed.clone();
                move || checkpointed.set(true)
            },
            &mut storage,
        );

        // Non-zero blocks flush asynchronously — poll until the user callback fires.
        let mut polls = 0;
        while !checkpointed.get() {
            forest.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint must complete");
        }

        assert!(forest.progress.is_none());

        // The manifest now holds one closed block; its references reflect the appends.
        let refs = forest.manifest_log.checkpoint_references();
        assert!(!refs.empty());
        assert_eq!(refs.block_count, 1);
        assert!(refs.oldest_address > 0);
        assert!(refs.newest_address > 0);
    }

    /// A checkpoint on a manifest with no pending entries fires its callback
    /// synchronously (zero blocks to flush). The references are empty/default.
    #[test]
    fn forest_checkpoint_empty_log() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest, &mut storage);

        let checkpointed = std::rc::Rc::new(std::cell::Cell::new(false));
        forest.checkpoint(
            {
                let checkpointed = checkpointed.clone();
                move || checkpointed.set(true)
            },
            &mut storage,
        );

        // Zero blocks → callback fires synchronously inside Forest::checkpoint's poll.
        assert!(checkpointed.get());
        assert!(forest.progress.is_none());
        assert!(forest.manifest_log.checkpoint_references().empty());
    }

    /// Compacting the manifest log re-appends live entries from the oldest blocks, so
    /// the entries remain present after compaction and subsequent appends populate a
    /// fresh block. Mirrors upstream `forest.compact` → `manifest_log.compact`.
    #[test]
    fn forest_compact_manifest_log_re_appends_live() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest, &mut storage);

        append_table(&mut forest, 0x1000, 1);
        append_table(&mut forest, 0x1001, 2);

        let compacted = std::rc::Rc::new(std::cell::Cell::new(false));
        forest.compact_manifest_log(
            {
                let compacted = compacted.clone();
                move || compacted.set(true)
            },
            constants::LSM_COMPACTION_OPS as u64,
            &mut storage,
        );
        let mut polls = 0;
        while !compacted.get() {
            forest.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "manifest compact must complete");
        }

        // After compaction the entries are still live, and a subsequent checkpoint
        // flushes a block reflecting them (a dropped entry would yield an empty log).
        let checkpointed = std::rc::Rc::new(std::cell::Cell::new(false));
        forest.checkpoint(
            {
                let checkpointed = checkpointed.clone();
                move || checkpointed.set(true)
            },
            &mut storage,
        );
        let mut polls = 0;
        while !checkpointed.get() {
            forest.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint after compact must complete");
        }
        assert!(!forest.manifest_log.checkpoint_references().empty());
    }
}
