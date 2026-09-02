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
use std::cell::RefCell;
use std::rc::Rc;
use tigerbeetle_core::constants::{CONFIG, LSM_GROWTH_FACTOR, LSM_LEVELS};
use tigerbeetle_lsm::manifest::ManifestLog as ManifestLogTrait;
use tigerbeetle_lsm::schema::manifest_node as mn;
use tigerbeetle_lsm::tree::table_count_max_for_tree;

/// Shared buffer for collecting manifest entries during `ManifestLog::open`.
type SharedTableBuffer = Rc<RefCell<Vec<mn::TableInfo>>>;

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
        /// Manifest entries replayed during `ManifestLog::open`. The manifest log's
        /// `open_event` callback pushes here (it cannot reach `self` — the callback runs while
        /// `manifest_log` is mutably borrowed); `Forest::poll` drains it into the owning trees'
        /// `open_table` once the open completes, before `open_complete` (upstream
        /// `manifest_log_open_event`, forest.zig:381).
        replayed: SharedTableBuffer,
    },
    /// Forest is checkpointing: the manifest log flush, then the grid free-set checkpoint,
    /// are sequenced in `Forest::poll`; the user callback fires only once both complete.
    Checkpoint {
        /// Set by the manifest log's flush-completion callback. `Forest::poll` reads this to
        /// know the flush is done (the completion callback itself runs while `manifest_log` is
        /// mutably borrowed, so it cannot touch `self.progress`).
        done: std::rc::Rc<std::cell::Cell<bool>>,
        /// Whether `Grid::checkpoint` has been kicked off (only once the manifest flush is
        /// done, so the encoded free set includes every manifest block).
        grid_started: bool,
        /// Whether the manifest log's grid reservation was forfeited to let the grid's own
        /// checkpoint run (`Grid::checkpoint` asserts zero outstanding reservations); it is
        /// restored once the grid checkpoint completes.
        reservation_released: bool,
        /// User callback invoked once the manifest flush and the grid checkpoint both complete.
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
    /// Port of upstream `forest.open` (forest.zig:369).
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
        // Port of upstream `manifest_log_open_event` (forest.zig:381): the manifest log
        // calls this per replayed entry; we collect them into a shared buffer and route them
        // by tree_id in `poll` (the callback cannot borrow `self` while `manifest_log` is
        // being polled, so routing is deferred until the open completes).
        let replayed = Rc::new(RefCell::new(Vec::new()));
        let event = {
            let replayed = replayed.clone();
            move |table: &mn::TableInfo| replayed.borrow_mut().push(*table)
        };
        self.manifest_log.open(event, || {}, &mut self.grid, storage);

        self.progress = Some(ForestProgress::Open { checkpoint_op, replayed });
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

        // Checkpoint completion: sequence the manifest flush, then the grid free-set
        // checkpoint, then finish. The manifest log's flush-completion callback marks
        // `done`; only after that do we forfeit the manifest log's grid reservation (the grid
        // checkpoint asserts zero outstanding reservations) and start `Grid::checkpoint`, so
        // the encoded free set includes every manifest block. When the grid checkpoint is no
        // longer in flight we restore the reservation and fire the user callback.
        let mut complete = false;
        if let Some(ForestProgress::Checkpoint { done, grid_started, reservation_released, .. }) =
            &mut self.progress
            && done.get()
        {
            if !*grid_started {
                self.manifest_log.forfeit_grid_reservation(&mut self.grid);
                *reservation_released = true;
                // The free set only encodes once the previous checkpoint is durable
                // (upstream `Grid.checkpoint_durable` before `Grid.checkpoint`).
                self.grid.checkpoint_durable();
                self.grid.checkpoint(storage);
                *grid_started = true;
            } else if !self.grid.is_checkpoint_in_flight() {
                if *reservation_released {
                    self.manifest_log.reserve_grid_blocks(&mut self.grid);
                }
                complete = true;
            }
        }

        if complete {
            let Some(ForestProgress::Checkpoint { mut callback, .. }) = self.progress.take() else {
                unreachable!("progress was just matched");
            };
            callback();
            return;
        }

        // Open completion: once the manifest log finishes opening, resume each groove.
        // Guard on the variant *before* `take()`, which unconditionally clears `progress`
        // (a `Checkpoint` must not be swallowed by this branch — it completes in its own
        // block above).
        if self.manifest_log.is_opened()
            && matches!(self.progress, Some(ForestProgress::Open { .. }))
            && let Some(ForestProgress::Open { checkpoint_op, replayed }) = self.progress.take()
        {
            // Upstream `manifest_log_open_callback` (forest.zig:402): the manifest log
            // finished opening, so replay each collected entry into its owning tree's
            // `open_table`, then resume each groove at `checkpoint_op`.
            let replayed = std::mem::take(&mut *replayed.borrow_mut());
            for table in &replayed {
                self.open_replay_table(table);
            }
            drop(replayed);

            self.manifest_log.init_blocks(&mut self.grid);
            self.accounts.open_complete(checkpoint_op);
            self.transfers.open_complete(checkpoint_op);
            self.transfers_pending.open_complete(checkpoint_op);
        }
    }

    /// Route one replayed manifest entry to its owning tree's `open_table`.
    ///
    /// Port of upstream `manifest_log_open_event` (forest.zig:381-397): assert the
    /// tree_id is known, then dispatch by `tree_id` to the specific tree. The routing mirrors
    /// upstream's `tree_for_id` switch using the canonical upstream tree ids.
    ///
    /// # Panics
    /// Panics if the manifest contains an unknown `tree_id`.
    fn open_replay_table(&mut self, table: &mn::TableInfo) {
        // Only the primary-key index trees are replayed as their own tables; the tree ids
        // below are upstream `tree_ids` (state_machine.zig:45-78).
        match table.tree_id {
            // Account groove.
            1 => self.accounts.id.open_table(table),
            2 => self.accounts.user_data_128.open_table(table),
            3 => self.accounts.user_data_64.open_table(table),
            4 => self.accounts.user_data_32.open_table(table),
            5 => self.accounts.ledger.open_table(table),
            6 => self.accounts.code.open_table(table),
            7 => self.accounts.objects.open_table(table),
            23 => self.accounts.imported.open_table(table),
            25 => self.accounts.closed.open_table(table),
            // Transfer groove.
            8 => self.transfers.id.open_table(table),
            9 => self.transfers.debit_account_id.open_table(table),
            10 => self.transfers.credit_account_id.open_table(table),
            11 => self.transfers.amount.open_table(table),
            12 => self.transfers.pending_id.open_table(table),
            13 => self.transfers.user_data_128.open_table(table),
            14 => self.transfers.user_data_64.open_table(table),
            15 => self.transfers.user_data_32.open_table(table),
            16 => self.transfers.ledger.open_table(table),
            17 => self.transfers.code.open_table(table),
            18 => self.transfers.objects.open_table(table),
            19 => self.transfers.expires_at.open_table(table),
            24 => self.transfers.imported.open_table(table),
            26 => self.transfers.closing.open_table(table),
            // TransferPending groove.
            20 | 21 => self.transfers_pending.open_table(table),
            other => panic!("unknown tree_id in manifest: {other}"),
        }
    }

    /// Compact every groove's mutable tables for the given op, sorting them with
    /// their per-tree radix-sort scratch and compacting the objects caches on the
    /// last beat of the bar; then drive the level-0 (immutable-flush) compaction of
    /// every tree through the bar cadence (upstream `forest.compact_trees_start`
    /// + `compact_finish`), factoring in the grid, storage, and manifest log.
    ///
    /// Port of upstream `forest.compact` (forest.zig:417), which drives each
    /// groove's `compact(op, &radix_buffer)` and the active level-0 compactions.
    pub fn compact(&mut self, op: u64, storage: &mut dyn Storage) {
        self.accounts.compact(op, &mut self.accounts_scratch);
        self.transfers.compact(op, &mut self.transfers_scratch);
        self.transfers_pending.compact(op, &mut self.transfers_pending_scratch);

        // DEVIATION: upstream plans and reserves grid blocks per beat for the whole forest
        // (`forest.compact_trees_reserve_grid_blocks` + `ResourcePool`); this port drives each
        // tree's level-0 compaction synchronously per beat, mirroring `compact_trees_start` +
        // `compact_finish` (which run `level_active` compactions; the level-0 compactions are
        // only advanced in the second half-bar, and every tree's mutable suffix is swapped on
        // the last beat regardless).
        self.compact_level0_trees(op, storage);
    }

    /// Drive the level-0 (immutable-flush) compaction of every tree in the forest, one op/beat.
    ///
    /// DEVIATION: upstream comptime-generates this loop (`inline for (std.enums.values(TreeID))`,
    /// forest.zig:567); here the 25 trees of the three grooves are enumerated explicitly, each
    /// with its own radix-sort scratch buffer (see the `*GrooveScratch` DEVIATION notes).
    #[allow(clippy::too_many_lines)]
    fn compact_level0_trees(&mut self, op: u64, storage: &mut dyn Storage) {
        self.accounts.objects.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.objects,
        );
        self.accounts.id.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.id,
        );
        self.accounts.user_data_128.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.user_data_128,
        );
        self.accounts.user_data_64.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.composite_key_64,
        );
        self.accounts.user_data_32.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.composite_key_64,
        );
        self.accounts.ledger.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.composite_key_64,
        );
        self.accounts.code.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.composite_key_64,
        );
        self.accounts.imported.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.composite_key_unit,
        );
        self.accounts.closed.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.accounts_scratch.composite_key_unit,
        );

        self.transfers.objects.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.objects,
        );
        self.transfers.id.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.id,
        );
        self.transfers.debit_account_id.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_128,
        );
        self.transfers.credit_account_id.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_128,
        );
        self.transfers.amount.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_128,
        );
        self.transfers.pending_id.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_128,
        );
        self.transfers.user_data_128.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_128,
        );
        self.transfers.user_data_64.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_64,
        );
        self.transfers.user_data_32.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_64,
        );
        self.transfers.ledger.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_64,
        );
        self.transfers.code.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_64,
        );
        self.transfers.expires_at.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_64,
        );
        self.transfers.imported.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_unit,
        );
        self.transfers.closing.compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_scratch.composite_key_unit,
        );

        self.transfers_pending.objects_mut().compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_pending_scratch.objects,
        );
        self.transfers_pending.status_mut().compact_level0(
            op,
            &mut self.grid,
            storage,
            &mut self.manifest_log,
            &mut self.transfers_pending_scratch.status,
        );
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

    /// Flush all pending manifest log blocks durably to storage and checkpoint the grid's
    /// free set (so the references below capture every manifest block).
    ///
    /// Port of upstream `forest.checkpoint` (forest.zig:798) plus the grid free-set
    /// checkpoint a replica drives alongside it. Sequence in `Forest::poll`: the manifest log
    /// flush completes first (all `WriteDone` events processed), then the manifest log's grid
    /// reservation is forfeited (the grid checkpoint asserts zero outstanding reservations),
    /// `Grid::checkpoint` runs, and finally the reservation is restored. The user callback
    /// fires once both complete.
    ///
    /// After the callback fires, [`references`](Self::references) returns the manifest block
    /// checksums/addresses and the free-set trailer references to persist into the superblock
    /// checkpoint trailer — the reopen inputs for a later [`open`](Self::open).
    ///
    /// For both zero-block manifests and a free set with nothing to encode, the phases still
    /// run through `poll`; the caller drives [`poll`](Self::poll) until the callback fires.
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
        self.progress = Some(ForestProgress::Checkpoint {
            done: done.clone(),
            grid_started: false,
            reservation_released: false,
            callback: Box::new(callback),
        });

        // Materialize any table entries buffered through the LSM `ManifestLog` trait seam
        // (recorded without a grid) into physical log blocks — this needs the grid, which is
        // only available here. Then the manifest log checkpoint closes the open block and
        // flushes durably.
        self.manifest_log.flush_pending_appends(&mut self.grid);

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
    use crate::table::TableKey;
    use tigerbeetle_core::checksum::checksum;
    use tigerbeetle_core::constants::{self, BLOCK_SIZE};
    use tigerbeetle_core::types::Account;
    use tigerbeetle_lsm::compaction;
    use tigerbeetle_lsm::schema::manifest_node::{self, Event, Label};
    use tigerbeetle_lsm::table_memory::Mutability;

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
            manifest_newest_address: 0,
            manifest_newest_checksum: 0,
            op_compacted: false,
        }
    }

    fn storage() -> MemoryStorage {
        MemoryStorage::new(Zone::Grid.start() + 64 * BLOCK_SIZE as u64)
    }

    /// A storage whose grid zone covers the forest grid's free-set capacity, so L0-flush
    /// dispatch writes (which address any free block) land in bounds.
    fn large_storage() -> MemoryStorage {
        MemoryStorage::new(Zone::Grid.start() + FREE_SET_BLOCKS as u64 * BLOCK_SIZE as u64)
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

        forest.compact(0, &mut storage());

        let values = forest.accounts.objects.table_mutable_ref().values_used();
        assert_eq!(
            values.iter().map(|account| account.timestamp).collect::<Vec<_>>(),
            vec![3, 5, 9]
        );
    }

    /// Compacting across a full bar (including the last-beat objects-cache compact and
    /// the per-tree level-0 driver) runs without panicking for all three grooves.
    #[test]
    fn forest_compact_full_bar() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        for op in 0..(constants::LSM_COMPACTION_OPS as u64) {
            forest.compact(op, &mut storage);
        }
    }

    /// Drive `Tree::compact_level0` (the L0 immutable-flush driver) across a full
    /// compaction bar on the account object tree, using the forest's real grid, storage,
    /// manifest log, and scratch. Puts values into the mutable table, pops them into an
    /// *unflushed* immutable table via a prior swap, then runs the driver for one bar.
    ///
    /// This is the end-to-end seam the compact dispatch machinery was built for: the
    /// bar paces `half_bar_commence` (the level-0 compaction is `level_active` only in
    /// the second half-bar), drains the immutable source through the grid via `dispatch`,
    /// `half_bar_complete` inserts the real output table into the manifest, and the last
    /// beat swaps mutable→immutable.
    #[test]
    fn tree_compact_level0_driver_flushes_immutable_to_l0() {
        // The grid's free set holds `matrix_free_set_blocks` addresses (a few thousand), far
        // more than the 64 blocks of the default `storage()`. Size the grid zone to cover the
        // free-set capacity so the L0 flush's dispatched writes land in bounds.
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = large_storage();
        open_forest(&mut forest, &mut storage);

        // Seed an *unflushed* immutable table of 2 accounts. Same as the hand-filled
        // flush test: put → compact → swap leaves the immutable unflushed with count 2.
        forest.accounts.objects.put(&Account { id: 3, timestamp: 3, ..Account::default() });
        forest.accounts.objects.put(&Account { id: 5, timestamp: 5, ..Account::default() });
        forest.accounts.objects.compact(&mut forest.accounts_scratch.objects);
        forest.accounts.objects.swap_mutable_and_immutable(1, &mut forest.accounts_scratch.objects);
        assert_eq!(forest.accounts.objects.table_immutable_ref().count(), 2);

        // Run a full bar. The driver is keyed off op modulo LSM_COMPACTION_OPS; the first
        // valid half-bar begins at op = HALF_BAR_BEAT_COUNT (commence asserts op aligns to it),
        // so start the steady-state bar at `bar_start` (`bar_start..bar_start + bar_ops`).
        let bar_ops = constants::LSM_COMPACTION_OPS as u64;
        let bar_start = compaction::HALF_BAR_BEAT_COUNT as u64 * 2;
        for op in bar_start..bar_start + bar_ops {
            forest.accounts.objects.compact_level0(
                op,
                &mut forest.grid,
                &mut storage,
                &mut forest.manifest_log,
                &mut forest.accounts_scratch.objects,
            );
        }

        // The immutable source flushed to a real level-0 table.
        assert!(forest.accounts.objects.manifest_table_count() >= 1);
        assert_eq!(
            forest.accounts.objects.manifest_ref().levels[0].table_count_visible(),
            1,
            "the flushed L0 table should be visible"
        );
        // The immutable table is marked flushed after the L0 flush, and the mutable table
        // was absorbed (count drained back to 0) on the last beat.
        let mutability = forest.accounts.objects.table_immutable_ref().mutability();
        assert!(matches!(mutability, Mutability::Immutable(state) if state.flushed));
        assert_eq!(forest.accounts.objects.table_immutable_ref().count(), 0);
    }

    /// Higher-level end-to-end: `Forest::compact(op, storage)` drives the level-0
    /// immutable-flush driver across *every* tree in all three grooves, so a real
    /// unflushed immutable table in one groove flushes to a visible level-0 table.
    ///
    /// Seeds the account object tree's immutable (put → compact → swap, as above), runs one
    /// full bar through `Forest::compact`, and asserts the flushed L0 table lands in the
    /// manifest while the immutable is drained back to 0.
    #[test]
    fn forest_compact_level0_driver_flushes_immutable_to_l0() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = large_storage();
        open_forest(&mut forest, &mut storage);

        forest.accounts.objects.put(&Account { id: 3, timestamp: 3, ..Account::default() });
        forest.accounts.objects.put(&Account { id: 5, timestamp: 5, ..Account::default() });
        forest.accounts.objects.compact(&mut forest.accounts_scratch.objects);
        forest.accounts.objects.swap_mutable_and_immutable(1, &mut forest.accounts_scratch.objects);
        assert_eq!(forest.accounts.objects.table_immutable_ref().count(), 2);

        let bar_ops = constants::LSM_COMPACTION_OPS as u64;
        let bar_start = compaction::HALF_BAR_BEAT_COUNT as u64 * 2;
        for op in bar_start..bar_start + bar_ops {
            forest.compact(op, &mut storage);
        }

        assert_eq!(
            forest.accounts.objects.manifest_ref().levels[0].table_count_visible(),
            1,
            "the flushed L0 table should be visible"
        );
        assert!(matches!(
            forest.accounts.objects.table_immutable_ref().mutability(),
            Mutability::Immutable(state) if state.flushed
        ));
        assert_eq!(forest.accounts.objects.table_immutable_ref().count(), 0);

        // Every other groove tree stayed idle through the bar (no immutable to flush), so the
        // only manifest tables recorded are the account object tree's flushed L0 table.
        assert_eq!(forest.accounts.id.manifest_table_count(), 0);
        assert_eq!(forest.transfers.objects.manifest_table_count(), 0);
        assert_eq!(forest.transfers_pending.objects_table_count(), 0);
        assert_eq!(forest.transfers_pending.status_table_count(), 0);
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

    /// A manifest-node `TableInfo` whose key bytes are a valid u64 (properly zero-padded),
    /// so it can be replayed through `TreeTableInfo::decode` on reopen. Mirrors the local
    /// `info` helper in `forest_open_replay_routes_by_tree_id`.
    fn manifest_info_u64(address: u64, tree_id: u16) -> manifest_node::TableInfo {
        let key_min = 1_u64.to_le_bytes_padded();
        let key_max = 100_u64.to_le_bytes_padded();
        manifest_node::TableInfo {
            key_min,
            key_max,
            checksum: checksum(&address.to_le_bytes()),
            address,
            snapshot_min: 1,
            snapshot_max: u64::MAX,
            value_count: 1,
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

    /// A checkpoint on a manifest with no pending entries and an unpopulated free set still
    /// sequences both phases; the references are empty/default.
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

        // The callback fires only after the (empty) manifest flush and the free-set
        // checkpoint both traverse `poll`.
        let mut polls = 0;
        while !checkpointed.get() {
            forest.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint must complete");
        }
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

    /// `Forest::open` collects replayed manifest entries into a shared buffer and, once the
    /// manifest log finishes opening, routes each by `tree_id` into the owning tree's
    /// `open_table` before `open_complete`. This can only run before open completes, so we
    /// open without polling (the trees are in the pre-`open_complete` window) and drive the
    /// routing directly.
    ///
    /// The motivating bug was the `tree_id` collision between the Transfer and
    /// TransferPending grooves at ids 20/21 — routing would have ambiguously dispatched those
    /// entries. The canonical ids are now globally unique (upstream `tree_ids`,
    /// state_machine.zig:45-78); we root ids 18 (transfer objects) and 20 (pending objects),
    /// the former collision pair, and one account tree (7) to span all three grooves. All
    /// three are u64-timestamp-keyed, so a single key encoding is valid for each.
    #[test]
    fn forest_open_replay_routes_by_tree_id() {
        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        forest.open(empty_references(), &mut storage, 0);

        forest.open_replay_table(&manifest_info_u64(0x3000, 7)); // account objects
        forest.open_replay_table(&manifest_info_u64(0x3001, 18)); // transfer objects
        forest.open_replay_table(&manifest_info_u64(0x3002, 20)); // pending objects

        // Each routed entry landed in exactly its owning tree (no cross-groove ambiguity).
        assert_eq!(forest.accounts.objects.manifest_table_count(), 1);
        assert_eq!(forest.accounts.id.manifest_table_count(), 0);
        assert_eq!(forest.transfers.objects.manifest_table_count(), 1);
        assert_eq!(forest.transfers.id.manifest_table_count(), 0);
        assert_eq!(forest.transfers_pending.objects_table_count(), 1);
        assert_eq!(forest.transfers_pending.status_table_count(), 0);
    }

    /// A full restart seam: forest A appends manifest entries, checkpoints (making the
    /// manifest blocks durable AND encoding the grid free set that contains them), and a
    /// fresh forest B over the same storage reopens from the recovered free-set refs plus the
    /// manifest refs on its superblock view, replaying the entries into the owning trees.
    #[test]
    fn forest_open_recovers_manifest_from_durable_blocks() {
        let mut forest_a = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest_a, &mut storage);

        // u64-keyed (account object) entries, so they survive `TreeTableInfo::decode` on reopen.
        forest_a.manifest_log.append(&manifest_info_u64(0x1000, 7), &mut forest_a.grid);
        forest_a.manifest_log.append(&manifest_info_u64(0x1001, 7), &mut forest_a.grid);

        let a_done = std::rc::Rc::new(std::cell::Cell::new(false));
        forest_a.checkpoint(
            {
                let a_done = a_done.clone();
                move || a_done.set(true)
            },
            &mut storage,
        );
        let mut polls = 0;
        while !a_done.get() {
            forest_a.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint must complete");
        }

        // Capture the restart references forest A produced.
        let manifest_refs = forest_a.manifest_log.checkpoint_references();
        assert_eq!(manifest_refs.block_count, 1);
        let grid_refs = forest_a.grid.free_set_checkpoint_references();
        assert!(!grid_refs.blocks_acquired.empty(), "free set must encode the manifest blocks");

        // The recovered free set lives above the initial zone, so the reopened view's
        // storage size must cover the highest recorded address (as the superblock records it).
        let highest_address = grid_refs
            .blocks_acquired
            .last_block_address
            .max(grid_refs.blocks_released.last_block_address);
        let storage_size = DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64;

        // Carry the manifest refs on the reopen superblock view.
        let reopen_view = SuperBlockView {
            storage_size,
            manifest_block_count: manifest_refs.block_count,
            manifest_oldest_address: manifest_refs.oldest_address,
            manifest_oldest_checksum: manifest_refs.oldest_checksum,
            manifest_newest_address: manifest_refs.newest_address,
            manifest_newest_checksum: manifest_refs.newest_checksum,
            ..test_superblock()
        };

        // Fresh forest over the same storage, reopened with the recovered free set.
        let mut forest_b = Forest::init(reopen_view, grid_options(), 32);
        forest_b.open(grid_refs, &mut storage, 0);
        let mut b_done = false;
        for _ in 0..1000 {
            forest_b.poll(&mut storage);
            if forest_b.progress.is_none() && forest_b.accounts.objects.is_opened() {
                b_done = true;
                break;
            }
        }
        assert!(b_done, "reopen must complete");

        // Both entries replayed into the account object tree (tree 7) on reopen.
        assert_eq!(forest_b.accounts.objects.manifest_table_count(), 2);
    }

    /// The LSM `ManifestLog` trait seam (the grid-less `append(&entry)` used by grooves and
    /// trees) is buffered, then materialized into physical blocks at `checkpoint` and thus
    /// survives a reopen replay — the sans-IO answer to threading the grid through the trait.
    #[test]
    fn forest_lsm_trait_append_flushed_at_checkpoint() {
        let mut forest_a = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest_a, &mut storage);

        // Invoke the LSM trait `append` (the grid-less two-arg form) explicitly via UFCS: the
        // inherent `append` also takes 2 args plus a `&mut Grid`, so method syntax is ambiguous.
        // Entries buffer in `pending_appends` rather than opening a physical block.
        ManifestLogTrait::append(&mut forest_a.manifest_log, &manifest_info_u64(0x4000, 7));
        ManifestLogTrait::append(&mut forest_a.manifest_log, &manifest_info_u64(0x4001, 7));

        let a_done = std::rc::Rc::new(std::cell::Cell::new(false));
        forest_a.checkpoint(
            {
                let a_done = a_done.clone();
                move || a_done.set(true)
            },
            &mut storage,
        );
        let mut polls = 0;
        while !a_done.get() {
            forest_a.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint must complete");
        }

        let manifest_refs = forest_a.manifest_log.checkpoint_references();
        assert_eq!(manifest_refs.block_count, 1);
        let grid_refs = forest_a.grid.free_set_checkpoint_references();
        assert!(!grid_refs.blocks_acquired.empty(), "free set must encode the manifest blocks");

        let highest_address = grid_refs
            .blocks_acquired
            .last_block_address
            .max(grid_refs.blocks_released.last_block_address);
        let storage_size = DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64;

        let reopen_view = SuperBlockView {
            storage_size,
            manifest_block_count: manifest_refs.block_count,
            manifest_oldest_address: manifest_refs.oldest_address,
            manifest_oldest_checksum: manifest_refs.oldest_checksum,
            manifest_newest_address: manifest_refs.newest_address,
            manifest_newest_checksum: manifest_refs.newest_checksum,
            ..test_superblock()
        };

        let mut forest_b = Forest::init(reopen_view, grid_options(), 32);
        forest_b.open(grid_refs, &mut storage, 0);
        let mut b_done = false;
        for _ in 0..1000 {
            forest_b.poll(&mut storage);
            if forest_b.progress.is_none() && forest_b.accounts.objects.is_opened() {
                b_done = true;
                break;
            }
        }
        assert!(b_done, "reopen must complete");

        assert_eq!(forest_b.accounts.objects.manifest_table_count(), 2);
    }

    /// A real vsr `Compaction` (level_b = 1) is driven through the forest's own grid +
    /// manifest log for a move-table compaction. Seeding level 0 to its compaction
    /// threshold makes `compaction_table(0)` select a table whose level-1 overlap is empty,
    /// so `half_bar_complete` records a single MoveToLevelB manifest entry through the
    /// LSM-trait buffering seam (the previous slice). Checkpointing flushes it durable, and a
    /// reopened forest replays the moved table into level 1 — proving the compaction's
    /// manifest append survives restart (upstream `Compaction`, vsr/compaction.zig).
    #[test]
    fn forest_move_table_compaction_flushed_durably() {
        use crate::compaction::Compaction;
        use crate::groove::AccountObjectSpec;
        use tigerbeetle_lsm::manifest::TreeTableInfo;

        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest, &mut storage);

        // Acquire 4 addresses and mark them acquired so the seeded level-0 tables read as real
        // disk tables to the free set (`!free_set_is_free`), then forfeit the reservation so
        // the grid checkpoint's `count_reservations() == 0` assert holds.
        let reservation = forest.grid.reserve(4);
        let addresses: Vec<u64> = (0..4).map(|_| forest.grid.acquire(reservation)).collect();
        forest.grid.forfeit(reservation);

        // Seed level 0 to its compaction threshold (`table_count_max_for_level(4, 0) == 4`).
        for &address in &addresses {
            let wire = manifest_info_u64(address, 7);
            let table = TreeTableInfo::<u64>::decode(&wire, 7);
            forest.accounts.objects.manifest_mut().insert_table(
                &mut forest.manifest_log,
                0,
                &table,
            );
        }
        assert_eq!(forest.accounts.objects.manifest_table_count(), 4);

        // Drive the level-0 → level-1 move-table compaction (op aligned to HALF_BAR_BEAT_COUNT).
        let mut comp = Compaction::<AccountObjectSpec>::new(
            core::ptr::addr_of_mut!(forest.accounts.objects),
            core::ptr::addr_of_mut!(forest.grid),
            1,
        );
        let quota = comp.half_bar_commence(8, &forest.accounts.objects, &forest.grid);
        assert_eq!(quota, 0, "move-table compaction has no quota to process");
        comp.half_bar_complete(
            &mut forest.accounts.objects,
            &forest.grid,
            &mut forest.manifest_log,
        );

        // The selected table moved to level 1; the other three remain at level 0.
        assert_eq!(forest.accounts.objects.manifest_table_count(), 4);
        assert_eq!(forest.accounts.objects.manifest_ref().levels[1].table_count_visible(), 1);
        assert_eq!(forest.accounts.objects.manifest_ref().levels[0].table_count_visible(), 3);

        // Checkpoint makes the manifest (4 inserts + 1 move) durable.
        let done = std::rc::Rc::new(std::cell::Cell::new(false));
        forest.checkpoint(
            {
                let done = done.clone();
                move || done.set(true)
            },
            &mut storage,
        );
        let mut polls = 0;
        while !done.get() {
            forest.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint must complete");
        }

        let manifest_refs = forest.manifest_log.checkpoint_references();
        let grid_refs = forest.grid.free_set_checkpoint_references();

        let highest_address = grid_refs
            .blocks_acquired
            .last_block_address
            .max(grid_refs.blocks_released.last_block_address);
        let reopen_view = SuperBlockView {
            storage_size: DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64,
            manifest_block_count: manifest_refs.block_count,
            manifest_oldest_address: manifest_refs.oldest_address,
            manifest_oldest_checksum: manifest_refs.oldest_checksum,
            manifest_newest_address: manifest_refs.newest_address,
            manifest_newest_checksum: manifest_refs.newest_checksum,
            ..test_superblock()
        };

        // Reopen over the same storage: all five entries replay, the move landing at level 1.
        let mut forest_b = Forest::init(reopen_view, grid_options(), 32);
        forest_b.open(grid_refs, &mut storage, 0);
        let mut b_done = false;
        for _ in 0..1000 {
            forest_b.poll(&mut storage);
            if forest_b.progress.is_none() && forest_b.accounts.objects.is_opened() {
                b_done = true;
                break;
            }
        }
        assert!(b_done, "reopen must complete");

        // The manifest log dedupes each table to its latest state on open, so the moved
        // table replays once (at level 1); the other three at level 0 — 4 tables total.
        assert_eq!(forest_b.accounts.objects.manifest_table_count(), 4);
        assert_eq!(forest_b.accounts.objects.manifest_ref().levels[1].table_count_visible(), 1);
        assert_eq!(forest_b.accounts.objects.manifest_ref().levels[0].table_count_visible(), 3);
    }

    /// Drive the *non-move* L0-flush branch of `Compaction::half_bar_complete`: an unflushed
    /// immutable table is compacted into level 0, merged with an overlapping level-0 disk table,
    /// the merged output table is inserted at level 0, and the now-invisible source table is
    /// hidden. The resulting manifest is flushed durably and replayed on reopen.
    ///
    /// The deferred dispatch loop (Phase 3) normally populates `manifest_entries`/`counters`/
    /// `quotas` by merging the input tables; here we hand-set those fields to the values the
    /// merge would produce so `half_bar_complete`'s manifest orchestration (insert_table,
    /// update_table, remove_invisible_tables, counter accounting) is exercised end-to-end.
    #[test]
    fn forest_l0_flush_half_bar_complete_inserts_output_hides_input() {
        use crate::compaction::{
            Compaction, CompactionCounters, ManifestEntry, ManifestEntryOperation,
        };
        use crate::groove::AccountObjectSpec;
        use tigerbeetle_lsm::manifest::TreeTableInfo;

        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest, &mut storage);

        // Populate an *unflushed* immutable table: put into the mutable table, compact, then
        // swap into the immutable (a fresh tree's immutable is empty & flushed, so the first
        // swap compacts and leaves the immutable unflushed with `count() == 2`).
        forest.accounts.objects.put(&Account { id: 3, timestamp: 3, ..Account::default() });
        forest.accounts.objects.put(&Account { id: 5, timestamp: 5, ..Account::default() });
        forest.accounts.objects.compact(&mut forest.accounts_scratch.objects);
        forest.accounts.objects.swap_mutable_and_immutable(1, &mut forest.accounts_scratch.objects);
        assert_eq!(forest.accounts.objects.table_immutable_ref().count(), 2);

        // Seed an overlapping level-0 disk table ([1,100] ⊇ immutable [3,5]) whose address
        // reads as acquired to the free set, then forfeit the reservation (grid checkpoint's
        // `count_reservations() == 0` assert must hold).
        let reservation = forest.grid.reserve(2);
        let range_b_addr = forest.grid.acquire(reservation);
        let output_addr = forest.grid.acquire(reservation);
        forest.grid.forfeit(reservation);

        let range_b = TreeTableInfo::<u64>::decode(&manifest_info_u64(range_b_addr, 7), 7);
        forest.accounts.objects.manifest_mut().insert_table(&mut forest.manifest_log, 0, &range_b);
        assert_eq!(forest.accounts.objects.manifest_table_count(), 1);

        // Drive the level-0 flush compaction (op aligned to HALF_BAR_BEAT_COUNT).
        let mut comp = Compaction::<AccountObjectSpec>::new(
            core::ptr::addr_of_mut!(forest.accounts.objects),
            core::ptr::addr_of_mut!(forest.grid),
            0,
        );
        let quota = comp.half_bar_commence(8, &forest.accounts.objects, &forest.grid);
        // Immutable count (2) + range_b value_count (1).
        assert_eq!(quota, 3);

        // Simulate the deferred dispatch (Phase 3): the merged output table holds 2 of the 3
        // input values; the range_b table's sole value is dropped (shadowed/tombstoned).
        let out: u32 = 2;
        let dropped: u32 = 1;
        let output = TreeTableInfo::<u64> {
            checksum: checksum(&output_addr.to_le_bytes()),
            address: output_addr,
            snapshot_min: 1,
            snapshot_max: u64::MAX,
            key_min: 1,
            key_max: 100,
            value_count: out,
        };
        comp.manifest_entries.push(ManifestEntry {
            operation: ManifestEntryOperation::InsertToLevelB,
            table: output,
        });
        comp.counters =
            CompactionCounters { in_: quota, dropped: u64::from(dropped), out: u64::from(out) };
        comp.quotas.half_bar_done = comp.quotas.half_bar;

        // The input range_b table's block was read and released by the (deferred) dispatch.
        forest.grid.release(&[range_b_addr]);

        comp.half_bar_complete(
            &mut forest.accounts.objects,
            &forest.grid,
            &mut forest.manifest_log,
        );

        // The output table is inserted at level 0; the range_b table was updated to an
        // invisible snapshot_max (= 9) and then removed by `remove_invisible_tables`, so a
        // single visible table remains.
        assert_eq!(forest.accounts.objects.manifest_table_count(), 1);
        assert_eq!(forest.accounts.objects.manifest_ref().levels[0].table_count_visible(), 1);

        // Checkpoint makes the manifest (inserts/updates/remove) durable.
        let done = std::rc::Rc::new(std::cell::Cell::new(false));
        forest.checkpoint(
            {
                let done = done.clone();
                move || done.set(true)
            },
            &mut storage,
        );
        let mut polls = 0;
        while !done.get() {
            forest.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint must complete");
        }

        let manifest_refs = forest.manifest_log.checkpoint_references();
        let grid_refs = forest.grid.free_set_checkpoint_references();

        let highest_address = grid_refs
            .blocks_acquired
            .last_block_address
            .max(grid_refs.blocks_released.last_block_address);
        let reopen_view = SuperBlockView {
            storage_size: DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64,
            manifest_block_count: manifest_refs.block_count,
            manifest_oldest_address: manifest_refs.oldest_address,
            manifest_oldest_checksum: manifest_refs.oldest_checksum,
            manifest_newest_address: manifest_refs.newest_address,
            manifest_newest_checksum: manifest_refs.newest_checksum,
            ..test_superblock()
        };

        // Reopen over the same storage: the output table replays at level 0; the range_b table
        // replays its trailing Remove event, so it does not reappear.
        let mut forest_b = Forest::init(reopen_view, grid_options(), 32);
        forest_b.open(grid_refs, &mut storage, 0);
        let mut b_done = false;
        for _ in 0..1000 {
            forest_b.poll(&mut storage);
            if forest_b.progress.is_none() && forest_b.accounts.objects.is_opened() {
                b_done = true;
                break;
            }
        }
        assert!(b_done, "reopen must complete");

        assert_eq!(forest_b.accounts.objects.manifest_table_count(), 1);
        assert_eq!(forest_b.accounts.objects.manifest_ref().levels[0].table_count_visible(), 1);
    }

    /// Drive the *non-move* `TableInfoA::Disk` branch of `Compaction::half_bar_complete`
    /// (level 0→1 merge with a non-empty level-1 overlap): the selected level-0 input table and
    /// its overlapping level-1 tables are updated to an invisible snapshot_max, the merged
    /// output is inserted at level 1, and `remove_invisible_tables` runs at BOTH consulted
    /// levels (the `level_b > 0` branch the move-table and L0-flush tests skipped). Flushed
    /// durably and replayed on reopen.
    ///
    /// Like the L0-flush test, the deferred dispatch (Phase 3) would populate
    /// `manifest_entries`/`counters`/`quotas` by merging; here we hand-set them to the merged
    /// values and drive `half_bar_complete`'s manifest orchestration end-to-end.
    #[test]
    #[allow(clippy::too_many_lines)] // full setup + checkpoint + reopen scaffolding
    fn forest_disk_merge_half_bar_complete_updates_inputs_inserts_output() {
        use crate::compaction::{
            Compaction, CompactionCounters, ManifestEntry, ManifestEntryOperation, TableInfoA,
        };
        use crate::groove::AccountObjectSpec;
        use tigerbeetle_lsm::manifest::TreeTableInfo;

        let mut forest = Forest::init(test_superblock(), grid_options(), 32);
        let mut storage = storage();
        open_forest(&mut forest, &mut storage);

        // Seed level 0 to its compaction threshold (`table_count_max_for_level(4, 0) == 4`) with
        // tiled u64-keyed tables, plus a single level-1 table whose range covers them all, so a
        // non-empty overlap forces a real merge (not a move) of the least-overlap level-0 table.
        let reservation = forest.grid.reserve(6);
        let addresses: Vec<u64> = (0..6).map(|_| forest.grid.acquire(reservation)).collect();
        forest.grid.forfeit(reservation);

        let level_0_ranges: [(u64, u64); 4] = [(1, 100), (101, 200), (201, 300), (301, 400)];
        for (i, &(lo, hi)) in level_0_ranges.iter().enumerate() {
            let wire = manifest_node::TableInfo {
                key_min: lo.to_le_bytes_padded(),
                key_max: hi.to_le_bytes_padded(),
                checksum: checksum(&addresses[i].to_le_bytes()),
                address: addresses[i],
                snapshot_min: 1,
                snapshot_max: u64::MAX,
                value_count: 1,
                tree_id: 7,
                label: Label { level: 0, event: Event::Insert },
            };
            let table = TreeTableInfo::<u64>::decode(&wire, 7);
            forest.accounts.objects.manifest_mut().insert_table(
                &mut forest.manifest_log,
                0,
                &table,
            );
        }
        let level_1_addr = addresses[4];
        let level_1_wire = manifest_node::TableInfo {
            key_min: 1_u64.to_le_bytes_padded(),
            key_max: 400_u64.to_le_bytes_padded(),
            checksum: checksum(&level_1_addr.to_le_bytes()),
            address: level_1_addr,
            snapshot_min: 1,
            snapshot_max: u64::MAX,
            value_count: 1,
            tree_id: 7,
            label: Label { level: 0, event: Event::Insert },
        };
        let level_1 = TreeTableInfo::<u64>::decode(&level_1_wire, 7);
        forest.accounts.objects.manifest_mut().insert_table(&mut forest.manifest_log, 1, &level_1);
        assert_eq!(forest.accounts.objects.manifest_table_count(), 5);

        // Drive a level 0→1 merge compaction (level_b=1, level_a=0).
        let mut comp = Compaction::<AccountObjectSpec>::new(
            core::ptr::addr_of_mut!(forest.accounts.objects),
            core::ptr::addr_of_mut!(forest.grid),
            1,
        );
        let quota = comp.half_bar_commence(8, &forest.accounts.objects, &forest.grid);
        // table_a value_count 1 + range_b value_count 1.
        assert_eq!(quota, 2);
        assert!(!comp.move_table, "a non-empty level-1 overlap forces a merge, not a move");

        // Collect the input-table addresses the (deferred) dispatch would have read then
        // released: the selected table_a plus every overlapping range_b table.
        let (Some(TableInfoA::Disk(table_a_ref)), Some(range_b)) =
            (&comp.table_info_a, &comp.range_b)
        else {
            unreachable!("level_b > 0 selected a disk table_a with a non-empty range_b")
        };
        let mut released_inputs = vec![table_a_ref.table_info.address];
        for table in range_b.tables.tables.slice() {
            released_inputs.push(table.table_info.address);
        }

        // Simulate the deferred dispatch (Phase 3): the merged output holds the single
        // surviving value; the range_b table's value is dropped (shadowed/tombstoned).
        let output = TreeTableInfo::<u64> {
            checksum: checksum(&addresses[5].to_le_bytes()),
            address: addresses[5],
            snapshot_min: 1,
            snapshot_max: u64::MAX,
            key_min: 1,
            key_max: 400,
            value_count: 1,
        };
        comp.manifest_entries.push(ManifestEntry {
            operation: ManifestEntryOperation::InsertToLevelB,
            table: output,
        });
        comp.counters = CompactionCounters { in_: quota, dropped: 1, out: 1 };
        comp.quotas.half_bar_done = comp.quotas.half_bar;

        // The dispatch released the read input blocks; `half_bar_complete` asserts both the
        // table_a and range_b inputs are released (they were acquired during commence).
        forest.grid.release(&released_inputs);

        comp.half_bar_complete(
            &mut forest.accounts.objects,
            &forest.grid,
            &mut forest.manifest_log,
        );

        // table_a (level 0) + the level-1 range table are updated to an invisible snapshot_max
        // (= 9) and removed by `remove_invisible_tables`; the output lands at level 1.
        assert_eq!(forest.accounts.objects.manifest_table_count(), 4);
        assert_eq!(forest.accounts.objects.manifest_ref().levels[0].table_count_visible(), 3);
        assert_eq!(forest.accounts.objects.manifest_ref().levels[1].table_count_visible(), 1);

        // Checkpoint makes the manifest (inserts/updates/removes) durable.
        let done = std::rc::Rc::new(std::cell::Cell::new(false));
        forest.checkpoint(
            {
                let done = done.clone();
                move || done.set(true)
            },
            &mut storage,
        );
        let mut polls = 0;
        while !done.get() {
            forest.poll(&mut storage);
            polls += 1;
            assert!(polls < 1000, "checkpoint must complete");
        }

        let manifest_refs = forest.manifest_log.checkpoint_references();
        let grid_refs = forest.grid.free_set_checkpoint_references();

        let highest_address = grid_refs
            .blocks_acquired
            .last_block_address
            .max(grid_refs.blocks_released.last_block_address);
        let reopen_view = SuperBlockView {
            storage_size: DATA_FILE_SIZE_MIN as u64 + highest_address * BLOCK_SIZE as u64,
            manifest_block_count: manifest_refs.block_count,
            manifest_oldest_address: manifest_refs.oldest_address,
            manifest_oldest_checksum: manifest_refs.oldest_checksum,
            manifest_newest_address: manifest_refs.newest_address,
            manifest_newest_checksum: manifest_refs.newest_checksum,
            ..test_superblock()
        };

        // Reopen over the same storage: the compaction inputs replay their trailing Remove
        // events (so they do not reappear), the other three level-0 tables and the output
        // replay at level 0 / level 1 respectively.
        let mut forest_b = Forest::init(reopen_view, grid_options(), 32);
        forest_b.open(grid_refs, &mut storage, 0);
        let mut b_done = false;
        for _ in 0..1000 {
            forest_b.poll(&mut storage);
            if forest_b.progress.is_none() && forest_b.accounts.objects.is_opened() {
                b_done = true;
                break;
            }
        }
        assert!(b_done, "reopen must complete");

        assert_eq!(forest_b.accounts.objects.manifest_table_count(), 4);
        assert_eq!(forest_b.accounts.objects.manifest_ref().levels[0].table_count_visible(), 3);
        assert_eq!(forest_b.accounts.objects.manifest_ref().levels[1].table_count_visible(), 1);
    }
}
