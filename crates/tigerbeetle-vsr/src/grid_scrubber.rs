//! Scrub grid blocks.
//!
//! A "data scrubber" is a background task that gradually/incrementally reads the disk and
//! validates what it finds. Its purpose is to discover faults proactively — as early as
//! possible — rather than waiting for them to be discovered by normal database operation
//! (e.g. during compaction).
//!
//! Ported from `reference/tigerbeetle/src/vsr/grid_scrubber.zig`.
//!
//! The scrubber tours the grid one block at a time, in this order:
//!
//! 1. **Tables:** every table's index block then its value blocks, level-major
//!    (`LSM_LEVELS`) and tree-major (trees in ascending `tree_id` order), wrapping the
//!    level/tree cycle at the per-replica tour origin.
//! 2. **Manifest:** every flushed manifest-log block, oldest first.
//! 3. **Free-set trailer blocks** (acquired, then released).
//! 4. **Client sessions** (DEVIATION: no client-sessions checkpoint trailer is wired into
//!    the port yet, so this stage is always empty).
//!
//! A fault is reported via [`BlockStatus::Repair`]; the owner feeds the block to
//! `Grid::blocks_missing` so a repair write eventually restores it (the block is then
//! re-scrubbed and comes back [`BlockStatus::Ok`]).
//!
//! DEVIATION (structural): upstream embeds the scrubber (`GridScrubberType(Forest, ...)`)
//! inside the `Replica`, holding a `*Forest` and reading through `grid.read_block` with a
//! per-read callback. The sans-I/O ported `Replica` owns no `Forest`, so the scrubber lives
//! in the [`crate::forest::Forest`] instead, next to the `grid`/`manifest_log` it scrubs.
//! Consequences:
//!
//! - Scrub reads resolve onto `Grid::take_scrub_done` instead of the event queue (see
//!   [`crate::grid::ScrubReadDone`]).
//! - The table tour is driven by a closure supplied by the forest (the trees live there,
//!   see the `ForestTours` docs), instead of a `WrappingForestTableIterator` built over a
//!   `*Forest`.
//! - `tour_tables_origin` (the plain "is the scrubber opened/armed" flag) is replaced by the
//!   forest's table-tour cursor being set when the forest open completes; `read_next` gates
//!   on the manifest log being opened instead of `superblock.opened`.
//! - The table tour origin is chosen with a deterministic [`Reservoir`] seeded from a fixed
//!   constant (upstream uses the superblock's per-replica PRNG).
//! - `grid.verify_table` (grid.rs:14 TODO) is not yet ported; index blocks are validated by
//!   the read instead.
//! - `grid.callback == .cancel` is absent (state sync is not ported); there is no `.cancel`
//!   guard in `read_next`, and `cancel()` only marks reads (plus table-value → table-index).
//! - `client_sessions` stage skipped (no trailer).

use crate::checkpoint_trailer::Callback;
use crate::grid::{Grid, ReadBlockResult, ScrubReadDone, Slot};
use crate::manifest_log::ManifestLog;
use crate::message_header::BlockType;
use crate::schema::TableIndex;
use crate::storage::Storage;
use std::collections::VecDeque;
use tigerbeetle_core::constants::GRID_SCRUBBER_READS_MAX;
use tigerbeetle_lsm::manifest::ManifestLog as _;

const _: () = assert!(GRID_SCRUBBER_READS_MAX > 0, "scrubber reads pool must not be empty");

/// A block the scrubber wants to read (`BlockId` upstream).
///
/// Field names mirror upstream's `BlockId` (`block_address`/`block_checksum`/`block_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
pub(crate) struct BlockId {
    pub block_checksum: u128,
    pub block_address: u64,
    pub block_type: BlockType,
}

/// The outcome of a scrub read (`BlockStatus` upstream).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockStatus {
    /// If `read.done`: the scrub failed — the block must be repaired.
    /// If `!read.done`: the scrub is still in progress (this is the initial state).
    Repair,
    /// The scrub succeeded; don't repair the block.
    Ok,
    /// The scrub was aborted (state sync); don't repair the block.
    Canceled,
    /// The block was freed by a checkpoint while the read was in progress; don't repair it.
    ///
    /// (At checkpoint the FreeSet frees blocks released during the preceding checkpoint. We
    /// can scrub released blocks, but not free blocks.)
    Released,
}

/// A scrubber read slot (upstream `Read`).
#[derive(Debug, Clone)]
struct ScrubRead {
    address: u64,
    checksum: u128,
    block_type: BlockType,
    status: BlockStatus,
    /// Whether the read is ready to be released via [`GridScrubber::read_result_next`].
    done: bool,
    /// The grid read token resolving this read (via [`crate::grid::ScrubReadDone`]).
    token: u32,
}

/// Fixed-size pool of [`ScrubRead`] slots sized by `GRID_SCRUBBER_READS_MAX` (upstream's
/// `IOPSType(Read, grid_scrubber_reads_max)`); free/busy/done partition the slots.
#[derive(Debug)]
pub struct GridScrubber {
    /// Track the progress through the grid.
    ///
    /// Every full tour scrubs every acquired block that survives the entire span of the tour.
    tour: Tour,

    /// Contains a table index block when `tour == TableValue` (upstream
    /// `tour_index_block`, a `BlockPtr` the grid keeps until the next table).
    tour_index_block: Option<Vec<u8>>,

    /// Counters reset after every tour cycle. Includes repeat index-block reads (upstream
    /// `tour_blocks_scrubbed_count`).
    tour_blocks_scrubbed_count: u64,

    reads: Vec<Option<ScrubRead>>,
    reads_free: VecDeque<usize>,
    /// A list of reads that are in progress (upstream `reads_busy`).
    reads_busy: VecDeque<usize>,
    /// A list of reads ready to be released (upstream `reads_done`).
    reads_done: VecDeque<usize>,
}

/// Upstream `Glitch.tour` union. `Copy` so `tour_next` can match a single stage while
/// recording the next state back into the field.
#[derive(Clone, Copy, Debug)]
enum Tour {
    Init,
    Done,
    TableIndex,
    /// The index block of the table currently being scrubbed. Points at
    /// `tour_index_block` once the index has been read (`index_block == null` means "keep
    /// issuing index reads until one verifies", upstream `tour.table_value.index_block`).
    TableValue {
        index_checksum: u128,
        index_address: u64,
        value_block_index: u32,
    },
    /// The manifest log tour iterates manifest blocks in reverse order
    /// (so manifest compaction doesn't lead to missed blocks).
    ManifestLog {
        iterator: ManifestBlockIterator,
    },
    FreeSetAcquired {
        index: u32,
    },
    FreeSetReleased {
        index: u32,
    },
    /// Stage counter for the client-sessions trailer. Always empty in this port (DEVIATION:
    /// no trailer is wired), so the carry `index` is skipped (`ClientSessions` upstream
    /// carries a `usize`).
    ClientSessions,
}

/// Iterate over every flushed manifest block address/checksum in the manifest log
/// (upstream `ManifestBlockIteratorType`).
///
/// Stable across manifest-log mutation — guaranteed to iterate every block that survives
/// the entire iteration: each step re-locates the previously-iterated block (positions
/// shift as compaction removes old blocks) and continues one step older.
#[derive(Clone, Copy, Debug)]
enum ManifestBlockIterator {
    Init,
    Done,
    State { index: u32, address: u64, checksum: u128 },
}

impl GridScrubber {
    #[must_use]
    pub(crate) fn new() -> Self {
        let reads_max = GRID_SCRUBBER_READS_MAX as usize;
        Self {
            tour: Tour::Init,
            tour_index_block: None,
            tour_blocks_scrubbed_count: 0,
            reads: vec![None; reads_max],
            reads_free: (0..reads_max).collect(),
            reads_busy: VecDeque::new(),
            reads_done: VecDeque::new(),
        }
    }

    /// Whether the current cycle is finished (upstream `tour == .done`).
    #[must_use]
    pub(crate) fn tour_done(&self) -> bool {
        matches!(self.tour, Tour::Done)
    }

    /// Blocks read so far this cycle (upstream `tour_blocks_scrubbed_count`).
    ///
    /// Used by the forest's scrubber tests; unobserved in production builds so far.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn tour_blocks_scrubbed_count(&self) -> u64 {
        self.tour_blocks_scrubbed_count
    }

    /// Begin a new cycle: reset the tour to its start and zero the count, ready for the
    /// next table tour (upstream `wrap`).
    ///
    /// # Panics
    /// Panics unless the tour is complete.
    pub(crate) fn wrap(&mut self) {
        assert!(matches!(self.tour, Tour::Done));

        self.tour = Tour::Init;
        self.tour_blocks_scrubbed_count = 0;
    }

    /// Cancel queued reads: the replica is about to re-open/state-sync, results will be
    /// ignored (upstream `cancel`).
    ///
    /// Wired from the future forest-owner's recovery path; not yet called from production
    /// code (the sans-I/O replica owns no forest yet).
    #[allow(dead_code)]
    pub(crate) fn cancel(&mut self) {
        for &id in self.reads_busy.iter().chain(&self.reads_done) {
            let read =
                self.reads[id].as_mut().unwrap_or_else(|| unreachable!("queued read is live"));
            read.status = BlockStatus::Canceled;
        }

        if matches!(self.tour, Tour::TableValue { .. }) {
            // Skip scrubbing the table data; the table may not exist when the replica
            // finishes re-opening.
            self.tour = Tour::TableIndex;
        }
    }

    /// Cancel queued reads to blocks that will be freed, now that the current checkpoint is
    /// durable. The read still runs, but its result is ignored (upstream
    /// `checkpoint_durable`, called immediately before `FreeSet.mark_checkpoint_durable`).
    pub(crate) fn checkpoint_durable(&mut self, grid: &Grid) {
        for &id in self.reads_busy.iter().chain(&self.reads_done) {
            let read =
                self.reads[id].as_mut().unwrap_or_else(|| unreachable!("queued read is live"));
            if read.status == BlockStatus::Repair {
                assert!(!grid.free_set_is_free(read.address));
                // `to_be_freed_at_checkpoint_durability` (not `is_released`): only blocks
                // that will be freed *this* checkpoint are aborted.
                if grid.free_set().to_be_freed_at_checkpoint_durability(read.address) {
                    read.status = BlockStatus::Released;
                }
            }
        }

        if let Tour::TableValue { index_address, .. } = self.tour {
            assert!(!grid.free_set_is_free(index_address));
            if grid.free_set().to_be_freed_at_checkpoint_durability(index_address) {
                // Skip scrubbing the table data; the table is about to be released.
                self.tour = Tour::TableIndex;
            }
        }
    }

    /// Whether a new read was started (upstream `read_next`).
    ///
    /// `next_table` advances the forest's table tour; see the module-level deviation note.
    ///
    /// # Panics
    /// Asserts the manifest log is opened and the read-pool bookkeeping is consistent.
    pub(crate) fn read_next(
        &mut self,
        grid: &mut Grid,
        manifest_log: &ManifestLog,
        storage: &mut dyn Storage,
        next_table: &mut dyn FnMut() -> Option<(u64, u128)>,
    ) -> bool {
        self.assert_reads_bookkeeping();

        if self.reads_free.is_empty() {
            return false;
        }

        let Some(block_id) = self.tour_next(grid, manifest_log, next_table) else {
            return false;
        };
        self.tour_blocks_scrubbed_count += 1;

        let id = self.reads_free.pop_front().unwrap_or_else(|| unreachable!("pool checked above"));
        assert!(!self.reads_busy.contains(&id));
        assert!(!self.reads_done.contains(&id));

        self.reads_busy.push_back(id);
        let token = grid.read_block_scrub(storage, block_id.block_address, block_id.block_checksum);
        self.reads[id] = Some(ScrubRead {
            address: block_id.block_address,
            checksum: block_id.block_checksum,
            block_type: block_id.block_type,
            status: BlockStatus::Repair,
            done: false,
            token,
        });

        self.assert_reads_bookkeeping();
        true
    }

    /// Route completed scrub reads back into their slots and cache a newly-verified table
    /// index block (upstream `read_next_callback`, drained by the forest from
    /// `Grid::take_scrub_done`).
    pub(crate) fn on_read_done(&mut self, done: VecDeque<ScrubReadDone>, grid: &Grid) {
        for ScrubReadDone { token, address, checksum, result, data } in done {
            let Some(position) = self
                .reads_busy
                .iter()
                .position(|&id| self.reads[id].as_ref().is_some_and(|read| read.token == token))
            else {
                continue;
            };
            let id = self.reads_busy[position];
            self.reads_busy.remove(position);
            let read = self.reads[id].as_mut().unwrap_or_else(|| unreachable!("busy read is live"));
            assert!(!read.done);
            assert!(!self.reads_done.contains(&id));

            // Cache the current table's index block on first (valid) read: the value-block
            // tour is driven from it. An invalid index keeps being re-scrubbed until it is
            // repaired or released (upstream: "waiting for index repair").
            let is_current_index = read.status == BlockStatus::Repair
                && matches!(
                    self.tour,
                    Tour::TableValue { index_checksum, index_address, value_block_index: 0 }
                        if index_checksum == checksum && index_address == address
                )
                && self.tour_index_block.is_none();

            if is_current_index && result == ReadBlockResult::Valid {
                self.tour_index_block = Some(data);
            }

            if result == ReadBlockResult::Valid {
                if read.status == BlockStatus::Repair {
                    read.status = BlockStatus::Ok;
                }
            } else if grid.free_set_is_free(read.address) {
                // The block was freed (e.g. a table removed after our tour started). Do not
                // repair it. If it was the current table's index, advance past the table —
                // the tour would otherwise lob the same read forever.
                read.status = BlockStatus::Released;
                if is_current_index {
                    self.tour = Tour::TableIndex;
                }
            }

            read.done = true;
            self.reads_done.push_back(id);
        }
    }

    /// Release the oldest completed read, reporting its block + outcome
    /// (upstream `read_result_next`).
    #[must_use]
    pub(crate) fn read_result_next(&mut self) -> Option<(BlockId, BlockStatus)> {
        self.assert_reads_bookkeeping();

        let id = self.reads_done.pop_front()?;
        let read = self.reads[id].take().unwrap_or_else(|| unreachable!("done read is live"));
        assert!(read.done);
        self.reads_free.push_back(id);

        let block = BlockId {
            block_checksum: read.checksum,
            block_address: read.address,
            block_type: read.block_type,
        };
        self.assert_reads_bookkeeping();
        Some((block, read.status))
    }

    fn assert_reads_bookkeeping(&self) {
        assert_eq!(
            self.reads_busy.len() + self.reads_done.len() + self.reads_free.len(),
            self.reads.len()
        );
    }

    /// Advance the tour to the next block to scrub (upstream `tour_next`).
    ///
    /// Returns `None` at the end of a cycle (asserting the tour became `.done`) or while
    /// stalled on the free-set trailers (a grid open/checkpoint is in flight).
    #[allow(clippy::too_many_lines)] // one arm per tour stage, mirroring the upstream match
    fn tour_next(
        &mut self,
        grid: &Grid,
        manifest_log: &ManifestLog,
        next_table: &mut dyn FnMut() -> Option<(u64, u128)>,
    ) -> Option<BlockId> {
        // DEVIATION: `superblock.opened` is replaced by the manifest-log open gate (the
        // forest arms the table tour when open completes).
        assert!(manifest_log.is_opened());

        if matches!(self.tour, Tour::Init) {
            self.tour = Tour::TableIndex;
        }

        // Emit the next block of the current table's index/value tour. The index block is
        // `None` if it was corrupt, or if `GRID_SCRUBBER_READS_MAX > 1` and it is still in
        // flight — keep trying until a checkpoint removes it (see `on_read_done`).
        if let Tour::TableValue { index_checksum, index_address, value_block_index } = self.tour {
            let Some(index_block) = self.tour_index_block.as_deref() else {
                return Some(BlockId {
                    block_checksum: index_checksum,
                    block_address: index_address,
                    block_type: BlockType::Index,
                });
            };

            let index_schema = TableIndex::from_block_without_schema(index_block);
            let value_blocks_used = index_schema.value_blocks_used(index_block);
            if value_block_index < value_blocks_used {
                #[allow(clippy::cast_possible_truncation)] // `value_blocks_used` fits u32
                let address = index_schema.value_address(index_block, value_block_index as usize);
                #[allow(clippy::cast_possible_truncation)]
                let checksum = index_schema.value_checksum(index_block, value_block_index as usize);
                self.tour = Tour::TableValue {
                    index_checksum,
                    index_address,
                    value_block_index: value_block_index + 1,
                };
                return Some(BlockId {
                    block_checksum: checksum,
                    block_address: address,
                    block_type: BlockType::Value,
                });
            }
            assert_eq!(value_block_index, value_blocks_used);
            self.tour = Tour::TableIndex;
        }

        loop {
            match self.tour {
                Tour::Init | Tour::TableValue { .. } => unreachable!("handled above"),
                Tour::TableIndex => {
                    if let Some((address, checksum)) = next_table() {
                        // A fresh table: its index is not read yet (mirrors upstream
                        // defaulting `tour.table_value.index_block = null`).
                        self.tour_index_block = None;
                        self.tour = Tour::TableValue {
                            index_checksum: checksum,
                            index_address: address,
                            value_block_index: 0,
                        };
                        return Some(BlockId {
                            block_checksum: checksum,
                            block_address: address,
                            block_type: BlockType::Index,
                        });
                    }
                    self.tour = Tour::ManifestLog { iterator: ManifestBlockIterator::Init };
                }
                Tour::ManifestLog { mut iterator } => {
                    if let Some((address, checksum)) = iterator.next(manifest_log) {
                        self.tour = Tour::ManifestLog { iterator };
                        return Some(BlockId {
                            block_checksum: checksum,
                            block_address: address,
                            block_type: BlockType::Manifest,
                        });
                    }
                    self.tour = Tour::FreeSetAcquired { index: 0 };
                }
                Tour::FreeSetAcquired { mut index } => {
                    let free_set_trailer = grid.free_set_trailer(Slot::Acquired);
                    if free_set_trailer.callback != Callback::None {
                        return None; // grid open/checkpoint in flight; tour stalls.
                    }
                    if index < free_set_trailer.block_count() {
                        // A checkpoint can reduce the number of trailer blocks mid-tour.
                        index += 1;
                        self.tour = Tour::FreeSetAcquired { index };
                        return Some(BlockId {
                            block_checksum: free_set_trailer.checksums[index as usize - 1],
                            block_address: free_set_trailer.addresses[index as usize - 1],
                            block_type: BlockType::FreeSet,
                        });
                    }
                    self.tour = Tour::FreeSetReleased { index: 0 };
                }
                Tour::FreeSetReleased { mut index } => {
                    let free_set_trailer = grid.free_set_trailer(Slot::Released);
                    if free_set_trailer.callback != Callback::None {
                        return None;
                    }
                    if index < free_set_trailer.block_count() {
                        index += 1;
                        self.tour = Tour::FreeSetReleased { index };
                        return Some(BlockId {
                            block_checksum: free_set_trailer.checksums[index as usize - 1],
                            block_address: free_set_trailer.addresses[index as usize - 1],
                            block_type: BlockType::FreeSet,
                        });
                    }
                    self.tour = Tour::ClientSessions;
                }
                Tour::ClientSessions => {
                    // DEVIATION: no client-sessions checkpoint trailer is wired into the
                    // port, so the stage is always empty (upstream iterates
                    // `client_sessions_checkpoint` trailer blocks).
                    self.tour = Tour::Done;
                }
                Tour::Done => return None,
            }
        }
    }
}

impl ManifestBlockIterator {
    fn next(&mut self, manifest_log: &ManifestLog) -> Option<(u64, u128)> {
        // Don't scrub the trailing `blocks_closed`; they are not yet flushed to disk.
        let log_block_count =
            manifest_log.log_block_addresses().len() - manifest_log.blocks_closed() as usize;

        let position: Option<usize> = match *self {
            Self::Done => None,
            Self::Init => {
                if log_block_count == 0 {
                    None
                } else {
                    Some(log_block_count - 1)
                }
            }
            Self::State { index, address, checksum } => {
                // `index` may be beyond the limit due to blocks removed by manifest
                // compaction. Locate the previously-scrubbed block by its address/checksum,
                // then step one block older.
                if log_block_count == 0 {
                    None
                } else {
                    let mut position = (index as usize).min(log_block_count - 1);
                    let mut found = None;
                    while position > 0 {
                        if manifest_log.log_block_addresses()[position] == address
                            && manifest_log.log_block_checksums()[position] == checksum
                        {
                            found = Some(position - 1);
                            break;
                        }
                        position -= 1;
                    }
                    found
                }
            }
        };

        if let Some(index) = position {
            // `log_block_addresses` is sized by the manifest log's capacity (a u32).
            let index = u32::try_from(index).unwrap_or_else(|_| unreachable!());
            *self = Self::State {
                index,
                address: manifest_log.log_block_addresses()[index as usize],
                checksum: manifest_log.log_block_checksums()[index as usize],
            };
            Some((
                manifest_log.log_block_addresses()[index as usize],
                manifest_log.log_block_checksums()[index as usize],
            ))
        } else {
            *self = Self::Done;
            None
        }
    }
}
