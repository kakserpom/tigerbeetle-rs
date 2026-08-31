// =============================================================================
// Forest — owns all LSM grooves for a replica
// =============================================================================
//
// Ported from `src/lsm/forest.zig`. The Zig version uses comptime codegen to
// auto-generate a `Grooves` struct from a config. In Rust, we define the fields
// explicitly since there are only three grooves (accounts + transfers +
// transfers_pending).

use crate::groove::{
    AccountGroove, AccountGrooveScratch, TransferGroove, TransferGrooveScratch,
    TransferPendingGroove, TransferPendingGrooveScratch,
};

/// Forest owns all LSM grooves for a single replica.
///
/// This is a simplified version — full open/checkpoint (and the `manifest_log`,
/// which needs Grid + `SuperBlock` integration) are deferred; `TODO(port)`
/// markers below. The compaction seam is fully wired.
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
}

impl Forest {
    /// Construct a fresh [`Forest`] with empty grooves and scratch buffers, sized
    /// for `batch_value_count_limit` values per beat.
    #[must_use]
    pub fn init(batch_value_count_limit: u32) -> Self {
        Self {
            accounts: AccountGroove::new(batch_value_count_limit),
            transfers: TransferGroove::new(batch_value_count_limit),
            transfers_pending: TransferPendingGroove::new(batch_value_count_limit),
            accounts_scratch: AccountGrooveScratch::default(),
            transfers_scratch: TransferGrooveScratch::default(),
            transfers_pending_scratch: TransferPendingGrooveScratch::default(),
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
    use tigerbeetle_core::constants;
    use tigerbeetle_core::types::Account;

    /// `Forest::compact` drives each groove's `compact` with its scratch buffers,
    /// sorting + deduping the account object mutable table's suffix.
    ///
    /// Mirrors `tree::tests::tree_compact_sorts_and_dedups_the_mutable_suffix` at
    /// the forest level. The account object tree sorts by `timestamp` (its tree
    /// key), so out-of-order timestamps are put and the duplicate resolves to the
    /// latest put.
    #[test]
    fn forest_compact_sorts_and_dedups_groove_trees() {
        let mut forest = Forest::init(32);

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
        let mut forest = Forest::init(32);
        for op in 0..(constants::LSM_COMPACTION_OPS as u64) {
            forest.compact(op);
        }
    }
}
