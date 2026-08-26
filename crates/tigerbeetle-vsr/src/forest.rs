// =============================================================================
// Forest — owns all LSM grooves for a replica
// =============================================================================
//
// Ported from `src/lsm/forest.zig`. The Zig version uses comptime codegen to
// auto-generate a `Grooves` struct from a config. In Rust, we define the fields
// explicitly since there are only two grooves (accounts + transfers).

use crate::groove::{AccountGroove, TransferGroove, TransferPendingGroove};
use crate::manifest_log::ManifestLog;

/// Forest owns all LSM grooves for a single replica.
///
/// This is a simplified version — full init/open/compact/checkpoint deferred
/// until Grid and ManifestLog are fully integrated.
///
/// Upstream: `src/lsm/forest.zig:31` (`ForestType`).
pub struct Forest {
    pub accounts: AccountGroove,
    pub transfers: TransferGroove,
    pub transfers_pending: TransferPendingGroove,
    pub manifest_log: ManifestLog,
}

// The forest doesn't have runtime tests yet — lifecycle tests will come when
// Grid integration is complete.
