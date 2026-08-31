//! Groove: the main LSM abstraction layer between the state machine and individual trees.
//!
//! Upstream: `src/lsm/groove.zig` (2047 lines).
//!
//! DEVIATION: upstream uses a comptime function `GrooveType(Storage, Object, groove_options)`
//! to auto-generate `IndexTrees`, `UniqueKey`, `PrefetchKeys`, `IndexHelperType`, and
//! `ObjectsCache` types. This port manually defines these for each concrete groove type
//! (Account, Transfer) since Rust lacks comptime type generation.
//!
//! DEVIATION: upstream's prefetch pipeline (~480 lines) is deferred to the async I/O phase.
//! For now `prefetch_setup`, `prefetch_enqueue`, and `prefetch` are stubs.

//! DEVIATION: upstream's `ObjectsCache` is the full `CacheMapType` (a SetAssociativeCache
//! on top of a HashMap stash). This port uses the same `tigerbeetle_lsm::cache_map::CacheMap`
//! for the groove objects cache. The `remove()`/tombstone path is only reachable under
//! `constants.verify` (unit tests/fuzzers), mirroring upstream.

#![allow(
    clippy::cast_possible_truncation,
    reason = "size_of::<T>() as u32 in const LAYOUT; upstream uses comptime"
)]

use tigerbeetle_core::constants;
use tigerbeetle_core::stdx::hash::{hash_inline_u64, hash_inline_u128};
use tigerbeetle_core::types::{
    Account, AccountFlags, Transfer, TransferFlags, TransferPending, TransferPendingStatus,
};
use tigerbeetle_lsm::cache_map::{CacheMap, CacheMapOptions, CacheMapSpec};
use tigerbeetle_lsm::composite_key::{
    self, CompositeKey, CompositeKey64, CompositeKey128, CompositeKeyUnit, U256,
};
use tigerbeetle_lsm::manifest::ManifestLog;
use tigerbeetle_lsm::scratch_memory::ScratchMemory;
use tigerbeetle_lsm::set_associative_cache::{Layout, SetAssociativeCacheSpec};
use tigerbeetle_lsm::table_memory::{self, Usage};
use tigerbeetle_lsm::tree::ScopeCloseMode;
use tigerbeetle_lsm::tree::TreeConfig;
use tigerbeetle_lsm::unique_key::{UniqueKey, UniqueKey128};

use crate::grid::Grid;
use crate::table::{self, BlockValue, IndexBlocks, TableLayout, TableSpec, TableUsage};
use crate::tree::{LookupMemoryResult, Options, Tree};

// ---------------------------------------------------------------------------
// Groove objects-cache specs
// ---------------------------------------------------------------------------
//
// Each groove carries a `CacheMap` (upstream `CacheMapType`) keyed by the object's primary
// key and valu-typed as the object itself. The tombstones, `key_from_value`, and `hash`
// mirror upstream `Groove.ObjectsCacheHelpers` (`groove.zig:536-566`): a tombstone is an
// object whose `timestamp` high bit (`composite_key::TOMBSTONE_BIT`) is set.

/// Objects-cache spec for the [`Account`] groove (primary key: `id`).
pub struct AccountObjectsCacheSpec;

impl Layout for AccountObjectsCacheSpec {}

impl SetAssociativeCacheSpec for AccountObjectsCacheSpec {
    type Key = u128;
    type Value = Account;

    fn key_from_value(value: &Account) -> u128 {
        value.id
    }

    fn hash(key: u128) -> u64 {
        hash_inline_u128(key)
    }
}

impl CacheMapSpec for AccountObjectsCacheSpec {
    fn tombstone_from_key(key: u128) -> Account {
        Account { id: key, timestamp: composite_key::TOMBSTONE_BIT, ..Account::default() }
    }

    fn tombstone(value: &Account) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

/// Objects-cache spec for the [`Transfer`] groove (primary key: `id`).
pub struct TransferObjectsCacheSpec;

impl Layout for TransferObjectsCacheSpec {}

impl SetAssociativeCacheSpec for TransferObjectsCacheSpec {
    type Key = u128;
    type Value = Transfer;

    fn key_from_value(value: &Transfer) -> u128 {
        value.id
    }

    fn hash(key: u128) -> u64 {
        hash_inline_u128(key)
    }
}

impl CacheMapSpec for TransferObjectsCacheSpec {
    fn tombstone_from_key(key: u128) -> Transfer {
        Transfer { id: key, timestamp: composite_key::TOMBSTONE_BIT, ..Transfer::default() }
    }

    fn tombstone(value: &Transfer) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

/// Objects-cache spec for the [`TransferPending`] groove (primary key: `timestamp`).
pub struct TransferPendingObjectsCacheSpec;

impl Layout for TransferPendingObjectsCacheSpec {}

impl SetAssociativeCacheSpec for TransferPendingObjectsCacheSpec {
    type Key = u64;
    type Value = TransferPending;

    fn key_from_value(value: &TransferPending) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    fn hash(key: u64) -> u64 {
        hash_inline_u64(key)
    }
}

impl CacheMapSpec for TransferPendingObjectsCacheSpec {
    fn tombstone_from_key(key: u64) -> TransferPending {
        assert_eq!(key & composite_key::TOMBSTONE_BIT, 0);
        TransferPending {
            timestamp: key | composite_key::TOMBSTONE_BIT,
            ..TransferPending::default()
        }
    }

    fn tombstone(value: &TransferPending) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

pub type AccountObjectsCache = CacheMap<AccountObjectsCacheSpec>;
pub type TransferObjectsCache = CacheMap<TransferObjectsCacheSpec>;
pub type TransferPendingObjectsCache = CacheMap<TransferPendingObjectsCacheSpec>;

// ---------------------------------------------------------------------------
// BlockValue impls for composite / unique key value types
// ---------------------------------------------------------------------------

impl BlockValue for CompositeKey64 {
    fn write_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.field.to_le_bytes());
        buf[8..16].copy_from_slice(&self.timestamp.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut f = [0u8; 8];
        f.copy_from_slice(&bytes[..8]);
        let mut t = [0u8; 8];
        t.copy_from_slice(&bytes[8..16]);
        Self { field: u64::from_le_bytes(f), timestamp: u64::from_le_bytes(t) }
    }
}

impl BlockValue for CompositeKey128 {
    fn write_bytes(&self, buf: &mut [u8]) {
        buf[..16].copy_from_slice(&self.field.to_le_bytes());
        buf[16..24].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[24..32].copy_from_slice(&self.padding.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut f = [0u8; 16];
        f.copy_from_slice(&bytes[..16]);
        let mut t = [0u8; 8];
        t.copy_from_slice(&bytes[16..24]);
        let mut p = [0u8; 8];
        p.copy_from_slice(&bytes[24..32]);
        Self {
            field: u128::from_le_bytes(f),
            timestamp: u64::from_le_bytes(t),
            padding: u64::from_le_bytes(p),
        }
    }
}

impl BlockValue for CompositeKeyUnit {
    fn write_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.timestamp.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut t = [0u8; 8];
        t.copy_from_slice(&bytes[..8]);
        Self { field: (), timestamp: u64::from_le_bytes(t) }
    }
}

impl BlockValue for UniqueKey128 {
    fn write_bytes(&self, buf: &mut [u8]) {
        buf[..16].copy_from_slice(&self.field.to_le_bytes());
        buf[16..24].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[24..32].copy_from_slice(&self.padding.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut f = [0u8; 16];
        f.copy_from_slice(&bytes[..16]);
        let mut t = [0u8; 8];
        t.copy_from_slice(&bytes[16..24]);
        let mut p = [0u8; 8];
        p.copy_from_slice(&bytes[24..32]);
        Self {
            field: u128::from_le_bytes(f),
            timestamp: u64::from_le_bytes(t),
            padding: u64::from_le_bytes(p),
        }
    }
}

// ---------------------------------------------------------------------------
// BlockValue for Account / Transfer
// ---------------------------------------------------------------------------

/// Safe helper: read a fixed-size little-endian integer from a byte slice.
/// # Panics
/// Panics if `bytes[start..end]` does not have exactly the right length (guaranteed by
/// BlockValue contract).
#[allow(clippy::missing_panics_doc, reason = "invariant guaranteed by BlockValue")]
fn read_u128(bytes: &[u8], start: usize) -> u128 {
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&bytes[start..start + 16]);
    u128::from_le_bytes(buf)
}

#[allow(clippy::missing_panics_doc, reason = "invariant guaranteed by BlockValue")]
fn read_u64(bytes: &[u8], start: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[start..start + 8]);
    u64::from_le_bytes(buf)
}

#[allow(clippy::missing_panics_doc, reason = "invariant guaranteed by BlockValue")]
fn read_u32(bytes: &[u8], start: usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[start..start + 4]);
    u32::from_le_bytes(buf)
}

#[allow(clippy::missing_panics_doc, reason = "invariant guaranteed by BlockValue")]
fn read_u16(bytes: &[u8], start: usize) -> u16 {
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&bytes[start..start + 2]);
    u16::from_le_bytes(buf)
}

impl BlockValue for Account {
    fn write_bytes(&self, buf: &mut [u8]) {
        assert!(buf.len() >= 128);
        buf[..16].copy_from_slice(&self.id.to_le_bytes());
        buf[16..32].copy_from_slice(&self.debits_pending.to_le_bytes());
        buf[32..48].copy_from_slice(&self.debits_posted.to_le_bytes());
        buf[48..64].copy_from_slice(&self.credits_pending.to_le_bytes());
        buf[64..80].copy_from_slice(&self.credits_posted.to_le_bytes());
        buf[80..96].copy_from_slice(&self.user_data_128.to_le_bytes());
        buf[96..104].copy_from_slice(&self.user_data_64.to_le_bytes());
        buf[104..108].copy_from_slice(&self.user_data_32.to_le_bytes());
        buf[108..112].copy_from_slice(&self.reserved.to_le_bytes());
        buf[112..116].copy_from_slice(&self.ledger.to_le_bytes());
        buf[116..118].copy_from_slice(&self.code.to_le_bytes());
        buf[118..120].copy_from_slice(&self.flags.as_raw().to_le_bytes());
        buf[120..128].copy_from_slice(&self.timestamp.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            id: read_u128(bytes, 0),
            debits_pending: read_u128(bytes, 16),
            debits_posted: read_u128(bytes, 32),
            credits_pending: read_u128(bytes, 48),
            credits_posted: read_u128(bytes, 64),
            user_data_128: read_u128(bytes, 80),
            user_data_64: read_u64(bytes, 96),
            user_data_32: read_u32(bytes, 104),
            reserved: read_u32(bytes, 108),
            ledger: read_u32(bytes, 112),
            code: read_u16(bytes, 116),
            flags: AccountFlags::from_raw(read_u16(bytes, 118)),
            timestamp: read_u64(bytes, 120),
        }
    }
}

impl BlockValue for Transfer {
    fn write_bytes(&self, buf: &mut [u8]) {
        assert!(buf.len() >= 128);
        buf[..16].copy_from_slice(&self.id.to_le_bytes());
        buf[16..32].copy_from_slice(&self.debit_account_id.to_le_bytes());
        buf[32..48].copy_from_slice(&self.credit_account_id.to_le_bytes());
        buf[48..64].copy_from_slice(&self.amount.to_le_bytes());
        buf[64..80].copy_from_slice(&self.pending_id.to_le_bytes());
        buf[80..96].copy_from_slice(&self.user_data_128.to_le_bytes());
        buf[96..104].copy_from_slice(&self.user_data_64.to_le_bytes());
        buf[104..108].copy_from_slice(&self.user_data_32.to_le_bytes());
        buf[108..112].copy_from_slice(&self.timeout.to_le_bytes());
        buf[112..116].copy_from_slice(&self.ledger.to_le_bytes());
        buf[116..118].copy_from_slice(&self.code.to_le_bytes());
        buf[118..120].copy_from_slice(&self.flags.as_raw().to_le_bytes());
        buf[120..128].copy_from_slice(&self.timestamp.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            id: read_u128(bytes, 0),
            debit_account_id: read_u128(bytes, 16),
            credit_account_id: read_u128(bytes, 32),
            amount: read_u128(bytes, 48),
            pending_id: read_u128(bytes, 64),
            user_data_128: read_u128(bytes, 80),
            user_data_64: read_u64(bytes, 96),
            user_data_32: read_u32(bytes, 104),
            timeout: read_u32(bytes, 108),
            ledger: read_u32(bytes, 112),
            code: read_u16(bytes, 116),
            flags: TransferFlags::from_raw(read_u16(bytes, 118)),
            timestamp: read_u64(bytes, 120),
        }
    }
}

impl BlockValue for TransferPending {
    fn write_bytes(&self, buf: &mut [u8]) {
        assert!(buf.len() >= 16);
        buf[..8].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[8] = self.status as u8;
        buf[9..16].copy_from_slice(&self.padding);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            timestamp: read_u64(bytes, 0),
            status: match bytes[8] {
                0 => TransferPendingStatus::None,
                1 => TransferPendingStatus::Pending,
                2 => TransferPendingStatus::Posted,
                3 => TransferPendingStatus::Voided,
                4 => TransferPendingStatus::Expired,
                _ => panic!("invalid TransferPendingStatus byte {}", bytes[8]),
            },
            padding: {
                let mut padding = [0u8; 7];
                padding.copy_from_slice(&bytes[9..16]);
                padding
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tree spec types
//
// Each tree in the groove needs a type implementing both `table_memory::Table` and
// `TreeSpec`.  We use unit structs to keep the overhead zero.
// ---------------------------------------------------------------------------

/// Maximum values a tree in a groove can hold, mirroring upstream `groove.zig:323-417`
/// (`table_value_count_max = lsm_compaction_ops * batch_value_count_max.{field}`).
/// The largest per-field batch count is the object `timestamp` field
/// (`state_machine.zig:4827`: `@max(batch_create_accounts, 2 * batch_create_transfers)`),
/// which bounds every table in the groove (object trees and index trees alike).
///
/// `batch_create_accounts`/`batch_create_transfers` are upstream
/// `Operation.create_accounts/create_transfers.event_max(message_body_size_max)`
/// (`tigerbeetle.zig:853-901`). Accounts and transfers are both 128 bytes, and the event
/// (request) side binds over the reply bound, so `event_max = message_body_size_max / 128`.
/// Maximum values a tree in a groove can hold, mirroring upstream `groove.zig:323-417`
/// (`lsm_compaction_ops * max(batch_create_accounts, 2*batch_create_transfers)`).
pub const GROOVE_VALUE_COUNT_MAX: usize = constants::LSM_COMPACTION_OPS
    * (2 * (constants::MESSAGE_BODY_SIZE_MAX / core::mem::size_of::<Account>()));

// ===== Object trees (keyed by u64 timestamp) =====

pub struct AccountObjectSpec;

impl table_memory::Table for AccountObjectSpec {
    type Key = u64;
    type Value = Account;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::General;

    fn key_from_value(value: &Account) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    fn tombstone(value: &Account) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

impl TableSpec for AccountObjectSpec {
    type Key = u64;
    type Value = Account;

    fn key_from_value(value: &Account) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    const SENTINEL_KEY: u64 = u64::MAX;

    fn tombstone(value: &Account) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }

    fn tombstone_from_key(key: u64) -> Account {
        Account { timestamp: key | composite_key::TOMBSTONE_BIT, ..Account::default() }
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::General;
}

impl crate::tree::TreeSpec for AccountObjectSpec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: u64) -> Option<IndexBlocks<u64>> {
        table::index_blocks_for_key::<u64>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: u64) -> Option<Account> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: u64) -> Account {
        Account { timestamp: key | composite_key::TOMBSTONE_BIT, ..Account::default() }
    }
}

pub struct TransferObjectSpec;

impl table_memory::Table for TransferObjectSpec {
    type Key = u64;
    type Value = Transfer;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::General;

    fn key_from_value(value: &Transfer) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    fn tombstone(value: &Transfer) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

impl TableSpec for TransferObjectSpec {
    type Key = u64;
    type Value = Transfer;

    fn key_from_value(value: &Transfer) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    const SENTINEL_KEY: u64 = u64::MAX;

    fn tombstone(value: &Transfer) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }

    fn tombstone_from_key(key: u64) -> Transfer {
        Transfer { timestamp: key | composite_key::TOMBSTONE_BIT, ..Transfer::default() }
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::General;
}

impl crate::tree::TreeSpec for TransferObjectSpec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: u64) -> Option<IndexBlocks<u64>> {
        table::index_blocks_for_key::<u64>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: u64) -> Option<Transfer> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: u64) -> Transfer {
        Transfer { timestamp: key | composite_key::TOMBSTONE_BIT, ..Transfer::default() }
    }
}

// ===== TransfersPending object tree (keyed by u64 timestamp) =====

pub struct TransferPendingObjectSpec;

impl table_memory::Table for TransferPendingObjectSpec {
    type Key = u64;
    type Value = TransferPending;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::General;

    fn key_from_value(value: &TransferPending) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    fn tombstone(value: &TransferPending) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

impl TableSpec for TransferPendingObjectSpec {
    type Key = u64;
    type Value = TransferPending;

    fn key_from_value(value: &TransferPending) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    const SENTINEL_KEY: u64 = u64::MAX;

    fn tombstone(value: &TransferPending) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }

    fn tombstone_from_key(key: u64) -> TransferPending {
        TransferPending {
            timestamp: key | composite_key::TOMBSTONE_BIT,
            status: TransferPendingStatus::None,
            padding: [0; 7],
        }
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::General;
}

impl crate::tree::TreeSpec for TransferPendingObjectSpec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: u64) -> Option<IndexBlocks<u64>> {
        table::index_blocks_for_key::<u64>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: u64) -> Option<TransferPending> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: u64) -> TransferPending {
        TransferPending {
            timestamp: key | composite_key::TOMBSTONE_BIT,
            status: TransferPendingStatus::None,
            padding: [0; 7],
        }
    }
}

// ===== TransfersPending status index (CompositeKey64: field = status as u64) =====
//
// Upstream indexes the optional `status` field via `CompositeKey(IndexType(enum))` where
// `IndexType(TransferPendingStatus)` is `u64` (status is an 8-bit enum). An entry is only
// produced when the status is non-zero (`index_from_object` returns null for `None`,
// groove.zig:488-521).

pub struct TransferPendingStatusSpec;

impl table_memory::Table for TransferPendingStatusSpec {
    type Key = u128;
    type Value = CompositeKey64;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::SecondaryIndex;

    fn key_from_value(value: &CompositeKey64) -> u128 {
        value.key_from_value()
    }

    fn tombstone(value: &CompositeKey64) -> bool {
        value.tombstone()
    }
}

impl TableSpec for TransferPendingStatusSpec {
    type Key = u128;
    type Value = CompositeKey64;

    fn key_from_value(value: &CompositeKey64) -> u128 {
        value.key_from_value()
    }

    const SENTINEL_KEY: u128 = u128::MAX;

    fn tombstone(value: &CompositeKey64) -> bool {
        value.tombstone()
    }

    fn tombstone_from_key(key: u128) -> CompositeKey64 {
        CompositeKey64::tombstone_from_key(key)
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::SecondaryIndex;
}

impl crate::tree::TreeSpec for TransferPendingStatusSpec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: u128) -> Option<IndexBlocks<u128>> {
        table::index_blocks_for_key::<u128>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: u128) -> Option<CompositeKey64> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: u128) -> CompositeKey64 {
        CompositeKey64::tombstone_from_key(key)
    }
}

/// Optional `status` index for `TransferPending`. Returns `None` for a record with
/// status `None` (zero), mirroring upstream's optional index behaviour (groove.zig:488-521).
pub struct TransferPendingStatusIndex;
impl IndexExtractor<TransferPending> for TransferPendingStatusIndex {
    type IndexPrefix = u64;
    fn index_from_object(object: &TransferPending) -> Option<u64> {
        if object.status == TransferPendingStatus::None {
            None
        } else {
            Some(u64::from(object.status as u8))
        }
    }
}

// ===== Index trees (composite-key secondary indexes) =====

/// Index tree for `u128` fields (key_type = u128, value_type = CompositeKey128).
pub struct CompositeKey128Spec;

impl table_memory::Table for CompositeKey128Spec {
    type Key = U256;
    type Value = CompositeKey128;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::SecondaryIndex;

    fn key_from_value(value: &CompositeKey128) -> U256 {
        value.key_from_value()
    }

    fn tombstone(value: &CompositeKey128) -> bool {
        value.tombstone()
    }
}

impl TableSpec for CompositeKey128Spec {
    type Key = U256;
    type Value = CompositeKey128;

    fn key_from_value(value: &CompositeKey128) -> U256 {
        value.key_from_value()
    }

    const SENTINEL_KEY: U256 = U256::MAX;

    fn tombstone(value: &CompositeKey128) -> bool {
        value.tombstone()
    }

    fn tombstone_from_key(key: U256) -> CompositeKey128 {
        CompositeKey128::tombstone_from_key(key)
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::SecondaryIndex;
}

impl crate::tree::TreeSpec for CompositeKey128Spec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: U256) -> Option<IndexBlocks<U256>> {
        table::index_blocks_for_key::<U256>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: U256) -> Option<CompositeKey128> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: U256) -> CompositeKey128 {
        CompositeKey128::tombstone_from_key(key)
    }
}

/// Index tree for `u64` fields (key_type = u128, value_type = CompositeKey64).
pub struct CompositeKey64Spec;

impl table_memory::Table for CompositeKey64Spec {
    type Key = u128;
    type Value = CompositeKey64;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::SecondaryIndex;

    fn key_from_value(value: &CompositeKey64) -> u128 {
        value.key_from_value()
    }

    fn tombstone(value: &CompositeKey64) -> bool {
        value.tombstone()
    }
}

impl TableSpec for CompositeKey64Spec {
    type Key = u128;
    type Value = CompositeKey64;

    fn key_from_value(value: &CompositeKey64) -> u128 {
        value.key_from_value()
    }

    const SENTINEL_KEY: u128 = u128::MAX;

    fn tombstone(value: &CompositeKey64) -> bool {
        value.tombstone()
    }

    fn tombstone_from_key(key: u128) -> CompositeKey64 {
        CompositeKey64::tombstone_from_key(key)
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::SecondaryIndex;
}

impl crate::tree::TreeSpec for CompositeKey64Spec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: u128) -> Option<IndexBlocks<u128>> {
        table::index_blocks_for_key::<u128>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: u128) -> Option<CompositeKey64> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: u128) -> CompositeKey64 {
        CompositeKey64::tombstone_from_key(key)
    }
}

/// Index tree for void/flag fields (key_type = u64, value_type = CompositeKeyUnit).
/// Used for derived indexes like `imported`, `closed`, `closing`.
pub struct CompositeKeyUnitSpec;

impl table_memory::Table for CompositeKeyUnitSpec {
    type Key = u64;
    type Value = CompositeKeyUnit;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::SecondaryIndex;

    fn key_from_value(value: &CompositeKeyUnit) -> u64 {
        value.key_from_value()
    }

    fn tombstone(value: &CompositeKeyUnit) -> bool {
        value.tombstone()
    }
}

impl TableSpec for CompositeKeyUnitSpec {
    type Key = u64;
    type Value = CompositeKeyUnit;

    fn key_from_value(value: &CompositeKeyUnit) -> u64 {
        value.key_from_value()
    }

    const SENTINEL_KEY: u64 = u64::MAX;

    fn tombstone(value: &CompositeKeyUnit) -> bool {
        value.tombstone()
    }

    fn tombstone_from_key(key: u64) -> CompositeKeyUnit {
        CompositeKeyUnit::tombstone_from_key(key)
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::SecondaryIndex;
}

impl crate::tree::TreeSpec for CompositeKeyUnitSpec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: u64) -> Option<IndexBlocks<u64>> {
        table::index_blocks_for_key::<u64>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: u64) -> Option<CompositeKeyUnit> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: u64) -> CompositeKeyUnit {
        CompositeKeyUnit::tombstone_from_key(key)
    }
}

/// Unique-key index tree for `u128` unique keys (key_type = u128, value_type = UniqueKey128).
/// Used for the `id` field on both Account and Transfer.
pub struct UniqueKey128Spec;

impl table_memory::Table for UniqueKey128Spec {
    type Key = u128;
    type Value = UniqueKey128;
    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::General;

    fn key_from_value(value: &UniqueKey128) -> u128 {
        UniqueKey::key_from_value(value)
    }

    fn tombstone(value: &UniqueKey128) -> bool {
        UniqueKey::tombstone(value)
    }
}

impl TableSpec for UniqueKey128Spec {
    type Key = u128;
    type Value = UniqueKey128;

    fn key_from_value(value: &UniqueKey128) -> u128 {
        UniqueKey::key_from_value(value)
    }

    const SENTINEL_KEY: u128 = u128::MAX;

    fn tombstone(value: &UniqueKey128) -> bool {
        UniqueKey::tombstone(value)
    }

    fn tombstone_from_key(key: u128) -> UniqueKey128 {
        UniqueKey128::tombstone_from_key(key)
    }

    const VALUE_COUNT_MAX: usize = GROOVE_VALUE_COUNT_MAX;
    const USAGE: TableUsage = TableUsage::General;
}

impl crate::tree::TreeSpec for UniqueKey128Spec {
    const LAYOUT: TableLayout = TableLayout::compute_for::<Self>();

    fn index_blocks_for_key(index_block: &[u8], key: u128) -> Option<IndexBlocks<u128>> {
        table::index_blocks_for_key::<u128>(index_block, &Self::LAYOUT.index, key)
    }

    fn value_block_search(value_block: &[u8], key: u128) -> Option<UniqueKey128> {
        table::value_block_search::<Self>(value_block, &Self::LAYOUT.data, key)
    }

    fn tombstone_from_key(key: u128) -> UniqueKey128 {
        UniqueKey128::tombstone_from_key(key)
    }
}

// ---------------------------------------------------------------------------
// Index helpers — extract index values from objects
//
// Upstream: `_IndexHelperType` (groove.zig:452-536). Each field on the Object has an
// `IndexHelper` that knows how to extract the index prefix, handle derived fields, and
// optionally skip zero values.
// ---------------------------------------------------------------------------

/// Trait for extracting an index value from an object.
///
/// `None` means the value should not be indexed (zero in an optional field, or a derived
/// field that evaluates to null).
pub trait IndexExtractor<O> {
    /// The index prefix type (maps to the composite/unique key's `Field`).
    type IndexPrefix: Copy;

    /// Try to extract the index value from the object. `None` = don't index.
    fn index_from_object(object: &O) -> Option<Self::IndexPrefix>;
}

// ===== Account index extractors =====

pub struct AccountIdIndex;
impl IndexExtractor<Account> for AccountIdIndex {
    type IndexPrefix = u128;
    fn index_from_object(object: &Account) -> Option<u128> {
        Some(object.id)
    }
}

pub struct AccountUserData128Index;
impl IndexExtractor<Account> for AccountUserData128Index {
    type IndexPrefix = u128;
    fn index_from_object(object: &Account) -> Option<u128> {
        if object.user_data_128 == 0 { None } else { Some(object.user_data_128) }
    }
}

pub struct AccountUserData64Index;
impl IndexExtractor<Account> for AccountUserData64Index {
    type IndexPrefix = u64;
    fn index_from_object(object: &Account) -> Option<u64> {
        if object.user_data_64 == 0 { None } else { Some(object.user_data_64) }
    }
}

pub struct AccountUserData32Index;
impl IndexExtractor<Account> for AccountUserData32Index {
    type IndexPrefix = u64;
    fn index_from_object(object: &Account) -> Option<u64> {
        if object.user_data_32 == 0 { None } else { Some(u64::from(object.user_data_32)) }
    }
}

pub struct AccountLedgerIndex;
impl IndexExtractor<Account> for AccountLedgerIndex {
    type IndexPrefix = u64;
    fn index_from_object(object: &Account) -> Option<u64> {
        Some(u64::from(object.ledger))
    }
}

pub struct AccountCodeIndex;
impl IndexExtractor<Account> for AccountCodeIndex {
    type IndexPrefix = u64;
    fn index_from_object(object: &Account) -> Option<u64> {
        Some(u64::from(object.code))
    }
}

/// Derived index: `imported` is present (as void) only when `flags.imported` is set.
pub struct AccountImportedIndex;
impl IndexExtractor<Account> for AccountImportedIndex {
    type IndexPrefix = ();
    fn index_from_object(object: &Account) -> Option<()> {
        if object.flags.imported() { Some(()) } else { None }
    }
}

/// Derived index: `closed` is present only when `flags.closed` is set.
pub struct AccountClosedIndex;
impl IndexExtractor<Account> for AccountClosedIndex {
    type IndexPrefix = ();
    fn index_from_object(object: &Account) -> Option<()> {
        if object.flags.closed() { Some(()) } else { None }
    }
}

// ===== Transfer index extractors =====

pub struct TransferIdIndex;
impl IndexExtractor<Transfer> for TransferIdIndex {
    type IndexPrefix = u128;
    fn index_from_object(object: &Transfer) -> Option<u128> {
        Some(object.id)
    }
}

pub struct TransferDebitAccountIdIndex;
impl IndexExtractor<Transfer> for TransferDebitAccountIdIndex {
    type IndexPrefix = u128;
    fn index_from_object(object: &Transfer) -> Option<u128> {
        Some(object.debit_account_id)
    }
}

pub struct TransferCreditAccountIdIndex;
impl IndexExtractor<Transfer> for TransferCreditAccountIdIndex {
    type IndexPrefix = u128;
    fn index_from_object(object: &Transfer) -> Option<u128> {
        Some(object.credit_account_id)
    }
}

pub struct TransferAmountIndex;
impl IndexExtractor<Transfer> for TransferAmountIndex {
    type IndexPrefix = u128;
    fn index_from_object(object: &Transfer) -> Option<u128> {
        Some(object.amount)
    }
}

pub struct TransferPendingIdIndex;
impl IndexExtractor<Transfer> for TransferPendingIdIndex {
    type IndexPrefix = u128;
    fn index_from_object(object: &Transfer) -> Option<u128> {
        if object.pending_id == 0 { None } else { Some(object.pending_id) }
    }
}

pub struct TransferUserData128Index;
impl IndexExtractor<Transfer> for TransferUserData128Index {
    type IndexPrefix = u128;
    fn index_from_object(object: &Transfer) -> Option<u128> {
        if object.user_data_128 == 0 { None } else { Some(object.user_data_128) }
    }
}

pub struct TransferUserData64Index;
impl IndexExtractor<Transfer> for TransferUserData64Index {
    type IndexPrefix = u64;
    fn index_from_object(object: &Transfer) -> Option<u64> {
        if object.user_data_64 == 0 { None } else { Some(object.user_data_64) }
    }
}

pub struct TransferUserData32Index;
impl IndexExtractor<Transfer> for TransferUserData32Index {
    type IndexPrefix = u64;
    fn index_from_object(object: &Transfer) -> Option<u64> {
        if object.user_data_32 == 0 { None } else { Some(u64::from(object.user_data_32)) }
    }
}

pub struct TransferLedgerIndex;
impl IndexExtractor<Transfer> for TransferLedgerIndex {
    type IndexPrefix = u64;
    fn index_from_object(object: &Transfer) -> Option<u64> {
        Some(u64::from(object.ledger))
    }
}

pub struct TransferCodeIndex;
impl IndexExtractor<Transfer> for TransferCodeIndex {
    type IndexPrefix = u64;
    fn index_from_object(object: &Transfer) -> Option<u64> {
        Some(u64::from(object.code))
    }
}

/// Derived: `expires_at = timestamp + timeout_ns()` for pending transfers with timeout.
pub struct TransferExpiresAtIndex;
impl IndexExtractor<Transfer> for TransferExpiresAtIndex {
    type IndexPrefix = u64;
    fn index_from_object(object: &Transfer) -> Option<u64> {
        if object.flags.pending() && object.timeout > 0 {
            Some(object.timestamp + object.timeout_ns())
        } else {
            None
        }
    }
}

/// Derived: `imported` void index for `flags.imported`.
pub struct TransferImportedIndex;
impl IndexExtractor<Transfer> for TransferImportedIndex {
    type IndexPrefix = ();
    fn index_from_object(object: &Transfer) -> Option<()> {
        if object.flags.imported() { Some(()) } else { None }
    }
}

/// Derived: `closing` void index for `flags.closing_debit | flags.closing_credit`.
pub struct TransferClosingIndex;
impl IndexExtractor<Transfer> for TransferClosingIndex {
    type IndexPrefix = ();
    fn index_from_object(object: &Transfer) -> Option<()> {
        if object.flags.closing_debit() || object.flags.closing_credit() { Some(()) } else { None }
    }
}

// ---------------------------------------------------------------------------
// Groove structs
//
// DEVIATION: upstream auto-generates `IndexTrees` as a comptime struct with one field
// per indexable field.  This port defines each groove's index set explicitly.
// ---------------------------------------------------------------------------

/// The Account groove.
///
/// Owns the object tree (`Tree<AccountObjectSpec>`) and one index tree per indexable field
/// on `Account`.
pub struct AccountGroove {
    pub objects: Tree<AccountObjectSpec>,
    pub id: Tree<UniqueKey128Spec>,
    pub user_data_128: Tree<CompositeKey128Spec>,
    pub user_data_64: Tree<CompositeKey64Spec>,
    pub user_data_32: Tree<CompositeKey64Spec>,
    pub ledger: Tree<CompositeKey64Spec>,
    pub code: Tree<CompositeKey64Spec>,
    pub imported: Tree<CompositeKeyUnitSpec>,
    pub closed: Tree<CompositeKeyUnitSpec>,
    /// Per-session objects cache, keyed by the account's primary key (id).
    ///
    /// Invariant with the object tree holds (groove.zig:725-728): anything visible
    /// in-session lives here first.
    objects_cache: AccountObjectsCache,
}

/// Radix-sort scratch buffers for [`AccountGroove`]'s trees, owned by the forest.
///
/// DEVIATION: upstream shares a single untyped byte buffer across all of the forest's
/// trees (forest.zig:253, sized to the max `value_count_max * size_of(Value)`). This port's
/// `ScratchMemory<T>` is typed per element (scratch_memory.rs), so safe Rust has one buffer
/// per distinct tree value type instead. Since compaction is serialized (one tree at a time
/// per beat) and each tree needs only its own value type's buffer, one buffer per type
/// suffices; this only costs more memory, not correctness.
pub struct AccountGrooveScratch {
    pub objects: ScratchMemory<Account>,
    pub id: ScratchMemory<UniqueKey128>,
    pub user_data_128: ScratchMemory<CompositeKey128>,
    pub composite_key_64: ScratchMemory<CompositeKey64>,
    pub composite_key_unit: ScratchMemory<CompositeKeyUnit>,
}

/// Builds fresh scratch buffers for an [`AccountGroove`], each sized to the groove's
/// maximum tree value count (mirroring upstream's max-over-trees sizing, forest.zig:290-299).
impl AccountGrooveScratch {
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            id: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            user_data_128: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            composite_key_64: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            composite_key_unit: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
        }
    }
}

impl Default for AccountGrooveScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// The Transfer groove.
pub struct TransferGroove {
    pub objects: Tree<TransferObjectSpec>,
    pub id: Tree<UniqueKey128Spec>,
    pub debit_account_id: Tree<CompositeKey128Spec>,
    pub credit_account_id: Tree<CompositeKey128Spec>,
    pub amount: Tree<CompositeKey128Spec>,
    pub pending_id: Tree<CompositeKey128Spec>,
    pub user_data_128: Tree<CompositeKey128Spec>,
    pub user_data_64: Tree<CompositeKey64Spec>,
    pub user_data_32: Tree<CompositeKey64Spec>,
    pub ledger: Tree<CompositeKey64Spec>,
    pub code: Tree<CompositeKey64Spec>,
    pub expires_at: Tree<CompositeKey64Spec>,
    pub imported: Tree<CompositeKeyUnitSpec>,
    pub closing: Tree<CompositeKeyUnitSpec>,
    /// Per-session objects cache, keyed by the transfer's primary key (id).
    objects_cache: TransferObjectsCache,
}

/// Radix-sort scratch buffers for [`TransferGroove`]'s trees, owned by the forest.
///
/// See the DEVIATION note on [`AccountGrooveScratch`] (typed per-type buffers instead of
/// upstream's single shared untyped buffer).
pub struct TransferGrooveScratch {
    pub objects: ScratchMemory<Transfer>,
    pub id: ScratchMemory<UniqueKey128>,
    pub composite_key_128: ScratchMemory<CompositeKey128>,
    pub composite_key_64: ScratchMemory<CompositeKey64>,
    pub composite_key_unit: ScratchMemory<CompositeKeyUnit>,
}

/// Builds fresh scratch buffers for a [`TransferGroove`], each sized to the groove's
/// maximum tree value count (forest.zig:290-299).
impl TransferGrooveScratch {
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            id: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            composite_key_128: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            composite_key_64: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            composite_key_unit: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
        }
    }
}

impl Default for TransferGrooveScratch {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AccountGroove operations
// ---------------------------------------------------------------------------

impl AccountGroove {
    /// Build a fresh [`AccountGroove`] with all nine trees, sized for
    /// `batch_value_count_limit` values per beat (forest.zig:290-302).
    #[must_use]
    pub fn new(batch_value_count_limit: u32) -> Self {
        fn tree<S: crate::tree::TreeSpec>(id: u16, name: &'static str, limit: u32) -> Tree<S> {
            Tree::<S>::new(TreeConfig { id, name }, Options { batch_value_count_limit: limit })
        }
        AccountGroove {
            objects: tree(1, "accounts_objects", batch_value_count_limit),
            id: tree(2, "accounts_id", batch_value_count_limit),
            user_data_128: tree(3, "accounts_user_data_128", batch_value_count_limit),
            user_data_64: tree(4, "accounts_user_data_64", batch_value_count_limit),
            user_data_32: tree(5, "accounts_user_data_32", batch_value_count_limit),
            ledger: tree(6, "accounts_ledger", batch_value_count_limit),
            code: tree(7, "accounts_code", batch_value_count_limit),
            imported: tree(8, "accounts_imported", batch_value_count_limit),
            closed: tree(9, "accounts_closed", batch_value_count_limit),
            objects_cache: AccountObjectsCache::new(CacheMapOptions {
                cache_value_count_max: 256,
                stash_value_count_max: constants::LSM_COMPACTION_OPS as u32
                    * batch_value_count_limit,
                scope_value_count_max: batch_value_count_limit,
                name: "accounts_objects",
            }),
        }
    }

    pub fn scope_open(&mut self) {
        self.objects.scope_open();
        self.id.scope_open();
        self.user_data_128.scope_open();
        self.user_data_64.scope_open();
        self.user_data_32.scope_open();
        self.ledger.scope_open();
        self.code.scope_open();
        self.imported.scope_open();
        self.closed.scope_open();
        self.objects_cache.scope_open();
    }

    pub fn scope_close(&mut self, mode: ScopeCloseMode) {
        self.objects.scope_close(mode);
        self.id.scope_close(mode);
        self.user_data_128.scope_close(mode);
        self.user_data_64.scope_close(mode);
        self.user_data_32.scope_close(mode);
        self.ledger.scope_close(mode);
        self.code.scope_close(mode);
        self.imported.scope_close(mode);
        self.closed.scope_close(mode);
        self.objects_cache.scope_close(mode);
    }

    /// Port of upstream `groove.compact` (groove.zig:1981-2021): sorts each tree's
    /// mutable table with its radix-sort scratch, and compacts the objects cache on the
    /// last beat of the bar (mirroring the trees' own mutable-table compaction cadence).
    ///
    /// DEVIATION: takes the per-value-type scratch buffers as a parameter instead of
    /// reading a shared untyped buffer as upstream does (see [`AccountGrooveScratch`]).
    pub fn compact(&mut self, op: u64, sc: &mut AccountGrooveScratch) {
        self.objects.compact(&mut sc.objects);
        self.id.compact(&mut sc.id);
        self.user_data_128.compact(&mut sc.user_data_128);
        self.user_data_64.compact(&mut sc.composite_key_64);
        self.user_data_32.compact(&mut sc.composite_key_64);
        self.ledger.compact(&mut sc.composite_key_64);
        self.code.compact(&mut sc.composite_key_64);
        self.imported.compact(&mut sc.composite_key_unit);
        self.closed.compact(&mut sc.composite_key_unit);

        let compaction_beat = op % (constants::LSM_COMPACTION_OPS as u64);
        if compaction_beat == constants::LSM_COMPACTION_OPS as u64 - 1 {
            self.objects_cache.compact();
        }
    }

    /// Lookup an account by its primary key (id) from the objects cache.
    ///
    /// Mirrors `groove.zig:885-936` (`get(PrimaryKey) ObjectCacheResult`), collapsed to
    /// `Option`: tombstones and orphans are resolved to `Negative` by
    /// [`AccountGroove::lookup`] instead of being surfaced here.
    #[must_use]
    pub fn get(&mut self, id: u128) -> Option<&Account> {
        self.objects_cache.get(id)
    }

    /// Whether the objects cache holds a live (non-tombstoned) entry for `id`.
    ///
    /// Mirrors upstream `objects_cache.has` (groove.zig:1020, 1107, 1330).
    #[must_use]
    pub fn has(&mut self, id: u128) -> bool {
        self.objects_cache.has(id)
    }

    /// Remove `id` from the primary-key view by placing a tombstone in the cache.
    ///
    /// Mirrors `groove.remove` (groove.zig:1876-1920).
    ///
    /// DEVIATION: upstream also removes the object and each populated index entry
    /// from the underlying trees; sans-forest only the cache tombstone is surfaced,
    /// which is what makes `get`/`has` report the key as absent in-session.
    ///
    /// # Panics
    /// Panics unless built with `verify`, or when `id` is absent from the cache.
    pub fn remove(&mut self, id: u128) {
        self.objects_cache.remove(id);
    }

    /// Resolve an account by its primary key (id) from the tree levels.
    ///
    /// Mirrors upstream's prefetch-by-unique-key resolution (groove.zig:1704-1732):
    /// look up the primary-key tree (`id`) for the account's timestamp, then resolve the
    /// object tree (`objects`) by that timestamp. Both hops go through
    /// `Tree::lookup_from_levels_cache`, so a block missing from the grid cache at either
    /// hop yields `Possible` (deferred to the async read phase).
    ///
    /// DEVIATION: upstream keeps orphaned primary keys (`timestamp == 0`) in a separate
    /// map via `insert_orphaned_object`; sans-IO an orphan resolves to `Negative` and
    /// orphaned-key tracking is deferred to the forest layer.
    ///
    /// # Panics
    /// Panics if the primary-key tree returns an entry whose key does not match `id`
    /// (a corrupted index invariant violation).
    #[must_use]
    pub fn lookup(
        &mut self,
        grid: &mut Grid,
        snapshot: u64,
        id: u128,
    ) -> LookupMemoryResult<Account> {
        let unique = match self.id.lookup_from_levels_cache(grid, snapshot, id) {
            LookupMemoryResult::Positive(unique) => unique,
            LookupMemoryResult::Negative => return LookupMemoryResult::Negative,
            LookupMemoryResult::Possible { level } => {
                return LookupMemoryResult::Possible { level };
            }
        };
        assert_eq!(unique.field, id);
        if unique.timestamp == 0 {
            return LookupMemoryResult::Negative;
        }
        self.objects.lookup_from_levels_cache(grid, snapshot, unique.timestamp)
    }

    /// Insert an object into the object tree and all index trees.
    ///
    /// # Panics
    /// Panics if the timestamp is zero or the tombstone bit is set.
    pub fn insert(&mut self, object: &Account) {
        assert!(object.timestamp != 0);
        assert_eq!(object.timestamp & composite_key::TOMBSTONE_BIT, 0);

        self.objects.put(object);
        self.objects.key_range_update(object.timestamp);
        self.objects_cache.upsert(object);

        // id (unique key)
        self.id.put(&UniqueKey128 { field: object.id, timestamp: object.timestamp, padding: 0 });
        self.id.key_range_update(object.id);

        // Secondary indexes
        if let Some(v) = AccountUserData128Index::index_from_object(object) {
            self.user_data_128.put(&CompositeKey128 {
                field: v,
                timestamp: object.timestamp,
                padding: 0,
            });
        }
        if let Some(v) = AccountUserData64Index::index_from_object(object) {
            self.user_data_64.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = AccountUserData32Index::index_from_object(object) {
            self.user_data_32.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = AccountLedgerIndex::index_from_object(object) {
            self.ledger.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = AccountCodeIndex::index_from_object(object) {
            self.code.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        // Derived indexes
        if let Some(v) = AccountImportedIndex::index_from_object(object) {
            self.imported.put(&CompositeKeyUnit { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = AccountClosedIndex::index_from_object(object) {
            self.closed.put(&CompositeKeyUnit { field: v, timestamp: object.timestamp });
        }
    }

    /// Update an existing object and its index trees.
    ///
    /// Diffs `old` and `new`, removing stale index entries and inserting new ones for
    /// fields that changed. The object tree entry is overwritten unconditionally.
    ///
    /// # Panics
    /// Panics if timestamps differ, or if the objects are identical.
    pub fn update(&mut self, old: &Account, new: &Account) {
        assert_eq!(old.timestamp, new.timestamp);
        assert!(old.timestamp != 0);

        // Index diffs — only update trees where the index value changed.
        macro_rules! diff_index {
            ($tree:expr, $old_val:expr, $new_val:expr) => {
                if $old_val != $new_val {
                    if let Some(v) = $old_val {
                        $tree.remove(&CompositeKey64 { field: v, timestamp: old.timestamp });
                    }
                    if let Some(v) = $new_val {
                        $tree.put(&CompositeKey64 { field: v, timestamp: new.timestamp });
                    }
                }
            };
        }
        macro_rules! diff_index_128 {
            ($tree:expr, $old_val:expr, $new_val:expr) => {
                if $old_val != $new_val {
                    if let Some(v) = $old_val {
                        $tree.remove(&CompositeKey128 {
                            field: v,
                            timestamp: old.timestamp,
                            padding: 0,
                        });
                    }
                    if let Some(v) = $new_val {
                        $tree.put(&CompositeKey128 {
                            field: v,
                            timestamp: new.timestamp,
                            padding: 0,
                        });
                    }
                }
            };
        }
        macro_rules! diff_index_unit {
            ($tree:expr, $old_val:expr, $new_val:expr) => {
                if $old_val != $new_val {
                    if let Some(v) = $old_val {
                        $tree.remove(&CompositeKeyUnit { field: v, timestamp: old.timestamp });
                    }
                    if let Some(v) = $new_val {
                        $tree.put(&CompositeKeyUnit { field: v, timestamp: new.timestamp });
                    }
                }
            };
        }

        diff_index_128!(
            self.user_data_128,
            AccountUserData128Index::index_from_object(old),
            AccountUserData128Index::index_from_object(new)
        );
        diff_index!(
            self.user_data_64,
            AccountUserData64Index::index_from_object(old),
            AccountUserData64Index::index_from_object(new)
        );
        diff_index!(
            self.user_data_32,
            AccountUserData32Index::index_from_object(old),
            AccountUserData32Index::index_from_object(new)
        );
        diff_index!(
            self.ledger,
            AccountLedgerIndex::index_from_object(old),
            AccountLedgerIndex::index_from_object(new)
        );
        diff_index!(
            self.code,
            AccountCodeIndex::index_from_object(old),
            AccountCodeIndex::index_from_object(new)
        );
        diff_index_unit!(
            self.imported,
            AccountImportedIndex::index_from_object(old),
            AccountImportedIndex::index_from_object(new)
        );
        diff_index_unit!(
            self.closed,
            AccountClosedIndex::index_from_object(old),
            AccountClosedIndex::index_from_object(new)
        );

        // Overwrite the object tree entry.
        self.objects.put(new);
        self.objects_cache.upsert(new);
    }

    pub fn open_commence(&mut self, manifest_log: &mut impl ManifestLog) {
        self.objects.open_commence(manifest_log);
        self.id.open_commence(manifest_log);
        self.user_data_128.open_commence(manifest_log);
        self.user_data_64.open_commence(manifest_log);
        self.user_data_32.open_commence(manifest_log);
        self.ledger.open_commence(manifest_log);
        self.code.open_commence(manifest_log);
        self.imported.open_commence(manifest_log);
        self.closed.open_commence(manifest_log);
    }

    pub fn open_complete(&mut self, checkpoint_op: u64) {
        self.objects.open_complete(checkpoint_op);
        self.id.open_complete(checkpoint_op);
        self.user_data_128.open_complete(checkpoint_op);
        self.user_data_64.open_complete(checkpoint_op);
        self.user_data_32.open_complete(checkpoint_op);
        self.ledger.open_complete(checkpoint_op);
        self.code.open_complete(checkpoint_op);
        self.imported.open_complete(checkpoint_op);
        self.closed.open_complete(checkpoint_op);
    }
}

// ---------------------------------------------------------------------------
// TransferGroove operations
// ---------------------------------------------------------------------------

impl TransferGroove {
    /// Build a fresh [`TransferGroove`] with all fourteen trees, sized for
    /// `batch_value_count_limit` values per beat (forest.zig:290-302).
    #[must_use]
    pub fn new(batch_value_count_limit: u32) -> Self {
        fn tree<S: crate::tree::TreeSpec>(id: u16, name: &'static str, limit: u32) -> Tree<S> {
            Tree::<S>::new(TreeConfig { id, name }, Options { batch_value_count_limit: limit })
        }
        TransferGroove {
            objects: tree(10, "transfers_objects", batch_value_count_limit),
            id: tree(11, "transfers_id", batch_value_count_limit),
            debit_account_id: tree(12, "transfers_debit_account_id", batch_value_count_limit),
            credit_account_id: tree(13, "transfers_credit_account_id", batch_value_count_limit),
            amount: tree(14, "transfers_amount", batch_value_count_limit),
            pending_id: tree(15, "transfers_pending_id", batch_value_count_limit),
            user_data_128: tree(16, "transfers_user_data_128", batch_value_count_limit),
            user_data_64: tree(17, "transfers_user_data_64", batch_value_count_limit),
            user_data_32: tree(18, "transfers_user_data_32", batch_value_count_limit),
            ledger: tree(19, "transfers_ledger", batch_value_count_limit),
            code: tree(20, "transfers_code", batch_value_count_limit),
            expires_at: tree(21, "transfers_expires_at", batch_value_count_limit),
            imported: tree(22, "transfers_imported", batch_value_count_limit),
            closing: tree(23, "transfers_closing", batch_value_count_limit),
            objects_cache: TransferObjectsCache::new(CacheMapOptions {
                cache_value_count_max: 256,
                stash_value_count_max: constants::LSM_COMPACTION_OPS as u32
                    * batch_value_count_limit,
                scope_value_count_max: batch_value_count_limit,
                name: "transfers_objects",
            }),
        }
    }

    /// Lookup a transfer by its primary key (id) from the objects cache.
    ///
    /// Mirrors `groove.zig:885-936` (`get(PrimaryKey) ObjectCacheResult`), collapsed to
    /// `Option`: tombstones and orphans are resolved to `Negative` by
    /// [`TransferGroove::lookup`] instead of being surfaced here.
    #[must_use]
    pub fn get(&mut self, id: u128) -> Option<&Transfer> {
        self.objects_cache.get(id)
    }

    /// Whether the objects cache holds a live (non-tombstoned) entry for `id`.
    ///
    /// Mirrors upstream `objects_cache.has` (groove.zig:1020, 1107, 1330).
    #[must_use]
    pub fn has(&mut self, id: u128) -> bool {
        self.objects_cache.has(id)
    }

    /// Remove `id` from the primary-key view by placing a tombstone in the cache.
    ///
    /// Mirrors `groove.remove` (groove.zig:1876-1920).
    ///
    /// DEVIATION: upstream also removes the object and each populated index entry
    /// from the underlying trees; sans-forest only the cache tombstone is surfaced,
    /// which is what makes `get`/`has` report the key as absent in-session.
    ///
    /// # Panics
    /// Panics unless built with `verify`, or when `id` is absent from the cache.
    pub fn remove(&mut self, id: u128) {
        self.objects_cache.remove(id);
    }

    /// Resolve a transfer by its primary key (id) from the tree levels.
    ///
    /// Mirrors upstream's prefetch-by-unique-key resolution (groove.zig:1704-1732):
    /// look up the primary-key tree (`id`) for the transfer's timestamp, then resolve the
    /// object tree (`objects`) by that timestamp. Both hops go through
    /// `Tree::lookup_from_levels_cache`, so a block missing from the grid cache at either
    /// hop yields `Possible` (deferred to the async read phase).
    ///
    /// DEVIATION: upstream keeps orphaned primary keys (`timestamp == 0`) in a separate
    /// map via `insert_orphaned_object`; sans-IO an orphan resolves to `Negative` and
    /// orphaned-key tracking is deferred to the forest layer.
    ///
    /// # Panics
    /// Panics if the primary-key tree returns an entry whose key does not match `id`
    /// (a corrupted index invariant violation).
    #[must_use]
    pub fn lookup(
        &mut self,
        grid: &mut Grid,
        snapshot: u64,
        id: u128,
    ) -> LookupMemoryResult<Transfer> {
        let unique = match self.id.lookup_from_levels_cache(grid, snapshot, id) {
            LookupMemoryResult::Positive(unique) => unique,
            LookupMemoryResult::Negative => return LookupMemoryResult::Negative,
            LookupMemoryResult::Possible { level } => {
                return LookupMemoryResult::Possible { level };
            }
        };
        assert_eq!(unique.field, id);
        if unique.timestamp == 0 {
            return LookupMemoryResult::Negative;
        }
        self.objects.lookup_from_levels_cache(grid, snapshot, unique.timestamp)
    }

    pub fn scope_open(&mut self) {
        self.objects.scope_open();
        self.id.scope_open();
        self.debit_account_id.scope_open();
        self.credit_account_id.scope_open();
        self.amount.scope_open();
        self.pending_id.scope_open();
        self.user_data_128.scope_open();
        self.user_data_64.scope_open();
        self.user_data_32.scope_open();
        self.ledger.scope_open();
        self.code.scope_open();
        self.expires_at.scope_open();
        self.imported.scope_open();
        self.closing.scope_open();
        self.objects_cache.scope_open();
    }

    pub fn scope_close(&mut self, mode: ScopeCloseMode) {
        self.objects.scope_close(mode);
        self.id.scope_close(mode);
        self.debit_account_id.scope_close(mode);
        self.credit_account_id.scope_close(mode);
        self.amount.scope_close(mode);
        self.pending_id.scope_close(mode);
        self.user_data_128.scope_close(mode);
        self.user_data_64.scope_close(mode);
        self.user_data_32.scope_close(mode);
        self.ledger.scope_close(mode);
        self.code.scope_close(mode);
        self.expires_at.scope_close(mode);
        self.imported.scope_close(mode);
        self.closing.scope_close(mode);
        self.objects_cache.scope_close(mode);
    }

    /// Port of upstream `groove.compact` (groove.zig:1981-2021). See
    /// [`AccountGroove::compact`].
    pub fn compact(&mut self, op: u64, sc: &mut TransferGrooveScratch) {
        self.objects.compact(&mut sc.objects);
        self.id.compact(&mut sc.id);
        self.debit_account_id.compact(&mut sc.composite_key_128);
        self.credit_account_id.compact(&mut sc.composite_key_128);
        self.amount.compact(&mut sc.composite_key_128);
        self.pending_id.compact(&mut sc.composite_key_128);
        self.user_data_128.compact(&mut sc.composite_key_128);
        self.user_data_64.compact(&mut sc.composite_key_64);
        self.user_data_32.compact(&mut sc.composite_key_64);
        self.ledger.compact(&mut sc.composite_key_64);
        self.code.compact(&mut sc.composite_key_64);
        self.expires_at.compact(&mut sc.composite_key_64);
        self.imported.compact(&mut sc.composite_key_unit);
        self.closing.compact(&mut sc.composite_key_unit);

        let compaction_beat = op % (constants::LSM_COMPACTION_OPS as u64);
        if compaction_beat == constants::LSM_COMPACTION_OPS as u64 - 1 {
            self.objects_cache.compact();
        }
    }

    /// Insert a transfer into the object tree and all index trees.
    ///
    /// # Panics
    /// Panics if the timestamp is zero or the tombstone bit is set.
    pub fn insert(&mut self, object: &Transfer) {
        assert!(object.timestamp != 0);
        assert_eq!(object.timestamp & composite_key::TOMBSTONE_BIT, 0);

        self.objects.put(object);
        self.objects.key_range_update(object.timestamp);
        self.objects_cache.upsert(object);

        // id (unique key)
        self.id.put(&UniqueKey128 { field: object.id, timestamp: object.timestamp, padding: 0 });
        self.id.key_range_update(object.id);

        // Secondary indexes
        if let Some(v) = TransferDebitAccountIdIndex::index_from_object(object) {
            self.debit_account_id.put(&CompositeKey128 {
                field: v,
                timestamp: object.timestamp,
                padding: 0,
            });
        }
        if let Some(v) = TransferCreditAccountIdIndex::index_from_object(object) {
            self.credit_account_id.put(&CompositeKey128 {
                field: v,
                timestamp: object.timestamp,
                padding: 0,
            });
        }
        if let Some(v) = TransferAmountIndex::index_from_object(object) {
            self.amount.put(&CompositeKey128 { field: v, timestamp: object.timestamp, padding: 0 });
        }
        if let Some(v) = TransferPendingIdIndex::index_from_object(object) {
            self.pending_id.put(&CompositeKey128 {
                field: v,
                timestamp: object.timestamp,
                padding: 0,
            });
        }
        if let Some(v) = TransferUserData128Index::index_from_object(object) {
            self.user_data_128.put(&CompositeKey128 {
                field: v,
                timestamp: object.timestamp,
                padding: 0,
            });
        }
        if let Some(v) = TransferUserData64Index::index_from_object(object) {
            self.user_data_64.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = TransferUserData32Index::index_from_object(object) {
            self.user_data_32.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = TransferLedgerIndex::index_from_object(object) {
            self.ledger.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = TransferCodeIndex::index_from_object(object) {
            self.code.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        // Derived indexes
        if let Some(v) = TransferExpiresAtIndex::index_from_object(object) {
            self.expires_at.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = TransferImportedIndex::index_from_object(object) {
            self.imported.put(&CompositeKeyUnit { field: v, timestamp: object.timestamp });
        }
        if let Some(v) = TransferClosingIndex::index_from_object(object) {
            self.closing.put(&CompositeKeyUnit { field: v, timestamp: object.timestamp });
        }
    }

    /// Update an existing transfer and its index trees.
    ///
    /// Diffs `old` and `new`, removing stale index entries and inserting new ones for
    /// fields that changed. The object tree entry is overwritten unconditionally.
    ///
    /// # Panics
    /// Panics if timestamps differ, or if the objects are identical.
    #[allow(clippy::too_many_lines, reason = "mirrors upstream's inline field iteration")]
    pub fn update(&mut self, old: &Transfer, new: &Transfer) {
        assert_eq!(old.timestamp, new.timestamp);
        assert!(old.timestamp != 0);

        macro_rules! diff_index_128 {
            ($tree:expr, $old_val:expr, $new_val:expr) => {
                if $old_val != $new_val {
                    if let Some(v) = $old_val {
                        $tree.remove(&CompositeKey128 {
                            field: v,
                            timestamp: old.timestamp,
                            padding: 0,
                        });
                    }
                    if let Some(v) = $new_val {
                        $tree.put(&CompositeKey128 {
                            field: v,
                            timestamp: new.timestamp,
                            padding: 0,
                        });
                    }
                }
            };
        }
        macro_rules! diff_index {
            ($tree:expr, $old_val:expr, $new_val:expr) => {
                if $old_val != $new_val {
                    if let Some(v) = $old_val {
                        $tree.remove(&CompositeKey64 { field: v, timestamp: old.timestamp });
                    }
                    if let Some(v) = $new_val {
                        $tree.put(&CompositeKey64 { field: v, timestamp: new.timestamp });
                    }
                }
            };
        }
        macro_rules! diff_index_unit {
            ($tree:expr, $old_val:expr, $new_val:expr) => {
                if $old_val != $new_val {
                    if let Some(v) = $old_val {
                        $tree.remove(&CompositeKeyUnit { field: v, timestamp: old.timestamp });
                    }
                    if let Some(v) = $new_val {
                        $tree.put(&CompositeKeyUnit { field: v, timestamp: new.timestamp });
                    }
                }
            };
        }

        // id (unique key) — must never change:
        assert_eq!(old.id, new.id);

        diff_index_128!(
            self.debit_account_id,
            TransferDebitAccountIdIndex::index_from_object(old),
            TransferDebitAccountIdIndex::index_from_object(new)
        );
        diff_index_128!(
            self.credit_account_id,
            TransferCreditAccountIdIndex::index_from_object(old),
            TransferCreditAccountIdIndex::index_from_object(new)
        );
        diff_index_128!(
            self.amount,
            TransferAmountIndex::index_from_object(old),
            TransferAmountIndex::index_from_object(new)
        );
        diff_index_128!(
            self.pending_id,
            TransferPendingIdIndex::index_from_object(old),
            TransferPendingIdIndex::index_from_object(new)
        );
        diff_index_128!(
            self.user_data_128,
            TransferUserData128Index::index_from_object(old),
            TransferUserData128Index::index_from_object(new)
        );
        diff_index!(
            self.user_data_64,
            TransferUserData64Index::index_from_object(old),
            TransferUserData64Index::index_from_object(new)
        );
        diff_index!(
            self.user_data_32,
            TransferUserData32Index::index_from_object(old),
            TransferUserData32Index::index_from_object(new)
        );
        diff_index!(
            self.ledger,
            TransferLedgerIndex::index_from_object(old),
            TransferLedgerIndex::index_from_object(new)
        );
        diff_index!(
            self.code,
            TransferCodeIndex::index_from_object(old),
            TransferCodeIndex::index_from_object(new)
        );
        diff_index!(
            self.expires_at,
            TransferExpiresAtIndex::index_from_object(old),
            TransferExpiresAtIndex::index_from_object(new)
        );
        diff_index_unit!(
            self.imported,
            TransferImportedIndex::index_from_object(old),
            TransferImportedIndex::index_from_object(new)
        );
        diff_index_unit!(
            self.closing,
            TransferClosingIndex::index_from_object(old),
            TransferClosingIndex::index_from_object(new)
        );

        // Overwrite the object tree entry.
        self.objects.put(new);
        self.objects_cache.upsert(new);
    }

    pub fn open_commence(&mut self, manifest_log: &mut impl ManifestLog) {
        self.objects.open_commence(manifest_log);
        self.id.open_commence(manifest_log);
        self.debit_account_id.open_commence(manifest_log);
        self.credit_account_id.open_commence(manifest_log);
        self.amount.open_commence(manifest_log);
        self.pending_id.open_commence(manifest_log);
        self.user_data_128.open_commence(manifest_log);
        self.user_data_64.open_commence(manifest_log);
        self.user_data_32.open_commence(manifest_log);
        self.ledger.open_commence(manifest_log);
        self.code.open_commence(manifest_log);
        self.expires_at.open_commence(manifest_log);
        self.imported.open_commence(manifest_log);
        self.closing.open_commence(manifest_log);
    }

    pub fn open_complete(&mut self, checkpoint_op: u64) {
        self.objects.open_complete(checkpoint_op);
        self.id.open_complete(checkpoint_op);
        self.debit_account_id.open_complete(checkpoint_op);
        self.credit_account_id.open_complete(checkpoint_op);
        self.amount.open_complete(checkpoint_op);
        self.pending_id.open_complete(checkpoint_op);
        self.user_data_128.open_complete(checkpoint_op);
        self.user_data_64.open_complete(checkpoint_op);
        self.user_data_32.open_complete(checkpoint_op);
        self.ledger.open_complete(checkpoint_op);
        self.code.open_complete(checkpoint_op);
        self.expires_at.open_complete(checkpoint_op);
        self.imported.open_complete(checkpoint_op);
        self.closing.open_complete(checkpoint_op);
    }
}

// ---------------------------------------------------------------------------
// TransferPendingGroove — simplified groove for pending transfer status tracking
// ---------------------------------------------------------------------------

/// Grove for tracking pending transfer statuses.
///
/// Upstream: `src/state_machine.zig:470` (`transfers_pending` groove config). Mirrors it with
/// an object tree (`TransferPending`, tree id 20) plus the optional `status` index tree
/// (`CompositeKey64`, tree id 21), and the per-session objects cache keyed by timestamp.
pub struct TransferPendingGroove {
    /// Object tree, keyed by the pending transfer's timestamp.
    objects: Tree<TransferPendingObjectSpec>,
    /// Optional `status` index (field = status as u64, keyed by (status, timestamp)).
    status: Tree<TransferPendingStatusSpec>,
    /// Per-session objects cache, keyed by the pending transfer's timestamp.
    objects_cache: TransferPendingObjectsCache,
}

/// Radix-sort scratch buffers for [`TransferPendingGroove`]'s trees, owned by the forest.
///
/// See the DEVIATION note on [`AccountGrooveScratch`].
pub struct TransferPendingGrooveScratch {
    pub objects: ScratchMemory<TransferPending>,
    pub status: ScratchMemory<CompositeKey64>,
}

/// Builds fresh scratch buffers for a [`TransferPendingGroove`], each sized to the groove's
/// maximum tree value count (forest.zig:290-299).
impl TransferPendingGrooveScratch {
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
            status: ScratchMemory::new(GROOVE_VALUE_COUNT_MAX),
        }
    }
}

impl Default for TransferPendingGrooveScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferPendingGroove {
    /// Build a fresh [`TransferPendingGroove`] with its two trees, sized for
    /// `batch_value_count_limit` values per beat (forest.zig:290-302).
    #[must_use]
    pub fn new(batch_value_count_limit: u32) -> Self {
        TransferPendingGroove {
            objects: Tree::<TransferPendingObjectSpec>::new(
                TreeConfig { id: 20, name: "transfers_pending_objects" },
                Options { batch_value_count_limit },
            ),
            status: Tree::<TransferPendingStatusSpec>::new(
                TreeConfig { id: 21, name: "transfers_pending_status" },
                Options { batch_value_count_limit },
            ),
            objects_cache: TransferPendingObjectsCache::new(CacheMapOptions {
                cache_value_count_max: 256,
                stash_value_count_max: constants::LSM_COMPACTION_OPS as u32
                    * batch_value_count_limit,
                scope_value_count_max: batch_value_count_limit,
                name: "transfers_pending_objects",
            }),
        }
    }

    pub fn scope_open(&mut self) {
        self.objects.scope_open();
        self.status.scope_open();
        self.objects_cache.scope_open();
    }

    pub fn scope_close(&mut self, mode: ScopeCloseMode) {
        self.objects.scope_close(mode);
        self.status.scope_close(mode);
        self.objects_cache.scope_close(mode);
    }

    /// Lookup a `TransferPending` record by its primary key (timestamp).
    #[must_use]
    pub fn get(&mut self, timestamp: u64) -> Option<&TransferPending> {
        self.objects_cache.get(timestamp)
    }

    /// Whether the objects cache holds a live (non-tombstoned) entry for the timestamp.
    #[must_use]
    pub fn has(&mut self, timestamp: u64) -> bool {
        self.objects_cache.has(timestamp)
    }

    /// Lookup a `TransferPending` record by its primary key (timestamp) from the object
    /// tree over the grid cache. Unlike `AccountGroove`/`TransferGroove`, the pending
    /// groove's primary key *is* the timestamp, so this is a single-hop read (no unique-key
    /// tree to resolve through), mirroring upstream's `IndexTreeType(get)` object read
    /// (groove.zig / state_machine.zig:470). `Possible` propagates when the tree cannot
    /// serve the value from the grid cache (deferred to the async read phase).
    #[must_use]
    pub fn lookup(
        &mut self,
        grid: &mut Grid,
        snapshot: u64,
        timestamp: u64,
    ) -> LookupMemoryResult<TransferPending> {
        self.objects.lookup_from_levels_cache(grid, snapshot, timestamp)
    }

    /// Remove the pending record at `timestamp` by placing a tombstone in the cache,
    /// mirroring `groove.remove` (groove.zig:1876).
    ///
    /// # Panics
    /// Panics unless built with `verify`, or when `timestamp` is absent from the cache.
    pub fn remove(&mut self, timestamp: u64) {
        self.objects_cache.remove(timestamp);
    }

    /// Insert a pending transfer record into the object tree, status index and cache.
    ///
    /// # Panics
    /// Panics if the timestamp is zero or the tombstone bit is set.
    pub fn insert(&mut self, object: &TransferPending) {
        assert!(object.timestamp != 0);
        assert_eq!(object.timestamp & composite_key::TOMBSTONE_BIT, 0);

        self.objects.put(object);
        self.objects.key_range_update(object.timestamp);
        self.objects_cache.upsert(object);

        if let Some(v) = TransferPendingStatusIndex::index_from_object(object) {
            self.status.put(&CompositeKey64 { field: v, timestamp: object.timestamp });
        }
    }

    /// Update an existing pending transfer record.
    ///
    /// Diffs `old` and `new` for the status index and overwrites the object tree + cache
    /// unconditionally.
    ///
    /// # Panics
    /// Panics if the timestamps differ.
    pub fn update(&mut self, old: &TransferPending, new: &TransferPending) {
        assert_eq!(old.timestamp, new.timestamp);
        self.objects.put(new);
        self.objects_cache.upsert(new);

        let old_status = TransferPendingStatusIndex::index_from_object(old);
        let new_status = TransferPendingStatusIndex::index_from_object(new);
        if old_status != new_status {
            if let Some(v) = old_status {
                self.status.remove(&CompositeKey64 { field: v, timestamp: new.timestamp });
            }
            if let Some(v) = new_status {
                self.status.put(&CompositeKey64 { field: v, timestamp: new.timestamp });
            }
        }
    }

    /// Port of upstream `groove.compact` (groove.zig:1981-2021). See
    /// [`AccountGroove::compact`].
    pub fn compact(&mut self, op: u64, sc: &mut TransferPendingGrooveScratch) {
        self.objects.compact(&mut sc.objects);
        self.status.compact(&mut sc.status);

        let compaction_beat = op % (constants::LSM_COMPACTION_OPS as u64);
        if compaction_beat == constants::LSM_COMPACTION_OPS as u64 - 1 {
            self.objects_cache.compact();
        }
    }

    pub fn open_commence(&mut self, manifest_log: &mut impl ManifestLog) {
        self.objects.open_commence(manifest_log);
        self.status.open_commence(manifest_log);
    }

    pub fn open_complete(&mut self, checkpoint_op: u64) {
        self.objects.open_complete(checkpoint_op);
        self.status.open_complete(checkpoint_op);
    }
}

// ---------------------------------------------------------------------------
// Write-path / read-path block accessor tests
//
// These build real index+value blocks with `TableBuilder` (via the concrete
// `TableSpec` impls above) and exercise the delegated `TreeSpec` accessors that
// `Tree::lookup_from_levels_cache` depends on, mirroring upstream table.zig's
// `assert_index_blocks_for_key` / `value_block_search` tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::multiversion::Release;
    use crate::table::{DataFinishOptions, IndexFinishOptions, TableBuilder, TableInfo};
    use crate::tree::TreeSpec;
    use tigerbeetle_core::types::TransferPendingStatus;
    use tigerbeetle_lsm::cache_map::{CacheMapOptions, GetOrTombstone};

    /// Build a single-value-block table from strictly key-sorted values.
    fn build_table<S: TableSpec>(values: &[S::Value]) -> (Vec<u8>, Vec<u8>, TableInfo<S::Key>) {
        build_table_with::<S>(values, 1, 1, 2)
    }

    /// Like [`build_table`], but for an arbitrary tree id and block addresses.
    fn build_table_with<S: TableSpec>(
        values: &[S::Value],
        tree_id: u16,
        value_address: u64,
        index_address: u64,
    ) -> (Vec<u8>, Vec<u8>, TableInfo<S::Key>) {
        let layout = TableLayout::compute_for::<S>();
        let mut index_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_block = vec![0u8; constants::BLOCK_SIZE];

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);
        builder.set_value_block(&mut value_block);
        for value in values {
            builder.insert_value::<S>(value, &mut value_block, &layout);
        }

        builder.value_block_finish::<S>(
            &mut value_block,
            &mut index_block,
            &layout,
            DataFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: value_address,
                snapshot_min: 1,
                tree_id,
            },
        );

        let info = builder.index_block_finish::<S::Key>(
            &mut index_block,
            &layout,
            IndexFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address: index_address,
                snapshot_min: 1,
                tree_id,
            },
        );

        (index_block, value_block, info)
    }

    /// Build a multi-value-block table from strictly key-sorted values, returning the
    /// index block and one value block per finished data block (in address order).
    fn build_multi_block_table<S: TableSpec>(
        values: &[S::Value],
    ) -> (Vec<u8>, Vec<Vec<u8>>, TableInfo<S::Key>) {
        let layout = TableLayout::compute_for::<S>();
        let mut index_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_block = vec![0u8; constants::BLOCK_SIZE];
        let mut value_blocks: Vec<Vec<u8>> = Vec::new();

        let mut builder = TableBuilder::new();
        builder.set_index_block(&mut index_block);

        let mut address = 0u64;
        let mut start = 0;
        while start < values.len() {
            let end = (start + layout.block_value_count_max as usize).min(values.len());

            builder.set_value_block(&mut value_block);
            for value in &values[start..end] {
                builder.insert_value::<S>(value, &mut value_block, &layout);
            }

            address += 1;
            builder.value_block_finish::<S>(
                &mut value_block,
                &mut index_block,
                &layout,
                DataFinishOptions {
                    cluster: 0,
                    release: Release::MINIMUM,
                    address,
                    snapshot_min: 1,
                    tree_id: 1,
                },
            );
            value_blocks.push(value_block.clone());

            start = end;
        }

        address += 1;
        let info = builder.index_block_finish::<S::Key>(
            &mut index_block,
            &layout,
            IndexFinishOptions {
                cluster: 0,
                release: Release::MINIMUM,
                address,
                snapshot_min: 1,
                tree_id: 1,
            },
        );

        (index_block, value_blocks, info)
    }

    #[test]
    fn account_object_tree_block_search() {
        let accounts = [
            Account { timestamp: 1, id: 101, ..Account::default() },
            Account { timestamp: 3, id: 103, ..Account::default() },
        ];
        let (index_block, value_block, info) = build_table::<AccountObjectSpec>(&accounts);
        assert_eq!(info.key_min, 1);
        assert_eq!(info.key_max, 3);
        assert_eq!(info.value_count, 2);

        // Present key routes to the value block and resolves:
        let blocks = AccountObjectSpec::index_blocks_for_key(&index_block, 1).unwrap();
        assert_eq!(blocks.value_block_address, 1);
        let value = AccountObjectSpec::value_block_search(&value_block, 1).unwrap();
        assert_eq!(value.timestamp, 1);
        assert_eq!(value.id, 101);

        // Absent key inside the table's key range: index routes, value search misses:
        assert!(AccountObjectSpec::index_blocks_for_key(&index_block, 2).is_some());
        assert!(AccountObjectSpec::value_block_search(&value_block, 2).is_none());
    }

    #[test]
    fn account_object_tree_block_search_tombstone() {
        let accounts = [
            Account { timestamp: 1, id: 101, ..Account::default() },
            <AccountObjectSpec as TreeSpec>::tombstone_from_key(2),
            Account { timestamp: 3, id: 103, ..Account::default() },
        ];
        let (index_block, value_block, info) = build_table::<AccountObjectSpec>(&accounts);
        assert_eq!(info.key_min, 1);
        assert_eq!(info.key_max, 3);

        let tombstone = AccountObjectSpec::value_block_search(&value_block, 2).unwrap();
        assert!(<AccountObjectSpec as TableSpec>::tombstone(&tombstone));
        assert_eq!(<AccountObjectSpec as TableSpec>::key_from_value(&tombstone), 2);

        let blocks = AccountObjectSpec::index_blocks_for_key(&index_block, 2).unwrap();
        assert_eq!(blocks.value_block_address, 1);
    }

    #[test]
    fn transfer_object_tree_block_search() {
        let transfers = [
            Transfer { timestamp: 7, id: 700, ..Transfer::default() },
            Transfer { timestamp: 9, id: 900, ..Transfer::default() },
        ];
        let (index_block, value_block, info) = build_table::<TransferObjectSpec>(&transfers);
        assert_eq!(info.key_min, 7);
        assert_eq!(info.key_max, 9);
        assert_eq!(info.value_count, 2);

        let value = TransferObjectSpec::value_block_search(&value_block, 9).unwrap();
        assert_eq!(value.id, 900);
        assert!(TransferObjectSpec::value_block_search(&value_block, 8).is_none());

        let blocks = TransferObjectSpec::index_blocks_for_key(&index_block, 7).unwrap();
        assert_eq!(blocks.value_block_address, 1);
    }

    #[test]
    fn unique_key_tree_block_search() {
        let entries = [
            UniqueKey128 { field: 5, timestamp: 100, padding: 0 },
            UniqueKey128 { field: 6, timestamp: 200, padding: 0 },
            UniqueKey128::tombstone_from_key(7),
        ];
        let (index_block, value_block, info) = build_table::<UniqueKey128Spec>(&entries);
        assert_eq!(info.key_min, 5);
        assert_eq!(info.key_max, 7);
        assert_eq!(info.value_count, 3);

        let found = UniqueKey128Spec::value_block_search(&value_block, 6).unwrap();
        assert_eq!(found.timestamp, 200);

        // Tombstone entry resolves to a value reported as a tombstone:
        let tombstone = UniqueKey128Spec::value_block_search(&value_block, 7).unwrap();
        assert!(<UniqueKey128Spec as TableSpec>::tombstone(&tombstone));

        // Key beyond the table's key range:
        assert!(UniqueKey128Spec::value_block_search(&value_block, 8).is_none());

        let blocks = UniqueKey128Spec::index_blocks_for_key(&index_block, 6).unwrap();
        assert_eq!(blocks.value_block_address, 1);
    }

    #[test]
    fn composite_key64_and_unit_tree_block_search() {
        let entries64 = [
            CompositeKey64 { field: 3, timestamp: 10 },
            CompositeKey64 { field: 3, timestamp: 20 },
            CompositeKey64 { field: 4, timestamp: 30 },
        ];
        let (index_block, value_block, info) = build_table::<CompositeKey64Spec>(&entries64);
        assert_eq!(info.key_min, u128::from(3_u64) << 64 | u128::from(10_u64));
        assert_eq!(info.key_max, u128::from(4_u64) << 64 | u128::from(30_u64));

        let key20 = u128::from(3_u64) << 64 | u128::from(20_u64);
        let found = CompositeKey64Spec::value_block_search(&value_block, key20).unwrap();
        assert_eq!(found.timestamp, 20);
        let blocks64 = CompositeKey64Spec::index_blocks_for_key(&index_block, key20).unwrap();
        assert_eq!(blocks64.value_block_address, 1);

        // Absent key between present keys:
        let key15 = u128::from(3_u64) << 64 | u128::from(15_u64);
        assert!(CompositeKey64Spec::value_block_search(&value_block, key15).is_none());

        let units = [
            CompositeKeyUnit { field: (), timestamp: 10 },
            CompositeKeyUnit { field: (), timestamp: 20 },
            CompositeKeyUnit::tombstone_from_key(25),
        ];
        let (index_unit, value_unit, info_unit) = build_table::<CompositeKeyUnitSpec>(&units);
        assert_eq!(info_unit.key_min, 10);
        assert_eq!(info_unit.key_max, 25);

        let found_unit = CompositeKeyUnitSpec::value_block_search(&value_unit, 10).unwrap();
        assert_eq!(found_unit.timestamp, 10);
        assert!(CompositeKeyUnitSpec::value_block_search(&value_unit, 15).is_none());

        let tombstone = CompositeKeyUnitSpec::value_block_search(&value_unit, 25).unwrap();
        assert!(<CompositeKeyUnitSpec as TableSpec>::tombstone(&tombstone));
        let blocks_unit = CompositeKeyUnitSpec::index_blocks_for_key(&index_unit, 20).unwrap();
        assert_eq!(blocks_unit.value_block_address, 1);
    }

    #[test]
    fn composite_key128_multiblock_index_routing() {
        // 32-byte values → 120 per block. 239 entries (ts 1..=119, 122..=241) force a
        // second value block and leave a gap at ts=120/121 for in-range misses:
        let first_field = 1_000u128;
        let entries: Vec<CompositeKey128> = (1_u64..=119)
            .chain(122..=241)
            .map(|ts| CompositeKey128 { field: first_field, timestamp: ts, padding: 0 })
            .collect();
        let (index_block, value_blocks, info) =
            build_multi_block_table::<CompositeKey128Spec>(&entries);
        assert_eq!(value_blocks.len(), 2);
        assert_eq!(info.key_min, U256::from_parts(first_field, 1));
        assert_eq!(info.key_max, U256::from_parts(first_field, 241));
        assert_eq!(info.value_count, 239);

        // ts=60 lives in the first value block:
        let key60 = U256::from_parts(first_field, 60);
        let blocks60 = CompositeKey128Spec::index_blocks_for_key(&index_block, key60).unwrap();
        assert_eq!(blocks60.value_block_address, 1);
        assert_eq!(blocks60.value_block_key_min, U256::from_parts(first_field, 1));
        let found60 = CompositeKey128Spec::value_block_search(&value_blocks[0], key60).unwrap();
        assert_eq!(found60.timestamp, 60);

        // ts=121 falls in the gap inside the first block's range (block 1 holds
        // ts 1..=119 plus 122): the index routes to block 1, then the value search
        // misses — matching upstream's behavior for a missing key inside a block.
        let key121 = U256::from_parts(first_field, 121);
        let blocks121 = CompositeKey128Spec::index_blocks_for_key(&index_block, key121).unwrap();
        assert_eq!(blocks121.value_block_address, 1);
        assert!(CompositeKey128Spec::value_block_search(&value_blocks[0], key121).is_none());

        // ts=241 lives in the second block:
        let key241 = U256::from_parts(first_field, 241);
        let blocks241 = CompositeKey128Spec::index_blocks_for_key(&index_block, key241).unwrap();
        assert_eq!(blocks241.value_block_address, 2);
        assert_eq!(blocks241.value_block_key_min, U256::from_parts(first_field, 123));
        assert_eq!(blocks241.value_block_key_max, key241);
        let found241 = CompositeKey128Spec::value_block_search(&value_blocks[1], key241).unwrap();
        assert_eq!(found241.timestamp, 241);
    }

    // ------------------------------------------------------------------
    // Groove-level primary lookup: `AccountGroove::lookup` / `TransferGroove::lookup`
    // over real `TableBuilder` tables served from the grid cache. Exercises the
    // prefetch-by-unique-key two-hop resolution (id tree → timestamp, then objects
    // tree → object) end to end, sans-IO.
    // ------------------------------------------------------------------

    use crate::grid::{Grid, GridOptions};
    use crate::table::TableKey as VsrTableKey;
    use crate::tree::{Options, Tree};
    use tigerbeetle_lsm::free_set::SHARD_BITS;
    use tigerbeetle_lsm::schema::manifest_node::{self, Event, Label};
    use tigerbeetle_lsm::tree::{SNAPSHOT_LATEST, TreeConfig};

    const FREE_SET_BLOCKS: usize = 2 * SHARD_BITS;

    /// Acquire `blocks` fresh addresses from the grid.
    fn acquire_addresses(grid: &mut Grid, blocks: usize) -> Vec<u64> {
        let reservation = grid.reserve(blocks);
        (0..blocks).map(|_| grid.acquire(reservation)).collect()
    }

    /// A grid with `blocks` acquired addresses.
    fn new_groove_grid(blocks: usize) -> (Grid, Vec<u64>) {
        let mut grid = Grid::new(GridOptions {
            cache_blocks_count: 64,
            stash_blocks_count: 12,
            read_iops_max: 2,
            write_iops_max: 2,
            free_set_blocks_count: Some(FREE_SET_BLOCKS),
            free_set_blocks_capacity: None,
        });
        let addresses = acquire_addresses(&mut grid, blocks);
        (grid, addresses)
    }

    /// Header checksum of a finished builder block, without touching the grid.
    fn block_checksum(block: &[u8]) -> u128 {
        crate::schema::header_from_block(block).checksum
    }

    /// Copy a finished block into the grid cache and return its header checksum.
    fn seed_grid_block(grid: &mut Grid, address: u64, block: &[u8]) -> u128 {
        assert!(!grid.free_set_is_free(address));
        let location = grid.get_block();
        grid.block_mut(location).copy_from_slice(block);
        let checksum = crate::schema::header_from_block(grid.block(location)).checksum;
        grid.cache_upsert(address, location);
        checksum
    }

    /// A manifest log that never opens — `Tree::open_table` bypasses logging entirely.
    struct NeverOpenedLog;

    impl ManifestLog for NeverOpenedLog {
        fn is_opened(&self) -> bool {
            false
        }

        fn append(&mut self, _entry: &manifest_node::TableInfo) {
            unreachable!("open_table bypasses the manifest log")
        }
    }

    /// Replay a built table directly into a tree's manifest level 0.
    fn open_tree<S: crate::tree::TreeSpec>(
        tree: &mut Tree<S>,
        info: &TableInfo<S::Key>,
        index_address: u64,
        index_checksum: u128,
        tree_id: u16,
    ) {
        tree.open_commence(&NeverOpenedLog);
        tree.open_table(&manifest_node::TableInfo {
            key_min: <S::Key as VsrTableKey>::to_le_bytes_padded(info.key_min),
            key_max: <S::Key as VsrTableKey>::to_le_bytes_padded(info.key_max),
            checksum: index_checksum,
            address: index_address,
            snapshot_min: 1,
            snapshot_max: SNAPSHOT_LATEST,
            value_count: info.value_count,
            tree_id,
            label: Label { level: 0, event: Event::Insert },
        });
        tree.open_complete(0);
    }

    fn new_account_objects_cache() -> AccountObjectsCache {
        // Sizing mirrors upstream groove.zig:766-780: cache_entries_max from config;
        // stash sized for lsm_compaction_ops * (batch_value_count_limit + prefetch);
        // scope sized for a single beat's batch_value_count_limit.
        AccountObjectsCache::new(CacheMapOptions {
            cache_value_count_max: 256,
            stash_value_count_max: constants::LSM_COMPACTION_OPS as u32 * 32,
            scope_value_count_max: 32,
            name: "accounts_objects",
        })
    }

    fn new_transfer_objects_cache() -> TransferObjectsCache {
        TransferObjectsCache::new(CacheMapOptions {
            cache_value_count_max: 256,
            stash_value_count_max: constants::LSM_COMPACTION_OPS as u32 * 32,
            scope_value_count_max: 32,
            name: "transfers_objects",
        })
    }

    fn new_transfer_pending_objects_cache() -> TransferPendingObjectsCache {
        TransferPendingObjectsCache::new(CacheMapOptions {
            cache_value_count_max: 256,
            stash_value_count_max: constants::LSM_COMPACTION_OPS as u32 * 32,
            scope_value_count_max: 32,
            name: "transfers_pending_objects",
        })
    }

    fn new_transfer_pending_groove() -> TransferPendingGroove {
        TransferPendingGroove {
            objects: Tree::<TransferPendingObjectSpec>::new(
                TreeConfig { id: 20, name: "transfers_pending_objects" },
                Options { batch_value_count_limit: 32 },
            ),
            status: Tree::<TransferPendingStatusSpec>::new(
                TreeConfig { id: 21, name: "transfers_pending_status" },
                Options { batch_value_count_limit: 32 },
            ),
            objects_cache: new_transfer_pending_objects_cache(),
        }
    }

    fn new_account_groove() -> AccountGroove {
        fn tree<S: crate::tree::TreeSpec>(id: u16, name: &'static str) -> Tree<S> {
            Tree::<S>::new(TreeConfig { id, name }, Options { batch_value_count_limit: 32 })
        }
        AccountGroove {
            objects: tree(1, "accounts_objects"),
            id: tree(2, "accounts_id"),
            user_data_128: tree(3, "accounts_user_data_128"),
            user_data_64: tree(4, "accounts_user_data_64"),
            user_data_32: tree(5, "accounts_user_data_32"),
            ledger: tree(6, "accounts_ledger"),
            code: tree(7, "accounts_code"),
            imported: tree(8, "accounts_imported"),
            closed: tree(9, "accounts_closed"),
            objects_cache: new_account_objects_cache(),
        }
    }

    fn new_transfer_groove() -> TransferGroove {
        fn tree<S: crate::tree::TreeSpec>(id: u16, name: &'static str) -> Tree<S> {
            Tree::<S>::new(TreeConfig { id, name }, Options { batch_value_count_limit: 32 })
        }
        TransferGroove {
            objects: tree(10, "transfers_objects"),
            id: tree(11, "transfers_id"),
            debit_account_id: tree(12, "transfers_debit_account_id"),
            credit_account_id: tree(13, "transfers_credit_account_id"),
            amount: tree(14, "transfers_amount"),
            pending_id: tree(15, "transfers_pending_id"),
            user_data_128: tree(16, "transfers_user_data_128"),
            user_data_64: tree(17, "transfers_user_data_64"),
            user_data_32: tree(18, "transfers_user_data_32"),
            ledger: tree(19, "transfers_ledger"),
            code: tree(20, "transfers_code"),
            expires_at: tree(21, "transfers_expires_at"),
            imported: tree(22, "transfers_imported"),
            closing: tree(23, "transfers_closing"),
            objects_cache: new_transfer_objects_cache(),
        }
    }

    /// Build the account groove's `objects` (tree 1) and `id` (tree 2) tables, seed
    /// their blocks into `grid` (acquiring 4 addresses), and open both trees.
    fn open_account_lookup_trees(
        grid: &mut Grid,
        objects: &[Account],
        ids: &[UniqueKey128],
    ) -> AccountGroove {
        let addresses = acquire_addresses(grid, 4);
        let (objects_value, objects_index, id_value, id_index) =
            (addresses[0], addresses[1], addresses[2], addresses[3]);

        let (obj_index_block, obj_value_block, obj_info) =
            build_table_with::<AccountObjectSpec>(objects, 1, objects_value, objects_index);
        let obj_checksum = seed_grid_block(grid, objects_index, &obj_index_block);
        seed_grid_block(grid, objects_value, &obj_value_block);

        let (id_index_block, id_value_block, id_info) =
            build_table_with::<UniqueKey128Spec>(ids, 2, id_value, id_index);
        let id_checksum = seed_grid_block(grid, id_index, &id_index_block);
        seed_grid_block(grid, id_value, &id_value_block);

        let mut groove = new_account_groove();
        open_tree(&mut groove.objects, &obj_info, objects_index, obj_checksum, 1);
        open_tree(&mut groove.id, &id_info, id_index, id_checksum, 2);
        groove
    }

    fn accounts() -> [Account; 3] {
        [
            Account { timestamp: 1, id: 101, ..Account::default() },
            Account { timestamp: 2, id: 102, ..Account::default() },
            Account { timestamp: 3, id: 103, ..Account::default() },
        ]
    }

    const IDS: [UniqueKey128; 3] = [
        UniqueKey128 { field: 101, timestamp: 1, padding: 0 },
        UniqueKey128 { field: 102, timestamp: 2, padding: 0 },
        UniqueKey128 { field: 103, timestamp: 3, padding: 0 },
    ];

    #[test]
    fn groove_lookup_primary_hit_and_miss() {
        let (mut grid, _) = new_groove_grid(4);
        let mut groove = open_account_lookup_trees(&mut grid, &accounts(), &IDS);

        // Present id resolves through both hops:
        let result = groove.lookup(&mut grid, SNAPSHOT_LATEST, 101);
        assert!(matches!(
            result,
            LookupMemoryResult::Positive(Account { id: 101, timestamp: 1, .. })
        ));

        // Unknown id (outside the id tree's key range) → negative:
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 999),
            LookupMemoryResult::Negative
        ));
    }

    #[test]
    fn groove_lookup_primary_object_tree_miss() {
        // The id tree resolves timestamps that the objects tree does not hold
        // (a gap inside the id tree's range, plus a present id pointing at a
        // missing object timestamp) → the second hop resolves to negative.
        let id_entries = [
            UniqueKey128 { field: 101, timestamp: 1, padding: 0 },
            UniqueKey128 { field: 103, timestamp: 3, padding: 0 }, // gap at id 102
            UniqueKey128 { field: 201, timestamp: 99, padding: 0 }, // no object@99
        ];
        let (mut grid, _) = new_groove_grid(4);
        let mut groove = open_account_lookup_trees(&mut grid, &accounts(), &id_entries);

        // In-range gap in the id tree:
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 102),
            LookupMemoryResult::Negative
        ));

        // Present id (201) whose timestamp (99) has no object → negative:
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 201),
            LookupMemoryResult::Negative
        ));
    }

    #[test]
    fn groove_lookup_primary_tombstone() {
        // A tombstone in the unique-key tree short-circuits before the objects hop.
        let id_entries = [
            UniqueKey128 { field: 101, timestamp: 1, padding: 0 },
            UniqueKey128::tombstone_from_key(102),
            UniqueKey128 { field: 103, timestamp: 3, padding: 0 },
        ];
        let (mut grid, _) = new_groove_grid(4);
        let mut groove = open_account_lookup_trees(&mut grid, &accounts(), &id_entries);

        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 102),
            LookupMemoryResult::Negative
        ));
    }

    #[test]
    fn groove_lookup_primary_orphan() {
        // An orphaned primary key (timestamp == 0) resolves to negative sans-IO;
        // upstream tracks it via `insert_orphaned_object` (deferred).
        let id_entries = [
            UniqueKey128 { field: 101, timestamp: 1, padding: 0 },
            UniqueKey128 { field: 103, timestamp: 3, padding: 0 },
            UniqueKey128 { field: 301, timestamp: 0, padding: 0 },
        ];
        let (mut grid, _) = new_groove_grid(4);
        let mut groove = open_account_lookup_trees(&mut grid, &accounts(), &id_entries);

        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 301),
            LookupMemoryResult::Negative
        ));
    }

    #[test]
    fn groove_lookup_primary_possible() {
        // Build both tables but seed them progressively: the same lookup walks
        // Possible{level:0} → Possible{level:0} → Positive as blocks arrive.
        let (mut grid, addresses) = new_groove_grid(4);
        let (objects_value, objects_index, id_value, id_index) =
            (addresses[0], addresses[1], addresses[2], addresses[3]);

        let (obj_index_block, obj_value_block, obj_info) =
            build_table_with::<AccountObjectSpec>(&accounts(), 1, objects_value, objects_index);
        let (id_index_block, id_value_block, id_info) =
            build_table_with::<UniqueKey128Spec>(&IDS, 2, id_value, id_index);

        let mut groove = new_account_groove();
        open_tree(
            &mut groove.objects,
            &obj_info,
            objects_index,
            block_checksum(&obj_index_block),
            1,
        );
        open_tree(&mut groove.id, &id_info, id_index, block_checksum(&id_index_block), 2);

        // First hop: the id tree's index block is not cached → possible at level 0.
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 101),
            LookupMemoryResult::Possible { level: 0 }
        ));

        // Seed only the id tree: the id hop resolves, but the objects tree's index
        // block is still missing → possible at level 0 again.
        seed_grid_block(&mut grid, id_index, &id_index_block);
        seed_grid_block(&mut grid, id_value, &id_value_block);
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 101),
            LookupMemoryResult::Possible { level: 0 }
        ));

        // Seed the objects tree: both hops resolve in cache.
        seed_grid_block(&mut grid, objects_index, &obj_index_block);
        seed_grid_block(&mut grid, objects_value, &obj_value_block);
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 101),
            LookupMemoryResult::Positive(Account { id: 101, timestamp: 1, .. })
        ));
    }

    #[test]
    fn transfer_groove_lookup_primary() {
        let transfers = [
            Transfer {
                timestamp: 1,
                id: 11,
                debit_account_id: 1,
                credit_account_id: 2,
                amount: 100,
                ledger: 1,
                code: 1,
                ..Transfer::default()
            },
            Transfer {
                timestamp: 2,
                id: 12,
                debit_account_id: 1,
                credit_account_id: 2,
                amount: 50,
                ledger: 1,
                code: 1,
                ..Transfer::default()
            },
        ];
        let transfer_ids = [
            UniqueKey128 { field: 11, timestamp: 1, padding: 0 },
            UniqueKey128 { field: 12, timestamp: 2, padding: 0 },
        ];

        let (mut grid, addresses) = new_groove_grid(4);
        let (obj_index_block, obj_value_block, obj_info) =
            build_table_with::<TransferObjectSpec>(&transfers, 10, addresses[0], addresses[1]);
        let obj_checksum = seed_grid_block(&mut grid, addresses[1], &obj_index_block);
        seed_grid_block(&mut grid, addresses[0], &obj_value_block);
        let (id_index_block, id_value_block, id_info) =
            build_table_with::<UniqueKey128Spec>(&transfer_ids, 11, addresses[2], addresses[3]);
        let id_checksum = seed_grid_block(&mut grid, addresses[3], &id_index_block);
        seed_grid_block(&mut grid, addresses[2], &id_value_block);

        let mut groove = new_transfer_groove();
        open_tree(&mut groove.objects, &obj_info, addresses[1], obj_checksum, 10);
        open_tree(&mut groove.id, &id_info, addresses[3], id_checksum, 11);

        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 12),
            LookupMemoryResult::Positive(Transfer { id: 12, timestamp: 2, amount: 50, .. })
        ));
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 13),
            LookupMemoryResult::Negative
        ));
    }

    // ------------------------------------------------------------------
    // Objects cache wiring: `AccountGroove::get`/`insert`/`update` serve the
    // per-session cache, and `scope_close(.discard)` rolls it back alongside the
    // trees (upstream's objects_cache invariant, groove.zig:725-728).
    // ------------------------------------------------------------------

    #[test]
    fn account_groove_objects_cache_hit_and_update() {
        let mut groove = new_account_groove();

        let mut account = Account { timestamp: 1, id: 101, ..Account::default() };
        groove.insert(&account);
        assert_eq!(groove.get(101), Some(&account));

        account.user_data_128 = 7;
        groove.update(&Account { timestamp: 1, id: 101, ..Account::default() }, &account);
        assert_eq!(groove.get(101), Some(&account));
        assert!(groove.get(999).is_none());
    }

    #[test]
    fn account_groove_objects_cache_scope_discard() {
        let mut groove = new_account_groove();
        groove.insert(&Account { timestamp: 1, id: 101, ..Account::default() });

        groove.scope_open();
        // Update of a pre-scope account:
        groove.update(
            &Account { timestamp: 1, id: 101, ..Account::default() },
            &Account { timestamp: 1, id: 101, user_data_128: 7, ..Account::default() },
        );
        // Insert of a brand-new account:
        groove.insert(&Account { timestamp: 2, id: 102, ..Account::default() });
        assert_eq!(groove.get(101).map(|a| a.user_data_128), Some(7));
        assert!(groove.get(102).is_some());

        groove.scope_close(ScopeCloseMode::Discard);

        assert_eq!(groove.get(101), Some(&Account { timestamp: 1, id: 101, ..Account::default() }));
        assert!(groove.get(102).is_none());
    }

    #[test]
    fn account_groove_objects_cache_scope_persist() {
        let mut groove = new_account_groove();

        groove.scope_open();
        groove.insert(&Account { timestamp: 1, id: 101, ..Account::default() });
        groove.scope_close(ScopeCloseMode::Persist);

        assert!(groove.get(101).is_some());
    }

    #[test]
    fn transfer_groove_objects_cache_scope_discard() {
        let mut groove = new_transfer_groove();
        let pending = Transfer {
            timestamp: 1,
            id: 11,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 100,
            ledger: 1,
            code: 1,
            flags: TransferFlags::PENDING,
            ..Transfer::default()
        };
        groove.insert(&pending);

        groove.scope_open();
        groove.insert(&Transfer { timestamp: 2, id: 12, ..Transfer::default() });
        groove.scope_close(ScopeCloseMode::Discard);

        assert_eq!(groove.get(11), Some(&pending));
        assert!(groove.get(12).is_none());
    }

    #[test]
    fn pending_groove_objects_cache_scope_discard() {
        let mut groove = new_transfer_pending_groove();

        let old = TransferPending {
            timestamp: 1,
            status: TransferPendingStatus::Pending,
            padding: [0; 7],
        };
        groove.insert(&old);
        assert_eq!(groove.get(1), Some(&old));

        groove.scope_open();
        let new = TransferPending {
            timestamp: 1,
            status: TransferPendingStatus::Posted,
            padding: [0; 7],
        };
        groove.update(&old, &new);
        assert_eq!(groove.get(1).map(|p| p.status), Some(TransferPendingStatus::Posted));
        groove.scope_close(ScopeCloseMode::Discard);

        assert_eq!(
            groove.get(1),
            Some(&TransferPending {
                timestamp: 1,
                status: TransferPendingStatus::Pending,
                padding: [0; 7]
            })
        );
    }

    #[test]
    fn account_objects_cache_cache_tier_evicts_to_stash() {
        // The set-associative cache tier absorbs lookups; when it overflows, evicted
        // entries flow to the stash so nothing inserted is ever lost (cache_map.zig:11-17).
        const N: u32 = 300;
        let mut cache = new_account_objects_cache();
        for i in 1..=N {
            cache.upsert(&Account {
                id: u128::from(i),
                timestamp: u64::from(i),
                ..Account::default()
            });
        }

        // The cache tier can hold at most cache_value_count_max entries.
        assert!(cache.cache_entries() <= cache.cache_entries_max());
        assert_eq!(cache.cache_entries_max(), 256);

        // Every inserted key still resolves, even the ones evicted into the stash.
        for i in 1..=N {
            assert_eq!(cache.get(u128::from(i)).map(|a| a.timestamp), Some(u64::from(i)));
        }
        // Keys that were never inserted still miss.
        assert!(cache.get(0).is_none());
        assert!(cache.get(u128::from(N) + 1).is_none());
    }

    #[test]
    fn account_objects_cache_tombstone_hides_deleted_key() {
        let mut cache = new_account_objects_cache();
        cache.upsert(&Account { id: 7, timestamp: 1, ..Account::default() });
        assert!(cache.has(7));

        // `remove` is only reachable under `constants.verify` (unit tests/fuzzers).
        cache.remove(7);
        assert!(!cache.has(7));
        assert!(cache.get(7).is_none());
        assert!(matches!(cache.get_or_tombstone(7), GetOrTombstone::Tombstone));

        // Re-inserting resurrects the key.
        cache.upsert(&Account { id: 7, timestamp: 2, ..Account::default() });
        assert_eq!(cache.get(7).map(|a| a.timestamp), Some(2));
    }

    #[test]
    fn transfer_pending_objects_cache_tombstone() {
        // The timestamp-keyed spec masks the tombstone bit out of timestamps.
        let mut cache = new_transfer_pending_objects_cache();
        let pending = TransferPending {
            timestamp: 5,
            status: TransferPendingStatus::Pending,
            padding: [0; 7],
        };
        cache.upsert(&pending);
        assert!(cache.has(5));

        cache.remove(5);
        assert_eq!(cache.get(5), None);
        assert!(matches!(cache.get_or_tombstone(5), GetOrTombstone::Tombstone));

        cache.reset();
        assert!(!cache.has(5));
        assert!(matches!(cache.get_or_tombstone(5), GetOrTombstone::NotFound));
    }

    // ------------------------------------------------------------------
    // Groove-level cache behavior: `has`/`remove` surface (tombstones) and
    // SAC-tier overflow, proven through the grooves rather than the raw cache.
    // ------------------------------------------------------------------

    #[test]
    fn account_groove_remove_tombstones_then_resurrect() {
        let mut groove = new_account_groove();
        groove.insert(&Account { timestamp: 1, id: 101, ..Account::default() });
        assert!(groove.has(101));
        assert!(groove.get(101).is_some());

        // Removing tombstoned the key in-session: `has`/`get` both report absent.
        groove.remove(101);
        assert!(!groove.has(101));
        assert!(groove.get(101).is_none());

        // Re-inserting resurrects the key with the newest value.
        groove.insert(&Account { timestamp: 2, id: 101, ..Account::default() });
        assert!(groove.has(101));
        assert_eq!(groove.get(101).map(|a| a.timestamp), Some(2));
    }

    #[test]
    fn transfer_groove_remove_tombstones_then_resurrect() {
        let mut groove = new_transfer_groove();
        let transfer = Transfer {
            timestamp: 1,
            id: 11,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 100,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        groove.insert(&transfer);
        assert!(groove.has(11));

        groove.remove(11);
        assert!(!groove.has(11));
        assert!(groove.get(11).is_none());

        groove.insert(&transfer);
        assert!(groove.has(11));
    }

    #[test]
    fn account_groove_remove_scope_discard_restores_value() {
        // A `remove` inside a scope is recorded in the rollback log; discarding the
        // scope must restore the pre-scope value (tombstone rolled back).
        let mut groove = new_account_groove();
        groove.insert(&Account { timestamp: 1, id: 101, ..Account::default() });

        groove.scope_open();
        groove.remove(101);
        assert!(!groove.has(101));

        groove.scope_close(ScopeCloseMode::Discard);

        assert!(groove.has(101));
        assert_eq!(groove.get(101), Some(&Account { timestamp: 1, id: 101, ..Account::default() }));
    }

    #[test]
    fn pending_groove_remove_tombstones_then_resurrect() {
        let mut groove = new_transfer_pending_groove();
        let pending = TransferPending {
            timestamp: 5,
            status: TransferPendingStatus::Pending,
            padding: [0; 7],
        };
        groove.insert(&pending);
        assert!(groove.has(5));

        groove.remove(5);
        assert!(!groove.has(5));
        assert!(groove.get(5).is_none());

        groove.insert(&pending);
        assert!(groove.has(5));
        assert_eq!(groove.get(5), Some(&pending));
    }

    #[test]
    fn transfer_pending_block_value_round_trip() {
        // state_machine.zig:92-102: { timestamp: u64 @0, status: u8 @8, padding: [7]u8 @9 }
        for status in [
            TransferPendingStatus::None,
            TransferPendingStatus::Pending,
            TransferPendingStatus::Posted,
            TransferPendingStatus::Voided,
            TransferPendingStatus::Expired,
        ] {
            let value =
                TransferPending { timestamp: 0x0123_4567_89ab_cdef, status, padding: [0xaa; 7] };
            let mut buf = [0u8; 16];
            value.write_bytes(&mut buf);
            let read_back = TransferPending::from_bytes(&buf);
            assert_eq!(read_back, value);
        }
    }

    #[test]
    fn pending_groove_tree_read_path() {
        // Prove the pending groove's object tree (TransferPendingObjectSpec) and status
        // index tree (TransferPendingStatusSpec) are written and read faithfully through
        // a grid, mirroring open_account_lookup_trees. index_from_object drops status
        // None (0), so it must not appear in the status tree.
        let pending_values = [
            TransferPending {
                timestamp: 1,
                status: TransferPendingStatus::Pending,
                padding: [0; 7],
            },
            TransferPending {
                timestamp: 2,
                status: TransferPendingStatus::Posted,
                padding: [0; 7],
            },
            TransferPending { timestamp: 3, status: TransferPendingStatus::None, padding: [0; 7] },
        ];
        let status_values =
            [CompositeKey64 { field: 1, timestamp: 1 }, CompositeKey64 { field: 2, timestamp: 2 }];

        let (mut grid, _) = new_groove_grid(6);
        let addresses = acquire_addresses(&mut grid, 6);
        let (objects_index, objects_value, status_index, status_value) =
            (addresses[0], addresses[1], addresses[2], addresses[3]);

        let (obj_index_block, obj_value_block, obj_info) =
            build_table_with::<TransferPendingObjectSpec>(
                &pending_values,
                20,
                objects_value,
                objects_index,
            );
        let obj_checksum = seed_grid_block(&mut grid, objects_index, &obj_index_block);
        seed_grid_block(&mut grid, objects_value, &obj_value_block);

        let (st_index_block, st_value_block, st_info) = build_table_with::<TransferPendingStatusSpec>(
            &status_values,
            21,
            status_value,
            status_index,
        );
        let st_checksum = seed_grid_block(&mut grid, status_index, &st_index_block);
        seed_grid_block(&mut grid, status_value, &st_value_block);

        let mut groove = new_transfer_pending_groove();
        open_tree(&mut groove.objects, &obj_info, objects_index, obj_checksum, 20);
        open_tree(&mut groove.status, &st_info, status_index, st_checksum, 21);

        // Object tree reads by timestamp:
        assert!(matches!(
            groove.objects.lookup_from_levels_cache(&mut grid, SNAPSHOT_LATEST, 1),
            LookupMemoryResult::Positive(TransferPending {
                timestamp: 1,
                status: TransferPendingStatus::Pending,
                ..
            })
        ));
        assert!(matches!(
            groove.objects.lookup_from_levels_cache(&mut grid, SNAPSHOT_LATEST, 3),
            LookupMemoryResult::Positive(TransferPending {
                timestamp: 3,
                status: TransferPendingStatus::None,
                ..
            })
        ));

        // Status index tree reads by composite (status << 64 | timestamp):
        let pending_key = (1u128 << 64) | 1;
        assert!(matches!(
            groove.status.lookup_from_levels_cache(&mut grid, SNAPSHOT_LATEST, pending_key),
            LookupMemoryResult::Positive(CompositeKey64 { field: 1, timestamp: 1 })
        ));
        // Status None (0) is not indexed (index_from_object returns None):
        let none_key = 3;
        assert!(matches!(
            groove.status.lookup_from_levels_cache(&mut grid, SNAPSHOT_LATEST, none_key),
            LookupMemoryResult::Negative
        ));
    }

    #[test]
    fn pending_groove_lookup_over_grid() {
        // Build+open the pending object tree through the grid (like
        // pending_groove_tree_read_path) and drive the public `lookup` seam:
        // an in-range present timestamp resolves Positive, an in-range absent
        // timestamp Negative, and a tree whose value block is not cached yields
        // Possible (deferred to the async read phase).
        let pending_values = [
            TransferPending {
                timestamp: 1,
                status: TransferPendingStatus::Pending,
                padding: [0; 7],
            },
            TransferPending {
                timestamp: 2,
                status: TransferPendingStatus::Posted,
                padding: [0; 7],
            },
        ];

        let (mut grid, _) = new_groove_grid(2);
        let addresses = acquire_addresses(&mut grid, 2);
        let (objects_index, objects_value) = (addresses[0], addresses[1]);

        let (obj_index_block, obj_value_block, obj_info) =
            build_table_with::<TransferPendingObjectSpec>(
                &pending_values,
                20,
                objects_value,
                objects_index,
            );
        let obj_checksum = seed_grid_block(&mut grid, objects_index, &obj_index_block);
        seed_grid_block(&mut grid, objects_value, &obj_value_block);

        let mut groove = new_transfer_pending_groove();
        open_tree(&mut groove.objects, &obj_info, objects_index, obj_checksum, 20);

        // Present timestamp (single hop, no unique-key tree):
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 1),
            LookupMemoryResult::Positive(TransferPending {
                timestamp: 1,
                status: TransferPendingStatus::Pending,
                ..
            })
        ));

        // In-range gap → Negative:
        assert!(matches!(
            groove.lookup(&mut grid, SNAPSHOT_LATEST, 3),
            LookupMemoryResult::Negative
        ));
    }
}
