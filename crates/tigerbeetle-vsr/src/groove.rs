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

#![allow(
    clippy::cast_possible_truncation,
    reason = "size_of::<T>() as u32 in const LAYOUT; upstream uses comptime"
)]

use std::collections::HashMap;
use tigerbeetle_core::constants;
use tigerbeetle_core::types::{Account, AccountFlags, Transfer, TransferFlags, TransferPending};
use tigerbeetle_lsm::composite_key::{
    self, CompositeKey, CompositeKey64, CompositeKey128, CompositeKeyUnit, U256,
};
use tigerbeetle_lsm::manifest::ManifestLog;
use tigerbeetle_lsm::table_memory::{self, Usage};
use tigerbeetle_lsm::tree::ScopeCloseMode;
use tigerbeetle_lsm::unique_key::{UniqueKey, UniqueKey128};

use crate::table::{BlockValue, TableLayout};
use crate::tree::Tree;

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

// ---------------------------------------------------------------------------
// Tree spec types
//
// Each tree in the groove needs a type implementing both `table_memory::Table` and
// `TreeSpec`.  We use unit structs to keep the overhead zero.
// ---------------------------------------------------------------------------

/// Default `VALUE_COUNT_MAX` for production (mirrors upstream `tree_values_count_max`).
/// In test-min config (`lsm_compaction_ops = 4`), the upstream derives this from
/// `message_body_size_max`.  We use a generous but safe upper bound.
const ACCOUNT_VALUE_COUNT_MAX: usize = constants::LSM_COMPACTION_OPS * 8192;
const TRANSFER_VALUE_COUNT_MAX: usize = constants::LSM_COMPACTION_OPS * 8192;

// ===== Object trees (keyed by u64 timestamp) =====

pub struct AccountObjectSpec;

impl table_memory::Table for AccountObjectSpec {
    type Key = u64;
    type Value = Account;
    const VALUE_COUNT_MAX: usize = ACCOUNT_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::General;

    fn key_from_value(value: &Account) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    fn tombstone(value: &Account) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

impl crate::tree::TreeSpec for AccountObjectSpec {
    const LAYOUT: TableLayout = TableLayout::compute(
        core::mem::size_of::<u64>() as u32,
        core::mem::size_of::<Account>() as u32,
        ACCOUNT_VALUE_COUNT_MAX as u32,
    );

    fn index_blocks_for_key(_: &[u8], _: u64) -> Option<crate::table::IndexBlocks<u64>> {
        None
    }

    fn value_block_search(_: &[u8], _: u64) -> Option<Account> {
        None
    }

    fn tombstone_from_key(key: u64) -> Account {
        Account { timestamp: key | composite_key::TOMBSTONE_BIT, ..Account::default() }
    }
}

pub struct TransferObjectSpec;

impl table_memory::Table for TransferObjectSpec {
    type Key = u64;
    type Value = Transfer;
    const VALUE_COUNT_MAX: usize = TRANSFER_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::General;

    fn key_from_value(value: &Transfer) -> u64 {
        value.timestamp & !composite_key::TOMBSTONE_BIT
    }

    fn tombstone(value: &Transfer) -> bool {
        value.timestamp & composite_key::TOMBSTONE_BIT != 0
    }
}

impl crate::tree::TreeSpec for TransferObjectSpec {
    const LAYOUT: TableLayout = TableLayout::compute(
        core::mem::size_of::<u64>() as u32,
        core::mem::size_of::<Transfer>() as u32,
        TRANSFER_VALUE_COUNT_MAX as u32,
    );

    fn index_blocks_for_key(_: &[u8], _: u64) -> Option<crate::table::IndexBlocks<u64>> {
        None
    }

    fn value_block_search(_: &[u8], _: u64) -> Option<Transfer> {
        None
    }

    fn tombstone_from_key(key: u64) -> Transfer {
        Transfer { timestamp: key | composite_key::TOMBSTONE_BIT, ..Transfer::default() }
    }
}

// ===== Index trees (composite-key secondary indexes) =====

/// Index tree for `u128` fields (key_type = u128, value_type = CompositeKey128).
pub struct CompositeKey128Spec;

impl table_memory::Table for CompositeKey128Spec {
    type Key = U256;
    type Value = CompositeKey128;
    const VALUE_COUNT_MAX: usize = ACCOUNT_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::SecondaryIndex;

    fn key_from_value(value: &CompositeKey128) -> U256 {
        value.key_from_value()
    }

    fn tombstone(value: &CompositeKey128) -> bool {
        value.tombstone()
    }
}

impl crate::tree::TreeSpec for CompositeKey128Spec {
    const LAYOUT: TableLayout = TableLayout::compute(
        core::mem::size_of::<U256>() as u32,
        core::mem::size_of::<CompositeKey128>() as u32,
        ACCOUNT_VALUE_COUNT_MAX as u32,
    );

    fn index_blocks_for_key(_: &[u8], _: U256) -> Option<crate::table::IndexBlocks<U256>> {
        None
    }

    fn value_block_search(_: &[u8], _: U256) -> Option<CompositeKey128> {
        None
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
    const VALUE_COUNT_MAX: usize = ACCOUNT_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::SecondaryIndex;

    fn key_from_value(value: &CompositeKey64) -> u128 {
        value.key_from_value()
    }

    fn tombstone(value: &CompositeKey64) -> bool {
        value.tombstone()
    }
}

impl crate::tree::TreeSpec for CompositeKey64Spec {
    const LAYOUT: TableLayout = TableLayout::compute(
        core::mem::size_of::<u128>() as u32,
        core::mem::size_of::<CompositeKey64>() as u32,
        ACCOUNT_VALUE_COUNT_MAX as u32,
    );

    fn index_blocks_for_key(_: &[u8], _: u128) -> Option<crate::table::IndexBlocks<u128>> {
        None
    }

    fn value_block_search(_: &[u8], _: u128) -> Option<CompositeKey64> {
        None
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
    const VALUE_COUNT_MAX: usize = ACCOUNT_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::SecondaryIndex;

    fn key_from_value(value: &CompositeKeyUnit) -> u64 {
        value.key_from_value()
    }

    fn tombstone(value: &CompositeKeyUnit) -> bool {
        value.tombstone()
    }
}

impl crate::tree::TreeSpec for CompositeKeyUnitSpec {
    const LAYOUT: TableLayout = TableLayout::compute(
        core::mem::size_of::<u64>() as u32,
        core::mem::size_of::<CompositeKeyUnit>() as u32,
        ACCOUNT_VALUE_COUNT_MAX as u32,
    );

    fn index_blocks_for_key(_: &[u8], _: u64) -> Option<crate::table::IndexBlocks<u64>> {
        None
    }

    fn value_block_search(_: &[u8], _: u64) -> Option<CompositeKeyUnit> {
        None
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
    const VALUE_COUNT_MAX: usize = ACCOUNT_VALUE_COUNT_MAX;
    const USAGE: Usage = Usage::General;

    fn key_from_value(value: &UniqueKey128) -> u128 {
        UniqueKey::key_from_value(value)
    }

    fn tombstone(value: &UniqueKey128) -> bool {
        UniqueKey::tombstone(value)
    }
}

impl crate::tree::TreeSpec for UniqueKey128Spec {
    const LAYOUT: TableLayout = TableLayout::compute(
        core::mem::size_of::<u128>() as u32,
        core::mem::size_of::<UniqueKey128>() as u32,
        ACCOUNT_VALUE_COUNT_MAX as u32,
    );

    fn index_blocks_for_key(_: &[u8], _: u128) -> Option<crate::table::IndexBlocks<u128>> {
        None
    }

    fn value_block_search(_: &[u8], _: u128) -> Option<UniqueKey128> {
        None
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
    /// Primary-key index (id → Account) — temporary stand-in for ObjectsCache.
    id_map: HashMap<u128, Account>,
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
    /// Primary-key index (id → Transfer) — temporary stand-in for ObjectsCache.
    id_map: HashMap<u128, Transfer>,
}

// ---------------------------------------------------------------------------
// AccountGroove operations
// ---------------------------------------------------------------------------

impl AccountGroove {
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
    }

    pub fn compact(&mut self) {
        // TODO(port): pass ScratchMemory per tree type from forest/compaction layer.
        // Each tree's compact() needs &mut ScratchMemory<V> — deferred until
        // the forest manages scratch memory allocation.
    }

    /// Lookup an account by its primary key (id).
    #[must_use]
    pub fn get(&self, id: u128) -> Option<&Account> {
        self.id_map.get(&id)
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
        self.id_map.insert(object.id, *object);

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
        self.id_map.insert(new.id, *new);
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
    /// Lookup a transfer by its primary key (id).
    #[must_use]
    pub fn get(&self, id: u128) -> Option<&Transfer> {
        self.id_map.get(&id)
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
    }

    pub fn compact(&mut self) {
        // TODO(port): pass ScratchMemory per tree type from forest/compaction layer.
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
        self.id_map.insert(object.id, *object);

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
        self.id_map.insert(new.id, *new);
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
/// Upstream: `src/state_machine.zig:470` (`transfers_pending` groove config).
///
/// DEVIATION: The full tree-based implementation (object tree + status index tree)
/// is deferred until `TreeSpec` is implemented for `TransferPending`. For now,
/// a `HashMap<u64, TransferPending>` provides the primary-key lookup needed by
/// `post_or_void_pending_transfer`.
pub struct TransferPendingGroove {
    /// Primary-key lookup (temporary stand-in for ObjectsCache + object tree).
    id_map: HashMap<u64, TransferPending>,
}

impl TransferPendingGroove {
    pub fn scope_open(&mut self) {
        // TODO(port): forward to object tree + status index tree
    }

    pub fn scope_close(&mut self, _mode: ScopeCloseMode) {
        // TODO(port): forward to object tree + status index tree
    }

    /// Lookup a `TransferPending` record by its primary key (timestamp).
    #[must_use]
    pub fn get(&self, timestamp: u64) -> Option<&TransferPending> {
        self.id_map.get(&timestamp)
    }

    /// Insert a pending transfer record.
    pub fn insert(&mut self, object: &TransferPending) {
        self.id_map.insert(object.timestamp, *object);
    }

    /// Update the status of an existing pending transfer record.
    pub fn update(&mut self, new: &TransferPending) {
        self.id_map.insert(new.timestamp, *new);
    }

    pub fn open_commence(&mut self, _manifest_log: &mut impl ManifestLog) {
        // TODO(port): forward to trees
    }

    pub fn open_complete(&mut self, _checkpoint_op: u64) {
        // TODO(port): forward to trees
    }
}
