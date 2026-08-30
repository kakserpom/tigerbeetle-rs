//! Domain object types stored in LSM grooves.
//!
//! Upstream: `src/tigerbeetle.zig` — Account, Transfer, and their flags.
//!
//! DEVIATION: upstream uses `extern struct` with comptime `no_padding` assertions;
//! Rust uses `#[repr(C)]` and static `size_of` / `align_of` assertions instead.

use core::fmt;

/// Nanoseconds per second, used by [`Transfer::timeout_ns`].
pub const NS_PER_S: u64 = 1_000_000_000;

// ---------------------------------------------------------------------------
// Account
// ---------------------------------------------------------------------------

/// A `TigerBeetle` account (128 bytes, 16-byte aligned).
///
/// Upstream: `src/tigerbeetle.zig:10`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Account {
    pub id: u128,
    pub debits_pending: u128,
    pub debits_posted: u128,
    pub credits_pending: u128,
    pub credits_posted: u128,
    pub user_data_128: u128,
    pub user_data_64: u64,
    pub user_data_32: u32,
    pub reserved: u32,
    pub ledger: u32,
    pub code: u16,
    pub flags: AccountFlags,
    pub timestamp: u64,
}

const _: () = assert!(core::mem::size_of::<Account>() == 128);
const _: () = assert!(core::mem::align_of::<Account>() == 16);

impl Account {
    #[must_use]
    pub const fn debits_exceed_credits(self, amount: u128) -> bool {
        self.flags.debits_must_not_exceed_credits()
            && self.debits_pending.wrapping_add(self.debits_posted).wrapping_add(amount)
                > self.credits_posted
    }

    #[must_use]
    pub const fn credits_exceed_debits(self, amount: u128) -> bool {
        self.flags.credits_must_not_exceed_debits()
            && self.credits_pending.wrapping_add(self.credits_posted).wrapping_add(amount)
                > self.debits_posted
    }
}

impl fmt::Debug for Account {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Account")
            .field("id", &DisplayHex(self.id))
            .field("timestamp", &self.timestamp)
            .field("ledger", &self.ledger)
            .field("code", &self.code)
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// AccountFlags
// ---------------------------------------------------------------------------

/// Bitflags for [`Account`].
///
/// Upstream: `src/tigerbeetle.zig:45` — `packed struct(u16)`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct AccountFlags(u16);

impl AccountFlags {
    pub const LINKED: Self = Self(1 << 0);
    pub const DEBITS_MUST_NOT_EXCEED_CREDITS: Self = Self(1 << 1);
    pub const CREDITS_MUST_NOT_EXCEED_DEBITS: Self = Self(1 << 2);
    pub const HISTORY: Self = Self(1 << 3);
    pub const IMPORTED: Self = Self(1 << 4);
    pub const CLOSED: Self = Self(1 << 5);

    /// Construct from a raw `u16` (for block-value decoding).
    #[must_use]
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// Return the raw `u16` value.
    #[must_use]
    pub const fn as_raw(self) -> u16 {
        self.0
    }

    #[must_use]
    pub const fn linked(self) -> bool {
        (self.0 & Self::LINKED.0) != 0
    }

    #[must_use]
    pub const fn debits_must_not_exceed_credits(self) -> bool {
        (self.0 & Self::DEBITS_MUST_NOT_EXCEED_CREDITS.0) != 0
    }

    #[must_use]
    pub const fn credits_must_not_exceed_debits(self) -> bool {
        (self.0 & Self::CREDITS_MUST_NOT_EXCEED_DEBITS.0) != 0
    }

    #[must_use]
    pub const fn history(self) -> bool {
        (self.0 & Self::HISTORY.0) != 0
    }

    #[must_use]
    pub const fn imported(self) -> bool {
        (self.0 & Self::IMPORTED.0) != 0
    }

    #[must_use]
    pub const fn closed(self) -> bool {
        (self.0 & Self::CLOSED.0) != 0
    }

    /// Return these flags with `CLOSED` set (upstream `flags.closed = true`).
    #[must_use]
    pub const fn with_closed(self) -> Self {
        Self::from_raw(self.0 | Self::CLOSED.0)
    }

    /// Return these flags with `CLOSED` cleared (upstream `flags.closed = false`).
    #[must_use]
    pub const fn without_closed(self) -> Self {
        Self::from_raw(self.0 & !Self::CLOSED.0)
    }

    /// Returns `true` if any reserved/padding bits are set.
    #[must_use]
    pub const fn has_padding(self) -> bool {
        // Bits 9..15 are reserved padding in upstream.
        (self.0 & !0x1FF) != 0
    }

    /// Returns `true` if `debits_must_not_exceed_credits` and `credits_must_not_exceed_debits`
    /// are both set — these flags are mutually exclusive.
    #[must_use]
    pub const fn are_mutually_exclusive(self) -> bool {
        self.debits_must_not_exceed_credits() && self.credits_must_not_exceed_debits()
    }
}

impl core::ops::BitOr for AccountFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

const _: () = assert!(core::mem::size_of::<AccountFlags>() == 2);
const _: () = assert!(core::mem::align_of::<AccountFlags>() == 2);

impl fmt::Debug for AccountFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("AccountFlags");
        if self.linked() {
            s.field("linked", &true);
        }
        if self.debits_must_not_exceed_credits() {
            s.field("debits_must_not_exceed_credits", &true);
        }
        if self.credits_must_not_exceed_debits() {
            s.field("credits_must_not_exceed_debits", &true);
        }
        if self.history() {
            s.field("history", &true);
        }
        if self.imported() {
            s.field("imported", &true);
        }
        if self.closed() {
            s.field("closed", &true);
        }
        s.finish()
    }
}

// ---------------------------------------------------------------------------
// Transfer
// ---------------------------------------------------------------------------

/// A `TigerBeetle` transfer (128 bytes, 16-byte aligned).
///
/// Upstream: `src/tigerbeetle.zig:85`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Transfer {
    pub id: u128,
    pub debit_account_id: u128,
    pub credit_account_id: u128,
    pub amount: u128,
    pub pending_id: u128,
    pub user_data_128: u128,
    pub user_data_64: u64,
    pub user_data_32: u32,
    pub timeout: u32,
    pub ledger: u32,
    pub code: u16,
    pub flags: TransferFlags,
    pub timestamp: u64,
}

const _: () = assert!(core::mem::size_of::<Transfer>() == 128);
const _: () = assert!(core::mem::align_of::<Transfer>() == 16);

impl Transfer {
    /// Convert the timeout from seconds to nanoseconds.
    #[must_use]
    pub const fn timeout_ns(self) -> u64 {
        self.timeout as u64 * NS_PER_S
    }
}

const _: () = assert!(core::mem::size_of::<Account>() == 128);

impl fmt::Debug for Transfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transfer")
            .field("id", &DisplayHex(self.id))
            .field("debit_account_id", &DisplayHex(self.debit_account_id))
            .field("credit_account_id", &DisplayHex(self.credit_account_id))
            .field("amount", &self.amount)
            .field("timestamp", &self.timestamp)
            .field("ledger", &self.ledger)
            .field("code", &self.code)
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// AccountEvent
// ---------------------------------------------------------------------------

/// Change-data-capture record for an account balance change (256 bytes,
/// 16-byte aligned).
///
/// Written once per committed transfer (creation, posting, voiding; expiry
/// events later), always capturing both the debit and credit account snapshots
/// *after* the mutation. The `transfer_pending_*` fields identify the pending
/// transfer a posting/voiding/expiring event refers to.
///
/// Upstream: `src/state_machine.zig:104`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct AccountEvent {
    pub dr_account_id: u128,
    pub dr_debits_pending: u128,
    pub dr_debits_posted: u128,
    pub dr_credits_pending: u128,
    pub dr_credits_posted: u128,
    pub cr_account_id: u128,
    pub cr_debits_pending: u128,
    pub cr_debits_posted: u128,
    pub cr_credits_pending: u128,
    pub cr_credits_posted: u128,
    pub timestamp: u64,
    pub dr_account_timestamp: u64,
    pub cr_account_timestamp: u64,
    pub dr_account_flags: AccountFlags,
    pub cr_account_flags: AccountFlags,
    pub transfer_flags: TransferFlags,
    pub transfer_pending_flags: TransferFlags,
    pub transfer_pending_id: u128,
    pub amount_requested: u128,
    pub amount: u128,
    pub ledger: u32,
    pub transfer_pending_status: TransferPendingStatus,
    pub reserved: [u8; 11],
}

const _: () = assert!(core::mem::size_of::<AccountEvent>() == 256);
const _: () = assert!(core::mem::align_of::<AccountEvent>() == 16);

// ---------------------------------------------------------------------------
// ChangeEvent
// ---------------------------------------------------------------------------

/// The kind of balance-level change a [`ChangeEvent`] records.
///
/// Upstream: `src/tigerbeetle.zig:614`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ChangeEventType {
    #[default]
    SinglePhase = 0,
    TwoPhasePending = 1,
    TwoPhasePosted = 2,
    TwoPhaseVoided = 3,
    TwoPhaseExpired = 4,
}

const _: () = assert!(core::mem::size_of::<ChangeEventType>() == 1);

/// A single account-balance change returned by `get_change_events` (384 bytes,
/// 16-byte aligned — the size of one transfer plus two accounts).
///
/// Upstream: `src/tigerbeetle.zig:622`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ChangeEvent {
    pub transfer_id: u128,
    pub transfer_amount: u128,
    pub transfer_pending_id: u128,
    pub transfer_user_data_128: u128,
    pub transfer_user_data_64: u64,
    pub transfer_user_data_32: u32,
    pub transfer_timeout: u32,
    pub transfer_code: u16,
    pub transfer_flags: TransferFlags,

    pub ledger: u32,
    pub r#type: ChangeEventType,
    pub reserved: [u8; 39],

    pub debit_account_id: u128,
    pub debit_account_debits_pending: u128,
    pub debit_account_debits_posted: u128,
    pub debit_account_credits_pending: u128,
    pub debit_account_credits_posted: u128,
    pub debit_account_user_data_128: u128,
    pub debit_account_user_data_64: u64,
    pub debit_account_user_data_32: u32,
    pub debit_account_code: u16,
    pub debit_account_flags: AccountFlags,

    pub credit_account_id: u128,
    pub credit_account_debits_pending: u128,
    pub credit_account_debits_posted: u128,
    pub credit_account_credits_pending: u128,
    pub credit_account_credits_posted: u128,
    pub credit_account_user_data_128: u128,
    pub credit_account_user_data_64: u64,
    pub credit_account_user_data_32: u32,
    pub credit_account_code: u16,
    pub credit_account_flags: AccountFlags,

    pub timestamp: u64,
    pub transfer_timestamp: u64,
    pub debit_account_timestamp: u64,
    pub credit_account_timestamp: u64,
}

const _: () = assert!(core::mem::size_of::<ChangeEvent>() == 384);
const _: () = assert!(core::mem::align_of::<ChangeEvent>() == 16);
const _: () = assert!(core::mem::align_of::<ChangeEvent>() == 16);

// ---------------------------------------------------------------------------
// ChangeEventsFilter
// ---------------------------------------------------------------------------

/// Query filter for `get_change_events` (64 bytes). A zero bound is
/// "unbounded"; `limit != 0` is required.
///
/// Upstream: `src/tigerbeetle.zig:672`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ChangeEventsFilter {
    pub timestamp_min: u64,
    pub timestamp_max: u64,
    pub limit: u32,
    pub reserved: [u8; 44],
}

impl Default for ChangeEventsFilter {
    fn default() -> Self {
        Self { timestamp_min: 0, timestamp_max: 0, limit: 0, reserved: [0; 44] }
    }
}

const _: () = assert!(core::mem::size_of::<ChangeEventsFilter>() == 64);

// ---------------------------------------------------------------------------
// TransferFlags
// ---------------------------------------------------------------------------

/// Bitflags for [`Transfer`].
///
/// Upstream: `src/tigerbeetle.zig:132` — `packed struct(u16)`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct TransferFlags(u16);

impl TransferFlags {
    pub const LINKED: Self = Self(1 << 0);
    pub const PENDING: Self = Self(1 << 1);
    pub const POST_PENDING_TRANSFER: Self = Self(1 << 2);
    pub const VOID_PENDING_TRANSFER: Self = Self(1 << 3);
    pub const BALANCING_DEBIT: Self = Self(1 << 4);
    pub const BALANCING_CREDIT: Self = Self(1 << 5);
    pub const CLOSING_DEBIT: Self = Self(1 << 6);
    pub const CLOSING_CREDIT: Self = Self(1 << 7);
    pub const IMPORTED: Self = Self(1 << 8);

    /// Construct from a raw `u16` (for block-value decoding).
    #[must_use]
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// Return the raw `u16` value.
    #[must_use]
    pub const fn as_raw(self) -> u16 {
        self.0
    }

    #[must_use]
    pub const fn linked(self) -> bool {
        (self.0 & Self::LINKED.0) != 0
    }

    #[must_use]
    pub const fn pending(self) -> bool {
        (self.0 & Self::PENDING.0) != 0
    }

    #[must_use]
    pub const fn post_pending_transfer(self) -> bool {
        (self.0 & Self::POST_PENDING_TRANSFER.0) != 0
    }

    #[must_use]
    pub const fn void_pending_transfer(self) -> bool {
        (self.0 & Self::VOID_PENDING_TRANSFER.0) != 0
    }

    #[must_use]
    pub const fn balancing_debit(self) -> bool {
        (self.0 & Self::BALANCING_DEBIT.0) != 0
    }

    #[must_use]
    pub const fn balancing_credit(self) -> bool {
        (self.0 & Self::BALANCING_CREDIT.0) != 0
    }

    #[must_use]
    pub const fn closing_debit(self) -> bool {
        (self.0 & Self::CLOSING_DEBIT.0) != 0
    }

    #[must_use]
    pub const fn closing_credit(self) -> bool {
        (self.0 & Self::CLOSING_CREDIT.0) != 0
    }

    #[must_use]
    pub const fn imported(self) -> bool {
        (self.0 & Self::IMPORTED.0) != 0
    }

    /// Returns `true` if any reserved/padding bits are set.
    #[must_use]
    pub const fn has_padding(self) -> bool {
        // Bits 9..15 are reserved padding in upstream.
        (self.0 & !0x1FF) != 0
    }

    /// Returns `true` if exactly one of `post_pending_transfer` or `void_pending_transfer`
    /// is set (but not both).
    #[must_use]
    pub const fn post_or_void_exclusive(self) -> bool {
        self.post_pending_transfer() ^ self.void_pending_transfer()
    }
}

impl core::ops::BitOr for TransferFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

const _: () = assert!(core::mem::size_of::<TransferFlags>() == 2);
const _: () = assert!(core::mem::align_of::<TransferFlags>() == 2);

impl fmt::Debug for TransferFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("TransferFlags");
        if self.linked() {
            s.field("linked", &true);
        }
        if self.pending() {
            s.field("pending", &true);
        }
        if self.post_pending_transfer() {
            s.field("post_pending_transfer", &true);
        }
        if self.void_pending_transfer() {
            s.field("void_pending_transfer", &true);
        }
        if self.balancing_debit() {
            s.field("balancing_debit", &true);
        }
        if self.balancing_credit() {
            s.field("balancing_credit", &true);
        }
        if self.closing_debit() {
            s.field("closing_debit", &true);
        }
        if self.closing_credit() {
            s.field("closing_credit", &true);
        }
        if self.imported() {
            s.field("imported", &true);
        }
        s.finish()
    }
}

// ---------------------------------------------------------------------------
// TransferPendingStatus
// ---------------------------------------------------------------------------

/// Pending status for a transfer.
///
/// Upstream: `src/tigerbeetle.zig:118`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferPendingStatus {
    #[default]
    None = 0,
    Pending = 1,
    Posted = 2,
    Voided = 3,
    Expired = 4,
}

const _: () = assert!(core::mem::size_of::<TransferPendingStatus>() == 1);

// ---------------------------------------------------------------------------
// TransferPending
// ---------------------------------------------------------------------------

/// Pending transfer record — 16 bytes.
///
/// Stored in the `transfers_pending` groove. Keyed by the pending transfer's
/// creation timestamp.
///
/// Upstream: `src/state_machine.zig:92`.
///
/// Fields: `timestamp`, `status` ([`TransferPendingStatus`]), `padding`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct TransferPending {
    pub timestamp: u64,
    pub status: TransferPendingStatus,
    pub padding: [u8; 7],
}

const _: () = assert!(core::mem::size_of::<TransferPending>() == 16);
const _: () = assert!(core::mem::align_of::<TransferPending>() == 8);

// ---------------------------------------------------------------------------
// AccountBalance
// ---------------------------------------------------------------------------

/// Balance snapshot for an account (128 bytes, 16-byte aligned).
///
/// Upstream: `src/tigerbeetle.zig:70`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct AccountBalance {
    pub debits_pending: u128,
    pub debits_posted: u128,
    pub credits_pending: u128,
    pub credits_posted: u128,
    pub timestamp: u64,
    pub reserved: [u8; 56],
}

impl Default for AccountBalance {
    fn default() -> Self {
        Self {
            debits_pending: 0,
            debits_posted: 0,
            credits_pending: 0,
            credits_posted: 0,
            timestamp: 0,
            reserved: [0; 56],
        }
    }
}

const _: () = assert!(core::mem::size_of::<AccountBalance>() == 128);
const _: () = assert!(core::mem::align_of::<AccountBalance>() == 16);

// ---------------------------------------------------------------------------
// AccountFilter
// ---------------------------------------------------------------------------

/// Query flags for [`AccountFilter`] (a `u32` bitfield).
///
/// Upstream: `src/tigerbeetle.zig:599`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct AccountFilterFlags(u32);

impl AccountFilterFlags {
    /// Whether to include results where `debit_account_id` matches.
    pub const DEBITS: Self = Self(1 << 0);
    /// Whether to include results where `credit_account_id` matches.
    pub const CREDITS: Self = Self(1 << 1);
    /// Whether the results are sorted by timestamp in chronological or
    /// reverse-chronological order.
    pub const REVERSED: Self = Self(1 << 2);

    /// Construct from a raw `u32` (for request decoding).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Return the raw `u32` value.
    #[must_use]
    pub const fn as_raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn debits(self) -> bool {
        (self.0 & Self::DEBITS.0) != 0
    }

    #[must_use]
    pub const fn credits(self) -> bool {
        (self.0 & Self::CREDITS.0) != 0
    }

    #[must_use]
    pub const fn reversed(self) -> bool {
        (self.0 & Self::REVERSED.0) != 0
    }

    /// The `padding` bits (upstream `account_filter_flags.padding`) must be
    /// zero.
    #[must_use]
    pub const fn has_padding(self) -> bool {
        (self.0 & !(Self::DEBITS.0 | Self::CREDITS.0 | Self::REVERSED.0)) != 0
    }
}

impl core::ops::BitOr for AccountFilterFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Filters for the `get_account_transfers` and `get_account_balances` queries
/// (128 bytes, 16-byte aligned).
///
/// Zero-valued fields act as "no filter". `timestamp_min`/`timestamp_max` bound
/// the inclusive timestamp range of the results; `limit` caps the number of
/// results and must be non-zero.
///
/// Upstream: `src/tigerbeetle.zig:564`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct AccountFilter {
    pub account_id: u128,
    pub user_data_128: u128,
    pub user_data_64: u64,
    pub user_data_32: u32,
    pub code: u16,
    pub reserved: [u8; 58],
    pub timestamp_min: u64,
    pub timestamp_max: u64,
    pub limit: u32,
    pub flags: AccountFilterFlags,
}

impl Default for AccountFilter {
    fn default() -> Self {
        Self {
            account_id: 0,
            user_data_128: 0,
            user_data_64: 0,
            user_data_32: 0,
            code: 0,
            reserved: [0; 58],
            timestamp_min: 0,
            timestamp_max: 0,
            limit: 0,
            flags: AccountFilterFlags::default(),
        }
    }
}

const _: () = assert!(core::mem::size_of::<AccountFilter>() == 128);
const _: () = assert!(core::mem::align_of::<AccountFilter>() == 16);
const _: () = assert!(core::mem::size_of::<AccountFilterFlags>() == 4);

// ---------------------------------------------------------------------------
// QueryFilter
// ---------------------------------------------------------------------------

/// Query flags for the `query_accounts`/`query_transfers` operations (a `u32`
/// bitfield).
///
/// Upstream: `src/tigerbeetle.zig:552`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct QueryFilterFlags(u32);

impl QueryFilterFlags {
    /// Whether the results are sorted by timestamp in chronological or
    /// reverse-chronological order.
    pub const REVERSED: Self = Self(1 << 0);

    /// Construct from a raw `u32` (for request decoding).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Return the raw `u32` value.
    #[must_use]
    pub const fn as_raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn reversed(self) -> bool {
        (self.0 & Self::REVERSED.0) != 0
    }

    /// The `padding` bits (upstream `query_filter_flags.padding`) must be zero.
    #[must_use]
    pub const fn has_padding(self) -> bool {
        (self.0 & !Self::REVERSED.0) != 0
    }
}

impl core::ops::BitOr for QueryFilterFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Filters for the `query_accounts` and `query_transfers` operations
/// (64 bytes, 16-byte aligned).
///
/// Applies an AND of the non-zero `user_data_*`/`ledger`/`code` equality
/// filters over the timestamp range `[timestamp_min, timestamp_max]`; zero
/// fields act as "no filter". `limit` caps the number of results and must be
/// non-zero.
///
/// Upstream: `src/tigerbeetle.zig:517`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct QueryFilter {
    pub user_data_128: u128,
    pub user_data_64: u64,
    pub user_data_32: u32,
    pub ledger: u32,
    pub code: u16,
    pub reserved: [u8; 6],
    pub timestamp_min: u64,
    pub timestamp_max: u64,
    pub limit: u32,
    pub flags: QueryFilterFlags,
}

const _: () = assert!(core::mem::size_of::<QueryFilter>() == 64);
const _: () = assert!(core::mem::align_of::<QueryFilter>() == 16);
const _: () = assert!(core::mem::size_of::<QueryFilterFlags>() == 4);

// ---------------------------------------------------------------------------
// Tree IDs
// ---------------------------------------------------------------------------

/// Numeric tree IDs — one per tree in the forest. Each groove contains one object tree
/// and N index trees; every tree in the forest has a unique `u16` ID.
///
/// Upstream: `src/state_machine.zig:45`.
pub mod tree_ids {
    /// Account groove tree IDs (9 trees).
    pub mod account {
        pub const ID: u16 = 1;
        pub const USER_DATA_128: u16 = 2;
        pub const USER_DATA_64: u16 = 3;
        pub const USER_DATA_32: u16 = 4;
        pub const LEDGER: u16 = 5;
        pub const CODE: u16 = 6;
        pub const TIMESTAMP: u16 = 7;
        pub const IMPORTED: u16 = 23;
        pub const CLOSED: u16 = 25;
    }

    /// Transfer groove tree IDs (14 trees).
    pub mod transfer {
        pub const ID: u16 = 8;
        pub const DEBIT_ACCOUNT_ID: u16 = 9;
        pub const CREDIT_ACCOUNT_ID: u16 = 10;
        pub const AMOUNT: u16 = 11;
        pub const PENDING_ID: u16 = 12;
        pub const USER_DATA_128: u16 = 13;
        pub const USER_DATA_64: u16 = 14;
        pub const USER_DATA_32: u16 = 15;
        pub const LEDGER: u16 = 16;
        pub const CODE: u16 = 17;
        pub const TIMESTAMP: u16 = 18;
        pub const EXPIRES_AT: u16 = 19;
        pub const IMPORTED: u16 = 24;
        pub const CLOSING: u16 = 26;
    }

    /// `TransferPending` groove tree IDs (2 trees).
    pub mod transfer_pending {
        pub const ID: u16 = 20;
        pub const STATUS: u16 = 21;
    }
}

// ---------------------------------------------------------------------------
// CreateAccountStatus
// ---------------------------------------------------------------------------

/// Result code for `create_accounts`. Values are wire-protocol-stable.
///
/// Upstream: `src/tigerbeetle.zig:153`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CreateAccountStatus {
    DeprecatedOk = 0,
    LinkedEventFailed = 1,
    LinkedEventChainOpen = 2,
    TimestampMustBeZero = 3,
    ReservedField = 4,
    ReservedFlag = 5,
    IdMustNotBeZero = 6,
    IdMustNotBeIntMax = 7,
    FlagsAreMutuallyExclusive = 8,
    DebitsPendingMustBeZero = 9,
    DebitsPostedMustBeZero = 10,
    CreditsPendingMustBeZero = 11,
    CreditsPostedMustBeZero = 12,
    LedgerMustNotBeZero = 13,
    CodeMustNotBeZero = 14,
    ExistsWithDifferentFlags = 15,
    ExistsWithDifferentUserData128 = 16,
    ExistsWithDifferentUserData64 = 17,
    ExistsWithDifferentUserData32 = 18,
    ExistsWithDifferentLedger = 19,
    ExistsWithDifferentCode = 20,
    Exists = 21,
    ImportedEventExpected = 22,
    ImportedEventNotExpected = 23,
    ImportedEventTimestampOutOfRange = 24,
    ImportedEventTimestampMustNotAdvance = 25,
    ImportedEventTimestampMustNotRegress = 26,
    Created = u32::MAX,
}

// ---------------------------------------------------------------------------
// CreateTransferStatus
// ---------------------------------------------------------------------------

/// Result code for `create_transfers`. Values are wire-protocol-stable.
///
/// Upstream: `src/tigerbeetle.zig:220`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CreateTransferStatus {
    DeprecatedOk = 0,
    LinkedEventFailed = 1,
    LinkedEventChainOpen = 2,
    TimestampMustBeZero = 3,
    ReservedFlag = 4,
    IdMustNotBeZero = 5,
    IdMustNotBeIntMax = 6,
    FlagsAreMutuallyExclusive = 7,
    DebitAccountIdMustNotBeZero = 8,
    DebitAccountIdMustNotBeIntMax = 9,
    CreditAccountIdMustNotBeZero = 10,
    CreditAccountIdMustNotBeIntMax = 11,
    AccountsMustBeDifferent = 12,
    PendingIdMustBeZero = 13,
    PendingIdMustNotBeZero = 14,
    PendingIdMustNotBeIntMax = 15,
    PendingIdMustBeDifferent = 16,
    TimeoutReservedForPendingTransfer = 17,
    LedgerMustNotBeZero = 19,
    CodeMustNotBeZero = 20,
    DebitAccountNotFound = 21,
    CreditAccountNotFound = 22,
    AccountsMustHaveTheSameLedger = 23,
    TransferMustHaveTheSameLedgerAsAccounts = 24,
    PendingTransferNotFound = 25,
    PendingTransferNotPending = 26,
    PendingTransferHasDifferentDebitAccountId = 27,
    PendingTransferHasDifferentCreditAccountId = 28,
    PendingTransferHasDifferentLedger = 29,
    PendingTransferHasDifferentCode = 30,
    ExceedsPendingTransferAmount = 31,
    PendingTransferHasDifferentAmount = 32,
    PendingTransferAlreadyPosted = 33,
    PendingTransferAlreadyVoided = 34,
    PendingTransferExpired = 35,
    ExistsWithDifferentFlags = 36,
    ExistsWithDifferentDebitAccountId = 37,
    ExistsWithDifferentCreditAccountId = 38,
    ExistsWithDifferentAmount = 39,
    ExistsWithDifferentPendingId = 40,
    ExistsWithDifferentUserData128 = 41,
    ExistsWithDifferentUserData64 = 42,
    ExistsWithDifferentUserData32 = 43,
    ExistsWithDifferentTimeout = 44,
    ExistsWithDifferentCode = 45,
    Exists = 46,
    OverflowsDebitsPending = 47,
    OverflowsCreditsPending = 48,
    OverflowsDebitsPosted = 49,
    OverflowsCreditsPosted = 50,
    OverflowsDebits = 51,
    OverflowsCredits = 52,
    OverflowsTimeout = 53,
    ExceedsCredits = 54,
    ExceedsDebits = 55,
    ImportedEventExpected = 56,
    ImportedEventNotExpected = 57,
    ImportedEventTimestampOutOfRange = 58,
    ImportedEventTimestampMustNotAdvance = 59,
    ImportedEventTimestampMustNotRegress = 60,
    ImportedEventTimestampMustPostdateDebitAccount = 61,
    ImportedEventTimestampMustPostdateCreditAccount = 62,
    ImportedEventTimeoutMustBeZero = 63,
    ClosingTransferMustBePending = 64,
    DebitAccountAlreadyClosed = 65,
    CreditAccountAlreadyClosed = 66,
    ExistsWithDifferentLedger = 67,
    IdAlreadyFailed = 68,
    Created = u32::MAX,
}

impl CreateTransferStatus {
    /// Returns `true` if the error code depends on transient system status and retrying
    /// the same transfer with identical request data can produce different outcomes.
    #[must_use]
    pub const fn transient(self) -> bool {
        matches!(
            self,
            Self::DebitAccountNotFound
                | Self::CreditAccountNotFound
                | Self::PendingTransferNotFound
                | Self::ExceedsCredits
                | Self::ExceedsDebits
                | Self::DebitAccountAlreadyClosed
                | Self::CreditAccountAlreadyClosed
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Displays a `u128` as a hex string for `Debug` output.
struct DisplayHex(u128);

impl fmt::Debug for DisplayHex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:032x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn account_size_and_alignment() {
        assert_eq!(core::mem::size_of::<Account>(), 128);
        assert_eq!(core::mem::align_of::<Account>(), 16);
    }

    #[test]
    fn transfer_size_and_alignment() {
        assert_eq!(core::mem::size_of::<Transfer>(), 128);
        assert_eq!(core::mem::align_of::<Transfer>(), 16);
    }

    #[test]
    fn transfer_timeout_ns() {
        let t = Transfer { timeout: 60, ..Transfer::default() };
        assert_eq!(t.timeout_ns(), 60 * NS_PER_S);
    }

    #[test]
    fn account_balance_size() {
        assert_eq!(core::mem::size_of::<AccountBalance>(), 128);
    }
}
