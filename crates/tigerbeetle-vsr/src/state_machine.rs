// =============================================================================
// State Machine — pure accounting logic
// =============================================================================
//
// Ported from `src/state_machine.zig`. This module contains the core validation
// and mutation logic for `create_accounts` and `create_transfers`.
//
// The StateMachine struct implements the batch orchestrator (linked chains,
// imported timestamps and their validation, persist scopes) on top of plain
// id-keyed timestamp-indexed stores until the forest layer lands; prefetch and
// expiry scheduling remain deferred. Two standalone functions hold the
// per-event validation (`create_account`, `create_transfer`) and expect the
// caller to have looked up referenced objects beforehand.

#![allow(
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::manual_assert_eq,
    clippy::too_many_lines,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::unnecessary_cast,
    clippy::redundant_else,
    clippy::collapsible_if,
    clippy::match_same_arms,
    clippy::module_name_repetitions
)]

use std::collections::HashMap;

use tigerbeetle_core::types::{
    Account, AccountFlags, CreateAccountStatus, CreateTransferStatus, Transfer, TransferFlags,
    TransferPending, TransferPendingStatus,
};

use crate::Operation;

// ---------------------------------------------------------------------------
// Account creation
// ---------------------------------------------------------------------------

/// Validate and insert a single account.
///
/// `timestamp_event` is the VSR-assigned timestamp for this event.
/// `existing` is the result of looking up `a.id` in the account grooves.
///
/// Returns the status code. On `Created`, the caller must insert the account
/// into the groove (zeroed balances).
pub fn create_account(
    a: &Account,
    timestamp_event: u64,
    existing: Option<&Account>,
) -> CreateAccountStatus {
    assert!(timestamp_event != 0);

    if a.reserved != 0 {
        return CreateAccountStatus::ReservedField;
    }
    if a.flags.has_padding() {
        return CreateAccountStatus::ReservedFlag;
    }
    if a.id == 0 {
        return CreateAccountStatus::IdMustNotBeZero;
    }
    if a.id == u128::MAX {
        return CreateAccountStatus::IdMustNotBeIntMax;
    }

    if let Some(e) = existing {
        return create_account_exists(a, e);
    }

    if a.flags.are_mutually_exclusive() {
        return CreateAccountStatus::FlagsAreMutuallyExclusive;
    }
    if a.debits_pending != 0 {
        return CreateAccountStatus::DebitsPendingMustBeZero;
    }
    if a.debits_posted != 0 {
        return CreateAccountStatus::DebitsPostedMustBeZero;
    }
    if a.credits_pending != 0 {
        return CreateAccountStatus::CreditsPendingMustBeZero;
    }
    if a.credits_posted != 0 {
        return CreateAccountStatus::CreditsPostedMustBeZero;
    }
    if a.ledger == 0 {
        return CreateAccountStatus::LedgerMustNotBeZero;
    }
    if a.code == 0 {
        return CreateAccountStatus::CodeMustNotBeZero;
    }

    // Imported timestamps: the regression/collision checks need groove-global
    // state (objects.key_range, indirect_lookup) and run in the batch
    // orchestrator after the idempotency checks (upstream state_machine.zig:3648-3670).
    if !a.flags.imported() {
        assert_eq!(a.timestamp, 0);
    }

    CreateAccountStatus::Created
}

/// Idempotency check for an existing account.
fn create_account_exists(a: &Account, e: &Account) -> CreateAccountStatus {
    assert_eq!(a.id, e.id);
    if a.flags.as_raw() != e.flags.as_raw() {
        return CreateAccountStatus::ExistsWithDifferentFlags;
    }
    if a.user_data_128 != e.user_data_128 {
        return CreateAccountStatus::ExistsWithDifferentUserData128;
    }
    if a.user_data_64 != e.user_data_64 {
        return CreateAccountStatus::ExistsWithDifferentUserData64;
    }
    if a.user_data_32 != e.user_data_32 {
        return CreateAccountStatus::ExistsWithDifferentUserData32;
    }
    assert!(a.reserved == 0);
    assert!(e.reserved == 0);
    if a.ledger != e.ledger {
        return CreateAccountStatus::ExistsWithDifferentLedger;
    }
    if a.code != e.code {
        return CreateAccountStatus::ExistsWithDifferentCode;
    }
    CreateAccountStatus::Exists
}

// ---------------------------------------------------------------------------
// Transfer creation
// ---------------------------------------------------------------------------

/// Validate and insert a single transfer.
///
/// `timestamp_event` is the VSR-assigned timestamp.
/// `dr_account` and `cr_account` must be looked up beforehand by the caller.
/// Returns a `CreateTransferStatus` (created = success).
pub fn create_transfer(
    t: &Transfer,
    timestamp_event: u64,
    existing: Option<&Transfer>,
    dr_account: &Account,
    cr_account: &Account,
) -> CreateTransferStatus {
    assert!(timestamp_event != 0);

    if t.flags.has_padding() {
        return CreateTransferStatus::ReservedFlag;
    }
    if t.id == 0 {
        return CreateTransferStatus::IdMustNotBeZero;
    }
    if t.id == u128::MAX {
        return CreateTransferStatus::IdMustNotBeIntMax;
    }

    if let Some(e) = existing {
        return create_transfer_exists(t, e);
    }

    // Post/void pending transfers are handled separately.
    if t.flags.post_pending_transfer() || t.flags.void_pending_transfer() {
        return CreateTransferStatus::DeprecatedOk;
    }

    if t.debit_account_id == 0 {
        return CreateTransferStatus::DebitAccountIdMustNotBeZero;
    }
    if t.debit_account_id == u128::MAX {
        return CreateTransferStatus::DebitAccountIdMustNotBeIntMax;
    }
    if t.credit_account_id == 0 {
        return CreateTransferStatus::CreditAccountIdMustNotBeZero;
    }
    if t.credit_account_id == u128::MAX {
        return CreateTransferStatus::CreditAccountIdMustNotBeIntMax;
    }
    if t.credit_account_id == t.debit_account_id {
        return CreateTransferStatus::AccountsMustBeDifferent;
    }

    if t.pending_id != 0 {
        return CreateTransferStatus::PendingIdMustBeZero;
    }
    if !t.flags.pending() {
        if t.timeout != 0 {
            return CreateTransferStatus::TimeoutReservedForPendingTransfer;
        }
        if t.flags.closing_debit() || t.flags.closing_credit() {
            return CreateTransferStatus::ClosingTransferMustBePending;
        }
    }

    if t.ledger == 0 {
        return CreateTransferStatus::LedgerMustNotBeZero;
    }
    if t.code == 0 {
        return CreateTransferStatus::CodeMustNotBeZero;
    }

    assert_eq!(dr_account.id, t.debit_account_id);
    assert_eq!(cr_account.id, t.credit_account_id);

    if dr_account.ledger != cr_account.ledger {
        return CreateTransferStatus::AccountsMustHaveTheSameLedger;
    }
    if t.ledger != dr_account.ledger {
        return CreateTransferStatus::TransferMustHaveTheSameLedgerAsAccounts;
    }

    // Imported transfers keep their own (past) timestamp; the batch-level
    // regression/collision checks run in the orchestrator beforehand. The
    // postdate and timeout checks below must fire before the balance checks,
    // matching upstream state_machine.zig:3819-3828.
    let timestamp_actual = if t.flags.imported() {
        assert!(t.timestamp != 0);
        assert!(t.timestamp <= timestamp_event);
        if t.timestamp <= dr_account.timestamp {
            return CreateTransferStatus::ImportedEventTimestampMustPostdateDebitAccount;
        }
        if t.timestamp <= cr_account.timestamp {
            return CreateTransferStatus::ImportedEventTimestampMustPostdateCreditAccount;
        }
        if t.timeout != 0 {
            assert!(t.flags.pending());
            return CreateTransferStatus::ImportedEventTimeoutMustBeZero;
        }
        t.timestamp
    } else {
        assert_eq!(t.timestamp, 0);
        timestamp_event
    };

    assert!(timestamp_actual > dr_account.timestamp);
    assert!(timestamp_actual > cr_account.timestamp);

    if dr_account.flags.closed() {
        return CreateTransferStatus::DebitAccountAlreadyClosed;
    }
    if cr_account.flags.closed() {
        return CreateTransferStatus::CreditAccountAlreadyClosed;
    }

    let amount_actual = compute_amount_actual(t, dr_account, cr_account);

    // Overflow checks.
    if t.flags.pending() {
        if sum_overflows(amount_actual, dr_account.debits_pending) {
            return CreateTransferStatus::OverflowsDebitsPending;
        }
        if sum_overflows(amount_actual, cr_account.credits_pending) {
            return CreateTransferStatus::OverflowsCreditsPending;
        }
    }
    if sum_overflows(amount_actual, dr_account.debits_posted) {
        return CreateTransferStatus::OverflowsDebitsPosted;
    }
    if sum_overflows(amount_actual, cr_account.credits_posted) {
        return CreateTransferStatus::OverflowsCreditsPosted;
    }
    if sum_overflows(
        amount_actual,
        dr_account.debits_pending.wrapping_add(dr_account.debits_posted),
    ) {
        return CreateTransferStatus::OverflowsDebits;
    }
    if sum_overflows(
        amount_actual,
        cr_account.credits_pending.wrapping_add(cr_account.credits_posted),
    ) {
        return CreateTransferStatus::OverflowsCredits;
    }

    // Timeout overflow check (timeout expressed in nanoseconds must fit in u63).
    {
        let timeout_ns = (t.timeout as u64)
            .checked_mul(tigerbeetle_core::types::NS_PER_S)
            .expect("timeout * NS_PER_S overflow");
        if timeout_ns > i64::MAX as u64 {
            return CreateTransferStatus::OverflowsTimeout;
        }
        if timestamp_actual as u64 + timeout_ns > i64::MAX as u64 {
            return CreateTransferStatus::OverflowsTimeout;
        }
    }

    // Limit checks.
    if dr_account.debits_exceed_credits(amount_actual) {
        return CreateTransferStatus::ExceedsCredits;
    }
    if cr_account.credits_exceed_debits(amount_actual) {
        return CreateTransferStatus::ExceedsDebits;
    }

    CreateTransferStatus::Created
}

/// Outcome of validating a single transfer creation.
///
/// Mirrors upstream's tagged `CreateTransferResult` union (state_machine.zig:3703):
/// the `.created` payload carries the amount actually applied after balancing, and
/// every other tag carries zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CreateTransferOutcome {
    pub status: CreateTransferStatus,
    /// The amount applied to the account balances (`.created => |amount_actual|`).
    pub amount_actual: u128,
}

/// Validate a single transfer and report the amount that a successful creation
/// applies to the account balances.
///
/// Same inputs as [`create_transfer`]; production callers use this to obtain
/// `amount_actual` for the resulting balance mutation.
pub fn create_transfer_outcome(
    t: &Transfer,
    timestamp_event: u64,
    existing: Option<&Transfer>,
    dr_account: &Account,
    cr_account: &Account,
) -> CreateTransferOutcome {
    let status = create_transfer(t, timestamp_event, existing, dr_account, cr_account);
    let amount_actual = if status == CreateTransferStatus::Created {
        compute_amount_actual(t, dr_account, cr_account)
    } else {
        0
    };
    CreateTransferOutcome { status, amount_actual }
}

/// Balancing amount: cap amount to the available balance
/// (upstream `state_machine.zig:3795-3799`).
#[must_use]
fn compute_amount_actual(t: &Transfer, dr_account: &Account, cr_account: &Account) -> u128 {
    let mut amount_actual = t.amount;
    if t.flags.balancing_debit() {
        let dr_balance = dr_account.debits_posted.wrapping_add(dr_account.debits_pending);
        let available = dr_account.credits_posted.saturating_sub(dr_balance);
        amount_actual = amount_actual.min(available);
    }
    if t.flags.balancing_credit() {
        let cr_balance = cr_account.credits_posted.wrapping_add(cr_account.credits_pending);
        let available = cr_account.debits_posted.saturating_sub(cr_balance);
        amount_actual = amount_actual.min(available);
    }
    amount_actual
}

/// Compute the two account records after a created transfer's balance mutation
/// (upstream `commit_transfer`, `state_machine.zig:3913-3962`). Pure: both the
/// batch orchestrator's in-session working view and `StateMachine::commit_transfer`
/// derive the mutation from this single rule, so they stay in lockstep.
fn commit_transfer_accounts(
    event: &Transfer,
    amount_actual: u128,
    dr_account: &Account,
    cr_account: &Account,
) -> (Account, Account) {
    let mut dr = *dr_account;
    let mut cr = *cr_account;
    if event.flags.pending() {
        dr.debits_pending = dr.debits_pending.wrapping_add(amount_actual);
        cr.credits_pending = cr.credits_pending.wrapping_add(amount_actual);
    } else {
        dr.debits_posted = dr.debits_posted.wrapping_add(amount_actual);
        cr.credits_posted = cr.credits_posted.wrapping_add(amount_actual);
    }
    if event.flags.closing_debit() {
        dr.flags = dr.flags.with_closed();
    }
    if event.flags.closing_credit() {
        cr.flags = cr.flags.with_closed();
    }
    (dr, cr)
}

/// Compute the two account records after a posting/voiding transfer commits
/// (upstream `state_machine.zig:4240-4263`): the pending balance is released,
/// a post moves `amount_actual` to the posted balances, and a void reopens
/// accounts that the pending transfer's closing flags had closed.
fn commit_post_void_accounts(
    event: &Transfer,
    p: &Transfer,
    amount_actual: u128,
    dr_account: &Account,
    cr_account: &Account,
) -> (Account, Account) {
    let mut dr = *dr_account;
    let mut cr = *cr_account;
    dr.debits_pending = dr.debits_pending.wrapping_sub(p.amount);
    cr.credits_pending = cr.credits_pending.wrapping_sub(p.amount);

    if event.flags.post_pending_transfer() {
        assert!(!p.flags.closing_debit());
        assert!(!p.flags.closing_credit());
        assert!(amount_actual <= p.amount);
        dr.debits_posted = dr.debits_posted.wrapping_add(amount_actual);
        cr.credits_posted = cr.credits_posted.wrapping_add(amount_actual);
    } else {
        assert_eq!(amount_actual, p.amount);
        // Revert the closing account operation:
        if p.flags.closing_debit() {
            assert!(dr_account.flags.closed());
            dr.flags = dr.flags.without_closed();
        }
        if p.flags.closing_credit() {
            assert!(cr_account.flags.closed());
            cr.flags = cr.flags.without_closed();
        }
    }
    (dr, cr)
}

/// Idempotency check for an existing transfer.
fn create_transfer_exists(t: &Transfer, e: &Transfer) -> CreateTransferStatus {
    assert_eq!(t.id, e.id);

    if t.flags.as_raw() != e.flags.as_raw() {
        return CreateTransferStatus::ExistsWithDifferentFlags;
    }
    if t.debit_account_id != e.debit_account_id {
        return CreateTransferStatus::ExistsWithDifferentDebitAccountId;
    }
    if t.credit_account_id != e.credit_account_id {
        return CreateTransferStatus::ExistsWithDifferentCreditAccountId;
    }

    // For non-pending transfers, amount is compared exactly.
    // For pending transfers, the amount is an upper-bound.
    if t.flags.pending() {
        if t.amount > e.amount {
            return CreateTransferStatus::PendingTransferHasDifferentAmount;
        }
    } else if t.amount != e.amount {
        return CreateTransferStatus::ExistsWithDifferentAmount;
    }

    if t.pending_id != e.pending_id {
        return CreateTransferStatus::ExistsWithDifferentPendingId;
    }
    if t.user_data_128 != e.user_data_128 {
        return CreateTransferStatus::ExistsWithDifferentUserData128;
    }
    if t.user_data_64 != e.user_data_64 {
        return CreateTransferStatus::ExistsWithDifferentUserData64;
    }
    if t.user_data_32 != e.user_data_32 {
        return CreateTransferStatus::ExistsWithDifferentUserData32;
    }
    if t.timeout != e.timeout {
        return CreateTransferStatus::ExistsWithDifferentTimeout;
    }
    if t.code != e.code {
        return CreateTransferStatus::ExistsWithDifferentCode;
    }
    CreateTransferStatus::Exists
}

// ---------------------------------------------------------------------------
// Post / void pending transfers
// ---------------------------------------------------------------------------

/// Result of posting or voiding a pending transfer.
///
/// On `Created`, the caller must:
/// 1. Insert the new transfer into the transfers groove.
/// 2. Update the `TransferPending` status to `Posted` or `Voided`.
/// 3. Update the debit and credit accounts (decrement pending, increment posted for post).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostVoidPendingResult {
    pub status: CreateTransferStatus,
    pub amount_actual: u128,
    /// If `true`, this is a post (not a void).
    pub is_post: bool,
}

/// Post or void a pending transfer.
///
/// `t` is the posting/voiding transfer event.
/// `p` is the original pending transfer (looked up by `t.pending_id`).
/// `dr_account` and `cr_account` are the debit/credit accounts of the pending transfer.
///
/// Returns the result status and the amount to apply.
///
/// Upstream: `src/state_machine.zig:4053` (`post_or_void_pending_transfer`).
pub fn post_or_void_pending_transfer(
    t: &Transfer,
    timestamp_event: u64,
    p: &Transfer,
    dr_account: &Account,
    cr_account: &Account,
    pending_status: tigerbeetle_core::types::TransferPendingStatus,
) -> PostVoidPendingResult {
    use tigerbeetle_core::types::TransferPendingStatus;
    assert!(timestamp_event != 0);
    assert!(t.flags.post_pending_transfer() || t.flags.void_pending_transfer());

    // Mutually exclusive flags.
    if t.flags.post_pending_transfer() && t.flags.void_pending_transfer() {
        return PostVoidPendingResult {
            status: CreateTransferStatus::FlagsAreMutuallyExclusive,
            amount_actual: 0,
            is_post: false,
        };
    }
    if t.flags.pending()
        || t.flags.balancing_debit()
        || t.flags.balancing_credit()
        || t.flags.closing_debit()
        || t.flags.closing_credit()
    {
        return PostVoidPendingResult {
            status: CreateTransferStatus::FlagsAreMutuallyExclusive,
            amount_actual: 0,
            is_post: false,
        };
    }

    if t.pending_id == 0 {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingIdMustNotBeZero,
            amount_actual: 0,
            is_post: false,
        };
    }
    if t.pending_id == u128::MAX {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingIdMustNotBeIntMax,
            amount_actual: 0,
            is_post: false,
        };
    }
    if t.pending_id == t.id {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingIdMustBeDifferent,
            amount_actual: 0,
            is_post: false,
        };
    }
    if t.timeout != 0 {
        return PostVoidPendingResult {
            status: CreateTransferStatus::TimeoutReservedForPendingTransfer,
            amount_actual: 0,
            is_post: false,
        };
    }

    // p is the pending transfer (already looked up by caller).
    assert_eq!(p.id, t.pending_id);
    assert!(p.flags.pending());

    // Account IDs must match (if provided).
    if t.debit_account_id != 0 && t.debit_account_id != p.debit_account_id {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingTransferHasDifferentDebitAccountId,
            amount_actual: 0,
            is_post: false,
        };
    }
    if t.credit_account_id != 0 && t.credit_account_id != p.credit_account_id {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingTransferHasDifferentCreditAccountId,
            amount_actual: 0,
            is_post: false,
        };
    }

    // Ledger/code must match (if provided).
    if t.ledger != 0 && t.ledger != p.ledger {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingTransferHasDifferentLedger,
            amount_actual: 0,
            is_post: false,
        };
    }
    if t.code != 0 && t.code != p.code {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingTransferHasDifferentCode,
            amount_actual: 0,
            is_post: false,
        };
    }

    // Compute amount: void defaults to full pending amount; post defaults to full.
    let is_post = t.flags.post_pending_transfer();
    let amount_actual = if t.flags.void_pending_transfer() {
        if t.amount == 0 { p.amount } else { t.amount }
    } else {
        // post
        if t.amount == u128::MAX { p.amount } else { t.amount }
    };

    if amount_actual > p.amount {
        return PostVoidPendingResult {
            status: CreateTransferStatus::ExceedsPendingTransferAmount,
            amount_actual,
            is_post,
        };
    }

    // For void: amount must exactly match pending amount.
    if !is_post && amount_actual < p.amount {
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingTransferHasDifferentAmount,
            amount_actual,
            is_post,
        };
    }

    // Check pending transfer status (upstream state_machine.zig:4130-4143; the
    // expiry *time* check that follows runs after it).
    match pending_status {
        TransferPendingStatus::Pending => {}
        TransferPendingStatus::Posted => {
            return PostVoidPendingResult {
                status: CreateTransferStatus::PendingTransferAlreadyPosted,
                amount_actual,
                is_post,
            };
        }
        TransferPendingStatus::Voided => {
            return PostVoidPendingResult {
                status: CreateTransferStatus::PendingTransferAlreadyVoided,
                amount_actual,
                is_post,
            };
        }
        TransferPendingStatus::Expired => {
            return PostVoidPendingResult {
                status: CreateTransferStatus::PendingTransferExpired,
                amount_actual,
                is_post,
            };
        }
        TransferPendingStatus::None => unreachable!(),
    }

    // The pending transfer must not have expired by the event's timestamp
    // (upstream state_machine.zig:4145-4156). A pending that timed out but is
    // still `Pending` (expiry pumps are deferred) reports this too.
    if p.timeout != 0 && p.timestamp + p.timeout_ns() <= timestamp_event {
        assert!(!p.flags.imported());
        return PostVoidPendingResult {
            status: CreateTransferStatus::PendingTransferExpired,
            amount_actual,
            is_post,
        };
    }

    // Closed accounts: posting is rejected on a closed account; voiding is the
    // only movement allowed (upstream state_machine.zig:4185-4190).
    if dr_account.flags.closed() && is_post {
        return PostVoidPendingResult {
            status: CreateTransferStatus::DebitAccountAlreadyClosed,
            amount_actual,
            is_post,
        };
    }
    if cr_account.flags.closed() && is_post {
        return PostVoidPendingResult {
            status: CreateTransferStatus::CreditAccountAlreadyClosed,
            amount_actual,
            is_post,
        };
    }

    // After this point, the transfer must succeed.
    PostVoidPendingResult { status: CreateTransferStatus::Created, amount_actual, is_post }
}

/// Idempotency check for an existing posting/voiding transfer.
///
/// `t` is the resubmitted event, `e` the committed post/void transfer, and `p`
/// the pending transfer it references. The folder/folded relationship lets some
/// fields be zero in the event and fall back to the stored (`e`/`p`) values.
///
/// Upstream: `src/state_machine.zig:4301` (`post_or_void_pending_transfer_exists`).
fn post_or_void_pending_transfer_exists(
    t: &Transfer,
    e: &Transfer,
    p: &Transfer,
) -> CreateTransferStatus {
    assert_eq!(t.id, e.id);
    assert_ne!(t.id, p.id);
    assert!(t.flags.post_pending_transfer() || t.flags.void_pending_transfer());
    assert_eq!(t.flags.as_raw(), e.flags.as_raw());
    assert_eq!(t.pending_id, e.pending_id);
    assert_eq!(t.pending_id, p.id);
    assert!(p.flags.pending());
    assert_eq!(t.timeout, e.timeout);
    assert_eq!(t.timeout, 0);
    assert_eq!(e.debit_account_id, p.debit_account_id);
    assert_eq!(e.credit_account_id, p.credit_account_id);
    assert_eq!(e.ledger, p.ledger);
    assert_eq!(e.code, p.code);
    assert!(e.timestamp > p.timestamp);

    if t.debit_account_id != 0 && t.debit_account_id != e.debit_account_id {
        return CreateTransferStatus::ExistsWithDifferentDebitAccountId;
    }
    if t.credit_account_id != 0 && t.credit_account_id != e.credit_account_id {
        return CreateTransferStatus::ExistsWithDifferentCreditAccountId;
    }

    if t.flags.void_pending_transfer() {
        if t.amount == 0 {
            if e.amount != p.amount {
                return CreateTransferStatus::ExistsWithDifferentAmount;
            }
        } else if t.amount != e.amount {
            return CreateTransferStatus::ExistsWithDifferentAmount;
        }
    }
    if t.flags.post_pending_transfer() {
        assert!(e.amount <= p.amount);
        if t.amount == u128::MAX {
            if e.amount != p.amount {
                return CreateTransferStatus::ExistsWithDifferentAmount;
            }
        } else if t.amount != e.amount {
            return CreateTransferStatus::ExistsWithDifferentAmount;
        }
    }

    if t.user_data_128 == 0 {
        if e.user_data_128 != p.user_data_128 {
            return CreateTransferStatus::ExistsWithDifferentUserData128;
        }
    } else if t.user_data_128 != e.user_data_128 {
        return CreateTransferStatus::ExistsWithDifferentUserData128;
    }

    if t.user_data_64 == 0 {
        if e.user_data_64 != p.user_data_64 {
            return CreateTransferStatus::ExistsWithDifferentUserData64;
        }
    } else if t.user_data_64 != e.user_data_64 {
        return CreateTransferStatus::ExistsWithDifferentUserData64;
    }

    if t.user_data_32 == 0 {
        if e.user_data_32 != p.user_data_32 {
            return CreateTransferStatus::ExistsWithDifferentUserData32;
        }
    } else if t.user_data_32 != e.user_data_32 {
        return CreateTransferStatus::ExistsWithDifferentUserData32;
    }

    if t.ledger != 0 && t.ledger != e.ledger {
        return CreateTransferStatus::ExistsWithDifferentLedger;
    }
    if t.code != 0 && t.code != e.code {
        return CreateTransferStatus::ExistsWithDifferentCode;
    }

    CreateTransferStatus::Exists
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` if `a + b` would overflow.
const fn sum_overflows(a: u128, b: u128) -> bool {
    a.checked_add(b).is_none()
}

// ---------------------------------------------------------------------------
// Batch orchestrator — linked chains + imported validation
// ---------------------------------------------------------------------------

/// Result of executing one event in a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CreateAccountResult {
    pub status: CreateAccountStatus,
    pub timestamp: u64,
}

/// Result of executing one transfer event in a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CreateTransferResult {
    pub status: CreateTransferStatus,
    pub timestamp: u64,
    /// The amount applied to the account balances (zero unless `status` is `Created`).
    pub amount_actual: u128,
}

/// Execute a batch of `create_accounts` events with linked-chain support.
///
/// `timestamp` is the commit timestamp for the batch (one higher than the last committed).
/// The per-event timestamp is computed as `timestamp - events.len + index + 1`.
///
/// `accounts_key_max` is the maximum committed account timestamp (`accounts.objects.key_range.key_max`,
/// passes through the committed store; the orchestrator tracks the within-batch inserts itself).
/// `transfer_with_timestamp` resolves the transfer matching a timestamp, mirroring the cross-groove
/// `transfers.indirect_lookup` that upstream consults for imported events.
///
/// Returns a `Vec<CreateAccountResult>` parallel to the input events.
pub fn execute_create_accounts<F, FTransferAt>(
    events: &[Account],
    timestamp: u64,
    mut get_existing: F,
    accounts_key_max: u64,
    transfer_with_timestamp: FTransferAt,
) -> Vec<CreateAccountResult>
where
    F: FnMut(u128) -> Option<Account>,
    FTransferAt: Fn(u64) -> Option<u128>,
{
    let mut results: Vec<CreateAccountResult> = Vec::with_capacity(events.len());
    let mut chain: Option<usize> = None;
    let mut chain_broken = false;
    // The running object-tree key range: committed state plus the events of this
    // batch (upstream inserts each created account before the next event's check).
    let mut running_key_max = accounts_key_max;

    let batch_imported = !events.is_empty() && events[0].flags.imported();

    for (index, event) in events.iter().enumerate() {
        let timestamp_event = timestamp - events.len() as u64 + index as u64 + 1;

        let (status, ts) = 'result: {
            if event.flags.linked() {
                if chain.is_none() {
                    chain = Some(index);
                    assert!(!chain_broken);
                    // TODO(port): scope_open
                }
                if index == events.len() - 1 {
                    break 'result (CreateAccountStatus::LinkedEventChainOpen, timestamp_event);
                }
            }

            if chain_broken {
                break 'result (CreateAccountStatus::LinkedEventFailed, timestamp_event);
            }

            // Imported batch consistency check.
            if batch_imported != event.flags.imported() {
                if event.flags.imported() {
                    break 'result (CreateAccountStatus::ImportedEventNotExpected, timestamp_event);
                } else {
                    break 'result (CreateAccountStatus::ImportedEventExpected, timestamp_event);
                }
            }

            // Timestamp validation.
            if event.flags.imported() {
                if event.timestamp == 0 || event.timestamp >= timestamp {
                    break 'result (
                        CreateAccountStatus::ImportedEventTimestampMustNotAdvance,
                        timestamp_event,
                    );
                }
            } else if event.timestamp != 0 {
                break 'result (CreateAccountStatus::TimestampMustBeZero, timestamp_event);
            }

            let existing = get_existing(event.id);
            // Imported-timestamp regression/collision checks. Upstream runs these
            // inside `create_account` _after_ the idempotency checks (state_machine.zig:3656-3665),
            // so an existing record short-circuits to its `exists` code first.
            if event.flags.imported() && existing.is_none() {
                assert!(event.timestamp != 0);
                assert!(event.timestamp <= timestamp_event);
                // A past timestamp must not regress the account object index, and
                // must not collide with an existing transfer's timestamp.
                if event.timestamp <= running_key_max
                    || transfer_with_timestamp(event.timestamp).is_some()
                {
                    break 'result (
                        CreateAccountStatus::ImportedEventTimestampMustNotRegress,
                        timestamp_event,
                    );
                }
            }
            let status = create_account(event, timestamp_event, existing.as_ref());
            // Upstream `create_account` returns a tagged union whose payload is
            // the result timestamp (`.created` => the event's, `.exists` => the
            // existing record's); we recover it from the lookup. Imported events
            // keep their own timestamp; the object index absorbs it.
            let ts = match status {
                CreateAccountStatus::Created if event.flags.imported() => event.timestamp,
                CreateAccountStatus::Created => timestamp_event,
                CreateAccountStatus::Exists => match existing {
                    Some(existing) => existing.timestamp,
                    None => {
                        unreachable!("create_account returns Exists only for a matching record")
                    }
                },
                _ => timestamp_event,
            };
            if matches!(status, CreateAccountStatus::Created) {
                running_key_max = running_key_max.max(ts);
            }
            (status, ts)
        };

        // Chain error handling.
        if status != CreateAccountStatus::Created {
            if let Some(chain_start) = chain {
                if !chain_broken {
                    chain_broken = true;
                    // The chain has just been broken: mark every prior member as
                    // `linked_event_failed` (upstream state_machine.zig:3116-3145).
                    for result in &mut results[chain_start..index] {
                        result.status = CreateAccountStatus::LinkedEventFailed;
                    }
                }
            }
        }

        results.push(CreateAccountResult { status, timestamp: ts });

        // Chain completion.
        if chain.is_some()
            && (!event.flags.linked() || status == CreateAccountStatus::LinkedEventChainOpen)
        {
            if !chain_broken {
                // TODO(port): scope_close(persist)
            }
            chain = None;
            chain_broken = false;
        }
    }

    assert!(chain.is_none());
    assert!(!chain_broken);

    results
}

/// Execute a batch of `create_transfers` events with linked-chain support.
///
/// Same semantics as `execute_create_accounts` but for transfers.
///
/// `transfers_key_max` is the maximum committed transfer timestamp
/// (`transfers.objects.key_range.key_max`). `account_with_timestamp` resolves
/// the account matching a timestamp, mirroring the cross-groove
/// `accounts.indirect_lookup` that upstream consults for imported events.
///
/// `get_existing_transfer` looks up a transfer by id.
/// `get_account` looks up an account by id — needed for both debit and credit
/// accounts. It must return the committed state: this orchestrator overlays an
/// in-session working view on top, so events later in the batch observe the
/// balance mutations of earlier created transfers (upstream's in-session
/// scope), with each linked chain snapshotted at open and rolled back on
/// break.
///
/// `get_pending_status` returns the committed status (`transfers_pending`) of
/// the pending transfer with the given timestamp, which post/void events
/// validate against. A working overlay of in-session post/void status changes
/// sits on top, so a pending transfer posted earlier in the batch reports
/// `pending_transfer_already_posted` to a later event (a pending transfer
/// created and posted within the same batch is a documented `DEVIATION`:
/// `get_existing_transfer` does not see it, so the post reports
/// `pending_transfer_not_found`).
pub fn execute_create_transfers<F, G, FAccountAt, FPendingStatus>(
    events: &[Transfer],
    timestamp: u64,
    mut get_existing_transfer: F,
    mut get_account: G,
    transfers_key_max: u64,
    account_with_timestamp: FAccountAt,
    mut get_pending_status: FPendingStatus,
) -> Vec<CreateTransferResult>
where
    F: FnMut(u128) -> Option<Transfer>,
    G: FnMut(u128) -> Option<Account>,
    FAccountAt: Fn(u64) -> Option<u128>,
    FPendingStatus: FnMut(u64) -> TransferPendingStatus,
{
    let mut results: Vec<CreateTransferResult> = Vec::with_capacity(events.len());
    let mut chain: Option<usize> = None;
    let mut chain_broken = false;
    let mut running_key_max = transfers_key_max;
    // In-session working view of the accounts, layered over `get_account`
    // (upstream's open batch scope). Mutations from created events accumulate
    // here so later batch members validate against them.
    let mut working: HashMap<u128, Account> = HashMap::new();
    // Chain scoping: snapshot the working view (and the running key range) when
    // a chain opens, and restore them if it breaks (upstream `scope_close(.discard)`
    // rolls back only the chain's own writes — events since the chain opened).
    let mut chain_snapshot: Option<HashMap<u128, Account>> = None;
    let mut chain_key_max_snapshot: Option<u64> = None;
    // In-session view of pending-transfer statuses, layered over
    // `get_pending_status` (a post/void within the batch marks its pending as
    // posted/voided so a later event sees it).
    let mut working_pending: HashMap<u64, TransferPendingStatus> = HashMap::new();
    let mut chain_pending_snapshot: Option<HashMap<u64, TransferPendingStatus>> = None;

    let batch_imported = !events.is_empty() && events[0].flags.imported();

    for (index, event) in events.iter().enumerate() {
        let timestamp_event = timestamp - events.len() as u64 + index as u64 + 1;

        // `amount_actual` is only set when validation runs (upstream's `.created`
        // payload); short-circuited events report zero.
        let mut amount_actual = 0_u128;
        let (status, ts) = 'result: {
            if event.flags.linked() {
                if chain.is_none() {
                    chain = Some(index);
                    assert!(!chain_broken);
                    // TODO(port): scope_open — mirrored by the working-view
                    // snapshot/restore below.
                    chain_snapshot = Some(working.clone());
                    chain_key_max_snapshot = Some(running_key_max);
                    chain_pending_snapshot = Some(working_pending.clone());
                }
                if index == events.len() - 1 {
                    break 'result (CreateTransferStatus::LinkedEventChainOpen, timestamp_event);
                }
            }

            if chain_broken {
                break 'result (CreateTransferStatus::LinkedEventFailed, timestamp_event);
            }

            // Imported batch consistency check.
            if batch_imported != event.flags.imported() {
                if event.flags.imported() {
                    break 'result (
                        CreateTransferStatus::ImportedEventNotExpected,
                        timestamp_event,
                    );
                } else {
                    break 'result (CreateTransferStatus::ImportedEventExpected, timestamp_event);
                }
            }

            // Timestamp validation.
            if event.flags.imported() {
                if event.timestamp == 0 || event.timestamp >= timestamp {
                    break 'result (
                        CreateTransferStatus::ImportedEventTimestampMustNotAdvance,
                        timestamp_event,
                    );
                }
            } else if event.timestamp != 0 {
                break 'result (CreateTransferStatus::TimestampMustBeZero, timestamp_event);
            }

            // For post/void pending transfers, the transfer must not already exist.
            // For regular transfers, look up existing for idempotency.
            let existing = get_existing_transfer(event.id);

            let post_void =
                event.flags.post_pending_transfer() || event.flags.void_pending_transfer();

            let (status, ts) = if post_void {
                // Post/void transfers are validated by
                // `post_or_void_pending_transfer` (upstream state_machine.zig:4053).
                // The reserved-flags and id checks upstream runs for every transfer
                // before the post/void dispatch (state_machine.zig:3729-3732) apply
                // here too.
                if event.flags.has_padding() {
                    break 'result (CreateTransferStatus::ReservedFlag, timestamp_event);
                }
                if event.id == 0 {
                    break 'result (CreateTransferStatus::IdMustNotBeZero, timestamp_event);
                }
                if event.id == u128::MAX {
                    break 'result (CreateTransferStatus::IdMustNotBeIntMax, timestamp_event);
                }

                let pending = get_existing_transfer(event.pending_id);

                if let Some(e) = existing.as_ref() {
                    // Idempotency (`create_transfer_exists`,
                    // state_machine.zig:3988-4007): flags, pending_id and timeout
                    // must match the stored post/void; only then is the pending
                    // consulted (`.?`: a committed post/void implies a valid
                    // pending).
                    if event.flags.as_raw() != e.flags.as_raw() {
                        break 'result (
                            CreateTransferStatus::ExistsWithDifferentFlags,
                            timestamp_event,
                        );
                    }
                    if event.pending_id != e.pending_id {
                        break 'result (
                            CreateTransferStatus::ExistsWithDifferentPendingId,
                            timestamp_event,
                        );
                    }
                    if event.timeout != e.timeout {
                        break 'result (
                            CreateTransferStatus::ExistsWithDifferentTimeout,
                            timestamp_event,
                        );
                    }
                    let p = pending.as_ref().expect(
                        "a committed posting/voiding transfer implies its pending transfer",
                    );
                    let status = post_or_void_pending_transfer_exists(event, e, p);
                    // Like `.exists` for regular transfers: report the stored
                    // record's timestamp (upstream state_machine.zig:4423-4427).
                    let ts = if status == CreateTransferStatus::Exists {
                        e.timestamp
                    } else {
                        timestamp_event
                    };
                    (status, ts)
                } else {
                    let Some(p) = pending.as_ref() else {
                        break 'result (
                            CreateTransferStatus::PendingTransferNotFound,
                            timestamp_event,
                        );
                    };
                    if !p.flags.pending() {
                        break 'result (
                            CreateTransferStatus::PendingTransferNotPending,
                            timestamp_event,
                        );
                    }

                    // Imported-timestamp regression/collision checks. Upstream runs
                    // these after the pending-status checks
                    // (state_machine.zig:4158-4180); running them here makes regress
                    // take precedence over the flag/account-match checks below.
                    // `DEVIATION`: imported post/void events skip the postdate
                    // asserts (the pending's accounts make them moot).
                    if event.flags.imported() {
                        assert!(event.timestamp != 0);
                        assert!(event.timestamp <= timestamp_event);
                        if event.timestamp <= running_key_max
                            || account_with_timestamp(event.timestamp).is_some()
                        {
                            break 'result (
                                CreateTransferStatus::ImportedEventTimestampMustNotRegress,
                                timestamp_event,
                            );
                        }
                    }

                    // The pending transfer carries the account ids: a post/void
                    // event leaves them zero by convention.
                    let dr = working
                        .get(&p.debit_account_id)
                        .copied()
                        .or_else(|| get_account(p.debit_account_id))
                        .expect("post_or_void_pending_transfer: committed pending implies its debit account");
                    let cr = working
                        .get(&p.credit_account_id)
                        .copied()
                        .or_else(|| get_account(p.credit_account_id))
                        .expect("post_or_void_pending_transfer: committed pending implies its credit account");

                    let pending_status = working_pending
                        .get(&p.timestamp)
                        .copied()
                        .unwrap_or_else(|| get_pending_status(p.timestamp));
                    let result = post_or_void_pending_transfer(
                        event,
                        timestamp_event,
                        p,
                        &dr,
                        &cr,
                        pending_status,
                    );
                    let status = result.status;
                    amount_actual = result.amount_actual;
                    let ts = if status == CreateTransferStatus::Created {
                        // Apply the in-session side effects: the pending's status and
                        // accounts, mirroring what `persist` later re-derives from
                        // the committed state.
                        let is_post = event.flags.post_pending_transfer();
                        working_pending.insert(
                            p.timestamp,
                            if is_post {
                                TransferPendingStatus::Posted
                            } else {
                                TransferPendingStatus::Voided
                            },
                        );
                        let (dr_new, cr_new) =
                            commit_post_void_accounts(event, p, amount_actual, &dr, &cr);
                        debug_assert_eq!(dr_new.id, dr.id);
                        debug_assert_eq!(cr_new.id, cr.id);
                        working.insert(dr_new.id, dr_new);
                        working.insert(cr_new.id, cr_new);

                        if event.flags.imported() { event.timestamp } else { timestamp_event }
                    } else {
                        timestamp_event
                    };
                    (status, ts)
                }
            } else {
                // Look up debit/credit accounts: working (in-session) view first,
                // then the committed state.
                let dr = working
                    .get(&event.debit_account_id)
                    .copied()
                    .or_else(|| get_account(event.debit_account_id));
                let cr = working
                    .get(&event.credit_account_id)
                    .copied()
                    .or_else(|| get_account(event.credit_account_id));

                let (dr_account, cr_account) = match (dr, cr) {
                    (Some(d), Some(c)) => (d, c),
                    (None, _) => {
                        break 'result (
                            CreateTransferStatus::DebitAccountNotFound,
                            timestamp_event,
                        );
                    }
                    (_, None) => {
                        break 'result (
                            CreateTransferStatus::CreditAccountNotFound,
                            timestamp_event,
                        );
                    }
                };

                // Imported-timestamp regression/collision checks. Upstream runs them
                // inside `create_transfer` _after_ the idempotency checks
                // (state_machine.zig:3808-3817) and before the postdate checks, so an
                // existing record short-circuits to its `exists` code first and the
                // postdate ordering is preserved (error codes must take precedence in
                // the same order).
                if event.flags.imported() && existing.is_none() {
                    assert!(event.timestamp != 0);
                    assert!(event.timestamp <= timestamp_event);
                    // A past timestamp must not regress the transfer object index, and
                    // must not collide with an existing account's timestamp.
                    if event.timestamp <= running_key_max
                        || account_with_timestamp(event.timestamp).is_some()
                    {
                        break 'result (
                            CreateTransferStatus::ImportedEventTimestampMustNotRegress,
                            timestamp_event,
                        );
                    }
                }

                let outcome = create_transfer_outcome(
                    event,
                    timestamp_event,
                    existing.as_ref(),
                    &dr_account,
                    &cr_account,
                );
                let status = outcome.status;
                if status == CreateTransferStatus::Created {
                    amount_actual = outcome.amount_actual;
                    // Apply the mutation to the in-session view so later batch
                    // members validate against it; `persist_transfers` later derives
                    // the same values from the committed state in event order.
                    let (dr_new, cr_new) =
                        commit_transfer_accounts(event, amount_actual, &dr_account, &cr_account);
                    debug_assert_eq!(dr_new.id, dr_account.id);
                    debug_assert_eq!(cr_new.id, cr_account.id);
                    working.insert(dr_new.id, dr_new);
                    working.insert(cr_new.id, cr_new);
                } else {
                    amount_actual = 0;
                }
                // Same rule as accounts: `.exists` reports the existing record's
                // timestamp (upstream `state_machine.zig:3669-3721`).
                let ts = match status {
                    CreateTransferStatus::Created if event.flags.imported() => event.timestamp,
                    CreateTransferStatus::Created => timestamp_event,
                    CreateTransferStatus::Exists => match existing {
                        Some(existing) => existing.timestamp,
                        None => {
                            unreachable!(
                                "create_transfer returns Exists only for a matching record"
                            )
                        }
                    },
                    _ => timestamp_event,
                };
                (status, ts)
            };
            if matches!(status, CreateTransferStatus::Created) {
                running_key_max = running_key_max.max(ts);
            }
            (status, ts)
        };

        // Chain error handling.
        if status != CreateTransferStatus::Created {
            if let Some(chain_start) = chain {
                if !chain_broken {
                    chain_broken = true;
                    // TODO(port): scope_close(.discard) — roll back this chain's
                    // in-session writes (only the events since it opened).
                    if let (Some(snapshot), Some(key_max_snapshot), Some(pending_snapshot)) = (
                        chain_snapshot.as_ref(),
                        chain_key_max_snapshot.as_ref(),
                        chain_pending_snapshot.as_ref(),
                    ) {
                        working.clone_from(snapshot);
                        running_key_max = *key_max_snapshot;
                        working_pending.clone_from(pending_snapshot);
                    }
                    for result in &mut results[chain_start..index] {
                        result.status = CreateTransferStatus::LinkedEventFailed;
                    }
                }
            }
        }

        results.push(CreateTransferResult { status, timestamp: ts, amount_actual });

        // Chain completion.
        if chain.is_some()
            && (!event.flags.linked() || status == CreateTransferStatus::LinkedEventChainOpen)
        {
            if !chain_broken {
                // TODO(port): scope_close(.persist)
            }
            chain = None;
            chain_broken = false;
            chain_snapshot = None;
            chain_key_max_snapshot = None;
            chain_pending_snapshot = None;
        }
    }

    assert!(chain.is_none());
    assert!(!chain_broken);

    results
}

// ---------------------------------------------------------------------------
// Request/result body codecs
// ---------------------------------------------------------------------------
//
// The wire bodies are arrays of `#[repr(C)]` LE records — 128 bytes per
// account/transfer event, 16 bytes per result. Layout is pinned by the `size_of`
// asserts in `tigerbeetle-core/src/types.rs`.

fn le_u128(bytes: &[u8], offset: usize) -> u128 {
    let mut value = [0_u8; 16];
    value.copy_from_slice(&bytes[offset..offset + 16]);
    u128::from_le_bytes(value)
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut value = [0_u8; 8];
    value.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(value)
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0_u8; 4];
    value.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(value)
}

fn le_u16(bytes: &[u8], offset: usize) -> u16 {
    let mut value = [0_u8; 2];
    value.copy_from_slice(&bytes[offset..offset + 2]);
    u16::from_le_bytes(value)
}

/// Encode a batch of [`Account`]s as a request body (128 bytes each, LE).
#[must_use]
pub fn account_batch_to_bytes(events: &[Account]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(events.len() * 128);
    for event in events {
        let mut record = [0_u8; 128];
        record[0..16].copy_from_slice(&event.id.to_le_bytes());
        record[16..32].copy_from_slice(&event.debits_pending.to_le_bytes());
        record[32..48].copy_from_slice(&event.debits_posted.to_le_bytes());
        record[48..64].copy_from_slice(&event.credits_pending.to_le_bytes());
        record[64..80].copy_from_slice(&event.credits_posted.to_le_bytes());
        record[80..96].copy_from_slice(&event.user_data_128.to_le_bytes());
        record[96..104].copy_from_slice(&event.user_data_64.to_le_bytes());
        record[104..108].copy_from_slice(&event.user_data_32.to_le_bytes());
        record[108..112].copy_from_slice(&event.reserved.to_le_bytes());
        record[112..116].copy_from_slice(&event.ledger.to_le_bytes());
        record[116..118].copy_from_slice(&event.code.to_le_bytes());
        record[118..120].copy_from_slice(&event.flags.as_raw().to_le_bytes());
        record[120..128].copy_from_slice(&event.timestamp.to_le_bytes());
        bytes.extend_from_slice(&record);
    }
    bytes
}

/// Decode a request body into a batch of [`Account`]s.
///
/// Returns `None` if the body length is not a whole number of records.
#[must_use]
pub fn bytes_to_account_batch(bytes: &[u8]) -> Option<Vec<Account>> {
    if !bytes.len().is_multiple_of(128) {
        return None;
    }
    let (records, remainder) = bytes.as_chunks::<128>();
    assert!(remainder.is_empty());
    let mut events = Vec::with_capacity(records.len());
    for record in records {
        events.push(Account {
            id: le_u128(record, 0),
            debits_pending: le_u128(record, 16),
            debits_posted: le_u128(record, 32),
            credits_pending: le_u128(record, 48),
            credits_posted: le_u128(record, 64),
            user_data_128: le_u128(record, 80),
            user_data_64: le_u64(record, 96),
            user_data_32: le_u32(record, 104),
            reserved: le_u32(record, 108),
            ledger: le_u32(record, 112),
            code: le_u16(record, 116),
            flags: AccountFlags::from_raw(le_u16(record, 118)),
            timestamp: le_u64(record, 120),
        });
    }
    Some(events)
}

/// Encode a batch of [`Transfer`]s as a request body (128 bytes each, LE).
#[must_use]
pub fn transfer_batch_to_bytes(events: &[Transfer]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(events.len() * 128);
    for event in events {
        let mut record = [0_u8; 128];
        record[0..16].copy_from_slice(&event.id.to_le_bytes());
        record[16..32].copy_from_slice(&event.debit_account_id.to_le_bytes());
        record[32..48].copy_from_slice(&event.credit_account_id.to_le_bytes());
        record[48..64].copy_from_slice(&event.amount.to_le_bytes());
        record[64..80].copy_from_slice(&event.pending_id.to_le_bytes());
        record[80..96].copy_from_slice(&event.user_data_128.to_le_bytes());
        record[96..104].copy_from_slice(&event.user_data_64.to_le_bytes());
        record[104..108].copy_from_slice(&event.user_data_32.to_le_bytes());
        record[108..112].copy_from_slice(&event.timeout.to_le_bytes());
        record[112..116].copy_from_slice(&event.ledger.to_le_bytes());
        record[116..118].copy_from_slice(&event.code.to_le_bytes());
        record[118..120].copy_from_slice(&event.flags.as_raw().to_le_bytes());
        record[120..128].copy_from_slice(&event.timestamp.to_le_bytes());
        bytes.extend_from_slice(&record);
    }
    bytes
}

/// Decode a request body into a batch of [`Transfer`]s.
///
/// Returns `None` if the body length is not a whole number of records.
#[must_use]
pub fn bytes_to_transfer_batch(bytes: &[u8]) -> Option<Vec<Transfer>> {
    if !bytes.len().is_multiple_of(128) {
        return None;
    }
    let (records, remainder) = bytes.as_chunks::<128>();
    assert!(remainder.is_empty());
    let mut events = Vec::with_capacity(records.len());
    for record in records {
        events.push(Transfer {
            id: le_u128(record, 0),
            debit_account_id: le_u128(record, 16),
            credit_account_id: le_u128(record, 32),
            amount: le_u128(record, 48),
            pending_id: le_u128(record, 64),
            user_data_128: le_u128(record, 80),
            user_data_64: le_u64(record, 96),
            user_data_32: le_u32(record, 104),
            timeout: le_u32(record, 108),
            ledger: le_u32(record, 112),
            code: le_u16(record, 116),
            flags: TransferFlags::from_raw(le_u16(record, 118)),
            timestamp: le_u64(record, 120),
        });
    }
    Some(events)
}

/// Encode a batch of [`CreateAccountResult`]s as a reply body: one 16-byte
/// record per event — `timestamp` LE, `status` LE (a `u32`), `reserved`
/// zeroed. Upstream `CreateAccountResult` (`tigerbeetle.zig:471`).
#[must_use]
pub fn account_results_to_bytes(results: &[CreateAccountResult]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(results.len() * 16);
    for result in results {
        let mut record = [0_u8; 16];
        record[0..8].copy_from_slice(&result.timestamp.to_le_bytes());
        record[8..12].copy_from_slice(&(result.status as u32).to_le_bytes());
        bytes.extend_from_slice(&record);
    }
    bytes
}

/// Encode a batch of [`CreateTransferResult`]s as a reply body (see
/// [`account_results_to_bytes`] for the layout).
#[must_use]
pub fn transfer_results_to_bytes(results: &[CreateTransferResult]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(results.len() * 16);
    for result in results {
        let mut record = [0_u8; 16];
        record[0..8].copy_from_slice(&result.timestamp.to_le_bytes());
        record[8..12].copy_from_slice(&(result.status as u32).to_le_bytes());
        bytes.extend_from_slice(&record);
    }
    bytes
}

/// The accounting state machine as mounted on a replica.
///
/// DEVIATION: upstream (`src/state_machine.zig`) owns the account and transfer
/// grooves (their trees and caches) and executes operations via
/// `state_machine.commit`. The forest is deferred, so the stores below are
/// plain id-keyed maps and persistence replays the chain scope decisions from
/// the orchestrator results instead of rolling grooves back.
///
/// Upstream: `src/state_machine.zig`.
#[derive(Debug, Default)]
pub struct StateMachine {
    /// The timestamp of the last op committed to the state machine; every
    /// executed op must advance it strictly (upstream asserts the same in
    /// `Replica.execute_op`, replica.zig:5441).
    pub commit_timestamp: u64,
    /// Temporary primary-key store for accounts.
    accounts: HashMap<u128, Account>,
    /// Temporary primary-key store for transfers.
    transfers: HashMap<u128, Transfer>,
    /// Maximum committed account timestamp — mirrors `accounts.objects.key_range.key_max`.
    accounts_timestamp_max: u64,
    /// Maximum committed transfer timestamp — mirrors `transfers.objects.key_range.key_max`.
    transfers_timestamp_max: u64,
    /// Timestamp → account id for committed accounts, mirroring the object
    /// tree's timestamp index (queryable upstream via `accounts.indirect_lookup`).
    accounts_by_timestamp: HashMap<u64, u128>,
    /// Timestamp → transfer id for committed transfers.
    transfers_by_timestamp: HashMap<u64, u128>,
    /// Pending transfer index: timestamp → status, written at pending creation
    /// and updated when the pending is posted/voided. Mirrors upstream's
    /// `transfers_pending.objects` groove.
    /// TODO(port): expiry (`expire_pending_transfers.pulse_next_timestamp`) is
    /// deferred with the grooves.
    transfers_pending: HashMap<u64, TransferPending>,
}

impl StateMachine {
    /// Record that an op was executed against the state machine.
    ///
    /// DEVIATION: upstream threads the operation and its body through
    /// `state_machine.commit`, which executes and returns the reply body size;
    /// sans-IO the replica executes nothing yet, so this (called from
    /// `Replica::commit_execute`) only guards and advances the timestamp.
    ///
    /// # Panics
    /// Panics unless `timestamp` strictly advances `commit_timestamp`.
    pub fn execute_op(&mut self, timestamp: u64) {
        assert!(self.commit_timestamp < timestamp);
        self.commit_timestamp = timestamp;
    }

    /// Execute a banking operation's request body and return its reply body.
    ///
    /// `timestamp` is the batch's last event timestamp; the caller advances
    /// [`execute_op`](Self::execute_op) first, so this only sanity-checks the
    /// clock (upstream guards the same way in `Replica.execute_op`).
    ///
    /// # Panics
    /// Panics unless `operation` is a state-machine operation this port
    /// executes, or the body does not decode to a whole number of events.
    #[must_use]
    pub fn execute(&mut self, operation: Operation, timestamp: u64, body: &[u8]) -> Vec<u8> {
        match operation {
            Operation::CREATE_ACCOUNTS => {
                let Some(events) = bytes_to_account_batch(body) else {
                    unreachable!("create_accounts body must encode whole Account events");
                };
                self.create_accounts(&events, timestamp)
            }
            Operation::CREATE_TRANSFERS => {
                let Some(events) = bytes_to_transfer_batch(body) else {
                    unreachable!("create_transfers body must encode whole Transfer events");
                };
                self.create_transfers(&events, timestamp)
            }
            operation => unreachable!("unsupported state-machine operation: {operation:?}"),
        }
    }

    /// Execute a `create_accounts` batch and return the reply body.
    #[must_use]
    pub fn create_accounts(&mut self, events: &[Account], timestamp: u64) -> Vec<u8> {
        assert!(self.commit_timestamp <= timestamp);
        let results = execute_create_accounts(
            events,
            timestamp,
            |id| self.accounts.get(&id).copied(),
            self.accounts_timestamp_max,
            |ts| self.transfers_by_timestamp.get(&ts).copied(),
        );
        self.persist_accounts(events, &results, timestamp);
        account_results_to_bytes(&results)
    }

    /// Execute a `create_transfers` batch and return the reply body.
    ///
    /// Created transfers apply their balance mutation to the debit/credit
    /// accounts at persist time, and are stored with the amount actually
    /// applied (after balancing). Within a batch, the orchestrator maintains an
    /// in-session working view of the accounts, so later events validate
    /// against earlier events' mutations; each linked chain is rolled back as a
    /// unit if it breaks.
    ///
    /// DEVIATION: `AccountEvent` history is deferred with the grooves.
    #[must_use]
    pub fn create_transfers(&mut self, events: &[Transfer], timestamp: u64) -> Vec<u8> {
        assert!(self.commit_timestamp <= timestamp);
        let results = execute_create_transfers(
            events,
            timestamp,
            |id| self.transfers.get(&id).copied(),
            |id| self.accounts.get(&id).copied(),
            self.transfers_timestamp_max,
            |ts| self.accounts_by_timestamp.get(&ts).copied(),
            |ts| {
                self.transfers_pending
                    .get(&ts)
                    .copied()
                    .map_or(TransferPendingStatus::None, |p| p.status)
            },
        );
        self.persist_transfers(events, &results, timestamp);
        transfer_results_to_bytes(&results)
    }

    /// The implied timestamp of the event at `index` (upstream
    /// `timestamp - events.len + index + 1`, state_machine.zig:3031).
    fn timestamp_event(timestamp: u64, len: usize, index: usize) -> u64 {
        timestamp - len as u64 + index as u64 + 1
    }

    /// Persist the accounts that upstream would write under the chain scope.
    ///
    /// A chain persists only if it closes and every member is `Created` (any
    /// other status — including `Exists` — breaks the chain and rolls its
    /// preceding members back, upstream `scope_close(.discard)`). Records are
    /// stamped with their event time before insertion (upstream bumps
    /// `event.timestamp = timestamp_event` before `groove.put`); imported
    /// events keep the timestamp the orchestrator validated.
    fn persist_accounts(
        &mut self,
        events: &[Account],
        results: &[CreateAccountResult],
        timestamp: u64,
    ) {
        let mut index = 0;
        while index < events.len() {
            if !events[index].flags.linked() {
                if results[index].status == CreateAccountStatus::Created {
                    let mut event = events[index];
                    if !event.flags.imported() {
                        event.timestamp = Self::timestamp_event(timestamp, events.len(), index);
                    }
                    assert_eq!(event.timestamp, results[index].timestamp);
                    self.insert_account(event.id, event);
                }
                index += 1;
                continue;
            }
            let start = index;
            while index < events.len() && events[index].flags.linked() {
                index += 1;
            }
            if index == events.len() {
                // Trailing unclosed chain: upstream discards it.
                continue;
            }
            let intact =
                results[start..=index].iter().all(|r| r.status == CreateAccountStatus::Created);
            if intact {
                for (offset, event) in events[start..=index].iter().enumerate() {
                    let mut event = *event;
                    if !event.flags.imported() {
                        event.timestamp =
                            Self::timestamp_event(timestamp, events.len(), start + offset);
                    }
                    assert_eq!(event.timestamp, results[start + offset].timestamp);
                    self.insert_account(event.id, event);
                }
            }
            index += 1;
        }
    }

    /// Insert a committed account, maintaining the timestamp index that
    /// mirrors the object tree (upstream `accounts.objects.key_range` and
    /// `accounts.indirect_lookup`).
    fn insert_account(&mut self, id: u128, account: Account) {
        assert!(account.timestamp != 0);
        if let Some(previous) = self.accounts_by_timestamp.insert(account.timestamp, id) {
            assert_eq!(previous, id, "two accounts cannot share a timestamp");
        }
        self.accounts_timestamp_max = self.accounts_timestamp_max.max(account.timestamp);
        self.accounts.insert(id, account);
    }

    /// Persist the transfers that upstream would write under the chain scope,
    /// applying each created transfer's balance mutation to its debit and
    /// credit accounts (upstream `commit_transfer`,
    /// `state_machine.zig:3913-3962`).
    ///
    /// See [`Self::persist_accounts`] for the chain semantics. Because the walk
    /// follows event order, mutations accumulate across the batch — so within an
    /// intact chain, later events operate on accounts already updated by earlier
    /// members, matching upstream's in-scope application.
    fn persist_transfers(
        &mut self,
        events: &[Transfer],
        results: &[CreateTransferResult],
        timestamp: u64,
    ) {
        let mut index = 0;
        while index < events.len() {
            if !events[index].flags.linked() {
                if results[index].status == CreateTransferStatus::Created {
                    let mut event = events[index];
                    if !event.flags.imported() {
                        event.timestamp = Self::timestamp_event(timestamp, events.len(), index);
                    }
                    assert_eq!(event.timestamp, results[index].timestamp);
                    self.persist_transfer(event, results[index].amount_actual);
                }
                index += 1;
                continue;
            }
            let start = index;
            while index < events.len() && events[index].flags.linked() {
                index += 1;
            }
            if index == events.len() {
                continue;
            }
            let intact =
                results[start..=index].iter().all(|r| r.status == CreateTransferStatus::Created);
            if intact {
                for (offset, event) in events[start..=index].iter().enumerate() {
                    let mut event = *event;
                    if !event.flags.imported() {
                        event.timestamp =
                            Self::timestamp_event(timestamp, events.len(), start + offset);
                    }
                    assert_eq!(event.timestamp, results[start + offset].timestamp);
                    self.persist_transfer(event, results[start + offset].amount_actual);
                }
            }
            index += 1;
        }
    }

    /// Store a created transfer (with its validated amount) and apply the
    /// corresponding balance mutation to the debit/credit accounts.
    ///
    /// Pending transfers additionally record the `transfers_pending` status
    /// index entry (status `Pending`). Posting/voiding a pending transfer
    /// routes to [`Self::persist_post_void`] instead — it folds the pending's
    /// fields into the stored record and applies the release/reopen mutation.
    fn persist_transfer(&mut self, mut event: Transfer, amount_actual: u128) {
        event.amount = amount_actual;
        self.insert_transfer(event.id, event);

        if event.flags.pending() {
            // Upstream writes the pending status on pending creation
            // (state_machine.zig:3963-3982).
            let transfer_pending = TransferPending {
                timestamp: event.timestamp,
                status: TransferPendingStatus::Pending,
                padding: [0; 7],
            };
            assert!(
                self.transfers_pending.insert(event.timestamp, transfer_pending).is_none(),
                "each pending transfer has a unique timestamp"
            );
            self.commit_transfer(&event, amount_actual);
        } else if event.flags.post_pending_transfer() || event.flags.void_pending_transfer() {
            self.persist_post_void(event, amount_actual);
        } else {
            self.commit_transfer(&event, amount_actual);
        }
    }

    /// Store a posting/voiding transfer and apply its balance mutation
    /// (upstream `commit_transfer` for post/void, `state_machine.zig:4240-4298`).
    ///
    /// The stored record folds the pending transfer's debit/credit accounts,
    /// ledger, code and user-data fallbacks; the pending status index advances
    /// to `Posted`/`Voided`; and the accounts release the pending holds
    /// ([`commit_post_void_accounts`]).
    fn persist_post_void(&mut self, mut event: Transfer, amount_actual: u128) {
        let p =
            self.transfers.get(&event.pending_id).copied().expect(
                "posting/voiding a pending transfer implies the pending transfer is committed",
            );
        event.debit_account_id = p.debit_account_id;
        event.credit_account_id = p.credit_account_id;
        event.ledger = p.ledger;
        event.code = p.code;
        event.timeout = 0;
        event.amount = amount_actual;
        if event.user_data_128 == 0 {
            event.user_data_128 = p.user_data_128;
        }
        if event.user_data_64 == 0 {
            event.user_data_64 = p.user_data_64;
        }
        if event.user_data_32 == 0 {
            event.user_data_32 = p.user_data_32;
        }
        event.pending_id = p.id;
        self.insert_transfer(event.id, event);

        let transfer_pending = self
            .transfers_pending
            .get_mut(&p.timestamp)
            .expect("posting/voiding a pending transfer implies its pending status row");
        assert_eq!(transfer_pending.status, TransferPendingStatus::Pending);
        transfer_pending.status = if event.flags.post_pending_transfer() {
            TransferPendingStatus::Posted
        } else {
            TransferPendingStatus::Voided
        };

        let dr = self
            .accounts
            .get(&event.debit_account_id)
            .copied()
            .expect("debit account exists: post_or_void_pending_transfer validated it");
        let cr = self
            .accounts
            .get(&event.credit_account_id)
            .copied()
            .expect("credit account exists: post_or_void_pending_transfer validated it");
        let (dr, cr) = commit_post_void_accounts(&event, &p, amount_actual, &dr, &cr);
        self.accounts.insert(dr.id, dr);
        self.accounts.insert(cr.id, cr);
    }

    /// Apply a created transfer's balance mutation to its accounts
    /// (upstream `commit_transfer`, `state_machine.zig:3913-3962`):
    /// pending transfers raise `debits_pending`/`credits_pending`, posted
    /// transfers raise the `*_posted` balances, and `closing_debit` /
    /// `closing_credit` set the account `CLOSED` flag.
    ///
    /// The accounts are guaranteed to exist (validated by `create_transfer`)
    /// and, at commit time, to not be closed — upstream's in-session scope
    /// rejects any transfer referencing an account closed earlier in the batch
    /// (see [`Self::create_transfers`]).
    ///
    /// TODO(port): `AccountEvent` CDC and expiry
    /// (`expire_pending_transfers.pulse_next_timestamp`) are deferred.
    fn commit_transfer(&mut self, event: &Transfer, amount_actual: u128) {
        let dr =
            self.accounts.get(&event.debit_account_id).copied().expect(
                "debit account exists: create_transfer validated and overflow checks passed",
            );
        let cr =
            self.accounts.get(&event.credit_account_id).copied().expect(
                "credit account exists: create_transfer validated and overflow checks passed",
            );

        let (dr, cr) = commit_transfer_accounts(event, amount_actual, &dr, &cr);
        self.accounts.insert(dr.id, dr);
        self.accounts.insert(cr.id, cr);
    }

    /// Insert a committed transfer, maintaining the timestamp index that
    /// mirrors the object tree (upstream `transfers.objects.key_range` and
    /// `transfers.indirect_lookup`).
    fn insert_transfer(&mut self, id: u128, transfer: Transfer) {
        assert!(transfer.timestamp != 0);
        if let Some(previous) = self.transfers_by_timestamp.insert(transfer.timestamp, id) {
            assert_eq!(previous, id, "two transfers cannot share a timestamp");
        }
        self.transfers_timestamp_max = self.transfers_timestamp_max.max(transfer.timestamp);
        self.transfers.insert(id, transfer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── StateMachine tests ───────────────────────────────────────────────

    #[test]
    fn state_machine_tracks_monotonic_commit_timestamp() {
        let mut state_machine = StateMachine::default();
        assert_eq!(state_machine.commit_timestamp, 0);
        state_machine.execute_op(1);
        assert_eq!(state_machine.commit_timestamp, 1);
        state_machine.execute_op(2);
        assert_eq!(state_machine.commit_timestamp, 2);
    }

    #[test]
    #[should_panic(expected = "commit_timestamp < timestamp")]
    fn state_machine_rejects_stale_commit_timestamp() {
        let mut state_machine = StateMachine::default();
        state_machine.execute_op(2);
        state_machine.execute_op(2);
    }

    // ── execute (batch bodies) tests ─────────────────────────────────────

    #[test]
    fn execute_accounts_writes_16_byte_results() {
        let mut sm = StateMachine::default();
        let events = [Account { id: 1, ledger: 1, code: 1, ..Account::default() }];
        let body = sm.execute(Operation::CREATE_ACCOUNTS, 10, &account_batch_to_bytes(&events));
        assert_eq!(body.len(), 16);
        // timestamp LE, status LE, reserved zeroed.
        assert_eq!(u64::from_le_bytes(body[0..8].try_into().unwrap_or([0; 8])), 10);
        assert_eq!(u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])), u32::MAX);
        assert_eq!(&body[12..16], &[0; 4]);
        assert_eq!(sm.accounts.len(), 1);
    }

    #[test]
    fn execute_accounts_persists_created_and_reports_exists_with_original_timestamp() {
        let mut sm = StateMachine::default();
        let events = [Account { id: 1, ledger: 1, code: 1, ..Account::default() }];
        let first = sm.create_accounts(&events, 10);
        assert_eq!(first.len(), 16);
        assert_eq!(u64::from_le_bytes(first[0..8].try_into().unwrap_or([0; 8])), 10);
        assert_eq!(u32::from_le_bytes(first[8..12].try_into().unwrap_or([0; 4])), u32::MAX);
        assert_eq!(sm.accounts.len(), 1);

        let second = sm.create_accounts(&events, 20);
        // The duplicate reports `Exists` carrying the original record's timestamp.
        assert_eq!(u64::from_le_bytes(second[0..8].try_into().unwrap_or([0; 8])), 10);
        assert_eq!(
            u32::from_le_bytes(second[8..12].try_into().unwrap_or([0; 4])),
            CreateAccountStatus::Exists as u32
        );
        assert_eq!(sm.accounts.len(), 1);
    }

    #[test]
    fn execute_accounts_rolls_back_broken_linked_chain() {
        let mut sm = StateMachine::default();
        let events = vec![
            Account {
                id: 1,
                ledger: 1,
                code: 1,
                flags: AccountFlags::LINKED,
                ..Account::default()
            },
            Account { id: 2, ledger: 0, code: 1, ..Account::default() },
        ];
        let body = sm.create_accounts(&events, 10);
        assert_eq!(body.len(), 32);
        assert_eq!(
            u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])),
            CreateAccountStatus::LinkedEventFailed as u32
        );
        assert_eq!(
            u32::from_le_bytes(body[24..28].try_into().unwrap_or([0; 4])),
            CreateAccountStatus::LedgerMustNotBeZero as u32
        );
        assert!(sm.accounts.is_empty());
    }

    #[test]
    fn execute_accounts_discards_unclosed_trailing_chain() {
        let mut sm = StateMachine::default();
        let events = vec![
            Account {
                id: 1,
                ledger: 1,
                code: 1,
                flags: AccountFlags::LINKED,
                ..Account::default()
            },
            Account {
                id: 2,
                ledger: 1,
                code: 1,
                flags: AccountFlags::LINKED,
                ..Account::default()
            },
        ];
        let body = sm.create_accounts(&events, 10);
        assert_eq!(body.len(), 32);
        // The trailing unclosed chain breaks: its members are rolled back.
        assert_eq!(
            u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])),
            CreateAccountStatus::LinkedEventFailed as u32
        );
        assert_eq!(
            u32::from_le_bytes(body[24..28].try_into().unwrap_or([0; 4])),
            CreateAccountStatus::LinkedEventChainOpen as u32
        );
        assert!(sm.accounts.is_empty());
    }

    #[test]
    fn execute_accounts_persists_closed_chain() {
        let mut sm = StateMachine::default();
        let events = vec![
            Account {
                id: 1,
                ledger: 1,
                code: 1,
                flags: AccountFlags::LINKED,
                ..Account::default()
            },
            Account { id: 2, ledger: 1, code: 1, ..Account::default() },
        ];
        let body = sm.create_accounts(&events, 10);
        assert_eq!(body.len(), 32);
        assert_eq!(
            u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])),
            CreateAccountStatus::Created as u32
        );
        assert_eq!(
            u32::from_le_bytes(body[24..28].try_into().unwrap_or([0; 4])),
            CreateAccountStatus::Created as u32
        );
        assert_eq!(body[28..32], [0; 4]);
        assert_eq!(sm.accounts.len(), 2);
        assert!(sm.accounts.contains_key(&1));
        assert!(sm.accounts.contains_key(&2));
    }

    #[test]
    fn execute_transfers_require_existing_accounts_and_persist() {
        let mut sm = StateMachine::default();
        let account = |id| Account { id, ledger: 1, code: 1, ..Account::default() };
        let transfers = vec![Transfer {
            id: 100,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        }];
        // Neither account exists yet: the debit is looked up first.
        let body = sm.create_transfers(&transfers, 10);
        assert_eq!(
            u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])),
            CreateTransferStatus::DebitAccountNotFound as u32
        );
        assert!(sm.transfers.is_empty());

        let _ = sm.create_accounts(&[account(1), account(2)], 20);
        let body = sm.create_transfers(&transfers, 30);
        assert_eq!(body.len(), 16);
        assert_eq!(u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])), u32::MAX);
        assert_eq!(sm.transfers.len(), 1);

        // Duplicate: Exists carries the original timestamp.
        let body = sm.create_transfers(&transfers, 40);
        assert_eq!(u64::from_le_bytes(body[0..8].try_into().unwrap_or([0; 8])), 30);
        assert_eq!(sm.transfers.len(), 1);
    }

    #[test]
    fn account_batch_bytes_roundtrip() {
        let events = vec![Account {
            id: u128::MAX,
            debits_pending: 1,
            debits_posted: 2,
            credits_pending: 3,
            credits_posted: 4,
            user_data_128: 5,
            user_data_64: 6,
            user_data_32: 7,
            reserved: 8,
            ledger: 9,
            code: 10,
            flags: AccountFlags::LINKED | AccountFlags::IMPORTED,
            timestamp: 11,
        }];
        let bytes = account_batch_to_bytes(&events);
        assert_eq!(bytes.len(), 128);
        assert_eq!(bytes_to_account_batch(&bytes), Some(events));
        assert_eq!(bytes_to_account_batch(&bytes[..100]), None);
    }

    #[test]
    fn transfer_batch_bytes_roundtrip() {
        let events = vec![Transfer {
            id: 1,
            debit_account_id: 2,
            credit_account_id: 3,
            amount: 4,
            pending_id: 5,
            user_data_128: 6,
            user_data_64: 7,
            user_data_32: 8,
            timeout: 9,
            ledger: 10,
            code: 11,
            flags: TransferFlags::PENDING,
            timestamp: 12,
        }];
        let bytes = transfer_batch_to_bytes(&events);
        assert_eq!(bytes.len(), 128);
        assert_eq!(bytes_to_transfer_batch(&bytes), Some(events));
        assert_eq!(bytes_to_transfer_batch(&bytes[..127]), None);
    }

    // ── create_account tests ─────────────────────────────────────────────

    #[test]
    fn account_created() {
        let a = Account { id: 1, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::Created);
    }

    #[test]
    fn account_id_zero() {
        let a = Account { id: 0, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::IdMustNotBeZero);
    }

    #[test]
    fn account_id_max() {
        let a = Account { id: u128::MAX, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::IdMustNotBeIntMax);
    }

    #[test]
    fn account_ledger_zero() {
        let a = Account { id: 1, ledger: 0, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::LedgerMustNotBeZero);
    }

    #[test]
    fn account_code_zero() {
        let a = Account { id: 1, ledger: 1, code: 0, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::CodeMustNotBeZero);
    }

    #[test]
    fn account_reserved_nonzero() {
        let a = Account { id: 1, reserved: 1, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::ReservedField);
    }

    #[test]
    fn account_debits_pending_nonzero() {
        let a = Account { id: 1, debits_pending: 1, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::DebitsPendingMustBeZero);
    }

    #[test]
    fn account_debits_posted_nonzero() {
        let a = Account { id: 1, debits_posted: 1, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::DebitsPostedMustBeZero);
    }

    #[test]
    fn account_credits_pending_nonzero() {
        let a = Account { id: 1, credits_pending: 1, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::CreditsPendingMustBeZero);
    }

    #[test]
    fn account_credits_posted_nonzero() {
        let a = Account { id: 1, credits_posted: 1, ledger: 1, code: 1, ..Account::default() };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::CreditsPostedMustBeZero);
    }

    #[test]
    fn account_flags_mutually_exclusive() {
        let a = Account {
            id: 1,
            flags: AccountFlags::DEBITS_MUST_NOT_EXCEED_CREDITS
                | AccountFlags::CREDITS_MUST_NOT_EXCEED_DEBITS,
            ledger: 1,
            code: 1,
            ..Account::default()
        };
        assert_eq!(create_account(&a, 100, None), CreateAccountStatus::FlagsAreMutuallyExclusive);
    }

    #[test]
    fn account_exists_same() {
        let a = Account {
            id: 1,
            flags: AccountFlags::DEBITS_MUST_NOT_EXCEED_CREDITS,
            user_data_128: 42,
            ledger: 7,
            code: 3,
            ..Account::default()
        };
        let e = Account {
            id: 1,
            flags: AccountFlags::DEBITS_MUST_NOT_EXCEED_CREDITS,
            user_data_128: 42,
            ledger: 7,
            code: 3,
            timestamp: 99,
            ..Account::default()
        };
        assert_eq!(create_account(&a, 100, Some(&e)), CreateAccountStatus::Exists);
    }

    #[test]
    fn account_exists_different_flags() {
        let a = Account {
            id: 1,
            flags: AccountFlags::DEBITS_MUST_NOT_EXCEED_CREDITS,
            ledger: 1,
            code: 1,
            ..Account::default()
        };
        let e = Account {
            id: 1,
            flags: AccountFlags::CREDITS_MUST_NOT_EXCEED_DEBITS,
            ledger: 1,
            code: 1,
            ..Account::default()
        };
        assert_eq!(
            create_account(&a, 100, Some(&e)),
            CreateAccountStatus::ExistsWithDifferentFlags
        );
    }

    #[test]
    fn account_exists_different_ledger() {
        let a = Account { id: 1, ledger: 1, code: 1, ..Account::default() };
        let e = Account { id: 1, ledger: 2, code: 1, ..Account::default() };
        assert_eq!(
            create_account(&a, 100, Some(&e)),
            CreateAccountStatus::ExistsWithDifferentLedger
        );
    }

    // ── create_transfer tests ────────────────────────────────────────────

    fn dr_account() -> Account {
        Account { id: 100, credits_posted: 1_000, ledger: 1, code: 1, ..Account::default() }
    }

    fn cr_account() -> Account {
        Account { id: 200, debits_posted: 1_000, ledger: 1, code: 1, ..Account::default() }
    }

    #[test]
    fn transfer_created() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::Created
        );
    }

    #[test]
    fn transfer_id_zero() {
        let t = Transfer { id: 0, ..Transfer::default() };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::IdMustNotBeZero
        );
    }

    #[test]
    fn transfer_id_max() {
        let t = Transfer { id: u128::MAX, ..Transfer::default() };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::IdMustNotBeIntMax
        );
    }

    #[test]
    fn transfer_debit_account_zero() {
        let t = Transfer {
            id: 1,
            debit_account_id: 0,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::DebitAccountIdMustNotBeZero
        );
    }

    #[test]
    fn transfer_credit_account_zero() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 0,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::CreditAccountIdMustNotBeZero
        );
    }

    #[test]
    fn transfer_accounts_must_be_different() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 100,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::AccountsMustBeDifferent
        );
    }

    #[test]
    fn transfer_ledger_must_not_be_zero() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 0,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::LedgerMustNotBeZero
        );
    }

    #[test]
    fn transfer_code_must_not_be_zero() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 0,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::CodeMustNotBeZero
        );
    }

    #[test]
    fn transfer_different_ledger() {
        let dr = Account { id: 100, ledger: 1, code: 1, ..Account::default() };
        let cr = Account { id: 200, ledger: 2, code: 1, ..Account::default() };
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr, &cr),
            CreateTransferStatus::AccountsMustHaveTheSameLedger
        );
    }

    #[test]
    fn transfer_wrong_ledger() {
        let dr = Account { id: 100, ledger: 1, code: 1, ..Account::default() };
        let cr = Account { id: 200, ledger: 1, code: 1, ..Account::default() };
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 2,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr, &cr),
            CreateTransferStatus::TransferMustHaveTheSameLedgerAsAccounts
        );
    }

    #[test]
    fn transfer_pending_must_not_be_zero() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            pending_id: 1,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::PendingIdMustBeZero
        );
    }

    #[test]
    fn transfer_timeout_for_pending_only() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            timeout: 60,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::TimeoutReservedForPendingTransfer
        );
    }

    #[test]
    fn transfer_closing_requires_pending() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            flags: TransferFlags::CLOSING_DEBIT,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr_account()),
            CreateTransferStatus::ClosingTransferMustBePending
        );
    }

    #[test]
    fn transfer_exceeds_credits() {
        let dr = Account {
            id: 100,
            credits_posted: 10,
            flags: AccountFlags::DEBITS_MUST_NOT_EXCEED_CREDITS,
            ledger: 1,
            code: 1,
            ..Account::default()
        };
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 100,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr, &cr_account()),
            CreateTransferStatus::ExceedsCredits
        );
    }

    #[test]
    fn transfer_exceeds_debits() {
        let cr = Account {
            id: 200,
            debits_posted: 10,
            flags: AccountFlags::CREDITS_MUST_NOT_EXCEED_DEBITS,
            ledger: 1,
            code: 1,
            ..Account::default()
        };
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 100,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr),
            CreateTransferStatus::ExceedsDebits
        );
    }

    #[test]
    fn transfer_exists_same() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        let e = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, Some(&e), &dr_account(), &cr_account()),
            CreateTransferStatus::Exists
        );
    }

    #[test]
    fn transfer_exists_different_amount() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 100,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        let e = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, Some(&e), &dr_account(), &cr_account()),
            CreateTransferStatus::ExistsWithDifferentAmount
        );
    }

    #[test]
    fn transfer_exists_different_flags() {
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            flags: TransferFlags::PENDING,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        let e = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, Some(&e), &dr_account(), &cr_account()),
            CreateTransferStatus::ExistsWithDifferentFlags
        );
    }

    #[test]
    fn transfer_debit_account_already_closed() {
        let dr = Account {
            id: 100,
            flags: AccountFlags::CLOSED,
            ledger: 1,
            code: 1,
            ..Account::default()
        };
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr, &cr_account()),
            CreateTransferStatus::DebitAccountAlreadyClosed
        );
    }

    #[test]
    fn transfer_credit_account_already_closed() {
        let cr = Account {
            id: 200,
            flags: AccountFlags::CLOSED,
            ledger: 1,
            code: 1,
            ..Account::default()
        };
        let t = Transfer {
            id: 1,
            debit_account_id: 100,
            credit_account_id: 200,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        assert_eq!(
            create_transfer(&t, 100, None, &dr_account(), &cr),
            CreateTransferStatus::CreditAccountAlreadyClosed
        );
    }

    // ── batch orchestrator tests ─────────────────────────────────────────

    #[test]
    fn batch_single_account() {
        let events = vec![Account { id: 1, ledger: 1, code: 1, ..Account::default() }];
        let results = execute_create_accounts(&events, 10, |_| None, 0, |_| None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, CreateAccountStatus::Created);
        assert_eq!(results[0].timestamp, 10);
    }

    #[test]
    fn batch_multiple_accounts() {
        let events = vec![
            Account { id: 1, ledger: 1, code: 1, ..Account::default() },
            Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            Account { id: 3, ledger: 1, code: 1, ..Account::default() },
        ];
        let results = execute_create_accounts(&events, 10, |_| None, 0, |_| None);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].timestamp, 8);
        assert_eq!(results[1].timestamp, 9);
        assert_eq!(results[2].timestamp, 10);
        for r in &results {
            assert_eq!(r.status, CreateAccountStatus::Created);
        }
    }

    #[test]
    fn batch_linked_chain_all_succeed() {
        let events = vec![
            Account {
                id: 1,
                ledger: 1,
                code: 1,
                flags: AccountFlags::LINKED,
                ..Account::default()
            },
            Account { id: 2, ledger: 1, code: 1, ..Account::default() },
        ];
        let results = execute_create_accounts(&events, 10, |_| None, 0, |_| None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].status, CreateAccountStatus::Created);
        assert_eq!(results[1].status, CreateAccountStatus::Created);
    }

    #[test]
    fn batch_linked_chain_last_event_linked() {
        let events = vec![
            Account { id: 1, ledger: 1, code: 1, ..Account::default() },
            Account {
                id: 2,
                ledger: 1,
                code: 1,
                flags: AccountFlags::LINKED,
                ..Account::default()
            },
        ];
        let results = execute_create_accounts(&events, 10, |_| None, 0, |_| None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].status, CreateAccountStatus::Created);
        assert_eq!(results[1].status, CreateAccountStatus::LinkedEventChainOpen);
    }

    #[test]
    fn batch_timestamp_must_be_zero() {
        let events =
            vec![Account { id: 1, ledger: 1, code: 1, timestamp: 5, ..Account::default() }];
        let results = execute_create_accounts(&events, 10, |_| None, 0, |_| None);
        assert_eq!(results[0].status, CreateAccountStatus::TimestampMustBeZero);
    }

    #[test]
    fn batch_imported_mixed() {
        let events = vec![
            Account {
                id: 1,
                ledger: 1,
                code: 1,
                flags: AccountFlags::IMPORTED,
                timestamp: 1,
                ..Account::default()
            },
            Account { id: 2, ledger: 1, code: 1, ..Account::default() },
        ];
        let results = execute_create_accounts(&events, 10, |_| None, 0, |_| None);
        assert_eq!(results[0].status, CreateAccountStatus::Created);
        assert_eq!(results[1].status, CreateAccountStatus::ImportedEventExpected);
    }

    #[test]
    fn batch_imported_timestamp_must_not_advance() {
        let events = vec![Account {
            id: 1,
            ledger: 1,
            code: 1,
            flags: AccountFlags::IMPORTED,
            timestamp: 15,
            ..Account::default()
        }];
        let results = execute_create_accounts(&events, 10, |_| None, 0, |_| None);
        assert_eq!(results[0].status, CreateAccountStatus::ImportedEventTimestampMustNotAdvance);
    }

    #[test]
    fn batch_transfers_linked_chain_breaks() {
        let events = vec![
            Transfer {
                id: 1,
                debit_account_id: 100,
                credit_account_id: 200,
                amount: 50,
                ledger: 1,
                code: 1,
                flags: TransferFlags::LINKED,
                ..Transfer::default()
            },
            Transfer {
                id: 2,
                debit_account_id: 100,
                credit_account_id: 200,
                amount: 50,
                ledger: 1,
                code: 1,
                ..Transfer::default()
            },
        ];
        let dr =
            Account { id: 100, credits_posted: 1_000, ledger: 1, code: 1, ..Account::default() };
        let cr =
            Account { id: 200, debits_posted: 1_000, ledger: 1, code: 1, ..Account::default() };
        let results = execute_create_transfers(
            &events,
            10,
            |_| None,
            |id| {
                if id == 100 {
                    Some(dr)
                } else if id == 200 {
                    Some(cr)
                } else {
                    None
                }
            },
            0,
            |_| None,
            |_| TransferPendingStatus::None,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].status, CreateTransferStatus::Created);
        assert_eq!(results[1].status, CreateTransferStatus::Created);
    }

    // ── Imported timestamp validation tests ──────────────────────────────

    fn reply_status(body: &[u8]) -> u32 {
        u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4]))
    }

    fn reply_status_at(body: &[u8], index: usize) -> u32 {
        u32::from_le_bytes(body[16 * index + 8..16 * index + 12].try_into().unwrap_or([0; 4]))
    }

    fn imported_account(id: u128, timestamp: u64) -> Account {
        Account {
            id,
            ledger: 1,
            code: 1,
            flags: AccountFlags::IMPORTED,
            timestamp,
            ..Account::default()
        }
    }

    #[test]
    fn imported_account_keeps_and_indexes_its_timestamp() {
        let mut sm = StateMachine::default();
        let body = sm.create_accounts(&[imported_account(1, 5)], 10);
        // The account is stored at its imported timestamp, not the batch slot.
        assert_eq!(u64::from_le_bytes(body[0..8].try_into().unwrap_or([0; 8])), 5);
        assert_eq!(reply_status(&body), CreateAccountStatus::Created as u32);
        assert_eq!(sm.accounts.get(&1).expect("stored account").timestamp, 5);
        assert_eq!(sm.accounts_by_timestamp.get(&5), Some(&1));
        assert_eq!(sm.accounts_timestamp_max, 5);
    }

    #[test]
    fn imported_account_timestamp_must_not_regress() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(&[imported_account(1, 5)], 10);

        // Equal timestamp regresses the account index.
        let body = sm.create_accounts(&[imported_account(2, 5)], 20);
        assert_eq!(
            reply_status(&body),
            CreateAccountStatus::ImportedEventTimestampMustNotRegress as u32
        );
        assert_eq!(sm.accounts.len(), 1);

        // A later capacity check: the batch orchestrator tracks within-batch
        // inserts, so a second imported event going backward is rejected too.
        // (The reply's first record is legitimately Created; the regression
        // fires on the second event.)
        let batch = [imported_account(3, 30), imported_account(4, 25)];
        let body = sm.create_accounts(&batch, 40);
        assert_eq!(reply_status_at(&body, 0), CreateAccountStatus::Created as u32);
        assert_eq!(
            reply_status_at(&body, 1),
            CreateAccountStatus::ImportedEventTimestampMustNotRegress as u32
        );
        assert_eq!(sm.accounts.len(), 2);
    }

    #[test]
    fn imported_account_timestamp_must_not_collide_with_a_transfer() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // The transfer takes batch slot 20.
        let _ = sm.create_transfers(
            &[Transfer {
                id: 1,
                debit_account_id: 1,
                credit_account_id: 2,
                amount: 1,
                ledger: 1,
                code: 1,
                ..Transfer::default()
            }],
            20,
        );

        // An account imported at the transfer's timestamp must not regress
        // against the transfers index.
        let body = sm.create_accounts(&[imported_account(3, 20)], 30);
        assert_eq!(
            reply_status(&body),
            CreateAccountStatus::ImportedEventTimestampMustNotRegress as u32
        );
    }

    fn imported_transfer(id: u128, timestamp: u64, flags: TransferFlags) -> Transfer {
        Transfer {
            id,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 1,
            ledger: 1,
            code: 1,
            flags,
            timestamp,
            ..Transfer::default()
        }
    }

    fn accounts_for_transfers(sm: &mut StateMachine) {
        // Separate batches so the debit account is stamped 10 and the credit
        // account 20 — a gap that lets the postdate checks fire without the
        // regression/collision checks (which upstream runs first) colliding.
        let _ =
            sm.create_accounts(&[Account { id: 1, ledger: 1, code: 1, ..Account::default() }], 10);
        let _ =
            sm.create_accounts(&[Account { id: 2, ledger: 1, code: 1, ..Account::default() }], 20);
    }

    #[test]
    fn imported_transfer_timestamp_must_postdate_accounts() {
        let mut sm = StateMachine::default();
        accounts_for_transfers(&mut sm);

        // Below the debit account's timestamp (10): debit error.
        let body = sm.create_transfers(&[imported_transfer(3, 9, TransferFlags::IMPORTED)], 30);
        assert_eq!(
            reply_status(&body),
            CreateTransferStatus::ImportedEventTimestampMustPostdateDebitAccount as u32
        );

        // Below or equal to the credit account's timestamp (20): credit error.
        let body = sm.create_transfers(&[imported_transfer(4, 11, TransferFlags::IMPORTED)], 30);
        assert_eq!(
            reply_status(&body),
            CreateTransferStatus::ImportedEventTimestampMustPostdateCreditAccount as u32
        );

        // A timestamp after both accounts is accepted and stored as-is.
        let body = sm.create_transfers(&[imported_transfer(5, 21, TransferFlags::IMPORTED)], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
        assert_eq!(sm.transfers.get(&5).expect("stored transfer").timestamp, 21);
        assert_eq!(sm.transfers_by_timestamp.get(&21), Some(&5));
        assert_eq!(sm.transfers_timestamp_max, 21);
    }

    #[test]
    fn imported_pending_transfer_timeout_must_be_zero() {
        let mut sm = StateMachine::default();
        accounts_for_transfers(&mut sm);

        let timed = imported_transfer(3, 21, TransferFlags::IMPORTED | TransferFlags::PENDING);
        let timed = Transfer { timeout: 60, ..timed };
        let body = sm.create_transfers(&[timed], 30);
        assert_eq!(
            reply_status(&body),
            CreateTransferStatus::ImportedEventTimeoutMustBeZero as u32
        );
    }

    // ── Balance mutation (commit_transfer) tests ─────────────────────────

    fn t(dr: u128, cr: u128, amount: u128) -> Transfer {
        Transfer {
            id: dr * 1_000 + cr,
            debit_account_id: dr,
            credit_account_id: cr,
            amount,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        }
    }

    #[test]
    fn created_transfer_updates_posted_balances() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        let body = sm.create_transfers(&[t(1, 2, 100)], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert_eq!(dr.debits_posted, 100);
        assert_eq!(cr.credits_posted, 100);
        assert_eq!(dr.debits_pending, 0);
        assert_eq!(cr.credits_pending, 0);
        // The stored transfer records the amount actually applied.
        assert_eq!(sm.transfers.get(&1002).expect("transfer stored").amount, 100);
    }

    #[test]
    fn created_pending_transfer_updates_pending_balances() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        let pending = Transfer { flags: TransferFlags::PENDING, ..t(1, 2, 50) };
        let body = sm.create_transfers(&[pending], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert_eq!(dr.debits_pending, 50);
        assert_eq!(cr.credits_pending, 50);
        assert_eq!(dr.debits_posted, 0);
        assert_eq!(cr.credits_posted, 0);
    }

    #[test]
    fn closing_transfer_closes_the_debit_account() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        let closing = Transfer {
            flags: TransferFlags::CLOSING_DEBIT | TransferFlags::PENDING,
            ..t(1, 2, 10)
        };
        let body = sm.create_transfers(&[closing], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert!(dr.flags.closed());
        assert_eq!(dr.debits_pending, 10);
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert!(!cr.flags.closed());
    }

    #[test]
    fn balanced_pending_transfer_records_amount_actual() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
                Account { id: 3, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Give account 1 100 credit via a posted transfer.
        let _ = sm.create_transfers(&[t(3, 1, 100)], 20);
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.credits_posted, 100);

        // A balancing_debit request of 200 applies only the 100 available
        // (upstream `balancing_debit`, state_machine.zig:3795).
        let balanced = Transfer {
            amount: 200,
            flags: TransferFlags::BALANCING_DEBIT | TransferFlags::PENDING,
            ..t(1, 2, 200)
        };
        let body = sm.create_transfers(&[balanced], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 100);
        // The stored transfer pins the amount actually applied.
        assert_eq!(sm.transfers.get(&1002).expect("transfer stored").amount, 100);
    }

    #[test]
    fn sequential_created_transfers_cumulate_balances() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        let batch = [t(1, 2, 50), t(1, 2, 70)];
        let body = sm.create_transfers(&batch, 20);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::Created as u32);

        // Mutations accumulate across the batch in event order.
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_posted, 120);
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert_eq!(cr.credits_posted, 120);
    }

    #[test]
    fn broken_chain_does_not_apply_balance_mutations() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // A linked chain whose second member references a missing account.
        let first = Transfer { flags: TransferFlags::LINKED, ..t(1, 2, 100) };
        let second = Transfer { flags: TransferFlags::LINKED, ..t(99, 2, 100) };
        let body = sm.create_transfers(&[first, second], 20);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::LinkedEventFailed as u32);
        // The closing member of a linked chain short-circuits to
        // `linked_event_chain_open` without validating its body
        // (upstream state_machine.zig:3039-3042).
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::LinkedEventChainOpen as u32);

        // The whole chain is discarded: no transfers, no balance mutations.
        assert!(sm.transfers.is_empty());
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_posted, 0);
    }

    #[test]
    fn transfer_after_closing_account_in_same_batch_is_rejected() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // Close account 1 with the first event, then reference it in the second.
        let closing = Transfer {
            flags: TransferFlags::PENDING | TransferFlags::CLOSING_DEBIT,
            ..t(1, 2, 10)
        };
        let body = sm.create_transfers(&[closing, t(1, 2, 5)], 20);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        // The in-session closing is visible to the second event: it is rejected
        // (upstream's `create_transfer` sees the scoped mutation).
        assert_eq!(
            reply_status_at(&body, 1),
            CreateTransferStatus::DebitAccountAlreadyClosed as u32
        );

        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert!(dr.flags.closed());
        assert_eq!(dr.debits_pending, 10);
        assert_eq!(sm.transfers.len(), 1);
    }

    #[test]
    fn balancing_sees_earlier_in_batch_mutations() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
                Account { id: 3, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // First give account 1 100 credit in-session, then balance against it.
        let batch = [
            t(3, 1, 100),
            Transfer {
                amount: 200,
                flags: TransferFlags::PENDING | TransferFlags::BALANCING_DEBIT,
                ..t(1, 2, 200)
            },
        ];
        let body = sm.create_transfers(&batch, 20);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.credits_posted, 100);
        // The balancing cap of the second event sees the first event's credit.
        assert_eq!(dr.debits_pending, 100);
        assert_eq!(sm.transfers.get(&1002).expect("transfer stored").amount, 100);
    }

    #[test]
    fn broken_chain_rolls_back_in_session_mutations_before_later_events() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // A three-member chain whose middle member references a missing account
        // (the first member is rolled back, the middle reports its own error,
        // the last is marked failed).
        let chain_first = Transfer { flags: TransferFlags::LINKED, ..t(1, 2, 100) };
        let chain_middle = Transfer { flags: TransferFlags::LINKED, ..t(99, 2, 100) };
        let chain_last = Transfer { flags: TransferFlags::LINKED, ..t(1, 2, 100) };
        // The next non-linked event still sees the broken (open) chain and is
        // itself marked failed, closing it; only the following event recovers.
        let closing_event = t(1, 2, 5);
        let recovered_event = t(1, 2, 7);
        let batch = [chain_first, chain_middle, chain_last, closing_event, recovered_event];
        let body = sm.create_transfers(&batch, 40);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::LinkedEventFailed as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::DebitAccountNotFound as u32);
        assert_eq!(reply_status_at(&body, 2), CreateTransferStatus::LinkedEventFailed as u32);
        assert_eq!(reply_status_at(&body, 3), CreateTransferStatus::LinkedEventFailed as u32);
        assert_eq!(reply_status_at(&body, 4), CreateTransferStatus::Created as u32);

        // Only the recovered post-chain transfer persisted; the broken chain's
        // A applied no balance (rollback) and was not stored.
        assert_eq!(sm.transfers.len(), 1);
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_posted, 7);
    }

    fn pending_transfer_id() -> u128 {
        // `t(1, 2, _)` ids the transfer as `dr * 1000 + cr`.
        1002
    }

    /// A committed pending transfer of `amount` between accounts 1 and 2,
    /// stamped 20, plus the two accounts (stamped 10).
    fn pending_setup(sm: &mut StateMachine, amount: u128, flags: TransferFlags) {
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let pending = Transfer { flags: TransferFlags::PENDING | flags, ..t(1, 2, amount) };
        let body = sm.create_transfers(&[pending], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
    }

    fn post_event(id: u128, amount: u128) -> Transfer {
        Transfer {
            id,
            pending_id: pending_transfer_id(),
            amount,
            flags: TransferFlags::POST_PENDING_TRANSFER,
            ..Transfer::default()
        }
    }

    fn void_event(id: u128, amount: u128) -> Transfer {
        Transfer {
            id,
            pending_id: pending_transfer_id(),
            amount,
            flags: TransferFlags::VOID_PENDING_TRANSFER,
            ..Transfer::default()
        }
    }

    // ── Post/void pending transfer tests ─────────────────────────────────

    #[test]
    fn post_pending_transfer_moves_pending_to_posted() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        let body = sm.create_transfers(&[post_event(3, u128::MAX)], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 0);
        assert_eq!(dr.debits_posted, 50);
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert_eq!(cr.credits_pending, 0);
        assert_eq!(cr.credits_posted, 50);

        // The stored transfer folds the pending's account/ledger/code fields and
        // pins the amount actually applied.
        let stored = sm.transfers.get(&3).expect("post transfer stored");
        assert_eq!(stored.amount, 50);
        assert_eq!(stored.debit_account_id, 1);
        assert_eq!(stored.credit_account_id, 2);
        assert_eq!(stored.pending_id, pending_transfer_id());
        assert_eq!(stored.ledger, 1);
        assert_eq!(stored.code, 1);
        assert_eq!(stored.timestamp, 30);
        let transfer_pending = sm.transfers_pending.get(&20).expect("pending index entry").status;
        assert_eq!(transfer_pending, TransferPendingStatus::Posted);
    }

    #[test]
    fn post_pending_transfer_partial_amount_releases_full_hold() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        let body = sm.create_transfers(&[post_event(3, 30)], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 0);
        assert_eq!(dr.debits_posted, 30);
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert_eq!(cr.credits_pending, 0);
        assert_eq!(cr.credits_posted, 30);
        assert_eq!(sm.transfers.get(&3).expect("post transfer stored").amount, 30);
    }

    #[test]
    fn void_pending_transfer_returns_funds() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        // An explicit amount is required by void; `0` means "the full amount".
        let body = sm.create_transfers(&[void_event(3, 0)], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 0);
        assert_eq!(dr.debits_posted, 0);
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert_eq!(cr.credits_pending, 0);
        assert_eq!(cr.credits_posted, 0);
        assert_eq!(sm.transfers.get(&3).expect("void transfer stored").amount, 50);
        let transfer_pending = sm.transfers_pending.get(&20).expect("pending index entry").status;
        assert_eq!(transfer_pending, TransferPendingStatus::Voided);
    }

    #[test]
    fn void_pending_transfer_partial_amount_is_rejected() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        let body = sm.create_transfers(&[void_event(3, 10)], 30);
        assert_eq!(
            reply_status(&body),
            CreateTransferStatus::PendingTransferHasDifferentAmount as u32
        );
        // Nothing applied.
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 50);
        assert_eq!(sm.transfers.len(), 1);
    }

    #[test]
    fn void_pending_transfer_reopens_closing_account() {
        let mut sm = StateMachine::default();
        // The pending transfer closes its debit account.
        pending_setup(&mut sm, 10, TransferFlags::CLOSING_DEBIT);
        assert!(sm.accounts.get(&1).expect("debit account").flags.closed());
        assert_eq!(sm.accounts.get(&1).expect("debit account").debits_pending, 10);

        let body = sm.create_transfers(&[void_event(3, 0)], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
        let dr = sm.accounts.get(&1).expect("debit account stored");
        // Voiding reopens the account the pending closing operation shut.
        assert!(!dr.flags.closed());
        assert_eq!(dr.debits_pending, 0);
    }

    #[test]
    fn post_pending_transfer_on_closed_account_is_rejected() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 10, TransferFlags::CLOSING_DEBIT);

        let body = sm.create_transfers(&[post_event(3, u128::MAX)], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::DebitAccountAlreadyClosed as u32);
        // The account stays closed and holding.
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert!(dr.flags.closed());
        assert_eq!(dr.debits_pending, 10);
    }

    #[test]
    fn post_pending_transfer_without_pending_is_rejected() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // A post whose pending_id references nothing (the pending was never
        // created) is rejected before any account access.
        let evicted = Transfer {
            id: 3,
            pending_id: 999,
            amount: 10,
            flags: TransferFlags::POST_PENDING_TRANSFER,
            ..Transfer::default()
        };
        let body = sm.create_transfers(&[evicted], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::PendingTransferNotFound as u32);
    }

    #[test]
    fn post_pending_transfer_with_expired_pending_is_rejected() {
        let mut sm = StateMachine::default();
        // A 1-second timeout expires the pending at `20 + 1e9`.
        pending_setup(&mut sm, 10, TransferFlags::default());
        let pending = sm.transfers.get_mut(&1002).expect("pending stored");
        pending.timeout = 1;

        let body = sm.create_transfers(&[post_event(3, u128::MAX)], 1_000_000_021);
        assert_eq!(reply_status(&body), CreateTransferStatus::PendingTransferExpired as u32);
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 10);
    }

    #[test]
    fn double_post_of_same_pending_is_rejected_within_and_across_batches() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        // Across batches: the second post reports the committed status.
        let _ = sm.create_transfers(&[post_event(3, u128::MAX)], 30);
        let body = sm.create_transfers(&[post_event(4, u128::MAX)], 40);
        assert_eq!(reply_status(&body), CreateTransferStatus::PendingTransferAlreadyPosted as u32);

        // Within a batch: the second post sees the first's in-session status
        // change (the `working_pending` overlay).
        let mut sm2 = StateMachine::default();
        pending_setup(&mut sm2, 50, TransferFlags::default());
        let batch = [post_event(3, u128::MAX), post_event(4, u128::MAX)];
        let body = sm2.create_transfers(&batch, 30);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(
            reply_status_at(&body, 1),
            CreateTransferStatus::PendingTransferAlreadyPosted as u32
        );
        // Only the first post persisted (pending + one post).
        assert_eq!(sm2.transfers.len(), 2);
    }

    #[test]
    fn void_pending_transfer_after_post_is_rejected() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());
        let _ = sm.create_transfers(&[post_event(3, u128::MAX)], 30);

        let body = sm.create_transfers(&[void_event(4, 0)], 40);
        assert_eq!(reply_status(&body), CreateTransferStatus::PendingTransferAlreadyPosted as u32);
    }

    #[test]
    fn resubmitted_post_reports_exists_with_original_timestamp() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        let first = sm.create_transfers(&[post_event(3, u128::MAX)], 30);
        assert_eq!(reply_status(&first), CreateTransferStatus::Created as u32);

        // Idempotent resubmission: `Exists` carries the stored post's timestamp.
        let second = sm.create_transfers(&[post_event(3, u128::MAX)], 40);
        assert_eq!(u64::from_le_bytes(second[0..8].try_into().unwrap_or([0; 8])), 30);
        assert_eq!(reply_status(&second), CreateTransferStatus::Exists as u32);
        assert_eq!(sm.transfers.len(), 2);

        // A resubmission with a different amount is a conflict.
        let third = sm.create_transfers(&[post_event(3, 10)], 50);
        assert_eq!(reply_status(&third), CreateTransferStatus::ExistsWithDifferentAmount as u32);

        // A resubmission with a different pending id is a conflict.
        let different_pending = Transfer { pending_id: 77, ..post_event(3, u128::MAX) };
        let fourth = sm.create_transfers(&[different_pending], 50);
        assert_eq!(
            reply_status(&fourth),
            CreateTransferStatus::ExistsWithDifferentPendingId as u32
        );
    }

    #[test]
    fn post_folds_fields_from_event_or_pending() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        // Explicit (matching) account/ledger/code and user-data fold from the
        // event; the pending's values win only for the omitted fields.
        let post = Transfer {
            id: 3,
            debit_account_id: 1,
            credit_account_id: 2,
            pending_id: pending_transfer_id(),
            amount: 10,
            user_data_128: 700,
            ledger: 1,
            code: 1,
            flags: TransferFlags::POST_PENDING_TRANSFER,
            ..Transfer::default()
        };
        let body = sm.create_transfers(&[post], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
        let stored = sm.transfers.get(&3).expect("post transfer stored");
        assert_eq!(stored.user_data_128, 700);

        // A mismatching ledger is rejected.
        let mismatched = Transfer { ledger: 2, ..post_event(4, 10) };
        let body = sm.create_transfers(&[mismatched], 40);
        assert_eq!(
            reply_status(&body),
            CreateTransferStatus::PendingTransferHasDifferentLedger as u32
        );
    }

    #[test]
    fn post_of_pending_created_in_same_batch_is_not_visible() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // `DEVIATION`: `get_existing_transfer` reads committed state only, so a
        // pending transfer created earlier in the same batch is not yet visible
        // to its own post/void (upstream would validate it against the in-session
        // scope). Pinned here so the deferred behavior is explicit.
        let pending = Transfer { flags: TransferFlags::PENDING, ..t(1, 2, 50) };
        let post = Transfer {
            id: 3,
            pending_id: pending_transfer_id(),
            amount: u128::MAX,
            flags: TransferFlags::POST_PENDING_TRANSFER,
            ..Transfer::default()
        };
        let body = sm.create_transfers(&[pending, post], 30);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::PendingTransferNotFound as u32);
        // Only the pending transfer persisted.
        assert_eq!(sm.transfers.len(), 1);
    }
}
