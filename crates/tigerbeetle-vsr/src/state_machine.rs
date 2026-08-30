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
    clippy::too_many_arguments,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::unnecessary_cast,
    clippy::redundant_else,
    clippy::collapsible_if,
    clippy::match_same_arms,
    clippy::module_name_repetitions
)]

use std::collections::{HashMap, HashSet};

use tigerbeetle_core::constants;
use tigerbeetle_core::types::{
    Account, AccountBalance, AccountEvent, AccountFilter, AccountFilterFlags, AccountFlags,
    ChangeEvent, ChangeEventType, ChangeEventsFilter, CreateAccountStatus, CreateTransferStatus,
    QueryFilter, QueryFilterFlags, Transfer, TransferFlags, TransferPending, TransferPendingStatus,
};
use tigerbeetle_lsm::timestamp_range::TimestampRange;

use crate::Operation;

// ---------------------------------------------------------------------------
// Query reply limits
// ---------------------------------------------------------------------------

/// The maximum number of results in a scan-based query reply, given the result
/// size (upstream `Operation.result_max`, `tigerbeetle.zig:907`).
///
/// DEVIATION: upstream also constrains `get_change_events` by the runtime
/// `batch_size_limit`-derived prefetch budget
/// (`prefetch_get_change_events`, `state_machine.zig:2251-2262`); sans-IO the
/// batch/prefetch budget is deferred (Phase 3), and the message-body bound is
/// the binding constraint whenever `batch_size_limit >= message_body_size_max`.
fn result_max(result_size: usize) -> usize {
    constants::MESSAGE_BODY_SIZE_MAX / result_size
}

/// Like [`result_max`], but for multi-batch operations whose reply carries a
/// multi-batch trailer (upstream `Operation.result_max`,
/// `tigerbeetle.zig:923-930`, `multi_batch.trailer_total_size`).
fn result_max_multi_batch(result_size: usize) -> usize {
    // The trailer for an operand-count of one: a `TrailerItem` + `Postamble`
    // (2 bytes each), padded up to the element size.
    debug_assert!(result_size.is_power_of_two());

    let trailer_size = 4usize.div_ceil(result_size) * result_size;
    (constants::MESSAGE_BODY_SIZE_MAX - trailer_size) / result_size
}

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

    // Post/void pending transfers are dispatched upstream inside `create_transfer`
    // (`state_machine.zig:3744-3746`); in this port the orchestrator
    // (`execute_create_transfers`) routes them to `post_or_void_pending_transfer`
    // before calling this helper, so a plain creation must never carry those flags.
    assert!(
        !t.flags.post_pending_transfer() && !t.flags.void_pending_transfer(),
        "post/void transfers are handled by post_or_void_pending_transfer"
    );

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

/// Build the stored record for a posting/voiding transfer (upstream
/// `state_machine.zig:4195-4209`): the pending transfer's debit/credit,
/// ledger and code are folded in, `timeout` is zeroed, user-data fields fall
/// back to the pending's values, and the amount is the validated
/// `amount_actual`.
fn fold_post_void_transfer(
    event: &Transfer,
    p: &Transfer,
    amount_actual: u128,
    timestamp: u64,
) -> Transfer {
    // Mirror of the orchestrator's exists-path: the pending's fields must
    // match the stored post/void record.
    debug_assert_eq!(event.pending_id, p.id);
    Transfer {
        id: event.id,
        debit_account_id: p.debit_account_id,
        credit_account_id: p.credit_account_id,
        amount: amount_actual,
        pending_id: event.pending_id,
        user_data_128: if event.user_data_128 != 0 { event.user_data_128 } else { p.user_data_128 },
        user_data_64: if event.user_data_64 != 0 { event.user_data_64 } else { p.user_data_64 },
        user_data_32: if event.user_data_32 != 0 { event.user_data_32 } else { p.user_data_32 },
        timeout: 0,
        ledger: p.ledger,
        code: p.code,
        flags: event.flags,
        timestamp,
    }
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
/// `running_key_max` and `account_with_timestamp` back the imported-timestamp
/// regression/collision checks (the transfers object tree's key range and the
/// accounts timestamp index, per upstream `state_machine.zig:4158-4180`).
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
    running_key_max: u64,
    account_with_timestamp: &impl Fn(u64) -> Option<u128>,
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

    // Imported-timestamp regression/collision checks. Upstream runs these after
    // the pending-status and expiry-time checks — so those codes take precedence
    // over a regressing timestamp — and before the closed-account checks
    // (state_machine.zig:4158-4180). A past timestamp must not regress the
    // transfer object index, and must not collide with an account's timestamp.
    // Imported post/void events skip the postdate asserts (the pending's
    // accounts make them moot: the committed pending predates them).
    if t.flags.imported() {
        assert!(t.timestamp != 0);
        assert!(t.timestamp <= timestamp_event);
        if t.timestamp <= running_key_max || account_with_timestamp(t.timestamp).is_some() {
            return PostVoidPendingResult {
                status: CreateTransferStatus::ImportedEventTimestampMustNotRegress,
                amount_actual,
                is_post,
            };
        }
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
/// `get_existing_transfer` looks up a transfer by id. It must return the
/// committed state: the orchestrator overlays an in-session working view of
/// created transfers (`working_transfers`) on top, so later batch events see
/// transfers created earlier in the batch — including pending transfers that a
/// post/void event references (upstream inserts into the transfers groove at
/// each create, so the in-session scope is visible there too).
///
/// `get_pending_status` returns the committed status (`transfers_pending`) of
/// the pending transfer with the given timestamp, which post/void events
/// validate against. A working overlay of in-session post/void status changes
/// (and pending creations) sits on top.
///
/// `is_orphaned` reports whether an id sits in the committed orphaned
/// primary-key set (transfer ids that failed with a transient status and can
/// never be created again); ids orphaned earlier in this batch are layered on
/// top, mirroring upstream's in-session groove state.
pub fn execute_create_transfers<F, G, FAccountAt, FPendingStatus, FOrphaned>(
    events: &[Transfer],
    timestamp: u64,
    mut get_existing_transfer: F,
    mut get_account: G,
    transfers_key_max: u64,
    account_with_timestamp: FAccountAt,
    mut get_pending_status: FPendingStatus,
    is_orphaned: FOrphaned,
) -> Vec<CreateTransferResult>
where
    F: FnMut(u128) -> Option<Transfer>,
    G: FnMut(u128) -> Option<Account>,
    FAccountAt: Fn(u64) -> Option<u128>,
    FPendingStatus: FnMut(u64) -> TransferPendingStatus,
    FOrphaned: Fn(u128) -> bool,
{
    let mut results: Vec<CreateTransferResult> = Vec::with_capacity(events.len());
    let mut chain: Option<usize> = None;
    let mut chain_broken = false;
    let mut running_key_max = transfers_key_max;
    // In-session working view of the accounts, layered over `get_account`
    // (upstream's open batch scope). Mutations from created events accumulate
    // here so later batch members validate against them.
    let mut working: HashMap<u128, Account> = HashMap::new();
    // In-session working view of created transfers, layered over
    // `get_existing_transfer` (upstream inserts each transfer into the transfers
    // groove as it commits, so the batch observes it).
    let mut working_transfers: HashMap<u128, Transfer> = HashMap::new();
    // Chain scoping: snapshot the working views (and the running key range) when
    // a chain opens, and restore them if it breaks (upstream `scope_close(.discard)`
    // rolls back the accounts, transfers and transfers_pending grooves).
    let mut chain_snapshot: Option<HashMap<u128, Account>> = None;
    let mut chain_key_max_snapshot: Option<u64> = None;
    let mut chain_transfers_snapshot: Option<HashMap<u128, Transfer>> = None;
    // In-session view of orphaned ids, layered over `is_orphaned`. Upstream
    // writes orphans outside the chain scope (`transient_error`,
    // `state_machine.zig:3215-3252`), so they survive a chain rollback — the
    // chain snapshots above deliberately exclude this set.
    let mut working_orphaned: HashSet<u128> = HashSet::new();
    // In-session view of pending-transfer statuses, layered over
    // `get_pending_status` (a post/void within the batch marks its pending as
    // posted/voided so a later event sees it; pending creations seed `Pending`).
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
                    chain_transfers_snapshot = Some(working_transfers.clone());
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
            // For regular transfers, look up existing for idempotency. The
            // in-session working view takes precedence over the committed state
            // (a transfer created earlier in this batch is already visible).
            let existing = working_transfers
                .get(&event.id)
                .copied()
                .or_else(|| get_existing_transfer(event.id));

            // Upstream consults the orphaned primary-key set in the same switch
            // as the existing record (state_machine.zig:3734-3737): an id that
            // previously failed with a transient status can never be created
            // again, short-circuiting everything else (including post/void).
            if working_orphaned.contains(&event.id) || is_orphaned(event.id) {
                break 'result (CreateTransferStatus::IdAlreadyFailed, timestamp_event);
            }

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

                let pending = working_transfers
                    .get(&event.pending_id)
                    .copied()
                    .or_else(|| get_existing_transfer(event.pending_id));

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
                        running_key_max,
                        &account_with_timestamp,
                    );
                    let status = result.status;
                    amount_actual = result.amount_actual;
                    let ts = if status == CreateTransferStatus::Created {
                        // Apply the in-session side effects: the stored post/void
                        // record, the pending's status and accounts, mirroring
                        // what `persist` later re-derives from the committed
                        // state.
                        let ts =
                            if event.flags.imported() { event.timestamp } else { timestamp_event };
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

                        let record = fold_post_void_transfer(event, p, amount_actual, ts);
                        working_transfers.insert(record.id, record);
                        ts
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

                    // The created transfer is visible to later batch events
                    // (upstream inserts into the transfers groove at commit); a
                    // pending creation also seeds its status row.
                    let mut record = *event;
                    record.amount = amount_actual;
                    record.timestamp = ts;
                    working_transfers.insert(record.id, record);
                    if record.flags.pending() {
                        working_pending.insert(record.timestamp, TransferPendingStatus::Pending);
                    }
                } else {
                    amount_actual = 0;
                }
                (status, ts)
            };
            if matches!(status, CreateTransferStatus::Created) {
                running_key_max = running_key_max.max(ts);
            }
            (status, ts)
        };

        // Chain error handling.
        if status != CreateTransferStatus::Created {
            // Upstream `transient_error` (state_machine.zig:3215-3252): a
            // transfer that fails with a transient code poisons the id in the
            // orphaned primary-key set. This runs _after_ the chain rollback, so
            // the orphan survives it.
            if status.transient() {
                working_orphaned.insert(event.id);
            }
            if let Some(chain_start) = chain {
                if !chain_broken {
                    chain_broken = true;
                    // TODO(port): scope_close(.discard) — roll back this chain's
                    // in-session writes (only the events since it opened).
                    if let (
                        Some(snapshot),
                        Some(key_max_snapshot),
                        Some(transfers_snapshot),
                        Some(pending_snapshot),
                    ) = (
                        chain_snapshot.as_ref(),
                        chain_key_max_snapshot.as_ref(),
                        chain_transfers_snapshot.as_ref(),
                        chain_pending_snapshot.as_ref(),
                    ) {
                        working.clone_from(snapshot);
                        running_key_max = *key_max_snapshot;
                        working_transfers.clone_from(transfers_snapshot);
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
            chain_transfers_snapshot = None;
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

/// Encode a batch of [`AccountBalance`] snapshots as a reply body (128 bytes
/// each, LE).
#[must_use]
pub fn account_balances_to_bytes(snapshots: &[AccountBalance]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(snapshots.len() * 128);
    for snapshot in snapshots {
        let mut record = [0_u8; 128];
        record[0..16].copy_from_slice(&snapshot.debits_pending.to_le_bytes());
        record[16..32].copy_from_slice(&snapshot.debits_posted.to_le_bytes());
        record[32..48].copy_from_slice(&snapshot.credits_pending.to_le_bytes());
        record[48..64].copy_from_slice(&snapshot.credits_posted.to_le_bytes());
        record[64..72].copy_from_slice(&snapshot.timestamp.to_le_bytes());
        record[72..128].copy_from_slice(&snapshot.reserved);
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

/// Decode a request body into a single [`ChangeEventsFilter`] (64 bytes).
///
/// Returns `None` unless the body is exactly one filter record.
#[must_use]
pub fn bytes_to_change_events_filter(bytes: &[u8]) -> Option<ChangeEventsFilter> {
    if bytes.len() != 64 {
        return None;
    }
    let mut reserved = [0_u8; 44];
    reserved.copy_from_slice(&bytes[20..64]);
    Some(ChangeEventsFilter {
        timestamp_min: le_u64(bytes, 0),
        timestamp_max: le_u64(bytes, 8),
        limit: le_u32(bytes, 16),
        reserved,
    })
}

/// Parse a `lookup_accounts`/`lookup_transfers` request body: a batch of
/// `u128` ids (8 bytes each, LE), exactly dividing the buffer.
#[must_use]
fn bytes_to_lookup_ids(bytes: &[u8]) -> Option<Vec<u128>> {
    if !bytes.len().is_multiple_of(16) {
        return None;
    }
    let (chunks, remainder) = bytes.as_chunks::<16>();
    assert!(remainder.is_empty());
    Some(chunks.iter().map(|c| le_u128(c, 0)).collect())
}

/// Parse a `get_account_transfers`/`get_account_balances` request body: a
/// single 128-byte LE [`AccountFilter`].
#[must_use]
fn bytes_to_account_filter(bytes: &[u8]) -> Option<AccountFilter> {
    if bytes.len() != 128 {
        return None;
    }
    let mut reserved = [0_u8; 58];
    reserved.copy_from_slice(&bytes[46..104]);
    Some(AccountFilter {
        account_id: le_u128(bytes, 0),
        user_data_128: le_u128(bytes, 16),
        user_data_64: le_u64(bytes, 32),
        user_data_32: le_u32(bytes, 40),
        code: u16::from_le_bytes(bytes[44..46].try_into().expect("2 bytes")),
        reserved,
        timestamp_min: le_u64(bytes, 104),
        timestamp_max: le_u64(bytes, 112),
        limit: le_u32(bytes, 120),
        flags: AccountFilterFlags::from_raw(le_u32(bytes, 124)),
    })
}

/// Parse a `query_accounts`/`query_transfers` request body: a single 64-byte
/// LE [`QueryFilter`].
#[must_use]
fn bytes_to_query_filter(bytes: &[u8]) -> Option<QueryFilter> {
    if bytes.len() != 64 {
        return None;
    }
    let mut reserved = [0_u8; 6];
    reserved.copy_from_slice(&bytes[34..40]);
    Some(QueryFilter {
        user_data_128: le_u128(bytes, 0),
        user_data_64: le_u64(bytes, 16),
        user_data_32: le_u32(bytes, 24),
        ledger: le_u32(bytes, 28),
        code: u16::from_le_bytes(bytes[32..34].try_into().expect("2 bytes")),
        reserved,
        timestamp_min: le_u64(bytes, 40),
        timestamp_max: le_u64(bytes, 48),
        limit: le_u32(bytes, 56),
        flags: QueryFilterFlags::from_raw(le_u32(bytes, 60)),
    })
}

/// Validate an [`AccountFilter`] (upstream `get_scan_from_account_filter`,
/// `state_machine.zig:1737`): a query must name a non-zero, non-`MAX` account,
/// have at least one of `debits`/`credits` set, no padding or reserved bits,
/// a non-zero `limit`, and a well-formed timestamp range.
#[must_use]
fn account_filter_valid(filter: &AccountFilter) -> bool {
    let timestamp_valid = |t: u64| (1..=i64::MAX as u64).contains(&t);
    filter.account_id != 0
        && filter.account_id != u128::MAX
        && (filter.timestamp_min == 0 || timestamp_valid(filter.timestamp_min))
        && (filter.timestamp_max == 0 || timestamp_valid(filter.timestamp_max))
        && (filter.timestamp_max == 0 || filter.timestamp_min <= filter.timestamp_max)
        && filter.limit != 0
        && (filter.flags.credits() || filter.flags.debits())
        && !filter.flags.has_padding()
        && filter.reserved.iter().all(|&b| b == 0)
}

/// Whether a committed transfer satisfies the account-filter conditions
/// (the account as its debit and/or credit side per the flags, plus the
/// `user_data_*`/`code` equality filters; a zero filter value means "no
/// filter", upstream `state_machine.zig:1771-1802`).
#[must_use]
fn transfer_matches_account_filter(t: &Transfer, filter: &AccountFilter) -> bool {
    let side_matches = (filter.flags.debits() && t.debit_account_id == filter.account_id)
        || (filter.flags.credits() && t.credit_account_id == filter.account_id);
    if !side_matches {
        return false;
    }
    (filter.user_data_128 == 0 || t.user_data_128 == filter.user_data_128)
        && (filter.user_data_64 == 0 || t.user_data_64 == filter.user_data_64)
        && (filter.user_data_32 == 0 || t.user_data_32 == filter.user_data_32)
        && (filter.code == 0 || t.code == filter.code)
}

/// Validate a [`QueryFilter`] (upstream `get_scan_from_query_filter`,
/// `state_machine.zig:2062`): a query needs a non-zero `limit`, a well-formed
/// timestamp range, no padding or reserved bits, and — unlike
/// [`AccountFilter`] — does not require any user_data/ledger/code value.
#[must_use]
fn query_filter_valid(filter: &QueryFilter) -> bool {
    let timestamp_valid = |t: u64| (1..=i64::MAX as u64).contains(&t);
    (filter.timestamp_min == 0 || timestamp_valid(filter.timestamp_min))
        && (filter.timestamp_max == 0 || timestamp_valid(filter.timestamp_max))
        && (filter.timestamp_max == 0 || filter.timestamp_min <= filter.timestamp_max)
        && filter.limit != 0
        && !filter.flags.has_padding()
        && filter.reserved.iter().all(|&b| b == 0)
}

/// Whether an object carrying a `user_data_128/64/32/ledger/code/timestamp`
/// satisfies the query filter's AND conditions (each non-zero filter value is
/// an equality; upstream `state_machine.zig:2086-2107`).
#[must_use]
fn query_matches(
    user_data_128: u128,
    user_data_64: u64,
    user_data_32: u32,
    ledger: u32,
    code: u16,
    timestamp: u64,
    min: u64,
    max: u64,
    filter: &QueryFilter,
) -> bool {
    timestamp >= min
        && timestamp <= max
        && (filter.user_data_128 == 0 || user_data_128 == filter.user_data_128)
        && (filter.user_data_64 == 0 || user_data_64 == filter.user_data_64)
        && (filter.user_data_32 == 0 || user_data_32 == filter.user_data_32)
        && (filter.ledger == 0 || ledger == filter.ledger)
        && (filter.code == 0 || code == filter.code)
}

/// Encode a [`ChangeEvent`] as a 384-byte reply record (LE).
///
/// Upstream `ChangeEvent` (`tigerbeetle.zig:622`) layout; the `reserved` and
/// flag fields occupy fixed offsets so the wire order must be explicit.
#[must_use]
pub fn change_bytes(event: &ChangeEvent) -> [u8; 384] {
    let mut r = [0_u8; 384];
    let mut put = |off: usize, bytes: &[u8]| r[off..off + bytes.len()].copy_from_slice(bytes);
    put(0, &event.transfer_id.to_le_bytes());
    put(16, &event.transfer_amount.to_le_bytes());
    put(32, &event.transfer_pending_id.to_le_bytes());
    put(48, &event.transfer_user_data_128.to_le_bytes());
    put(64, &event.transfer_user_data_64.to_le_bytes());
    put(72, &event.transfer_user_data_32.to_le_bytes());
    put(76, &event.transfer_timeout.to_le_bytes());
    put(80, &event.transfer_code.to_le_bytes());
    put(82, &event.transfer_flags.as_raw().to_le_bytes());
    put(84, &event.ledger.to_le_bytes());
    put(88, &[event.r#type as u8]);
    put(89, &event.reserved);
    put(128, &event.debit_account_id.to_le_bytes());
    put(144, &event.debit_account_debits_pending.to_le_bytes());
    put(160, &event.debit_account_debits_posted.to_le_bytes());
    put(176, &event.debit_account_credits_pending.to_le_bytes());
    put(192, &event.debit_account_credits_posted.to_le_bytes());
    put(208, &event.debit_account_user_data_128.to_le_bytes());
    put(224, &event.debit_account_user_data_64.to_le_bytes());
    put(232, &event.debit_account_user_data_32.to_le_bytes());
    put(236, &event.debit_account_code.to_le_bytes());
    put(238, &event.debit_account_flags.as_raw().to_le_bytes());
    put(240, &event.credit_account_id.to_le_bytes());
    put(256, &event.credit_account_debits_pending.to_le_bytes());
    put(272, &event.credit_account_debits_posted.to_le_bytes());
    put(288, &event.credit_account_credits_pending.to_le_bytes());
    put(304, &event.credit_account_credits_posted.to_le_bytes());
    put(320, &event.credit_account_user_data_128.to_le_bytes());
    put(336, &event.credit_account_user_data_64.to_le_bytes());
    put(344, &event.credit_account_user_data_32.to_le_bytes());
    put(348, &event.credit_account_code.to_le_bytes());
    put(350, &event.credit_account_flags.as_raw().to_le_bytes());
    put(352, &event.timestamp.to_le_bytes());
    put(360, &event.transfer_timestamp.to_le_bytes());
    put(368, &event.debit_account_timestamp.to_le_bytes());
    put(376, &event.credit_account_timestamp.to_le_bytes());
    r
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
#[derive(Debug)]
pub struct StateMachine {
    /// The timestamp of the last op committed to the state machine; every
    /// executed op must advance it strictly (upstream asserts the same in
    /// `Replica.execute_op`, replica.zig:5441).
    pub commit_timestamp: u64,
    /// Temporary primary-key store for accounts.
    accounts: HashMap<u128, Account>,
    /// Temporary primary-key store for transfers.
    transfers: HashMap<u128, Transfer>,
    /// Transfer ids that failed with a transient status and therefore can
    /// never be created again (strong idempotency). Mirrors upstream's
    /// transfers-groove orphaned primary-key set (`insert_orphaned_primary_key`,
    /// `state_machine.zig:3248`); `create_transfer` reports `id_already_failed`
    /// for ids in this set (`state_machine.zig:3736`), and it is disjoint from
    /// [`Self::transfers`] because a transient failure never commits a record.
    transfers_orphaned: HashSet<u128>,
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
    /// and updated when the pending is posted, voided, or expired. Mirrors
    /// upstream's `transfers_pending.objects` groove.
    transfers_pending: HashMap<u64, TransferPending>,
    /// Change-data-capture store of account balance changes, keyed by event
    /// timestamp. Mirrors upstream's `account_events.objects` groove (written
    /// from [`Self::account_event`], read by [`Self::get_change_events`]).
    account_events: HashMap<u64, AccountEvent>,
    /// CDC index: account timestamp → event timestamps in increasing order,
    /// for accounts with `AccountFlags::HISTORY`. Mirrors upstream's
    /// `account_events.indexes.account_timestamp` derived index.
    account_events_index: HashMap<u64, Vec<u64>>,
    /// The timestamp of the next pending transfer to expire, or the sentinel
    /// that tells the driver whether a `pulse` is needed at any op timestamp
    /// (upstream `expire_pending_transfers.pulse_next_timestamp`,
    /// `state_machine.zig:4920`).
    ///
    /// Starts at [`TimestampRange::TIMESTAMP_MIN`] so the first pulse always
    /// runs (and finds any pre-existing expired pendings); becomes
    /// `timestamp_max` when there is nothing left to expire.
    pulse_next_timestamp: u64,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self {
            commit_timestamp: 0,
            accounts: HashMap::new(),
            transfers: HashMap::new(),
            transfers_orphaned: HashSet::new(),
            accounts_timestamp_max: 0,
            transfers_timestamp_max: 0,
            accounts_by_timestamp: HashMap::new(),
            transfers_by_timestamp: HashMap::new(),
            transfers_pending: HashMap::new(),
            account_events: HashMap::new(),
            account_events_index: HashMap::new(),
            pulse_next_timestamp: TimestampRange::TIMESTAMP_MIN,
        }
    }
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
            Operation::STATE_MACHINE_PULSE => {
                if !body.is_empty() {
                    unreachable!(
                        "pulse must carry an empty body (Operation.valid, state_machine.zig:1044)"
                    );
                }
                self.expire_pending_transfers(timestamp);
                Vec::new()
            }
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
            Operation::GET_CHANGE_EVENTS => {
                let Some(filter) = bytes_to_change_events_filter(body) else {
                    unreachable!("get_change_events body must be a single ChangeEventsFilter");
                };
                self.get_change_events(&filter)
            }
            Operation::LOOKUP_ACCOUNTS => {
                let Some(ids) = bytes_to_lookup_ids(body) else {
                    unreachable!("lookup_accounts body must encode whole u128 ids");
                };
                self.lookup_accounts(&ids)
            }
            Operation::LOOKUP_TRANSFERS => {
                let Some(ids) = bytes_to_lookup_ids(body) else {
                    unreachable!("lookup_transfers body must encode whole u128 ids");
                };
                self.lookup_transfers(&ids)
            }
            Operation::GET_ACCOUNT_TRANSFERS => {
                let Some(filter) = bytes_to_account_filter(body) else {
                    unreachable!("get_account_transfers body must be a single AccountFilter");
                };
                self.get_account_transfers(&filter)
            }
            Operation::GET_ACCOUNT_BALANCES => {
                let Some(filter) = bytes_to_account_filter(body) else {
                    unreachable!("get_account_balances body must be a single AccountFilter");
                };
                self.get_account_balances(&filter)
            }
            Operation::QUERY_ACCOUNTS => {
                let Some(filter) = bytes_to_query_filter(body) else {
                    unreachable!("query_accounts body must be a single QueryFilter");
                };
                self.query_accounts(&filter)
            }
            Operation::QUERY_TRANSFERS => {
                let Some(filter) = bytes_to_query_filter(body) else {
                    unreachable!("query_transfers body must be a single QueryFilter");
                };
                self.query_transfers(&filter)
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
            |id| self.transfers_orphaned.contains(&id),
        );
        self.persist_transfers(events, &results, timestamp);
        transfer_results_to_bytes(&results)
    }

    /// Execute a `lookup_accounts` query and return the reply body (upstream
    /// `execute_lookup_accounts`, `state_machine.zig:3255`).
    ///
    /// The request is a batch of account ids (`u128` each). The reply is the
    /// compact list of the accounts that exist, in request order (ids that do
    /// not exist are omitted). This is a read operation and does not advance
    /// `commit_timestamp`.
    ///
    /// DEVIATION: upstream bounds the reply by the message body — upstream
    /// `execute_lookup_accounts` writes into a reply buffer of the request
    /// size and asserts `events.len <= results.len` (state_machine.zig:3266),
    /// so it cannot return more objects than fit a message. Sans-IO there is no
    /// message buffer; the results are unbounded by request size.
    #[must_use]
    pub fn lookup_accounts(&self, ids: &[u128]) -> Vec<u8> {
        let mut results: Vec<Account> = Vec::new();
        for &id in ids {
            if let Some(account) = self.accounts.get(&id) {
                results.push(*account);
            }
        }
        account_batch_to_bytes(&results)
    }

    /// Execute a `lookup_transfers` query and return the reply body (upstream
    /// `execute_lookup_transfers`, `state_machine.zig:3275`).
    ///
    /// The request is a batch of transfer ids (`u128` each). The reply is the
    /// compact list of the transfers that exist, in request order (ids that do
    /// not exist are omitted). This is a read operation and does not advance
    /// `commit_timestamp`.
    ///
    /// DEVIATION: as for [`Self::lookup_accounts`], upstream only guarantees a
    /// reply that fits one message (state_machine.zig:3286); sans-IO the
    /// results are unbounded by request size.
    #[must_use]
    pub fn lookup_transfers(&self, ids: &[u128]) -> Vec<u8> {
        let mut results: Vec<Transfer> = Vec::new();
        for &id in ids {
            if let Some(transfer) = self.transfers.get(&id) {
                results.push(*transfer);
            }
        }
        transfer_batch_to_bytes(&results)
    }

    /// Execute a `get_change_events` query and return the reply body.
    ///
    /// Sweeps the change-data-capture events whose timestamps fall in
    /// `[timestamp_min, timestamp_max]` (a zero bound is unbounded) in
    /// ascending order, up to `filter.limit` events, and encodes each as a
    /// [`ChangeEvent`] (upstream `execute_get_change_events`,
    /// `state_machine.zig:3395`).
    ///
    /// Every recorded event references a committed transfer; a post/void
    /// resolves its referenced pending transfer from the committed store.
    ///
    /// DEVIATION: upstream drives this through an async scan over the
    /// `account_events` object groove plus account/transfer prefetch
    /// (state_machine.zig:2198-2380). Sans-IO the store is a plain map and the
    /// scan is synchronous; the produced [`ChangeEvent`]s are identical.
    #[must_use]
    pub fn get_change_events(&self, filter: &ChangeEventsFilter) -> Vec<u8> {
        debug_assert!(
            filter.reserved.iter().all(|&b| b == 0),
            "reserved bits must be zero: filter validated upstream"
        );
        let min = if filter.timestamp_min == 0 { u64::MIN } else { filter.timestamp_min };
        let max = if filter.timestamp_max == 0 { u64::MAX } else { filter.timestamp_max };
        assert!(min <= max, "timestamp_max must not be less than timestamp_min");
        assert!(filter.limit != 0, "limit must not be zero");

        let mut events: Vec<&AccountEvent> = self
            .account_events
            .iter()
            .filter(|&(&ts, _)| ts >= min && ts <= max)
            .map(|(_, e)| e)
            .collect();
        events.sort_unstable_by_key(|e| e.timestamp);
        // Upstream `prefetch_get_change_events` caps the reply at
        // `min(filter.limit, limit_max)` (state_machine.zig:2245-2275).
        events
            .truncate((filter.limit as usize).min(result_max(core::mem::size_of::<ChangeEvent>())));
        let mut reply = Vec::with_capacity(events.len() * core::mem::size_of::<ChangeEvent>());
        for result in events {
            let change = self.build_change_event(result);
            reply.extend_from_slice(&change_bytes(&change));
        }
        reply
    }

    /// The committed transfer a recorded [`AccountEvent`] refers to. Resolved by
    /// timestamp, except for expiry events, which carry no transfer with the
    /// event's timestamp and are resolved by `transfer_pending_id` (upstream
    /// `state_machine.zig:3428-3439`).
    fn account_event_transfer(&self, result: &AccountEvent) -> Option<Transfer> {
        match result.transfer_pending_status {
            TransferPendingStatus::Expired => {
                self.transfers.get(&result.transfer_pending_id).copied()
            }
            _ => self
                .transfers_by_timestamp
                .get(&result.timestamp)
                .and_then(|id| self.transfers.get(id))
                .copied(),
        }
    }

    /// Execute a `get_account_transfers` query and return the reply body
    /// (upstream `execute_get_account_transfers`, `state_machine.zig:3294`).
    ///
    /// Returns the transfers involving `filter.account_id` (as debit and/or
    /// credit per the filter flags) whose timestamp falls in the (inclusive)
    /// `[timestamp_min, timestamp_max]` range, optionally filtered by the
    /// transfer's `user_data_*`/`code`, up to `filter.limit`, in ascending or
    /// reverse-chronological order per `reversed`. A malformed filter or one
    /// naming account id `MAX` yields an empty reply (upstream returns the
    /// results but treats the filter as invalid).
    ///
    /// DEVIATION: upstream scans the `transfers` object groove via the
    /// `debit_account_id`/`credit_account_id`/`user_data_*`/`code` derived
    /// indexes (`get_scan_from_account_filter`); sans-IO the committed
    /// transfers are filtered in memory.
    #[must_use]
    pub fn get_account_transfers(&self, filter: &AccountFilter) -> Vec<u8> {
        if !account_filter_valid(filter) {
            return Vec::new();
        }

        let min = if filter.timestamp_min == 0 { u64::MIN } else { filter.timestamp_min };
        let max = if filter.timestamp_max == 0 { u64::MAX } else { filter.timestamp_max };

        let mut results: Vec<Transfer> = self
            .transfers
            .values()
            .copied()
            .filter(|t| {
                t.timestamp >= min
                    && t.timestamp <= max
                    && transfer_matches_account_filter(t, filter)
            })
            .collect();
        if filter.flags.reversed() {
            results.sort_unstable_by_key(|t| std::cmp::Reverse(t.timestamp));
        } else {
            results.sort_unstable_by_key(|t| t.timestamp);
        }
        // Upstream `prefetch_get_account_transfers` caps the reply at
        // `min(filter.limit, result_max(message_body_size_max))`
        // (state_machine.zig:1641-1648).
        results.truncate(
            (filter.limit as usize).min(result_max_multi_batch(core::mem::size_of::<Transfer>())),
        );
        transfer_batch_to_bytes(&results)
    }

    /// Execute a `get_account_balances` query and return the reply body
    /// (upstream `execute_get_account_balances`, `state_machine.zig:3312`).
    ///
    /// Scans the change-data-capture history for `filter.account_id` in the
    /// (inclusive) `[timestamp_min, timestamp_max]` range, up to `filter.limit`,
    /// in ascending or reverse-chronological order per `reversed`, and returns
    /// one [`AccountBalance`] snapshot per event for the account's side. The
    /// account must exist and advertise `AccountFlags::HISTORY`; otherwise the
    /// reply is empty.
    ///
    /// DEVIATION: upstream scans the `account_events` groove's
    /// `account_timestamp` derived index built from the transfers-groove scan
    /// conditions — events are yielded only when a *transfer* exists at the
    /// event's timestamp (`AccountBalancesScanLookup`), so expiry events,
    /// which carry no transfer at their own timestamp, never appear in the
    /// balance history; sans-IO the events are filtered in memory (including
    /// excluding `Expired` events and resolving an event's originating
    /// transfer to apply the `user_data_*`/`code` filters).
    #[must_use]
    pub fn get_account_balances(&self, filter: &AccountFilter) -> Vec<u8> {
        if !account_filter_valid(filter) {
            return Vec::new();
        }
        let Some(account) = self.accounts.get(&filter.account_id) else {
            return Vec::new();
        };
        if !account.flags.history() {
            return Vec::new();
        }

        let min = if filter.timestamp_min == 0 { u64::MIN } else { filter.timestamp_min };
        let max = if filter.timestamp_max == 0 { u64::MAX } else { filter.timestamp_max };

        let mut events: Vec<&AccountEvent> = self
            .account_events
            .iter()
            .filter(|(ts, e)| {
                **ts >= min
                    && **ts <= max
                    && e.transfer_pending_status != TransferPendingStatus::Expired
                    && self
                        .account_event_transfer(e)
                        .is_some_and(|t| transfer_matches_account_filter(&t, filter))
            })
            .map(|(_, e)| e)
            .collect();
        if filter.flags.reversed() {
            events.sort_unstable_by_key(|e| std::cmp::Reverse(e.timestamp));
        } else {
            events.sort_unstable_by_key(|e| e.timestamp);
        }
        // Upstream `prefetch_get_account_balances` caps the reply at
        // `min(filter.limit, result_max(message_body_size_max))`, where
        // `result_max` divides the body by the RESULT size — `AccountBalance`
        // (128 bytes; 4032/128 = 31 in `test_min`), not the `AccountEvent`
        // (256) events are scanned from (state_machine.zig:1641-1648).
        events.truncate(
            (filter.limit as usize)
                .min(result_max_multi_batch(core::mem::size_of::<AccountBalance>())),
        );

        let mut results: Vec<AccountBalance> = Vec::with_capacity(events.len());
        for event in events {
            assert_ne!(event.dr_account_id, event.cr_account_id);
            let snapshot = if filter.account_id == event.dr_account_id {
                AccountBalance {
                    timestamp: event.timestamp,
                    debits_pending: event.dr_debits_pending,
                    debits_posted: event.dr_debits_posted,
                    credits_pending: event.dr_credits_pending,
                    credits_posted: event.dr_credits_posted,
                    reserved: [0; 56],
                }
            } else {
                assert_eq!(filter.account_id, event.cr_account_id);
                AccountBalance {
                    timestamp: event.timestamp,
                    debits_pending: event.cr_debits_pending,
                    debits_posted: event.cr_debits_posted,
                    credits_pending: event.cr_credits_pending,
                    credits_posted: event.cr_credits_posted,
                    reserved: [0; 56],
                }
            };
            results.push(snapshot);
        }
        account_balances_to_bytes(&results)
    }

    /// Execute a `query_accounts` operation and return the reply body
    /// (upstream `execute_query_accounts`, `state_machine.zig:3359`).
    ///
    /// Returns the accounts whose timestamp falls in the (inclusive)
    /// `[timestamp_min, timestamp_max]` range (a zero bound is unbounded) and
    /// that satisfy the AND of the non-zero `user_data_128`/`user_data_64`/
    /// `user_data_32`/`ledger`/`code` equality filters, up to `filter.limit`,
    /// in ascending or reverse-chronological order per `reversed`.
    ///
    /// DEVIATION: upstream scans the `accounts` object groove via the
    /// `user_data_*`/`ledger`/`code` derived index prefixes intersected with
    /// the timestamp range (`get_scan_from_query_filter`); sans-IO the
    /// committed accounts are filtered in memory.
    #[must_use]
    pub fn query_accounts(&self, filter: &QueryFilter) -> Vec<u8> {
        if !query_filter_valid(filter) {
            return Vec::new();
        }
        let min = if filter.timestamp_min == 0 { u64::MIN } else { filter.timestamp_min };
        let max = if filter.timestamp_max == 0 { u64::MAX } else { filter.timestamp_max };

        let mut results: Vec<Account> = self
            .accounts
            .values()
            .copied()
            .filter(|a| {
                query_matches(
                    a.user_data_128,
                    a.user_data_64,
                    a.user_data_32,
                    a.ledger,
                    a.code,
                    a.timestamp,
                    min,
                    max,
                    filter,
                )
            })
            .collect();
        if filter.flags.reversed() {
            results.sort_unstable_by_key(|a| std::cmp::Reverse(a.timestamp));
        } else {
            results.sort_unstable_by_key(|a| a.timestamp);
        }
        // Upstream `prefetch_query_accounts_scan` caps the reply at
        // `min(filter.limit, result_max)` (state_machine.zig:1887-1894).
        results.truncate(
            (filter.limit as usize).min(result_max_multi_batch(core::mem::size_of::<Account>())),
        );
        account_batch_to_bytes(&results)
    }

    /// Execute a `query_transfers` operation and return the reply body
    /// (upstream `execute_query_transfers`, `state_machine.zig:3377`).
    ///
    /// Returns the committed transfers whose timestamp falls in the
    /// (inclusive) `[timestamp_min, timestamp_max]` range (a zero bound is
    /// unbounded) and that satisfy the AND of the non-zero
    /// `user_data_128`/`user_data_64`/`user_data_32`/`ledger`/`code` equality
    /// filters, up to `filter.limit`, in ascending or reverse-chronological
    /// order per `reversed`.
    ///
    /// DEVIATION: upstream scans the `transfers` object groove via the derived
    /// index prefixes (`get_scan_from_query_filter`); sans-IO the committed
    /// transfers are filtered in memory.
    #[must_use]
    pub fn query_transfers(&self, filter: &QueryFilter) -> Vec<u8> {
        if !query_filter_valid(filter) {
            return Vec::new();
        }
        let min = if filter.timestamp_min == 0 { u64::MIN } else { filter.timestamp_min };
        let max = if filter.timestamp_max == 0 { u64::MAX } else { filter.timestamp_max };

        let mut results: Vec<Transfer> = self
            .transfers
            .values()
            .copied()
            .filter(|t| {
                query_matches(
                    t.user_data_128,
                    t.user_data_64,
                    t.user_data_32,
                    t.ledger,
                    t.code,
                    t.timestamp,
                    min,
                    max,
                    filter,
                )
            })
            .collect();
        if filter.flags.reversed() {
            results.sort_unstable_by_key(|t| std::cmp::Reverse(t.timestamp));
        } else {
            results.sort_unstable_by_key(|t| t.timestamp);
        }
        results.truncate(
            (filter.limit as usize).min(result_max_multi_batch(core::mem::size_of::<Transfer>())),
        );
        transfer_batch_to_bytes(&results)
    }

    /// Build the [`ChangeEvent`] reported for a recorded account event
    /// (upstream `get_change_event`, `state_machine.zig:3424-3527`).
    #[must_use]
    fn build_change_event(&self, result: &AccountEvent) -> ChangeEvent {
        let transfer = self
            .account_event_transfer(result)
            .expect("each account event references a committed transfer");
        let dr_account =
            self.accounts.get(&result.dr_account_id).copied().expect("debit account committed");
        let cr_account =
            self.accounts.get(&result.cr_account_id).copied().expect("credit account committed");
        assert_eq!(transfer.debit_account_id, dr_account.id);
        assert_eq!(transfer.credit_account_id, cr_account.id);
        assert_eq!(transfer.ledger, result.ledger);
        assert_eq!(dr_account.ledger, result.ledger);
        assert_eq!(cr_account.ledger, result.ledger);

        // For expiry events the event timestamp carries no transfer, but expiry
        // writes are deferred; every recorded event here is timestamp-linked.
        let event_type: ChangeEventType = match result.transfer_pending_status {
            TransferPendingStatus::None => {
                assert_eq!(transfer.timestamp, result.timestamp);
                assert!(!transfer.flags.pending());
                assert!(!transfer.flags.post_pending_transfer());
                assert!(!transfer.flags.void_pending_transfer());
                assert_eq!(transfer.pending_id, 0);
                ChangeEventType::SinglePhase
            }
            TransferPendingStatus::Pending => {
                assert_eq!(transfer.timestamp, result.timestamp);
                assert!(transfer.flags.pending());
                assert_eq!(transfer.pending_id, 0);
                ChangeEventType::TwoPhasePending
            }
            TransferPendingStatus::Posted => {
                assert_eq!(transfer.timestamp, result.timestamp);
                assert!(transfer.flags.post_pending_transfer());
                assert_eq!(transfer.pending_id, result.transfer_pending_id);
                ChangeEventType::TwoPhasePosted
            }
            TransferPendingStatus::Voided => {
                assert_eq!(transfer.timestamp, result.timestamp);
                assert!(transfer.flags.void_pending_transfer());
                assert_eq!(transfer.pending_id, result.transfer_pending_id);
                ChangeEventType::TwoPhaseVoided
            }
            TransferPendingStatus::Expired => {
                assert!(transfer.flags.pending());
                assert_eq!(transfer.id, result.transfer_pending_id);
                assert!(transfer.timeout > 0);
                assert!(transfer.timestamp < result.timestamp);
                ChangeEventType::TwoPhaseExpired
            }
        };

        ChangeEvent {
            transfer_id: transfer.id,
            transfer_amount: result.amount,
            transfer_pending_id: transfer.pending_id,
            transfer_user_data_128: transfer.user_data_128,
            transfer_user_data_64: transfer.user_data_64,
            transfer_user_data_32: transfer.user_data_32,
            transfer_timeout: transfer.timeout,
            transfer_code: transfer.code,
            transfer_flags: transfer.flags,
            ledger: result.ledger,
            r#type: event_type,
            reserved: [0; 39],
            debit_account_id: dr_account.id,
            debit_account_debits_pending: result.dr_debits_pending,
            debit_account_debits_posted: result.dr_debits_posted,
            debit_account_credits_pending: result.dr_credits_pending,
            debit_account_credits_posted: result.dr_credits_posted,
            debit_account_user_data_128: dr_account.user_data_128,
            debit_account_user_data_64: dr_account.user_data_64,
            debit_account_user_data_32: dr_account.user_data_32,
            debit_account_code: dr_account.code,
            debit_account_flags: result.dr_account_flags,
            credit_account_id: cr_account.id,
            credit_account_debits_pending: result.cr_debits_pending,
            credit_account_debits_posted: result.cr_debits_posted,
            credit_account_credits_pending: result.cr_credits_pending,
            credit_account_credits_posted: result.cr_credits_posted,
            credit_account_user_data_128: cr_account.user_data_128,
            credit_account_user_data_64: cr_account.user_data_64,
            credit_account_user_data_32: cr_account.user_data_32,
            credit_account_code: cr_account.code,
            credit_account_flags: result.cr_account_flags,
            timestamp: result.timestamp,
            transfer_timestamp: transfer.timestamp,
            debit_account_timestamp: dr_account.timestamp,
            credit_account_timestamp: cr_account.timestamp,
        }
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
        // Upstream records transient failures in the transfers groove's
        // orphaned primary-key set (`transient_error`,
        // `state_machine.zig:3215-3252`), outside the chain scope — so the
        // orphan applies even when the failed event belonged to a chain that
        // was rolled back. Replay it over the full event list before the
        // created-only walk below.
        for (event, result) in events.iter().zip(results) {
            if result.status.transient() {
                self.transfers_orphaned.insert(event.id);
            }
        }

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
        // The amount the user requested (upstream `t.amount`); recorded in the
        // CDC event even when balancing or posting reduced the applied amount.
        let amount_requested = event.amount;
        if event.flags.pending() {
            event.amount = amount_actual;
            self.insert_transfer(event.id, event);
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
            self.commit_transfer(&event, amount_actual, amount_requested);
        } else if event.flags.post_pending_transfer() || event.flags.void_pending_transfer() {
            self.persist_post_void(event, amount_actual);
        } else {
            event.amount = amount_actual;
            self.insert_transfer(event.id, event);
            self.commit_transfer(&event, amount_actual, amount_requested);
        }
    }

    /// Store a posting/voiding transfer and apply its balance mutation
    /// (upstream `commit_transfer` for post/void, `state_machine.zig:4240-4298`).
    ///
    /// The stored record folds the pending transfer's debit/credit accounts,
    /// ledger, code and user-data fallbacks; the pending status index advances
    /// to `Posted`/`Voided`; and the accounts release the pending holds
    /// ([`commit_post_void_accounts`]).
    fn persist_post_void(&mut self, event: Transfer, amount_actual: u128) {
        let p =
            self.transfers.get(&event.pending_id).copied().expect(
                "posting/voiding a pending transfer implies the pending transfer is committed",
            );
        let amount_requested = event.amount;
        let record = fold_post_void_transfer(&event, &p, amount_actual, event.timestamp);
        self.insert_transfer(record.id, record);

        let transfer_pending = self
            .transfers_pending
            .get_mut(&p.timestamp)
            .expect("posting/voiding a pending transfer implies its pending status row");
        assert_eq!(transfer_pending.status, TransferPendingStatus::Pending);
        transfer_pending.status = if record.flags.post_pending_transfer() {
            TransferPendingStatus::Posted
        } else {
            TransferPendingStatus::Voided
        };
        let pending_status = transfer_pending.status;

        let dr = self
            .accounts
            .get(&record.debit_account_id)
            .copied()
            .expect("debit account exists: post_or_void_pending_transfer validated it");
        let cr = self
            .accounts
            .get(&record.credit_account_id)
            .copied()
            .expect("credit account exists: post_or_void_pending_transfer validated it");
        let (dr, cr) = commit_post_void_accounts(&record, &p, amount_actual, &dr, &cr);
        self.accounts.insert(dr.id, dr);
        self.accounts.insert(cr.id, cr);
        self.account_event(
            event.timestamp,
            &dr,
            &cr,
            record.flags,
            pending_status,
            Some(&p),
            amount_requested,
            amount_actual,
        );

        // The posted/voided pending is no longer outstanding; if it was the
        // next due, re-point the pulse (upstream resets to `timestamp_min` to
        // force a rescan, `state_machine.zig:4226-4230`).
        self.update_pulse_next_timestamp();
    }

    /// Record a change-data-capture event for a committed transfer mutation
    /// (upstream `state_machine.zig:4384-4465`).
    ///
    /// `dr_account` and `cr_account` are the accounts *after* the mutation
    /// applies. `transfer_pending_status` describes the event (`None` for a
    /// plain transfer, `Pending`/`Posted`/`Voided` for the corresponding
    /// pending-transfer lifecycle); `transfer_pending` is the referenced
    /// pending transfer for post/void (and later, expiry) events.
    ///
    /// The event is inserted unconditionally (CDC is written regardless of
    /// `AccountFlags::HISTORY`); the account index is populated only for
    /// accounts advertising `HISTORY`.
    fn account_event(
        &mut self,
        timestamp_event: u64,
        dr_account: &Account,
        cr_account: &Account,
        transfer_flags: TransferFlags,
        transfer_pending_status: TransferPendingStatus,
        transfer_pending: Option<&Transfer>,
        amount_requested: u128,
        amount: u128,
    ) {
        assert!(timestamp_event > 0);
        match transfer_pending_status {
            TransferPendingStatus::None | TransferPendingStatus::Pending => {
                assert!(transfer_pending.is_none());
            }
            TransferPendingStatus::Posted
            | TransferPendingStatus::Voided
            | TransferPendingStatus::Expired => {
                assert!(transfer_pending.is_some());
            }
        }
        assert_eq!(dr_account.ledger, cr_account.ledger);

        let transfer_pending = transfer_pending.copied();
        let event = AccountEvent {
            dr_account_id: dr_account.id,
            dr_debits_pending: dr_account.debits_pending,
            dr_debits_posted: dr_account.debits_posted,
            dr_credits_pending: dr_account.credits_pending,
            dr_credits_posted: dr_account.credits_posted,
            cr_account_id: cr_account.id,
            cr_debits_pending: cr_account.debits_pending,
            cr_debits_posted: cr_account.debits_posted,
            cr_credits_pending: cr_account.credits_pending,
            cr_credits_posted: cr_account.credits_posted,
            timestamp: timestamp_event,
            dr_account_timestamp: dr_account.timestamp,
            cr_account_timestamp: cr_account.timestamp,
            dr_account_flags: dr_account.flags,
            cr_account_flags: cr_account.flags,
            transfer_flags,
            transfer_pending_flags: transfer_pending.map_or(TransferFlags::default(), |p| p.flags),
            transfer_pending_id: transfer_pending.map_or(0, |p| p.id),
            amount_requested,
            amount,
            ledger: dr_account.ledger,
            transfer_pending_status,
            reserved: [0; 11],
        };
        assert!(
            self.account_events.insert(timestamp_event, event).is_none(),
            "each account event has a unique timestamp"
        );

        if dr_account.flags.history() {
            self.account_events_index
                .entry(dr_account.timestamp)
                .or_default()
                .push(timestamp_event);
        }
        if cr_account.flags.history() {
            self.account_events_index
                .entry(cr_account.timestamp)
                .or_default()
                .push(timestamp_event);
        }
    }

    /// Whether a `pulse` operation is needed at `timestamp` (upstream
    /// `StateMachine.pulse_needed`, `state_machine.zig:1138`): a pending
    /// transfer is already due (or a scan is still owed), so the driver calls
    /// [`Self::expire_pending_transfers`] at `timestamp`.
    #[must_use]
    pub fn pulse_needed(&self, timestamp: u64) -> bool {
        self.pulse_next_timestamp <= timestamp
    }

    /// The next timestamp at which a pending transfer expires (upstream
    /// `expire_pending_transfers.pulse_next_timestamp`, `state_machine.zig:4920`):
    /// `timestamp_min` while a pulse is owed, `timestamp_max` when nothing is
    /// due, otherwise the soonest `expires_at`.
    #[must_use]
    pub fn pulse_next_timestamp(&self) -> u64 {
        self.pulse_next_timestamp
    }

    /// Re-derive `pulse_next_timestamp` from the outstanding pending transfers:
    /// the soonest `expires_at` among remaining `Pending` transfers, or
    /// `timestamp_max` when none remain.
    ///
    /// DEVIATION: upstream fixes up the value incrementally (min-update on
    /// creation, `state_machine.zig:3979-3980`; reset to `timestamp_min` on
    /// post/void when the removed pending was the next due, `state_machine.zig`
    /// `:4227`), driving the `expires_at` index scan on the next pulse. Sans-IO
    /// the pending map is authoritative and cheap to scan, so recomputing is
    /// exact and always leaves `pulse_needed` pointing at the true next expiry.
    fn update_pulse_next_timestamp(&mut self) {
        self.pulse_next_timestamp = TimestampRange::TIMESTAMP_MAX;
        for (&pending_ts, pending_status) in &self.transfers_pending {
            if pending_status.status != TransferPendingStatus::Pending {
                continue;
            }
            let id = self
                .transfers_by_timestamp
                .get(&pending_ts)
                .copied()
                .expect("the pending status index implies a committed pending transfer");
            let p = self
                .transfers
                .get(&id)
                .copied()
                .expect("the pending status index implies a committed pending transfer");
            assert!(p.flags.pending());
            assert!(p.timeout > 0);
            let expires_at = p.timestamp + p.timeout_ns();
            self.pulse_next_timestamp = self.pulse_next_timestamp.min(expires_at);
        }
    }

    /// Expire pending transfers whose timeout has elapsed by `timestamp`
    /// (upstream `execute_expire_pending_transfers`, `state_machine.zig:4511`).
    ///
    /// Upstream is driven by a pulse: a derived `expires_at` index pairs each
    /// pending transfer's timestamp with `expires_at = timestamp + timeout_ns`,
    /// and the next pulse fires at the soonest expiry. Sans-IO there is no beat
    /// loop yet, so this consumes the same batching of due pendings directly:
    /// every pending with `status == Pending` and `expires_at <= timestamp` is
    /// expired in ascending `expires_at` order (mirroring the index scan),
    /// `commit_timestamp` advances by one synthetic timestamp per expired
    /// transfer, pending balances are returned to the accounts' pools
    /// (`closing_debit`/`closing_credit` accounts are reopened), the
    /// `transfers_pending` status becomes `Expired`, and an expired
    /// [`AccountEvent`] is recorded.
    ///
    /// DEVIATION: `expire_pending_transfers` is invoked directly with the next
    /// expiry timestamp rather than through the VSR pulse/beat machinery and
    /// the `expires_at` derived index; the caller is responsible for calling it
    /// once the next pending is due. This operation produces no reply body.
    pub fn expire_pending_transfers(&mut self, timestamp: u64) {
        assert!(
            timestamp > self.commit_timestamp,
            "expiry must advance commit_timestamp (upstream assert: timestamp > commit_timestamp)"
        );
        assert!(
            timestamp > 0,
            "expiry timestamps are assigned from a positive base and cannot overflow"
        );

        // Collect due pendings in the order the `expires_at` derived index would
        // scan them: ascending `expires_at`, then ascending pending timestamp.
        let mut due: Vec<(u64, u64, u128)> = Vec::new(); // (expires_at, pending_ts, pending_id)
        for (&pending_ts, pending_status) in &self.transfers_pending {
            if pending_status.status != TransferPendingStatus::Pending {
                continue;
            }
            let id = self
                .transfers_by_timestamp
                .get(&pending_ts)
                .copied()
                .expect("pending transfer status row implies a committed pending transfer");
            let p = self
                .transfers
                .get(&id)
                .copied()
                .expect("pending transfer status row implies a committed pending transfer");
            assert!(p.flags.pending());
            assert!(p.timeout > 0);
            let expires_at = p.timestamp + p.timeout_ns();
            if expires_at <= timestamp {
                due.push((expires_at, pending_ts, id));
            }
        }
        due.sort_unstable();

        for (index, (_, _, id)) in due.iter().enumerate() {
            let timestamp_event = timestamp - due.len() as u64 + index as u64 + 1;
            assert!(self.commit_timestamp < timestamp_event);
            self.commit_timestamp = timestamp_event;

            let p = self.transfers.get(id).copied().expect("due pending transfer is committed");
            let expires_at = p.timestamp + p.timeout_ns();
            assert!(expires_at <= timestamp_event);

            let dr = self
                .accounts
                .get(&p.debit_account_id)
                .copied()
                .expect("debit account exists: asserted debits_pending >= amount at creation");
            let cr =
                self.accounts.get(&p.credit_account_id).copied().expect(
                    "credit account exists: asserted credits_pending >= amount at creation",
                );
            assert!(dr.debits_pending >= p.amount);
            assert!(cr.credits_pending >= p.amount);

            let mut dr_new = dr;
            let mut cr_new = cr;
            dr_new.debits_pending = dr_new.debits_pending.wrapping_sub(p.amount);
            cr_new.credits_pending = cr_new.credits_pending.wrapping_sub(p.amount);

            if p.flags.closing_debit() {
                assert!(dr_new.flags.closed());
                dr_new.flags = dr_new.flags.without_closed();
            }
            if p.flags.closing_credit() {
                assert!(cr_new.flags.closed());
                cr_new.flags = cr_new.flags.without_closed();
            }

            let dr_updated = p.amount > 0 || dr_new.flags.closed() != dr.flags.closed();
            let cr_updated = p.amount > 0 || cr_new.flags.closed() != cr.flags.closed();
            if dr_updated {
                self.accounts.insert(dr_new.id, dr_new);
            }
            if cr_updated {
                self.accounts.insert(cr_new.id, cr_new);
            }

            let transfer_pending = self
                .transfers_pending
                .get_mut(&p.timestamp)
                .expect("due pending transfer has a pending status row");
            assert_eq!(transfer_pending.timestamp, p.timestamp);
            assert_eq!(transfer_pending.status, TransferPendingStatus::Pending);
            transfer_pending.status = TransferPendingStatus::Expired;

            self.account_event(
                timestamp_event,
                &dr_new,
                &cr_new,
                TransferFlags::default(),
                TransferPendingStatus::Expired,
                Some(&p),
                0,
                p.amount,
            );
        }

        // Re-point the pulse at the next outstanding expiry (or `timestamp_max`
        // when nothing remains). Upstream advances the value in the expiry
        // pump's `finish`, `state_machine.zig:4984-4996`.
        self.update_pulse_next_timestamp();
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
    fn commit_transfer(&mut self, event: &Transfer, amount_actual: u128, amount_requested: u128) {
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
        self.account_event(
            event.timestamp,
            &dr,
            &cr,
            event.flags,
            if event.flags.pending() {
                TransferPendingStatus::Pending
            } else {
                TransferPendingStatus::None
            },
            None,
            amount_requested,
            amount_actual,
        );

        // A created pending transfer binds the next pulse to its expiry
        // (upstream `state_machine.zig:3975-3982`); the pending status row and
        // `timeout > 0` together imply a created (non-imported) pending, so the
        // asserts mirror upstream's.
        if event.timeout > 0 {
            assert!(event.flags.pending());
            assert!(!event.flags.imported());
            let expires_at = event.timestamp + event.timeout_ns();
            if expires_at < self.pulse_next_timestamp {
                self.pulse_next_timestamp = expires_at;
            }
        }
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
    use std::fmt::Write as _;

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

        // The transient failure poisoned the id: retrying it reports
        // `id_already_failed` (upstream `transient_error`).
        let body = sm.create_transfers(&transfers, 20);
        assert_eq!(
            u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])),
            CreateTransferStatus::IdAlreadyFailed as u32
        );
        assert!(sm.transfers.is_empty());

        // A fresh id succeeds once the accounts exist.
        let _ = sm.create_accounts(&[account(1), account(2)], 30);
        let mut retry = transfers[0];
        retry.id = 101;
        let body = sm.create_transfers(&[retry], 40);
        assert_eq!(body.len(), 16);
        assert_eq!(u32::from_le_bytes(body[8..12].try_into().unwrap_or([0; 4])), u32::MAX);
        assert_eq!(sm.transfers.len(), 1);

        // Duplicate: Exists carries the original timestamp.
        let body = sm.create_transfers(&[retry], 50);
        assert_eq!(u64::from_le_bytes(body[0..8].try_into().unwrap_or([0; 8])), 40);
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
            |_| false,
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

    /// An imported post (or void, when `is_post` is false) of `pending_id`,
    /// posting the full pending amount and naming the pending's ledger/code.
    fn imported_post_or_void(
        id: u128,
        pending_id: u128,
        timestamp: u64,
        is_post: bool,
    ) -> Transfer {
        Transfer {
            id,
            pending_id,
            amount: u128::MAX,
            ledger: 1,
            code: 1,
            flags: if is_post {
                TransferFlags::IMPORTED | TransferFlags::POST_PENDING_TRANSFER
            } else {
                TransferFlags::IMPORTED | TransferFlags::VOID_PENDING_TRANSFER
            },
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

    #[test]
    fn imported_post_void_status_and_expiry_checks_precede_timestamp_regression() {
        let mut sm = StateMachine::default();
        accounts_for_transfers(&mut sm);

        // A pending imported at 21 (past both accounts' stamps, no regress).
        let pending = imported_transfer(1002, 21, TransferFlags::IMPORTED | TransferFlags::PENDING);
        let body = sm.create_transfers(&[pending], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        // Imported post of the full amount at 25 — accepted.
        let body = sm.create_transfers(&[imported_post_or_void(1003, 1002, 25, true)], 40);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
        assert_eq!(
            sm.transfers_pending.get(&21).expect("pending status row").status,
            TransferPendingStatus::Posted
        );

        // A second post of the same (now posted) pending whose timestamp regresses
        // (21 <= transfers key-max 25): the status check must win over the
        // regression check — upstream runs the pending-status switch
        // (state_machine.zig:4130-4143) before the imported regression
        // (state_machine.zig:4158-4180). Same for a full-amount void (amount 0)
        // of the posted pending.
        let body = sm.create_transfers(&[imported_post_or_void(1004, 1002, 21, true)], 50);
        assert_eq!(reply_status(&body), CreateTransferStatus::PendingTransferAlreadyPosted as u32);
        let void = Transfer { amount: 0, ..imported_post_or_void(1005, 1002, 21, false) };
        let body = sm.create_transfers(&[void], 50);
        assert_eq!(reply_status(&body), CreateTransferStatus::PendingTransferAlreadyPosted as u32);
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

        // Distinct ids: the transfers groove would report `.exists` for a
        // second event reusing the first id (in-session idempotency), so the
        // balance accumulation uses fresh ids.
        let batch = [t(1, 2, 50), Transfer { id: 1003, ..t(1, 2, 70) }];
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

        // Close account 1 with the first event, then reference it in the
        // second (a fresh id — reusing the first's id would report `.exists`).
        let closing = Transfer {
            flags: TransferFlags::PENDING | TransferFlags::CLOSING_DEBIT,
            ..t(1, 2, 10)
        };
        let body = sm.create_transfers(&[closing, Transfer { id: 1003, ..t(1, 2, 5) }], 20);
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
    fn post_of_pending_created_in_same_batch_is_created() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // A pending transfer created earlier in the batch is visible to its own
        // post/void (the transfers groove accepts the in-session writes, so the
        // post validates against the working view).
        let pending = Transfer { flags: TransferFlags::PENDING, ..t(1, 2, 50) };
        let post = post_event(3, u128::MAX);
        let body = sm.create_transfers(&[pending, post], 30);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::Created as u32);
        // The stored post folded the pending's account/ledger/code and the
        // pending status advanced to `Posted`, releasing the holds.
        let stored_post = sm.transfers.get(&3).expect("post transfer stored");
        assert_eq!(stored_post.debit_account_id, 1);
        assert_eq!(stored_post.credit_account_id, 2);
        assert_eq!(stored_post.amount, 50);
        assert_eq!(
            sm.transfers_pending.get(&29).expect("pending status stored").status,
            TransferPendingStatus::Posted
        );
        let dr = sm.accounts.get(&1).expect("debit account stored");
        let cr = sm.accounts.get(&2).expect("credit account stored");
        assert_eq!(dr.debits_pending, 0);
        assert_eq!(cr.credits_pending, 0);
        assert_eq!(dr.debits_posted, 50);
        assert_eq!(cr.credits_posted, 50);
    }

    #[test]
    fn void_of_pending_created_in_same_batch_is_created() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        let pending = Transfer { flags: TransferFlags::PENDING, ..t(1, 2, 25) };
        let void = void_event(3, 0);
        let body = sm.create_transfers(&[pending, void], 30);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::Created as u32);
        assert_eq!(
            sm.transfers_pending.get(&29).expect("pending status stored").status,
            TransferPendingStatus::Voided
        );
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 0);
        assert_eq!(dr.debits_posted, 0);
    }

    #[test]
    fn post_after_vacating_pending_in_same_batch_is_rejected() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        // Post and void of the same pending within one batch: the post commits,
        // the void then reports the pending as already posted.
        let post = post_event(3, u128::MAX);
        let void = void_event(4, 0);
        let body = sm.create_transfers(&[post, void], 30);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(
            reply_status_at(&body, 1),
            CreateTransferStatus::PendingTransferAlreadyPosted as u32
        );
    }

    #[test]
    fn duplicate_transfer_id_within_same_batch_reports_exists() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // The second event with the same id sees the first via the working
        // view and reports `.exists` with the first's timestamp.
        let body = sm.create_transfers(&[t(1, 2, 5), t(1, 2, 5)], 30);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::Created as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::Exists as u32);
        assert_eq!(sm.transfers.len(), 1);
    }

    #[test]
    fn chain_break_rolls_back_same_batch_transfer_visibility() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // A linked chain whose second member references a missing account. The
        // chain's pending (member A) is rolled back on break, so a post of it
        // after the chain closed reports pending-not-found.
        let chain_first =
            Transfer { flags: TransferFlags::LINKED | TransferFlags::PENDING, ..t(1, 2, 50) };
        let chain_middle = Transfer { flags: TransferFlags::LINKED, ..t(99, 2, 100) };
        let chain_last = Transfer { flags: TransferFlags::LINKED, ..t(1, 2, 100) };
        let closing_event = t(1, 2, 5);
        let post = post_event(4, u128::MAX);
        let batch = [chain_first, chain_middle, chain_last, closing_event, post];
        let body = sm.create_transfers(&batch, 40);
        assert_eq!(reply_status_at(&body, 0), CreateTransferStatus::LinkedEventFailed as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::DebitAccountNotFound as u32);
        assert_eq!(reply_status_at(&body, 2), CreateTransferStatus::LinkedEventFailed as u32);
        assert_eq!(reply_status_at(&body, 3), CreateTransferStatus::LinkedEventFailed as u32);
        // The pending was rolled back with the broken chain: the post cannot see
        // it and the void's account lookups never ran.
        assert_eq!(reply_status_at(&body, 4), CreateTransferStatus::PendingTransferNotFound as u32);
        // Nothing from the broken chain persisted.
        assert_eq!(sm.transfers.len(), 0);
        let dr = sm.accounts.get(&1).expect("debit account stored");
        assert_eq!(dr.debits_pending, 0);
        assert_eq!(dr.debits_posted, 0);
    }

    // ── Account-change (CDC) event tests ─────────────────────────────────

    #[test]
    fn transfer_creation_records_account_event() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        let body = sm.create_transfers(&[t(1, 2, 7)], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        let event = sm.account_events.get(&20).expect("account event recorded");
        assert_eq!(event.dr_account_id, 1);
        assert_eq!(event.cr_account_id, 2);
        assert_eq!(event.dr_account_timestamp, 9);
        assert_eq!(event.cr_account_timestamp, 10);
        assert_eq!(event.dr_debits_posted, 7);
        assert_eq!(event.cr_credits_posted, 7);
        assert_eq!(event.transfer_flags, TransferFlags::default());
        assert_eq!(event.transfer_pending_status, TransferPendingStatus::None);
        assert_eq!(event.transfer_pending_id, 0);
        assert_eq!(event.transfer_pending_flags, TransferFlags::default());
        assert_eq!(event.amount_requested, 7);
        assert_eq!(event.amount, 7);
        assert_eq!(event.ledger, 1);
    }

    #[test]
    fn pending_transfer_creation_records_account_event() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        let pending = Transfer { flags: TransferFlags::PENDING, ..t(1, 2, 7) };
        let _ = sm.create_transfers(&[pending], 20);

        let event = sm.account_events.get(&20).expect("account event recorded");
        assert_eq!(event.transfer_flags, TransferFlags::PENDING);
        assert_eq!(event.transfer_pending_status, TransferPendingStatus::Pending);
        assert_eq!(event.transfer_pending_id, 0);
        assert_eq!(event.dr_debits_pending, 7);
        assert_eq!(event.cr_credits_pending, 7);
    }

    #[test]
    fn post_records_account_event() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        let _ = sm.create_transfers(&[post_event(3, u128::MAX)], 30);

        let event = sm.account_events.get(&30).expect("post event recorded");
        assert_eq!(event.transfer_flags, TransferFlags::POST_PENDING_TRANSFER);
        assert_eq!(event.transfer_pending_status, TransferPendingStatus::Posted);
        assert_eq!(event.transfer_pending_id, pending_transfer_id());
        assert_eq!(event.transfer_pending_flags, TransferFlags::PENDING);
        // The post requested the full pending amount.
        assert_eq!(event.amount_requested, u128::MAX);
        assert_eq!(event.amount, 50);
        // Balances are snapshotted after the mutation: holds released, posted applied.
        assert_eq!(event.dr_debits_pending, 0);
        assert_eq!(event.cr_credits_pending, 0);
        assert_eq!(event.dr_debits_posted, 50);
        assert_eq!(event.cr_credits_posted, 50);
    }

    #[test]
    fn void_records_account_event() {
        let mut sm = StateMachine::default();
        pending_setup(&mut sm, 50, TransferFlags::default());

        let _ = sm.create_transfers(&[void_event(3, 0)], 30);

        let event = sm.account_events.get(&30).expect("void event recorded");
        assert_eq!(event.transfer_flags, TransferFlags::VOID_PENDING_TRANSFER);
        assert_eq!(event.transfer_pending_status, TransferPendingStatus::Voided);
        assert_eq!(event.transfer_pending_id, pending_transfer_id());
        assert_eq!(event.amount_requested, 0);
        // A full void (amount 0) applies the entire pending amount.
        assert_eq!(event.amount, 50);
        assert_eq!(event.dr_debits_pending, 0);
        assert_eq!(event.cr_credits_pending, 0);
    }

    #[test]
    fn account_event_amount_requested_is_original_for_balancing() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
                Account { id: 3, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );

        // Give account 1 100 credit, then balance 200 against it: 100 is applied
        // but the event records the requested 200.
        let batch = [
            t(3, 1, 100),
            Transfer {
                amount: 200,
                flags: TransferFlags::PENDING | TransferFlags::BALANCING_DEBIT,
                ..t(1, 2, 200)
            },
        ];
        let _ = sm.create_transfers(&batch, 20);

        let event = sm.account_events.get(&20).expect("balancing event recorded");
        assert_eq!(event.amount_requested, 200);
        assert_eq!(event.amount, 100);
        assert_eq!(event.dr_debits_pending, 100);
    }

    #[test]
    fn account_events_indexed_only_for_history_accounts() {
        let mut sm = StateMachine::default();
        // Account 1 (timestamp 19) advertises HISTORY; account 2 does not.
        let _ = sm.create_accounts(
            &[
                Account {
                    id: 1,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            20,
        );

        let _ = sm.create_transfers(&[t(1, 2, 5)], 30);

        assert_eq!(sm.account_events.len(), 1);
        assert_eq!(sm.account_events_index.get(&19), Some(&vec![30]));
        assert!(!sm.account_events_index.contains_key(&20));
    }

    #[test]
    fn account_events_index_lists_events_in_increasing_order() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account {
                    id: 1,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            20,
        );

        // Two events touching the history account in one batch: event timestamps 39, 40.
        let batch = [Transfer { id: 1003, ..t(1, 2, 5) }, Transfer { id: 1004, ..t(1, 2, 6) }];
        let _ = sm.create_transfers(&batch, 40);

        assert_eq!(sm.account_events_index.get(&19), Some(&vec![39, 40]));
        // The store mirrors the object groove: one event per timestamp.
        assert_eq!(sm.account_events.len(), 2);
    }

    // ── get_change_events tests ──────────────────────────────────────────

    /// Decode a single `ChangeEvent` from the reply body offset `i`.
    fn change_event_at(body: &[u8], i: usize) -> ChangeEvent {
        let r = &body[i * 384..(i + 1) * 384];
        ChangeEvent {
            transfer_id: u128::from_le_bytes(r[0..16].try_into().unwrap_or([0; 16])),
            transfer_amount: u128::from_le_bytes(r[16..32].try_into().unwrap_or([0; 16])),
            transfer_pending_id: u128::from_le_bytes(r[32..48].try_into().unwrap_or([0; 16])),
            transfer_user_data_128: u128::from_le_bytes(r[48..64].try_into().unwrap_or([0; 16])),
            transfer_user_data_64: u64::from_le_bytes(r[64..72].try_into().unwrap_or([0; 8])),
            transfer_user_data_32: u32::from_le_bytes(r[72..76].try_into().unwrap_or([0; 4])),
            transfer_timeout: u32::from_le_bytes(r[76..80].try_into().unwrap_or([0; 4])),
            transfer_code: u16::from_le_bytes(r[80..82].try_into().unwrap_or([0; 2])),
            transfer_flags: TransferFlags::from_raw(u16::from_le_bytes(
                r[82..84].try_into().unwrap_or([0; 2]),
            )),
            ledger: u32::from_le_bytes(r[84..88].try_into().unwrap_or([0; 4])),
            r#type: match r[88] {
                0 => ChangeEventType::SinglePhase,
                1 => ChangeEventType::TwoPhasePending,
                2 => ChangeEventType::TwoPhasePosted,
                3 => ChangeEventType::TwoPhaseVoided,
                4 => ChangeEventType::TwoPhaseExpired,
                other => unreachable!("invalid ChangeEventType: {other}"),
            },
            reserved: r[89..128].try_into().unwrap_or([0; 39]),
            debit_account_id: u128::from_le_bytes(r[128..144].try_into().unwrap_or([0; 16])),
            debit_account_debits_pending: u128::from_le_bytes(
                r[144..160].try_into().unwrap_or([0; 16]),
            ),
            debit_account_debits_posted: u128::from_le_bytes(
                r[160..176].try_into().unwrap_or([0; 16]),
            ),
            debit_account_credits_pending: u128::from_le_bytes(
                r[176..192].try_into().unwrap_or([0; 16]),
            ),
            debit_account_credits_posted: u128::from_le_bytes(
                r[192..208].try_into().unwrap_or([0; 16]),
            ),
            debit_account_user_data_128: u128::from_le_bytes(
                r[208..224].try_into().unwrap_or([0; 16]),
            ),
            debit_account_user_data_64: u64::from_le_bytes(
                r[224..232].try_into().unwrap_or([0; 8]),
            ),
            debit_account_user_data_32: u32::from_le_bytes(
                r[232..236].try_into().unwrap_or([0; 4]),
            ),
            debit_account_code: u16::from_le_bytes(r[236..238].try_into().unwrap_or([0; 2])),
            debit_account_flags: AccountFlags::from_raw(u16::from_le_bytes(
                r[238..240].try_into().unwrap_or([0; 2]),
            )),
            credit_account_id: u128::from_le_bytes(r[240..256].try_into().unwrap_or([0; 16])),
            credit_account_debits_pending: u128::from_le_bytes(
                r[256..272].try_into().unwrap_or([0; 16]),
            ),
            credit_account_debits_posted: u128::from_le_bytes(
                r[272..288].try_into().unwrap_or([0; 16]),
            ),
            credit_account_credits_pending: u128::from_le_bytes(
                r[288..304].try_into().unwrap_or([0; 16]),
            ),
            credit_account_credits_posted: u128::from_le_bytes(
                r[304..320].try_into().unwrap_or([0; 16]),
            ),
            credit_account_user_data_128: u128::from_le_bytes(
                r[320..336].try_into().unwrap_or([0; 16]),
            ),
            credit_account_user_data_64: u64::from_le_bytes(
                r[336..344].try_into().unwrap_or([0; 8]),
            ),
            credit_account_user_data_32: u32::from_le_bytes(
                r[344..348].try_into().unwrap_or([0; 4]),
            ),
            credit_account_code: u16::from_le_bytes(r[348..350].try_into().unwrap_or([0; 2])),
            credit_account_flags: AccountFlags::from_raw(u16::from_le_bytes(
                r[350..352].try_into().unwrap_or([0; 2]),
            )),
            timestamp: u64::from_le_bytes(r[352..360].try_into().unwrap_or([0; 8])),
            transfer_timestamp: u64::from_le_bytes(r[360..368].try_into().unwrap_or([0; 8])),
            debit_account_timestamp: u64::from_le_bytes(r[368..376].try_into().unwrap_or([0; 8])),
            credit_account_timestamp: u64::from_le_bytes(r[376..384].try_into().unwrap_or([0; 8])),
        }
    }

    fn change_filter(timestamp_min: u64, timestamp_max: u64, limit: u32) -> ChangeEventsFilter {
        ChangeEventsFilter { timestamp_min, timestamp_max, limit, reserved: [0; 44] }
    }

    #[test]
    fn get_change_events_returns_single_phase_event() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(&[t(1, 2, 7)], 20);

        let body = sm.get_change_events(&change_filter(0, 0, 100));
        assert_eq!(body.len(), 384);
        let e = change_event_at(&body, 0);
        assert_eq!(e.timestamp, 20);
        assert_eq!(e.transfer_id, 1002);
        assert_eq!(e.transfer_timestamp, 20);
        assert_eq!(e.r#type, ChangeEventType::SinglePhase);
        assert_eq!(e.debit_account_id, 1);
        assert_eq!(e.credit_account_id, 2);
        assert_eq!(e.debit_account_timestamp, 9);
        assert_eq!(e.credit_account_timestamp, 10);
        assert_eq!(e.debit_account_debits_posted, 7);
        assert_eq!(e.credit_account_credits_posted, 7);
        assert_eq!(e.transfer_amount, 7);
    }

    #[test]
    fn get_change_events_returns_pending_and_post_events_in_order() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Pending at 20, then a post at 30 (separate batch).
        let pending = Transfer { flags: TransferFlags::PENDING, ..t(1, 2, 50) };
        let _ = sm.create_transfers(&[pending], 20);
        let _ = sm.create_transfers(&[post_event(3, u128::MAX)], 30);

        let body = sm.get_change_events(&change_filter(0, 0, 100));
        assert_eq!(body.len(), 2 * 384);
        let pending_event = change_event_at(&body, 0);
        let post_event_ = change_event_at(&body, 1);
        assert_eq!(pending_event.r#type, ChangeEventType::TwoPhasePending);
        assert_eq!(pending_event.timestamp, 20);
        assert_eq!(post_event_.r#type, ChangeEventType::TwoPhasePosted);
        assert_eq!(post_event_.timestamp, 30);
        assert_eq!(post_event_.transfer_pending_id, pending_transfer_id());
        assert_eq!(post_event_.transfer_amount, 50);
        assert_eq!(post_event_.debit_account_debits_posted, 50);
        assert_eq!(post_event_.debit_account_debits_pending, 0);
    }

    #[test]
    fn get_change_events_respects_timestamp_range_and_limit() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Two events at timestamps 20 (single), 40 (pending).
        let _ = sm.create_transfers(&[t(1, 2, 1)], 20);
        let pending = Transfer { id: 1003, flags: TransferFlags::PENDING, ..t(1, 2, 2) };
        let _ = sm.create_transfers(&[pending], 40);

        // Range [20, 20] → only the first.
        let body = sm.get_change_events(&change_filter(20, 20, 100));
        assert_eq!(body.len(), 384);
        assert_eq!(change_event_at(&body, 0).timestamp, 20);

        // Range starting at 21 → only the second.
        let body = sm.get_change_events(&change_filter(21, 0, 100));
        assert_eq!(body.len(), 384);
        assert_eq!(change_event_at(&body, 0).timestamp, 40);

        // Limit 1 → only the first (ascending order).
        let body = sm.get_change_events(&change_filter(0, 0, 1));
        assert_eq!(body.len(), 384);
        assert_eq!(change_event_at(&body, 0).timestamp, 20);
    }

    #[test]
    fn get_change_events_via_execute_decodes_filter() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(&[t(1, 2, 7)], 20);

        // Encode the filter and route through `execute` like the replica would.
        let filter = change_filter(0, 0, 100);
        let mut body_bytes = Vec::with_capacity(64);
        body_bytes.extend_from_slice(&filter.timestamp_min.to_le_bytes());
        body_bytes.extend_from_slice(&filter.timestamp_max.to_le_bytes());
        body_bytes.extend_from_slice(&filter.limit.to_le_bytes());
        body_bytes.extend_from_slice(&[0_u8; 44]);

        let reply = sm.execute(Operation::GET_CHANGE_EVENTS, 0, &body_bytes);
        assert_eq!(reply.len(), 384);
        assert_eq!(change_event_at(&reply, 0).timestamp, 20);
    }

    // ── id_already_failed orphan tests ───────────────────────────────────
    // Upstream `transient_error` (state_machine.zig:3215-3252): a transfer that
    // fails with a transient status poisons its id so it can never be created
    // again (strong idempotency, reported as `id_already_failed`).

    /// A transfer with a missing debit account fails transiently; retrying the
    /// same id reports `id_already_failed` instead.
    #[test]
    fn transient_failure_poisons_the_id() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let transfer = |id| Transfer {
            id,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };

        // Credit account 3 is missing: `credit_account_not_found`, transient.
        let mut missing_cr = transfer(100);
        missing_cr.credit_account_id = 3;
        let body = sm.create_transfers(&[missing_cr], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::CreditAccountNotFound as u32);

        // The id is poisoned even once the account exists.
        let _ =
            sm.create_accounts(&[Account { id: 3, ledger: 1, code: 1, ..Account::default() }], 30);
        let body = sm.create_transfers(&[transfer(100)], 40);
        assert_eq!(reply_status(&body), CreateTransferStatus::IdAlreadyFailed as u32);

        // The same request under a fresh id succeeds.
        let body = sm.create_transfers(&[transfer(101)], 50);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
    }

    /// An event that fails transiently poisons the id for later events of the
    /// same batch.
    #[test]
    fn orphaned_id_fails_in_same_batch() {
        let mut sm = StateMachine::default();
        let _ =
            sm.create_accounts(&[Account { id: 1, ledger: 1, code: 1, ..Account::default() }], 10);
        let a = Transfer {
            id: 100,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        // Second event reuses the same id after the first poisons it.
        let body = sm.create_transfers(&[a, a], 20);
        // Account 2 is missing: the credit lookup fails transiently.
        assert_eq!(reply_status(&body), CreateTransferStatus::CreditAccountNotFound as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::IdAlreadyFailed as u32);
    }

    /// A post/void event whose id is orphaned reports `id_already_failed` — the
    /// orphan check runs before the post/void dispatch (state_machine.zig:3734-3737).
    #[test]
    fn orphaned_id_fails_for_post_void() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Poisons id 100 via a regular transfer against a missing account.
        let failed = Transfer {
            id: 100,
            debit_account_id: 3,
            credit_account_id: 2,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        let _ = sm.create_transfers(&[failed], 20);

        // A post event reusing id 100 is rejected before any pending lookup.
        let post = Transfer {
            id: 100,
            pending_id: 1,
            amount: 50,
            flags: TransferFlags::POST_PENDING_TRANSFER,
            ..Transfer::default()
        };
        let body = sm.create_transfers(&[post], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::IdAlreadyFailed as u32);
    }

    /// Non-transient failures do not poison the id: a fixable event retries
    /// under the same id.
    #[test]
    fn non_transient_failure_does_not_poison() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let transfer = Transfer {
            id: 100,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 50,
            ledger: 1,
            code: 0,
            ..Transfer::default()
        };
        let body = sm.create_transfers(&[transfer], 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::CodeMustNotBeZero as u32);

        let mut fixed = transfer;
        fixed.code = 1;
        let body = sm.create_transfers(&[fixed], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
    }

    /// A chain rollback orphans the breaking event's id but not the rolled-back
    /// members' ids: the former resolves to `id_already_failed` forever, the
    /// latter can be recreated.
    #[test]
    fn chain_rollback_orphans_only_the_breaker() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let transfer = |id| Transfer {
            id,
            debit_account_id: 1,
            credit_account_id: 2,
            amount: 50,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        let mut breaker = transfer(200);
        breaker.credit_account_id = 3;
        let events = [Transfer { flags: TransferFlags::LINKED, ..transfer(100) }, breaker];
        let body = sm.create_transfers(&events, 20);
        assert_eq!(reply_status(&body), CreateTransferStatus::LinkedEventFailed as u32);
        assert_eq!(reply_status_at(&body, 1), CreateTransferStatus::CreditAccountNotFound as u32);

        // Member 100 was rolled back: recreatable.
        let body = sm.create_transfers(&[transfer(100)], 30);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);

        // Breaker 200 stays poisoned even after account 3 exists.
        let _ =
            sm.create_accounts(&[Account { id: 3, ledger: 1, code: 1, ..Account::default() }], 40);
        let body = sm.create_transfers(&[transfer(200)], 50);
        assert_eq!(reply_status(&body), CreateTransferStatus::IdAlreadyFailed as u32);
    }

    // ── expire_pending_transfers tests ───────────────────────────────────

    const NS_PER_S: u64 = tigerbeetle_core::types::NS_PER_S;

    #[test]
    fn expire_pending_transfer_returns_pending_balances_and_advances_timestamp() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Pending of 50 between 1 and 2 with a 1-second timeout, stamped 20.
        let pending = Transfer { flags: TransferFlags::PENDING, timeout: 1, ..t(1, 2, 50) };
        let _ = sm.create_transfers(&[pending], 20);
        assert_eq!(sm.accounts.get(&1).expect("account 1 stored").debits_pending, 50);
        assert_eq!(sm.accounts.get(&2).expect("account 2 stored").credits_pending, 50);

        // Expire at `20 + 1s`, the exact expiry instant.
        sm.expire_pending_transfers(20 + NS_PER_S);

        // Pending balances are returned to the pools; posted stays unchanged.
        assert_eq!(sm.accounts.get(&1).expect("account 1 stored").debits_pending, 0);
        assert_eq!(sm.accounts.get(&2).expect("account 2 stored").credits_pending, 0);
        assert_eq!(sm.accounts.get(&1).expect("account 1 stored").debits_posted, 0);
        assert_eq!(sm.accounts.get(&2).expect("account 2 stored").credits_posted, 0);

        // The status row is now Expired.
        let pending_status = sm.transfers_pending.get(&20).expect("pending status row");
        assert_eq!(pending_status.status, TransferPendingStatus::Expired);

        // commit_timestamp advanced by the single synthetic expiry stamp.
        assert_eq!(sm.commit_timestamp, 20 + NS_PER_S);

        // The expired event was recorded (timestamp-linked, amount = pending amount).
        let event = sm.account_events.get(&(20 + NS_PER_S)).expect("expired account event");
        assert_eq!(event.transfer_pending_status, TransferPendingStatus::Expired);
        assert_eq!(event.amount, 50);
        assert_eq!(event.amount_requested, 0);
        assert_eq!(event.dr_debits_pending, 0);
        assert_eq!(event.cr_credits_pending, 0);
    }

    #[test]
    fn expire_pending_transfers_only_expires_due_pendings() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Two pendings with different timeouts: id 1002 (1s) stamped 20, id 1003 (10s) stamped 40.
        let p1 = Transfer { flags: TransferFlags::PENDING, timeout: 1, ..t(1, 2, 50) };
        let _ = sm.create_transfers(&[p1], 20);
        let p2 = Transfer { id: 1003, flags: TransferFlags::PENDING, timeout: 10, ..t(1, 2, 7) };
        let _ = sm.create_transfers(&[p2], 40);

        // Expire early (before either is due) → nothing happens.
        sm.expire_pending_transfers(20 + NS_PER_S - 1);
        assert_eq!(
            sm.transfers_pending.get(&20).expect("pending status row").status,
            TransferPendingStatus::Pending
        );
        assert_eq!(
            sm.transfers_pending.get(&40).expect("pending status row").status,
            TransferPendingStatus::Pending
        );
        assert_eq!(sm.transfers_pending.len(), 2);
        assert_eq!(sm.account_events.len(), 2); // the two creation events only

        // Expire at p1's deadline: only p1 (due) expires, p2 (10s) stays Pending.
        sm.expire_pending_transfers(20 + NS_PER_S);
        assert_eq!(
            sm.transfers_pending.get(&20).expect("pending status row").status,
            TransferPendingStatus::Expired
        );
        assert_eq!(
            sm.transfers_pending.get(&40).expect("pending status row").status,
            TransferPendingStatus::Pending
        );
        assert_eq!(sm.accounts.get(&1).expect("account 1 stored").debits_pending, 7);
        assert_eq!(sm.accounts.get(&2).expect("account 2 stored").credits_pending, 7);
        assert_eq!(sm.commit_timestamp, 20 + NS_PER_S);
    }

    #[test]
    fn expire_multiple_pendings_in_index_order() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // p1 (id 1002) stamped 20, timeout 1s → expires 20+1s.
        // p2 (id 1003) stamped 30, timeout 2s → expires 30+2s = 20+1s+10.
        let _ = sm.create_transfers(
            &[Transfer { flags: TransferFlags::PENDING, timeout: 1, ..t(1, 2, 10) }],
            20,
        );
        let _ = sm.create_transfers(
            &[Transfer { id: 1003, flags: TransferFlags::PENDING, timeout: 2, ..t(1, 2, 20) }],
            30,
        );

        // Expire both with a single call past the later deadline.
        sm.expire_pending_transfers(30 + 2 * NS_PER_S);

        // Two synthetic stamps were consumed: the second event is the later one.
        let mut ts: Vec<u64> = sm.account_events.keys().copied().filter(|&ts| ts > 30).collect();
        ts.sort_unstable();
        assert_eq!(
            ts,
            vec![
                30 + 2 * NS_PER_S - 1, // p1 expires first (earlier expires_at)
                30 + 2 * NS_PER_S,     // p2 expires second
            ]
        );

        // Balances fully returned to the pools.
        assert_eq!(sm.accounts.get(&1).expect("account 1 stored").debits_pending, 0);
        assert_eq!(sm.accounts.get(&2).expect("account 2 stored").credits_pending, 0);
        assert_eq!(
            sm.transfers_pending.get(&20).expect("pending status row").status,
            TransferPendingStatus::Expired
        );
        assert_eq!(
            sm.transfers_pending.get(&30).expect("pending status row").status,
            TransferPendingStatus::Expired
        );
    }

    #[test]
    fn expire_pending_transfer_reopens_closing_accounts() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Debit account closes once the pending (its only balance) clears.
        let pending = Transfer {
            flags: TransferFlags::PENDING | TransferFlags::CLOSING_DEBIT,
            timeout: 1,
            ..t(1, 2, 10)
        };
        let _ = sm.create_transfers(&[pending], 20);
        assert!(sm.accounts.get(&1).expect("account 1 stored").flags.closed());

        sm.expire_pending_transfers(20 + NS_PER_S);
        assert!(!sm.accounts.get(&1).expect("account 1 stored").flags.closed());
        assert_eq!(sm.accounts.get(&1).expect("account 1 stored").debits_pending, 0);
    }

    #[test]
    fn expire_pending_transfer_emits_expired_change_event() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(
            &[Transfer { flags: TransferFlags::PENDING, timeout: 1, ..t(1, 2, 50) }],
            20,
        );
        sm.expire_pending_transfers(20 + NS_PER_S);

        // The read path must canonicalize the transfer via `transfer_pending_id`,
        // since no transfer carries the expiry event's timestamp.
        let body = sm.get_change_events(&change_filter(0, 0, 100));
        let expired = change_event_at(&body, 1); // creation (20) then expiry
        assert_eq!(expired.r#type, ChangeEventType::TwoPhaseExpired);
        assert_eq!(expired.timestamp, 20 + NS_PER_S);
        assert_eq!(expired.transfer_id, pending_transfer_id());
        // The ChangeEvent's `transfer_pending_id` is the transfer's own
        // `pending_id`, which is 0 for the original pending.
        assert_eq!(expired.transfer_pending_id, 0);
        assert_eq!(expired.transfer_amount, 50);
        assert_eq!(expired.transfer_timeout, 1);
        assert_eq!(expired.debit_account_debits_pending, 0);
        assert_eq!(expired.credit_account_credits_pending, 0);
        // Debts cleared, nothing posted.
        assert_eq!(expired.debit_account_debits_posted, 0);
        assert_eq!(expired.credit_account_credits_posted, 0);
    }

    // ── state-machine pulse tests ────────────────────────────────────────

    /// `pulse_next_timestamp` starts at `timestamp_min`, so the first possible
    /// pulse fires; a no-op pulse (no pendings outstanding) parks it at
    /// `timestamp_max`.
    #[test]
    fn pulse_needed_starts_owed_and_noop_pulse_parks_it() {
        let mut sm = StateMachine::default();
        assert_eq!(sm.pulse_next_timestamp(), TimestampRange::TIMESTAMP_MIN);
        assert!(!sm.pulse_needed(0));
        assert!(sm.pulse_needed(1));

        let reply = sm.execute(Operation::STATE_MACHINE_PULSE, 1, &[]);
        assert!(reply.is_empty());
        assert_eq!(sm.pulse_next_timestamp(), TimestampRange::TIMESTAMP_MAX);
        assert!(!sm.pulse_needed(1));
    }

    /// Creating a pending transfer arms the pulse as *owed* (`timestamp_min`); the
    /// first pulse scans, then arms the next expiry exactly. Executing the pulse at
    /// that expiry expires the pending and parks the pulse at `timestamp_max` once
    /// nothing is outstanding.
    #[test]
    fn execute_pulse_expires_due_pending_transfer() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Pending stamped 20 with a 1-second timeout.
        let pending = Transfer { flags: TransferFlags::PENDING, timeout: 1, ..t(1, 2, 50) };
        let _ = sm.create_transfers(&[pending], 20);
        let expires_at = 20 + NS_PER_S;

        // A scan is owed immediately (pulse armed at `timestamp_min`); the
        // first pulse finds nothing due and re-arms at the exact expiry
        // (upstream `execute_expire_pending_transfers` + `finish`).
        assert_eq!(sm.pulse_next_timestamp(), TimestampRange::TIMESTAMP_MIN);
        assert!(sm.pulse_needed(1));
        let reply = sm.execute(Operation::STATE_MACHINE_PULSE, 21, &[]);
        assert!(reply.is_empty());
        assert_eq!(sm.pulse_next_timestamp(), expires_at);
        assert!(!sm.pulse_needed(expires_at - 1));
        assert!(sm.pulse_needed(expires_at));

        let reply = sm.execute(Operation::STATE_MACHINE_PULSE, expires_at, &[]);
        assert!(reply.is_empty());

        assert_eq!(sm.accounts.get(&1).expect("account 1 stored").debits_pending, 0);
        assert_eq!(sm.accounts.get(&2).expect("account 2 stored").credits_pending, 0);
        let pending_status = sm.transfers_pending.get(&20).expect("pending status row");
        assert_eq!(pending_status.status, TransferPendingStatus::Expired);
        assert_eq!(sm.pulse_next_timestamp(), TimestampRange::TIMESTAMP_MAX);
        assert!(!sm.pulse_needed(expires_at));
    }

    /// Posting or voiding a pending re-points the pulse at the next outstanding
    /// expiry (or parks it at `timestamp_max` when none remain).
    #[test]
    fn pulse_next_repoints_when_a_pending_is_posted_or_voided() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Two pendings, expiring in order: id 1002 (p1) at `20 + 1s`,
        // id 1003 (p2) at `30 + 2s`. A scan is owed until the next pulse.
        let p1 = Transfer { id: 1002, flags: TransferFlags::PENDING, timeout: 1, ..t(1, 2, 50) };
        let p2 = Transfer { id: 1003, flags: TransferFlags::PENDING, timeout: 2, ..t(1, 2, 25) };
        let _ = sm.create_transfers(&[p1], 20);
        let _ = sm.create_transfers(&[p2], 30);
        assert_eq!(sm.pulse_next_timestamp(), TimestampRange::TIMESTAMP_MIN);

        // Posting p1 leaves p2 as the next expiry.
        let body = sm.create_transfers(&[post_event(1004, u128::MAX)], 40);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
        assert_eq!(sm.pulse_next_timestamp(), 30 + 2 * NS_PER_S);

        // Voiding p2 parks the pulse at `timestamp_max` (`void_event` targets
        // `pending_id` 1002, so build the p2 void directly; a void's zero
        // amount means "full amount", and a void must cover the whole pending).
        let void = Transfer {
            id: 1005,
            pending_id: 1003,
            amount: 0,
            flags: TransferFlags::VOID_PENDING_TRANSFER,
            ..Transfer::default()
        };
        let body = sm.create_transfers(&[void], 50);
        assert_eq!(reply_status(&body), CreateTransferStatus::Created as u32);
        assert_eq!(sm.pulse_next_timestamp(), TimestampRange::TIMESTAMP_MAX);
        assert!(!sm.pulse_needed(50));
    }

    /// A pulse carries no body; `execute` rejects any bytes (upstream
    /// `Operation.valid` returns `batch.len == 0` for `.pulse`, and the replica
    /// never sends one).
    #[test]
    #[should_panic(expected = "pulse must carry an empty body")]
    fn execute_pulse_rejects_nonempty_body() {
        let mut sm = StateMachine::default();
        let _ = sm.execute(Operation::STATE_MACHINE_PULSE, 1, &[0]);
    }

    // ── lookup_accounts / lookup_transfers tests ─────────────────────────

    #[test]
    fn lookup_accounts_returns_matching_accounts_in_request_order() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
                Account { id: 3, ledger: 1, code: 1, ..Account::default() },
            ],
            30,
        );

        // Out-of-order request, with a missing id (99) omitted.
        let reply = sm.lookup_accounts(&[3, 99, 1, 2]);
        let results = bytes_to_account_batch(&reply).expect("valid account batch");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 3);
        assert_eq!(results[1].id, 1);
        assert_eq!(results[2].id, 2);
        // Accounts keep their committed timestamps (account 2 <- 30 - 3 + 1).
        assert_eq!(results[2].timestamp, 29);
    }

    #[test]
    fn lookup_accounts_empty_when_nothing_matches() {
        let sm = StateMachine::default();
        let reply = sm.lookup_accounts(&[1, 2]);
        assert!(reply.is_empty());
    }

    #[test]
    fn lookup_transfers_returns_matching_transfers_in_request_order() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Transfer id 1002 = t(1,2,1), plus a second transfer 1003 -> t(1,2,2).
        let _ = sm.create_transfers(&[t(1, 2, 1), Transfer { id: 1003, ..t(1, 2, 2) }], 20);

        let reply = sm.lookup_transfers(&[1003, 9999, 1002]);
        let results = bytes_to_transfer_batch(&reply).expect("valid transfer batch");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1003);
        assert_eq!(results[1].id, 1002);
        assert_eq!(results[1].amount, 1);
    }

    #[test]
    fn lookup_via_execute_decodes_ids() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(&[t(1, 2, 7)], 20);

        // Encode a lookup request body (one id, 16 bytes LE) and route through
        // `execute` like the replica would; read ops take timestamp 0.
        let mut body = Vec::with_capacity(16);
        body.extend_from_slice(&1_u128.to_le_bytes());

        let accounts = sm.execute(Operation::LOOKUP_ACCOUNTS, 0, &body);
        let parsed = bytes_to_account_batch(&accounts).expect("valid account batch");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, 1);

        let mut tbody = Vec::with_capacity(16);
        tbody.extend_from_slice(&1002_u128.to_le_bytes());
        let transfers = sm.execute(Operation::LOOKUP_TRANSFERS, 0, &tbody);
        let tparsed = bytes_to_transfer_batch(&transfers).expect("valid transfer batch");
        assert_eq!(tparsed.len(), 1);
        assert_eq!(tparsed[0].amount, 7);
    }

    // ── get_account_transfers tests ──────────────────────────────────────

    fn account_filter(account_id: u128, flags: AccountFilterFlags) -> AccountFilter {
        AccountFilter { account_id, flags, limit: 100, ..AccountFilter::default() }
    }

    #[test]
    fn get_account_transfers_returns_transfers_for_debits_and_credits() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
                Account { id: 3, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Account 1 is debited (1->2), account 2 is credited (1->2) and debited
        // (2->3); account 3 is only credited (2->3).
        let _ = sm.create_transfers(&[t(1, 2, 10), t(2, 3, 20)], 20);

        let dr_filter = account_filter(1, AccountFilterFlags::DEBITS);
        let dr = sm.get_account_transfers(&dr_filter);
        let dr_results = bytes_to_transfer_batch(&dr).expect("valid transfer batch");
        assert_eq!(dr_results.len(), 1);
        assert_eq!(dr_results[0].debit_account_id, 1);

        let cr_filter = account_filter(2, AccountFilterFlags::CREDITS);
        let cr = sm.get_account_transfers(&cr_filter);
        let cr_results = bytes_to_transfer_batch(&cr).expect("valid transfer batch");
        assert_eq!(cr_results.len(), 1);
        assert_eq!(cr_results[0].credit_account_id, 2);

        let both_filter =
            account_filter(2, AccountFilterFlags::DEBITS | AccountFilterFlags::CREDITS);
        let both = sm.get_account_transfers(&both_filter);
        let both_results = bytes_to_transfer_batch(&both).expect("valid transfer batch");
        assert_eq!(both_results.len(), 2);
    }

    #[test]
    fn get_account_transfers_posted_sorted_chronologically_and_reversed() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        // Same batch of four transfers; timestamps 20..23.
        let _ = sm.create_transfers(
            &[
                t(1, 2, 1),
                Transfer { id: 1003, ..t(1, 2, 2) },
                Transfer { id: 1004, ..t(1, 2, 3) },
                Transfer { id: 1005, ..t(1, 2, 4) },
            ],
            20,
        );

        let asc_filter = account_filter(1, AccountFilterFlags::DEBITS);
        let asc = sm.get_account_transfers(&asc_filter);
        let asc_results = bytes_to_transfer_batch(&asc).expect("valid transfer batch");
        let asc_ts: Vec<u64> = asc_results.iter().map(|t| t.timestamp).collect();
        assert_eq!(asc_ts, vec![17, 18, 19, 20]);

        let desc_filter =
            account_filter(1, AccountFilterFlags::DEBITS | AccountFilterFlags::REVERSED);
        let desc = sm.get_account_transfers(&desc_filter);
        let desc_results = bytes_to_transfer_batch(&desc).expect("valid transfer batch");
        let desc_ts: Vec<u64> = desc_results.iter().map(|t| t.timestamp).collect();
        assert_eq!(desc_ts, vec![20, 19, 18, 17]);

        let limited = AccountFilter { limit: 2, ..account_filter(1, AccountFilterFlags::DEBITS) };
        let lim = sm.get_account_transfers(&limited);
        let lim_results = bytes_to_transfer_batch(&lim).expect("valid transfer batch");
        let lim_ts: Vec<u64> = lim_results.iter().map(|t| t.timestamp).collect();
        assert_eq!(lim_ts, vec![17, 18]);
    }

    #[test]
    fn get_account_transfers_filters_by_user_data_code_and_timestamp_range() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let mut a = t(1, 2, 5);
        a.code = 7;
        a.user_data_64 = 99;
        let mut b = t(1, 2, 6);
        b.code = 8;
        let _ = sm.create_transfers(&[a, b], 20); // timestamps 20, 21

        let code_filter =
            AccountFilter { code: 7, ..account_filter(1, AccountFilterFlags::DEBITS) };
        let code = sm.get_account_transfers(&code_filter);
        let code_results = bytes_to_transfer_batch(&code).expect("valid transfer batch");
        assert_eq!(code_results.len(), 1);
        assert_eq!(code_results[0].code, 7);

        let range_filter = AccountFilter {
            timestamp_min: 19,
            timestamp_max: 19,
            ..account_filter(1, AccountFilterFlags::DEBITS)
        };
        let range = sm.get_account_transfers(&range_filter);
        let range_results = bytes_to_transfer_batch(&range).expect("valid transfer batch");
        assert_eq!(range_results.len(), 1);
        assert_eq!(range_results[0].timestamp, 19);
    }

    #[test]
    fn get_account_transfers_rejects_invalid_filter() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(&[t(1, 2, 5)], 20);

        // No debits/credits flag, limit of zero, and padding bits are all
        // invalid and must yield an empty reply.
        let no_side = account_filter(1, AccountFilterFlags::default());
        assert!(sm.get_account_transfers(&no_side).is_empty());

        let no_limit = AccountFilter { limit: 0, ..account_filter(1, AccountFilterFlags::DEBITS) };
        assert!(sm.get_account_transfers(&no_limit).is_empty());

        let padding = AccountFilter {
            flags: AccountFilterFlags::from_raw(0xFFFF_FFFF),
            ..account_filter(1, AccountFilterFlags::default())
        };
        assert!(sm.get_account_transfers(&padding).is_empty());
    }

    // ── get_account_balances tests ───────────────────────────────────────

    #[test]
    fn get_account_balances_requires_history_flag() {
        let mut sm = StateMachine::default();
        // Accounts without the history flag still record events, but the query
        // honors the account's `flags.history`, so the reply is empty.
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(&[t(1, 2, 5)], 20);

        let filter = account_filter(1, AccountFilterFlags::DEBITS);
        assert!(sm.get_account_balances(&filter).is_empty());

        // A non-existent account is also empty.
        let missing = account_filter(999, AccountFilterFlags::DEBITS);
        assert!(sm.get_account_balances(&missing).is_empty());
    }

    #[test]
    fn get_account_balances_snapshots_each_matching_side() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account {
                    id: 1,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
                Account {
                    id: 2,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
            ],
            10,
        );
        // Two transfers: 1->2 (50), then 1->2 (30): account 1 ends with
        // debits_posted 80; account 2 credits_posted 80.
        let _ = sm.create_transfers(&[t(1, 2, 50), Transfer { id: 1003, ..t(1, 2, 30) }], 20);

        let dr_filter = account_filter(1, AccountFilterFlags::DEBITS);
        let dr_view = sm.get_account_balances(&dr_filter);
        let (dr_snapshots, _) = dr_view.as_chunks::<128>();
        assert_eq!(dr_snapshots.len(), 2);
        // Chronological: first event (ts 19) has 50 debits posted, then 80
        // (ts 20).
        assert_eq!(le_u128(&dr_snapshots[0], 16), 50);
        assert_eq!(le_u128(&dr_snapshots[1], 16), 80);
        assert_eq!(le_u64(&dr_snapshots[0], 64), 19);
        assert_eq!(le_u64(&dr_snapshots[1], 64), 20);

        let cr_filter = account_filter(2, AccountFilterFlags::CREDITS);
        let cr_view = sm.get_account_balances(&cr_filter);
        let (cr_snapshots, _) = cr_view.as_chunks::<128>();
        assert_eq!(cr_snapshots.len(), 2);
        // Credit side: credits_posted field is at offset 48.
        assert_eq!(le_u128(&cr_snapshots[1], 48), 80);
    }

    #[test]
    fn get_account_balances_reversed_and_limited() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account {
                    id: 1,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
                Account {
                    id: 2,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
            ],
            10,
        );
        let _ = sm.create_transfers(
            &[t(1, 2, 1), Transfer { id: 1003, ..t(1, 2, 2) }, Transfer { id: 1004, ..t(1, 2, 3) }],
            20,
        );

        let desc = AccountFilter {
            limit: 2,
            flags: AccountFilterFlags::DEBITS | AccountFilterFlags::REVERSED,
            ..account_filter(1, AccountFilterFlags::default())
        };
        let view = sm.get_account_balances(&desc);
        let (snapshots, _) = view.as_chunks::<128>();
        assert_eq!(snapshots.len(), 2);
        // Most recent first: timestamps 20, 19.
        assert_eq!(le_u64(&snapshots[0], 64), 20);
        assert_eq!(le_u64(&snapshots[1], 64), 19);
    }

    #[test]
    fn get_account_balances_excludes_expired_events() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account {
                    id: 1,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
                Account {
                    id: 2,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
            ],
            10,
        );
        let _ = sm.create_transfers(
            &[Transfer { flags: TransferFlags::PENDING, timeout: 1, ..t(1, 2, 50) }],
            20,
        );
        let _ = sm.execute(Operation::STATE_MACHINE_PULSE, 20 + NS_PER_S, &[]);

        // The CDC records the expiry event, but the balance history does not:
        // upstream yields balances only for events with a transfer at their
        // timestamp (AccountBalancesScanLookup scans via the transfers groove).
        let filter =
            ChangeEventsFilter { timestamp_min: 0, timestamp_max: 0, limit: 10, reserved: [0; 44] };
        assert_eq!(sm.get_change_events(&filter).len(), 2 * 384);

        let view = sm.get_account_balances(&account_filter(1, AccountFilterFlags::DEBITS));
        let (snapshots, _) = view.as_chunks::<128>();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(le_u64(&snapshots[0], 64), 20);
        assert_eq!(le_u128(&snapshots[0], 0), 50);
    }

    #[test]
    fn get_account_balances_via_execute_decodes_filter() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account {
                    id: 1,
                    ledger: 1,
                    code: 1,
                    flags: AccountFlags::HISTORY,
                    ..Account::default()
                },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(&[t(1, 2, 5)], 20);

        // Encode a 128-byte AccountFilter for account 1 (debits, limit 100).
        let mut body = vec![0_u8; 128];
        body[0..16].copy_from_slice(&1_u128.to_le_bytes());
        body[120..124].copy_from_slice(&100_u32.to_le_bytes());
        body[124..128].copy_from_slice(&AccountFilterFlags::DEBITS.as_raw().to_le_bytes());

        let reply = sm.execute(Operation::GET_ACCOUNT_BALANCES, 0, &body);
        let (snapshots, _) = reply.as_chunks::<128>();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(le_u64(&snapshots[0], 64), 20);
    }

    // ── query_accounts / query_transfers tests ───────────────────────────

    fn query_filter() -> QueryFilter {
        QueryFilter { limit: 100, ..QueryFilter::default() }
    }

    #[test]
    fn query_accounts_filters_by_user_data_ledger_code_and_range() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 2, code: 1, user_data_64: 99, ..Account::default() },
                Account { id: 3, ledger: 1, code: 7, ..Account::default() },
            ],
            10,
        );

        // All three created in one batch at base ts 10 (N=3): timestamps
        // 8, 9, 10.
        let all = sm.query_accounts(&query_filter());
        let all_results = bytes_to_account_batch(&all).expect("valid account batch");
        let ts: Vec<u64> = all_results.iter().map(|a| a.timestamp).collect();
        assert_eq!(ts, vec![8, 9, 10]);

        let ledger = QueryFilter { ledger: 2, ..query_filter() };
        let l = sm.query_accounts(&ledger);
        let lr = bytes_to_account_batch(&l).expect("valid account batch");
        assert_eq!(lr.len(), 1);
        assert_eq!(lr[0].id, 2);

        let code = QueryFilter { code: 7, ..query_filter() };
        let c = sm.query_accounts(&code);
        let cr = bytes_to_account_batch(&c).expect("valid account batch");
        assert_eq!(cr.len(), 1);
        assert_eq!(cr[0].id, 3);

        let range = QueryFilter { timestamp_min: 8, timestamp_max: 9, ..query_filter() };
        let r = sm.query_accounts(&range);
        let rr = bytes_to_account_batch(&r).expect("valid account batch");
        assert_eq!(rr.len(), 2);
        assert_eq!(rr[0].timestamp, 8);
        assert_eq!(rr[1].timestamp, 9);
    }

    #[test]
    fn query_accounts_reversed_and_limited() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
                Account { id: 3, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let desc = QueryFilter { flags: QueryFilterFlags::REVERSED, limit: 2, ..query_filter() };
        let view = sm.query_accounts(&desc);
        let results = bytes_to_account_batch(&view).expect("valid account batch");
        // Most recent first: ts 10, 9.
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].timestamp, 10);
        assert_eq!(results[1].timestamp, 9);
    }

    #[test]
    fn query_transfers_filters_and_orders() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
                Account { id: 5, ledger: 2, code: 1, ..Account::default() },
                Account { id: 6, ledger: 2, code: 1, ..Account::default() },
            ],
            10,
        );
        // `a`: 1->2 on ledger 1, user_data_32 = 7. `b`: 5->6 on ledger 2.
        let mut a = t(1, 2, 5);
        a.user_data_32 = 7;
        let mut b = Transfer { id: 1003, ..t(5, 6, 6) };
        b.ledger = 2;
        let _ = sm.create_transfers(&[a, b], 20); // timestamps 20, 21

        let user_data = QueryFilter { user_data_32: 7, ..query_filter() };
        let u = sm.query_transfers(&user_data);
        let ur = bytes_to_transfer_batch(&u).expect("valid transfer batch");
        assert_eq!(ur.len(), 1);
        assert_eq!(ur[0].amount, 5);

        let ledger = QueryFilter { ledger: 2, ..query_filter() };
        let l = sm.query_transfers(&ledger);
        let lr = bytes_to_transfer_batch(&l).expect("valid transfer batch");
        assert_eq!(lr.len(), 1);
        assert_eq!(lr[0].amount, 6);

        // No conditions: all in ascending timestamp order.
        let all = sm.query_transfers(&query_filter());
        let ar = bytes_to_transfer_batch(&all).expect("valid transfer batch");
        let ts: Vec<u64> = ar.iter().map(|t| t.timestamp).collect();
        assert_eq!(ts, vec![19, 20]);
    }

    #[test]
    fn query_rejects_invalid_filter() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 1, code: 1, ..Account::default() },
                Account { id: 2, ledger: 1, code: 1, ..Account::default() },
            ],
            10,
        );
        let _ = sm.create_transfers(&[t(1, 2, 5)], 20);

        let no_limit = QueryFilter { limit: 0, ..QueryFilter::default() };
        assert!(sm.query_accounts(&no_limit).is_empty());
        assert!(sm.query_transfers(&no_limit).is_empty());

        let padding =
            QueryFilter { flags: QueryFilterFlags::from_raw(0xFFFF_FFFF), ..query_filter() };
        assert!(sm.query_accounts(&padding).is_empty());
    }

    #[test]
    fn query_via_execute_decodes_filter() {
        let mut sm = StateMachine::default();
        let _ = sm.create_accounts(
            &[
                Account { id: 1, ledger: 5, code: 1, ..Account::default() },
                Account { id: 2, ledger: 5, code: 1, ..Account::default() },
            ],
            10,
        );

        // Encode a 64-byte QueryFilter for ledger 5, limit 100.
        let mut body = vec![0_u8; 64];
        body[28..32].copy_from_slice(&5_u32.to_le_bytes());
        body[56..60].copy_from_slice(&100_u32.to_le_bytes());

        let reply = sm.execute(Operation::QUERY_ACCOUNTS, 0, &body);
        let results = bytes_to_account_batch(&reply).expect("valid account batch");
        assert_eq!(results.len(), 2);

        let transfers = sm.execute(Operation::QUERY_TRANSFERS, 0, &body);
        // No transfers exist; empty reply.
        assert!(transfers.is_empty());
    }

    // ── Upstream golden cross-validation ─────────────────────────────────

    /// Snake-cases a [`CreateAccountStatus`] into upstream's `@tagName`
    /// string; the harness prints `{timestamp}:{@tagName(status)}`.
    fn account_status_name(status: u32) -> &'static str {
        match status {
            s if s == CreateAccountStatus::LinkedEventFailed as u32 => "linked_event_failed",
            s if s == CreateAccountStatus::LinkedEventChainOpen as u32 => "linked_event_chain_open",
            s if s == CreateAccountStatus::Exists as u32 => "exists",
            s if s == CreateAccountStatus::Created as u32 => "created",
            s if s == CreateAccountStatus::ImportedEventTimestampMustNotRegress as u32 => {
                "imported_event_timestamp_must_not_regress"
            }
            other => panic!("unexpected account status {other}"),
        }
    }

    /// Snake-cases a [`CreateTransferStatus`] into upstream's `@tagName`
    /// string.
    fn transfer_status_name(status: u32) -> &'static str {
        match status {
            s if s == CreateTransferStatus::AccountsMustHaveTheSameLedger as u32 => {
                "accounts_must_have_the_same_ledger"
            }
            s if s == CreateTransferStatus::PendingTransferNotFound as u32 => {
                "pending_transfer_not_found"
            }
            s if s == CreateTransferStatus::PendingTransferAlreadyPosted as u32 => {
                "pending_transfer_already_posted"
            }
            s if s == CreateTransferStatus::PendingTransferNotPending as u32 => {
                "pending_transfer_not_pending"
            }
            s if s == CreateTransferStatus::PendingTransferHasDifferentAmount as u32 => {
                "pending_transfer_has_different_amount"
            }
            s if s == CreateTransferStatus::PendingTransferAlreadyVoided as u32 => {
                "pending_transfer_already_voided"
            }
            s if s == CreateTransferStatus::PendingTransferExpired as u32 => {
                "pending_transfer_expired"
            }
            s if s == CreateTransferStatus::CreditAccountNotFound as u32 => {
                "credit_account_not_found"
            }
            s if s == CreateTransferStatus::DebitAccountAlreadyClosed as u32 => {
                "debit_account_already_closed"
            }
            s if s == CreateTransferStatus::CreditAccountAlreadyClosed as u32 => {
                "credit_account_already_closed"
            }
            s if s == CreateTransferStatus::IdAlreadyFailed as u32 => "id_already_failed",
            s if s == CreateTransferStatus::ImportedEventTimestampMustNotRegress as u32 => {
                "imported_event_timestamp_must_not_regress"
            }
            s if s == CreateTransferStatus::Exists as u32 => "exists",
            s if s == CreateTransferStatus::Created as u32 => "created",
            other => panic!("unexpected transfer status {other}"),
        }
    }

    /// Returns the `{timestamp}:{dp}:{dpost}:{cp}:{cpost}` line of a
    /// 128-byte `AccountBalance` record.
    fn format_balance(row: &[u8]) -> String {
        let ts = u64::from_le_bytes(row[64..72].try_into().expect("8-byte slice"));
        let dp = u128::from_le_bytes(row[0..16].try_into().expect("16-byte slice"));
        let dpost = u128::from_le_bytes(row[16..32].try_into().expect("16-byte slice"));
        let cp = u128::from_le_bytes(row[32..48].try_into().expect("16-byte slice"));
        let cpost = u128::from_le_bytes(row[48..64].try_into().expect("16-byte slice"));
        format!("{ts}:{dp}:{dpost}:{cp}:{cpost}")
    }

    /// Returns the `{id}:{ts}:{amount}:{pending_id}:{dr}:{cr}:{ledger}:{code}:{timeout}:{flags}`
    /// line of a `Transfer` (upstream field order).
    fn format_transfer(t: &Transfer) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            t.id,
            t.timestamp,
            t.amount,
            t.pending_id,
            t.debit_account_id,
            t.credit_account_id,
            t.ledger,
            t.code,
            t.timeout,
            t.flags.as_raw()
        )
    }

    /// Returns the `{id}:{ts}:{dp}:{dpost}:{cp}:{cpost}:{ledger}:{code}:{flags}`
    /// line of an `Account` (same line used by lookup and query sections).
    fn format_account(acc: &Account) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}",
            acc.id,
            acc.timestamp,
            acc.debits_pending,
            acc.debits_posted,
            acc.credits_pending,
            acc.credits_posted,
            acc.ledger,
            acc.code,
            acc.flags.as_raw()
        )
    }

    /// Upstream `@tagName` of a [`ChangeEventType`].
    fn change_event_type_name(t: ChangeEventType) -> &'static str {
        match t {
            ChangeEventType::SinglePhase => "single_phase",
            ChangeEventType::TwoPhasePending => "two_phase_pending",
            ChangeEventType::TwoPhasePosted => "two_phase_posted",
            ChangeEventType::TwoPhaseVoided => "two_phase_voided",
            ChangeEventType::TwoPhaseExpired => "two_phase_expired",
        }
    }

    /// Returns the `{ts}:{type}:{transfer_id}:{transfer_amount}:
    /// {transfer_pending_id}:{dr}:{dr_dp}:{dr_dpost}:{dr_cp}:{dr_cpost}:
    /// {cr}:{cr_dp}:{cr_dpost}:{cr_cp}:{cr_cpost}` line of a `ChangeEvent`
    /// (the recorded account balance snapshots, not the live state).
    fn format_change_event(e: &ChangeEvent) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            e.timestamp,
            change_event_type_name(e.r#type),
            e.transfer_id,
            e.transfer_amount,
            e.transfer_pending_id,
            e.debit_account_id,
            e.debit_account_debits_pending,
            e.debit_account_debits_posted,
            e.debit_account_credits_pending,
            e.debit_account_credits_posted,
            e.credit_account_id,
            e.credit_account_debits_pending,
            e.credit_account_debits_posted,
            e.credit_account_credits_pending,
            e.credit_account_credits_posted,
        )
    }

    /// The exact stdout of `reference/tigerbeetle/src/tbcross_accounting.zig`
    /// driven against the real upstream `StateMachine` (`test_min` config).
    ///
    /// Recompile the harness and rerun it whenever this script changes:
    ///
    /// ```sh
    /// cd reference/tigerbeetle/src && ../zig/zig build-exe \
    ///   --dep stdx -Mroot=tbcross_accounting.zig \
    ///   -Mstdx=$PWD/stdx/stdx.zig -femit-bin=/tmp/tbaccounting && \
    /// /tmp/tbaccounting 2>/dev/null
    /// ```
    ///
    /// Each batch commits at `1 + batch_len` past the previous batch's last
    /// timestamp (upstream `prepare()` advances by the event count), giving
    /// this port its exact per-event timestamps. Compare statuses, timestamps,
    /// balances and lookups line-by-line.
    const GOLDEN_ACCOUNTING: &str = "\
create_accounts;
2:created
3:created
create_accounts;
5:created
create_transfers;
7:created
create_transfers;
9:created
create_transfers;
7:exists
12:accounts_must_have_the_same_ledger
create_transfers;
14:created
create_transfers;
16:pending_transfer_not_found
create_transfers;
18:pending_transfer_already_posted
create_transfers;
20:pending_transfer_not_pending
create_transfers;
22:credit_account_not_found
create_transfers;
24:id_already_failed
get_account_balances account=1;
7:500:0:0:0
9:500:100:0:0
14:0:600:0:0
get_account_balances account=2;
7:0:0:500:0
9:0:0:500:100
14:0:0:0:600
lookup_accounts n=3;
1:2:0:600:0:0:1:1:8
3:5:0:0:0:0:2:1:8
lookup_transfers n=4;
1:7:500:0:1:2:1:1:0:2
2:9:100:0:1:2:1:1:0:0
4:14:500:1:1:2:1:1:0:4
get_account_transfers account=1 reversed=0;
1:7:500:0:1:2:1:1:0:2
2:9:100:0:1:2:1:1:0:0
4:14:500:1:1:2:1:1:0:4
get_account_transfers account=2 reversed=1;
4:14:500:1:1:2:1:1:0:4
2:9:100:0:1:2:1:1:0:0
1:7:500:0:1:2:1:1:0:2
query_accounts ts=[0,0] limit=10 reversed=0;
1:2:0:600:0:0:1:1:8
2:3:0:0:0:600:1:2:8
3:5:0:0:0:0:2:1:8
query_accounts ts=[3,0] limit=1 reversed=1;
3:5:0:0:0:0:2:1:8
query_transfers ts=[0,0] limit=10 reversed=0;
1:7:500:0:1:2:1:1:0:2
2:9:100:0:1:2:1:1:0:0
4:14:500:1:1:2:1:1:0:4
query_transfers ts=[0,0] limit=2 reversed=1;
4:14:500:1:1:2:1:1:0:4
2:9:100:0:1:2:1:1:0:0
get_change_events ts=[0,0] limit=10;
7:two_phase_pending:1:500:0:1:500:0:0:0:2:0:0:500:0
9:single_phase:2:100:0:1:500:100:0:0:2:0:0:500:100
14:two_phase_posted:4:500:1:1:0:600:0:0:2:0:0:0:600
create_accounts;
37:imported_event_timestamp_must_not_regress
create_accounts;
6:created
30:created
create_accounts;
6:exists
create_accounts;
30:exists
create_accounts;
40:created
47:imported_event_timestamp_must_not_regress
create_transfers;
49:imported_event_timestamp_must_not_regress
create_transfers;
15:created
create_transfers;
53:imported_event_timestamp_must_not_regress
create_transfers;
16:created
create_transfers;
17:created
lookup_accounts n=2;
10:6:0:0:0:0:1:1:16
20:30:0:0:0:0:1:1:16
lookup_transfers n=3;
24:16:1:0:1:2:1:1:0:258
25:17:1:24:1:2:1:1:0:260
22:15:1:0:1:2:1:1:0:256
get_account_balances account=1;
7:500:0:0:0
9:500:100:0:0
14:0:600:0:0
15:0:601:0:0
16:1:601:0:0
17:0:602:0:0
get_change_events ts=[0,0] limit=10;
7:two_phase_pending:1:500:0:1:500:0:0:0:2:0:0:500:0
9:single_phase:2:100:0:1:500:100:0:0:2:0:0:500:100
14:two_phase_posted:4:500:1:1:0:600:0:0:2:0:0:0:600
15:single_phase:22:1:0:1:0:601:0:0:2:0:0:0:601
16:two_phase_pending:24:1:0:1:1:601:0:0:2:0:0:1:601
17:two_phase_posted:25:1:24:1:0:602:0:0:2:0:0:0:602
create_transfers;
63:created
create_transfers;
65:created
pulse at 3000000096;
lookup_transfers n=2;
333:63:1234:0:1:2:1:1:2:2
334:65:500:0:1:2:1:1:100:2
get_account_balances account=1;
7:500:0:0:0
9:500:100:0:0
14:0:600:0:0
15:0:601:0:0
16:1:601:0:0
17:0:602:0:0
63:1234:602:0:0
65:1734:602:0:0
get_account_balances account=2;
7:0:0:500:0
9:0:0:500:100
14:0:0:0:600
15:0:0:0:601
16:0:0:1:601
17:0:0:0:602
63:0:0:1234:602
65:0:0:1734:602
get_change_events ts=[0,0] limit=10;
7:two_phase_pending:1:500:0:1:500:0:0:0:2:0:0:500:0
9:single_phase:2:100:0:1:500:100:0:0:2:0:0:500:100
14:two_phase_posted:4:500:1:1:0:600:0:0:2:0:0:0:600
15:single_phase:22:1:0:1:0:601:0:0:2:0:0:0:601
16:two_phase_pending:24:1:0:1:1:601:0:0:2:0:0:1:601
17:two_phase_posted:25:1:24:1:0:602:0:0:2:0:0:0:602
63:two_phase_pending:333:1234:0:1:1234:602:0:0:2:0:0:1234:602
65:two_phase_pending:334:500:0:1:1734:602:0:0:2:0:0:1734:602
3000000096:two_phase_expired:333:1234:0:1:500:602:0:0:2:0:0:500:602
create_transfers;
3000000102:created
create_transfers;
3000000104:pending_transfer_expired
create_transfers;
3000000106:pending_transfer_has_different_amount
create_transfers;
3000000108:pending_transfer_already_posted
create_transfers;
3000000110:created
create_transfers;
3000000112:created
create_transfers;
3000000114:pending_transfer_already_voided
lookup_transfers n=7;
340:3000000102:201:334:1:2:1:1:0:4
350:3000000110:777:0:1:2:1:1:200:2
351:3000000112:777:350:1:2:1:1:0:8
get_account_balances account=1;
7:500:0:0:0
9:500:100:0:0
14:0:600:0:0
15:0:601:0:0
16:1:601:0:0
17:0:602:0:0
63:1234:602:0:0
65:1734:602:0:0
3000000102:0:803:0:0
3000000110:777:803:0:0
3000000112:0:803:0:0
get_account_balances account=2;
7:0:0:500:0
9:0:0:500:100
14:0:0:0:600
15:0:0:0:601
16:0:0:1:601
17:0:0:0:602
63:0:0:1234:602
65:0:0:1734:602
3000000102:0:0:0:803
3000000110:0:0:777:803
3000000112:0:0:0:803
get_change_events ts=[0,0] limit=15;
7:two_phase_pending:1:500:0:1:500:0:0:0:2:0:0:500:0
9:single_phase:2:100:0:1:500:100:0:0:2:0:0:500:100
14:two_phase_posted:4:500:1:1:0:600:0:0:2:0:0:0:600
15:single_phase:22:1:0:1:0:601:0:0:2:0:0:0:601
16:two_phase_pending:24:1:0:1:1:601:0:0:2:0:0:1:601
17:two_phase_posted:25:1:24:1:0:602:0:0:2:0:0:0:602
63:two_phase_pending:333:1234:0:1:1234:602:0:0:2:0:0:1234:602
65:two_phase_pending:334:500:0:1:1734:602:0:0:2:0:0:1734:602
3000000096:two_phase_expired:333:1234:0:1:500:602:0:0:2:0:0:500:602
3000000102:two_phase_posted:340:201:334:1:0:803:0:0:2:0:0:0:803
create_transfers;
3000000120:created
lookup_accounts n=1;
1:2:50:803:0:0:1:1:40
pulse at 4000000152;
lookup_accounts n=1;
1:2:0:803:0:0:1:1:8
get_account_balances account=1;
7:500:0:0:0
9:500:100:0:0
14:0:600:0:0
15:0:601:0:0
16:1:601:0:0
17:0:602:0:0
63:1234:602:0:0
65:1734:602:0:0
3000000102:0:803:0:0
3000000110:777:803:0:0
3000000112:0:803:0:0
3000000120:50:803:0:0
get_account_balances account=2;
7:0:0:500:0
9:0:0:500:100
14:0:0:0:600
15:0:0:0:601
16:0:0:1:601
17:0:0:0:602
63:0:0:1234:602
65:0:0:1734:602
3000000102:0:0:0:803
3000000110:0:0:777:803
3000000112:0:0:0:803
3000000120:0:0:50:803
get_change_events ts=[3000000116,0] limit=15;
3000000120:two_phase_pending:360:50:0:1:50:803:0:0:2:0:0:50:803
4000000152:two_phase_expired:360:50:0:1:0:803:0:0:2:0:0:0:803
create_accounts;
4000000158:created
create_transfers;
4000000160:created
lookup_accounts n=1;
4:4000000158:100:0:0:0:1:1:40
create_transfers;
4000000163:debit_account_already_closed
create_transfers;
4000000165:credit_account_already_closed
pulse at 5000000196;
lookup_accounts n=1;
4:4000000158:0:0:0:0:1:1:8
create_transfers;
5000000199:created
get_account_balances account=4;
4000000160:100:0:0:0
5000000199:0:10:0:0
create_transfers;
5000000202:created
create_transfers;
5000000204:created
create_transfers;
5000000206:created
create_transfers;
5000000208:created
create_transfers;
5000000210:created
get_account_balances account=1;
7:500:0:0:0
9:500:100:0:0
14:0:600:0:0
15:0:601:0:0
16:1:601:0:0
17:0:602:0:0
63:1234:602:0:0
65:1734:602:0:0
3000000102:0:803:0:0
3000000110:777:803:0:0
3000000112:0:803:0:0
3000000120:50:803:0:0
5000000202:0:804:0:0
5000000204:0:806:0:0
5000000206:0:809:0:0
5000000208:0:813:0:0
5000000210:0:818:0:0
get_account_balances account=1 ts=[5000000202,0] limit=100;
5000000202:0:804:0:0
5000000204:0:806:0:0
5000000206:0:809:0:0
5000000208:0:813:0:0
5000000210:0:818:0:0
create_transfers;
5000000214:created
create_transfers;
5000000216:created
create_transfers;
5000000218:created
create_transfers;
5000000220:created
create_transfers;
5000000222:created
create_transfers;
5000000224:created
create_transfers;
5000000226:created
create_transfers;
5000000228:created
create_transfers;
5000000230:created
create_transfers;
5000000232:created
create_transfers;
5000000234:created
create_transfers;
5000000236:created
create_transfers;
5000000238:created
create_transfers;
5000000240:created
create_transfers;
5000000242:created
query_transfers ts=[0,0] limit=50 reversed=0;
1:7:500:0:1:2:1:1:0:2
2:9:100:0:1:2:1:1:0:0
4:14:500:1:1:2:1:1:0:4
22:15:1:0:1:2:1:1:0:256
24:16:1:0:1:2:1:1:0:258
25:17:1:24:1:2:1:1:0:260
333:63:1234:0:1:2:1:1:2:2
334:65:500:0:1:2:1:1:100:2
340:3000000102:201:334:1:2:1:1:0:4
350:3000000110:777:0:1:2:1:1:200:2
351:3000000112:777:350:1:2:1:1:0:8
360:3000000120:50:0:1:2:1:1:1:66
370:4000000160:100:0:4:2:1:1:1:66
373:5000000199:10:0:4:2:1:1:0:0
380:5000000202:1:0:1:2:1:1:0:0
381:5000000204:2:0:1:2:1:1:0:0
382:5000000206:3:0:1:2:1:1:0:0
383:5000000208:4:0:1:2:1:1:0:0
384:5000000210:5:0:1:2:1:1:0:0
385:5000000214:6:0:1:2:1:1:0:0
386:5000000216:7:0:1:2:1:1:0:0
387:5000000218:8:0:1:2:1:1:0:0
388:5000000220:9:0:1:2:1:1:0:0
389:5000000222:10:0:1:2:1:1:0:0
390:5000000224:11:0:1:2:1:1:0:0
391:5000000226:12:0:1:2:1:1:0:0
392:5000000228:13:0:1:2:1:1:0:0
393:5000000230:14:0:1:2:1:1:0:0
394:5000000232:15:0:1:2:1:1:0:0
get_account_transfers account=1 reversed=0;
1:7:500:0:1:2:1:1:0:2
2:9:100:0:1:2:1:1:0:0
4:14:500:1:1:2:1:1:0:4
22:15:1:0:1:2:1:1:0:256
24:16:1:0:1:2:1:1:0:258
25:17:1:24:1:2:1:1:0:260
333:63:1234:0:1:2:1:1:2:2
334:65:500:0:1:2:1:1:100:2
340:3000000102:201:334:1:2:1:1:0:4
350:3000000110:777:0:1:2:1:1:200:2
351:3000000112:777:350:1:2:1:1:0:8
360:3000000120:50:0:1:2:1:1:1:66
380:5000000202:1:0:1:2:1:1:0:0
381:5000000204:2:0:1:2:1:1:0:0
382:5000000206:3:0:1:2:1:1:0:0
383:5000000208:4:0:1:2:1:1:0:0
384:5000000210:5:0:1:2:1:1:0:0
385:5000000214:6:0:1:2:1:1:0:0
386:5000000216:7:0:1:2:1:1:0:0
387:5000000218:8:0:1:2:1:1:0:0
388:5000000220:9:0:1:2:1:1:0:0
389:5000000222:10:0:1:2:1:1:0:0
390:5000000224:11:0:1:2:1:1:0:0
391:5000000226:12:0:1:2:1:1:0:0
392:5000000228:13:0:1:2:1:1:0:0
393:5000000230:14:0:1:2:1:1:0:0
394:5000000232:15:0:1:2:1:1:0:0
395:5000000234:16:0:1:2:1:1:0:0
396:5000000236:17:0:1:2:1:1:0:0
get_account_balances account=1;
7:500:0:0:0
9:500:100:0:0
14:0:600:0:0
15:0:601:0:0
16:1:601:0:0
17:0:602:0:0
63:1234:602:0:0
65:1734:602:0:0
3000000102:0:803:0:0
3000000110:777:803:0:0
3000000112:0:803:0:0
3000000120:50:803:0:0
5000000202:0:804:0:0
5000000204:0:806:0:0
5000000206:0:809:0:0
5000000208:0:813:0:0
5000000210:0:818:0:0
5000000214:0:824:0:0
5000000216:0:831:0:0
5000000218:0:839:0:0
5000000220:0:848:0:0
5000000222:0:858:0:0
5000000224:0:869:0:0
5000000226:0:881:0:0
5000000228:0:894:0:0
5000000230:0:908:0:0
5000000232:0:923:0:0
5000000234:0:939:0:0
5000000236:0:956:0:0
lookup_transfers n=29;
384:5000000210:5:0:1:2:1:1:0:0
380:5000000202:1:0:1:2:1:1:0:0
399:5000000242:20:0:1:2:1:1:0:0
398:5000000240:19:0:1:2:1:1:0:0
1:7:500:0:1:2:1:1:0:2
2:9:100:0:1:2:1:1:0:0
4:14:500:1:1:2:1:1:0:4
22:15:1:0:1:2:1:1:0:256
24:16:1:0:1:2:1:1:0:258
25:17:1:24:1:2:1:1:0:260
333:63:1234:0:1:2:1:1:2:2
334:65:500:0:1:2:1:1:100:2
340:3000000102:201:334:1:2:1:1:0:4
350:3000000110:777:0:1:2:1:1:200:2
351:3000000112:777:350:1:2:1:1:0:8
360:3000000120:50:0:1:2:1:1:1:66
370:4000000160:100:0:4:2:1:1:1:66
373:5000000199:10:0:4:2:1:1:0:0
381:5000000204:2:0:1:2:1:1:0:0
382:5000000206:3:0:1:2:1:1:0:0
383:5000000208:4:0:1:2:1:1:0:0
385:5000000214:6:0:1:2:1:1:0:0
386:5000000216:7:0:1:2:1:1:0:0
387:5000000218:8:0:1:2:1:1:0:0
388:5000000220:9:0:1:2:1:1:0:0
389:5000000222:10:0:1:2:1:1:0:0
390:5000000224:11:0:1:2:1:1:0:0
391:5000000226:12:0:1:2:1:1:0:0
392:5000000228:13:0:1:2:1:1:0:0
";

    #[test]
    fn accounting_matches_upstream_golden() {
        let mut sm = StateMachine::default();
        let mut out = String::new();

        let a = |id: u128, ledger: u32, code: u16| Account {
            id,
            ledger,
            code,
            flags: AccountFlags::HISTORY,
            ..Account::default()
        };
        let pending_never = |id: u128, amount: u128| Transfer {
            id,
            debit_account_id: 1,
            credit_account_id: 2,
            amount,
            ledger: 1,
            code: 1,
            flags: TransferFlags::PENDING,
            ..Transfer::default()
        };

        // Accounts 1 and 2 (ledger 1) and account 3 (ledger 2, for a
        // cross-ledger failure).
        let a1 = a(1, 1, 1);
        let a2 = a(2, 1, 2);
        let a3 = a(3, 2, 1);

        out.push_str("create_accounts;\n");
        for row in sm.create_accounts(&[a1, a2], 3).as_chunks::<16>().0 {
            let ts = u64::from_le_bytes(row[0..8].try_into().expect("8-byte slice"));
            let status = u32::from_le_bytes(row[8..12].try_into().expect("4-byte slice"));
            let _ = writeln!(out, "{ts}:{}", account_status_name(status));
        }

        out.push_str("create_accounts;\n");
        for row in sm.create_accounts(&[a3], 5).as_chunks::<16>().0 {
            let ts = u64::from_le_bytes(row[0..8].try_into().expect("8-byte slice"));
            let status = u32::from_le_bytes(row[8..12].try_into().expect("4-byte slice"));
            let _ = writeln!(out, "{ts}:{}", account_status_name(status));
        }

        // A pending transfer and a plain transfer.
        let t1 = pending_never(1, 500);
        let t2 = Transfer { id: 2, amount: 100, flags: TransferFlags::default(), ..t1 };
        let t3 = Transfer {
            id: 3,
            debit_account_id: 1,
            amount: 1,
            credit_account_id: 3,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        let bulk =
            |sm: &mut StateMachine, out: &mut String, events: &[Transfer], timestamp: u64| {
                out.push_str("create_transfers;\n");
                for row in sm.create_transfers(events, timestamp).as_chunks::<16>().0 {
                    let ts = u64::from_le_bytes(row[0..8].try_into().expect("8-byte slice"));
                    let status = u32::from_le_bytes(row[8..12].try_into().expect("4-byte slice"));
                    let _ = writeln!(out, "{ts}:{}", transfer_status_name(status));
                }
            };

        bulk(&mut sm, &mut out, &[t1], 7);
        bulk(&mut sm, &mut out, &[t2], 9);
        // Duplicate of t1 (`.exists`) and a ledger-mismatched transfer
        // (`.accounts_must_have_the_same_ledger`).
        bulk(&mut sm, &mut out, &[t1, t3], 12);

        // Post the full pending amount, void of an unknown pending, post an
        // already-posted pending (different id), void a non-pending transfer.
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 4,
                amount: 500,
                pending_id: 1,
                flags: TransferFlags::POST_PENDING_TRANSFER,
                ..Transfer::default()
            }],
            14,
        );
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 5,
                amount: 0,
                pending_id: 9,
                flags: TransferFlags::VOID_PENDING_TRANSFER,
                ..Transfer::default()
            }],
            16,
        );
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 6,
                amount: 500,
                pending_id: 1,
                flags: TransferFlags::POST_PENDING_TRANSFER,
                ..Transfer::default()
            }],
            18,
        );
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 7,
                amount: 0,
                pending_id: 2,
                flags: TransferFlags::VOID_PENDING_TRANSFER,
                ..Transfer::default()
            }],
            20,
        );

        // Id 8 fails with a transient code (credit account 4 never exists);
        // retrying the same id reports `id_already_failed` (orphaned key,
        // state_machine.zig:3215-3252).
        let orphan_fail = Transfer {
            id: 8,
            debit_account_id: 1,
            credit_account_id: 4,
            amount: 1,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        bulk(&mut sm, &mut out, &[orphan_fail], 22);
        bulk(&mut sm, &mut out, &[orphan_fail], 24);

        // Both sides of the accounts, debit for 1 and credit for 2.
        for account_id in [1, 2] {
            let filter = AccountFilter {
                account_id,
                flags: AccountFilterFlags::DEBITS | AccountFilterFlags::CREDITS,
                limit: 100,
                ..AccountFilter::default()
            };
            let _ = writeln!(out, "get_account_balances account={account_id};");
            for row in sm.get_account_balances(&filter).as_chunks::<128>().0 {
                let _ = writeln!(out, "{}", format_balance(row));
            }
        }

        out.push_str("lookup_accounts n=3;\n");
        for acc in
            bytes_to_account_batch(&sm.lookup_accounts(&[1, 3, 99])).expect("valid account batch")
        {
            let _ = writeln!(out, "{}", format_account(&acc));
        }

        out.push_str("lookup_transfers n=4;\n");
        for transfer in bytes_to_transfer_batch(&sm.lookup_transfers(&[1, 2, 4, 9]))
            .expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }

        // Index-scanned queries: debit side of account 1 (ascending), credit
        // side of account 2 (reversed).
        let transfers_filter = |account_id: u128, debits: bool, reversed: bool| AccountFilter {
            account_id,
            flags: if debits { AccountFilterFlags::DEBITS } else { AccountFilterFlags::CREDITS }
                | if reversed {
                    AccountFilterFlags::REVERSED
                } else {
                    AccountFilterFlags::default()
                },
            limit: 100,
            ..AccountFilter::default()
        };
        for (account_id, debits, reversed) in [(1, true, false), (2, false, true)] {
            let filter = transfers_filter(account_id, debits, reversed);
            let _ = writeln!(
                out,
                "get_account_transfers account={account_id} reversed={};",
                reversed as u8
            );
            for transfer in bytes_to_transfer_batch(&sm.get_account_transfers(&filter))
                .expect("valid transfer batch")
            {
                let _ = writeln!(out, "{}", format_transfer(&transfer));
            }
        }

        // Filter queries: full range ascending, and a trimmed descending window.
        let query = |min: u64, max: u64, limit: u32, reversed: bool| QueryFilter {
            timestamp_min: min,
            timestamp_max: max,
            limit,
            flags: if reversed { QueryFilterFlags::REVERSED } else { QueryFilterFlags::default() },
            ..QueryFilter::default()
        };

        let q = query(0, 0, 10, false);
        let _ = writeln!(
            out,
            "query_accounts ts=[{},{}] limit={} reversed={};",
            q.timestamp_min,
            q.timestamp_max,
            q.limit,
            q.flags.reversed() as u8
        );
        for acc in bytes_to_account_batch(&sm.query_accounts(&q)).expect("valid account batch") {
            let _ = writeln!(out, "{}", format_account(&acc));
        }
        let q = query(3, 0, 1, true);
        let _ = writeln!(
            out,
            "query_accounts ts=[{},{}] limit={} reversed={};",
            q.timestamp_min,
            q.timestamp_max,
            q.limit,
            q.flags.reversed() as u8
        );
        for acc in bytes_to_account_batch(&sm.query_accounts(&q)).expect("valid account batch") {
            let _ = writeln!(out, "{}", format_account(&acc));
        }

        let q = query(0, 0, 10, false);
        let _ = writeln!(
            out,
            "query_transfers ts=[{},{}] limit={} reversed={};",
            q.timestamp_min,
            q.timestamp_max,
            q.limit,
            q.flags.reversed() as u8
        );
        for transfer in
            bytes_to_transfer_batch(&sm.query_transfers(&q)).expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }
        let q = query(0, 0, 2, true);
        let _ = writeln!(
            out,
            "query_transfers ts=[{},{}] limit={} reversed={};",
            q.timestamp_min,
            q.timestamp_max,
            q.limit,
            q.flags.reversed() as u8
        );
        for transfer in
            bytes_to_transfer_batch(&sm.query_transfers(&q)).expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }

        // CDC: every recorded change event (pending, single-phase, posted).
        let filter =
            ChangeEventsFilter { timestamp_min: 0, timestamp_max: 0, limit: 10, reserved: [0; 44] };
        let _ = writeln!(
            out,
            "get_change_events ts=[{},{}] limit={};",
            filter.timestamp_min, filter.timestamp_max, filter.limit
        );
        for row in sm.get_change_events(&filter).as_chunks::<384>().0 {
            let event = change_event_at(row, 0);
            let _ = writeln!(out, "{}", format_change_event(&event));
        }

        // Imported events: past timestamps are stored as-is. The batch heads
        // mirror the harness `submit` cadence — a single-event batch's batch
        // head IS the error-slot timestamp pinned above (37/49/53), the
        // two-event batches land at 40/47 — so the slot lines match exactly.
        let imported_account = |id: u128, timestamp: u64| Account {
            id,
            ledger: 1,
            code: 1,
            flags: AccountFlags::IMPORTED,
            timestamp,
            ..Account::default()
        };
        let accounts_batch =
            |sm: &mut StateMachine, out: &mut String, events: &[Account], head: u64| {
                out.push_str("create_accounts;\n");
                for row in sm.create_accounts(events, head).as_chunks::<16>().0 {
                    let ts = u64::from_le_bytes(row[0..8].try_into().expect("8-byte slice"));
                    let status = u32::from_le_bytes(row[8..12].try_into().expect("4-byte slice"));
                    let _ = writeln!(out, "{ts}:{}", account_status_name(status));
                }
            };

        // An account imported at a transfer's timestamp (7) is a cross-index
        // regression.
        accounts_batch(&mut sm, &mut out, &[imported_account(33, 7)], 37);
        // Imported accounts stored at their own timestamps.
        accounts_batch(&mut sm, &mut out, &[imported_account(10, 6), imported_account(20, 30)], 40);
        // Retry: idempotency precedes the regression check (reports `.exists`
        // at the stored timestamp even when the imported one differs).
        accounts_batch(&mut sm, &mut out, &[imported_account(10, 6)], 42);
        accounts_batch(&mut sm, &mut out, &[imported_account(20, 29)], 44);
        // A within-batch regression against the running key max (35 < 40).
        accounts_batch(
            &mut sm,
            &mut out,
            &[imported_account(31, 40), imported_account(32, 35)],
            47,
        );

        let imported_transfer = |id: u128,
                                 dr: u128,
                                 cr: u128,
                                 amount: u128,
                                 pending_id: u128,
                                 ledger: u32,
                                 code: u16,
                                 flags: TransferFlags,
                                 timestamp: u64| Transfer {
            id,
            debit_account_id: dr,
            credit_account_id: cr,
            amount,
            pending_id,
            ledger,
            code,
            flags: flags | TransferFlags::IMPORTED,
            timestamp,
            ..Transfer::default()
        };
        // Below the existing transfer key max (14) is a regression; at 30 an
        // imported account's timestamp collides; 15 stores as-is.
        bulk(
            &mut sm,
            &mut out,
            &[imported_transfer(21, 1, 2, 1, 0, 1, 1, TransferFlags::default(), 12)],
            49,
        );
        bulk(
            &mut sm,
            &mut out,
            &[imported_transfer(22, 1, 2, 1, 0, 1, 1, TransferFlags::default(), 15)],
            51,
        );
        bulk(
            &mut sm,
            &mut out,
            &[imported_transfer(23, 1, 2, 1, 0, 1, 1, TransferFlags::default(), 30)],
            53,
        );
        // An imported pending (timeout 0 required) and its imported posting.
        bulk(
            &mut sm,
            &mut out,
            &[imported_transfer(24, 1, 2, 1, 0, 1, 1, TransferFlags::PENDING, 16)],
            55,
        );
        bulk(
            &mut sm,
            &mut out,
            &[imported_transfer(
                25,
                0,
                0,
                u128::MAX,
                24,
                0,
                0,
                TransferFlags::POST_PENDING_TRANSFER,
                17,
            )],
            57,
        );

        // The imported records are stored at their own timestamps (account
        // flags 16 = imported; the post carries the pending's dr/cr/ledger/
        // code), account 1's balance moved per event, and the CDC dump grew.
        out.push_str("lookup_accounts n=2;\n");
        for acc in
            bytes_to_account_batch(&sm.lookup_accounts(&[10, 20])).expect("valid account batch")
        {
            let _ = writeln!(out, "{}", format_account(&acc));
        }
        out.push_str("lookup_transfers n=3;\n");
        for transfer in bytes_to_transfer_batch(&sm.lookup_transfers(&[24, 25, 22]))
            .expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }
        let account_balances = |sm: &mut StateMachine, out: &mut String, account_id: u128| {
            let filter = AccountFilter {
                account_id,
                limit: 100,
                flags: AccountFilterFlags::DEBITS | AccountFilterFlags::CREDITS,
                ..AccountFilter::default()
            };
            let _ = writeln!(out, "get_account_balances account={account_id};");
            for row in sm.get_account_balances(&filter).as_chunks::<128>().0 {
                let _ = writeln!(out, "{}", format_balance(row));
            }
        };
        account_balances(&mut sm, &mut out, 1);
        let filter =
            ChangeEventsFilter { timestamp_min: 0, timestamp_max: 0, limit: 10, reserved: [0; 44] };
        let _ = writeln!(
            out,
            "get_change_events ts=[{},{}] limit={};",
            filter.timestamp_min, filter.timestamp_max, filter.limit
        );
        for row in sm.get_change_events(&filter).as_chunks::<384>().0 {
            let event = change_event_at(row, 0);
            let _ = writeln!(out, "{}", format_change_event(&event));
        }

        // Expiry: a short-lived pending (333, 2s) and a long-lived one (334,
        // 100s). A pulse at 3_000_000_096 expires only 333, releasing its
        // holds (both sides down by 1234); 334 stays pending. The balance
        // history omits the expiry event (no transfer at its timestamp), but
        // the CDC records it.
        bulk(&mut sm, &mut out, &[Transfer { timeout: 2, ..pending_never(333, 1234) }], 63);
        bulk(&mut sm, &mut out, &[Transfer { timeout: 100, ..pending_never(334, 500) }], 65);

        out.push_str("pulse at 3000000096;\n");
        let reply = sm.execute(Operation::STATE_MACHINE_PULSE, 3_000_000_096, &[]);
        assert!(reply.is_empty());

        out.push_str("lookup_transfers n=2;\n");
        for transfer in bytes_to_transfer_batch(&sm.lookup_transfers(&[333, 334]))
            .expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }
        account_balances(&mut sm, &mut out, 1);
        account_balances(&mut sm, &mut out, 2);
        let filter =
            ChangeEventsFilter { timestamp_min: 0, timestamp_max: 0, limit: 10, reserved: [0; 44] };
        let _ = writeln!(
            out,
            "get_change_events ts=[{},{}] limit={};",
            filter.timestamp_min, filter.timestamp_max, filter.limit
        );
        for row in sm.get_change_events(&filter).as_chunks::<384>().0 {
            let event = change_event_at(row, 0);
            let _ = writeln!(out, "{}", format_change_event(&event));
        }

        // Post/void after expiry: a partial post (201 of the 500 pending)
        // closes 334 — the remainder is released, not posted; the expired 333
        // and the now-posted 334 reject (post and void have different rules):
        // voiding 334 for 300 fails before the status check (amount must
        // equal the full pending), and re-posting fails with
        // `pending_transfer_already_posted`. A fresh 777 pending (350) is then
        // fully voided (successful `two_phase_voided` pin) and re-voided.
        let posting = |id: u128, amount: u128, pending_id: u128| Transfer {
            id,
            amount,
            pending_id,
            flags: TransferFlags::POST_PENDING_TRANSFER,
            ..Transfer::default()
        };
        let voiding = |id: u128, amount: u128, pending_id: u128| Transfer {
            id,
            amount,
            pending_id,
            flags: TransferFlags::VOID_PENDING_TRANSFER,
            ..Transfer::default()
        };
        bulk(&mut sm, &mut out, &[posting(340, 201, 334)], 3_000_000_102);
        bulk(&mut sm, &mut out, &[posting(341, 1234, 333)], 3_000_000_104);
        bulk(&mut sm, &mut out, &[voiding(342, 300, 334)], 3_000_000_106);
        bulk(&mut sm, &mut out, &[posting(344, 1, 334)], 3_000_000_108);
        bulk(
            &mut sm,
            &mut out,
            &[Transfer { timeout: 200, ..pending_never(350, 777) }],
            3_000_000_110,
        );
        bulk(&mut sm, &mut out, &[voiding(351, 777, 350)], 3_000_000_112);
        bulk(&mut sm, &mut out, &[voiding(352, 777, 350)], 3_000_000_114);

        out.push_str("lookup_transfers n=7;\n");
        for transfer in
            bytes_to_transfer_batch(&sm.lookup_transfers(&[340, 341, 342, 344, 350, 351, 352]))
                .expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }
        account_balances(&mut sm, &mut out, 1);
        account_balances(&mut sm, &mut out, 2);
        // Upstream caps the change-events reply at the message-body
        // `result_max` (10 events in `test_min`) — the two newest events
        // (350/351) fall beyond it even though `limit` is 15
        // (state_machine.zig:2245-2275).
        let filter =
            ChangeEventsFilter { timestamp_min: 0, timestamp_max: 0, limit: 15, reserved: [0; 44] };
        let _ = writeln!(
            out,
            "get_change_events ts=[{},{}] limit={};",
            filter.timestamp_min, filter.timestamp_max, filter.limit
        );
        for row in sm.get_change_events(&filter).as_chunks::<384>().0 {
            let event = change_event_at(row, 0);
            let _ = writeln!(out, "{}", format_change_event(&event));
        }

        // A pending that closes its debit account: 360 debits account 1 with
        // `closing_debit`, so account 1 reports CLOSED (40 = HISTORY|CLOSED)
        // while the hold is outstanding; the expiry full-clear reopens it (8)
        // and the hold returns to the pool. Balance history omits the expiry
        // row (as if the pending never existed); the change-event window pins
        // the `two_phase_expired` canonicalized via `transfer_pending_id`.
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                timeout: 1,
                flags: TransferFlags::PENDING | TransferFlags::CLOSING_DEBIT,
                ..pending_never(360, 50)
            }],
            3_000_000_120,
        );
        out.push_str("lookup_accounts n=1;\n");
        for acc in bytes_to_account_batch(&sm.lookup_accounts(&[1])).expect("valid account batch") {
            let _ = writeln!(out, "{}", format_account(&acc));
        }
        out.push_str("pulse at 4000000152;\n");
        let reply = sm.execute(Operation::STATE_MACHINE_PULSE, 4_000_000_152, &[]);
        assert!(reply.is_empty());
        out.push_str("lookup_accounts n=1;\n");
        for acc in bytes_to_account_batch(&sm.lookup_accounts(&[1])).expect("valid account batch") {
            let _ = writeln!(out, "{}", format_account(&acc));
        }
        account_balances(&mut sm, &mut out, 1);
        account_balances(&mut sm, &mut out, 2);
        let filter = ChangeEventsFilter {
            timestamp_min: 3_000_000_116,
            timestamp_max: 0,
            limit: 15,
            reserved: [0; 44],
        };
        let _ = writeln!(
            out,
            "get_change_events ts=[{},{}] limit={};",
            filter.timestamp_min, filter.timestamp_max, filter.limit
        );
        for row in sm.get_change_events(&filter).as_chunks::<384>().0 {
            let event = change_event_at(row, 0);
            let _ = writeln!(out, "{}", format_change_event(&event));
        }

        // Account-level closure: a `closing_debit` pending (370) closes
        // account 4 as 360 closed account 1; while it, single-phase
        // debits/credits into account 4 are rejected
        // (`debit_account_already_closed` / `credit_account_already_closed`).
        // The expiry full-clear reopens 4 (flags back to 8) and a plain
        // transfer then succeeds; account 4's balance history shows the hold
        // and the plain transfer, but not the expiry.
        out.push_str("create_accounts;\n");
        for row in sm.create_accounts(&[a(4, 1, 1)], 4_000_000_158).as_chunks::<16>().0 {
            let ts = u64::from_le_bytes(row[0..8].try_into().expect("8-byte slice"));
            let status = u32::from_le_bytes(row[8..12].try_into().expect("4-byte slice"));
            let _ = writeln!(out, "{ts}:{}", account_status_name(status));
        }
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 370,
                debit_account_id: 4,
                credit_account_id: 2,
                amount: 100,
                ledger: 1,
                code: 1,
                timeout: 1,
                flags: TransferFlags::PENDING | TransferFlags::CLOSING_DEBIT,
                ..Transfer::default()
            }],
            4_000_000_160,
        );
        out.push_str("lookup_accounts n=1;\n");
        for acc in bytes_to_account_batch(&sm.lookup_accounts(&[4])).expect("valid account batch") {
            let _ = writeln!(out, "{}", format_account(&acc));
        }
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 371,
                debit_account_id: 4,
                credit_account_id: 2,
                amount: 10,
                ledger: 1,
                code: 1,
                ..Transfer::default()
            }],
            4_000_000_163,
        );
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 372,
                debit_account_id: 2,
                credit_account_id: 4,
                amount: 10,
                ledger: 1,
                code: 1,
                ..Transfer::default()
            }],
            4_000_000_165,
        );
        out.push_str("pulse at 5000000196;\n");
        let reply = sm.execute(Operation::STATE_MACHINE_PULSE, 5_000_000_196, &[]);
        assert!(reply.is_empty());
        out.push_str("lookup_accounts n=1;\n");
        for acc in bytes_to_account_batch(&sm.lookup_accounts(&[4])).expect("valid account batch") {
            let _ = writeln!(out, "{}", format_account(&acc));
        }
        bulk(
            &mut sm,
            &mut out,
            &[Transfer {
                id: 373,
                debit_account_id: 4,
                credit_account_id: 2,
                amount: 10,
                ledger: 1,
                code: 1,
                ..Transfer::default()
            }],
            5_000_000_199,
        );
        account_balances(&mut sm, &mut out, 4);

        // Multi-batch reply caps. `test_min` message body =
        // `message_size_max_min(4)` - 256-byte header = alignForward(5*256,
        // 4096) - 256 = 3840. `get_account_balances` and `get_account_transfers`
        // are MULTI-batch ops like the queries (tigerbeetle.zig `is_multi_batch`
        // 819-833), so their caps subtract the one-batch trailer
        // `div_ceil(4, 128)*128` = 128: (3840-128)/128 = 29. The port got this
        // wrong twice: originally capping balances at `size_of::<AccountEvent>`
        // (256) → 15, caught in c7e43a8 by the 17-row dump below; then at the
        // plain `result_max` (30) instead of the multi-batch variant — caught
        // here. Transfers 380-384 (17-balance dump) plus 385-399 raise account
        // 1 to 32 events, so the capped dumps that follow all pin 29 rows
        // exactly (and query_transfers, with 34 transfers, 29 as well).
        let debit1 = |id: u128, amount: u128| Transfer {
            id,
            debit_account_id: 1,
            credit_account_id: 2,
            amount,
            ledger: 1,
            code: 1,
            ..Transfer::default()
        };
        bulk(&mut sm, &mut out, &[debit1(380, 1)], 5_000_000_202);
        bulk(&mut sm, &mut out, &[debit1(381, 2)], 5_000_000_204);
        bulk(&mut sm, &mut out, &[debit1(382, 3)], 5_000_000_206);
        bulk(&mut sm, &mut out, &[debit1(383, 4)], 5_000_000_208);
        bulk(&mut sm, &mut out, &[debit1(384, 5)], 5_000_000_210);
        account_balances(&mut sm, &mut out, 1);
        let filter = AccountFilter {
            account_id: 1,
            timestamp_min: 5_000_000_202,
            timestamp_max: 0,
            limit: 100,
            flags: AccountFilterFlags::DEBITS | AccountFilterFlags::CREDITS,
            ..AccountFilter::default()
        };
        let _ = writeln!(
            out,
            "get_account_balances account={} ts=[{},0] limit=100;",
            filter.account_id, filter.timestamp_min
        );
        for row in sm.get_account_balances(&filter).as_chunks::<128>().0 {
            let _ = writeln!(out, "{}", format_balance(row));
        }

        bulk(&mut sm, &mut out, &[debit1(385, 6)], 5_000_000_214);
        bulk(&mut sm, &mut out, &[debit1(386, 7)], 5_000_000_216);
        bulk(&mut sm, &mut out, &[debit1(387, 8)], 5_000_000_218);
        bulk(&mut sm, &mut out, &[debit1(388, 9)], 5_000_000_220);
        bulk(&mut sm, &mut out, &[debit1(389, 10)], 5_000_000_222);
        bulk(&mut sm, &mut out, &[debit1(390, 11)], 5_000_000_224);
        bulk(&mut sm, &mut out, &[debit1(391, 12)], 5_000_000_226);
        bulk(&mut sm, &mut out, &[debit1(392, 13)], 5_000_000_228);
        bulk(&mut sm, &mut out, &[debit1(393, 14)], 5_000_000_230);
        bulk(&mut sm, &mut out, &[debit1(394, 15)], 5_000_000_232);
        bulk(&mut sm, &mut out, &[debit1(395, 16)], 5_000_000_234);
        bulk(&mut sm, &mut out, &[debit1(396, 17)], 5_000_000_236);
        bulk(&mut sm, &mut out, &[debit1(397, 18)], 5_000_000_238);
        bulk(&mut sm, &mut out, &[debit1(398, 19)], 5_000_000_240);
        bulk(&mut sm, &mut out, &[debit1(399, 20)], 5_000_000_242);
        let q = query(0, 0, 50, false);
        let _ = writeln!(
            out,
            "query_transfers ts=[{},{}] limit={} reversed={};",
            q.timestamp_min,
            q.timestamp_max,
            q.limit,
            q.flags.reversed() as u8
        );
        for transfer in
            bytes_to_transfer_batch(&sm.query_transfers(&q)).expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }
        let filter = AccountFilter {
            account_id: 1,
            limit: 100,
            flags: AccountFilterFlags::DEBITS | AccountFilterFlags::CREDITS,
            ..AccountFilter::default()
        };
        let _ = writeln!(out, "get_account_transfers account={} reversed=0;", filter.account_id);
        for transfer in bytes_to_transfer_batch(&sm.get_account_transfers(&filter))
            .expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }
        account_balances(&mut sm, &mut out, 1);

        // Upstream fills lookup replies in REQUEST order (execute_lookup_transfers
        // iterates the request ids, state_machine.zig:3286-3297), which the
        // earlier small lookups could not distinguish from timestamp order.
        // Ids 384 (ts 210) before 380 (ts 202) and 399 (ts 242) before 398
        // (ts 240) pin it. 29 is the harness maximum — its submit MultiBatch-
        // encodes even lookups, so the 16-byte u128 trailer inflates the count
        // over `event_max` (30) with 30 ids.
        out.push_str("lookup_transfers n=29;\n");
        for transfer in bytes_to_transfer_batch(&sm.lookup_transfers(&[
            384, 380, 399, 398, 1, 2, 4, 22, 24, 25, 333, 334, 340, 350, 351, 360, 370, 373, 381,
            382, 383, 385, 386, 387, 388, 389, 390, 391, 392,
        ]))
        .expect("valid transfer batch")
        {
            let _ = writeln!(out, "{}", format_transfer(&transfer));
        }

        assert_eq!(out, GOLDEN_ACCOUNTING);
    }
}
