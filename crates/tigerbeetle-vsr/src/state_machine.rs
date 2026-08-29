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

    // Balancing amount: cap amount to the available balance.
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

    // Check pending transfer status.
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

    // Closed accounts: voiding is allowed on closed accounts, posting is not.
    if dr_account.flags.closed() && !is_post {
        return PostVoidPendingResult {
            status: CreateTransferStatus::DebitAccountAlreadyClosed,
            amount_actual,
            is_post,
        };
    }
    if cr_account.flags.closed() && !is_post {
        return PostVoidPendingResult {
            status: CreateTransferStatus::CreditAccountAlreadyClosed,
            amount_actual,
            is_post,
        };
    }

    // After this point, the transfer must succeed.
    PostVoidPendingResult { status: CreateTransferStatus::Created, amount_actual, is_post }
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
/// `get_account` looks up an account by id — needed for both debit and credit accounts.
pub fn execute_create_transfers<F, G, FAccountAt>(
    events: &[Transfer],
    timestamp: u64,
    mut get_existing_transfer: F,
    mut get_account: G,
    transfers_key_max: u64,
    account_with_timestamp: FAccountAt,
) -> Vec<CreateTransferResult>
where
    F: FnMut(u128) -> Option<Transfer>,
    G: FnMut(u128) -> Option<Account>,
    FAccountAt: Fn(u64) -> Option<u128>,
{
    let mut results: Vec<CreateTransferResult> = Vec::with_capacity(events.len());
    let mut chain: Option<usize> = None;
    let mut chain_broken = false;
    let mut running_key_max = transfers_key_max;

    let batch_imported = !events.is_empty() && events[0].flags.imported();

    for (index, event) in events.iter().enumerate() {
        let timestamp_event = timestamp - events.len() as u64 + index as u64 + 1;

        let (status, ts) = 'result: {
            if event.flags.linked() {
                if chain.is_none() {
                    chain = Some(index);
                    assert!(!chain_broken);
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

            // Look up debit/credit accounts.
            let dr = get_account(event.debit_account_id);
            let cr = get_account(event.credit_account_id);

            let (dr_account, cr_account) = match (dr, cr) {
                (Some(d), Some(c)) => (d, c),
                (None, _) => {
                    break 'result (CreateTransferStatus::DebitAccountNotFound, timestamp_event);
                }
                (_, None) => {
                    break 'result (CreateTransferStatus::CreditAccountNotFound, timestamp_event);
                }
            };

            // Imported-timestamp regression/collision checks. Upstream runs these
            // inside `create_transfer` _after_ the idempotency checks
            // (state_machine.zig:3808-3817) and before the postdate checks, so an
            // existing record short-circuits to its `exists` code first and the
            // postdate ordering is preserved (error codes must take precedence in
            // the same order). Post/void transfers never run them: upstream routes
            // those through separate functions that do not validate regression.
            if event.flags.imported()
                && existing.is_none()
                && !event.flags.post_pending_transfer()
                && !event.flags.void_pending_transfer()
            {
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

            let status = create_transfer(
                event,
                timestamp_event,
                existing.as_ref(),
                &dr_account,
                &cr_account,
            );
            // Same rule as accounts: `.exists` reports the existing record's
            // timestamp (upstream `state_machine.zig:3669-3721`).
            let ts = match status {
                CreateTransferStatus::Created if event.flags.imported() => event.timestamp,
                CreateTransferStatus::Created => timestamp_event,
                CreateTransferStatus::Exists => match existing {
                    Some(existing) => existing.timestamp,
                    None => {
                        unreachable!("create_transfer returns Exists only for a matching record")
                    }
                },
                _ => timestamp_event,
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
                    for result in &mut results[chain_start..index] {
                        result.status = CreateTransferStatus::LinkedEventFailed;
                    }
                }
            }
        }

        results.push(CreateTransferResult { status, timestamp: ts });

        // Chain completion.
        if chain.is_some()
            && (!event.flags.linked() || status == CreateTransferStatus::LinkedEventChainOpen)
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
    /// DEVIATION: pending/imported balancing (balance mutations, expiry) are
    /// deferred with the grooves.
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

    /// Persist the transfers that upstream would write under the chain scope.
    ///
    /// See [`Self::persist_accounts`] for the chain semantics.
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
                    self.insert_transfer(event.id, event);
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
                    self.insert_transfer(event.id, event);
                }
            }
            index += 1;
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
}
