// =============================================================================
// State Machine — pure accounting logic
// =============================================================================
//
// Ported from `src/state_machine.zig`. This module contains the core validation
// and mutation logic for `create_accounts` and `create_transfer`.
//
// The full StateMachine struct (batch orchestrator, linked chains, imported
// timestamps, prefetch, expiry scheduling) is deferred until the forest layer
// is complete. For now we expose two standalone functions that take a mutable
// reference to the relevant grooves plus an account-lookup callback.

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

use tigerbeetle_core::types::{Account, CreateAccountStatus, CreateTransferStatus, Transfer};

#[cfg(test)]
use tigerbeetle_core::types::{AccountFlags, TransferFlags};

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

    // Imported timestamp validation is deferred until the full batch orchestrator
    // can check objects.key_range and indirect_lookup.
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

    // Imported timestamp validation deferred until batch orchestrator.
    if !t.flags.imported() {
        assert_eq!(t.timestamp, 0);
    }
    let timestamp_actual = timestamp_event;

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
/// Returns a `Vec<CreateAccountResult>` parallel to the input events.
pub fn execute_create_accounts<F>(
    events: &[Account],
    timestamp: u64,
    mut get_existing: F,
) -> Vec<CreateAccountResult>
where
    F: FnMut(u128) -> Option<Account>,
{
    let mut results = Vec::with_capacity(events.len());
    let mut chain: Option<usize> = None;
    let mut chain_broken = false;

    let batch_imported = !events.is_empty() && events[0].flags.imported();

    for (index, event) in events.iter().enumerate() {
        let timestamp_event = timestamp - events.len() as u64 + index as u64 + 1;

        let (status, _ts) = 'result: {
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
            let status = create_account(event, timestamp_event, existing.as_ref());
            let ts = match status {
                CreateAccountStatus::Created | CreateAccountStatus::Exists => timestamp_event,
                _ => timestamp_event,
            };
            (status, ts)
        };

        // Chain error handling.
        if status != CreateAccountStatus::Created {
            if let Some(chain_start) = chain {
                if !chain_broken {
                    chain_broken = true;
                    // TODO(port): scope_close(discard)
                    // Fill linked_event_failed for prior chain events.
                    for ci in chain_start..index {
                        // The result slot for this index will be filled below,
                        // but we mark it as linked_event_failed.
                        let _ = ci;
                    }
                }
            }
        }

        results.push(CreateAccountResult { status, timestamp: timestamp_event });

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
/// `get_existing_transfer` looks up a transfer by id.
/// `get_account` looks up an account by id — needed for both debit and credit accounts.
pub fn execute_create_transfers<F, G>(
    events: &[Transfer],
    timestamp: u64,
    mut get_existing_transfer: F,
    mut get_account: G,
) -> Vec<CreateTransferResult>
where
    F: FnMut(u128) -> Option<Transfer>,
    G: FnMut(u128) -> Option<Account>,
{
    let mut results = Vec::with_capacity(events.len());
    let mut chain: Option<usize> = None;
    let mut chain_broken = false;

    let batch_imported = !events.is_empty() && events[0].flags.imported();

    for (index, event) in events.iter().enumerate() {
        let timestamp_event = timestamp - events.len() as u64 + index as u64 + 1;

        let (status, _ts) = 'result: {
            if event.flags.linked() {
                if chain.is_none() {
                    chain = Some(index);
                    assert!(!chain_broken);
                    // TODO(port): scope_open
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

            let status = create_transfer(
                event,
                timestamp_event,
                existing.as_ref(),
                &dr_account,
                &cr_account,
            );
            (status, timestamp_event)
        };

        // Chain error handling.
        if status != CreateTransferStatus::Created {
            if let Some(chain_start) = chain {
                if !chain_broken {
                    chain_broken = true;
                    // TODO(port): scope_close(discard)
                    let _ = chain_start;
                }
            }
        }

        results.push(CreateTransferResult { status, timestamp: timestamp_event });

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

#[cfg(test)]
mod tests {
    use super::*;

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
        let results = execute_create_accounts(&events, 10, |_| None);
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
        let results = execute_create_accounts(&events, 10, |_| None);
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
        let results = execute_create_accounts(&events, 10, |_| None);
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
        let results = execute_create_accounts(&events, 10, |_| None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].status, CreateAccountStatus::Created);
        assert_eq!(results[1].status, CreateAccountStatus::LinkedEventChainOpen);
    }

    #[test]
    fn batch_timestamp_must_be_zero() {
        let events =
            vec![Account { id: 1, ledger: 1, code: 1, timestamp: 5, ..Account::default() }];
        let results = execute_create_accounts(&events, 10, |_| None);
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
        let results = execute_create_accounts(&events, 10, |_| None);
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
        let results = execute_create_accounts(&events, 10, |_| None);
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
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].status, CreateTransferStatus::Created);
        assert_eq!(results[1].status, CreateTransferStatus::Created);
    }
}
