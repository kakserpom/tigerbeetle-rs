//! Port of `src/vsr/journal.zig` — part 1: ring/slot geometry and the WAL recovery
//! decision table.
//!
#![allow(clippy::cast_possible_truncation)] // slot indices derive from op % slot_count < SLOT_COUNT

//! TODO(port): src/vsr/journal.zig — the IOPS-pooled read/write paths
//! (`read_prepare*`, `recover*`, `write_prepare*`, `torn_prepares`) require the Replica
//! and message bus and are ported together with them.

use crate::message_header::{self, TypedHeader};

/// A slot is an index within:
///
/// - the on-disk headers ring
/// - the on-disk prepares ring
/// - `journal.headers`
/// - `journal.headers_redundant`
/// - `journal.dirty`
/// - `journal.faulty`
///
/// A header's slot is `header.op % constants.journal_slot_count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub index: usize,
}

impl Slot {
    /// (upstream: `Journal.slot_for_op`, which ignores its receiver).
    #[must_use]
    pub fn for_op(op: u64) -> Self {
        Slot { index: (op % SLOT_COUNT as u64) as usize }
    }
}

/// An inclusive, non-empty range of slots.
#[derive(Clone, Copy, Debug)]
pub struct SlotRange {
    pub head: Slot,
    pub tail: Slot,
}

impl SlotRange {
    /// Returns whether this range (inclusive) includes the specified slot.
    ///
    /// Cases (`·`=included, ` `=excluded):
    ///
    /// ```text
    /// head < tail  →   head··tail
    /// head > tail  → ··tail  head··   (The range wraps around).
    /// ```
    ///
    /// # Panics
    /// Asserts the range is non-empty (`head != tail`); the caller must handle the empty
    /// range separately.
    #[must_use]
    pub fn contains(&self, slot: Slot) -> bool {
        // To avoid confusion, the empty range must be checked separately by the caller.
        assert_ne!(self.head.index, self.tail.index);

        if self.head.index < self.tail.index {
            return self.head.index <= slot.index && slot.index <= self.tail.index;
        }
        // The range wraps around:
        slot.index <= self.tail.index || self.head.index <= slot.index
    }
}

/// The WAL consists of two contiguous circular buffers on disk:
/// - [`crate::Zone::WalHeaders`]
/// - [`crate::Zone::WalPrepares`]
///
/// In each ring, the `op` for reserved headers is set to the corresponding slot index.
/// This helps WAL recovery detect misdirected reads/writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ring {
    /// A circular buffer of (redundant) prepare message headers.
    Headers,
    /// A circular buffer of prepare messages. Each slot is padded to `MESSAGE_SIZE_MAX`.
    Prepares,
}

impl Ring {
    /// Returns the slot's offset relative to the start of the ring.
    ///
    /// # Panics
    /// Asserts the slot is within range and the computed offset stays inside the ring.
    #[must_use]
    pub fn offset(self, slot: Slot) -> u64 {
        assert!(slot.index < SLOT_COUNT);
        match self {
            Ring::Headers => {
                let ring_offset = sector_floor(slot.index * HEADER_SIZE);
                assert!(ring_offset < HEADERS_SIZE as u64);
                ring_offset
            }
            Ring::Prepares => {
                let ring_offset = MESSAGE_SIZE_MAX as u64 * slot.index as u64;
                assert!(ring_offset < PREPARES_SIZE as u64);
                ring_offset
            }
        }
    }
}

pub const SLOT_COUNT: usize = tigerbeetle_core::constants::JOURNAL_SLOT_COUNT as usize;
const HEADERS_SIZE: usize = tigerbeetle_core::constants::JOURNAL_SIZE_HEADERS;
const PREPARES_SIZE: usize = tigerbeetle_core::constants::JOURNAL_SIZE_PREPARES;
const MESSAGE_SIZE_MAX: usize = tigerbeetle_core::constants::MESSAGE_SIZE_MAX as usize;
const SECTOR_SIZE: usize = tigerbeetle_core::constants::SECTOR_SIZE;

/// `headers_size + prepares_size` (upstream: `write_ahead_log_zone_size`).
pub const WRITE_AHEAD_LOG_ZONE_SIZE: usize = HEADERS_SIZE + PREPARES_SIZE;

const HEADER_SIZE: usize = message_header::SIZE;
const HEADERS_PER_SECTOR: usize = SECTOR_SIZE / HEADER_SIZE;
const HEADERS_PER_MESSAGE: usize = MESSAGE_SIZE_MAX / HEADER_SIZE;

const _: () = {
    assert!(HEADERS_PER_SECTOR > 0);
    assert!(HEADERS_PER_MESSAGE > 0);
    assert!(SECTOR_SIZE.is_multiple_of(HEADER_SIZE));

    assert!(SLOT_COUNT > 0);
    assert!(SLOT_COUNT.is_multiple_of(2));
    assert!(SLOT_COUNT.is_multiple_of(HEADERS_PER_SECTOR));
    assert!(SLOT_COUNT >= HEADERS_PER_SECTOR);

    assert!(HEADERS_SIZE > 0);
    assert!(HEADERS_SIZE.is_multiple_of(SECTOR_SIZE));
    // It's important that the replica doesn't write all redundant headers simultaneously.
    // Otherwise, a crash could lead to a series of torn writes making the entire journal
    // faulty. Rather than adding simulator-only locking to the journal, the simulator
    // itself prevents correlated torn writes at runtime, and we just exclude non-production
    // configs from the assert:
    assert!(
        HEADERS_SIZE / SECTOR_SIZE
            > tigerbeetle_core::constants::CONFIG.process.journal_iops_write_max as usize
            || !tigerbeetle_core::constants::CONFIG.is_production()
    );

    assert!(PREPARES_SIZE > 0);
    assert!(PREPARES_SIZE.is_multiple_of(SECTOR_SIZE));
    assert!(PREPARES_SIZE.is_multiple_of(MESSAGE_SIZE_MAX));
};

/// Floor to a logical sector boundary (upstream: `vsr.sector_floor`).
fn sector_floor(size: usize) -> u64 {
    (size - size % SECTOR_SIZE) as u64
}

/// Returns the header, only if the header:
/// * has a valid checksum, and
/// * has command=prepare (guaranteed statically by the [`message_header::Prepare`] type),
/// * has the expected cluster, and
/// * resides in the correct slot.
///
/// A header with the wrong cluster, or in the wrong slot, may indicate a misdirected
/// read/write. All journalled headers should be reserved or else prepares.
// DEVIATION: upstream takes `*const Header.Prepare` and re-checks `command`; our typed
// `Prepare` cannot hold another command, so that check is unrepresentable.
#[must_use]
pub fn header_ok(
    cluster: u128,
    slot: Slot,
    header: &message_header::Prepare,
) -> Option<message_header::Prepare> {
    // We must first validate the header checksum before accessing any fields.
    // Otherwise, we may hit undefined data or an out-of-bounds enum and cause a crash.
    if !header.valid_checksum() {
        return None;
    }

    let valid_cluster_and_slot = if header.operation == crate::Operation::RESERVED {
        header.cluster == cluster && slot.index as u64 == header.op
    } else {
        header.cluster == cluster && slot.index as u64 == header.op % SLOT_COUNT as u64
    };

    // Do not check the checksum here, because that would run only after the other field
    // accesses.
    if valid_cluster_and_slot { Some(*header) } else { None }
}

/// Returns the highest op number prepared, as per [`header_ok`], in untrusted headers
/// (upstream: `op_maximum_headers_untrusted`).
#[must_use]
pub fn op_maximum_headers_untrusted(
    cluster: u128,
    headers_untrusted: &[message_header::Prepare],
) -> u64 {
    let mut op: u64 = 0;
    for (slot_index, header_untrusted) in headers_untrusted.iter().enumerate() {
        let slot = Slot { index: slot_index };
        if let Some(header) = header_ok(cluster, slot, header_untrusted)
            .filter(|header| header.operation != crate::Operation::RESERVED && header.op > op)
        {
            op = header.op;
        }
    }
    op
}

/// The decision the recovery process makes for one slot (upstream: `RecoveryDecision`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryDecision {
    /// The header and prepare are identical; no repair necessary.
    Eql,
    /// Reserved; dirty/faulty are clear, no repair necessary.
    Nil,
    /// Use intact prepare to repair redundant header. Dirty/faulty are clear.
    Fix,
    /// If replica_count>1 or standby: repair with VSR `get_prepare`. Mark dirty, mark faulty.
    /// If replica_count=1 and !standby: fail; cannot recover safely.
    Vsr,
    /// The prepare is from the next checkpoint. Truncate, set to reserved, clear dirty/faulty.
    Cut,
    /// Truncate the op, setting it to reserved. Dirty/faulty are clear.
    CutTorn,
    /// Unreachable combination of header and prepare states.
    Unr,
}

/// One pattern atom of a recovery case (upstream: `Matcher`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Matcher {
    Any,
    IsFalse,
    IsTrue,
    AssertIsFalse,
    AssertIsTrue,
}

/// Number of compared properties per case (upstream: `Case.pattern_size`).
pub const PATTERN_SIZE: usize = 11;

/// Error returned by [`Case::check`] when an assertion-pattern (`a0`/`a1`) is violated —
/// the replica would abort on such state upstream.
#[derive(Debug)]
enum CaseCheckError {
    ExpectTrue,
    ExpectFalse,
}

/// One row of the recovery decision table (upstream: `Case`).
pub struct Case {
    pub label: &'static str,
    /// Decision when replica_count>1.
    decision_multiple: RecoveryDecision,
    /// Decision when replica_count=1.
    decision_single: RecoveryDecision,
    /// 0: header_ok(header)
    /// 1: header.operation == reserved
    /// 2: header_ok(prepare) ∧ valid_checksum_body
    /// 3: prepare.operation == reserved
    /// 4: prepare.op is maximum of all prepare.ops
    /// 5: prepare.op > op_prepare_max
    /// 6: header.op > op_prepare_max
    /// 7: header.checksum == prepare.checksum
    /// 8: header.op == prepare.op
    /// 9: header.op < prepare.op
    /// 10: header.view == prepare.view
    pattern: [Matcher; PATTERN_SIZE],
}

impl Case {
    const fn new(
        label: &'static str,
        decision_multiple: RecoveryDecision,
        decision_single: RecoveryDecision,
        pattern: [Matcher; PATTERN_SIZE],
    ) -> Self {
        Self { label, decision_multiple, decision_single, pattern }
    }

    /// Returns whether `parameters` satisfies this case's pattern.
    fn check(&self, parameters: [bool; PATTERN_SIZE]) -> Result<bool, CaseCheckError> {
        for (&pattern, &parameter) in self.pattern.iter().zip(parameters.iter()) {
            match pattern {
                Matcher::Any => {}
                Matcher::IsFalse => {
                    if parameter {
                        return Ok(false);
                    }
                }
                Matcher::IsTrue => {
                    if !parameter {
                        return Ok(false);
                    }
                }
                Matcher::AssertIsFalse => {
                    if parameter {
                        return Err(CaseCheckError::ExpectFalse);
                    }
                }
                Matcher::AssertIsTrue => {
                    if !parameter {
                        return Err(CaseCheckError::ExpectTrue);
                    }
                }
            }
        }
        Ok(true)
    }

    #[must_use]
    pub fn decision(&self, solo: bool) -> RecoveryDecision {
        if solo { self.decision_single } else { self.decision_multiple }
    }
}

mod cases {
    use super::{Case, Matcher, RecoveryDecision};

    // Mnemonics kept for easy diffing against upstream's table:
    const __: Matcher = Matcher::Any;
    const _0: Matcher = Matcher::IsFalse;
    const _1: Matcher = Matcher::IsTrue;
    // The replica will abort if any of these checks fail:
    const A0: Matcher = Matcher::AssertIsFalse;
    const A1: Matcher = Matcher::AssertIsTrue;

    pub const RECOVERY_CASES: [Case; 16] = [
        Case::new(
            "@A",
            RecoveryDecision::Vsr,
            RecoveryDecision::Vsr,
            [_0, __, _0, __, __, A0, A0, __, __, __, __],
        ),
        // @B/@C: this prepare is corrupt. We may have a valid redundant header, but need to
        // recover the full message. @B may be caused by crashing while writing the prepare
        // (torn write).
        Case::new(
            "@B",
            RecoveryDecision::Vsr,
            RecoveryDecision::Vsr,
            [_1, _1, _0, __, __, A0, __, __, __, __, __],
        ),
        Case::new(
            "@C",
            RecoveryDecision::Vsr,
            RecoveryDecision::Vsr,
            [_1, _0, _0, __, __, A0, __, __, __, __, __],
        ),
        // @D: possibly a torn write to the redundant headers, so when replica_count=1 we
        // must repair this locally.
        Case::new(
            "@D",
            RecoveryDecision::Vsr,
            RecoveryDecision::Fix,
            [_0, __, _1, _1, __, __, A0, __, __, __, __],
        ),
        // @E: valid prepare, corrupt header (crashed while writing the redundant header,
        // corrupt/misdirected read, or multiple faults).
        Case::new(
            "@E",
            RecoveryDecision::Vsr,
            RecoveryDecision::Fix,
            [_0, __, _1, _0, _0, _0, A0, __, __, __, __],
        ),
        // @F/@G: recovering from a crash after writing the prepare, but before writing the
        // redundant header.
        Case::new(
            "@F",
            RecoveryDecision::Fix,
            RecoveryDecision::Fix,
            [_0, __, _1, _0, _1, _0, A0, __, __, __, __],
        ),
        Case::new(
            "@G",
            RecoveryDecision::Fix,
            RecoveryDecision::Fix,
            [_1, _1, _1, _0, __, _0, __, __, __, __, __],
        ),
        // @H/@I/@J: valid prepare/header past the prepare_max for the replica's checkpoint —
        // truncate so all prepares in the checkpoint can be replayed.
        // @H: prepare.op > op_prepare_max.
        Case::new(
            "@H",
            RecoveryDecision::Cut,
            RecoveryDecision::Unr,
            [__, __, _1, _0, __, _1, __, __, __, __, __],
        ),
        // @I: header.op > op_prepare_max, prepare !reserved.
        Case::new(
            "@I",
            RecoveryDecision::Cut,
            RecoveryDecision::Unr,
            [_1, _0, _1, _0, __, _0, _1, __, __, A0, __],
        ),
        // @J: header.op > op_prepare_max, prepare reserved.
        Case::new(
            "@J",
            RecoveryDecision::Cut,
            RecoveryDecision::Unr,
            [_1, _0, _1, _1, __, __, _1, __, __, A0, __],
        ),
        // @K: redundant header present & valid, but the corresponding prepare was a lost or
        // misdirected read/write.
        Case::new(
            "@K",
            RecoveryDecision::Vsr,
            RecoveryDecision::Vsr,
            [_1, _0, _1, _1, __, __, _0, __, __, __, __],
        ),
        // @L: legitimately reserved — this may be the first fill of the log.
        Case::new(
            "@L",
            RecoveryDecision::Nil,
            RecoveryDecision::Nil,
            [_1, _1, _1, _1, __, __, __, A1, A1, A0, A1],
        ),
        // @M/@N: both valid but distinct ops — always pick the higher op (@M: repair
        // locally, @N: mark faulty).
        Case::new(
            "@M",
            RecoveryDecision::Fix,
            RecoveryDecision::Fix,
            [
                _1, _0, _1, _0, __, _0, _0, _0, _0, _1, __, // header.op < prepare.op
            ],
        ),
        Case::new(
            "@N",
            RecoveryDecision::Vsr,
            RecoveryDecision::Vsr,
            [
                _1, _0, _1, _0, __, _0, _0, _0, _0, _0, __, // header.op > prepare.op
            ],
        ),
        // @O: views differ (rewrite due to view change / lost prepare write); recovery can't
        // distinguish which is actually newer.
        Case::new(
            "@O",
            RecoveryDecision::Vsr,
            RecoveryDecision::Vsr,
            [
                _1, _0, _1, _0, __, _0, _0, _0, _1, A0, A0, // header.view != prepare.view
            ],
        ),
        // @P: redundant header matches the message's header — the usual, correct case.
        Case::new(
            "@P",
            RecoveryDecision::Eql,
            RecoveryDecision::Eql,
            [_1, _0, _1, _0, __, _0, _0, _1, A1, A0, A1],
        ),
    ];
}

pub use cases::RECOVERY_CASES;

/// The case applied to slots identified by `torn_prepares()` (upstream: `case_cut_torn`).
pub const CASE_CUT_TORN: Case = Case::new(
    "@TruncateTorn",
    RecoveryDecision::CutTorn,
    RecoveryDecision::CutTorn,
    [Matcher::Any; PATTERN_SIZE],
);

/// Extra context used to classify a slot (upstream: the anonymous `data` struct of
/// `recovery_case()`).
#[derive(Clone, Copy, Debug)]
pub struct RecoveryData {
    pub op_max: u64,
    pub op_prepare_max: u64,
    pub op_checkpoint: u64,
}

/// Classifies one recovered slot into exactly one [`Case`] (upstream: `recovery_case`).
///
/// The recovery table is exhaustive: every combination of parameters matches exactly one
/// case.
///
/// # Panics
/// Panics when an assertion-pattern is violated (the replica would abort on such state
/// upstream) or when no/multiple cases match (unreachable by construction).
#[must_use]
pub fn recovery_case(
    header: Option<message_header::Prepare>,
    prepare: Option<message_header::Prepare>,
    data: RecoveryData,
) -> &'static Case {
    let h_reserved = matches!(&header, Some(h) if h.operation == crate::Operation::RESERVED);
    let p_reserved = matches!(&prepare, Some(p) if p.operation == crate::Operation::RESERVED);
    let checksum_match =
        matches!((&header, &prepare), (Some(h), Some(p)) if h.checksum == p.checksum);
    let op_equal = matches!((&header, &prepare), (Some(h), Some(p)) if h.op == p.op);
    let op_less = matches!((&header, &prepare), (Some(h), Some(p)) if h.op < p.op);
    let view_equal = matches!((&header, &prepare), (Some(h), Some(p)) if h.view == p.view);

    let parameters: [bool; PATTERN_SIZE] = [
        header.is_some(),
        h_reserved,
        prepare.is_some(),
        p_reserved,
        matches!(&prepare, Some(p) if p.op == data.op_max),
        matches!(&prepare, Some(p) if p.op > data.op_prepare_max),
        matches!(&header, Some(h) if h.op > data.op_prepare_max),
        checksum_match,
        op_equal,
        op_less,
        view_equal,
    ];

    let mut result: Option<&'static Case> = None;
    for case in &RECOVERY_CASES {
        match case.check(parameters) {
            Ok(true) => {
                assert!(result.is_none(), "multiple cases matched");
                result = Some(case);
            }
            Ok(false) => {}
            Err(CaseCheckError::ExpectTrue | CaseCheckError::ExpectFalse) => {
                panic!("recovery_case: impossible state: case={}", case.label);
            }
        }
    }
    match result {
        Some(case) => case,
        None => unreachable!("recovery table is exhaustive"),
    }
}

#[cfg(test)]
mod journal_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A checksum-valid prepare with arbitrary fields (bypasses the staging-header
    /// assertions of `set_checksum()` by assigning the checksum directly).
    fn make_prepare(
        cluster: u128,
        op: u64,
        view: u32,
        operation: crate::Operation,
    ) -> message_header::Prepare {
        let mut header = message_header::Prepare::reserve(cluster, op % SLOT_COUNT as u64);
        header.operation = operation;
        header.view = view;
        header.op = op;
        header.checksum = header.calculate_checksum();
        header
    }

    #[test]
    fn slot_for_op_wraps_around_the_ring() {
        assert_eq!(Slot::for_op(0), Slot { index: 0 });
        assert_eq!(Slot::for_op(SLOT_COUNT as u64 - 1), Slot { index: SLOT_COUNT - 1 });
        assert_eq!(Slot::for_op(SLOT_COUNT as u64), Slot { index: 0 });
        assert_eq!(Slot::for_op(SLOT_COUNT as u64 + 7), Slot { index: 7 });
    }

    #[test]
    fn slot_range_contains_straight_and_wrapped() {
        // head < tail → inclusive between:
        let straight = SlotRange { head: Slot { index: 2 }, tail: Slot { index: 5 } };
        assert!(!straight.contains(Slot { index: 1 }));
        assert!(straight.contains(Slot { index: 2 }));
        assert!(straight.contains(Slot { index: 5 }));
        assert!(!straight.contains(Slot { index: 6 }));

        // head > tail → wraps around the end:
        let wrapped = SlotRange { head: Slot { index: 8 }, tail: Slot { index: 2 } };
        assert!(wrapped.contains(Slot { index: 8 }));
        assert!(wrapped.contains(Slot { index: 9 }));
        assert!(wrapped.contains(Slot { index: 0 }));
        assert!(wrapped.contains(Slot { index: 2 }));
        assert!(!wrapped.contains(Slot { index: 3 }));
        assert!(!wrapped.contains(Slot { index: 7 }));
    }

    #[test]
    fn ring_offsets_are_sector_and_message_aligned() {
        let slot = Slot { index: 3 };

        let headers_offset = Ring::Headers.offset(slot);
        assert_eq!(headers_offset % SECTOR_SIZE as u64, 0);
        assert!(headers_offset < HEADERS_SIZE as u64);

        let prepares_offset = Ring::Prepares.offset(slot);
        assert_eq!(prepares_offset % MESSAGE_SIZE_MAX as u64, 0);
        assert!(prepares_offset < PREPARES_SIZE as u64);

        assert_eq!(Ring::Prepares.offset(Slot { index: 0 }), 0);
    }

    #[test]
    fn header_ok_accepts_reserved_and_prepare_headers_in_the_right_slot() {
        let cluster = 0xA1_B2_C3;
        let slot = Slot { index: 5 };

        // Reserved headers live at their slot index directly:
        let reserved = message_header::Prepare::reserve(cluster, 5);
        assert_eq!(header_ok(cluster, slot, &reserved), Some(reserved));

        // Wrong slot → rejected:
        assert_eq!(header_ok(cluster, Slot { index: 6 }, &reserved), None);

        // Wrong cluster → rejected:
        assert_eq!(header_ok(0xDEAD, slot, &reserved), None);

        // Corrupt checksum → rejected before any field access:
        let mut corrupt = reserved;
        corrupt.cluster ^= 1;
        assert_eq!(header_ok(cluster, slot, &corrupt), None);

        // A real prepare maps through `op % slot_count`:
        let prepare = make_prepare(cluster, SLOT_COUNT as u64 + 5, 3, crate::Operation::ROOT);
        assert_eq!(header_ok(cluster, Slot { index: 5 }, &prepare), Some(prepare));
    }

    #[test]
    fn op_maximum_scans_only_valid_non_reserved_headers() {
        let cluster = 42_u128;
        let mut headers = vec![message_header::Prepare::reserve(cluster, 0); SLOT_COUNT];

        // Nothing set yet:
        assert_eq!(op_maximum_headers_untrusted(cluster, &headers), 0);

        // Ops land at their own slots (`op % slot_count`):
        let slot_a = (103 % SLOT_COUNT as u64) as usize;
        let slot_b = (107 % SLOT_COUNT as u64) as usize;
        headers[slot_a] = make_prepare(cluster, 103, 0, crate::Operation::ROOT);
        headers[slot_b] = make_prepare(cluster, 107, 0, crate::Operation::ROOT);
        assert_eq!(op_maximum_headers_untrusted(cluster, &headers), 107);

        // A header from another cluster doesn't count:
        let slot_c = (111 % SLOT_COUNT as u64) as usize;
        headers[slot_c] = make_prepare(999, 111, 0, crate::Operation::ROOT);
        assert_eq!(op_maximum_headers_untrusted(cluster, &headers), 107);
    }

    #[test]
    fn recovery_cases_table_is_exhaustive() {
        // Verify that every pattern matches exactly one case.
        //
        // Every possible combination of parameters must either:
        // * have a matching case
        // * have a case that fails (which would result in a panic).
        for i in 0..(1 << PATTERN_SIZE) {
            let parameters: [bool; PATTERN_SIZE] = std::array::from_fn(|j| i & (1 << j) != 0);

            let mut case_fail: bool = false;
            let mut case_match: Option<&Case> = None;
            for case in &RECOVERY_CASES {
                // Assertion patterns (a0/a1) act as wildcards for the purpose of matching.
                // Thus, it is possible for multiple cases to "match" a pattern iff they all
                // fail an assertion. (For example, simultaneous op= and op<).
                match case.check(parameters) {
                    Err(_) => {
                        assert!(case_match.is_none());
                        case_fail = true;
                    }
                    Ok(true) => {
                        assert!(!case_fail);
                        assert!(case_match.is_none());
                        case_match = Some(case);
                    }
                    Ok(false) => {}
                }
            }
            assert_eq!(case_fail, case_match.is_none());
        }
    }

    #[test]
    fn recovery_case_classifies_normal_paths() {
        let cluster = 7_u128;
        let data = RecoveryData { op_max: 209, op_prepare_max: 200, op_checkpoint: 100 };

        // @P: matching prepare + redundant header:
        let header_p = make_prepare(cluster, 20, 2, crate::Operation::ROOT);
        assert_eq!(recovery_case(Some(header_p), Some(header_p), data).label, "@P");

        // @L: both sides reserved:
        let reserved_l = message_header::Prepare::reserve(cluster, 4);
        assert_eq!(recovery_case(Some(reserved_l), Some(reserved_l), data).label, "@L");

        // @B: valid-but-reserved redundant header over a corrupt prepare → VSR repair:
        let redundant_b = message_header::Prepare::reserve(cluster, 6);
        assert_eq!(
            recovery_case(Some(redundant_b), None, data).decision(false),
            RecoveryDecision::Vsr
        );

        // @H: prepare beyond prepare_max → truncate (unreachable solo):
        let header_h = message_header::Prepare::reserve(cluster, 9);
        let prepare_h = make_prepare(cluster, 209, 1, crate::Operation::ROOT);
        let case_h = recovery_case(Some(header_h), Some(prepare_h), data);
        assert_eq!(case_h.label, "@H");
        assert_eq!(case_h.decision(false), RecoveryDecision::Cut);
        assert_eq!(case_h.decision(true), RecoveryDecision::Unr);

        // @D: invalid redundant header, reserved prepare — vsr for replicas>1, fix solo:
        let prepare_d = message_header::Prepare::reserve(cluster, 12);
        let case_d = recovery_case(None, Some(prepare_d), data);
        assert_eq!(case_d.label, "@D");
        assert_eq!(case_d.decision(false), RecoveryDecision::Vsr);
        assert_eq!(case_d.decision(true), RecoveryDecision::Fix);
    }

    #[test]
    fn case_cut_torn_truncates_regardless_of_solo() {
        assert_eq!(CASE_CUT_TORN.label, "@TruncateTorn");
        assert_eq!(CASE_CUT_TORN.decision(false), RecoveryDecision::CutTorn);
        assert_eq!(CASE_CUT_TORN.decision(true), RecoveryDecision::CutTorn);
        // The pattern is undefined upstream — every parameter combination "matches":
        assert!(matches!(CASE_CUT_TORN.check([true; PATTERN_SIZE]), Ok(true)));
        assert!(matches!(CASE_CUT_TORN.check([false; PATTERN_SIZE]), Ok(true)));
    }
}
