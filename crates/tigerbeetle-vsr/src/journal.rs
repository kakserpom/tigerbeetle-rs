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

// ---------------------------------------------------------------------------
// Journal — in-memory WAL header storage and lookups
// ---------------------------------------------------------------------------

/// Lifecycle of the journal (upstream: `Journal.Status`).
///
/// DEVIATION: upstream begins in `.init` and moves through `.recovering` to
/// `.recovered` asynchronously (`recover*`). Recovery is not ported yet, so
/// journals are constructed directly in the `.recovered` state; the variants
/// exist so the recovery port can land later without changing callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JournalStatus {
    Init,
    Recovering,
    Recovered,
}

/// A set bit in `dirty`/`faulty` marks the corresponding slot (upstream:
/// `Journal.BitSet`).
///
/// Spans enough words to cover [`SLOT_COUNT`] bits; `count` is maintained
/// incrementally, matching upstream's `DynamicBitSetUnmanaged` + `count`.
#[derive(Clone, Debug, Default)]
pub struct BitSet {
    words: Vec<u64>,
    count: usize,
}

impl BitSet {
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(64)], count: 0 }
    }

    fn position(index: usize) -> (usize, usize) {
        (index / 64, index % 64)
    }

    /// Whether the bit for a slot is set:
    #[must_use]
    pub fn bit(&self, slot: Slot) -> bool {
        let (word, bit) = Self::position(slot.index);
        self.words[word] & (1 << bit) != 0
    }

    /// Set the bit for a slot (idempotent):
    ///
    /// # Panics
    /// Panics if the number of set bits exceeds the bitset length (internal
    /// invariant; cannot happen while slots stay within the ring).
    pub fn set(&mut self, slot: Slot) {
        let (word, bit) = Self::position(slot.index);
        if self.words[word] & (1 << bit) == 0 {
            self.words[word] |= 1 << bit;
            self.count += 1;
            assert!(self.count <= self.words.len() * 64);
        }
    }

    /// Clear the bit for a slot (idempotent):
    pub fn clear(&mut self, slot: Slot) {
        let (word, bit) = Self::position(slot.index);
        if self.words[word] & (1 << bit) != 0 {
            self.words[word] &= !(1 << bit);
            self.count -= 1;
        }
    }
}

/// The journal stores the latest prepared op in the header ring, plus redundant
/// headers written after the corresponding prepares hit disk.
///
/// This is the synchronous in-memory data model: slot geometry, header storage,
/// lookups, hash-chain break detection, and truncation. The IOPS-pooled async
/// read/write + recovery paths are ported with the message bus.
///
/// Upstream: `src/vsr/journal.zig` (`JournalType`, header/slot/bitset fields).
pub struct Journal {
    pub cluster: u128,
    pub replica_index: u16,
    /// A header is located at `slot == header.op % headers.len()`.
    pub headers: Vec<message_header::Prepare>,
    /// Headers whose prepares are on disk; written after the prepare data.
    pub headers_redundant: Vec<message_header::Prepare>,
    /// Whether an entry is in memory only and needs to be written or is being
    /// written (dirty = not yet prepared on disk, or needing repair).
    pub dirty: BitSet,
    /// Whether an entry was written to disk and subsequently lost (corruption,
    /// misdirected read/write, or latent sector error).
    pub faulty: BitSet,
    /// Checksum of the prepare in the corresponding slot (used to answer
    /// `get_prepare` even when the slot is faulty).
    pub prepare_checksums: Vec<u128>,
    /// Whether the slot holds a prepare (see `prepare_checksums`).
    pub prepare_inhabited: Vec<bool>,
    /// The prepare body for the corresponding slot.
    ///
    /// DEVIATION: upstream keeps the whole prepare (header + body) in the WAL
    /// entry. sans-IO the body rides along in memory, keyed to the slot; a
    /// header-only prepare (repair, message-layer bodies deferred) records an
    /// empty body here.
    pub prepare_bodies: Vec<Vec<u8>>,
    pub status: JournalStatus,
}

/// An inclusive range of ops where the log needs repair
/// (upstream: private `HeaderRange`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderRange {
    pub op_min: u64,
    pub op_max: u64,
}

impl Journal {
    /// Construct an empty, recovered journal: every slot holds a reserved
    /// header at its slot index.
    ///
    /// # Panics
    /// Panics if `Prepare::reserve` fails for any slot (cannot happen).
    #[must_use]
    pub fn new(cluster: u128, replica_index: u16) -> Self {
        let headers: Vec<message_header::Prepare> = (0..SLOT_COUNT as u64)
            .map(|slot| message_header::Prepare::reserve(cluster, slot))
            .collect();
        let headers_redundant = headers.clone();
        Self {
            cluster,
            replica_index,
            headers,
            headers_redundant,
            dirty: BitSet::new(SLOT_COUNT),
            faulty: BitSet::new(SLOT_COUNT),
            prepare_checksums: vec![0; SLOT_COUNT],
            prepare_inhabited: vec![false; SLOT_COUNT],
            prepare_bodies: vec![Vec::new(); SLOT_COUNT],
            status: JournalStatus::Recovered,
        }
    }

    /// (upstream: `Journal.slot_for_op`).
    #[must_use]
    pub fn slot_for_op(op: u64) -> Slot {
        Slot::for_op(op)
    }

    #[must_use]
    pub fn slot_with_op(&self, op: u64) -> Option<Slot> {
        if self.header_with_op(op).is_some() { Some(Self::slot_for_op(op)) } else { None }
    }

    #[must_use]
    pub fn slot_with_op_and_checksum(&self, op: u64, checksum: u128) -> Option<Slot> {
        if self.header_with_op_and_checksum(op, checksum).is_some() {
            Some(Self::slot_for_op(op))
        } else {
            None
        }
    }

    /// # Panics
    /// Panics if the header is reserved (`operation == .reserved`).
    #[must_use]
    pub fn slot_for_header(&self, header: &message_header::Prepare) -> Slot {
        assert_ne!(header.operation, crate::Operation::RESERVED);
        Self::slot_for_op(header.op)
    }

    /// # Panics
    /// Panics if the header is reserved (`operation == .reserved`).
    #[must_use]
    pub fn slot_with_header(&self, header: &message_header::Prepare) -> Option<Slot> {
        assert_ne!(header.operation, crate::Operation::RESERVED);
        self.slot_with_op_and_checksum(header.op, header.checksum)
    }

    /// Returns any existing header at the location indicated by `header.op`.
    /// The existing header may have an older or newer op number.
    ///
    /// # Panics
    /// Panics if the header is reserved (`operation == .reserved`).
    #[must_use]
    pub fn header_for_prepare(
        &self,
        header: &message_header::Prepare,
    ) -> Option<&message_header::Prepare> {
        assert_ne!(header.operation, crate::Operation::RESERVED);
        self.header_for_op(header.op)
    }

    /// We use `op` directly to index into the headers array and locate ops
    /// without a scan. The existing header may have an older or newer op number.
    ///
    /// # Panics
    /// Panics if the existing header's op locates to a different slot (internal
    /// invariant).
    #[must_use]
    pub fn header_for_op(&self, op: u64) -> Option<&message_header::Prepare> {
        let slot = Self::slot_for_op(op);
        let existing = &self.headers[slot.index];
        // DEVIATION: upstream re-checks `existing.command == .prepare`; our typed
        // `Prepare` cannot hold another command, so this is unrepresentable.

        if existing.operation == crate::Operation::RESERVED {
            assert_eq!(existing.op, slot.index as u64);
            None
        } else {
            assert_eq!(Self::slot_for_op(existing.op).index, slot.index);
            Some(existing)
        }
    }

    /// Returns the entry at `@mod(op)` location, but only if `entry.op == op`,
    /// else `None`. Be careful of using this without considering that there may
    /// still be an existing op.
    #[must_use]
    pub fn header_with_op(&self, op: u64) -> Option<&message_header::Prepare> {
        // The existing header may have an older or newer op number:
        match self.header_for_op(op) {
            Some(existing) if existing.op == op => Some(existing),
            _ => None,
        }
    }

    /// As per `header_with_op()`, but only if there is a checksum match.
    #[must_use]
    pub fn header_with_op_and_checksum(
        &self,
        op: u64,
        checksum: u128,
    ) -> Option<&message_header::Prepare> {
        match self.header_with_op(op) {
            Some(existing) if existing.checksum == checksum => Some(existing),
            _ => None,
        }
    }

    /// The prepare body recorded for `op` (empty for header-only prepares).
    #[must_use]
    pub fn body_with_op(&self, op: u64) -> Option<&[u8]> {
        self.header_with_op(op).map(|_| &self.prepare_bodies[Self::slot_for_op(op).index][..])
    }

    /// Record the prepare body for `op`.
    ///
    /// # Panics
    /// Panics if `op` is not currently journaled (upstream records the body as
    /// part of the prepare message, so a body always has a header).
    pub fn set_prepare_body(&mut self, op: u64, body: &[u8]) {
        assert!(self.header_with_op(op).is_some(), "prepare body requires a journaled header");
        self.prepare_bodies[Self::slot_for_op(op).index] = body.to_vec();
    }

    #[must_use]
    pub fn previous_entry(
        &self,
        header: &message_header::Prepare,
    ) -> Option<&message_header::Prepare> {
        if header.op == 0 { None } else { self.header_for_op(header.op - 1) }
    }

    #[must_use]
    pub fn next_entry(&self, header: &message_header::Prepare) -> Option<&message_header::Prepare> {
        self.header_for_op(header.op + 1)
    }

    /// Returns the highest op number prepared, in any slot without reference to
    /// the checkpoint.
    ///
    /// # Panics
    /// Panics unless the journal status is `.recovered`.
    #[must_use]
    pub fn op_maximum(&self) -> u64 {
        assert_eq!(self.status, JournalStatus::Recovered);
        let mut op: u64 = 0;
        for header in &self.headers {
            if header.operation != crate::Operation::RESERVED && header.op > op {
                op = header.op;
            }
        }
        op
    }

    /// # Panics
    /// Panics unless the journal status is `.recovered` and the header is a
    /// non-reserved prepare.
    #[must_use]
    pub fn has_header(&self, header: &message_header::Prepare) -> bool {
        assert_eq!(self.status, JournalStatus::Recovered);
        assert_ne!(header.operation, crate::Operation::RESERVED);
        self.header_with_op_and_checksum(header.op, header.checksum).is_some()
    }

    /// # Panics
    /// Panics if the header is not present, is dirty, and the slot is not
    /// marked as inhabited (upstream asserts the same).
    #[must_use]
    pub fn has_prepare(&self, header: &message_header::Prepare) -> bool {
        match self.slot_with_op_and_checksum(header.op, header.checksum) {
            Some(slot) if !self.dirty.bit(slot) => {
                assert!(self.prepare_inhabited[slot.index]);
                assert_eq!(self.prepare_checksums[slot.index], header.checksum);
                true
            }
            _ => false,
        }
    }

    /// # Panics
    /// Panics if the header is present but its slot-with-header lookup fails
    /// (cannot happen).
    #[must_use]
    pub fn has_dirty(&self, header: &message_header::Prepare) -> bool {
        if !self.has_header(header) {
            return false;
        }
        match self.slot_with_header(header) {
            Some(slot) => self.dirty.bit(slot),
            None => unreachable!("has_header implies the slot is present"),
        }
    }

    /// Copies latest headers between `op_min` and `op_max` (both inclusive) as
    /// fit in `dest`. Reverses the order when copying so that latest headers
    /// are copied first, which protects against the callsite slicing the buffer
    /// the wrong way and incorrectly, and which is required by message handlers
    /// that use the hash chain for repairs. Skips `.reserved` headers (gaps
    /// between headers). Returns the number of headers actually copied.
    ///
    /// # Panics
    /// Panics unless the journal status is `.recovered`, `op_min <= op_max`, and
    /// `dest` is non-empty.
    ///
    /// DEVIATION: upstream `@memset`s `dest` to `undefined`; Rust has no
    /// undefined values, so the caller must only read the first `copied`
    /// entries of `dest`.
    #[must_use]
    pub fn copy_latest_headers_between(
        &self,
        op_min: u64,
        op_max: u64,
        dest: &mut [message_header::Prepare],
    ) -> usize {
        assert_eq!(self.status, JournalStatus::Recovered);
        assert!(op_min <= op_max);
        assert!(!dest.is_empty());

        let mut copied: usize = 0;

        // Start at op_max + 1 and do the decrement upfront to avoid overflow
        // when op_min == 0:
        let mut op = op_max + 1;
        while op > op_min {
            op -= 1;

            if let Some(header) = self.header_with_op(op) {
                dest[copied] = *header;
                copied += 1;
                if copied == dest.len() {
                    break;
                }
            }
        }

        copied
    }

    /// Finds the latest break in headers between `op_min` and `op_max` (both
    /// inclusive). A break is a missing header or a header not connected to the
    /// next header by hash chain. On finding the highest break, extends the
    /// range downwards to cover as much as possible.
    ///
    /// We expect that `op_max` (`replica.op`) must exist.
    /// `op_min` may exist or not.
    ///
    /// A range will never include `op_max` because this must be up to date as
    /// the latest op. A range may include `op_min`.
    ///
    /// For example: If ops 3, 9 and 10 are missing, returns `{9, 10}`.
    ///
    /// Another example: If op 17 is disconnected from op 18, 16 is connected to
    /// 17, and 12-15 are missing, returns `{12, 17}`.
    ///
    /// # Panics
    /// Panics unless the journal status is `.recovered`, `header_with_op(op_max)`
    /// exists, and the examined span fits within one ring.
    #[must_use]
    pub fn find_latest_headers_break_between(
        &self,
        op_min: u64,
        op_max: u64,
    ) -> Option<HeaderRange> {
        assert_eq!(self.status, JournalStatus::Recovered);
        assert!(self.header_with_op(op_max).is_some());
        assert!(op_max >= op_min);
        assert!(op_max - op_min < SLOT_COUNT as u64);
        let mut range: Option<HeaderRange> = None;

        // We set B, the op after op_max, to null because we only examine
        // breaks < op_max:
        let mut b: Option<message_header::Prepare> = None;

        let mut op = op_max + 1;
        while op > op_min {
            op -= 1;

            let a = self.header_with_op(op).copied();
            if let Some(a) = &a {
                if let Some(b) = &b {
                    // If A was reordered then A may have a newer op than B (but an
                    // older view). header_with_op() guarantees a.op + 1 == b.op:
                    assert_eq!(a.op + 1, b.op);
                    // We do not assert a.view <= b.view here unless the chain is
                    // intact, because repair_header() may put a newer view to the
                    // left of an older view.

                    if let Some(r) = &mut range {
                        assert_eq!(b.op, r.op_min);
                        if a.op == op_min {
                            // A is committed, because we pass `commit_min` as
                            // `op_min`: A cannot be a break if committed.
                            break;
                        } else if a.checksum == b.parent {
                            // A is connected to B, but B is disconnected, add A:
                            assert!(a.view <= b.view);
                            r.op_min = a.op;
                        } else if a.view < b.view {
                            // A is not connected to B, and A is older: add A:
                            r.op_min = a.op;
                        } else if a.view > b.view {
                            // A is not connected to B, but A is newer: close:
                            break;
                        } else {
                            // Op numbers in the same view must be connected.
                            unreachable!("op numbers in the same view must be connected");
                        }
                    } else if a.checksum == b.parent {
                        // A is connected to B, and B is connected or B is op_max.
                        assert!(a.view <= b.view);
                    } else if a.view != b.view {
                        // A is not connected to B, open range:
                        assert!(b.op <= op_max);
                        range = Some(HeaderRange { op_min: a.op, op_max: a.op });
                    } else {
                        unreachable!("op numbers in the same view must be connected");
                    }
                } else {
                    // A exists and B does not exist (or B has a older/newer op):
                    if let Some(r) = &range {
                        // A may be older/newer, close range:
                        assert_eq!(r.op_min, op + 1);
                        break;
                    }
                    // We expect a range if B does not exist, unless:
                    assert_eq!(a.op, op_max);
                }
            } else {
                assert!(op < op_max);

                // A does not exist, or A has an older (or newer if reordered) op:
                if let Some(r) = &mut range {
                    // Add A to range:
                    assert_eq!(r.op_min, op + 1);
                    r.op_min = op;
                } else {
                    // Open range:
                    assert!(b.is_some());
                    range = Some(HeaderRange { op_min: op, op_max: op });
                }
            }

            b = a;
        }

        if let Some(r) = &range {
            assert!(r.op_min >= op_min);
            // We can never repair op_max (replica.op) since that is the latest op:
            assert!(r.op_max < op_max);
        }

        range
    }

    /// Removes entries from `op_min` (inclusive) onwards. Used after a view
    /// change to remove uncommitted entries discarded by the new primary.
    ///
    /// # Panics
    /// Panics unless the journal status is `.recovered` and `op_min > 0`.
    pub fn remove_entries_from(&mut self, op_min: u64) {
        assert_eq!(self.status, JournalStatus::Recovered);
        assert!(op_min > 0);

        let mut slots = Vec::new();
        for (index, header) in self.headers.iter().enumerate() {
            // We must remove the header regardless of whether it is a prepare or
            // reserved, since a reserved header may have been marked faulty for
            // case @K, and since the caller expects the WAL to be truncated,
            // with clean slots.
            if header.op >= op_min {
                let slot = Self::slot_for_op(header.op);
                assert_eq!(slot.index, index);
                slots.push(slot);
            }
        }
        for slot in slots {
            self.remove_entry(slot);
        }
    }

    /// Replace a slot's headers with a reserved header and clear its dirty and
    /// faulty bits. The prepare is untouched on disk — it may still be useful
    /// later (see upstream's comment for why `prepare_inhabited` is not cleared).
    pub fn remove_entry(&mut self, slot: Slot) {
        let reserved = message_header::Prepare::reserve(self.cluster, slot.index as u64);
        self.headers[slot.index] = reserved;
        self.headers_redundant[slot.index] = reserved;
        self.dirty.clear(slot);
        self.faulty.clear(slot);
        self.prepare_bodies[slot.index] = Vec::new();
    }

    /// Mark a header as dirty (needing to be written), overwriting any previous
    /// entry in the slot.
    ///
    /// # Panics
    /// Panics unless the journal status is `.recovered`, the header is
    /// non-reserved, and the slot does not hold a newer op.
    pub fn set_header_as_dirty(&mut self, header: &message_header::Prepare) {
        assert_eq!(self.status, JournalStatus::Recovered);
        assert_ne!(header.operation, crate::Operation::RESERVED);

        let slot = self.slot_for_header(header);

        if self.has_header(header) {
            assert!(self.dirty.bit(slot));
            // Do not clear any faulty bit for the same entry.
        } else {
            // Overwriting a new op with an old op would be a correctness bug; it
            // could cause a message to be uncommitted.
            assert!(self.headers[slot.index].op <= header.op);

            if self.headers[slot.index].operation == crate::Operation::RESERVED {
                // The WAL might have written/prepared this exact header before
                // crashing — leave the entry marked faulty because we cannot
                // safely nack it.
            } else {
                // The WAL definitely did not hold this exact header, so it is safe
                // to reset the faulty bit + nack this header.
                self.faulty.clear(slot);
                self.headers_redundant[slot.index] =
                    message_header::Prepare::reserve(self.cluster, slot.index as u64);
            }

            self.headers[slot.index] = *header;
            self.dirty.set(slot);
        }
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

    // ── Journal header storage & lookups ────────────────────────────────

    /// Build a journal holding a hash-chained run of prepares `0..=max`.
    /// `missing` ops are left reserved (a gap), simulating a slot that never
    /// received its entry.
    fn journal_with_chain(cluster: u128, max: u64, missing: &[u64]) -> Journal {
        let mut journal = Journal::new(cluster, 0);
        let mut parent = 0u128;
        for op in 0..=max {
            if missing.contains(&op) {
                continue;
            }
            let mut header = make_prepare(cluster, op, 0, crate::Operation::ROOT);
            header.parent = parent;
            header.checksum = header.calculate_checksum();
            journal.set_header_as_dirty(&header);
            parent = header.checksum;
        }
        journal
    }

    #[test]
    fn journal_headers_start_reserved() {
        let journal = Journal::new(42, 1);
        assert_eq!(journal.headers.len(), SLOT_COUNT);
        assert_eq!(journal.headers_redundant.len(), SLOT_COUNT);
        for (i, header) in journal.headers.iter().enumerate() {
            assert_eq!(header.operation, crate::Operation::RESERVED);
            assert_eq!(header.op, i as u64);
        }
        assert_eq!(journal.header_with_op(0), None);
        assert_eq!(journal.op_maximum(), 0);
        assert!(!journal.dirty.bit(Slot { index: 0 }));
        assert!(!journal.faulty.bit(Slot { index: 0 }));
    }

    #[test]
    fn journal_insert_and_lookup() {
        let cluster = 9_u128;
        let mut journal = Journal::new(cluster, 1);
        let mut header = make_prepare(cluster, 4, 0, crate::Operation::ROOT);
        header.parent = 0xDEAD_BEEF;
        header.checksum = header.calculate_checksum();
        journal.set_header_as_dirty(&header);

        // header_with_op by op + checksum:
        assert_eq!(journal.header_with_op(4), Some(&header));
        assert_eq!(journal.header_with_op_and_checksum(4, header.checksum), Some(&header));
        assert_eq!(journal.header_with_op_and_checksum(4, header.checksum ^ 1), None);
        assert_eq!(journal.header_with_op(5), None);

        // slot lookups:
        assert_eq!(journal.slot_with_op(4), Some(Slot { index: 4 }));
        assert_eq!(journal.slot_with_op_and_checksum(4, header.checksum), Some(Slot { index: 4 }));
        assert_eq!(journal.slot_with_header(&header), Some(Slot { index: 4 }));

        // previous/next entries across the gap:
        let header = journal.header_with_op(4).unwrap();
        assert_eq!(journal.previous_entry(header), None); // no op 3
        assert_eq!(journal.next_entry(header), None); // no op 5

        // dirty bit is set by set_header_as_dirty:
        assert!(journal.has_header(&header.clone()));
        assert!(journal.has_dirty(&header.clone()));
    }

    #[test]
    fn journal_lookup_after_ring_wrap() {
        let cluster = 9_u128;
        let mut journal = Journal::new(cluster, 1);
        // Op 8 lives at slot 8; a later op 40 overwrites the same slot, so
        // header_for_op(8) sees op 40's (newer) header but header_with_op(8) is None.
        for (op, view) in [(8, 0u32), (40, 1u32)] {
            let mut header = make_prepare(cluster, op, view, crate::Operation::ROOT);
            header.checksum = header.calculate_checksum();
            journal.set_header_as_dirty(&header);
        }

        assert_eq!(Journal::slot_for_op(40), Slot { index: 8 });
        let existing = journal.header_for_op(8).unwrap();
        assert_eq!(existing.op, 40);
        assert_eq!(journal.header_with_op(8), None);
        assert_eq!(journal.header_with_op(40), Some(existing));
    }

    #[test]
    fn journal_op_maximum_finds_highest() {
        let cluster = 5_u128;
        let journal = journal_with_chain(cluster, 6, &[]);
        assert_eq!(journal.op_maximum(), 6);

        // After truncation, reserved headers are ignored:
        let mut journal = journal;
        journal.remove_entries_from(4);
        assert_eq!(journal.op_maximum(), 3);
    }

    #[test]
    fn journal_has_prepare_requires_clean_inhabited_slot() {
        let cluster = 5_u128;
        let mut journal = journal_with_chain(cluster, 6, &[]);
        let header = *journal.header_with_op(4).unwrap();

        // has_header is true, but the slot is dirty + not marked inhabited:
        assert!(!journal.has_prepare(&header));

        // Mark the prepare as on-disk:
        journal.dirty.clear(Journal::slot_for_op(4));
        journal.prepare_inhabited[4] = true;
        journal.prepare_checksums[4] = header.checksum;
        assert!(journal.has_prepare(&header));

        // Wrong checksum → not present:
        let mut other = header;
        other.checksum ^= 1;
        assert!(!journal.has_prepare(&other));
    }

    #[test]
    fn journal_copy_latest_headers_reverses_and_stops() {
        let cluster = 5_u128;
        let journal = journal_with_chain(cluster, 6, &[]);

        // Copies newest-first, up to dest capacity:
        let mut dest = vec![message_header::Prepare::reserve(cluster, 0); 4];
        let copied = journal.copy_latest_headers_between(0, 6, &mut dest);
        assert_eq!(copied, 4);
        assert_eq!(dest[0].op, 6);
        assert_eq!(dest[1].op, 5);
        assert_eq!(dest[2].op, 4);
        assert_eq!(dest[3].op, 3);

        // Skips reserved gaps (missing ops) but copies around them:
        let journal = journal_with_chain(cluster, 6, &[3, 4]);
        let mut dest = vec![message_header::Prepare::reserve(cluster, 0); 6];
        let copied = journal.copy_latest_headers_between(0, 6, &mut dest);
        assert_eq!(copied, 5);
        assert_eq!(dest[0].op, 6);
        assert_eq!(dest[1].op, 5);
        assert_eq!(dest[2].op, 2);
        assert_eq!(dest[3].op, 1);
        assert_eq!(dest[4].op, 0);
    }

    #[test]
    fn journal_find_latest_headers_break_none_when_contiguous() {
        let cluster = 5_u128;
        let journal = journal_with_chain(cluster, 6, &[]);
        assert_eq!(journal.find_latest_headers_break_between(0, 6), None);
    }

    #[test]
    fn journal_find_latest_headers_break_missing_middle() {
        // Ops 0..6 all present except op 3; the highest break is just {3, 3}.
        let cluster = 5_u128;
        let journal = journal_with_chain(cluster, 6, &[3]);
        assert_eq!(
            journal.find_latest_headers_break_between(0, 6),
            Some(HeaderRange { op_min: 3, op_max: 3 })
        );
    }

    #[test]
    fn journal_find_latest_headers_break_runs_of_missing() {
        // Ops 0..7 with 3, 4, 5 missing: the break from the head is {3, 5}.
        let cluster = 5_u128;
        let journal = journal_with_chain(cluster, 7, &[3, 4, 5]);
        assert_eq!(
            journal.find_latest_headers_break_between(0, 7),
            Some(HeaderRange { op_min: 3, op_max: 5 })
        );
    }

    #[test]
    fn journal_settings_remove_and_clean() {
        let cluster = 5_u128;
        let mut journal = journal_with_chain(cluster, 6, &[]);

        // set_header_as_dirty over an existing (same op) header keeps it dirty:
        let mut header = make_prepare(cluster, 6, 1, crate::Operation::ROOT);
        header.checksum = header.calculate_checksum();
        journal.set_header_as_dirty(&header);
        assert!(journal.has_header(&header));
        assert!(journal.dirty.bit(Slot { index: 6 }));

        // remove_entries_from truncates from op_min onwards and clears bits:
        let mut journal = Journal::new(cluster, 0);
        for op in 4..=6 {
            let mut h = make_prepare(cluster, op, 0, crate::Operation::ROOT);
            h.checksum = h.calculate_checksum();
            journal.set_header_as_dirty(&h);
        }
        journal.dirty.set(Slot { index: 4 });
        journal.faulty.set(Slot { index: 4 });
        journal.remove_entries_from(4);
        assert_eq!(journal.header_with_op(4), None);
        assert_eq!(journal.header_with_op(6), None);
        assert_eq!(journal.op_maximum(), 0);
        for slot in 4..=6 {
            let slot = Slot { index: slot };
            assert!(!journal.dirty.bit(slot));
            assert!(!journal.faulty.bit(slot));
            assert_eq!(journal.headers[slot.index].operation, crate::Operation::RESERVED);
        }
    }
}
