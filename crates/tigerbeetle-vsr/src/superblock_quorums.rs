//! Superblock quorums: grouping the on-disk superblock copies into quorums, selecting the
//! working superblock at startup, and planning repairs.
//!
//! Upstream: `src/vsr/superblock_quorums.zig` (`QuorumsType(.{ .superblock_copies = … })`).
//!
//! DEVIATION: upstream is a comptime-generic function of `superblock_copies`; this port is
//! generic over `COPIES` with [`SuperBlockQuorums`] as the instantiated alias.

use core::fmt;

use crate::superblock::{SuperBlockHeader, VSRState};
use tigerbeetle_core::constants::SUPERBLOCK_COPIES;

/// Port of `QuorumCount = stdx.BitSetType(superblock_copies)`: a fixed-capacity bitset over
/// copy indices (≤ 8 in every supported configuration).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuorumCount {
    bits: [bool; 8],
}

impl QuorumCount {
    #[must_use]
    pub fn count(&self) -> usize {
        self.bits.iter().filter(|&&bit| bit).count()
    }

    #[must_use]
    pub fn full(&self) -> bool {
        self.count() == self.bits.len()
    }

    /// # Panics
    /// Panics if `index` exceeds the bitset capacity (upstream asserts).
    #[must_use]
    pub fn is_set(&self, index: usize) -> bool {
        assert!(index < self.bits.len());
        self.bits[index]
    }

    /// # Panics
    /// Panics if `index` exceeds the bitset capacity (upstream asserts).
    pub fn set(&mut self, index: usize) {
        assert!(index < self.bits.len());
        self.bits[index] = true;
    }
}

/// Port of `superblock_quorums.Error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuorumError {
    Fork,
    NotFound,
    QuorumLost,
    ParentNotConnected,
    ParentSkipped,
    VSRStateNotMonotonic,
}

impl fmt::Display for QuorumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Fork => "fork",
            Self::NotFound => "not found",
            Self::QuorumLost => "quorum lost",
            Self::ParentNotConnected => "parent not connected",
            Self::ParentSkipped => "parent skipped",
            Self::VSRStateNotMonotonic => "vsr state not monotonic",
        };
        write!(f, "superblock quorum error: {name}")
    }
}

impl std::error::Error for QuorumError {}

/// Port of `Threshold`.
///
/// We use flexible quorums for even quorums with write quorum > read quorum, for example:
/// * When writing, we must verify that at least 3/4 copies were written.
/// * At startup, we must verify that at least 2/4 copies were read.
///
/// This ensures that our read and write quorums will intersect.
/// Using flexible quorums in this way increases resiliency of the superblock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threshold {
    Verify,
    // Working these threshold out by formula is easy to get wrong, so enumerate them:
    // The rule is that the write quorum plus the read quorum must be exactly copies + 1.
    /// The open quorum must allow for at least two copy faults, because we update
    /// copies in place, temporarily impairing one copy.
    Open,
}

impl Threshold {
    /// # Panics
    /// Panics for unsupported copy counts (upstream: `unreachable`; only {4, 6, 8} are valid).
    #[must_use]
    pub fn count<const COPIES: usize>(self) -> u8 {
        match self {
            Self::Verify => match COPIES {
                4 => 3,
                6 => 4,
                8 => 5,
                _ => unreachable!("superblock_copies must be either 4, 6, or 8"),
            },
            Self::Open => match COPIES {
                4 => 2,
                6 => 3,
                8 => 4,
                _ => unreachable!("superblock_copies must be either 4, 6, or 8"),
            },
        }
    }
}

/// Port of `QuorumsType(options).Quorum`.
#[derive(Clone, Debug)]
pub struct Quorum<'a> {
    header: &'a SuperBlockHeader,
    valid: bool,
    /// Track which copies are a member of the quorum.
    /// Used to ignore duplicate copies of a header when determining a quorum.
    copies: QuorumCount,
    /// An integer value indicates the copy index found in the corresponding slot.
    /// A `None` value indicates that the copy is invalid or not a member of the working
    /// quorum. All copies belong to the same (valid, working) quorum.
    slots: [Option<u8>; SUPERBLOCK_COPIES],
}

impl<'a> Quorum<'a> {
    /// Port of `Quorum.repairs`.
    ///
    /// # Panics
    /// Panics if the quorum is not valid (upstream asserts).
    #[must_use]
    pub fn repairs(&self) -> RepairIterator {
        assert!(self.valid);
        RepairIterator { slots: self.slots }
    }

    /// The header shared by all members of this quorum.
    #[must_use]
    pub fn header(&self) -> &'a SuperBlockHeader {
        self.header
    }

    /// Whether the quorum reached the threshold (upstream: `quorum.valid`).
    #[must_use]
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// How many copies joined the quorum (upstream: `quorum.copies.count()`).
    #[must_use]
    pub fn copies_count(&self) -> usize {
        self.copies.count()
    }
}

/// Port of `QuorumsType(options)`, instantiated for [`SUPERBLOCK_COPIES`].
#[derive(Debug, Default)]
pub struct SuperBlockQuorums<'a> {
    array: [Option<Quorum<'a>>; SUPERBLOCK_COPIES],
    count: usize,
}

impl<'a> SuperBlockQuorums<'a> {
    fn quorum(&self, index: usize) -> &Quorum<'a> {
        assert!(index < self.count);
        self.array[index].as_ref().unwrap_or_else(|| unreachable!("slots below count are filled"))
    }

    /// Returns the working superblock according to the quorum with the highest sequence number.
    ///
    /// * When a member of the parent quorum is still present, verify that the highest quorum is
    ///   connected.
    /// * When there are 2 quorums: 1/4 new and 3/4 old, favor the 3/4 old since it is safer to
    ///   repair.
    ///   TODO Re-examine this now that there are no superblock trailers to worry about.
    ///
    /// # Errors
    /// Returns [`QuorumError::NotFound`] if no copy has a valid checksum,
    /// [`QuorumError::QuorumLost`] if even the best quorum misses the threshold, or one of
    ///   `Fork`, `ParentSkipped`, `ParentNotConnected`, `VSRStateNotMonotonic`
    ///   when the surviving sequences cannot be safely chained together.
    ///
    /// # Panics
    /// Panics unless `copies.len() == SUPERBLOCK_COPIES` and the threshold is within 2..=5
    /// (upstream asserts).
    pub fn working(
        &mut self,
        copies: &'a [SuperBlockHeader],
        threshold: Threshold,
    ) -> Result<Quorum<'a>, QuorumError> {
        assert_eq!(copies.len(), SUPERBLOCK_COPIES);
        let threshold_count = threshold.count::<SUPERBLOCK_COPIES>();
        assert!((2..=5).contains(&threshold_count));

        self.array = Default::default();
        self.count = 0;

        for (index, copy) in copies.iter().enumerate() {
            self.count_copy(copy, index, threshold_count);
        }

        // Sort by repair priority (see `sort_priority_descending`):
        let mut order: Vec<usize> = (0..self.count).collect();
        order.sort_by(|&i, &j| Self::sort_priority_descending(self.quorum(i), self.quorum(j)));

        // No working copies of any sequence number exist in the superblock storage zone at all.
        if let Some(&best) = order.first() {
            let b = self.quorum(best);

            // Verify that the remaining quorums are correctly sorted:
            for &index in &order[1..] {
                let a = self.quorum(index);
                assert_eq!(Self::sort_priority_descending(b, a), core::cmp::Ordering::Less);
                assert!(a.header.valid_checksum());
            }

            // Even the best copy with the most quorum still has inadequate quorum.
            if !b.valid {
                return Err(QuorumError::QuorumLost);
            }

            // If a parent quorum is present (either complete or incomplete) it must be connected
            // to the new working quorum. The parent quorum can exist due to:
            // - a crash during checkpoint()/view_change() before writing all copies
            // - a lost or misdirected write
            // - a latent sector error that prevented a write
            for &index in &order[1..] {
                let a = self.quorum(index);
                if a.header.cluster != b.header.cluster {
                    continue;
                }

                if a.header.vsr_state.replica_id != b.header.vsr_state.replica_id {
                    continue;
                }

                if a.header.sequence == b.header.sequence {
                    // Two quorums, same cluster+replica+sequence, but different checksums.
                    // This shouldn't ever happen — but if it does, we can't safely repair.
                    assert_ne!(a.header.checksum, b.header.checksum);
                    return Err(QuorumError::Fork);
                }

                if a.header.sequence > b.header.sequence + 1 {
                    // We read sequences such as (2,2,2,4) — 2 isn't safe to use, but there isn't
                    // a valid quorum for 4 either.
                    return Err(QuorumError::ParentSkipped);
                }

                if a.header.sequence + 1 == b.header.sequence {
                    assert_ne!(a.header.checksum, b.header.checksum);
                    assert_eq!(a.header.cluster, b.header.cluster);
                    assert_eq!(a.header.vsr_state.replica_id, b.header.vsr_state.replica_id);

                    if a.header.checksum != b.header.parent {
                        return Err(QuorumError::ParentNotConnected);
                    }
                    if !VSRState::monotonic(&a.header.vsr_state, &b.header.vsr_state) {
                        return Err(QuorumError::VSRStateNotMonotonic);
                    }

                    assert!(b.header.valid_checksum());
                    return Ok(b.clone());
                }
            }

            assert!(b.header.valid_checksum());
            return Ok(b.clone());
        }
        Err(QuorumError::NotFound)
    }

    fn count_copy(&mut self, copy: &'a SuperBlockHeader, slot: usize, threshold_count: u8) {
        assert!(slot < SUPERBLOCK_COPIES);
        assert!((2..=5).contains(&threshold_count));

        if !copy.valid_checksum() {
            return;
        }

        // Either the entire copy was misdirected, or just the copy field is corrupted.
        // We definitely still want to count the copy.
        // We must just be careful to count it idempotently.

        let quorum_index = self.find_or_insert_quorum_for_copy(copy);
        let quorum =
            self.array[quorum_index].as_mut().unwrap_or_else(|| unreachable!("just inserted"));
        assert_eq!(quorum.header.checksum, copy.checksum);
        assert!(quorum.header.equal(copy));

        if usize::from(copy.copy) >= SUPERBLOCK_COPIES {
            // This header is a valid member of the quorum, but with an unexpected copy number.
            // The "SuperBlockHeader.copy" field is not protected by the checksum, so if that
            // byte (and only that byte) is corrupted, the superblock is still valid — but we
            // don't know for certain which copy this was supposed to be.
            // We make the assumption that this was not a double-fault (corrupt + misdirect) —
            // that is, the copy is in the correct slot, and its copy index is simply corrupt.
            quorum.slots[slot] = Some(u8::try_from(slot).unwrap_or(u8::MAX));
            quorum.copies.set(slot);
        } else if quorum.copies.is_set(usize::from(copy.copy)) {
            // Ignore the duplicate copy.
        } else {
            quorum.slots[slot] = match u8::try_from(copy.copy) {
                Ok(copy) => Some(copy),
                Err(_) => unreachable!("copy.copy < SUPERBLOCK_COPIES"),
            };
            quorum.copies.set(usize::from(copy.copy));
        }
        assert!(quorum.copies.count() >= 1);

        quorum.valid = quorum.copies.count() >= usize::from(threshold_count);
    }

    fn find_or_insert_quorum_for_copy(&mut self, copy: &'a SuperBlockHeader) -> usize {
        assert!(copy.valid_checksum());

        for index in 0..self.count {
            let quorum = self.array[index]
                .as_ref()
                .unwrap_or_else(|| unreachable!("slots below count are filled"));
            if copy.checksum == quorum.header.checksum {
                return index;
            }
        }
        self.array[self.count] = Some(Quorum {
            header: copy,
            valid: false,
            copies: QuorumCount::default(),
            slots: [None; SUPERBLOCK_COPIES],
        });
        self.count += 1;

        self.count - 1
    }

    fn sort_priority_descending(a: &Quorum<'_>, b: &Quorum<'_>) -> core::cmp::Ordering {
        assert_ne!(a.header.checksum, b.header.checksum);

        if a.valid && !b.valid {
            return core::cmp::Ordering::Less;
        }
        if b.valid && !a.valid {
            return core::cmp::Ordering::Greater;
        }

        if a.header.sequence != b.header.sequence {
            return b.header.sequence.cmp(&a.header.sequence);
        }

        if a.copies.count() != b.copies.count() {
            return b.copies.count().cmp(&a.copies.count());
        }

        // The sort order must be stable and deterministic:
        b.header.checksum.cmp(&a.header.checksum)
    }
}

/// Port of `RepairIterator`: repair a quorum's copies in the safest known order.
/// Repair is complete when every copy is on-disk (not necessarily in its home slot).
///
/// We must be careful when repairing superblock headers to avoid endangering our quorum if
/// an additional fault occurs. We primarily guard against torn header writes — preventing a
/// misdirected write from derailing repair is far more expensive and complex — but they are
/// likewise far less likely to occur.
///
/// For example, consider this case:
///
/// ```text
/// 0. Sequence is initially A.
/// 1. Checkpoint sequence B.
/// 2.   Write B₀ — ok.
/// 3.   Write B₁ — misdirected to B₂'s slot.
/// 4. Crash.
/// 5. Recover with quorum B[B₀,A₁,B₁,A₃].
///    If we repair the superblock quorum while only considering the valid copies (and not
///    slots) the following scenario could occur:
///      6. We already have a valid B₀ and B₁, so begin writing B₂.
///      7. Crash, tearing the B₂ write.
///      8. Recover with quorum A[B₀,A₁,_,A₂].
///    The working quorum backtracked from B to A!
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RepairIterator {
    /// An integer value indicates the copy index found in the corresponding slot.
    /// A `None` value indicates that the copy is invalid or not a member of the working
    /// quorum. All copies belong to the same (valid, working) quorum.
    slots: [Option<u8>; SUPERBLOCK_COPIES],
}

impl RepairIterator {
    /// Returns the slot/copy to repair next.
    /// We never (deliberately) write a copy to a slot other than its own. This is simpler
    /// to implement, and also reduces risk when one of open()'s reads was misdirected.
    ///
    /// # Panics
    /// Panics if a stored copy index exceeds the copy count (upstream asserts; corrupt
    /// indices are normalized by `count_copy`).
    pub fn next_slot(&mut self) -> Option<u8> {
        // Corrupt copy indices have already been normalized.
        for slot in &self.slots {
            assert!(
                slot.is_none_or(|copy| usize::from(copy) < SUPERBLOCK_COPIES),
                "corrupt copy index"
            );
        }

        // Set bits indicate that the corresponding copy was found at least once.
        let mut copies_any = QuorumCount::default();
        // Set bits indicate that the corresponding copy was found more than once.
        let mut copies_duplicate = QuorumCount::default();

        for slot in self.slots.iter().flatten() {
            let copy = usize::from(*slot);
            if copies_any.is_set(copy) {
                copies_duplicate.set(copy);
            }
            copies_any.set(copy);
        }

        // In descending order, our priorities for repair are:
        // 1. The slot holds no header, and the copy was not found anywhere.
        // 2. The slot holds no header, but its copy was found elsewhere.
        // 3. The slot holds a misdirected header, but that copy is in another slot as well.
        let mut a: Option<u8> = None;
        let mut b: Option<u8> = None;
        let mut c: Option<u8> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            match *slot {
                None => {
                    if !copies_any.is_set(i) {
                        a = Some(u8::try_from(i).unwrap_or(u8::MAX));
                    }
                    if copies_any.is_set(i) {
                        b = Some(u8::try_from(i).unwrap_or(u8::MAX));
                    }
                }
                Some(slot_copy) => {
                    if usize::from(slot_copy) != i
                        && copies_duplicate.is_set(usize::from(slot_copy))
                    {
                        c = Some(u8::try_from(i).unwrap_or(u8::MAX));
                    }
                }
            }
        }

        let repair = a.or(b).or(c)?;
        self.slots[usize::from(repair)] = Some(repair);
        Some(repair)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        QuorumCount, QuorumError, RepairIterator, SUPERBLOCK_COPIES, SuperBlockQuorums, Threshold,
    };
    use crate::message_header;
    use crate::multiversion::Release;
    use crate::superblock::{SUPERBLOCK_VERSION, SuperBlockHeader, VSRState, VSRStateRootOptions};
    use tigerbeetle_core::constants::MEMBERS_MAX;

    fn test_header(sequence: u64) -> SuperBlockHeader {
        let mut members = [0u128; MEMBERS_MAX];
        members[0] = 11;
        members[1] = 22;
        members[2] = 33;
        let vsr_state = VSRState::root(VSRStateRootOptions {
            cluster: 7,
            replica_id: 11,
            members,
            replica_count: 3,
            release: Release::MINIMUM,
            view: 0,
        });
        let mut header = SuperBlockHeader {
            checksum: 0,
            checksum_padding: 0,
            copy: 0,
            version: SUPERBLOCK_VERSION,
            release_format: Release::MINIMUM,
            sequence,
            cluster: 7,
            parent: 0,
            parent_padding: 0,
            vsr_state,
            flags: 0,
            view_headers_count: 0,
            view_headers_all: [[0; message_header::SIZE]; VIEW_HEADERS_MAX as usize],
        };
        header.set_checksum();
        header
    }

    use tigerbeetle_core::constants::VIEW_HEADERS_MAX;

    fn four_headers(sequence: u64) -> [SuperBlockHeader; SUPERBLOCK_COPIES] {
        core::array::from_fn(|slot| {
            let mut header = test_header(sequence);
            header.copy = u16::try_from(slot).unwrap_or(u16::MAX);
            header
        })
    }

    fn corrupt(header: &mut SuperBlockHeader) {
        header.version += 1;
        assert!(!header.valid_checksum());
    }

    #[test]
    fn threshold_counts_enumerated() {
        assert_eq!(Threshold::Verify.count::<4>(), 3);
        assert_eq!(Threshold::Open.count::<4>(), 2);
        assert_eq!(Threshold::Verify.count::<6>(), 4);
        assert_eq!(Threshold::Open.count::<6>(), 3);
        assert_eq!(Threshold::Verify.count::<8>(), 5);
        assert_eq!(Threshold::Open.count::<8>(), 4);
        // The write quorum plus the read quorum must be exactly copies + 1:
        assert_eq!(
            u16::from(Threshold::Verify.count::<4>()) + u16::from(Threshold::Open.count::<4>()),
            5
        );
        assert_eq!(
            u16::from(Threshold::Verify.count::<6>()) + u16::from(Threshold::Open.count::<6>()),
            7
        );
        assert_eq!(
            u16::from(Threshold::Verify.count::<8>()) + u16::from(Threshold::Open.count::<8>()),
            9
        );
    }

    #[test]
    fn working_prefers_full_quorum_with_connected_parent() {
        // An incomplete parent quorum (sequence 5) that is properly connected to sequence 6.
        let mut parent = test_header(5);
        parent.copy = 2;

        let mut copies = four_headers(6);
        copies[2] = parent.clone();
        for copy in copies.iter_mut().filter(|c| c.sequence == 6) {
            copy.parent = parent.checksum;
            // `set_checksum` asserts a pristine staging header: recompute with copy zeroed.
            let slot = copy.copy;
            copy.copy = 0;
            copy.set_checksum();
            copy.copy = slot;
        }
        let mut quorums = SuperBlockQuorums::default();
        let quorum = quorums.working(&copies, Threshold::Open).expect("quorum");

        assert_eq!(quorum.header().sequence, 6);
        assert!(quorum.valid());
        assert_eq!(quorum.copies_count(), 3);

        // The parent's slot must be rewritten with the working sequence before repair completes:
        let mut repairs = quorum.repairs();
        assert_eq!(repairs.next_slot(), Some(2));
        assert_eq!(repairs.next_slot(), None);
    }

    #[test]
    fn working_repairs_nothing_when_every_copy_matches() {
        let copies = four_headers(6);
        let mut quorums = SuperBlockQuorums::default();
        let quorum = quorums.working(&copies, Threshold::Verify).expect("quorum");

        assert!(quorum.valid());
        assert_eq!(quorum.copies_count(), SUPERBLOCK_COPIES);

        let mut repairs = quorum.repairs();
        assert!(repairs.next_slot().is_none(), "every copy is on disk");
    }

    #[test]
    fn working_not_found_when_all_checksums_invalid() {
        let mut copies = four_headers(0);
        for copy in &mut copies {
            corrupt(copy);
        }
        let mut quorums = SuperBlockQuorums::default();
        let result = quorums.working(&copies, Threshold::Open);
        assert_eq!(result.unwrap_err(), QuorumError::NotFound);
    }

    #[test]
    fn working_quorum_lost_below_threshold() {
        let mut copies = four_headers(0);
        for copy in copies.iter_mut().skip(1) {
            corrupt(copy);
        }
        let mut quorums = SuperBlockQuorums::default();
        // Only one valid copy: neither the open nor the verify threshold is reached.
        let open = quorums.working(&copies, Threshold::Open);
        assert_eq!(open.unwrap_err(), QuorumError::QuorumLost);
        let verify = quorums.working(&copies, Threshold::Verify);
        assert_eq!(verify.unwrap_err(), QuorumError::QuorumLost);
    }

    #[test]
    fn working_fork_between_two_same_sequence_quorums() {
        let mut a = test_header(6);
        a.view_headers_count = 1; // distinct checksum, same cluster/replica/sequence
        a.set_checksum();
        let b = test_header(6);

        let mut c = b.clone();
        c.copy = 2;
        let mut d = a.clone();
        d.copy = 3;
        let copies = [a, b, c, d];
        let mut quorums = SuperBlockQuorums::default();
        let result = quorums.working(&copies, Threshold::Open);
        assert_eq!(result.unwrap_err(), QuorumError::Fork);
    }

    #[test]
    fn repair_iterator_priorities() {
        // slots=[_,B,B,_] where B was found twice (duplicate): repair order is 3, 0, then 1 —
        // missing-and-nowhere-found first (last index wins), then missing-but-found-elsewhere.
        let mut iterator = RepairIterator { slots: [None, Some(2), Some(2), None] };
        assert_eq!(iterator.next_slot(), Some(3));
        assert_eq!(iterator.next_slot(), Some(0));
        assert_eq!(iterator.next_slot(), Some(1));
        assert_eq!(iterator.next_slot(), None);
    }

    #[test]
    fn quorum_count_tracks_copies() {
        let mut count = QuorumCount::default();
        assert_eq!(count.count(), 0);
        assert!(!count.full());
        count.set(3);
        count.set(5);
        assert_eq!(count.count(), 2);
        assert!(count.is_set(3));
        assert!(!count.is_set(4));
        assert!(!count.full());

        for index in 0..8 {
            count.set(index);
        }
        assert!(count.full());
    }
}
