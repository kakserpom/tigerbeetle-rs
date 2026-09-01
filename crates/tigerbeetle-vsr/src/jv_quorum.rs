//! Port of the JoinView quorum protocol: `JVQuorum` and the `quorum_headers`
//! algorithm from `src/vsr/replica.zig` (struct `JVQuorum`, ~line 11605).
//!
//! Upstream collects the JoinView messages of a view change into a message array
//! indexed by replica (each slot an optional `*Message.JoinView`), then — from the
//! JVs whose `log_view` is maximal, the *canonical* JVs — determines:
//!
//! - `commit_max`: the highest definitely-committed op;
//! - `op_head`: the head of the new log, truncating definitely-uncommitted ops via
//!   the "Ctrl" variant of Protocol-Aware Recovery (an op is truncated once
//!   `quorum_nack_prepare` replicas nack it);
//! - the consecutive run of verified headers to install, high-to-low op.
//!
//! This port is sans-I/O: JVs arrive already decoded as [`JoinedView`] (typed
//! header + the JV body as a `Vec<Prepare>`), and the message array is a plain
//! `[Option<JoinedView>]` indexed by replica.
//!
//! DEVIATION: upstream stores `*const Message` and computes lazily; here the
//! canonical headers are materialized eagerly into a `Vec` when the quorum is
//! complete.

use tigerbeetle_core::constants;

use crate::Operation;
use crate::message_header;

/// A received, decoded JoinView message.
#[derive(Clone, Debug)]
pub struct JoinedView {
    /// The typed JV header.
    pub header: message_header::JoinView,
    /// The JV body: prepare headers in descending-op order; index 0 corresponds
    /// to `header.op`. May contain blanks (ops the sender did not prepare).
    pub headers: Vec<message_header::Prepare>,
}

/// Whether a JV body header is a real prepare or a placeholder ("blank": the op
/// was not prepared / is not present).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JvHeaderType {
    Blank,
    Valid,
}

/// `vsr.Headers.jv_header_type`.
///
/// DEVIATION: upstream compares against a zero-filled blank template (all fields
/// zero except `op`) and asserts checksum validity of non-blank headers; since
/// sans-I/O prepares are checksum-valid by construction and `send_join_view` never
/// produces blanks, a header is blank iff its operation is `RESERVED`.
#[must_use]
pub fn jv_header_type(header: &message_header::Prepare) -> JvHeaderType {
    if header.operation == Operation::RESERVED { JvHeaderType::Blank } else { JvHeaderType::Valid }
}

/// The JVs collected so far, indexed by replica slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuorumOptions {
    pub replica_count: u16,
    pub quorum_view_change: u16,
    pub quorum_nack_prepare: u16,
}

/// Outcome of `quorum_headers`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuorumHeadersResult {
    /// The quorum has fewer than `quorum_view_change` JVs; keep waiting.
    AwaitingQuorum,
    /// The quorum has at least `quorum_view_change` JVs but fewer than
    /// `replica_count`; the JVs collected are insufficient to determine which
    /// headers can be truncated (excess of faults). Wait for more JVs.
    AwaitingRepair,
    /// Every replica contributed a JV but there are too many faults to start a
    /// new view; the cluster cannot rebuild the log.
    CompleteInvalid,
    /// The quorum is complete and sufficient to start the new view.
    CompleteValid {
        /// The head op of the new log.
        op_head: u64,
        /// The highest definitely-committed op (`commit_max`).
        op_min: u64,
        /// Headers from `op_head` down to `op_min` (inclusive), consecutive and
        /// verified (as yielded by the upstream `HeaderIterator`).
        headers: Vec<message_header::Prepare>,
    },
}

fn jvs_all(jvs: &[Option<JoinedView>]) -> Vec<&JoinedView> {
    let mut array = Vec::new();
    for (replica, received) in jvs.iter().enumerate() {
        if let Some(message) = received {
            assert_eq!(
                message.header.replica,
                u8::try_from(replica).unwrap_or_else(|_| panic!("replica index exceeds u8"))
            );
            array.push(message);
        }
    }
    array
}

/// The JVs whose `log_view` is the highest of any JV. Their headers are
/// canonical: the replica(s) have knowledge of previous view changes in which
/// headers were replaced.
#[must_use]
pub fn jvs_canonical(jvs: &[Option<JoinedView>]) -> Vec<&JoinedView> {
    jvs_with_log_view(jvs, log_view_max(jvs))
}

/// The JVs with a specific `log_view`.
#[must_use]
pub fn jvs_with_log_view(jvs: &[Option<JoinedView>], log_view: u32) -> Vec<&JoinedView> {
    let mut array = Vec::new();
    for message in jvs_all(jvs) {
        if message.header.log_view == log_view {
            array.push(message);
        }
    }
    array
}

/// The JVs with a `log_view` below the maximum.
///
/// # Panics
/// Panics if a JV's `log_view` exceeds the maximum (an invariant violation), or
/// if the quorum contains no JVs.
#[must_use]
pub fn jvs_uncanonical(jvs: &[Option<JoinedView>]) -> Vec<&JoinedView> {
    let log_view_max_ = log_view_max(jvs);
    let mut array = Vec::new();
    for message in jvs_all(jvs) {
        assert!(message.header.log_view <= log_view_max_);
        if message.header.log_view < log_view_max_ {
            array.push(message);
        }
    }
    array
}

/// The highest `log_view` of any JV.
///
/// # Panics
/// Panics if the quorum contains no JVs, or if a JV's `log_view >= view`
/// (an invariant violation).
#[must_use]
pub fn log_view_max(jvs: &[Option<JoinedView>]) -> u32 {
    let mut maximum: Option<u32> = None;
    for message in jvs_all(jvs) {
        // The `log_view` may be higher than the view in any of the prepare
        // headers, but must be lower than the view of this view change.
        assert!(message.header.log_view < message.header.view);
        let log_view = message.header.log_view;
        maximum = Some(match maximum {
            None => log_view,
            Some(current) => current.max(log_view),
        });
    }
    maximum.unwrap_or_else(|| panic!("empty JV quorum"))
}

/// The highest `checkpoint_op` of any JV.
///
/// # Panics
/// Panics if the quorum contains no JVs.
#[must_use]
pub fn op_checkpoint_max(jvs: &[Option<JoinedView>]) -> u64 {
    let mut maximum: Option<u64> = None;
    for jv in jvs_all(jvs) {
        let checkpoint = jv.header.checkpoint_op;
        maximum = Some(match maximum {
            None => checkpoint,
            Some(current) => current.max(checkpoint),
        });
    }
    maximum.unwrap_or_else(|| panic!("empty JV quorum"))
}

/// The highest op that is definitely committed (or an op that is uncommitted but
/// outside the pipeline, and therefore also definitely committed).
///
/// # Panics
/// Panics if the quorum contains no JVs.
#[must_use]
pub fn commit_max(jvs: &[Option<JoinedView>]) -> u64 {
    let all = jvs_all(jvs);
    assert!(!all.is_empty());

    let mut commit_maximum: u64 = 0;
    for jv in all {
        let jv_headers = &jv.headers;
        // JV generation stops when a header with op <= commit_max is appended.
        let jv_commit_max_tail = jv_headers[jv_headers.len() - 1].op;
        // An op cannot be uncommitted if it is definitely outside the pipeline.
        // Use `join_view_op_head` instead of `replica.op` since the former is
        // about to become the new `replica.op`.
        let jv_commit_max_pipeline =
            jv.header.op.saturating_sub(u64::from(constants::PIPELINE_PREPARE_QUEUE_MAX));

        commit_maximum = commit_maximum.max(jv_commit_max_tail);
        commit_maximum = commit_maximum.max(jv_commit_max_pipeline);
        commit_maximum = commit_maximum.max(jv.header.commit_min);
        commit_maximum = commit_maximum.max(jv_headers[0].commit);
    }
    commit_maximum
}

/// The highest `timestamp` of any JV (for the new primary's `prepare_timestamp`).
///
/// # Panics
/// Panics if the quorum contains no JVs.
#[must_use]
pub fn timestamp_max(jvs: &[Option<JoinedView>]) -> u64 {
    let mut maximum: Option<u64> = None;
    for jv in jvs_all(jvs) {
        let jv_head = &jv.headers[0];
        maximum = Some(match maximum {
            None => jv_head.timestamp,
            Some(current) => current.max(jv_head.timestamp),
        });
    }
    maximum.unwrap_or_else(|| panic!("empty JV quorum"))
}

/// The highest head op of any canonical JV.
///
/// # Panics
/// Panics if there are no canonical JVs.
#[must_use]
pub fn op_max_canonical(jvs: &[Option<JoinedView>]) -> u64 {
    let mut maximum: Option<u64> = None;
    for message in jvs_canonical(jvs) {
        maximum = Some(match maximum {
            None => message.header.op,
            Some(current) => current.max(message.header.op),
        });
    }
    maximum.unwrap_or_else(|| panic!("no canonical JVs"))
}

#[must_use]
fn bit(bitset: u128, index: usize) -> bool {
    assert!(index < 128);
    bitset & (1u128 << index) != 0
}

/// Decide whether the view may begin, and if so, which ops survive.
///
/// Port of `JVQuorum.quorum_headers` (`src/vsr/replica.zig:11800`).
///
/// # Panics
/// Panics on invariant violations (e.g. `commit_max > op_head_max`, or a
/// mismatch between a *canonical* JV and another JV admitted with the same
/// `log_view`).
#[must_use]
#[allow(clippy::cast_possible_truncation)] // op indices are bounded by the JV body length
pub fn quorum_headers(jvs: &[Option<JoinedView>], options: QuorumOptions) -> QuorumHeadersResult {
    assert!(options.replica_count >= 2);
    assert!(options.quorum_view_change >= 2);
    assert!(options.quorum_view_change <= options.replica_count);
    if options.replica_count == 2 {
        assert_eq!(options.quorum_nack_prepare, 1);
    } else {
        assert_eq!(options.quorum_nack_prepare, options.quorum_view_change);
    }

    let all = jvs_all(jvs);
    if all.len() < usize::from(options.quorum_view_change) {
        return QuorumHeadersResult::AwaitingQuorum;
    }

    let log_view_canonical = log_view_max(jvs);
    let jvs_canonical_ = jvs_canonical(jvs);
    assert!(!jvs_canonical_.is_empty());
    assert!(jvs_canonical_.len() <= all.len());

    let op_head_max = op_max_canonical(jvs);
    let op_head_min = commit_max(jvs);
    assert!(op_head_min <= op_head_max);

    // Iterate the highest definitely committed op and all maybe-uncommitted ops.
    let mut op = op_head_min;
    let mut op_head = op_head_max;
    while op <= op_head_max {
        let header_canonical: Option<&message_header::Prepare> = {
            let mut canonical: Option<&message_header::Prepare> = None;
            for jv in &jvs_canonical_ {
                // This JV is canonical, but lagging far behind.
                if jv.header.op < op {
                    continue;
                }
                let headers = &jv.headers;
                let header_index = (jv.header.op - op) as usize;
                assert!(header_index < headers.len());

                let header = &headers[header_index];
                assert_eq!(header.op, op);

                if jv_header_type(header) == JvHeaderType::Valid {
                    canonical = Some(header);
                    break;
                }
            }
            canonical
        };

        let mut copies: usize = 0;
        let mut nacks: usize = 0;
        for jv in &all {
            if jv.header.op < op {
                nacks += 1;
                continue;
            }

            let headers = &jv.headers;
            let header_index = (jv.header.op - op) as usize;
            if header_index >= headers.len() {
                nacks += 1;
                continue;
            }

            let header = &headers[header_index];
            assert_eq!(header.op, op);
            assert!(header.view <= log_view_canonical);

            if jv_header_type(header) == JvHeaderType::Valid
                && bit(jv.header.present_bitset, header_index)
                && header_canonical.is_some_and(|h| h.checksum == header.checksum)
            {
                copies += 1;
            }

            if bit(jv.header.nack_bitset, header_index) {
                // The op is nacked explicitly.
                nacks += 1;
            } else if jv_header_type(header) == JvHeaderType::Valid {
                if header_canonical.is_some_and(|h| h.checksum != header.checksum) {
                    assert!(jv.header.log_view < log_view_canonical);
                    // The op is nacked implicitly: the replica has a different header.
                    nacks += 1;
                }
                if header_canonical.is_none() {
                    assert!(header.view < log_view_canonical);
                    assert!(jv.header.log_view < log_view_canonical);
                    // The op is nacked implicitly: the header has already been
                    // truncated in the latest log_view.
                    nacks += 1;
                }
            }
        }

        // An abbreviated version of Protocol-Aware Recovery's Ctrl protocol.
        // When we can confirm that an op is definitely uncommitted, truncate it
        // to improve availability.
        if nacks >= usize::from(options.quorum_nack_prepare) {
            // Never nack `op_head_min` (aka `commit_max`).
            assert!(op > op_head_min);
            op_head = op - 1;
            break;
        }

        if header_canonical.is_none() || (header_canonical.is_some() && copies == 0) {
            if all.len() < usize::from(options.replica_count) {
                return QuorumHeadersResult::AwaitingRepair;
            }
            return QuorumHeadersResult::CompleteInvalid;
        }

        // This op is eligible to be the view's head.
        assert!(header_canonical.is_some() && copies > 0);
        op += 1;
    }
    assert!(op_head >= op_head_min);
    assert!(op_head <= op_head_max);

    let headers = headers_for_view(&jvs_canonical_, op_head, op_head_min);
    QuorumHeadersResult::CompleteValid { op_head, op_min: op_head_min, headers }
}

/// Iterate the consecutive headers of a set of same-`log_view` JVs, from
/// high-to-low op. Port of the upstream `HeaderIterator` (`replica.zig:11945`).
///
/// # Panics
/// Panics on invariant violations (e.g. a checksum mismatch between copies of
/// the same op in different JVs, or a broken parent chain).
#[must_use]
#[allow(clippy::cast_possible_truncation)] // op indices are bounded by the JV body length
pub fn headers_for_view(
    jvs_canonical: &[&JoinedView],
    op_max: u64,
    op_min: u64,
) -> Vec<message_header::Prepare> {
    assert!(!jvs_canonical.is_empty());
    assert!(op_min <= op_max);

    let mut result = Vec::new();
    let mut child_op: Option<u64> = None;
    let mut child_parent: Option<u128> = None;

    loop {
        assert_eq!(child_op.is_some(), child_parent.is_some());
        if child_op.is_some_and(|op| op == op_min) {
            return result;
        }

        let op = child_op.unwrap_or(op_max + 1) - 1;

        let mut header: Option<&message_header::Prepare> = None;
        let log_view = jvs_canonical[0].header.log_view;
        for jv in jvs_canonical {
            assert_eq!(log_view, jv.header.log_view);

            if op > jv.header.op {
                continue;
            }

            let jv_headers = &jv.headers;
            let jv_header_index = (jv.header.op - op) as usize;
            if jv_header_index >= jv_headers.len() {
                continue;
            }

            let jv_header = &jv_headers[jv_header_index];
            if jv_header_type(jv_header) == JvHeaderType::Valid {
                if let Some(h) = header {
                    assert_eq!(h.checksum, jv_header.checksum);
                } else {
                    header = Some(jv_header);
                }
            }
        }

        if let Some(parent) = child_parent {
            assert!(header.is_some_and(|h| h.checksum == parent), "broken parent chain at op {op}");
        }
        let Some(header_found) = header else {
            panic!("no header found for op {op}");
        };
        child_op = Some(op);
        child_parent = Some(header_found.parent);
        result.push(*header_found);
    }
}

/// Verify that the JVs collected so far are consistent: no two JVs with the same
/// `log_view` disagree about the checksum of any op.
///
/// Port of `JVQuorum.verify` (`replica.zig:11608`).
///
/// # Panics
/// Panics if a JV is structurally invalid or if two same-`log_view` JVs conflict.
#[allow(clippy::cast_possible_truncation)] // op indices are bounded by the JV body length
pub fn verify(jvs: &[Option<JoinedView>]) {
    let all = jvs_all(jvs);
    for message in &all {
        verify_message(message);
    }

    // Verify that JVs with the same `log_view` do not conflict.
    for (i, jv_a) in all.iter().enumerate() {
        for jv_b in &all[0..i] {
            if jv_a.header.log_view != jv_b.header.log_view {
                continue;
            }

            let headers_a = &jv_a.headers;
            let headers_b = &jv_b.headers;
            // Find the intersection of the ops covered by each JV.
            let op_max = jv_a.header.op.min(jv_b.header.op);
            let op_min = headers_a[headers_a.len() - 1].op.max(headers_b[headers_b.len() - 1].op);
            // If a replica is lagging, its headers may not overlap at all.
            if op_min > op_max {
                continue;
            }

            for op in op_min..=op_max {
                let left_header = &headers_a[(jv_a.header.op - op) as usize];
                let right_header = &headers_b[(jv_b.header.op - op) as usize];
                if jv_header_type(left_header) == JvHeaderType::Valid
                    && jv_header_type(right_header) == JvHeaderType::Valid
                {
                    assert_eq!(left_header.checksum, right_header.checksum);
                }
            }
        }
    }
}

#[allow(clippy::cast_possible_truncation)] // JV bodies are far smaller than u32::MAX
fn verify_message(message: &JoinedView) {
    assert!(message.header.commit_min <= message.header.op);
    assert!(message.header.checkpoint_op <= message.header.commit_min);

    let log_view = message.header.log_view;
    assert!(log_view < message.header.view);

    let headers = &message.headers;
    assert!(!headers.is_empty());
    assert!(
        headers.len()
            <= usize::try_from(constants::PIPELINE_PREPARE_QUEUE_MAX)
                .unwrap_or_else(|_| panic!("PIPELINE_PREPARE_QUEUE_MAX exceeds usize"))
                + 1
    );
    assert_eq!(headers[0].op, message.header.op);
    assert!(headers[0].view <= log_view);

    assert!(message.header.nack_bitset.count_ones() <= headers.len() as u32);
    assert!(message.header.nack_bitset.leading_zeros() + headers.len() as u32 >= 128);

    assert!(message.header.present_bitset.count_ones() <= headers.len() as u32);
    assert!(message.header.present_bitset.leading_zeros() + headers.len() as u32 >= 128);
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::message_header::{Prepare, TypedHeader};
    use crate::multiversion::Release;

    fn make_prepare(cluster: u128, op: u64, parent: u128, view: u32, commit: u64) -> Prepare {
        let mut header = Prepare {
            cluster,
            release: Release::ZERO,
            operation: crate::Operation::NOOP,
            op,
            parent,
            view,
            commit,
            ..Prepare::default()
        };
        header.set_checksum_body(&[]);
        header.set_checksum();
        header
    }

    /// Build a contiguous chain of prepares for ops `1..=op_max` (root parent).
    fn chain(cluster: u128, op_max: u64, view: u32, commit: u64) -> Vec<Prepare> {
        let root = Prepare::root(cluster);
        let mut headers = Vec::new();
        let mut parent = root.checksum();
        for op in 1..=op_max {
            let h = make_prepare(cluster, op, parent, view, commit);
            parent = h.checksum();
            headers.push(h);
        }
        headers
    }

    /// The JV body preceding from `headers` (indexed by `op - 1`, ascending op):
    /// a descending slice from op `head` down to `tail`.
    #[allow(clippy::cast_possible_truncation)] // test-only helper, op <= headers.len() <= u16
    fn jv_headers(headers: &[Prepare], head: u64, tail: u64) -> Vec<Prepare> {
        assert!(tail >= 1);
        assert!(tail <= head);
        assert!(head <= headers.len() as u64);
        headers[(tail as usize - 1)..(head as usize)].iter().rev().copied().collect()
    }

    #[allow(clippy::too_many_arguments)] // test-only helper mirroring the JV header fields
    fn joined_view(
        replica_index: u8,
        view: u32,
        log_view: u32,
        op: u64,
        commit_min: u64,
        checkpoint_op: u64,
        present_bitset: u128,
        nack_bitset: u128,
        headers: Vec<Prepare>,
    ) -> JoinedView {
        let header = message_header::JoinView {
            replica: replica_index,
            view,
            log_view,
            op,
            commit_min,
            checkpoint_op,
            present_bitset,
            nack_bitset,
            ..message_header::JoinView::default()
        };
        // Body is strictly descending from `op` (no blanks in these tests).
        assert_eq!(headers.first().unwrap().op, op);
        for pair in headers.windows(2) {
            assert_eq!(pair[0].op, pair[1].op + 1);
        }
        JoinedView { header, headers }
    }

    #[test]
    fn jv_header_type_detects_reserved_as_blank() {
        let mut blank = Prepare { operation: crate::Operation::RESERVED, ..Prepare::default() };
        blank.op = 7;
        assert_eq!(jv_header_type(&blank), JvHeaderType::Blank);

        let valid = make_prepare(0, 7, 1, 0, 0);
        assert_eq!(jv_header_type(&valid), JvHeaderType::Valid);
    }

    #[test]
    fn commit_max_takes_max_of_all_contributions() {
        // Replica 0: head commit=5 (op 7, body [7,6,5,4]), commit_min=3.
        let h = chain(0, 7, 0, 5);
        let jv0 = joined_view(0, 1, 0, 7, 3, 0, 0, 0, jv_headers(&h, 7, 4));

        // Replica 1: lagging, but commit_min=8 while head op=9.
        let k = chain(0, 9, 0, 2);
        let jv1 = joined_view(1, 1, 0, 9, 8, 0, 0, 0, jv_headers(&k, 9, 2));

        let jvs = vec![Some(jv0), Some(jv1)];
        assert_eq!(commit_max(&jvs), 8);
        assert_eq!(op_checkpoint_max(&jvs), 0);
    }

    #[test]
    fn quorum_headers_complete_valid_preserves_head_with_quorum_copies() {
        // 3-replica cluster. All three replicas prepared ops 1..3 in view 0 and
        // committed through op 1. Replica 3 lost op 3 (nacks it).
        let h = chain(0, 3, 0, 1);
        let r1 = joined_view(0, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        let r2 = joined_view(1, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        let r3 = joined_view(2, 1, 0, 3, 1, 0, 0b110, 0b001, jv_headers(&h, 3, 1));
        let jvs = vec![Some(r1), Some(r2), Some(r3)];

        verify(&jvs);

        let options =
            QuorumOptions { replica_count: 3, quorum_view_change: 2, quorum_nack_prepare: 2 };
        match quorum_headers(&jvs, options) {
            QuorumHeadersResult::CompleteValid { op_head, op_min, headers } => {
                assert_eq!(op_head, 3);
                assert_eq!(op_min, 1);
                // High-to-low, consecutive and consistent with the chain.
                assert_eq!(headers.iter().map(|h| h.op).collect::<Vec<_>>(), vec![3, 2, 1]);
                assert_eq!(headers, jv_headers(&h, 3, 1));
            }
            otherwise => panic!("expected CompleteValid, got {otherwise:?}"),
        }
    }

    #[test]
    fn quorum_headers_nack_quorum_truncates_uncommitted_head() {
        // Only replica 0 has op 3 present; both others nack it. The nack quorum
        // (2) confirms op 3 was never replicated to a quorum, so it is
        // truncated: op_head drops to 2 (the committed op).
        let h = chain(0, 3, 0, 1);
        let r1 = joined_view(0, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        let r2 = joined_view(1, 1, 0, 3, 1, 0, 0b110, 0b001, jv_headers(&h, 3, 1));
        let r3 = joined_view(2, 1, 0, 3, 1, 0, 0b110, 0b001, jv_headers(&h, 3, 1));
        let jvs = vec![Some(r1), Some(r2), Some(r3)];

        match quorum_headers(
            &jvs,
            QuorumOptions { replica_count: 3, quorum_view_change: 2, quorum_nack_prepare: 2 },
        ) {
            QuorumHeadersResult::CompleteValid { op_head, op_min, headers } => {
                assert_eq!(op_head, 2);
                assert_eq!(op_min, 1);
                assert_eq!(headers.iter().map(|h| h.op).collect::<Vec<_>>(), vec![2, 1]);
            }
            otherwise => panic!("expected CompleteValid, got {otherwise:?}"),
        }
    }

    #[test]
    fn quorum_headers_awaiting_quorum_below_view_change_threshold() {
        let h = chain(0, 3, 0, 1);
        let r1 = joined_view(0, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        // Only one JV; the view-change quorum is 2.
        let jvs = vec![Some(r1), None, None];
        assert_eq!(
            quorum_headers(
                &jvs,
                QuorumOptions { replica_count: 3, quorum_view_change: 2, quorum_nack_prepare: 2 }
            ),
            QuorumHeadersResult::AwaitingQuorum
        );
    }

    #[test]
    fn quorum_headers_awaiting_repair_with_excess_faults() {
        // 5-replica cluster; 4 JVs collected. The head op is present in no JV
        // (copies == 0) and is nacked by only 2 replicas (< nack_prepare 3), so
        // the quorum cannot yet determine whether it was committed.
        let h = chain(0, 5, 0, 2);
        let body = jv_headers(&h, 5, 2); // [5,4,3,2]
        let r0 = joined_view(0, 1, 0, 5, 2, 0, 0b1000, 0, body.clone());
        let r1 = joined_view(1, 1, 0, 5, 2, 0, 0b1000, 0, body.clone());
        let r2 = joined_view(2, 1, 0, 5, 2, 0, 0b1000, 0b0111, body.clone());
        let r3 = joined_view(3, 1, 0, 5, 2, 0, 0b1000, 0b0111, body);
        let jvs = vec![Some(r0), Some(r1), Some(r2), Some(r3), None];

        assert_eq!(
            quorum_headers(
                &jvs,
                QuorumOptions { replica_count: 5, quorum_view_change: 3, quorum_nack_prepare: 3 }
            ),
            QuorumHeadersResult::AwaitingRepair
        );
    }

    #[test]
    fn quorum_headers_complete_invalid_when_all_replicas_faulty_at_head() {
        // 3-replica cluster, all JVs collected, but the head op is present in
        // none of them and nacked by none — the log cannot be rebuilt.
        let h = chain(0, 3, 0, 1);
        let body = jv_headers(&h, 3, 1); // [3,2,1]
        let r0 = joined_view(0, 1, 0, 3, 1, 0, 0b100, 0, body.clone());
        let r1 = joined_view(1, 1, 0, 3, 1, 0, 0b100, 0, body.clone());
        let r2 = joined_view(2, 1, 0, 3, 1, 0, 0b100, 0, body);
        let jvs = vec![Some(r0), Some(r1), Some(r2)];

        assert_eq!(
            quorum_headers(
                &jvs,
                QuorumOptions { replica_count: 3, quorum_view_change: 2, quorum_nack_prepare: 2 }
            ),
            QuorumHeadersResult::CompleteInvalid
        );
    }

    #[test]
    fn headers_for_view_yields_consecutive_headers_high_to_low() {
        // Two same-log_view JVs with overlapping bodies; the iterator must agree
        // on checksums and return [3,2,1].
        let h = chain(0, 3, 0, 1);
        let r0 = joined_view(0, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        let r1 = joined_view(1, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        let jvs = [Some(r0), Some(r1)];
        let canonical = jvs_canonical(&jvs);

        let iterated = headers_for_view(&canonical, 3, 1);
        assert_eq!(iterated.iter().map(|h| h.op).collect::<Vec<_>>(), vec![3, 2, 1]);
        assert_eq!(iterated, jv_headers(&h, 3, 1));
    }

    #[test]
    fn verify_rejects_conflicting_same_log_view_jvs() {
        let h = chain(0, 3, 0, 1);
        let mut r0 = joined_view(0, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        // Corrupt replica 0's copy of op 2: a different (valid) checksum.
        r0.headers[1] = make_prepare(0, 2, 42, 0, 1);
        let r1 = joined_view(1, 1, 0, 3, 1, 0, 0b111, 0, jv_headers(&h, 3, 1));
        let jvs = vec![Some(r0), Some(r1), None];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify(&jvs);
        }));
        assert!(result.is_err());
    }

    // -- Aggregate helpers (untested canonical/max/filter paths) --

    #[test]
    fn log_view_max_takes_max_and_panics_on_empty_or_invariant() {
        let h = chain(0, 4, 3, 1); // log_view 3
        let jvs = vec![
            Some(joined_view(0, 5, 3, 4, 1, 0, 0, 0, jv_headers(&h, 4, 1))),
            Some(joined_view(1, 5, 3, 2, 1, 0, 0, 0, jv_headers(&h, 2, 1))),
            Some(joined_view(2, 5, 1, 3, 1, 0, 0, 0, jv_headers(&chain(0, 3, 1, 1), 3, 1))),
        ];
        assert_eq!(log_view_max(&jvs), 3);

        // An empty quorum has no max.
        let empty: Vec<Option<JoinedView>> = vec![None, None];
        assert!(std::panic::catch_unwind(|| log_view_max(&empty)).is_err());

        // log_view must stay below the JV's own `view` (protocol invariant).
        let bad =
            vec![Some(joined_view(0, 4, 5, 3, 1, 0, 0, 0, jv_headers(&chain(0, 3, 4, 1), 3, 1)))];
        assert!(std::panic::catch_unwind(|| log_view_max(&bad)).is_err());
    }

    #[test]
    fn canonical_split_selects_max_log_view_quorum() {
        let h = chain(0, 4, 3, 1); // log_view 3
        let jvs = vec![
            Some(joined_view(0, 5, 3, 4, 1, 0, 0, 0, jv_headers(&h, 4, 1))),
            Some(joined_view(1, 5, 3, 3, 1, 0, 0, 0, jv_headers(&h, 3, 1))),
            Some(joined_view(2, 5, 1, 2, 1, 0, 0, 0, jv_headers(&chain(0, 2, 1, 1), 2, 1))),
            None,
        ];

        let canonical = jvs_canonical(&jvs);
        assert_eq!(canonical.len(), 2);
        assert!(canonical.iter().all(|jv| jv.header.log_view == 3));
        assert_eq!(jvs_with_log_view(&jvs, 3).len(), 2);

        let uncanonical = jvs_uncanonical(&jvs);
        assert_eq!(uncanonical.len(), 1);
        assert_eq!(uncanonical[0].header.log_view, 1);

        // op_max_canonical only looks at the canonical (max log_view) JVs.
        assert_eq!(op_max_canonical(&jvs), 4);

        // No canonical JVs to maximize over.
        let none: Vec<Option<JoinedView>> = vec![None, None];
        assert!(std::panic::catch_unwind(|| op_max_canonical(&none)).is_err());
    }

    #[test]
    fn jvs_all_rejects_misplaced_replica_slots() {
        // The JV body is indexed by replica slot; a JV whose header names a
        // different replica would corrupt the quorum accounting.
        let h = chain(0, 2, 1, 1);
        let misplaced = vec![None, Some(joined_view(0, 3, 1, 2, 1, 0, 0, 0, jv_headers(&h, 2, 1)))];
        let result = std::panic::catch_unwind(|| jvs_canonical(&misplaced));
        assert!(result.is_err());
    }

    #[test]
    fn op_checkpoint_max_and_timestamp_max_aggregate() {
        let h = chain(0, 10, 1, 1); // log_view 1

        let mut body0 = jv_headers(&h, 5, 2);
        body0[0].timestamp = 1_000;
        let jv0 = joined_view(0, 3, 1, 5, 3, 2, 0, 0, body0);

        let mut body1 = jv_headers(&h, 10, 2);
        body1[0].timestamp = 42;
        let jv1 = joined_view(1, 3, 1, 10, 9, 7, 0, 0, body1);

        let jvs: Vec<Option<JoinedView>> = vec![Some(jv0), Some(jv1)];
        assert_eq!(op_checkpoint_max(&jvs), 7);
        assert_eq!(timestamp_max(&jvs), 1_000);

        // Both helpers panic on an empty quorum.
        let empty: Vec<Option<JoinedView>> = vec![None];
        assert!(std::panic::catch_unwind(|| op_checkpoint_max(&empty)).is_err());
        assert!(std::panic::catch_unwind(|| timestamp_max(&empty)).is_err());
    }
}
