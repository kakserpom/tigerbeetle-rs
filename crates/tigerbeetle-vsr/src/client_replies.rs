//! Store the latest reply to every active client session.
//!
//! This allows them to be resent to the corresponding client if the client missed the
//! original reply message (e.g. dropped packet).
//!
//! - Client replies' headers are stored in the `client_sessions` trailer.
//! - Client replies (header and body) are only stored here in the `ClientReplies` zone when
//!   `reply.header.size != sizeOf(Header)` — that is, when the body is non-empty.
//! - Corrupt client replies can be repaired from other replicas.
//!
//! Replies are written asynchronously. Subsequent writes for the same client may be
//! coalesced — we only care about the last reply to each client session.
//!
//! ClientReplies guarantees that the latest replies are durable at checkpoint.
//!
//! If the same reply is corrupted by all replicas, the cluster is still available. If the
//! respective client also never received the reply (due to a network fault), the client may
//! be "locked out" of the cluster — continually retrying a request which has been executed,
//! but whose reply has been permanently lost. This can be resolved by the operator
//! restarting the client to create a new session.
//!
//! DEVIATION: upstream drives this component with function-pointer callbacks fired from its
//! IO event loop (`read_reply_callback`, `ready(callback)`,
//! `checkpoint(callback)`). Under our interim poll-based [`Storage`], completions are
//! correlated FIFO and surfaced through [`ClientReplies::poll`] +
//! [`ClientReplies::take_events`]. Messages are owned per `message.rs`'s deviation (no
//! refcounts), so resolving a pending read through a concurrent write clones the message.

#![allow(clippy::cast_possible_truncation)] // slot indices and reply sizes fit in u32/usize

use std::collections::VecDeque;

use tigerbeetle_core::constants::{
    CLIENT_REPLIES_IOPS_READ_MAX, CLIENT_REPLIES_IOPS_WRITE_MAX, CLIENTS_MAX, SECTOR_SIZE,
};
use tigerbeetle_core::stdx::bitset::BitSet;

use crate::message::{MESSAGE_SIZE_MAX, Message};
use crate::message_header;
use crate::storage::{Completion, ReadRequest, Storage, WriteRequest, zeroed_buffer};
use crate::{Zone, client_sessions};

const READS_MAX: usize = CLIENT_REPLIES_IOPS_READ_MAX as usize;
const WRITES_MAX: usize = CLIENT_REPLIES_IOPS_WRITE_MAX as usize;
const CLIENTS_COUNT: usize = CLIENTS_MAX as usize;

fn slot_offset(slot: client_sessions::ReplySlot) -> u64 {
    (slot.index * MESSAGE_SIZE_MAX) as u64
}

fn sector_ceil(size: u32) -> usize {
    (size as usize).div_ceil(SECTOR_SIZE) * SECTOR_SIZE
}

/// Why a reply write was started (upstream: `WriteTrigger`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteTrigger {
    /// A new commit; the write must be durable before the next checkpoint.
    Commit,
    /// Repairing a corrupt/missing reply.
    Repair,
}

/// The result of a completed read (upstream delivers these via the read callback).
#[derive(Debug)]
pub enum ReadOutcome {
    /// The on-disk reply matches the expected header.
    Found(Message),
    /// A pending read was satisfied by a concurrent `write_reply` of the same reply.
    ResolvedByWrite(Message),
    /// The reply failed its checksum validation.
    Corrupt,
    /// The slot holds a different reply than expected (older/newer/misdirected).
    Unexpected,
}

/// Events for the owner, drained via [`ClientReplies::take_events`]
/// (upstream: callback invocations).
#[allow(clippy::large_enum_variant)] // ReadReply mirrors upstream's callback payload
#[derive(Debug)]
pub enum Event {
    ReadReply {
        slot: client_sessions::ReplySlot,
        /// Header of the expected reply (from the session).
        header: message_header::Reply,
        destination_replica: Option<u8>,
        outcome: ReadOutcome,
    },
    /// Another `write_reply()` may be started ([`ClientReplies::ready`] was waiting).
    Ready,
    /// All commit-trigger writes are durable; the checkpoint may proceed.
    CheckpointDone,
}

/// Error returned by [`ClientReplies::read_reply`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// All read IOPs are currently in use.
    Busy,
}

#[derive(Debug)]
struct ReadOp {
    slot: client_sessions::ReplySlot,
    /// Header of the expected reply.
    header: message_header::Reply,
    destination_replica: Option<u8>,
    /// Whether the owner still wants an event for this read (upstream:
    /// `callback != null`; cleared when resolved by a concurrent write).
    wanted: bool,
}

#[derive(Debug)]
struct WriteOp {
    slot: client_sessions::ReplySlot,
    message: Message,
    trigger: WriteTrigger,
}

/// Port of `ClientRepliesType(Storage)` (see module-level deviations).
#[derive(Debug)]
pub struct ClientReplies {
    replica: u8,

    /// Acquired read operation slots; `None` when free
    /// (upstream: `IOPSType(Read, client_replies_iops_read_max)`).
    reads: Vec<Option<ReadOp>>,
    /// Outstanding reads in issue order, for correlating storage completions.
    reads_in_flight: VecDeque<usize>,

    /// Acquired write operation slots; `None` when free.
    writes: Vec<Option<WriteOp>>,
    /// Acquired writes (queued *and* executing) in acquisition order, matching the
    /// storage's FIFO completion order.
    writes_in_flight: VecDeque<usize>,
    /// Indices into `writes` that are queued behind another write to the same slot
    /// (upstream: `write_queue`).
    write_queue: VecDeque<usize>,

    /// Which slots have a write currently in progress.
    writing: BitSet,
    /// Which slots hold a corrupt reply, or are otherwise missing the reply that
    /// ClientSessions believes they should hold.
    ///
    /// Invariants:
    /// - Set bits must correspond to occupied slots in ClientSessions.
    /// - Set bits must correspond to entries in ClientSessions whose header has a body
    ///   (`size != sizeOf(Header)`).
    faulty: BitSet,

    /// Set by [`ClientReplies::ready`]; fires [`Event::Ready`] when capacity frees up
    /// (upstream: `ready_callback`).
    ready_waiting: bool,
    /// Set by [`ClientReplies::checkpoint`]; fires [`Event::CheckpointDone`] once no
    /// commit-trigger writes remain (upstream: `checkpoint_callback` + next tick).
    checkpoint_waiting: bool,

    events: VecDeque<Event>,
}

impl ClientReplies {
    #[must_use]
    pub fn new(replica_index: u8) -> Self {
        Self {
            replica: replica_index,
            reads: (0..READS_MAX).map(|_| None).collect(),
            reads_in_flight: VecDeque::new(),
            writes: (0..WRITES_MAX).map(|_| None).collect(),
            writes_in_flight: VecDeque::new(),
            write_queue: VecDeque::new(),
            writing: BitSet::new_empty(CLIENTS_COUNT),
            faulty: BitSet::new_empty(CLIENTS_COUNT),
            ready_waiting: false,
            checkpoint_waiting: false,
            events: VecDeque::new(),
        }
    }

    #[must_use]
    pub fn replica(&self) -> u8 {
        self.replica
    }

    /// Returns true if the reply at the given slot is durably persisted to disk. The
    /// difference with the `faulty` bit is that `faulty` is cleared at the start of a write
    /// while the reply is still in RAM. In contrast, `reply_durable` checks that the
    /// corresponding reply hit the disk.
    #[must_use]
    pub fn reply_durable(&self, slot: client_sessions::ReplySlot) -> bool {
        !self.faulty.get(slot.index) && !self.writing.get(slot.index)
    }

    /// If a write is in progress for `client`'s slot, returns the newest in-flight message
    /// so callers can serve the reply from RAM instead of disk (upstream:
    /// `read_reply_sync`).
    ///
    /// Returns `None` both when nothing is being written to the slot and when what is being
    /// written does not match the session's expectation. The latter happens after state
    /// sync, where `client_sessions` is updated without waiting for in-flight writes.
    ///
    /// # Panics
    /// Asserts (as upstream) that a checksum-mismatching in-flight write implies the slot
    /// was marked faulty.
    #[must_use]
    pub fn write_in_flight_latest(
        &self,
        slot: client_sessions::ReplySlot,
        session: &client_sessions::Entry,
    ) -> Option<&Message> {
        if !self.writing.get(slot.index) {
            return None;
        }

        let client = session.header.client;
        let mut latest: Option<(u32, usize)> = None;
        for &index in &self.writes_in_flight {
            let Some(write) = self.writes[index].as_ref() else { continue };
            let Some(header) = write.message.header::<message_header::Reply>() else {
                continue;
            };
            if header.client != client {
                continue;
            }
            if latest.is_none_or(|(request, _)| header.request > request) {
                latest = Some((header.request, index));
            }
        }

        let (_, index) = latest?;
        let write = self.writes[index].as_ref()?;
        let header = write.message.header::<message_header::Reply>()?;

        // We are writing something, but that may be the wrong reply according to
        // `client_sessions`; such a slot must be marked faulty:
        assert!(
            header.checksum == session.header.checksum || self.faulty.get(slot.index),
            "mismatching in-flight write without faulty bit"
        );
        if header.checksum != session.header.checksum {
            return None;
        }

        Some(&write.message)
    }

    /// Starts reading a reply from disk.
    ///
    /// # Errors
    /// [`ReadError::Busy`] when all read operations are in use (upstream: `error.Busy`).
    ///
    /// # Panics
    /// Asserts the caller checked [`ClientReplies::write_in_flight_latest`] first, so a
    /// reply that is still being written is served from RAM instead of disk.
    pub fn read_reply(
        &mut self,
        storage: &mut dyn Storage,
        slot: client_sessions::ReplySlot,
        session: &client_sessions::Entry,
        destination_replica: Option<u8>,
    ) -> Result<(), ReadError> {
        assert!(
            self.write_in_flight_latest(slot, session).is_none(),
            "caller must check write_in_flight_latest() first"
        );

        let Some(read_index) = self.reads.iter().position(Option::is_none) else {
            return Err(ReadError::Busy);
        };

        let size_ceil = sector_ceil(session.header.size);
        storage.read_sectors(ReadRequest {
            zone: Zone::ClientReplies,
            offset_in_zone: slot_offset(slot),
            buffer: zeroed_buffer(size_ceil),
        });

        self.reads[read_index] =
            Some(ReadOp { slot, header: session.header, destination_replica, wanted: true });
        self.reads_in_flight.push_back(read_index);
        Ok(())
    }

    /// Whether another `write_reply()` can be started right now.
    #[must_use]
    pub fn ready_sync(&self) -> bool {
        self.writes_in_flight.len() < WRITES_MAX
    }

    /// Registers interest in being told when another write can start; the notification
    /// arrives as [`Event::Ready`] during a subsequent [`ClientReplies::poll`]
    /// (upstream: `ready(callback)`).
    ///
    /// Caller must check [`ClientReplies::ready_sync`] first.
    ///
    /// # Panics
    /// Asserts no wait is already registered and capacity is indeed exhausted.
    pub fn ready(&mut self) {
        assert!(!self.ready_waiting);
        assert!(!self.ready_sync());
        self.ready_waiting = true;
    }

    /// Clears any faulty marking for the slot (upstream: `remove_reply`).
    pub fn remove_reply(&mut self, slot: client_sessions::ReplySlot) {
        self.faulty.unset(slot.index);
    }

    /// Queues a reply write. Coalesces with any queued (not yet started) write to the same
    /// slot, keeping only the newest reply.
    ///
    /// The caller is responsible for ensuring capacity by calling [`ClientReplies::ready`]
    /// and waiting for [`Event::Ready`] first.
    ///
    /// # Panics
    /// Asserts capacity is available, the message is a body-ful reply, and (for repairs)
    /// the slot was marked faulty.
    pub fn write_reply(
        &mut self,
        storage: &mut dyn Storage,
        slot: client_sessions::ReplySlot,
        message: Message,
        trigger: WriteTrigger,
    ) {
        assert!(self.ready_sync());

        let header = message
            .header::<message_header::Reply>()
            .unwrap_or_else(|| unreachable!("caller passes a reply message"));
        // There is never any need to write a body-less message, since the header is stored
        // safely in the `client_sessions` trailer:
        assert_ne!(header.size, message_header::SIZE as u32);

        match trigger {
            WriteTrigger::Commit => {
                assert!(!self.checkpoint_waiting);
            }
            WriteTrigger::Repair => {
                assert!(self.faulty.get(slot.index));
            }
        }

        // Resolve any pending reads for this reply. If we don't do this, an earlier-started
        // read can complete with an error and erroneously clobber the faulty bit. For
        // simplicity, resolve the reads synchronously instead of going through next-tick
        // machinery.
        for read_index in 0..READS_MAX {
            let Some(read) = self.reads[read_index].as_mut() else { continue };
            if !read.wanted {
                continue; // Already resolved.
            }
            if read.header.checksum == header.checksum {
                read.wanted = false;
                self.events.push_back(Event::ReadReply {
                    slot: read.slot,
                    header: read.header,
                    destination_replica: read.destination_replica,
                    outcome: ReadOutcome::ResolvedByWrite(message.clone()),
                });
            }
        }

        // Clear the fault *before* the write completes, not after. Otherwise, a replica
        // exiting state sync might mark a reply as faulty, then this clears that bit due to
        // an unrelated write that was already queued.
        self.faulty.unset(slot.index);

        let write_index = self
            .writes
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| unreachable!("capacity checked by ready_sync"));
        self.writes[write_index] = Some(WriteOp { slot, message, trigger });
        self.writes_in_flight.push_back(write_index);

        // If there is already a write to the same slot queued (but not started), replace it:
        // the queued operation is swapped for this one, in place.
        let superseded = self.write_queue.iter().copied().find(|&index| {
            self.writes[index].as_ref().is_some_and(|write| write.slot.index == slot.index)
        });
        if let Some(old_index) = superseded {
            self.writes[old_index] = None;
            for index in &mut self.write_queue {
                if *index == old_index {
                    *index = write_index;
                }
            }
            for index in &mut self.writes_in_flight {
                if *index == old_index {
                    *index = write_index;
                }
            }
        } else {
            self.write_queue.push_back(write_index);
        }

        self.write_reply_next(storage);

        assert!(
            self.writing.get(slot.index)
                || self.write_queue.iter().any(|&index| {
                    self.writes[index].as_ref().is_some_and(|write| write.slot.index == slot.index)
                })
        );
    }

    /// Start queued writes whose slots are not already being written
    /// (upstream: `write_reply_next`).
    fn write_reply_next(&mut self, storage: &mut dyn Storage) {
        while let Some(&head_index) = self.write_queue.front() {
            let Some(head) = self.writes[head_index].as_ref() else {
                unreachable!("queue only holds live writes");
            };
            if self.writing.get(head.slot.index) {
                return;
            }
            self.write_queue.pop_front();

            // Padding must be zero to ensure deterministic storage:
            let size = head.message.size_raw() as usize;
            let size_ceil = sector_ceil(head.message.size_raw());
            assert!(head.message.buffer()[size..size_ceil].iter().all(|&byte| byte == 0));

            self.writing.set(head.slot.index);

            let mut buffer = zeroed_buffer(size_ceil);
            buffer[..size].copy_from_slice(&head.message.buffer()[..size]);
            storage.write_sectors(WriteRequest {
                zone: Zone::ClientReplies,
                offset_in_zone: slot_offset(head.slot),
                buffer,
            });
        }
    }

    /// Waits until all commit-trigger writes are done, then emits
    /// [`Event::CheckpointDone`]. Writes with trigger=[`WriteTrigger::Repair`] may still be
    /// in progress (upstream: `checkpoint()`).
    ///
    /// # Panics
    /// Asserts no checkpoint wait is already registered.
    pub fn checkpoint(&mut self) {
        assert!(!self.checkpoint_waiting);
        self.checkpoint_waiting = true;
        // The event is emitted by the next `poll()` round, standing in for upstream's
        // `on_next_tick` deferral.
    }

    fn writes_executing_by_trigger(&self, trigger: WriteTrigger) -> usize {
        self.writes_in_flight
            .iter()
            .filter_map(|&index| self.writes[index].as_ref())
            .filter(|write| write.trigger == trigger)
            .count()
    }

    /// Drives storage completions, emitting events (upstream: the callback halves of every
    /// async step).
    ///
    /// # Panics
    /// Asserts internal invariants: every storage completion correlates with an outstanding
    /// operation, and completed writes had their slot marked as writing.
    pub fn poll(&mut self, storage: &mut dyn Storage) {
        while let Some(completion) = storage.next_completion() {
            match completion {
                Completion::Read(request) => {
                    let read_index = self
                        .reads_in_flight
                        .pop_front()
                        .unwrap_or_else(|| unreachable!("completion without an outstanding read"));
                    let _ = request;
                    let Some(read) = self.reads[read_index].take() else {
                        unreachable!("in-flight read is live");
                    };

                    // Upstream resolves pending reads against concurrent writes
                    // synchronously; such reads carry `wanted=false` here and produce no
                    // event.
                    if read.wanted {
                        let outcome = Self::classify_read_buffer(&request.buffer, &read.header);
                        self.events.push_back(Event::ReadReply {
                            slot: read.slot,
                            header: read.header,
                            destination_replica: read.destination_replica,
                            outcome,
                        });
                    }
                }
                Completion::Write(request) => {
                    let write_index = self
                        .writes_in_flight
                        .pop_front()
                        .unwrap_or_else(|| unreachable!("completion without an outstanding write"));
                    let _ = request;
                    let Some(write) = self.writes[write_index].take() else {
                        unreachable!("in-flight write is live");
                    };

                    assert!(self.writing.get(write.slot.index));
                    self.writing.unset(write.slot.index);
                    drop(write.message);

                    self.write_reply_next(storage);
                }
            }
        }

        // Fire deferred notifications once the storage queue is drained. Upstream fires
        // these from within individual callbacks; under polling, the observable order of
        // events is unchanged.
        if self.ready_waiting && self.ready_sync() {
            self.ready_waiting = false;
            self.events.push_back(Event::Ready);
        }
        if self.checkpoint_waiting && self.writes_executing_by_trigger(WriteTrigger::Commit) == 0 {
            self.checkpoint_waiting = false;
            self.events.push_back(Event::CheckpointDone);
        }
    }

    fn classify_read_buffer(buffer: &[u8], expected: &message_header::Reply) -> ReadOutcome {
        let mut message = Message::new();
        message.buffer_mut()[..buffer.len()].copy_from_slice(buffer);

        let frame: &[u8; message_header::SIZE] = message
            .frame()
            .try_into()
            .unwrap_or_else(|_| unreachable!("frame is HEADER_SIZE bytes"));
        let base = message_header::Header::from_wire(frame);

        // Checksum failures mean the reply is corrupt (latent sector error, torn write, …):
        let header_valid = base.as_ref().is_some_and(message_header::Header::valid_checksum);
        let body_valid = message.size_raw() >= message_header::SIZE as u32
            && base.as_ref().is_some_and(|header| header.valid_checksum_body(message.body_used()));
        if !(header_valid && body_valid) {
            return ReadOutcome::Corrupt;
        }

        let Some(header) = message.header::<message_header::Reply>() else {
            return ReadOutcome::Corrupt;
        };
        assert_eq!(header.command, expected.command);

        // Possible causes:
        // - The read targets an older reply.
        // - The read targets a newer reply (that we haven't seen/written yet).
        // - The read targets a reply that we wrote, but was misdirected.
        if header.checksum != expected.checksum {
            return ReadOutcome::Unexpected;
        }

        ReadOutcome::Found(message)
    }

    /// Drains accumulated events (upstream: callbacks invoked synchronously).
    pub fn take_events(&mut self) -> Vec<Event> {
        self.events.drain(..).collect()
    }
}

#[cfg(test)]
mod client_replies_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::message_header::TypedHeader;
    use crate::storage::MemoryStorage;

    fn zone_start() -> u64 {
        Zone::ClientReplies.start()
    }

    /// A body-ful, checksum-valid reply message.
    fn make_reply(cluster: u128, client: u128, request: u32, body: &[u8]) -> Message {
        let mut reply = message_header::Reply::default();
        reply.cluster = cluster;
        reply.size = (message_header::SIZE + body.len()) as u32;
        reply.release = crate::multiversion::Release { value: 1 };
        reply.client = client;
        reply.op = u64::from(request);
        reply.commit = u64::from(request);
        reply.timestamp = 1;
        reply.request = request;
        reply.operation = crate::Operation::ROOT;
        reply.checksum_body = reply.calculate_checksum_body(body);
        reply.set_checksum();

        let mut message = Message::new();
        message.set_header(&reply);
        message.buffer_mut()[message_header::SIZE..message_header::SIZE + body.len()]
            .copy_from_slice(body);
        message
    }

    fn entry_for(reply: &Message) -> client_sessions::Entry {
        let header = reply.header::<message_header::Reply>().unwrap();
        client_sessions::Entry { session: 1, header }
    }

    fn slot(index: usize) -> client_sessions::ReplySlot {
        client_sessions::ReplySlot { index }
    }

    fn poll_all(client_replies: &mut ClientReplies, storage: &mut MemoryStorage) -> Vec<Event> {
        client_replies.poll(storage);
        client_replies.take_events()
    }

    #[test]
    fn write_then_read_round_trip_is_durable() {
        let mut storage =
            MemoryStorage::new(Zone::ClientReplies.start() + 4 * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        let reply = make_reply(7, 0xC1, 3, b"hello world");
        cr.write_reply(&mut storage, slot(0), reply, WriteTrigger::Commit);
        assert!(!cr.reply_durable(slot(0)), "write has not completed yet");

        let events = poll_all(&mut cr, &mut storage);
        assert!(
            events.iter().all(|event| matches!(event, Event::ReadReply { .. })),
            "no checkpoint was waited for: {events:?}"
        );
        assert!(cr.reply_durable(slot(0)));

        // Read it back through the same session:
        let session = entry_for(&make_reply(7, 0xC1, 3, b"hello world"));
        cr.read_reply(&mut storage, slot(0), &session, None).expect("read capacity available");
        let events = poll_all(&mut cr, &mut storage);
        match events.first() {
            Some(Event::ReadReply { outcome: ReadOutcome::Found(message), .. }) => {
                assert_eq!(message.body_used(), b"hello world");
            }
            other => panic!("expected Found, got {other:?}"),
        }
    }

    #[test]
    fn latent_sector_error_reads_as_corrupt() {
        let mut storage = MemoryStorage::new(zone_start() + 2 * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        let reply = make_reply(7, 0xC2, 1, b"payload");
        cr.write_reply(&mut storage, slot(1), reply, WriteTrigger::Commit);
        let _ = poll_all(&mut cr, &mut storage);

        // Corrupt the reply's first sector on disk:
        let sector = (zone_start() + slot_offset(slot(1))) / SECTOR_SIZE as u64;
        storage.faulty_sectors.insert(sector);

        let session = entry_for(&make_reply(7, 0xC2, 1, b"payload"));
        cr.read_reply(&mut storage, slot(1), &session, None).unwrap();
        let events = poll_all(&mut cr, &mut storage);
        assert!(matches!(
            events.first(),
            Some(Event::ReadReply { outcome: ReadOutcome::Corrupt, .. })
        ));
    }

    #[test]
    fn unexpected_reply_when_slot_holds_different_checksum() {
        let mut storage = MemoryStorage::new(zone_start() + 3 * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        let written = make_reply(7, 0xC3, 5, b"a");
        cr.write_reply(&mut storage, slot(2), written, WriteTrigger::Commit);
        let _ = poll_all(&mut cr, &mut storage);

        // Session expects a different reply than what the slot holds:
        let stale_session = entry_for(&make_reply(7, 0xC3, 4, b"a"));
        cr.read_reply(&mut storage, slot(2), &stale_session, Some(3)).unwrap();
        let events = poll_all(&mut cr, &mut storage);
        match events.first() {
            Some(Event::ReadReply { destination_replica, outcome, .. }) => {
                assert_eq!(*destination_replica, Some(3));
                assert!(matches!(outcome, ReadOutcome::Unexpected));
            }
            other => panic!("expected ReadReply, got {other:?}"),
        }
    }

    #[test]
    fn queued_writes_to_the_same_slot_coalesce() {
        let mut storage = MemoryStorage::new(zone_start() + 4 * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        // First write starts immediately (slot becomes "writing"):
        let first = make_reply(7, 0xC4, 1, b"one");
        cr.write_reply(&mut storage, slot(3), first, WriteTrigger::Commit);
        assert!(!cr.reply_durable(slot(3)));

        // Second write to the same slot queues behind the first:
        let second = make_reply(7, 0xC4, 2, b"two");
        cr.write_reply(&mut storage, slot(3), second, WriteTrigger::Commit);

        let _ = poll_all(&mut cr, &mut storage); // completes first, starts second
        let _ = poll_all(&mut cr, &mut storage); // completes second

        // The durable reply is the newest one:
        let session = entry_for(&make_reply(7, 0xC4, 2, b"two"));
        cr.read_reply(&mut storage, slot(3), &session, None).unwrap();
        let events = poll_all(&mut cr, &mut storage);
        match events.first() {
            Some(Event::ReadReply { outcome: ReadOutcome::Found(message), .. }) => {
                assert_eq!(message.body_used(), b"two");
            }
            other => panic!("expected newest reply, got {other:?}"),
        }
    }

    #[test]
    fn reads_are_bounded_and_report_busy() {
        let mut storage =
            MemoryStorage::new(zone_start() + (READS_MAX as u64 + 1) * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        // Persist distinct replies into READS_MAX distinct slots:
        for index in 0..READS_MAX {
            let reply = make_reply(7, 0x50 + index as u128, 1, b"x");
            cr.write_reply(&mut storage, slot(index), reply, WriteTrigger::Commit);
        }
        let _ = poll_all(&mut cr, &mut storage);

        for index in 0..READS_MAX {
            let session = entry_for(&make_reply(7, 0x50 + index as u128, 1, b"x"));
            cr.read_reply(&mut storage, slot(index), &session, None)
                .unwrap_or_else(|_| panic!("read {index} should fit"));
        }
        let session = entry_for(&make_reply(7, 999, 1, b"x"));
        assert_eq!(
            cr.read_reply(&mut storage, slot(READS_MAX), &session, None),
            Err(ReadError::Busy)
        );

        let events = poll_all(&mut cr, &mut storage);
        assert_eq!(events.len(), READS_MAX);
    }

    #[test]
    fn ready_fires_once_capacity_frees_up() {
        let mut storage =
            MemoryStorage::new(zone_start() + (WRITES_MAX as u64 + 1) * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        // Exhaust all write capacity with distinct-slot writes:
        for index in 0..WRITES_MAX {
            let reply = make_reply(7, 0x60 + index as u128, 1, b"y");
            cr.write_reply(&mut storage, slot(index), reply, WriteTrigger::Commit);
        }
        assert!(!cr.ready_sync());
        cr.ready();

        let events = poll_all(&mut cr, &mut storage);
        assert!(
            events.iter().any(|event| matches!(event, Event::Ready)),
            "expected Ready among {events:?}"
        );
        assert!(cr.ready_sync());
    }

    #[test]
    fn checkpoint_waits_for_commit_writes_only() {
        let mut storage = MemoryStorage::new(zone_start() + 4 * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        // Mark slot 1 faulty so a repair write is legal:
        cr.faulty.set(1);

        let commit = make_reply(7, 0xC7, 1, b"c");
        let repair = make_reply(7, 0xC8, 1, b"r");
        cr.write_reply(&mut storage, slot(0), commit, WriteTrigger::Commit);
        cr.write_reply(&mut storage, slot(1), repair, WriteTrigger::Repair);
        cr.checkpoint();

        let events = poll_all(&mut cr, &mut storage);
        assert!(
            events.iter().any(|event| matches!(event, Event::CheckpointDone)),
            "checkpoint must complete once commit writes drain: {events:?}"
        );
        // Both writes (including the repair) finished under the synchronous storage:
        assert!(cr.reply_durable(slot(0)));
        assert!(cr.reply_durable(slot(1)));
    }

    #[test]
    fn pending_read_resolved_by_concurrent_write_of_same_reply() {
        let mut storage = MemoryStorage::new(zone_start() + 2 * MESSAGE_SIZE_MAX as u64);
        let mut cr = ClientReplies::new(0);

        // Nothing on disk yet and no write in flight: start the read first.
        let reply = make_reply(7, 0xC9, 9, b"fresh");
        let session = entry_for(&reply);
        cr.read_reply(&mut storage, slot(0), &session, Some(2))
            .unwrap_or_else(|_| panic!("reads available"));

        // Writing the very same reply resolves the pending read synchronously
        // (upstream fires the read callback with `reply = write.message`).
        cr.write_reply(&mut storage, slot(0), reply.clone(), WriteTrigger::Commit);

        let events = poll_all(&mut cr, &mut storage);
        let resolved = events
            .iter()
            .filter(|event| {
                matches!(event, Event::ReadReply { outcome: ReadOutcome::ResolvedByWrite(_), .. })
            })
            .count();
        assert_eq!(resolved, 1, "exactly one ResolvedByWrite event: {events:?}");
        match events.first() {
            Some(Event::ReadReply {
                destination_replica,
                outcome: ReadOutcome::ResolvedByWrite(message),
                ..
            }) => {
                assert_eq!(*destination_replica, Some(2));
                assert_eq!(message.body_used(), b"fresh");
            }
            other => panic!("expected ResolvedByWrite, got {other:?}"),
        }
    }
}
