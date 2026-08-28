//! Client sessions: track the headers of the latest reply for each active client.
//! Serialized/deserialized to/from the trailer on-disk. For the reply bodies, see
//! ClientReplies.
//!
//! Upstream: `src/vsr/client_sessions.zig`.
//!
//! DEVIATION: upstream's `EntriesByClient` is an `AutoHashMapUnmanaged(u128, usize)`; this
//! port uses `std`'s `HashMap`. Iteration order over the map never influences state or
//! output (`evictee` scans the entry array, which has a deterministic order).

use std::collections::HashMap;

use crate::message_header::{self, TypedHeader as _};
use tigerbeetle_core::constants::{CLIENTS_MAX, HEADER_SIZE};

/// `CLIENTS_MAX` as a `usize` (it is a `u32` upstream).
const CLIENTS: usize = CLIENTS_MAX as usize;

/// There is a slot corresponding to every active client (i.e. a total of [`CLIENTS_MAX`] slots).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplySlot {
    pub index: usize,
}

/// Port of `ClientSessions.Entry`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    /// The client's session number as committed to the cluster by a register request.
    pub session: u64,
    /// The header of the reply corresponding to the client's latest committed request.
    pub header: message_header::Reply,
}

impl Entry {
    /// A free entry: all zero, including the header's wire bytes (upstream zeroes the
    /// extern struct; our typed headers' `Default` sets size/protocol/command, so this is
    /// spelled out explicitly).
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            session: 0,
            header: message_header::Reply {
                size: 0,
                protocol: 0,
                command: crate::command::Command::Reserved,
                ..Default::default()
            },
        }
    }
}

/// We found two bugs in the VRR paper relating to the client table:
///
/// 1. a correctness bug, where successive client crashes may cause request numbers to collide
///    for different request payloads, resulting in requests receiving the wrong reply, and
///
/// 2. a liveness bug, where if the client table is updated for request and prepare messages
///    with the client's latest request number, then the client may be locked out from the cluster
///    if the request is ever reordered through a view change.
///
/// We therefore take a different approach with the implementation of our client table, to:
///
/// 1. register client sessions explicitly through the state machine to ensure that
///    session numbers always increase, and
///
/// 2. make a more careful distinction between uncommitted and committed request numbers,
///    considering that uncommitted requests may not survive a view change.
#[derive(Debug, Default)]
pub struct ClientSessions {
    /// Values are indexes into `entries`.
    entries_by_client: HashMap<u128, usize>,
    /// Free entries are zeroed, both in `entries` and on-disk.
    entries: Vec<Entry>,
    entries_present: EntriesPresent,
}

/// Port of `EntriesPresent = stdx.BitSetType(clients_max)`: one bit per possible entry.
#[derive(Clone, Debug, PartialEq, Eq)]
struct EntriesPresent {
    bits: [bool; CLIENTS],
}

impl Default for EntriesPresent {
    fn default() -> Self {
        Self { bits: [false; CLIENTS] }
    }
}

impl EntriesPresent {
    fn count(&self) -> usize {
        self.bits.iter().filter(|&&bit| bit).count()
    }

    fn full(&self) -> bool {
        self.count() == self.bits.len()
    }

    fn empty(&self) -> bool {
        self.count() == 0
    }

    fn is_set(&self, index: usize) -> bool {
        assert!(index < self.bits.len());
        self.bits[index]
    }

    fn set(&mut self, index: usize) {
        assert!(index < self.bits.len());
        self.bits[index] = true;
    }

    fn unset(&mut self, index: usize) {
        assert!(index < self.bits.len());
        self.bits[index] = false;
    }

    fn first_unset(&self) -> Option<usize> {
        self.bits.iter().position(|&bit| !bit)
    }
}

impl ClientSessions {
    /// Size of the buffer needed to encode the client sessions on disk.
    /// (Not rounded up to a sector boundary.)
    ///
    /// First go the vsr headers for the entries (16-byte aligned), then the session values
    /// for the entries (8-byte aligned). For encoding/decoding simplicity, the ClientSessions
    /// always fits in a single block.
    pub const ENCODE_SIZE: usize = {
        const fn align_forward(size: usize, alignment: usize) -> usize {
            size.div_ceil(alignment) * alignment
        }
        let mut size_max = 0usize;

        // First goes the vsr headers for the entries.
        // This takes advantage of the buffer alignment to avoid adding padding for the headers.
        size_max = align_forward(size_max, 16);
        size_max += HEADER_SIZE * CLIENTS;

        // Then follows the session values for the entries.
        size_max = align_forward(size_max, 8);
        size_max += core::mem::size_of::<u64>() * CLIENTS;

        size_max
    };

    /// # Panics
    /// Panics if the encoded size does not fit in a single block minus its header
    /// (upstream asserts at compile time).
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries_by_client: HashMap::with_capacity(CLIENTS),
            entries: vec![Entry::zeroed(); CLIENTS],
            entries_present: EntriesPresent::default(),
        }
    }

    pub fn reset(&mut self) {
        for entry in &mut self.entries {
            *entry = Entry::zeroed();
        }
        self.entries_by_client.clear();
        self.entries_present = EntriesPresent::default();
    }

    /// Port of `encode`: writes headers then sessions into `target`.
    ///
    /// # Panics
    /// Panics unless `target.len() >= ENCODE_SIZE` (upstream asserts).
    pub fn encode(&self, target: &mut [u8]) -> u64 {
        assert!(target.len() >= Self::ENCODE_SIZE);

        let mut size: usize = 0;

        // Write all headers:
        let new_size = size.div_ceil(16) * 16;
        target[size..new_size].fill(0);
        size = new_size;

        for entry in &self.entries {
            target[size..size + HEADER_SIZE].copy_from_slice(&entry.header.to_wire());
            size += HEADER_SIZE;
        }

        // Write all sessions:
        let new_size = size.div_ceil(8) * 8;
        target[size..new_size].fill(0);
        size = new_size;

        for entry in &self.entries {
            target[size..size + 8].copy_from_slice(&entry.session.to_le_bytes());
            size += 8;
        }

        assert_eq!(size, Self::ENCODE_SIZE);
        size as u64
    }

    /// Port of `decode`: loads from an encoding produced by [`Self::encode`] (or disk).
    ///
    /// # Panics
    /// Panics unless the sessions are empty and every free entry is zeroed (upstream asserts),
    /// or if a non-empty entry fails validation.
    pub fn decode(&mut self, source: &[u8]) {
        assert_eq!(self.count(), 0);
        assert!(self.entries_present.empty());
        for entry in &self.entries {
            assert_eq!(entry.session, 0);
            assert!(
                entry.header.to_wire().iter().all(|&byte| byte == 0),
                "free entry must be zeroed"
            );
        }

        assert!(!source.is_empty());
        assert!(source.len() <= Self::ENCODE_SIZE);

        let mut size: usize = 0;
        size = size.div_ceil(16) * 16;
        let headers_offset = size;
        size += CLIENTS * HEADER_SIZE;

        size = size.div_ceil(8) * 8;
        let sessions_offset = size;
        size += CLIENTS * 8;

        assert_eq!(size, Self::ENCODE_SIZE);

        for index in 0..CLIENTS {
            let header_offset = headers_offset + index * HEADER_SIZE;
            let header_bytes: &[u8; HEADER_SIZE] = source
                [header_offset..header_offset + HEADER_SIZE]
                .try_into()
                .unwrap_or_else(|_| unreachable!("HEADER_SIZE bytes"));
            let session = u64::from_le_bytes(
                source[sessions_offset + index * 8..sessions_offset + index * 8 + 8]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("8 bytes")),
            );

            if session == 0 {
                assert!(header_bytes.iter().all(|&byte| byte == 0));
            } else {
                let header = message_header::Reply::from_wire(header_bytes)
                    .unwrap_or_else(|| unreachable!("reply command"));
                assert!(header.valid_checksum());
                assert!(header.commit >= session);

                self.entries_by_client.insert(header.client, index);
                self.entries_present.set(index);
                self.entries[index] = Entry { session, header };
            }
        }

        assert_eq!(
            self.entries_present.count(),
            self.entries_by_client.len(),
            "a client must have at most one entry"
        );
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.entries_by_client.len()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        CLIENTS
    }

    /// # Panics
    /// Panics if the entry found is inconsistent (upstream asserts).
    #[must_use]
    pub fn get(&self, client: u128) -> Option<&Entry> {
        let entry_index = *self.entries_by_client.get(&client)?;
        let entry = &self.entries[entry_index];
        assert_ne!(entry.session, 0);
        assert_eq!(entry.header.command, crate::command::Command::Reply);
        assert_eq!(entry.header.client, client);
        Some(entry)
    }

    /// # Panics
    /// Panics if the entry is absent or internally inconsistent (upstream asserts).
    #[must_use]
    pub fn get_mut(&mut self, client: u128) -> Option<&mut Entry> {
        let entry_index = *self.entries_by_client.get(&client)?;
        let entry = &mut self.entries[entry_index];
        assert_ne!(entry.session, 0);
        assert_eq!(entry.header.command, crate::command::Command::Reply);
        assert_eq!(entry.header.client, client);
        Some(entry)
    }

    #[must_use]
    pub fn get_slot_for_client(&self, client: u128) -> Option<ReplySlot> {
        let index = *self.entries_by_client.get(&client)?;
        Some(ReplySlot { index })
    }

    #[must_use]
    pub fn get_slot_for_header(&self, header: &message_header::Reply) -> Option<ReplySlot> {
        if let Some(entry_index) = self.entries_by_client.get(&header.client) {
            let entry = &self.entries[*entry_index];
            if entry.header.checksum == header.checksum {
                return Some(ReplySlot { index: *entry_index });
            }
        }
        None
    }

    /// If the entry is from a newly-registered client, the caller is responsible for ensuring
    /// the ClientSessions has available capacity.
    ///
    /// # Panics
    /// Panics if the existing entry is inconsistent with the new reply (upstream asserts).
    pub fn put(&mut self, session: u64, header: &message_header::Reply) -> ReplySlot {
        assert_ne!(session, 0);
        assert_eq!(header.command, crate::command::Command::Reply);
        let client = header.client;

        if let Some(entry_index) = self.entries_by_client.get_mut(&client) {
            let entry_index = *entry_index;
            assert!(self.entries_present.is_set(entry_index));

            let existing = &mut self.entries[entry_index];
            assert_eq!(existing.session, session);
            assert_eq!(existing.header.cluster, header.cluster);
            assert_eq!(existing.header.client, header.client);
            assert!(existing.header.commit < header.commit);

            existing.header = *header;
            return ReplySlot { index: entry_index };
        }

        let entry_index = self
            .entries_present
            .first_unset()
            .unwrap_or_else(|| unreachable!("caller ensures capacity"));
        self.entries_present.set(entry_index);

        let e = &mut self.entries[entry_index];
        assert_eq!(e.session, 0);

        self.entries_by_client.insert(client, entry_index);
        e.session = session;
        e.header = *header;
        ReplySlot { index: entry_index }
    }

    /// For correctness, it's critical that all replicas evict deterministically:
    /// We cannot depend on `HashMap.capacity()` since `HashMap.ensureTotalCapacity()` may
    /// change across versions of the Zig std lib. We therefore rely on
    /// `constants.clients_max`, which must be the same across all replicas, and must not
    /// change after initializing a cluster.
    /// We also do not depend on `HashMap.valueIterator()` being deterministic here. However,
    /// we do require that all entries have different commit numbers and are iterated.
    /// This ensures that we will always pick the entry with the oldest commit number.
    /// We also check that a client has only one entry in the hash map (or it's buggy).
    ///
    /// # Panics
    /// Panics unless the table is full and entries are mutually consistent (upstream asserts).
    #[must_use]
    pub fn evictee(&self) -> u128 {
        assert!(self.entries_present.full());
        assert_eq!(self.count(), CLIENTS);

        let mut evictee: Option<&message_header::Reply> = None;
        let mut iterated: usize = 0;
        for entry in self.iterator() {
            assert_eq!(entry.header.command, crate::command::Command::Reply);
            assert_eq!(entry.header.op, entry.header.commit);
            assert!(entry.header.commit >= entry.session);

            match evictee {
                Some(evictee_reply) => {
                    assert_ne!(entry.header.client, evictee_reply.client);
                    assert_ne!(entry.header.commit, evictee_reply.commit);

                    if entry.header.commit < evictee_reply.commit {
                        evictee = Some(&entry.header);
                    }
                }
                None => evictee = Some(&entry.header),
            }
            iterated += 1;
        }
        assert_eq!(iterated, CLIENTS);

        evictee.unwrap_or_else(|| unreachable!("table is full")).client
    }

    /// # Panics
    /// Panics if the client has no entry (upstream asserts via `.?`).
    pub fn remove(&mut self, client: u128) {
        let entry_index = self
            .entries_by_client
            .remove(&client)
            .unwrap_or_else(|| unreachable!("client has an entry"));

        assert!(self.entries_present.is_set(entry_index));
        self.entries_present.unset(entry_index);

        assert_eq!(self.entries[entry_index].header.client, client);
        self.entries[entry_index] = Entry::zeroed();

        assert!(!self.entries_by_client.contains_key(&client));
    }

    pub fn iterator(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|entry| entry.session != 0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{CLIENTS, ClientSessions, Entry};
    use crate::command::Command;
    use crate::message_header::{self, TypedHeader as _};
    use crate::multiversion::Release;
    use tigerbeetle_core::constants::VSR_OPERATIONS_RESERVED;

    fn reply(client: u128, commit: u64) -> message_header::Reply {
        let mut header = message_header::Reply {
            client,
            op: commit,
            commit,
            timestamp: commit + 1000,
            request: 42,
            operation: crate::Operation(VSR_OPERATIONS_RESERVED + 1),
            release: Release::MINIMUM,
            command: Command::Reply,
            ..Default::default()
        };
        header.set_checksum();
        header
    }

    #[test]
    fn encode_size_fits_single_block() {
        // 16-byte-aligned headers followed by 8-byte-aligned sessions:
        assert_eq!(
            ClientSessions::ENCODE_SIZE,
            CLIENTS * message_header::SIZE + CLIENTS * core::mem::size_of::<u64>()
        );
        // (The fits-in-a-single-block invariant is asserted in `new()` and pinned by
        // `superblock::CLIENT_SESSIONS_ENCODE_SIZE`.)
    }

    #[test]
    fn put_get_remove_round_trip() {
        let mut sessions = ClientSessions::new();
        assert_eq!(sessions.count(), 0);
        assert_eq!(sessions.capacity(), CLIENTS);

        let slot = sessions.put(7, &reply(1000, 9));
        assert_eq!(slot.index, 0);
        assert_eq!(sessions.count(), 1);

        let entry = sessions.get(1000).expect("client entry");
        assert_eq!(entry.session, 7);
        assert_eq!(entry.header.commit, 9);
        assert_eq!(sessions.get_slot_for_client(1000), Some(super::ReplySlot { index: 0 }));
        assert_eq!(sessions.get_slot_for_client(2000), None);

        // Same client + session with a newer commit updates in place:
        let slot = sessions.put(7, &reply(1000, 10));
        assert_eq!(slot.index, 0);
        assert_eq!(sessions.count(), 1);
        assert_eq!(sessions.get(1000).expect("client entry").header.commit, 10);
        assert_eq!(
            sessions.get_slot_for_header(&reply(1000, 10)),
            Some(super::ReplySlot { index: 0 })
        );
        assert_eq!(sessions.get_slot_for_header(&reply(1000, 9)), None);

        // A second client takes the next slot:
        let slot = sessions.put(8, &reply(2000, 11));
        assert_eq!(slot.index, 1);

        sessions.remove(1000);
        assert_eq!(sessions.count(), 1);
        assert_eq!(sessions.get(1000), None);

        // The freed slot is reused:
        let slot = sessions.put(9, &reply(3000, 12));
        assert_eq!(slot.index, 0);
    }

    #[test]
    fn encode_decode_round_trip() {
        let mut sessions = ClientSessions::new();
        sessions.put(7, &reply(1000, 9));
        sessions.put(8, &reply(2000, 10));

        let mut wire = vec![0u8; ClientSessions::ENCODE_SIZE];
        assert_eq!(
            usize::try_from(sessions.encode(&mut wire)).unwrap_or(usize::MAX),
            ClientSessions::ENCODE_SIZE
        );

        let mut decoded = ClientSessions::new();
        decoded.decode(&wire);
        assert_eq!(decoded.count(), 2);
        assert_eq!(decoded.get(1000), Some(&Entry { session: 7, header: reply(1000, 9) }));
        assert_eq!(decoded.get(2000), Some(&Entry { session: 8, header: reply(2000, 10) }));
    }

    #[test]
    fn evictee_picks_oldest_commit_when_full() {
        let mut sessions = ClientSessions::new();
        for (index, _) in (0..CLIENTS).enumerate() {
            let client = u128::try_from(index).unwrap_or(u128::MAX) + 10;
            let commit = u64::try_from(index).unwrap_or(u64::MAX) + 100;
            sessions.put(commit, &reply(client, commit));
        }
        assert_eq!(sessions.count(), CLIENTS);

        // Client 10 has the oldest commit:
        assert_eq!(sessions.evictee(), 10);

        // Refresh client 10 (keeping its session number) so that client 11 becomes the evictee:
        sessions.put(100, &reply(10, 500));
        assert_eq!(sessions.evictee(), 11);
    }
}
