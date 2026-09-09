//! Port of `src/vsr/replica_format.zig`: initialize a replica's data file.
//!
//! `format()` writes the minimum data needed for a fresh replica: the journal (one prepare per
//! slot + the contiguous headers block) is written first, then `SuperBlock.format()` lays down
//! the superblock copies, the storage is flushed, and finally a bitset check verifies that
//! *exactly* the expected sectors were touched (so a forgotten `wal_headers` sector or an extra
//! `grid` write is caught).
//!
//! DEVIATION (async model): upstream completes each write through an IO event loop driven by
//! `storage.run()`; here writes are issued synchronously and drained through
//! `Storage::next_completion()` (see `src/storage.zig` port notes). `format()` additionally
//! returns the formatted `SuperBlock` so the caller can print its state (upstream returns void).

#![allow(clippy::cast_possible_truncation)] // data-file offsets/lengths are 64-bit-only

use crate::Zone;
use crate::message_header::{self, Prepare, TypedHeader as _};
use crate::storage::{Completion, Storage, WriteRequest, zeroed_buffer};
use crate::superblock::{DATA_FILE_SIZE_MIN, Event, FormatOptions, SuperBlock};
use tigerbeetle_core::constants::{
    JOURNAL_SIZE_HEADERS, JOURNAL_SIZE_PREPARES, JOURNAL_SLOT_COUNT, MESSAGE_SIZE_MAX, SECTOR_SIZE,
};

/// The number of writes `format()` issues: one prepare per journal slot plus one headers block.
/// (Upstream: `replica_format.writes_max`.)
pub const WRITES_MAX: usize = JOURNAL_SLOT_COUNT as usize + 1;

/// `DATA_FILE_SIZE_MIN` expressed as a sector count (upstream: `@divExact(data_file_size_min,
/// sector_size)` — the bitset capacity).
const SECTORS_TOTAL: usize = DATA_FILE_SIZE_MIN / SECTOR_SIZE;

const _: () = assert!(message_header::SIZE < SECTOR_SIZE);

/// The on-disk prepare header for `slot` of `cluster`: the root prepare for slot 0, a reserved
/// (blank) prepare otherwise (upstream: `replica_format.slot_header`).
///
/// # Panics
/// Panics if `slot` does not fit in the journal (upstream asserts).
#[must_use]
pub fn slot_header(cluster: u128, slot: u64) -> Prepare {
    assert!(slot < u64::from(JOURNAL_SLOT_COUNT));
    assert!(slot * (message_header::SIZE as u64) < JOURNAL_SIZE_HEADERS as u64);
    assert!(slot * u64::from(MESSAGE_SIZE_MAX) < JOURNAL_SIZE_PREPARES as u64);

    if slot == 0 { Prepare::root(cluster) } else { Prepare::reserve(cluster, slot) }
}

/// Initialize a TigerBeetle replica data file (upstream: `vsr.replica_format.format`).
///
/// REQUIRES the storage to be exactly [`DATA_FILE_SIZE_MIN`] bytes (upstream formats a file of
/// that size); the caller holds it freshly created/truncated.
///
/// # Panics
/// Panics if the storage is not the minimum data-file size, a WAL write leaves a sector
/// untracked or unverified, the superblock reports an error, or `format` fails (upstream
/// asserts/`vsr.fatal`).
#[must_use]
pub fn format(storage: &mut dyn Storage, options: FormatOptions) -> SuperBlock {
    assert_eq!(
        storage.size(),
        DATA_FILE_SIZE_MIN as u64,
        "format requires a data file of exactly data_file_size_min bytes"
    );

    let mut sectors_written = vec![false; SECTORS_TOTAL];
    let mut writes_pending = WRITES_MAX;

    // The logical offset *within* the zone. Even though the prepare zone follows the redundant
    // header zone, write the prepares first. This allows the test Storage to check the
    // invariant "never write the redundant header before the prepare".
    for slot in 0..JOURNAL_SLOT_COUNT {
        // Direct I/O requires the buffer to be sector-aligned. The prepares are spread out with
        // zeros in-between, so each slot gets its own sector-sized buffer.
        let mut buffer = zeroed_buffer(SECTOR_SIZE);
        let header = slot_header(options.cluster, u64::from(slot));
        buffer[..message_header::SIZE].copy_from_slice(&header.to_wire());

        storage.write_sectors(WriteRequest {
            zone: Zone::WalPrepares,
            offset_in_zone: u64::from(slot) * u64::from(MESSAGE_SIZE_MAX),
            buffer,
        });
    }

    // The headers zone is contiguous, so a single buffer covers it (sector-ceiled for the
    // padding, which is zeroed to produce identical checksums of an empty data file).
    let headers_len = sector_ceil(JOURNAL_SIZE_HEADERS);
    let mut headers_buffer = zeroed_buffer(headers_len);
    for slot in 0..JOURNAL_SLOT_COUNT {
        let header = slot_header(options.cluster, u64::from(slot));
        let start = (slot as usize) * message_header::SIZE;
        headers_buffer[start..start + message_header::SIZE].copy_from_slice(&header.to_wire());
    }
    storage.write_sectors(WriteRequest {
        zone: Zone::WalHeaders,
        offset_in_zone: 0,
        buffer: headers_buffer,
    });

    while writes_pending > 0 {
        let Some(completion) = storage.next_completion() else {
            unreachable!("a WAL format write is still pending");
        };
        let Completion::Write(request) = completion else {
            unreachable!("no reads are issued during the WAL format phase");
        };
        writes_pending -= 1;
        mark_sectors_written(
            &mut sectors_written,
            request.zone,
            request.offset_in_zone,
            request.buffer.len(),
        );
    }

    let mut superblock = SuperBlock::new(storage.size());
    superblock.format(storage, options);
    superblock.poll(storage);
    let events = superblock.take_events();
    assert!(events.contains(&Event::FormatDone), "format did not complete: {events:?}");

    storage.flush_sectors();

    verify_writes(&sectors_written);
    superblock
}

/// Marks the sectors covered by a completed write in the bitset (upstream:
/// `ReplicaFormat.write_sectors_callback`).
fn mark_sectors_written(
    sectors_written: &mut [bool],
    zone: Zone,
    offset_in_zone: u64,
    buffer_len: usize,
) {
    let sector_offset = (zone.offset(offset_in_zone) / SECTOR_SIZE as u64) as usize;
    let sector_count = buffer_len / SECTOR_SIZE;
    for sector in &mut sectors_written[sector_offset..sector_offset + sector_count] {
        assert!(!*sector, "the same sector must not be formatted twice");
        *sector = true;
    }
}

/// Verifies that exactly the minimum sectors were written (upstream:
/// `ReplicaFormat.verify_writes`): every `wal_headers` sector, the first sector in every
/// `message_size_max` of the prepare zone, and nothing else. The superblock zone is covered by
/// `SuperBlock.format`'s own read-back validation, so it is excluded here too.
fn verify_writes(sectors_written: &[bool]) {
    assert!(sectors_written.iter().filter(|written| **written).count() > 0);
    assert_eq!(sectors_written.len(), SECTORS_TOTAL);

    for (sector, written) in sectors_written.iter().copied().enumerate() {
        let sector_start = sector as u64 * SECTOR_SIZE as u64;

        let zone = zone_at(sector_start);
        match zone {
            // Every sector in the wal_headers zone has been written:
            Zone::WalHeaders => assert!(written, "wal_headers sector {sector} not written"),

            // The first sector in every message_size_max has been written:
            Zone::WalPrepares => {
                let within =
                    (sector_start - Zone::WalPrepares.start()) % u64::from(MESSAGE_SIZE_MAX);
                if within == 0 {
                    assert!(written, "prepare sector {sector} not written");
                } else {
                    assert!(!written, "prepare sector {sector} unexpectedly written");
                }
            }

            // Nothing else has been written:
            _ => assert!(!written, "sector {sector} ({zone:?}) unexpectedly written"),
        }
    }
}

/// The fixed-size zone containing the given absolute file offset (upstream's values-array walk in
/// `verify_writes`; the open-ended grid zone is past `DATA_FILE_SIZE_MIN` and never reached).
fn zone_at(offset: u64) -> Zone {
    for zone in [
        Zone::Superblock,
        Zone::WalHeaders,
        Zone::WalPrepares,
        Zone::ClientReplies,
        Zone::GridPadding,
    ] {
        let start = zone.start();
        let Some(size) = zone.size() else { continue };
        if offset >= start && offset < start + size {
            return zone;
        }
    }
    unreachable!("sector offset {offset} is beyond the formatted data file");
}

/// Round `size` up to a whole number of `SECTOR_SIZE` bytes (upstream `vsr.sector_ceil`).
fn sector_ceil(size: usize) -> usize {
    size.div_ceil(SECTOR_SIZE) * SECTOR_SIZE
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{format, slot_header};
    use crate::command::Command;
    use crate::message_header::{self, Prepare, TypedHeader as _};
    use crate::storage::{Completion, MemoryStorage, ReadRequest, Storage, zeroed_buffer};
    use crate::superblock::{DATA_FILE_SIZE_MIN, FormatOptions, SuperBlock};
    use crate::{Operation, Zone};
    use tigerbeetle_core::constants::{
        JOURNAL_SIZE_HEADERS, JOURNAL_SLOT_COUNT, MESSAGE_SIZE_MAX, SECTOR_SIZE,
    };

    fn format_options(replica: u8) -> FormatOptions {
        FormatOptions {
            cluster: 0,
            release: crate::multiversion::Release::MINIMUM,
            replica,
            replica_count: 1,
            view: None,
        }
    }

    /// Port of upstream `replica_format_zig slot_header` test.
    #[test]
    fn slot_header_matches_upstream_test() {
        for slot in 0..JOURNAL_SLOT_COUNT {
            let header = slot_header(0, u64::from(slot));

            assert!(header.valid_checksum());
            assert!(header.valid_checksum_body(&[]));
            assert!(header.invalid().is_none());
            assert_eq!(header.cluster, 0);
            assert_eq!(header.op, u64::from(slot));
            assert_eq!(header.size, message_header::SIZE as u32);
            assert_eq!(header.command, Command::Prepare);
            if slot == 0 {
                assert_eq!(header.operation, Operation::ROOT);
            } else {
                assert_eq!(header.operation, Operation::RESERVED);
            }
        }
    }

    /// Reads `len` bytes from `storage` at `offset_in_zone` within `zone` and returns the
    /// (zero-padded-to-sector) filled buffer.
    fn read_back(
        storage: &mut dyn Storage,
        zone: Zone,
        offset_in_zone: u64,
        len: usize,
    ) -> Vec<u8> {
        let buffer = zeroed_buffer(super::sector_ceil(len));
        storage.read_sectors(ReadRequest { zone, offset_in_zone, buffer });
        let Completion::Read(request) =
            storage.next_completion().expect("read must complete immediately")
        else {
            unreachable!("read was expected");
        };
        request.buffer
    }

    fn prepare_at(bytes: &[u8], index: usize) -> Prepare {
        let start = index * message_header::SIZE;
        Prepare::from_wire(bytes[start..start + message_header::SIZE].try_into().unwrap())
            .expect("prepare must decode")
    }

    /// Every sector of the given range must still hold the 0xAA poison (never written).
    fn assert_poisoned(storage: &mut dyn Storage, zone: Zone) {
        if zone.size().unwrap_or(0) == 0 {
            return;
        }
        let buffer = read_back(storage, zone, 0, SECTOR_SIZE);
        assert!(buffer.iter().all(|b| *b == 0xAA), "{zone:?} must never have been written");
    }

    /// Upstream `replica_format_zig "format"` test: over a poisoned storage only the minimum
    /// sectors are written, and the WAL content is exactly the root/reserved prepares.
    #[test]
    fn format_writes_minimum_sectors() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        storage.poison_image();

        let sb = format(&mut storage, format_options(0));
        assert_eq!(sb.working().sequence, 1);
        assert_eq!(sb.working().cluster, 0);
        assert_eq!(sb.working().vsr_state.commit_max, 0);
        assert!(storage.next_completion().is_none(), "all completions must have been drained");

        // WAL headers: every slot decodes to a valid checksummed prepare.
        let headers = read_back(&mut storage, Zone::WalHeaders, 0, JOURNAL_SIZE_HEADERS);
        for slot in 0..JOURNAL_SLOT_COUNT {
            let header = prepare_at(&headers, slot as usize);
            assert!(header.valid_checksum());
            assert!(header.valid_checksum_body(&[]));
            assert_eq!(header.op, u64::from(slot));
            if slot == 0 {
                assert_eq!(header.operation, Operation::ROOT);
            } else {
                assert_eq!(header.operation, Operation::RESERVED);
            }
        }

        // WAL prepares: the first sector of every message_size_max is the same prepare as the
        // header block; the remainder of the block is the poison (never written).
        for slot in 0..JOURNAL_SLOT_COUNT {
            let slot = u64::from(slot);
            let block = read_back(
                &mut storage,
                Zone::WalPrepares,
                slot * u64::from(MESSAGE_SIZE_MAX),
                MESSAGE_SIZE_MAX as usize,
            );
            assert_eq!(
                prepare_at(&block, 0),
                prepare_at(&headers, slot as usize),
                "prepare block {slot} must match the headers block"
            );
            if MESSAGE_SIZE_MAX as usize > SECTOR_SIZE {
                assert!(block[SECTOR_SIZE..].iter().all(|b| *b == 0xAA));
            }
        }

        // Nothing outside the journal and superblock zones was written.
        assert_poisoned(&mut storage, Zone::ClientReplies);
        assert_poisoned(&mut storage, Zone::GridPadding);
    }

    /// Format a real data file and reopen it — the strongest end-to-end check that the file's
    /// bytes (superblock + journal) are internally consistent.
    #[test]
    fn format_round_trips_through_file_storage() {
        let path = std::env::temp_dir()
            .join(format!("tigerbeetle-rs-format-{}-data.tigerbeetle", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut storage =
            crate::storage::FileStorage::open_format(&path, DATA_FILE_SIZE_MIN as u64).unwrap();
        let sb = format(&mut storage, format_options(0));
        assert_eq!(sb.working().sequence, 1);
        assert_eq!(sb.working().cluster, 0);
        assert_eq!(sb.working().vsr_state.replica_id, crate::root_members(0)[0]);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            DATA_FILE_SIZE_MIN as u64,
            "the data file must be exactly data_file_size_min bytes"
        );
        drop(storage);

        // A fresh instance reopens the file (validating every superblock copy's checksums).
        let mut reopened = crate::storage::FileStorage::open(&path, 0).unwrap();
        let mut superblock = SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        superblock.open(&mut reopened);
        superblock.poll(&mut reopened);
        assert!(superblock.opened());
        assert_eq!(superblock.working().sequence, 1);
        assert_eq!(superblock.working().cluster, 0);
        assert_eq!(superblock.replica_index(), Some(0));
        assert_eq!(superblock.working().vsr_state.commit_max, 0);

        let _ = std::fs::remove_file(&path);
    }
}
