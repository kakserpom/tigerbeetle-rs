//! Sector-level reads/writes against the data file, tagged by [`Zone`].
//!
//! Upstream: `src/storage.zig` (`StorageType(IO)`).
//!
//! DEVIATION (async model): upstream is callback-based over a pluggable IO event loop
//! (`io.read/io.write` + completion pointers); this port exposes the same operations as an
//! explicit submission/completion queue (`read_sectors`/`write_sectors` +
//! [`Storage::next_completion`]) so that callers are written against the final hand-rolled
//! reactor shape while the interim implementation runs synchronously. O_DIRECT and aligned
//! buffers are deferred with it (std pread-style I/O needs neither).
//!
//! The latent-sector-error recovery in the read path is ported 1:1: reads are subdivided in
//! a recursive binary search (`target_max` halving) and unreadable logical sectors are
//! zeroed, to be surfaced later as checksum failures.

#![allow(clippy::cast_possible_truncation)] // data-file offsets/lengths are 64-bit-only, as upstream

use std::collections::HashSet;

use crate::Zone;
use tigerbeetle_core::constants::SECTOR_SIZE;

/// A pending read issued through [`Storage::read_sectors`].
///
/// The buffer is moved in at submission and returned (filled) in the completion.
/// Upstream: `Storage.Read`.
#[derive(Debug)]
pub struct ReadRequest {
    pub zone: Zone,
    /// Offset relative to the start of the zone (upstream: `offset_in_zone`).
    pub offset_in_zone: u64,
    /// Length must be a positive multiple of [`SECTOR_SIZE`] (upstream asserts via
    /// `Zone.verify_iop`). Moved back out through the completion.
    pub buffer: Vec<u8>,
}

/// A pending write issued through [`Storage::write_sectors`].
/// Upstream: `Storage.Write`.
#[derive(Debug)]
pub struct WriteRequest {
    pub zone: Zone,
    pub offset_in_zone: u64,
    /// Length must be a positive multiple of [`SECTOR_SIZE`] (upstream asserts via
    /// `Zone.verify_iop`). Moved back out through the completion.
    pub buffer: Vec<u8>,
}

/// A finished operation (upstream: invoking the operation's callback).
#[derive(Debug)]
pub enum Completion {
    /// The buffer has been fully populated. Unreadable sectors are zeroed.
    Read(ReadRequest),
    /// The buffer has been fully written.
    Write(WriteRequest),
}

impl Completion {
    #[must_use]
    pub fn zone(&self) -> Zone {
        match self {
            Self::Read(request) => request.zone,
            Self::Write(request) => request.zone,
        }
    }
}

/// Port of `StorageType(IO)`'s public surface.
pub trait Storage {
    /// Size of the data file (upstream: `storage.size`, maintained by the superblock).
    fn size(&self) -> u64;

    /// Issue an asynchronous sector read. The completion (with the filled buffer) is
    /// delivered through [`Self::next_completion`].
    ///
    /// # Panics
    /// Panics if the request fails zone verification or targets `GridPadding`
    /// (upstream asserts).
    fn read_sectors(&mut self, request: ReadRequest);

    /// Issue an asynchronous sector write. The completion (with the consumed buffer) is
    /// delivered through [`Self::next_completion`].
    ///
    /// # Panics
    /// Panics under the same conditions as [`Self::read_sectors`]; a failed write
    /// is fatal (upstream panics/vsr.fatal).
    fn write_sectors(&mut self, request: WriteRequest);

    /// Dequeue one completed operation, if any.
    /// (Upstream: the callback invocation itself.)
    fn next_completion(&mut self) -> Option<Completion>;
}

/// A zeroed `len`-byte buffer whose length is a whole number of logical sectors.
///
/// # Panics
/// Asserts `len` is a multiple of [`SECTOR_SIZE`].
#[must_use]
pub fn zeroed_buffer(len: usize) -> Vec<u8> {
    assert!(len.is_multiple_of(SECTOR_SIZE));
    vec![0u8; len]
}

/// Verify a request the way upstream's `read_sectors`/`write_sectors` do, returning the
/// absolute offset in the data file.
// DEVIATION (interim): upstream calls `Zone.verify_iop()`, which also asserts that the
// buffer's *memory* starts on a sector boundary — an O_DIRECT requirement satisfied by
// its message-pool buffers. Our interim `std::fs`/heap-backed I/O uses ordinary `Vec`
// buffers whose address cannot be aligned safely, so the pointer check is omitted here;
// every other check matches `Zone::verify_iop`. TODO(port): restore the alignment check
// together with O_DIRECT support.
fn verify_request(zone: Zone, buffer: &[u8], offset_in_zone: u64) -> u64 {
    use tigerbeetle_core::constants::BLOCK_SIZE;

    if let Some(zone_size) = zone.size() {
        assert!(u64::try_from(buffer.len()).is_ok_and(|len| offset_in_zone + len <= zone_size));
    }
    assert!(buffer.len().is_multiple_of(SECTOR_SIZE));
    assert!(!buffer.is_empty());
    let offset_in_storage = zone.offset(offset_in_zone);
    assert!(offset_in_storage.is_multiple_of(SECTOR_SIZE as u64));
    if zone == Zone::Grid {
        assert!(offset_in_storage.is_multiple_of(BLOCK_SIZE as u64));
    }
    assert_ne!(zone, Zone::GridPadding, "padding is never touched");
    offset_in_storage
}

/// Read-progress state shared by all implementations (port of `Storage.Read`'s
/// `target()`/LSE bookkeeping).
#[derive(Debug)]
struct ReadState {
    /// Remaining bytes to fill, counted from the start of the request
    /// (upstream: `read.buffer` shrinking from the front).
    remaining_len: usize,
    /// Absolute file offset of the next byte to read (upstream: `read.offset`).
    offset: u64,
    /// The maximum amount of bytes to read per syscall. We use this to subdivide
    /// troublesome reads into smaller reads to work around latent sector errors (LSEs)
    /// (upstream: `target_max`).
    target_max: usize,
}

impl ReadState {
    fn new(len: usize) -> Self {
        Self { remaining_len: len, offset: 0, target_max: len }
    }

    /// Returns how much to read in this step, capped by `target_max`; if the previous read
    /// was a partial read of physical sectors (e.g. 512 bytes) less than our logical sector
    /// size (e.g. 4 KiB), the amount is capped further to get back onto a logical sector
    /// boundary (upstream: `Read.target`).
    fn target_len(&self) -> usize {
        // A worked example of a partial read that leaves the rest of the buffer unaligned:
        // This could happen for non-Advanced Format disks with a physical
        // sector of 512 bytes.
        //
        // We want to read 8 KiB: len=8192, and then experience a partial read of only 512
        // bytes: remaining_len=7680. Now `remaining_len % SECTOR_SIZE == 3584` is the part
        // of the partially-read physical sector still missing, so `target_max` (4 KiB) is
        // reduced by exactly that to get back onto the boundary.
        let mut max = self.target_max;

        let partial_sector_read_remainder = self.remaining_len % SECTOR_SIZE;
        if partial_sector_read_remainder != 0 {
            let partial_sector_read = SECTOR_SIZE - partial_sector_read_remainder;
            max -= partial_sector_read;
        }

        self.remaining_len.min(max)
    }

    /// Ceiling division of `len` into sectors (upstream inline arithmetic).
    fn sectors(len: usize) -> usize {
        (len - 1) / SECTOR_SIZE + 1
    }

    /// Halve `target_max` for the recursive binary search over failing sectors
    /// (upstream: `(@divFloor(target_sectors - 1, 2) + 1) * constants.sector_size`).
    fn halve_target_max(&mut self) {
        let target_sectors = Self::sectors(self.target_len());
        self.target_max = ((target_sectors - 1) / 2 + 1) * SECTOR_SIZE;
        assert!(self.target_max >= SECTOR_SIZE);
    }

    /// AIMD: if our target was limited to a single sector, perhaps because of a latent
    /// sector error, then increase `target_max` now that we have read successfully and
    /// hopefully cleared the faulty zone (upstream: `target_max += sector_size`).
    fn aimd_restore(&mut self) {
        if self.target_max == SECTOR_SIZE {
            self.target_max += SECTOR_SIZE;
        }
    }

    fn advance(&mut self, bytes: usize) {
        self.offset += bytes as u64;
        self.remaining_len -= bytes;
    }

    /// Zero-fill the unreadable logical sector at the current position and move past it
    /// (upstream: `@memset(target, 0)` + retry).
    fn zero_current_sector(&mut self, buffer: &mut [u8], base_offset: u64) {
        let target_len = self.target_len();
        assert!(target_len > 0);
        let start = (self.offset - base_offset) as usize;
        buffer[start..start + target_len].fill(0);
        self.advance(target_len);
    }

    /// Zero-fill everything from the current position to the end of the request and finish
    /// (upstream: `@memset(read.buffer, 0)` on a zero-byte read).
    fn zero_rest(&mut self, buffer: &mut [u8], base_offset: u64) {
        let start = (self.offset - base_offset) as usize;
        buffer[start..].fill(0);
        self.remaining_len = 0;
    }
}

/// Error mirroring the subset of `IO.ReadError` the read state machine distinguishes.
#[derive(Debug)]
enum FileReadError {
    /// `error.InputOutput`: a latent sector error.
    InputOutput,
    /// Short read because we (thought we could) read beyond the end of the file descriptor.
    EndOfFile,
}

/// Fill `buffer` completely from `offset`, retrying partial reads.
fn read_step(file: &std::fs::File, buffer: &mut [u8], offset: u64) -> Result<(), FileReadError> {
    use std::os::unix::fs::FileExt;

    let mut filled = 0usize;
    while filled < buffer.len() {
        match file.read_at(&mut buffer[filled..], offset + filled as u64) {
            Ok(0) => return Err(FileReadError::EndOfFile),
            Ok(bytes) => filled += bytes,
            Err(err) => match err.raw_os_error() {
                // 5 is EIO on every platform we support (Linux and macOS):
                Some(5) => return Err(FileReadError::InputOutput),
                _ => panic!("impossible read: offset={offset} error={err:?}"),
            },
        }
    }
    Ok(())
}

/// Drive the shared read state machine synchronously to completion
/// (upstream: `start_read`/`on_read` recursion across async steps).
fn drive_read(
    request: &mut ReadRequest,
    base_offset: u64,
    mut read_one: impl FnMut(&mut [u8], u64) -> Result<(), FileReadError>,
) {
    let mut state = ReadState::new(request.buffer.len());
    state.offset = base_offset;

    while state.remaining_len > 0 {
        let target_len = state.target_len();
        assert!(target_len > 0);
        assert_eq!(state.offset % SECTOR_SIZE as u64, 0);

        let start = (state.offset - base_offset) as usize;
        match read_one(&mut request.buffer[start..start + target_len], state.offset) {
            Ok(()) => {
                state.advance(target_len);
                state.aimd_restore();
            }
            Err(FileReadError::InputOutput) => {
                // The disk was unable to read some sectors (an internal CRC or hardware
                // failure): We may also have already experienced a partial unaligned read,
                // so we cannot expect `target_len` to be an exact logical sector multiple.
                if target_len > SECTOR_SIZE {
                    // Divide the buffer in half and try to read each half separately:
                    // This creates a recursive binary search for the sector(s)
                    // causing the error.
                    state.halve_target_max();
                    // Retry the same offset with a smaller window (bytes_read = 0).
                } else {
                    // We tried to read at (or less than) logical sector granularity and
                    // failed: zero this logical sector which can't be read. We will treat
                    // these EIO errors the same as a checksum failure.
                    state.zero_current_sector(&mut request.buffer, base_offset);
                }
            }
            Err(FileReadError::EndOfFile) => {
                // We tried to read more than there really is available to read. Some possible
                // causes (upstream): truncated/corrupted inode size, a trailing grid block
                // smaller than block_size, or a peer requesting an address beyond our file.
                // Zero-fill the remainder and complete.
                state.zero_rest(&mut request.buffer, base_offset);
            }
        }
    }
}

/// In-process data file backed by `std::fs`.
/// DEVIATION: no O_DIRECT and no buffer alignment requirements yet (see module docs).
pub struct FileStorage {
    file: std::fs::File,
    file_size: u64,
    completions: Vec<Completion>,
}

impl FileStorage {
    /// Opens (or creates) the data file and ensures at least `size_min` bytes.
    ///
    /// # Errors
    /// Returns any filesystem error from open/resize.
    pub fn open(path: impl AsRef<std::path::Path>, size_min: u64) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let mut file_size = file.metadata()?.len();
        if file_size < size_min {
            file.set_len(size_min)?;
            file_size = size_min;
        }
        Ok(Self { file, file_size, completions: Vec::new() })
    }

    fn drive_read_request(&self, mut request: ReadRequest) -> Completion {
        let base_offset = verify_request(request.zone, &request.buffer, request.offset_in_zone);
        drive_read(&mut request, base_offset, |slice, offset| read_step(&self.file, slice, offset));
        Completion::Read(request)
    }

    fn drive_write_request(&mut self, request: WriteRequest) -> Completion {
        use std::io::{Seek, SeekFrom, Write};

        let base_offset = verify_request(request.zone, &request.buffer, request.offset_in_zone);
        assert_eq!(base_offset % SECTOR_SIZE as u64, 0);

        self.file
            .seek(SeekFrom::Start(base_offset))
            .unwrap_or_else(|err| panic!("impossible write: {err:?}"));
        self.file.write_all(&request.buffer).unwrap_or_else(|err| match err.kind() {
            std::io::ErrorKind::StorageFull => {
                // NB: Intentionally crash on physical space exhaustion.
                // Low space condition is handled logically, via `--limit-storage` argument.
                panic!(
                    "write failed: no space left on device (offset={} size={})",
                    base_offset,
                    request.buffer.len()
                );
            }
            _ => panic!(
                "impossible write: offset={} buffer.len={} error={err:?}",
                base_offset,
                request.buffer.len()
            ),
        });

        Completion::Write(request)
    }
}

impl Storage for FileStorage {
    fn size(&self) -> u64 {
        self.file_size
    }

    fn read_sectors(&mut self, request: ReadRequest) {
        self.completions.push(self.drive_read_request(request));
    }

    fn write_sectors(&mut self, request: WriteRequest) {
        let completion = self.drive_write_request(request);
        self.completions.push(completion);
    }

    fn next_completion(&mut self) -> Option<Completion> {
        if self.completions.is_empty() { None } else { Some(self.completions.remove(0)) }
    }
}

/// In-memory data file with fault injection, standing in for upstream's
/// `testing/storage.zig` (which drives the real code against a simulated disk).
///
/// Fault injection models whole-syscall EIO: any read whose current window contains a
/// faulty sector fails like `error.InputOutput`, driving the same subdivision logic until
/// exactly the faulty logical sector(s) are isolated and zeroed.
pub struct MemoryStorage {
    image: Vec<u8>,
    /// Absolute sector offsets that behave as latent sector errors (reads fail with EIO).
    pub faulty_sectors: HashSet<u64>,
    completions: Vec<Completion>,
}

impl MemoryStorage {
    /// Creates an all-zero image of `size` bytes.
    ///
    /// # Panics
    /// Asserts `size` is a multiple of [`SECTOR_SIZE`].
    #[must_use]
    pub fn new(size: u64) -> Self {
        assert!(size.is_multiple_of(SECTOR_SIZE as u64));
        Self {
            image: vec![0; size as usize],
            faulty_sectors: HashSet::new(),
            completions: Vec::new(),
        }
    }

    fn drive_read_request(&self, mut request: ReadRequest) -> Completion {
        let base_offset = verify_request(request.zone, &request.buffer, request.offset_in_zone);
        drive_read(&mut request, base_offset, |slice, offset| {
            let contains_fault = (0..slice.len()).step_by(SECTOR_SIZE).any(|step| {
                let sector = (offset + step as u64) / SECTOR_SIZE as u64;
                self.faulty_sectors.contains(&sector)
            });
            if contains_fault {
                return Err(FileReadError::InputOutput);
            }

            let available = self.image.len();
            for (step, byte) in slice.iter_mut().enumerate() {
                let src = offset as usize + step;
                *byte = if src >= available {
                    0 // Reading beyond the end of the file descriptor zero-fills.
                } else {
                    self.image[src]
                };
            }
            Ok(())
        });
        Completion::Read(request)
    }
}

impl Storage for MemoryStorage {
    fn size(&self) -> u64 {
        self.image.len() as u64
    }

    fn read_sectors(&mut self, request: ReadRequest) {
        self.completions.push(self.drive_read_request(request));
    }

    fn write_sectors(&mut self, request: WriteRequest) {
        let base_offset = verify_request(request.zone, &request.buffer, request.offset_in_zone);
        let start = base_offset as usize;
        let end = start + request.buffer.len();
        assert!(end <= self.image.len(), "write beyond end of storage");
        self.image[start..end].copy_from_slice(&request.buffer);

        self.completions.push(Completion::Write(request));
    }

    fn next_completion(&mut self) -> Option<Completion> {
        if self.completions.is_empty() { None } else { Some(self.completions.remove(0)) }
    }
}

#[cfg(test)]
mod storage_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::zeroed_buffer;
    use super::{
        Completion, FileStorage, MemoryStorage, ReadRequest, ReadState, SECTOR_SIZE, Storage,
        WriteRequest,
    };
    use crate::Zone;
    use tigerbeetle_core::constants::BLOCK_SIZE;

    fn temp_file_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tigerbeetle-rs-storage-test-{name}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir.join("data.dat")
    }

    fn temp_file_storage(name: &str) -> FileStorage {
        let path = temp_file_path(name);
        let _ = std::fs::remove_file(&path);
        FileStorage::open(&path, Zone::WalPrepares.start()).expect("open temp data file")
    }

    #[test]
    fn write_then_read_round_trip_through_zones() {
        let mut storage = temp_file_storage("round_trip");

        let mut buffer = zeroed_buffer(2 * BLOCK_SIZE);
        for (index, byte) in buffer.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        let buffer_len = buffer.len();

        storage.write_sectors(WriteRequest { zone: Zone::WalPrepares, offset_in_zone: 0, buffer });
        assert!(matches!(storage.next_completion(), Some(Completion::Write(_))));

        storage.read_sectors(ReadRequest {
            zone: Zone::WalPrepares,
            offset_in_zone: 0,
            buffer: zeroed_buffer(buffer_len),
        });
        match storage.next_completion().expect("read completion") {
            Completion::Read(request) => {
                assert_eq!(request.buffer.len(), buffer_len);
                for (index, byte) in request.buffer.iter().enumerate() {
                    assert_eq!(*byte, (index % 251) as u8);
                }
            }
            Completion::Write(_) => unreachable!("expected a read completion"),
        }
    }

    #[test]
    fn read_beyond_eof_zero_fills() {
        let mut storage = temp_file_storage("eof");

        storage.read_sectors(ReadRequest {
            zone: Zone::Grid,
            offset_in_zone: BLOCK_SIZE as u64,
            buffer: zeroed_buffer(BLOCK_SIZE),
        });
        match storage.next_completion().expect("read completion") {
            Completion::Read(request) => {
                assert!(request.buffer.iter().all(|&byte| byte == 0));
            }
            Completion::Write(_) => unreachable!("expected a read completion"),
        }
    }

    #[test]
    fn latent_sector_errors_zero_only_the_faulty_sector() {
        let mut storage = MemoryStorage::new(16 * BLOCK_SIZE as u64);

        let mut pattern = zeroed_buffer(4 * BLOCK_SIZE);
        pattern.fill(7);
        storage.write_sectors(WriteRequest {
            zone: Zone::WalPrepares,
            offset_in_zone: 0,
            buffer: pattern,
        });
        assert!(matches!(storage.next_completion(), Some(Completion::Write(_))));

        // Inject faults into the third and fourth sector of the second block:
        let block_base = Zone::WalPrepares.start() + BLOCK_SIZE as u64;
        let sector_a = block_base / SECTOR_SIZE as u64 + 2;
        let sector_b = block_base / SECTOR_SIZE as u64 + 3;
        storage.faulty_sectors.insert(sector_a);
        storage.faulty_sectors.insert(sector_b);

        storage.read_sectors(ReadRequest {
            zone: Zone::WalPrepares,
            offset_in_zone: BLOCK_SIZE as u64,
            buffer: zeroed_buffer(BLOCK_SIZE),
        });
        match storage.next_completion().expect("read completion") {
            Completion::Read(request) => {
                for (index, &byte) in request.buffer.iter().enumerate() {
                    let in_faulty_sector = index / SECTOR_SIZE == 2 || index / SECTOR_SIZE == 3;
                    if in_faulty_sector {
                        assert_eq!(byte, 0, "faulty sector must be zeroed");
                    } else {
                        assert_eq!(byte, 7, "healthy sector must survive");
                    }
                }
            }
            Completion::Write(_) => unreachable!("expected a read completion"),
        }
    }

    #[test]
    fn adjacent_latent_sector_errors_zero_both_sectors() {
        let mut storage = MemoryStorage::new(4 * BLOCK_SIZE as u64);

        let mut pattern = zeroed_buffer(BLOCK_SIZE);
        pattern.fill(9);
        storage.write_sectors(WriteRequest {
            zone: Zone::WalPrepares,
            offset_in_zone: 0,
            buffer: pattern,
        });
        assert!(matches!(storage.next_completion(), Some(Completion::Write(_))));

        // Two adjacent faulty sectors: the binary search must isolate both.
        let base = Zone::WalPrepares.start() / SECTOR_SIZE as u64;
        storage.faulty_sectors.insert(base);
        storage.faulty_sectors.insert(base + 1);

        storage.read_sectors(ReadRequest {
            zone: Zone::WalPrepares,
            offset_in_zone: 0,
            buffer: zeroed_buffer(BLOCK_SIZE),
        });
        match storage.next_completion().expect("read completion") {
            Completion::Read(request) => {
                for (index, &byte) in request.buffer.iter().enumerate() {
                    if index / SECTOR_SIZE <= 1 {
                        assert_eq!(byte, 0, "faulty sector must be zeroed");
                    } else {
                        assert_eq!(byte, 9, "healthy sector must survive");
                    }
                }
            }
            Completion::Write(_) => unreachable!("expected a read completion"),
        }
    }

    #[test]
    fn read_state_target_halving_and_alignment_cap() {
        // Full-window target:
        let state = ReadState::new(4 * SECTOR_SIZE);
        assert_eq!(state.target_len(), 4 * SECTOR_SIZE);

        // A partial (physical-sector) read leaves the remainder unaligned; the next target
        // is capped back onto the logical sector boundary (see the worked example):
        // 7680 remaining, target_max one sector: 3584 bytes land us back on the boundary.
        let mut state = ReadState::new(8192);
        state.remaining_len = 8192 - 512;
        state.target_max = SECTOR_SIZE;
        assert_eq!(state.target_len(), 3584);

        // Halving performs the binary search over failing sectors:
        let mut state = ReadState::new(8 * SECTOR_SIZE);
        state.halve_target_max();
        assert_eq!(state.target_max, 4 * SECTOR_SIZE);
        state.halve_target_max();
        assert_eq!(state.target_max, 2 * SECTOR_SIZE);
        state.halve_target_max();
        assert_eq!(state.target_max, SECTOR_SIZE);

        // AIMD restores the window after a healthy single-sector read:
        let mut state = ReadState::new(SECTOR_SIZE);
        state.aimd_restore();
        assert_eq!(state.target_max, 2 * SECTOR_SIZE);

        // zero_current_sector/zero_rest clear from the base offset:
        let mut state = ReadState::new(3 * SECTOR_SIZE);
        let mut buffer = vec![0xFF; 3 * SECTOR_SIZE];
        let base = 100 * SECTOR_SIZE as u64;
        state.offset = base + 2 * SECTOR_SIZE as u64;
        state.remaining_len -= 2 * SECTOR_SIZE;
        state.zero_current_sector(&mut buffer, base);
        assert!(buffer[2 * SECTOR_SIZE..3 * SECTOR_SIZE].iter().all(|&b| b == 0));
        assert_eq!(state.remaining_len, 0);
    }

    #[test]
    #[should_panic(expected = "padding is never touched")]
    fn grid_padding_is_rejected() {
        let mut storage = MemoryStorage::new(BLOCK_SIZE as u64);
        storage.write_sectors(WriteRequest {
            zone: Zone::GridPadding,
            offset_in_zone: 0,
            buffer: zeroed_buffer(SECTOR_SIZE),
        });
    }
}
