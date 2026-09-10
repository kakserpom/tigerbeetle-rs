//! Port of upstream `src/tigerbeetle/inspect.zig`'s read-only `inspect superblock`,
//! `inspect manifest`, and `inspect wal` data-file exporters.
//!
//! `inspect_superblock` prints the four stored copies of the superblock (from a data file's
//! superblock zone), grouping the copies by the value of each field, exactly as upstream does
//! in `inspect_superblock`/`print_struct`.
//!
//! `inspect_manifest` walks the manifest log from `checkpoint.manifest_newest_*` backwards
//! through the `previous_manifest_block_*` links, printing each block's address/checksum and
//! its per-event, per-level table-entry counts.
//!
//! `inspect_wal` prints one line per journal slot: the two header copies (the `wal_headers`
//! zone and the prepare's own header in `wal_prepares`) group by whole-header equality, and
//! each group's line carries the header's checksum/fields with a `|`/`X` valid-mark label like
//! `||___0: `. `inspect_wal_slot` prints a single slot the same way plus its prepare body.
//!
//! Upstream relies on Zig's comptime `std.meta.fields(SuperBlockHeader)` to enumerate the
//! struct layout; Rust has no reflection over struct fields, so the layout is spelled out in
//! the [`Field`] trees below (mirroring superblock.zig's declaration order) and pinned against
//! `superblock.rs`'s offset constants by unit tests.

use core::fmt;

use tigerbeetle_core::checksum::checksum;
use tigerbeetle_core::constants::{
    JOURNAL_SIZE_HEADERS, JOURNAL_SLOT_COUNT, MEMBERS_MAX, MESSAGE_SIZE_MAX, SECTOR_SIZE,
    SUPERBLOCK_COPIES, VIEW_HEADERS_MAX,
};

use crate::Operation;
use crate::Zone;
use crate::message_header::{SIZE, format_prepare_raw};
use crate::multiversion::Release;
use crate::storage::{Completion, ReadRequest, Storage};
use crate::superblock::{SUPERBLOCK_HEADER_SIZE, SuperBlockHeader};

/// Port of `SuperBlockHeader.CHECKSUM_IGNORE_SIZE` (src/vsr/superblock.zig): the checksum
/// covers the header starting at this byte offset, ignoring `checksum`, `checksum_padding`,
/// and `copy`.
const CHECKSUM_IGNORE_SIZE: usize = 34;

/// Each field of the superblock header, in declaration order.
#[derive(Clone, Copy)]
struct Field {
    /// Field name (matches the Zig struct member name).
    name: &'static str,
    /// Byte offset of the field within its containing struct.
    offset: usize,
    /// Byte length of the field (used to slice the field bytes for grouping/printing).
    len: usize,
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    U128,
    U64,
    U32,
    U16,
    U8,
    Release,
    /// `[len]u8` array, printed as `[len]u8{0}` or `[len]u8{nonzero}`.
    U8Array,
    /// `[MEMBERS_MAX]u128` — each element printed as `label[i]=0x…`.
    Members,
    /// `[VIEW_HEADERS_MAX]Prepare` — each element printed as `label[i]=Prepare{…}`.
    ViewHeadersAll,
    /// A single `Prepare` struct (which has a format fn) — printed as `label=Prepare{…}`.
    Prepare,
    /// A struct without a format fn — recurses into [`Field::sub`] with `label.field`.
    Struct(&'static [Field]),
}

impl Field {
    const fn new(name: &'static str, offset: usize, len: usize, kind: Kind) -> Self {
        Self { name, offset, len, kind }
    }
}

/// `SuperBlockHeader` fields (upstream: `src/vsr/superblock.zig` `SuperBlockHeader`), in Zig
/// declaration order.
const HEADER_FIELDS: &[Field] = &[
    Field::new("checksum", 0, 16, Kind::U128),
    Field::new("checksum_padding", 16, 16, Kind::U128),
    Field::new("copy", 32, 2, Kind::U16),
    Field::new("version", 34, 2, Kind::U16),
    Field::new("release_format", 36, 4, Kind::Release),
    Field::new("sequence", 40, 8, Kind::U64),
    Field::new("cluster", 48, 16, Kind::U128),
    Field::new("parent", 64, 16, Kind::U128),
    Field::new("parent_padding", 80, 16, Kind::U128),
    Field::new("vsr_state", 96, VSR_STATE_SIZE, Kind::Struct(VSR_STATE_FIELDS)),
    Field::new("flags", 2144, 8, Kind::U64),
    Field::new("view_headers_count", 2152, 4, Kind::U32),
    Field::new("reserved", 2156, SECTOR_SIZE - 2156, Kind::U8Array),
    Field::new(
        "view_headers_all",
        SECTOR_SIZE,
        VIEW_HEADERS_MAX as usize * SIZE,
        Kind::ViewHeadersAll,
    ),
    Field::new(
        "view_headers_reserved",
        SECTOR_SIZE + VIEW_HEADERS_MAX as usize * SIZE,
        SUPERBLOCK_HEADER_SIZE - (SECTOR_SIZE + VIEW_HEADERS_MAX as usize * SIZE),
        Kind::U8Array,
    ),
];

/// `VSRState` size (upstream: `vsr_state_size`, fixed at 2048 bytes).
const VSR_STATE_SIZE: usize = 2048;

const _: () = assert!(HEADER_FIELDS[12].len == 1940);

/// `VSRState` fields (upstream `VSRState`), relative to the `vsr_state` struct base.
const VSR_STATE_FIELDS: &[Field] = &[
    Field::new("checkpoint", 0, CHECKPOINT_STATE_SIZE, Kind::Struct(CHECKPOINT_STATE_FIELDS)),
    Field::new("replica_id", 1024, 16, Kind::U128),
    Field::new("members", 1040, MEMBERS_MAX * 16, Kind::Members),
    Field::new("commit_max", 1232, 8, Kind::U64),
    Field::new("sync_op_min", 1240, 8, Kind::U64),
    Field::new("sync_op_max", 1248, 8, Kind::U64),
    Field::new("sync_view", 1256, 4, Kind::U32),
    Field::new("log_view", 1260, 4, Kind::U32),
    Field::new("view", 1264, 4, Kind::U32),
    Field::new("replica_count", 1268, 1, Kind::U8),
    Field::new("reserved", 1269, VSR_STATE_SIZE - 1269, Kind::U8Array),
];

const _: () = assert!(VSR_STATE_FIELDS[10].len == 779);

/// `CheckpointState` size (upstream: `checkpoint_state_size`, fixed at 1024 bytes).
const CHECKPOINT_STATE_SIZE: usize = 1024;

/// `CheckpointState` fields (upstream `CheckpointState`), relative to the `checkpoint` struct
/// base.
const CHECKPOINT_STATE_FIELDS: &[Field] = &[
    Field::new("header", 0, SIZE, Kind::Prepare),
    Field::new("free_set_blocks_acquired_last_block_checksum", 256, 16, Kind::U128),
    Field::new("free_set_blocks_acquired_last_block_checksum_padding", 272, 16, Kind::U128),
    Field::new("free_set_blocks_released_last_block_checksum", 288, 16, Kind::U128),
    Field::new("free_set_blocks_released_last_block_checksum_padding", 304, 16, Kind::U128),
    Field::new("client_sessions_last_block_checksum", 320, 16, Kind::U128),
    Field::new("client_sessions_last_block_checksum_padding", 336, 16, Kind::U128),
    Field::new("manifest_oldest_checksum", 352, 16, Kind::U128),
    Field::new("manifest_oldest_checksum_padding", 368, 16, Kind::U128),
    Field::new("manifest_newest_checksum", 384, 16, Kind::U128),
    Field::new("manifest_newest_checksum_padding", 400, 16, Kind::U128),
    Field::new("snapshots_block_checksum", 416, 16, Kind::U128),
    Field::new("snapshots_block_checksum_padding", 432, 16, Kind::U128),
    Field::new("free_set_blocks_acquired_checksum", 448, 16, Kind::U128),
    Field::new("free_set_blocks_released_checksum", 464, 16, Kind::U128),
    Field::new("client_sessions_checksum", 480, 16, Kind::U128),
    Field::new("parent_checkpoint_id", 496, 16, Kind::U128),
    Field::new("grandparent_checkpoint_id", 512, 16, Kind::U128),
    Field::new("free_set_blocks_acquired_last_block_address", 528, 8, Kind::U64),
    Field::new("free_set_blocks_released_last_block_address", 536, 8, Kind::U64),
    Field::new("client_sessions_last_block_address", 544, 8, Kind::U64),
    Field::new("manifest_oldest_address", 552, 8, Kind::U64),
    Field::new("manifest_newest_address", 560, 8, Kind::U64),
    Field::new("snapshots_block_address", 568, 8, Kind::U64),
    Field::new("storage_size", 576, 8, Kind::U64),
    Field::new("free_set_blocks_acquired_size", 584, 8, Kind::U64),
    Field::new("free_set_blocks_released_size", 592, 8, Kind::U64),
    Field::new("client_sessions_size", 600, 8, Kind::U64),
    Field::new("manifest_block_count", 608, 4, Kind::U32),
    Field::new("release", 612, 4, Kind::Release),
    Field::new("reserved", 616, CHECKPOINT_STATE_SIZE - 616, Kind::U8Array),
];

const _: () = assert!(CHECKPOINT_STATE_FIELDS[30].len == 408);

/// Port of `inspect_superblock`: prints the superblock zone, grouping copies by field value.
///
/// `copies` are the four raw superblock header copies (+padding-ignored) read straight from
/// the data file's superblock zone, as upstream's `bytesAsValue` reads them — `from_wire`
/// asserts reserved areas are zeroed and would panic on corrupt files, so we never decode.
///
/// # Errors
///
/// Returns the underlying formatting error if writing to `output` fails.
pub fn inspect_superblock(
    output: &mut impl fmt::Write,
    copies: &[[u8; SUPERBLOCK_HEADER_SIZE]],
) -> fmt::Result {
    let valid: Vec<bool> = copies.iter().map(valid_checksum).collect();

    for field in HEADER_FIELDS {
        for group in groups(copies, field) {
            // Upstream: `matches.first_set() == a` keeps the group in first-appearance order,
            // so `group.first_set()` is the index of the copy whose value we print.
            let header_index = first_set(group);
            let mark = if valid[header_index] { '|' } else { 'X' };

            let mut label = String::new();
            for copy in 0..SUPERBLOCK_COPIES {
                let bit = 1 << copy;
                label.push(if group & bit != 0 { mark } else { '_' });
            }
            label.push(' ');
            label.push_str(field.name);

            print_struct(output, &label, &copies[header_index], field, 0)?;
        }
    }
    Ok(())
}

/// Decode a raw superblock copy into a [`SuperBlockHeader`] for the CLI's working-quorum and
/// version checks.
///
/// DEVIATION: upstream overlays the raw bytes (`bytesAsValue`), while `SuperBlockHeader::from_wire`
/// asserts that every reserved/padding area is zeroed and would panic on a corrupt copy. We zero
/// those areas up front: `SuperBlockHeader::valid_checksum()` recomputes the checksum over a
/// zeroed `to_wire()`, so any copy whose raw reserved bytes are nonzero fails checksum validation
/// here exactly as it does upstream — zeroing before decode cannot change whether the copy is
/// valid, it only keeps `inspect superblock` from panicking on corrupt files.
#[must_use]
pub fn decode_lenient(bytes: &[u8; SUPERBLOCK_HEADER_SIZE]) -> Option<SuperBlockHeader> {
    let mut clean = *bytes;

    // `SuperBlockHeader` reserved + `view_headers_reserved`.
    clean[HEADER_FIELDS[12].offset..HEADER_FIELDS[12].offset + HEADER_FIELDS[12].len].fill(0);
    clean[HEADER_FIELDS[14].offset..].fill(0);

    // `VSRState.reserved`.
    let vsr_base = HEADER_FIELDS[9].offset;
    let vsr_field = VSR_STATE_FIELDS[10];
    clean[vsr_base + vsr_field.offset..vsr_base + vsr_field.offset + vsr_field.len].fill(0);

    // `CheckpointState.reserved` and the six trailing checksum paddings.
    let checkpoint_base = vsr_base + VSR_STATE_FIELDS[0].offset;
    let checkpoint_reserved = CHECKPOINT_STATE_FIELDS[30];
    clean[checkpoint_base + checkpoint_reserved.offset
        ..checkpoint_base + checkpoint_reserved.offset + checkpoint_reserved.len]
        .fill(0);
    // The six `*_padding` fields are interleaved with their checksums at (1,2), (3,4), ...,
    // (11,12) — i.e. even indices 2,4,6,8,10,12. `CheckpointState::from_wire` asserts these
    // are zero, and `inspect_superblock` groups the *raw* copies by value, so zeroing must
    // never touch the checksum-value fields.
    for field in CHECKPOINT_STATE_FIELDS.iter().take(13).skip(2).step_by(2) {
        clean[checkpoint_base + field.offset..checkpoint_base + field.offset + field.len].fill(0);
    }

    SuperBlockHeader::from_wire(&clean)
}

/// Port of `GroupByType.groups()`: partitions the copies into groups of equal field bytes,
/// emitted in order of each value's first appearance.
fn groups(copies: &[[u8; SUPERBLOCK_HEADER_SIZE]], field: &Field) -> Vec<u16> {
    let mut groups = Vec::new();
    for a in 0..SUPERBLOCK_COPIES {
        let mut matches = 0u16;
        for b in 0..SUPERBLOCK_COPIES {
            if equal(copies, field, a, b) {
                matches |= 1 << b;
            }
        }
        if first_set(matches) == a {
            groups.push(matches);
        }
    }
    groups
}

/// DEVIATION: upstream groups copies by `vsr.checksum(&as_bytes(&field))`; we compare the field
/// bytes directly, which is strictly stronger (equal bytes imply equal checksums) and avoids
/// hashing every field twice.
fn equal(copies: &[[u8; SUPERBLOCK_HEADER_SIZE]], field: &Field, a: usize, b: usize) -> bool {
    copies[a][field.offset..field.offset + field.len]
        == copies[b][field.offset..field.offset + field.len]
}

/// First set bit of a copy mask, or `usize::MAX` if unset (upstream's
/// `bitset.first_set()`).
#[must_use]
fn first_set(mask: u16) -> usize {
    let trailing = mask.trailing_zeros();
    if trailing >= u16::BITS { usize::MAX } else { trailing as usize }
}

/// Port of `SuperBlockHeader.valid_checksum()` on raw bytes: the checksum (covering the
/// header from `CHECKSUM_IGNORE_SIZE`) and `checksum_padding` must match.
#[must_use]
fn valid_checksum(bytes: &[u8; SUPERBLOCK_HEADER_SIZE]) -> bool {
    if checksum(&bytes[CHECKSUM_IGNORE_SIZE..]) != get_u128(bytes, 0) {
        return false;
    }
    if get_u128(bytes, 16) != 0 {
        return false;
    }
    true
}

/// Port of `print_struct`: prints a struct value field-by-field (recursing into sub-structs),
/// or prints the scalar value as `label=value`.
///
/// `base` is the absolute offset of the struct containing `field` (upstream threads
/// `offset + field.offset` down the recursion), so nested offsets yield absolute positions.
fn print_struct(
    output: &mut impl fmt::Write,
    label: &str,
    bytes: &[u8; SUPERBLOCK_HEADER_SIZE],
    field: &Field,
    base: usize,
) -> fmt::Result {
    let abs = base + field.offset;
    let value = match field.kind {
        Kind::Struct(fields) => {
            for sub in fields {
                let sublabel = format!("{label}.{}", sub.name);
                print_struct(output, &sublabel, bytes, sub, abs)?;
            }
            return Ok(());
        }
        Kind::U8Array => {
            let len = field.len;
            let array = &bytes[abs..abs + len];
            let nonzero = array.iter().any(|&byte| byte != 0);
            if nonzero { format!("[{len}]u8{{nonzero}}") } else { format!("[{len}]u8{{0}}") }
        }
        Kind::Members => {
            for i in 0..MEMBERS_MAX {
                writeln!(output, "{label}[{i}]=0x{:0>32x}", get_u128(bytes, abs + i * 16))?;
            }
            return Ok(());
        }
        Kind::ViewHeadersAll => {
            for i in 0..VIEW_HEADERS_MAX as usize {
                write!(output, "{label}[{i}]=")?;
                format_prepare_raw(
                    output,
                    bytes[abs + i * SIZE..abs + (i + 1) * SIZE]
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("slice length checked")),
                )?;
                writeln!(output)?;
            }
            return Ok(());
        }
        Kind::Prepare => {
            write!(output, "{label}=")?;
            format_prepare_raw(
                output,
                bytes[abs..abs + SIZE]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("slice length checked")),
            )?;
            writeln!(output)?;
            return Ok(());
        }
        Kind::U128 => format!("0x{:0>32x}", get_u128(bytes, abs)),
        Kind::U64 => get_u64(bytes, abs).to_string(),
        Kind::U32 => get_u32(bytes, abs).to_string(),
        Kind::U16 => get_u16(bytes, abs).to_string(),
        Kind::U8 => bytes[abs].to_string(),
        Kind::Release => Release { value: get_u32(bytes, abs) }.to_string(),
    };
    writeln!(output, "{label}={value}")
}

fn get_u128(bytes: &[u8], offset: usize) -> u128 {
    u128::from_le_bytes(
        bytes[offset..offset + 16]
            .try_into()
            .unwrap_or_else(|_| unreachable!("slice length checked")),
    )
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .unwrap_or_else(|_| unreachable!("slice length checked")),
    )
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .unwrap_or_else(|_| unreachable!("slice length checked")),
    )
}

fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .unwrap_or_else(|_| unreachable!("slice length checked")),
    )
}

fn get_u8(bytes: &[u8], offset: usize) -> u8 {
    bytes[offset]
}

/// Port of `Inspector.read_block` (src/tigerbeetle/inspect.zig): read one grid block and
/// verify its header checksum, body checksum, and (when given) expected block checksum.
///
/// Returns `None` for every failure (unknown command, bad checksum, wrong block, or an
/// out-of-bounds address/size), mirroring upstream's `error` returns without ever panicking on
/// disk input — `inspect_manifest` collapses each failure into `error: manifest block not
/// found`.
#[must_use]
fn read_block(
    storage: &mut dyn Storage,
    address: u64,
    expected_checksum: Option<u128>,
) -> Option<Vec<u8>> {
    use tigerbeetle_core::constants::BLOCK_SIZE;

    use crate::Zone;
    use crate::message_header::{Block, SIZE as HEADER_SIZE, TypedHeader};
    use crate::storage::{Completion, ReadRequest};

    // Upstream computes `(address - 1) * block_size`; a manifest never references block 0, so
    // both the underflow and an overflow of the byte offset read as "not found".
    let offset_in_zone =
        if address == 0 { None } else { (address - 1).checked_mul(BLOCK_SIZE as u64) }?;

    storage.read_sectors(ReadRequest {
        zone: Zone::Grid,
        offset_in_zone,
        buffer: vec![0_u8; BLOCK_SIZE],
    });
    let block = match storage
        .next_completion()
        .unwrap_or_else(|| unreachable!("read completes synchronously"))
    {
        Completion::Read(request) => request.buffer,
        Completion::Write(_) => unreachable!("only a read was submitted"),
    };

    let header = Block::from_wire(block[..HEADER_SIZE].try_into().ok()?)?;
    if !header.valid_checksum() {
        return None;
    }
    // The body slice is `[SIZE..size]`; reject sizes that would make it invalid before
    // `valid_checksum_body` (which asserts the exact body length).
    if !(HEADER_SIZE..=BLOCK_SIZE).contains(&(header.size as usize)) {
        return None;
    }
    if !header.valid_checksum_body(&block[HEADER_SIZE..header.size as usize]) {
        return None;
    }
    if let Some(expected_checksum) = expected_checksum
        && header.checksum != expected_checksum
    {
        return None;
    }
    Some(block)
}

/// Port of `Inspector.inspect_manifest` (src/tigerbeetle/inspect.zig): walk the manifest log
/// from the superblock's newest block backwards, printing each block's header line and its
/// per-event, per-level table-entry counts.
///
/// The walk starts at `checkpoint.manifest_newest_*` and follows the
/// `previous_manifest_block_*` links for `checkpoint.manifest_block_count` blocks; a block
/// that fails to read prints `error: manifest block not found` and stops, exactly as upstream.
///
/// # Errors
///
/// Returns the underlying formatting error if writing to `output` fails.
pub fn inspect_manifest(
    output: &mut impl fmt::Write,
    storage: &mut dyn Storage,
    superblock: &SuperBlockHeader,
) -> fmt::Result {
    use tigerbeetle_core::constants::LSM_LEVELS;

    use crate::schema::ManifestNode::{self, ENTRY_COUNT_MAX, Event};

    let checkpoint = &superblock.vsr_state.checkpoint;

    let mut manifest_block_address = checkpoint.manifest_newest_address;
    let mut manifest_block_checksum = checkpoint.manifest_newest_checksum;

    for block_index in 0..checkpoint.manifest_block_count {
        write!(
            output,
            "manifest_log.blocks[{block_index}]: address={manifest_block_address} \
             checksum={manifest_block_checksum:032x} "
        )?;

        let Some(block) =
            read_block(storage, manifest_block_address, Some(manifest_block_checksum))
        else {
            writeln!(output, "error: manifest block not found")?;
            break;
        };

        let metadata = ManifestNode::metadata(&block);
        let tables = ManifestNode::tables(&block);
        let mut entry_counts = [[0_usize; LSM_LEVELS as usize]; 4];
        for table in &tables {
            entry_counts[table.label.event as usize][table.label.level as usize] += 1;
        }

        write!(output, "entries={}/{}", metadata.entry_count, ENTRY_COUNT_MAX)?;
        for (event, name) in
            [(Event::Insert, "insert"), (Event::Update, "update"), (Event::Remove, "remove")]
        {
            write!(output, " {name}=")?;
            for (level, count) in entry_counts[event as usize].iter().enumerate() {
                if level != 0 {
                    write!(output, ",")?;
                }
                write!(output, "{count}")?;
            }
        }
        writeln!(output)?;

        manifest_block_address = metadata.previous_manifest_block_address;
        manifest_block_checksum = metadata.previous_manifest_block_checksum;
    }
    Ok(())
}

/// Port of `inspect_op` (src/tigerbeetle/inspect.zig): print an aligned table of the five
/// checkpoint points adjacent to `op` — `checkpoint`, `op`, `trigger`, `prepare_max`,
/// `checkpoint_next` — with the delta to each point's successor.
///
/// Pure computation over [`crate::checkpoint`] (no file access), so it runs directly from the
/// config; the column/width format is upstream's (`{label:<20}` header, then per pair the
/// `{op:<15}` / `+{diff:<4}` body, and the final `{op:<20}`).
///
/// # Panics
///
/// Panics (matching upstream's `@divExact` assertion) when the op→checkpoint reduction would
/// perform non-exact integer division. This cannot happen for any valid `u64` op value because
/// `CHECKPOINT_OPS` always divides `(op + 1 - r)` exactly by construction.
///
/// # Errors
///
/// Propagates [`fmt::Error`] from the underlying writer — impossible with [`String`], but
/// callers using a fallible writer must handle it.
pub fn inspect_op(output: &mut impl fmt::Write, op: u64) -> fmt::Result {
    use tigerbeetle_core::constants::VSR_CHECKPOINT_OPS as CHECKPOINT_OPS;

    use crate::checkpoint::{checkpoint_after, prepare_max_for_checkpoint, trigger_for_checkpoint};

    struct Point {
        label: &'static str,
        op: u64,
    }

    let checkpoints = CHECKPOINT_OPS as u64;
    let checkpoint = if op < checkpoints - 1 {
        0
    } else {
        // op = q * checkpoints - 1 + r; the division is exact (upstream `@divExact`).
        let r = (op + 1) % checkpoints;
        assert_eq!((op + 1 - r) % checkpoints, 0);
        let q = (op + 1 - r) / checkpoints;
        q * checkpoints - 1
    };
    let checkpoint_next = checkpoint_after(checkpoint);

    let mut points = [
        Point { label: "checkpoint", op: checkpoint },
        // Upstream `orelse 0`: a checkpoint with no trigger/prepare_max prints 0.
        Point { label: "trigger", op: trigger_for_checkpoint(checkpoint).unwrap_or(0) },
        Point { label: "prepare_max", op: prepare_max_for_checkpoint(checkpoint).unwrap_or(0) },
        Point { label: "checkpoint_next", op: checkpoint_next },
        Point { label: "op", op },
    ];
    // Upstream sorts with insertion sort (stable), so equal op values keep the struct order.
    points.sort_by_key(|point| point.op);

    for point in &points {
        write!(output, "{:<20}", point.label)?;
    }
    writeln!(output)?;
    for pair in points.windows(2) {
        write!(output, "{:<15}+{:<4}", pair[0].op, pair[1].op - pair[0].op)?;
    }
    writeln!(output, "{:<20}", points[4].op)
}

/// Port of `Inspector.read_buffer` (src/tigerbeetle/inspect.zig): submit a `read_sectors` from
/// a zone and block for its (synchronous) completion, returning the buffer.
#[must_use]
fn read_buffer(storage: &mut dyn Storage, zone: Zone, offset_in_zone: u64, len: usize) -> Vec<u8> {
    storage.read_sectors(ReadRequest { zone, offset_in_zone, buffer: vec![0_u8; len] });
    match storage.next_completion() {
        Some(Completion::Read(request)) => request.buffer,
        None | Some(Completion::Write(_)) => unreachable!("a read completes synchronously"),
    }
}

/// Port of `Header.valid_checksum()` on raw header bytes: for the WAL inspector the header is
/// read as bytes (upstream `bytesAsValue`), not decoded, so the checksum is computed over
/// `bytes[16..]` — the whole header minus its first (checksum-sum) 16 bytes, exactly as
/// `Header.calculate_checksum` does.
#[must_use]
fn valid_header_checksum(bytes: &[u8; SIZE]) -> bool {
    get_u128(bytes, 0) == checksum(&bytes[16..])
}

/// Port of `Header.valid_checksum_body()` on raw header bytes: the `checksum_body` field
/// (offset 32) must match the checksum of `body`.
#[must_use]
fn valid_header_checksum_body(bytes: &[u8; SIZE], body: &[u8]) -> bool {
    get_u128(bytes, 32) == checksum(body)
}

/// Group the two WAL header copies (the `wal_headers` slot header and the prepare's own header)
/// by whole-header equality — upstream `GroupByType(2).compare(as_bytes)` for both copies.
fn wal_groups(wal_header: &[u8; SIZE], wal_prepare: &[u8; SIZE]) -> Vec<u16> {
    let copies: [&[u8; SIZE]; 2] = [wal_header, wal_prepare];
    let mut groups = Vec::new();
    for a in 0..copies.len() {
        let mut matches = 0u16;
        for b in 0..copies.len() {
            if copies[a] == copies[b] {
                matches |= 1 << b;
            }
        }
        if first_set(matches) == a {
            groups.push(matches);
        }
    }
    groups
}

/// Whether `mask` has bit `copy` set (upstream `bitset.is_set`).
const fn is_set(mask: u16, copy: usize) -> bool {
    mask & (1 << copy) != 0
}

/// Port of `print_value` for `vsr.Operation`: `Operation.valid(StateMachine.Operation)` +
/// `tag_name` — known control-plane and state-machine operations print their bare tag name,
/// unknown ordinals print `{n}!`.
///
/// DEVIATION: upstream resolves tag names at comptime over both enums; the port matches the
/// control-plane ordinals here and delegates state-machine ordinals (≥
/// `vsr_operations_reserved`) to [`Operation`]'s `Display`, which prints the same names.
fn write_operation_name(output: &mut impl fmt::Write, operation: Operation) -> fmt::Result {
    match operation {
        Operation::RESERVED => write!(output, "reserved"),
        Operation::ROOT => write!(output, "root"),
        Operation::REGISTER => write!(output, "register"),
        Operation::RECONFIGURE => write!(output, "reconfigure"),
        Operation::PULSE => write!(output, "pulse"),
        Operation::UPGRADE => write!(output, "upgrade"),
        Operation::NOOP => write!(output, "noop"),
        _ => write!(output, "{operation}"),
    }
}

/// Port of `Inspector.inspect_wal` (src/tigerbeetle/inspect.zig): print one line per journal
/// slot, per group of equal header copies, showing the slot's header checksum and fields.
///
/// Each slot holds two header copies (the `wal_headers` zone and the `wal_prepares` zone);
/// identical copies form a single `||` group, differing copies split into `|_` (redundant
/// header) and `_|` (prepare header). The mark is `|` when the copy's checksum is valid — and
/// for groups covering the prepare header, also when its body checksum is valid — `X`
/// otherwise. The label's `{slot:_>4}: ` suffix reproduces upstream's `{:_>4}: ` slot prefix.
///
/// Upstream's four `log.info` preamble lines explaining the column meanings go to stderr, so
/// they are not part of the printed output.
///
/// # Errors
///
/// Returns the underlying formatting error if writing to `output` fails.
pub fn inspect_wal(output: &mut impl fmt::Write, storage: &mut dyn Storage) -> fmt::Result {
    let headers_buffer = read_buffer(storage, Zone::WalHeaders, 0, JOURNAL_SIZE_HEADERS);

    for slot in 0..JOURNAL_SLOT_COUNT as usize {
        let prepare_buffer = read_buffer(
            storage,
            Zone::WalPrepares,
            (slot * MESSAGE_SIZE_MAX as usize) as u64,
            MESSAGE_SIZE_MAX as usize,
        );

        let wal_header: [u8; SIZE] = headers_buffer[slot * SIZE..(slot + 1) * SIZE]
            .try_into()
            .unwrap_or_else(|_| unreachable!("slice length checked"));
        let wal_prepare: [u8; SIZE] = prepare_buffer[..SIZE]
            .try_into()
            .unwrap_or_else(|_| unreachable!("slice length checked"));

        // Upstream slices `prepare_buffer[@sizeOf(Header)..wal_prepare.size]`; a corrupt `size`
        // out of the buffer would OOB-panic there, so per the no-panics-on-disk-input rule the
        // body is treated as invalid instead.
        let wal_prepare_body_valid = {
            let size = get_u32(&wal_prepare, 96) as usize;
            if (SIZE..=prepare_buffer.len()).contains(&size) {
                valid_header_checksum(&wal_prepare)
                    && valid_header_checksum_body(&wal_prepare, &prepare_buffer[SIZE..size])
            } else {
                false
            }
        };

        for group in wal_groups(&wal_header, &wal_prepare) {
            let header_index = first_set(group);
            let header: &[u8; SIZE] = if header_index == 0 { &wal_header } else { &wal_prepare };
            let header_valid =
                valid_header_checksum(header) && (!is_set(group, 1) || wal_prepare_body_valid);

            let mark = if header_valid { '|' } else { 'X' };
            // The two mark chars reproduce upstream's `GroupByType(2)` bit order: `wal_headers`
            // (redundant) first, `wal_prepares` second.
            write!(output, "{}", if is_set(group, 0) { mark } else { '_' })?;
            write!(output, "{}", if is_set(group, 1) { mark } else { '_' })?;
            write!(output, "{slot:_>4}: ")?;

            let checksum = get_u128(header, 0);
            let release = Release { value: get_u32(header, 108) };
            let view = get_u32(header, 104);
            let op = get_u64(header, 224);
            let size = get_u32(header, 96);
            let operation = Operation(get_u8(header, 252));

            write!(
                output,
                "checksum=0x{checksum:032x} release={release} view={view} op={op} \
                 size={size} operation=",
            )?;
            write_operation_name(output, operation)?;
            writeln!(output, " ")?;
        }
    }
    Ok(())
}

/// Port of `Inspector.inspect_wal_slot` (src/tigerbeetle/inspect.zig): print a single journal
/// slot — the two header copies grouped as in [`inspect_wal`], each printed with
/// `label=Prepare{…}`, then the prepare body.
///
/// The slot headers print via the `Prepare` format fn (`format_header`), reproduced here by
/// [`format_prepare_raw`] on the raw bytes. The body prints `(no body)` for zero-length
/// bodies; event decoding is deferred (see [`write_prepare_body`]).
///
/// # Errors
///
/// Returns the underlying formatting error if writing to `output` fails.
///
/// # Panics
///
/// Panics if `slot` does not fit in the journal (upstream asserts).
pub fn inspect_wal_slot(
    output: &mut impl fmt::Write,
    storage: &mut dyn Storage,
    slot: usize,
) -> fmt::Result {
    assert!(slot <= JOURNAL_SLOT_COUNT as usize, "slot exceeds {}", JOURNAL_SLOT_COUNT - 1);

    let headers_buffer = read_buffer(storage, Zone::WalHeaders, 0, JOURNAL_SIZE_HEADERS);
    let prepare_buffer = read_buffer(
        storage,
        Zone::WalPrepares,
        (slot * MESSAGE_SIZE_MAX as usize) as u64,
        MESSAGE_SIZE_MAX as usize,
    );

    let wal_header: [u8; SIZE] = headers_buffer[slot * SIZE..(slot + 1) * SIZE]
        .try_into()
        .unwrap_or_else(|_| unreachable!("slice length checked"));
    let wal_prepare: [u8; SIZE] =
        prepare_buffer[..SIZE].try_into().unwrap_or_else(|_| unreachable!("slice length checked"));

    let prepare_body_valid = {
        let size = get_u32(&wal_prepare, 96) as usize;
        if (SIZE..=prepare_buffer.len()).contains(&size) {
            valid_header_checksum(&wal_prepare)
                && valid_header_checksum_body(&wal_prepare, &prepare_buffer[SIZE..size])
        } else {
            false
        }
    };

    for group in wal_groups(&wal_header, &wal_prepare) {
        let header_index = first_set(group);
        let header: &[u8; SIZE] = if header_index == 0 { &wal_header } else { &wal_prepare };
        let header_mark = if valid_header_checksum(header) { '|' } else { 'X' };

        let mut label = String::new();
        label.push(if is_set(group, 0) { header_mark } else { '_' });
        label.push(if is_set(group, 1) { header_mark } else { '_' });

        // `print_struct(label, header)` where `Prepare` has a format fn prints
        // `{label}=Prepare{…}` (via `format_header` → `format_prepare_raw`).
        write!(output, "{label}=")?;
        format_prepare_raw(output, header)?;
        writeln!(output)?;
    }

    write_prepare_body(output, &prepare_buffer)?;

    if !prepare_body_valid {
        writeln!(output, "error: invalid prepare body!")?;
    }
    Ok(())
}

/// Port of `print_prepare_body` (src/tigerbeetle/inspect.zig): decode a prepare's body against
/// the operation schema.
///
/// The control-plane schemas that have bodies (`reconfigure`, `upgrade`) would be printed
/// event-by-event via `print_struct`; decoding them is deferred, so a nonzero body on those
/// prints `error: unexpected body size=…, @sizeOf(Event)=…` like upstream does for body sizes
/// that don't divide evenly. `(no body)` is printed for zero-length bodies — the only case a
/// freshly formatted WAL reaches.
///
/// DEVIATION: upstream's `else` branch prints `@tagName(header.operation)`; the port writes
/// the same bare name via [`write_operation_name`].
///
/// # Errors
///
/// Returns the underlying formatting error if writing to `output` fails.
fn write_prepare_body(output: &mut impl fmt::Write, prepare: &[u8]) -> fmt::Result {
    let header: &[u8; SIZE] =
        &prepare[..SIZE].try_into().unwrap_or_else(|_| unreachable!("slice length checked"));
    let operation = Operation(get_u8(header, 252));
    let size = get_u32(header, 96) as usize;
    let body_size = size.saturating_sub(SIZE);

    // Upstream `operation_schemas` entry sizes: `.reserved`/`.root`/`.pulse` and the register
    // body are `extern struct {}` (size 0); `.reconfigure` is `ReconfigurationRequest` (256);
    // `.upgrade` is `UpgradeRequest` (16). State-machine event types are not yet ported.
    let event_size = match operation {
        Operation::RECONFIGURE => size_of::<crate::ReconfigurationRequest>(),
        Operation::UPGRADE => size_of::<crate::UpgradeRequest>(),
        _ => 0,
    };

    if body_size == 0 {
        writeln!(output, "(no body)")
    } else if event_size != 0 && body_size.is_multiple_of(event_size) {
        // DEVIATION: per-event `print_struct` decoding deferred (needs the event schemas).
        write!(output, "error: unimplemented operation=")?;
        write_operation_name(output, operation)?;
        writeln!(output)
    } else {
        writeln!(output, "error: unexpected body size={size}, @sizeOf(Event)={event_size}")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::schema::ManifestNode::ENTRY_COUNT_MAX;
    use crate::storage::MemoryStorage;
    use tigerbeetle_lsm::schema::manifest_node::Event;

    /// Builds a raw superblock copy with a valid checksum (bytes[0..34] are excluded from the
    /// checksum) and the given per-copy overrides.
    fn make_copy(checksum_padding: u128, copy: u16) -> [u8; SUPERBLOCK_HEADER_SIZE] {
        let mut bytes = [0u8; SUPERBLOCK_HEADER_SIZE];
        // `copy` at offset 32 is excluded from the checksum (`CHECKSUM_IGNORE_SIZE` = 34).
        bytes[32..34].copy_from_slice(&copy.to_le_bytes());
        // `version` (upstream: 2) at offset 34 is covered by the checksum.
        bytes[34..36].copy_from_slice(&2u16.to_le_bytes());
        let sum = checksum(&bytes[CHECKSUM_IGNORE_SIZE..]);
        bytes[0..16].copy_from_slice(&sum.to_le_bytes());
        bytes[16..32].copy_from_slice(&checksum_padding.to_le_bytes());
        bytes
    }

    fn copies_full() -> Vec<[u8; SUPERBLOCK_HEADER_SIZE]> {
        (0..SUPERBLOCK_COPIES)
            .map(|i| make_copy(0, u16::try_from(i).expect("index fits in u16")))
            .collect()
    }

    #[test]
    fn valid_checksum_rejects_bad_padding() {
        let good = make_copy(0, 0);
        assert!(valid_checksum(&good));
        let bad = make_copy(1, 0);
        assert!(!valid_checksum(&bad));
    }

    #[test]
    fn decode_lenient_preserves_manifest_checksums() {
        // Regression: `decode_lenient` used to zero the six *checksum-value* fields
        // (indices 1,3,5,7,9,11 — e.g. offset 352 = manifest_oldest_checksum) instead of
        // their interleaved paddings (272..432). Any nonzero manifest checksum on disk was
        // wiped, breaking `valid_checksum()` and so the superblock quorum in `inspect manifest`.
        let mut header = superblock_with_manifest(2, 4, 0xDEAD_BEEF, 6, 0xCAFE_F00D);
        header.set_checksum();
        let bytes = header.to_wire();

        let decoded = decode_lenient(&bytes).unwrap();
        assert_eq!(decoded.checksum, header.checksum);
        assert_eq!(decoded.vsr_state.checkpoint.manifest_oldest_checksum, 0xDEAD_BEEF);
        assert_eq!(decoded.vsr_state.checkpoint.manifest_newest_checksum, 0xCAFE_F00D);
        assert!(decoded.valid_checksum());
        // Round-trip must reproduce the raw wire exactly (modulo checksum/copy, which the
        // inspector reinstates).
        let restored = decoded.to_wire();
        assert_eq!(restored[34..SUPERBLOCK_HEADER_SIZE], bytes[34..SUPERBLOCK_HEADER_SIZE]);
    }

    #[test]
    fn groups_by_equality_per_field() {
        let mut copies = copies_full();
        // Distinct values in a covered field → distinct groups, in first-appearance order.
        // `cluster` at offset 48 spans bytes[48..64], which are covered by the checksum.
        copies[1][48..64]
            .copy_from_slice(&0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10u128.to_le_bytes());
        // Keep copy 1's checksum valid so its mark is `|`.
        let cluster_bytes = copies[1][CHECKSUM_IGNORE_SIZE..].to_vec();
        copies[1][0..16].copy_from_slice(&checksum(&cluster_bytes).to_le_bytes());

        let mut output = String::new();
        inspect_superblock(&mut output, &copies).unwrap();

        let lines: Vec<&str> = output.lines().collect();
        // `cluster`: copies 0, 2, 3 share the zero value; copy 1 is alone.
        assert!(lines.contains(&"|_|| cluster=0x00000000000000000000000000000000"));
        assert!(lines.contains(&"_|__ cluster=0x0102030405060708090a0b0c0d0e0f10"));
        // Unchanged fields group all copies as `||||`.
        assert!(lines.contains(&"|||| version=2"));
        // `copy` (not checksum-covered): every copy differs, four single-member groups.
        assert!(lines.contains(&"|___ copy=0"));
        assert!(lines.contains(&"_|__ copy=1"));
        assert!(lines.contains(&"__|_ copy=2"));
        assert!(lines.contains(&"___| copy=3"));
    }

    #[test]
    fn corrupt_copy_marks_group_with_x() {
        let mut copies = copies_full();
        // Flip a checksum-covered byte so this copy's checksum no longer matches.
        copies[2][48] ^= 0xFF;

        let mut output = String::new();
        inspect_superblock(&mut output, &copies).unwrap();

        let lines: Vec<&str> = output.lines().collect();
        // The corrupted copy 2 still groups by value (equal bytes), but its mark is `X`.
        assert!(lines.contains(&"||_| cluster=0x00000000000000000000000000000000"));
        assert!(lines.iter().any(|line| line.starts_with("__X_ cluster=0x")));
        // Uncorrupted fields still group all valid copies as `||||`.
        assert!(lines.contains(&"|||| version=2"));
    }

    /// A superblock checkpoint pointing at a manifest log, for driving `inspect_manifest`
    /// without the (checksum-validated) CLI quorum gate.
    fn superblock_with_manifest(
        manifest_block_count: u32,
        manifest_oldest_address: u64,
        manifest_oldest_checksum: u128,
        manifest_newest_address: u64,
        manifest_newest_checksum: u128,
    ) -> SuperBlockHeader {
        use crate::multiversion::Release;
        use crate::superblock::{SUPERBLOCK_VERSION, VSRState, VSRStateRootOptions};
        use tigerbeetle_core::constants::{MEMBERS_MAX, VIEW_HEADERS_MAX};

        let mut members = [0_u128; MEMBERS_MAX];
        members[0] = 11;
        let mut vsr_state = VSRState::root(VSRStateRootOptions {
            cluster: 7,
            replica_id: 11,
            members,
            replica_count: 1,
            release: Release::MINIMUM,
            view: 0,
        });
        let checkpoint = &mut vsr_state.checkpoint;
        checkpoint.manifest_block_count = manifest_block_count;
        checkpoint.manifest_oldest_address = manifest_oldest_address;
        checkpoint.manifest_oldest_checksum = manifest_oldest_checksum;
        checkpoint.manifest_newest_address = manifest_newest_address;
        checkpoint.manifest_newest_checksum = manifest_newest_checksum;

        SuperBlockHeader {
            checksum: 0,
            checksum_padding: 0,
            copy: 0,
            version: SUPERBLOCK_VERSION,
            release_format: Release::MINIMUM,
            sequence: 0,
            cluster: 7,
            parent: 0,
            parent_padding: 0,
            vsr_state,
            flags: 0,
            view_headers_count: 0,
            view_headers_all: [[0; SIZE]; VIEW_HEADERS_MAX as usize],
        }
    }

    /// Storage big enough to hold `put_manifest_block`'s low grid addresses (mirrors the
    /// forest test helper of the same name).
    fn manifest_storage() -> MemoryStorage {
        use tigerbeetle_core::constants::BLOCK_SIZE;
        MemoryStorage::new(crate::Zone::Grid.start() + 8 * BLOCK_SIZE as u64)
    }

    /// Write a valid manifest block (entries `(tree_id, level, event)`) into `storage`'s grid
    /// zone at `address` (grid addresses are 1-based), linked to `previous_*`, returning its
    /// checksum.
    fn put_manifest_block(
        storage: &mut MemoryStorage,
        address: u64,
        entries: &[(u16, u8, Event)],
        previous_address: u64,
        previous_checksum: u128,
    ) -> u128 {
        use crate::message_header::{Block, BlockType, SIZE as HEADER_SIZE, TypedHeader};
        use crate::schema::ManifestNode;
        use crate::storage::{Completion, WriteRequest};
        use tigerbeetle_core::constants::BLOCK_SIZE;
        use tigerbeetle_lsm::schema::manifest_node::{ENTRY_SIZE, Label, Metadata, TableInfo};

        let mut block = vec![0_u8; BLOCK_SIZE];
        let size = ManifestNode::size(u32::try_from(entries.len()).expect("test entry count fits"));

        for (i, &(tree_id, level, event)) in entries.iter().enumerate() {
            let table = TableInfo {
                key_min: [u8::try_from(i).expect("test entry index fits in u8"); 32],
                key_max: [u8::MAX; 32],
                checksum: checksum(&address.to_le_bytes()),
                address,
                snapshot_min: 1,
                snapshot_max: u64::MAX,
                value_count: 1,
                tree_id,
                label: Label { level, event },
            };
            let wire = table.to_wire();
            block[HEADER_SIZE + i * ENTRY_SIZE..HEADER_SIZE + (i + 1) * ENTRY_SIZE]
                .copy_from_slice(&wire);
        }

        let mut header = Block::default();
        header.cluster = 1;
        header.address = address;
        header.size = size;
        header.block_type_ordinal = BlockType::Manifest as u8;
        header.release = crate::multiversion::Release::MINIMUM;
        header.metadata_bytes = Metadata {
            previous_manifest_block_checksum: previous_checksum,
            previous_manifest_block_address: previous_address,
            entry_count: u32::try_from(entries.len()).expect("test entry count fits"),
        }
        .to_wire();
        block[..HEADER_SIZE].copy_from_slice(&TypedHeader::to_wire(&header));

        let mut header = Block::from_wire(block[..HEADER_SIZE].try_into().unwrap()).unwrap();
        header.set_checksum_body(&block[HEADER_SIZE..size as usize]);
        header.set_checksum();
        let checksum = header.checksum;
        block[..HEADER_SIZE].copy_from_slice(&TypedHeader::to_wire(&header));

        storage.write_sectors(WriteRequest {
            zone: crate::Zone::Grid,
            offset_in_zone: (address - 1) * BLOCK_SIZE as u64,
            buffer: block,
        });
        match storage.next_completion().unwrap() {
            Completion::Write(_) => {}
            Completion::Read(_) => unreachable!("only a write was submitted"),
        }
        checksum
    }

    /// Mirrors `inspect_manifest` output for a two-block manifest: the eager walk follows the
    /// `previous_manifest_block_*` links newest→oldest, and entry counts accumulate per event
    /// per level.
    #[test]
    fn inspect_manifest_prints_walk_and_counts() {
        let mut storage = manifest_storage();

        let old_checksum = put_manifest_block(
            &mut storage,
            3,
            &[(1, 0, Event::Insert), (2, 1, Event::Insert)],
            0,
            0,
        );
        let new_checksum = put_manifest_block(
            &mut storage,
            5,
            &[
                (1, 0, Event::Insert),
                (2, 0, Event::Insert),
                (1, 2, Event::Update),
                (3, 6, Event::Remove),
            ],
            3,
            old_checksum,
        );

        let superblock = superblock_with_manifest(2, 3, old_checksum, 5, new_checksum);

        let mut output = String::new();
        inspect_manifest(&mut output, &mut storage, &superblock).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            format!(
                "manifest_log.blocks[0]: address=5 checksum={new_checksum:032x} \
                 entries=4/{ENTRY_COUNT_MAX} insert=2,0,0,0,0,0,0 update=0,0,1,0,0,0,0 \
                 remove=0,0,0,0,0,0,1"
            )
        );
        assert_eq!(
            lines[1],
            format!(
                "manifest_log.blocks[1]: address=3 checksum={old_checksum:032x} \
                 entries=2/{ENTRY_COUNT_MAX} insert=1,1,0,0,0,0,0 update=0,0,0,0,0,0,0 \
                 remove=0,0,0,0,0,0,0"
            )
        );
    }

    /// `checkpoint.manifest_block_count == 0` (a freshly formatted data file) prints nothing.
    #[test]
    fn inspect_manifest_empty_checkpoint_prints_nothing() {
        let mut storage = manifest_storage();
        let superblock = superblock_with_manifest(0, 0, 0, 0, 0);

        let mut output = String::new();
        inspect_manifest(&mut output, &mut storage, &superblock).unwrap();
        assert!(output.is_empty());
    }

    /// A manifest reference whose block is missing prints the error line and stops the walk,
    /// without panicking.
    #[test]
    fn inspect_manifest_stops_on_missing_block() {
        let mut storage = manifest_storage();

        // Only the newest block exists at address 7; its `previous_manifest_block_*` link
        // points into the void (999), so the second walk step must print the error line.
        let new_checksum =
            put_manifest_block(&mut storage, 7, &[(1, 0, Event::Insert)], 999, 0xDEAD_BEEF);
        let superblock = superblock_with_manifest(2, 999, 0xDEAD_BEEF, 7, new_checksum);

        let mut output = String::new();
        inspect_manifest(&mut output, &mut storage, &superblock).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("manifest_log.blocks[0]: address=7 checksum="));
        // Upstream writes the failure message right after the address/checksum header, without
        // a newline, so the error shares the `blocks[1]` line.
        assert_eq!(
            lines[1],
            "manifest_log.blocks[1]: address=999 checksum=000000000000000000000000deadbeef \
             error: manifest block not found"
        );
    }

    /// `read_block` accepts a checksum-valid manifest block and rejects every corruption mode,
    /// matching upstream's `read_block` error returns.
    #[test]
    fn read_block_rejects_corruption() {
        use crate::storage::{Completion, WriteRequest};

        let mut storage = manifest_storage();
        let checksum = put_manifest_block(&mut storage, 5, &[(1, 0, Event::Insert)], 0, 0);

        // Valid block + matching expected checksum.
        assert!(read_block(&mut storage, 5, Some(checksum)).is_some());
        // Wrong expected checksum (upstream `error.WrongBlock`).
        assert!(read_block(&mut storage, 5, Some(0)).is_none());
        // Missing block → all-zero grid, not even a `Block` command frame.
        assert!(read_block(&mut storage, 99, Some(0)).is_none());
        // Block 0 is never a valid address (upstream would underflow).
        assert!(read_block(&mut storage, 0, None).is_none());

        // A garbage block that still has a plausible address: flip a checksum-covered header
        // byte and rewrite it, so `valid_checksum` fails (upstream `error.InvalidChecksum`).
        let block = read_block(&mut storage, 5, Some(checksum)).unwrap();
        let mut corrupt = block;
        corrupt[100] ^= 0xFF;
        storage.write_sectors(WriteRequest {
            zone: crate::Zone::Grid,
            offset_in_zone: (5 - 1) * tigerbeetle_core::constants::BLOCK_SIZE as u64,
            buffer: corrupt,
        });
        match storage.next_completion().unwrap() {
            Completion::Write(_) => {}
            Completion::Read(_) => unreachable!("only a write was submitted"),
        }
        assert!(read_block(&mut storage, 5, Some(checksum)).is_none());
    }

    #[test]
    fn inspect_op_at_zero() {
        let mut output = String::new();
        inspect_op(&mut output, 0).unwrap();
        // Under test-min (VSR_CHECKPOINT_OPS=20, trigger+=4, prepare_max+=8),
        // op=0 → checkpoint=0, trigger=0, prepare_max=0, op=0, checkpoint_next=19.
        // Stable sort keeps the original array order among equal-keyed points.
        assert_eq!(
            output,
            concat!(
                "checkpoint          trigger             prepare_max         op                  checkpoint_next     \n",
                "0              +0   0              +0   0              +0   0              +19  19                  \n",
            )
        );
    }

    #[test]
    fn inspect_op_at_checkpoint_boundary() {
        let mut output = String::new();
        inspect_op(&mut output, 19).unwrap();
        // checkpoint=19, trigger=23, prepare_max=31, checkpoint_next=39, op=19.
        assert_eq!(
            output,
            concat!(
                "checkpoint          op                  trigger             prepare_max         checkpoint_next     \n",
                "19             +0   19             +4   23             +8   31             +8   39                  \n",
            )
        );
    }

    #[test]
    fn inspect_op_between_checkpoints() {
        let mut output = String::new();
        inspect_op(&mut output, 21).unwrap();
        // op=21 → same checkpoint=19; sorted op values: [19, 21, 23, 31, 39].
        assert_eq!(
            output,
            concat!(
                "checkpoint          op                  trigger             prepare_max         checkpoint_next     \n",
                "19             +2   21             +2   23             +8   31             +8   39                  \n",
            )
        );
    }

    /// A `MemoryStorage` formatted like a fresh replica data file, so both WAL zones hold the
    /// per-slot `slot_header` prepares (root at slot 0, reserved elsewhere).
    fn wal_storage() -> MemoryStorage {
        use crate::replica_format::format;
        use crate::superblock::{DATA_FILE_SIZE_MIN, FormatOptions};

        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        let _ = format(
            &mut storage,
            FormatOptions {
                cluster: 0,
                release: crate::multiversion::Release::MINIMUM,
                replica: 0,
                replica_count: 1,
                view: None,
            },
        );
        storage
    }

    /// The exact `inspect_wal` line for a valid formatted slot: one `||` group, the slot
    /// header's fields, and the trailing space upstream's tuple `print_struct` writes after
    /// every value (including the last).
    fn wal_line(slot: u64) -> String {
        use crate::replica_format::slot_header;

        let header = slot_header(0, slot);
        let operation = if slot == 0 { "root" } else { "reserved" };
        format!(
            "||{slot:_>4}: checksum=0x{:032x} release=0.0.0 view=0 op={slot} size=256 \
             operation={operation} ",
            header.checksum,
        )
    }

    /// A freshly formatted WAL prints one line per slot: every slot's two header copies are
    /// byte-identical (`||`), with valid checksums and no body.
    #[test]
    fn inspect_wal_fresh_format() {
        let mut storage = wal_storage();
        let mut output = String::new();
        inspect_wal(&mut output, &mut storage).unwrap();

        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), JOURNAL_SLOT_COUNT as usize);
        for slot in 0..JOURNAL_SLOT_COUNT {
            assert_eq!(lines[slot as usize], wal_line(u64::from(slot)));
        }
    }

    /// `--slot` prints the single slot as `label=Prepare{…}` (the prepare's format fn, through
    /// `format_prepare_raw`) plus its body — `(no body)` for the root prepare of a fresh file.
    #[test]
    fn inspect_wal_slot_prints_prepare_and_body() {
        use crate::replica_format::slot_header;

        let mut storage = wal_storage();
        let mut output = String::new();
        inspect_wal_slot(&mut output, &mut storage, 0).unwrap();

        let header = slot_header(0, 0);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        // `prepare_at`-style Display (format_header) must equal the raw-bytes `format_prepare_raw`.
        assert_eq!(lines[0], format!("||={header}"));
        assert_eq!(lines[1], "(no body)");

        // A reserved slot is identical except for op/operation.
        let mut output = String::new();
        inspect_wal_slot(&mut output, &mut storage, 5).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], format!("||={}", slot_header(0, 5)));
        assert_eq!(lines[1], "(no body)");
    }

    /// A prepare whose header is corrupted in the `wal_prepares` zone only splits the slot into
    /// two groups: the valid redundant header (`|_`) and the corrupt prepare (`_X`), whose
    /// stored checksum value is still printed (it is the bytes at the front of the header).
    #[test]
    fn inspect_wal_marks_differing_headers() {
        use crate::message_header::TypedHeader;
        use crate::replica_format::slot_header;
        use crate::storage::{Completion, WriteRequest};
        use tigerbeetle_core::constants::MESSAGE_SIZE_MAX;

        let mut storage = wal_storage();

        let header = slot_header(0, 3);
        // Flip a checksum-covered byte (100 is covered by the range 16..256) in the prepare
        // copy only, so the two header copies differ and the prepare's checksum is invalid.
        // The slot's prepare buffer is one `message_size_max` frame (sector-aligned).
        let mut corrupt = vec![0_u8; MESSAGE_SIZE_MAX as usize];
        corrupt[..SIZE].copy_from_slice(&header.to_wire());
        corrupt[100] ^= 0xFF;
        storage.write_sectors(WriteRequest {
            zone: crate::Zone::WalPrepares,
            offset_in_zone: 3 * u64::from(MESSAGE_SIZE_MAX),
            buffer: corrupt,
        });
        let Completion::Write(_) = storage.next_completion().unwrap() else {
            unreachable!("write was expected");
        };

        let mut output = String::new();
        inspect_wal(&mut output, &mut storage).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        // Slot 3 gains a second (group) line.
        assert_eq!(lines.len(), JOURNAL_SLOT_COUNT as usize + 1);
        for slot in 0..JOURNAL_SLOT_COUNT as usize {
            if slot != 3 {
                let expected = wal_line(slot as u64);
                let index = if slot < 3 { slot } else { slot + 1 };
                assert_eq!(lines[index], expected);
            }
        }
        // `|_` is the valid redundant (wal_headers) copy; `_X` is the corrupt prepare copy.
        assert_eq!(
            lines[3],
            format!(
                "|____3: checksum=0x{:032x} release=0.0.0 view=0 op=3 size=256 \
                 operation=reserved ",
                header.checksum,
            )
        );
        assert_eq!(
            lines[4],
            format!(
                "_X___3: checksum=0x{:032x} release=0.0.0 view=0 op=3 size=256 \
                 operation=reserved ",
                header.checksum,
            )
        );
    }

    /// A `--slot` whose header's size is out of range (in both copies) groups as one `||`/`XX`
    /// line and reports both the unexpected body size and the invalid body checksum error.
    #[test]
    fn inspect_wal_slot_rejects_invalid_body_size() {
        use crate::message_header::TypedHeader;
        use crate::replica_format::slot_header;
        use crate::storage::{Completion, WriteRequest};
        use tigerbeetle_core::constants::MESSAGE_SIZE_MAX;

        let mut storage = wal_storage();

        // Corrupt the size of slot 5 (bytes 96..100, covered by the checksum) so the body size
        // is out of the prepare buffer's range. Write the identical header into both zones so
        // the slot still groups as one `||` line.
        let corrupt_header = {
            let mut header = slot_header(0, 5).to_wire();
            header[96..100].copy_from_slice(&u32::MAX.to_le_bytes());
            header
        };

        let mut prepare = vec![0_u8; MESSAGE_SIZE_MAX as usize];
        prepare[..SIZE].copy_from_slice(&corrupt_header);
        storage.write_sectors(WriteRequest {
            zone: crate::Zone::WalPrepares,
            offset_in_zone: 5 * u64::from(MESSAGE_SIZE_MAX),
            buffer: prepare,
        });
        let Completion::Write(_) = storage.next_completion().unwrap() else {
            unreachable!("write was expected");
        };

        // The slot-5 header sits inside the headers zone's first sector; reread the whole zone,
        // swap in the corrupt header, and rewrite.
        let mut headers =
            read_buffer(&mut storage, crate::Zone::WalHeaders, 0, JOURNAL_SIZE_HEADERS);
        headers[5 * SIZE..6 * SIZE].copy_from_slice(&corrupt_header);
        storage.write_sectors(WriteRequest {
            zone: crate::Zone::WalHeaders,
            offset_in_zone: 0,
            buffer: headers,
        });
        let Completion::Write(_) = storage.next_completion().unwrap() else {
            unreachable!("write was expected");
        };

        let mut output = String::new();
        inspect_wal_slot(&mut output, &mut storage, 5).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 3);
        // Equal copies → `||`; the size change breaks the checksum → `X`. The `Prepare`
        // Display (format_header) must match the raw-bytes `format_prepare_raw`.
        assert_eq!(
            lines[0],
            format!("XX={}", crate::message_header::Prepare::from_wire(&corrupt_header).unwrap())
        );
        assert_eq!(lines[1], "error: unexpected body size=4294967295, @sizeOf(Event)=0");
        assert_eq!(lines[2], "error: invalid prepare body!");
    }
}
