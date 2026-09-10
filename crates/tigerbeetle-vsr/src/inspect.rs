//! Port of upstream `src/tigerbeetle/inspect.zig`'s `inspect superblock` exporter.
//!
//! Prints the four stored copies of the superblock (from a data file's superblock zone),
//! grouping the copies by the value of each field, exactly as upstream does in
//! `inspect_superblock`/`print_struct`.
//!
//! Upstream relies on Zig's comptime `std.meta.fields(SuperBlockHeader)` to enumerate the
//! struct layout; Rust has no reflection over struct fields, so the layout is spelled out in
//! the [`Field`] trees below (mirroring superblock.zig's declaration order) and pinned against
//! `superblock.rs`'s offset constants by unit tests.

use core::fmt;

use tigerbeetle_core::checksum::checksum;
use tigerbeetle_core::constants::{MEMBERS_MAX, SECTOR_SIZE, SUPERBLOCK_COPIES, VIEW_HEADERS_MAX};

use crate::message_header::{SIZE, format_prepare_raw};
use crate::multiversion::Release;
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
    for field in CHECKPOINT_STATE_FIELDS.iter().take(12).skip(1).step_by(2) {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

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
}
