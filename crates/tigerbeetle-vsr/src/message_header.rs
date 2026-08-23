//! Network message, prepare, and grid block header.
//!
//! We reuse the same header for both so that prepare messages from the primary can simply be
//! journalled as is by the backups without requiring any further modification.
//!
//! Port of `src/vsr/message_header.zig` (base `Header` frame; per-command typed headers land
//! incrementally as their subsystems are ported).
//!
//! DEVIATION: upstream is an `extern struct` reinterpreted via `bytesAsValue`; this port keeps a
//! `#[repr(C)]` struct for in-memory use and converts to/from wire bytes explicitly
//! (`to_wire`/`from_wire`, little-endian). This avoids byte-punning (`unsafe`) while producing
//! byte-identical on-disk/on-wire layouts on upstream's little-endian targets.
//!
//! TODO(port): src/vsr/message_header.zig typed headers (Ping, Pong, Request, Prepare, Reply, …),
//! their `invalid_header()` checks, `peer_type()`, `format`.

#![allow(clippy::doc_markdown)]
// doc comments are ported verbatim from upstream
// SIZE (256) and body lengths are bounded far below u32::MAX; truncation cannot occur.
#![allow(clippy::cast_possible_truncation)]

use tigerbeetle_core::checksum::checksum;
use tigerbeetle_core::{constants, stdx};

use crate::VERSION;
use crate::command::Command;
use crate::multiversion::Release;

pub const SIZE: usize = 256;

/// The vsr checksum of an empty body (upstream: `vsr.checksum(&.{})`).
#[must_use]
pub fn checksum_body_empty() -> u128 {
    checksum(&[])
}

/// The base header frame. Field order, sizes, and alignment match upstream exactly
/// (size 256, align 16, no padding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Header {
    /// A checksum covering only the remainder of this header.
    /// This allows the header to be trusted without having to recv() or read() the associated
    /// body. This checksum is enough to uniquely identify a network message or prepare.
    pub checksum: u128,

    // TODO(zig): When Zig supports u256 in extern-structs, merge this into `checksum`.
    pub checksum_padding: u128,

    /// A checksum covering only the associated body after this header.
    pub checksum_body: u128,

    // TODO(zig): When Zig supports u256 in extern-structs, merge this into `checksum_body`.
    pub checksum_body_padding: u128,

    /// Reserved for future use by AEAD.
    pub nonce_reserved: u128,

    /// The cluster number binds intention into the header, so that a client or replica can
    /// indicate the cluster it believes it is speaking to, instead of accidentally talking to the
    /// wrong cluster (for example, staging vs production).
    pub cluster: u128,

    /// The size of the Header structure (always), plus any associated body.
    pub size: u32,

    /// The cluster reconfiguration epoch number (for future use).
    pub epoch: u32,

    /// Every message sent from one replica to another contains the sending replica's current view.
    /// A `u32` allows for a minimum lifetime of 136 years at a rate of one view change per second.
    pub view: u32,

    /// The release version set by the state machine.
    /// (This field is not set for all message types.)
    pub release: Release,

    /// The version of the protocol implementation that originated this message.
    pub protocol: u16,

    /// The Viewstamped Replication protocol command for this message.
    pub command: Command,

    /// The index of the replica in the cluster configuration array that authored this message.
    /// This identifies only the ultimate author because messages may be forwarded amongst
    /// replicas.
    pub replica: u8,

    /// Reserved for future use by the header frame (i.e. to be shared by all message types).
    pub reserved_frame: [u8; 12],

    /// This data's schema is different depending on the `Header.command`.
    /// (No default value – `Header`s should not be constructed directly.)
    pub reserved_command: [u8; 128],
}

// Upstream comptime assertions:
const _: () = assert!(size_of::<Header>() == 256);
const _: () = assert!(align_of::<Header>() == 16);
const _: () = assert!(std::mem::offset_of!(Header, reserved_frame) == 116);
const _: () = assert!(std::mem::offset_of!(Header, reserved_command) == 128);
const _: () = assert!(std::mem::offset_of!(Header, reserved_command) % 32 == 0); // upstream: % sizeOf(u256)

impl Header {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            checksum: 0,
            checksum_padding: 0,
            checksum_body: 0,
            checksum_body_padding: 0,
            nonce_reserved: 0,
            cluster: 0,
            size: SIZE as u32,
            epoch: 0,
            view: 0,
            release: Release::ZERO,
            protocol: VERSION,
            command: Command::Reserved,
            replica: 0,
            reserved_frame: [0; 12],
            reserved_command: [0; 128],
        }
    }

    /// Serializes to the on-disk/on-wire byte representation (little-endian fields).
    #[must_use]
    pub fn to_wire(&self) -> [u8; SIZE] {
        let mut out = [0u8; SIZE];
        put_u128(&mut out, 0, self.checksum);
        put_u128(&mut out, 16, self.checksum_padding);
        put_u128(&mut out, 32, self.checksum_body);
        put_u128(&mut out, 48, self.checksum_body_padding);
        put_u128(&mut out, 64, self.nonce_reserved);
        put_u128(&mut out, 80, self.cluster);
        put_u32(&mut out, 96, self.size);
        put_u32(&mut out, 100, self.epoch);
        put_u32(&mut out, 104, self.view);
        put_u32(&mut out, 108, self.release.value);
        put_u16(&mut out, 112, self.protocol);
        out[114] = self.command as u8;
        out[115] = self.replica;
        out[116..128].copy_from_slice(&self.reserved_frame);
        out[128..256].copy_from_slice(&self.reserved_command);
        out
    }

    /// Parses the wire representation. Unlike upstream's unchecked casts, this rejects bytes that
    /// do not decode into the frame (e.g. an unknown command) by returning `None`.
    #[must_use]
    pub fn from_wire(bytes: &[u8; SIZE]) -> Option<Self> {
        Some(Self {
            checksum: get_u128(bytes, 0),
            checksum_padding: get_u128(bytes, 16),
            checksum_body: get_u128(bytes, 32),
            checksum_body_padding: get_u128(bytes, 48),
            nonce_reserved: get_u128(bytes, 64),
            cluster: get_u128(bytes, 80),
            size: get_u32(bytes, 96),
            epoch: get_u32(bytes, 100),
            view: get_u32(bytes, 104),
            release: Release { value: get_u32(bytes, 108) },
            protocol: get_u16(bytes, 112),
            command: Command::from_u8(bytes[114])?,
            replica: bytes[115],
            reserved_frame: bytes[116..128].try_into().ok()?,
            reserved_command: bytes[128..256].try_into().ok()?,
        })
    }

    #[must_use]
    pub fn calculate_checksum(&self) -> u128 {
        let bytes = self.to_wire();
        checksum(&bytes[16..])
    }

    /// # Panics
    /// Panics if `self.size != SIZE + body.len()` (upstream asserts the same).
    #[must_use]
    pub fn calculate_checksum_body(&self, body: &[u8]) -> u128 {
        assert_eq!(self.size, SIZE as u32 + body.len() as u32);
        checksum(body)
    }

    /// This must be called only after set_checksum_body() so that checksum_body is also covered:
    pub fn set_checksum(&mut self) {
        self.checksum = self.calculate_checksum();
    }

    pub fn set_checksum_body(&mut self, body: &[u8]) {
        self.checksum_body = self.calculate_checksum_body(body);
    }

    #[must_use]
    pub fn valid_checksum(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }

    #[must_use]
    pub fn valid_checksum_body(&self, body: &[u8]) -> bool {
        self.checksum_body == self.calculate_checksum_body(body)
    }
}

impl WireField for crate::Operation {
    const WIRE_LEN: usize = 1;
    fn put_wire(self, dst: &mut [u8; SIZE], offset: usize) {
        u8::put_wire(self.0, dst, offset);
    }
    fn get_wire(src: &[u8; SIZE], offset: usize) -> Self {
        crate::Operation(u8::get_wire(src, offset))
    }
}

/// Port of `Header.Eviction.Reason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Reason {
    Reserved = 0,
    NoSession = 1,
    ClientReleaseTooLow = 2,
    ClientReleaseTooHigh = 3,
    InvalidRequestOperation = 4,
    InvalidRequestBody = 5,
    InvalidRequestBodySize = 6,
    SessionTooLow = 7,
    SessionReleaseMismatch = 8,
}

const REASONS_ALL: [Reason; 9] = [
    Reason::Reserved,
    Reason::NoSession,
    Reason::ClientReleaseTooLow,
    Reason::ClientReleaseTooHigh,
    Reason::InvalidRequestOperation,
    Reason::InvalidRequestBody,
    Reason::InvalidRequestBodySize,
    Reason::SessionTooLow,
    Reason::SessionReleaseMismatch,
];

impl DefaultField for crate::Operation {
    const DEFAULT: Self = crate::Operation::RESERVED;
}
impl DefaultField for Reason {
    const DEFAULT: Self = Reason::Reserved;
}

fn get_u128(src: &[u8; SIZE], offset: usize) -> u128 {
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&src[offset..offset + 16]);
    u128::from_le_bytes(buf)
}

fn get_u32(src: &[u8; SIZE], offset: usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&src[offset..offset + 4]);
    u32::from_le_bytes(buf)
}

fn get_u16(src: &[u8; SIZE], offset: usize) -> u16 {
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&src[offset..offset + 2]);
    u16::from_le_bytes(buf)
}

fn put_u128(dst: &mut [u8; SIZE], offset: usize, value: u128) {
    dst[offset..offset + 16].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(dst: &mut [u8; SIZE], offset: usize, value: u32) {
    dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u16(dst: &mut [u8; SIZE], offset: usize, value: u16) {
    dst[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Little-endian wire codec for a single typed-header field.
trait WireField: Copy {
    const WIRE_LEN: usize;
    fn put_wire(self, dst: &mut [u8; SIZE], offset: usize);
    fn get_wire(src: &[u8; SIZE], offset: usize) -> Self;
}

macro_rules! impl_wire_int {
    ($($t:ty),+) => {$(
        impl WireField for $t {
            const WIRE_LEN: usize = size_of::<$t>();
            fn put_wire(self, dst: &mut [u8; SIZE], offset: usize) {
                dst[offset..offset + Self::WIRE_LEN].copy_from_slice(&self.to_le_bytes());
            }
            fn get_wire(src: &[u8; SIZE], offset: usize) -> Self {
                let mut buf = [0u8; Self::WIRE_LEN];
                buf.copy_from_slice(&src[offset..offset + Self::WIRE_LEN]);
                <$t>::from_le_bytes(buf)
            }
        }
    )+};
}

impl_wire_int!(u8, u16, u32, u64, u128);

impl WireField for Release {
    const WIRE_LEN: usize = 4;
    fn put_wire(self, dst: &mut [u8; SIZE], offset: usize) {
        self.value.put_wire(dst, offset);
    }
    fn get_wire(src: &[u8; SIZE], offset: usize) -> Self {
        Release { value: u32::get_wire(src, offset) }
    }
}

impl<const N: usize> WireField for [u8; N] {
    const WIRE_LEN: usize = N;
    fn put_wire(self, dst: &mut [u8; SIZE], offset: usize) {
        dst[offset..offset + N].copy_from_slice(&self);
    }
    fn get_wire(src: &[u8; SIZE], offset: usize) -> Self {
        let mut buf = [0u8; N];
        buf.copy_from_slice(&src[offset..offset + N]);
        buf
    }
}

/// Zero default for a typed-header field (upstream uses per-field `= 0` defaults).
trait DefaultField {
    const DEFAULT: Self;
}
impl DefaultField for u8 {
    const DEFAULT: Self = 0;
}
impl DefaultField for u16 {
    const DEFAULT: Self = 0;
}
impl DefaultField for u32 {
    const DEFAULT: Self = 0;
}
impl DefaultField for u64 {
    const DEFAULT: Self = 0;
}
impl DefaultField for u128 {
    const DEFAULT: Self = 0;
}
impl<const N: usize> DefaultField for [u8; N] {
    const DEFAULT: Self = [0; N];
}
impl DefaultField for Release {
    const DEFAULT: Self = Release::ZERO;
}

fn put_arr12(dst: &mut [u8; SIZE], offset: usize, value: [u8; 12]) {
    dst[offset..offset + 12].copy_from_slice(&value);
}

fn get_arr12(src: &[u8; SIZE], offset: usize) -> [u8; 12] {
    let mut buf = [0u8; 12];
    buf.copy_from_slice(&src[offset..offset + 12]);
    buf
}

/// Shared behavior of the per-command header types (upstream `Header.Ping`, `Header.Pong`, …).
///
/// Upstream shares these functions by reinterpreting the command-specific struct's bytes as the
/// base `Header`; here every generated type carries the full frame fields and shared logic runs
/// over `to_wire()` (see module-level DEVIATION).
pub trait TypedHeader: Copy + PartialEq + Eq + std::fmt::Debug {
    /// The command whose schema this header type is (upstream `Type(command)`).
    const COMMAND: Command;

    /// Serializes to the on-disk/on-wire 256-byte representation.
    fn to_wire(&self) -> [u8; SIZE];

    /// Parses the wire representation; rejects frames whose command does not match
    /// [`Self::COMMAND`] (mirrors upstream's guarded `into()/into_const()` casts).
    fn from_wire(bytes: &[u8; SIZE]) -> Option<Self>;

    /// Per-command validation (upstream `invalid_header()`); assumes checksums already verified.
    fn invalid_header(&self) -> Option<&'static str>;

    #[must_use]
    fn calculate_checksum(&self) -> u128 {
        let wire = self.to_wire();
        checksum(&wire[16..])
    }

    /// # Panics
    /// Panics if `size != SIZE + body.len()` (upstream asserts the same).
    #[must_use]
    fn calculate_checksum_body(&self, body: &[u8]) -> u128 {
        assert_eq!(self.size(), SIZE as u32 + body.len() as u32);
        checksum(body)
    }

    /// This must be called only after set_checksum_body() so that checksum_body is also covered:
    fn set_checksum(&mut self);

    fn set_checksum_body(&mut self, body: &[u8]);

    #[must_use]
    fn valid_checksum(&self) -> bool;

    #[must_use]
    fn valid_checksum_body(&self, body: &[u8]) -> bool;

    /// Returns null if all fields are set correctly according to the command, or else a warning.
    /// This does not verify that checksum is valid, and expects that this has already been done.
    ///
    /// Unlike upstream, this skips the base-frame checks — use [`Header::invalid`] for those.
    fn invalid(&self) -> Option<&'static str> {
        self.invalid_header()
    }

    /// The base `Header` view of this message (upstream `frame_const()`).
    #[must_use]
    fn frame(&self) -> Header {
        match Header::from_wire(&self.to_wire()) {
            Some(header) => header,
            None => unreachable!("frame decodes by construction"),
        }
    }

    fn size(&self) -> u32;
    fn set_size(&mut self, size: u32);
}

/// Defines one per-command header type.
///
/// The generated struct starts with the exact base-`Header` frame fields (same order/offsets),
/// then the command-specific fields at their explicit byte offsets — mirroring upstream, where
/// each `Header.<Command>` repeats the frame. `to_wire`/`from_wire` encode field-by-field (LE).
/// The two checksum fields are carried explicitly; the three always-zero padding/nonce fields
/// stay implicit.
macro_rules! typed_header {
    (
        $(#[$struct_meta:meta])*
        $vis:vis struct $name:ident : $command:expr,
        {
            $(
                $(#[$field_meta:meta])*
                $offset:literal $field:vis $field_name:ident : $field_ty:ty,
            )*
        }
        invalid_header(|$self_id:ident| $invalid_body:block)
    ) => {
        $(#[$struct_meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            pub checksum: u128,
            pub checksum_padding: u128,
            pub checksum_body: u128,
            pub checksum_body_padding: u128,
            pub nonce_reserved: u128,
            pub cluster: u128,
            pub size: u32,
            pub epoch: u32,
            pub view: u32,
            pub release: Release,
            pub protocol: u16,
            pub command: Command,
            pub replica: u8,
            pub reserved_frame: [u8; 12],
            $(
                $(#[$field_meta])*
                $field $field_name: $field_ty,
            )*
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    checksum: 0,
                    checksum_padding: 0,
                    checksum_body: 0,
                    checksum_body_padding: 0,
                    nonce_reserved: 0,
                    cluster: 0,
                    size: SIZE as u32,
                    epoch: 0,
                    view: 0,
                    release: Release::ZERO,
                    protocol: VERSION,
                    command: $command,
                    replica: 0,
                    reserved_frame: [0; 12],
                    $(
                        $field_name: <$field_ty as DefaultField>::DEFAULT,
                    )*
                }
            }
        }

        impl $name {
            #[must_use]
            pub fn checksum(&self) -> u128 { self.checksum }
            #[must_use]
            pub fn checksum_body_field(&self) -> u128 { self.checksum_body }
        }

        impl TypedHeader for $name {
            const COMMAND: Command = $command;

            fn to_wire(&self) -> [u8; SIZE] {
                let mut out = [0u8; SIZE];
                put_u128(&mut out, 0, self.checksum);
                put_u128(&mut out, 16, self.checksum_padding);
                put_u128(&mut out, 32, self.checksum_body);
                put_u128(&mut out, 48, self.checksum_body_padding);
                put_u128(&mut out, 64, self.nonce_reserved);
                put_u128(&mut out, 80, self.cluster);
                put_u32(&mut out, 96, self.size);
                put_u32(&mut out, 100, self.epoch);
                put_u32(&mut out, 104, self.view);
                put_u32(&mut out, 108, self.release.value);
                put_u16(&mut out, 112, self.protocol);
                out[114] = self.command as u8;
                out[115] = self.replica;
                put_arr12(&mut out, 116, self.reserved_frame);
                $(
                    <$field_ty as WireField>::put_wire(self.$field_name, &mut out, $offset);
                )*
                out
            }

            fn from_wire(bytes: &[u8; SIZE]) -> Option<Self> {
                if bytes[114] != <$name as TypedHeader>::COMMAND as u8 {
                    return None;
                }
                Some(Self {
                    checksum: get_u128(bytes, 0),
                    checksum_padding: get_u128(bytes, 16),
                    checksum_body: get_u128(bytes, 32),
                    checksum_body_padding: get_u128(bytes, 48),
                    nonce_reserved: get_u128(bytes, 64),
                    cluster: get_u128(bytes, 80),
                    size: get_u32(bytes, 96),
                    epoch: get_u32(bytes, 100),
                    view: get_u32(bytes, 104),
                    release: Release { value: get_u32(bytes, 108) },
                    protocol: get_u16(bytes, 112),
                    command: $command,
                    replica: bytes[115],
                    reserved_frame: get_arr12(bytes, 116),
                    $(
                        $field_name: <$field_ty as WireField>::get_wire(bytes, $offset),
                    )*
                })
            }

            fn invalid_header(&self) -> Option<&'static str> {
                let $self_id = self;
                assert_eq!($self_id.command, <$name as TypedHeader>::COMMAND);
                $invalid_body
            }

            fn set_checksum(&mut self) { self.checksum = self.calculate_checksum(); }

            fn set_checksum_body(&mut self, body: &[u8]) {
                self.checksum_body = self.calculate_checksum_body(body);
            }

            fn valid_checksum(&self) -> bool { self.checksum == self.calculate_checksum() }

            fn valid_checksum_body(&self, body: &[u8]) -> bool {
                self.checksum_body == self.calculate_checksum_body(body)
            }

            fn size(&self) -> u32 { self.size }

            fn set_size(&mut self, size: u32) { self.size = size; }
        }
    };
}

impl Header {
    /// Returns null if all fields are set correctly according to the command, or else a warning.
    /// This does not verify that checksum is valid, and expects that this has already been done.
    #[must_use]
    pub fn invalid(&self) -> Option<&'static str> {
        if self.checksum_padding != 0 {
            return Some("checksum_padding != 0");
        }
        if self.checksum_body_padding != 0 {
            return Some("checksum_body_padding != 0");
        }
        if self.nonce_reserved != 0 {
            return Some("nonce_reserved != 0");
        }
        if self.size < SIZE as u32 {
            return Some("size < @sizeOf(Header)");
        }
        if self.size > constants::MESSAGE_SIZE_MAX {
            return Some("size > message_size_max");
        }
        if self.epoch != 0 {
            return Some("epoch != 0");
        }
        if !stdx::zeroed(&self.reserved_frame) {
            return Some("reserved_frame != 0");
        }

        if self.command == Command::Block {
            if self.protocol > VERSION {
                return Some("block: protocol > Version");
            }
        } else if self.protocol != VERSION {
            return Some("protocol != Version");
        }

        match self.command {
            Command::Reserved => Some("reserved is invalid"),
            Command::Ping => self.into_typed::<Ping>().and_then(|h| h.invalid_header()),
            Command::Pong => self.into_typed::<Pong>().and_then(|h| h.invalid_header()),
            Command::PingClient => self.into_typed::<PingClient>().and_then(|h| h.invalid_header()),
            Command::PongClient => self.into_typed::<PongClient>().and_then(|h| h.invalid_header()),
            Command::Request => self.into_typed::<Request>().and_then(|h| h.invalid_header()),
            Command::Prepare => self.into_typed::<Prepare>().and_then(|h| h.invalid_header()),
            Command::PrepareOk => self.into_typed::<PrepareOk>().and_then(|h| h.invalid_header()),
            Command::Reply => self.into_typed::<Reply>().and_then(|h| h.invalid_header()),
            Command::Commit => self.into_typed::<Commit>().and_then(|h| h.invalid_header()),
            Command::ExitView => self.into_typed::<ExitView>().and_then(|h| h.invalid_header()),
            Command::JoinView => self.into_typed::<JoinView>().and_then(|h| h.invalid_header()),
            Command::GetView => self.into_typed::<GetView>().and_then(|h| h.invalid_header()),
            Command::GetHeaders => self.into_typed::<GetHeaders>().and_then(|h| h.invalid_header()),
            Command::GetPrepare => self.into_typed::<GetPrepare>().and_then(|h| h.invalid_header()),
            Command::GetReply => self.into_typed::<GetReply>().and_then(|h| h.invalid_header()),
            Command::Headers => self.into_typed::<Headers>().and_then(|h| h.invalid_header()),
            Command::Eviction => self.into_typed::<Eviction>().and_then(|h| h.invalid_header()),
            Command::GetBlocks => self.into_typed::<GetBlocks>().and_then(|h| h.invalid_header()),
            Command::View => self.into_typed::<View>().and_then(|h| h.invalid_header()),
            Command::Block => self.into_typed::<Block>().and_then(|h| h.invalid_header()),
            Command::Deprecated12
            | Command::Deprecated21
            | Command::Deprecated22
            | Command::Deprecated23 => Some("deprecated message type"),
        }
    }

    /// Upstream `into_const(command)`: views this frame's bytes as the given command's header
    /// type, or `None` when the commands do not match.
    #[must_use]
    pub fn into_typed<T: TypedHeader>(&self) -> Option<T> {
        if self.command != T::COMMAND {
            return None;
        }
        T::from_wire(&self.to_wire())
    }
}

typed_header! {
    pub struct Ping : Command::Ping,
    {
        /// Current checkpoint id.
        128 pub checkpoint_id: u128,
        /// Current checkpoint op.
        144 pub checkpoint_op: u64,
        152 pub ping_timestamp_monotonic: u64,
        160 pub release_count: u16,
        162 pub reserved: [u8; 94],
    }
    invalid_header(|ping| {
        // NB: unlike every other message, pings and pongs use on disk view, rather than in-memory
        // view, to avoid disrupting clock synchronization while the view is being updated.
        let size_expected = SIZE as u32 + size_of::<Release>() as u32 * constants::VSR_RELEASES_MAX;
        if ping.size != size_expected {
            return Some("size != @sizeOf(Header) + @sizeOf(vsr.Release) * constants.vsr_releases_max");
        }
        if ping.release.value == 0 {
            return Some("release == 0");
        }
        if !crate::checkpoint::valid(ping.checkpoint_op) {
            return Some("checkpoint_op invalid");
        }
        if ping.ping_timestamp_monotonic == 0 {
            return Some("ping_timestamp_monotonic != expected");
        }
        if ping.release_count == 0 {
            return Some("release_count == 0");
        }
        if ping.release_count > constants::VSR_RELEASES_MAX as u16 {
            return Some("release_count > vsr_releases_max");
        }
        if !stdx::zeroed(&ping.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct Pong : Command::Pong,
    {
        128 pub ping_timestamp_monotonic: u64,
        136 pub pong_timestamp_wall: u64,
        144 pub reserved: [u8; 112],
    }
    invalid_header(|pong| {
        if pong.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if pong.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if pong.release.value == 0 {
            return Some("release == 0");
        }
        if pong.ping_timestamp_monotonic == 0 {
            return Some("ping_timestamp_monotonic == 0");
        }
        if pong.pong_timestamp_wall == 0 {
            return Some("pong_timestamp_wall == 0");
        }
        if !stdx::zeroed(&pong.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct PingClient : Command::PingClient,
    {
        128 pub client: u128,
        144 pub ping_timestamp_monotonic: u64,
        // NB: Introduced in 0.17.6, and was implicitly 0 before that.
        152 pub session: u64,
        160 pub reserved: [u8; 96],
    }
    invalid_header(|ping| {
        if ping.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if ping.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if ping.release.value == 0 {
            return Some("release == 0");
        }
        if ping.replica != 0 {
            return Some("replica != 0");
        }
        if ping.view != 0 {
            return Some("view != 0");
        }
        if ping.client == 0 {
            return Some("client == 0");
        }
        if !stdx::zeroed(&ping.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct PongClient : Command::PongClient,
    {
        128 pub ping_timestamp_monotonic: u64,
        136 pub reserved: [u8; 120],
    }
    invalid_header(|pong| {
        if pong.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if pong.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if pong.release.value == 0 {
            return Some("release == 0");
        }
        if !stdx::zeroed(&pong.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct Request : Command::Request,
    {
        /// Clients hash-chain their requests to verify linearizability:
        /// - A session's first request (operation=register) sets `parent=0`.
        /// - A session's subsequent requests (operation≠register) set `parent` to the checksum of
        ///   the preceding reply.
        128 pub parent: u128,
        144 pub parent_padding: u128,
        /// Each client process generates a unique, random and ephemeral client ID at
        /// initialization. The client ID identifies connections made by the client to the cluster
        /// for the sake of routing messages back to the client.
        ///
        /// With the client ID in hand, the client then registers a monotonically increasing session
        /// number (committed through the cluster) to allow the client's session to be evicted
        /// safely from the client table if too many concurrent clients cause the client table to
        /// overflow. The monotonically increasing session number prevents duplicate client requests
        /// from being replayed.
        ///
        /// The problem of routing is therefore solved by the 128-bit client ID, and the problem of
        /// detecting whether a session has been evicted is solved by the session number.
        160 pub client: u128,
        /// When operation=register, this is zero.
        /// When operation≠register, this is the commit number of register.
        176 pub session: u64,
        /// Only nonzero during AOF recovery.
        /// TODO: Use this for bulk-import to state machine?
        184 pub timestamp: u64,
        /// Each request is given a number by the client and later requests must have larger numbers
        /// than earlier ones. The request number is used by the replicas to avoid running requests
        /// more than once; it is also used by the client to discard duplicate replies to its
        /// requests.
        ///
        /// A client is allowed to have at most one request inflight at a time.
        192 pub request: u32,
        196 pub operation: crate::Operation,
        197 previous_request_latency_padding: [u8; 3],
        /// Microsecond (0.17.0+) / Nanosecond interval measuring the time between when the client
        /// first began to construct the previous request's body and the time that the client
        /// received the corresponding reply.
        200 pub previous_request_latency: u32,
        204 pub reserved: [u8; 52],
    }
    invalid_header(|request| {
        if request.release.value == 0 {
            return Some("release == 0");
        }
        if request.parent_padding != 0 {
            return Some("parent_padding != 0");
        }
        if request.operation == crate::Operation::RESERVED {
            return Some("operation == .reserved");
        } else if request.operation == crate::Operation::ROOT {
            return Some("operation == .root");
        } else if request.operation == crate::Operation::REGISTER {
            // The first request a client makes must be to register with the cluster:
            if request.replica != 0 {
                return Some("register: replica != 0");
            }
            if request.client == 0 {
                return Some("register: client == 0");
            }
            if request.parent != 0 {
                return Some("register: parent != 0");
            }
            if request.session != 0 {
                return Some("register: session != 0");
            }
            if request.request != 0 {
                return Some("register: request != 0");
            }
            // Support `register` requests without the body to correctly
            // reply with `client_release_too_low` for clients <= v0.15.3.
            let size_with_body = SIZE as u32 + size_of::<crate::RegisterRequest>() as u32;
            if request.size != SIZE as u32 && request.size != size_with_body {
                return Some("register: size != @sizeOf(Header) [+ @sizeOf(vsr.RegisterRequest)]");
            }
        } else if request.operation == crate::Operation::PULSE {
            // These requests don't originate from a real client or session.
            if request.client != 0 {
                return Some("pulse: client != 0");
            }
            if request.parent != 0 {
                return Some("pulse: parent != 0");
            }
            if request.session != 0 {
                return Some("pulse: session != 0");
            }
            if request.request != 0 {
                return Some("pulse: request != 0");
            }
            if request.size != SIZE as u32 {
                return Some("pulse: size != @sizeOf(Header)");
            }
        } else if request.operation == crate::Operation::UPGRADE {
            // These requests don't originate from a real client or session.
            if request.client != 0 {
                return Some("upgrade: client != 0");
            }
            if request.parent != 0 {
                return Some("upgrade: parent != 0");
            }
            if request.session != 0 {
                return Some("upgrade: session != 0");
            }
            if request.request != 0 {
                return Some("upgrade: request != 0");
            }
            let size_with_body = SIZE as u32 + size_of::<crate::UpgradeRequest>() as u32;
            if request.size != size_with_body {
                return Some("upgrade: size != @sizeOf(Header) + @sizeOf(vsr.UpgradeRequest)");
            }
        } else {
            if request.operation == crate::Operation::RECONFIGURE {
                let size_with_body =
                    SIZE as u32 + size_of::<crate::ReconfigurationRequest>() as u32;
                if request.size != size_with_body {
                    return Some("size != @sizeOf(Header) + @sizeOf(ReconfigurationRequest)");
                }
            } else if request.operation == crate::Operation::NOOP {
                if request.size != SIZE as u32 {
                    return Some("size != @sizeOf(Header)");
                }
            } else if request.operation.vsr_reserved() {
                return Some("operation is reserved");
            }
            if request.replica != 0 {
                return Some("replica != 0");
            }
            if request.client == 0 {
                return Some("client == 0");
            }
            // Thereafter, the client must provide the session number:
            // These requests should set `parent` to the `checksum` of the previous reply.
            if request.session == 0 {
                return Some("session == 0");
            }
            if request.request == 0 {
                return Some("request == 0");
            }
            // The Replica is responsible for checking the `Operation` is a valid variant –
            // the check requires the StateMachine type.
        }
        if !stdx::zeroed(&request.previous_request_latency_padding) {
            return Some("padding != 0");
        }
        if !stdx::zeroed(&request.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct Prepare : Command::Prepare,
    {
        /// A backpointer to the previous prepare checksum for hash chain verification.
        /// This provides a strong guarantee for linearizability across our distributed log
        /// of prepares.
        ///
        /// This may also be used as the initialization vector for AEAD encryption at rest, provided
        /// that the primary ratchets the encryption key every view change to ensure that prepares
        /// reordered through a view change never repeat the same IV for the same encryption key.
        128 pub parent: u128,
        144 pub parent_padding: u128,
        /// The checksum of the client's request.
        160 pub request_checksum: u128,
        176 pub request_checksum_padding: u128,
        /// The id of the checkpoint where:
        ///
        ///   prepare.op > checkpoint_op
        ///   prepare.op ≤ checkpoint_after(checkpoint_op)
        ///
        /// The purpose of including the checkpoint id is to strictly bound the number of commits
        /// that it may take to discover a divergent replica. If a replica diverges, then that
        /// divergence will be discovered *at latest* when the divergent replica attempts to commit
        /// the first op after the next checkpoint.
        192 pub checkpoint_id: u128,
        208 pub client: u128,
        /// The op number of the latest prepare that may or may not yet be committed. Uncommitted
        /// ops may be replaced by different ops if they do not survive through a view change.
        224 pub op: u64,
        /// The commit number of the latest committed prepare. Committed ops are immutable.
        232 pub commit: u64,
        /// The primary's state machine `prepare_timestamp`.
        /// For `create_accounts` and `create_transfers` this is the batch's highest timestamp.
        240 pub timestamp: u64,
        248 pub request: u32,
        /// The state machine operation to apply.
        252 pub operation: crate::Operation,
        253 pub reserved: [u8; 3],
    }
    invalid_header(|prepare| {
        if prepare.parent_padding != 0 {
            return Some("parent_padding != 0");
        }
        if prepare.request_checksum_padding != 0 {
            return Some("request_checksum_padding != 0");
        }
        if prepare.operation == crate::Operation::RESERVED {
            if prepare.size != SIZE as u32 {
                return Some("reserved: size != @sizeOf(Header)");
            }
            if prepare.checksum_body != checksum_body_empty() {
                return Some("reserved: checksum_body != expected");
            }
            if prepare.view != 0 {
                return Some("reserved: view != 0");
            }
            if prepare.release.value != 0 {
                return Some("release != 0");
            }
            if prepare.replica != 0 {
                return Some("reserved: replica != 0");
            }
            if prepare.parent != 0 {
                return Some("reserved: parent != 0");
            }
            if prepare.client != 0 {
                return Some("reserved: client != 0");
            }
            if prepare.request_checksum != 0 {
                return Some("reserved: request_checksum != 0");
            }
            if prepare.checkpoint_id != 0 {
                return Some("reserved: checkpoint_id != 0");
            }
            if prepare.commit != 0 {
                return Some("reserved: commit != 0");
            }
            if prepare.request != 0 {
                return Some("reserved: request != 0");
            }
            if prepare.timestamp != 0 {
                return Some("reserved: timestamp != 0");
            }
        } else if prepare.operation == crate::Operation::ROOT {
            if prepare.size != SIZE as u32 {
                return Some("root: size != @sizeOf(Header)");
            }
            if prepare.checksum_body != checksum_body_empty() {
                return Some("root: checksum_body != expected");
            }
            if prepare.view != 0 {
                return Some("root: view != 0");
            }
            if prepare.release.value != 0 {
                return Some("release != 0");
            }
            if prepare.replica != 0 {
                return Some("root: replica != 0");
            }
            if prepare.parent != 0 {
                return Some("root: parent != 0");
            }
            if prepare.client != 0 {
                return Some("root: client != 0");
            }
            if prepare.request_checksum != 0 {
                return Some("root: request_checksum != 0");
            }
            if prepare.checkpoint_id != 0 {
                return Some("root: checkpoint_id != 0");
            }
            if prepare.op != 0 {
                return Some("root: op != 0");
            }
            if prepare.commit != 0 {
                return Some("root: commit != 0");
            }
            if prepare.timestamp != 0 {
                return Some("root: timestamp != 0");
            }
            if prepare.request != 0 {
                return Some("root: request != 0");
            }
        } else {
            if prepare.release.value == 0 {
                return Some("release == 0");
            }
            if prepare.operation == crate::Operation::PULSE
                || prepare.operation == crate::Operation::UPGRADE
            {
                if prepare.client != 0 {
                    return Some("client != 0");
                }
            } else if prepare.client == 0 {
                return Some("client == 0");
            }
            if prepare.op == 0 {
                return Some("op == 0");
            }
            if prepare.op <= prepare.commit {
                return Some("op <= commit");
            }
            if prepare.timestamp == 0 {
                return Some("timestamp == 0");
            }
            if prepare.operation == crate::Operation::REGISTER
                || prepare.operation == crate::Operation::PULSE
                || prepare.operation == crate::Operation::UPGRADE
            {
                if prepare.request != 0 {
                    return Some("request != 0");
                }
            } else if prepare.request == 0 {
                return Some("request == 0");
            }
        }
        if !stdx::zeroed(&prepare.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

impl Prepare {
    /// Port of `Header.Prepare.reserve`.
    ///
    /// # Panics
    /// Panics if `slot >= journal_slot_count` or if the constructed header is invalid.
    #[must_use]
    pub fn reserve(cluster: u128, slot: u64) -> Self {
        assert!(slot < u64::from(tigerbeetle_core::constants::JOURNAL_SLOT_COUNT));

        let mut header = Self {
            cluster,
            release: crate::multiversion::Release::ZERO,
            operation: crate::Operation::RESERVED,
            op: slot,
            ..Self::default()
        };
        header.set_checksum_body(&[]);
        header.set_checksum();
        assert!(header.invalid().is_none());
        header
    }

    /// Port of `Header.Prepare.root`.
    ///
    /// # Panics
    /// Panics if the constructed header is invalid (cannot happen by construction).
    #[must_use]
    pub fn root(cluster: u128) -> Self {
        let mut header = Self {
            cluster,
            release: crate::multiversion::Release::ZERO,
            operation: crate::Operation::ROOT,
            ..Self::default()
        };
        header.set_checksum_body(&[]);
        header.set_checksum();
        assert!(header.invalid().is_none());
        header
    }
}

typed_header! {
    #[allow(clippy::struct_field_names)] // upstream field names
    pub struct PrepareOk : Command::PrepareOk,
    {
        /// The previous prepare's checksum.
        /// (Same as the corresponding Prepare's `parent`.)
        128 pub parent: u128,
        144 pub parent_padding: u128,
        /// The corresponding prepare's checksum.
        160 pub prepare_checksum: u128,
        176 pub prepare_checksum_padding: u128,
        /// The corresponding prepare's checkpoint_id.
        192 pub checkpoint_id: u128,
        208 pub client: u128,
        224 pub op: u64,
        232 pub commit_min: u64,
        240 pub timestamp: u64,
        248 pub request: u32,
        252 pub operation: crate::Operation,
        253 pub reserved: [u8; 3],
    }
    invalid_header(|prepare_ok| {
        if prepare_ok.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if prepare_ok.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if prepare_ok.release.value != 0 {
            return Some("release != 0");
        }
        if prepare_ok.prepare_checksum_padding != 0 {
            return Some("prepare_checksum_padding != 0");
        }
        if prepare_ok.operation == crate::Operation::RESERVED {
            return Some("operation == .reserved");
        } else if prepare_ok.operation == crate::Operation::ROOT {
            let root_checksum = Prepare::root(prepare_ok.cluster).checksum;
            if prepare_ok.parent != 0 {
                return Some("root: parent != 0");
            }
            if prepare_ok.client != 0 {
                return Some("root: client != 0");
            }
            if prepare_ok.prepare_checksum != root_checksum {
                return Some("root: prepare_checksum != expected");
            }
            if prepare_ok.request != 0 {
                return Some("root: request != 0");
            }
            if prepare_ok.op != 0 {
                return Some("root: op != 0");
            }
            if prepare_ok.timestamp != 0 {
                return Some("root: timestamp != 0");
            }
        } else {
            if prepare_ok.operation == crate::Operation::UPGRADE
                || prepare_ok.operation == crate::Operation::PULSE
            {
                if prepare_ok.client != 0 {
                    return Some("client != 0");
                }
            } else if prepare_ok.client == 0 {
                return Some("client == 0");
            }
            if prepare_ok.op == 0 {
                return Some("op == 0");
            }
            if prepare_ok.timestamp == 0 {
                return Some("timestamp == 0");
            }
            if (prepare_ok.operation == crate::Operation::REGISTER
                || prepare_ok.operation == crate::Operation::UPGRADE
                || prepare_ok.client == 0)
                && prepare_ok.request != 0
            {
                return Some("request != 0");
            }
            if !(prepare_ok.operation == crate::Operation::REGISTER
                || prepare_ok.operation == crate::Operation::UPGRADE
                || prepare_ok.client == 0)
                && prepare_ok.request == 0
            {
                return Some("request == 0");
            }
        }
        if !stdx::zeroed(&prepare_ok.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct Reply : Command::Reply,
    {
        /// The checksum of the corresponding Request.
        128 pub request_checksum: u128,
        144 pub request_checksum_padding: u128,
        /// The checksum to be included with the next request as parent checksum.
        /// It's almost exactly the same as entire header's checksum, except that it is computed
        /// with a fixed view and remains stable if reply is retransmitted in a newer view.
        /// This allows for strong guarantees beyond request, op, and commit numbers, which
        /// have low entropy and may otherwise collide in the event of any correctness bugs.
        160 pub context: u128,
        176 pub context_padding: u128,
        192 pub client: u128,
        208 pub op: u64,
        216 pub commit: u64,
        /// The corresponding `prepare`'s timestamp.
        /// This allows the test workload to verify transfer timeouts.
        224 pub timestamp: u64,
        232 pub request: u32,
        236 pub operation: crate::Operation,
        237 pub reserved: [u8; 19],
    }
    invalid_header(|reply| {
        if reply.release.value == 0 {
            return Some("release == 0");
        }
        // Initialization within `client.zig` asserts that client `id` is greater than zero:
        if reply.client == 0 {
            return Some("client == 0");
        }
        if reply.request_checksum_padding != 0 {
            return Some("request_checksum_padding != 0");
        }
        if reply.context_padding != 0 {
            return Some("context_padding != 0");
        }
        if reply.op != reply.commit {
            return Some("op != commit");
        }
        if reply.timestamp == 0 {
            return Some("timestamp == 0");
        }
        if reply.operation == crate::Operation::REGISTER {
            let size_with_body = SIZE as u32 + size_of::<crate::RegisterResult>() as u32;
            if reply.size != size_with_body {
                return Some("register: size != @sizeOf(Header) + @sizeOf(vsr.RegisterResult)");
            }
            // In this context, the commit number is the newly registered session number.
            // The `0` commit number is reserved for cluster initialization.
            if reply.commit == 0 {
                return Some("commit == 0");
            }
            if reply.request != 0 {
                return Some("request != 0");
            }
        } else {
            if reply.commit == 0 {
                return Some("commit == 0");
            }
            if reply.request == 0 {
                return Some("request == 0");
            }
        }
        if !stdx::zeroed(&reply.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    /// Port of `Header.Commit`.
    #[allow(clippy::struct_field_names)] // upstream field names
    pub struct Commit : Command::Commit,
    {
        /// The latest committed prepare's checksum.
        128 pub commit_checksum: u128,
        144 commit_checksum_padding: u128,
        /// Current checkpoint id.
        160 pub checkpoint_id: u128,
        /// Current checkpoint op.
        176 pub checkpoint_op: u64,
        /// The latest committed prepare's op.
        184 pub commit: u64,
        192 pub timestamp_monotonic: u64,
        200 pub reserved: [u8; 56],
    }
    invalid_header(|commit| {
        if commit.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if commit.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if commit.release.value != 0 {
            return Some("release != 0");
        }
        if commit.commit < commit.checkpoint_op {
            return Some("commit < checkpoint_op");
        }
        if commit.timestamp_monotonic == 0 {
            return Some("timestamp_monotonic == 0");
        }
        if !stdx::zeroed(&commit.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct ExitView : Command::ExitView,
    {
        128 pub reserved: [u8; 128],
    }
    invalid_header(|exit_view| {
        if exit_view.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if exit_view.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if exit_view.release.value != 0 {
            return Some("release != 0");
        }
        if !stdx::zeroed(&exit_view.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct JoinView : Command::JoinView,
    {
        /// A bitset of "present" prepares. If a bit is set, then the corresponding header is not
        /// "blank", the replica has the prepare, and the prepare is not known to be faulty.
        128 pub present_bitset: u128,
        /// A bitset, with set bits indicating headers in the message body which it has definitely
        /// not prepared (i.e. "nack"). The corresponding header may be an actual prepare header, or
        /// it may be a "blank" header.
        144 pub nack_bitset: u128,
        160 pub op: u64,
        /// Set to `commit_min`, to indicate the sending replica's progress.
        /// The sending replica may continue to commit after sending the JV.
        168 pub commit_min: u64,
        176 pub checkpoint_op: u64,
        184 pub log_view: u32,
        188 pub reserved: [u8; 68],
    }
    invalid_header(|join_view| {
        if !(join_view.size as usize - SIZE).is_multiple_of(SIZE) {
            return Some("size multiple invalid");
        }
        if join_view.release.value != 0 {
            return Some("release != 0");
        }
        if join_view.op < join_view.commit_min {
            return Some("op < commit_min");
        }
        if join_view.commit_min < join_view.checkpoint_op {
            return Some("commit_min < checkpoint_op");
        }
        if !stdx::zeroed(&join_view.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct View : Command::View,
    {
        /// Set to zero for a new view, and to a nonce from an RV when responding to the RV.
        128 pub nonce: u128,
        144 pub op: u64,
        /// Equal to `commit_min` if the View message is being sent by a .normal primary,
        /// but may not be equal if sent by potential primary in .view_change status.
        152 pub commit_max: u64,
        /// The replica's `op_checkpoint`.
        160 pub checkpoint_op: u64,
        168 pub reserved: [u8; 88],
    }
    invalid_header(|view| {
        let body_size = view.size as usize - SIZE;
        if body_size < tigerbeetle_core::constants::CHECKPOINT_STATE_SIZE {
            return Some("checkpointstate missing");
        }
        let headers_size = body_size - tigerbeetle_core::constants::CHECKPOINT_STATE_SIZE;
        if !headers_size.is_multiple_of(SIZE) {
            return Some("headers size multiple invalid");
        }
        if view.release.value != 0 {
            return Some("release != 0");
        }
        if view.op < view.commit_max {
            return Some("op < commit_max");
        }
        if view.commit_max < view.checkpoint_op {
            return Some("commit_max < checkpoint_op");
        }
        if !stdx::zeroed(&view.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct GetView : Command::GetView,
    {
        128 pub nonce: u128,
        144 pub reserved: [u8; 112],
    }
    invalid_header(|get_view| {
        if get_view.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if get_view.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if get_view.release.value != 0 {
            return Some("release != 0");
        }
        if get_view.nonce == 0 {
            return Some("nonce == 0");
        }
        if !stdx::zeroed(&get_view.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct GetHeaders : Command::GetHeaders,
    {
        /// The minimum op requested (inclusive).
        128 pub op_min: u64,
        /// The maximum op requested (inclusive).
        136 pub op_max: u64,
        144 pub reserved: [u8; 112],
    }
    invalid_header(|get_headers| {
        if get_headers.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if get_headers.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if get_headers.view != 0 {
            return Some("view == 0");
        }
        if get_headers.release.value != 0 {
            return Some("release != 0");
        }
        if get_headers.op_min > get_headers.op_max {
            return Some("op_min > op_max");
        }
        if !stdx::zeroed(&get_headers.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct GetPrepare : Command::GetPrepare,
    {
        128 pub prepare_checksum: u128,
        144 pub prepare_checksum_padding: u128,
        160 pub prepare_op: u64,
        168 pub reserved: [u8; 88],
    }
    invalid_header(|get_prepare| {
        if get_prepare.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if get_prepare.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if get_prepare.view != 0 && get_prepare.prepare_checksum != 0 {
            return Some("view != 0 and checksum != 0");
        }
        if get_prepare.release.value != 0 {
            return Some("release != 0");
        }
        if get_prepare.prepare_checksum_padding != 0 {
            return Some("prepare_checksum_padding != 0");
        }
        if !stdx::zeroed(&get_prepare.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    #[allow(clippy::struct_field_names)] // upstream field names
    pub struct GetReply : Command::GetReply,
    {
        128 pub reply_checksum: u128,
        144 reply_checksum_padding: u128,
        160 pub reply_client: u128,
        176 pub reply_op: u64,
        184 pub reserved: [u8; 72],
    }
    invalid_header(|get_reply| {
        if get_reply.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if get_reply.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if get_reply.release.value != 0 {
            return Some("release != 0");
        }
        if get_reply.reply_checksum_padding != 0 {
            return Some("reply_checksum_padding != 0");
        }
        if get_reply.view != 0 {
            return Some("view == 0");
        }
        if get_reply.reply_client == 0 {
            return Some("reply_client == 0");
        }
        if !stdx::zeroed(&get_reply.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    pub struct Headers : Command::Headers,
    {
        128 pub reserved: [u8; 128],
    }
    invalid_header(|headers| {
        if headers.size == SIZE as u32 {
            return Some("size == @sizeOf(Header)");
        }
        if !(headers.size as usize - SIZE).is_multiple_of(SIZE) {
            return Some("size multiple invalid");
        }
        if headers.release.value != 0 {
            return Some("release != 0");
        }
        if !stdx::zeroed(&headers.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

typed_header! {
    /// Port of `Header.Eviction`.
    ///
    /// The `reason` is carried as its raw on-wire ordinal so that corrupt values survive decoding
    /// (upstream decodes into the enum directly); use [`Eviction::reason`] for the typed view.
    pub struct Eviction : Command::Eviction,
    {
        128 pub client: u128,
        144 pub reserved: [u8; 111],
        /// Raw on-wire ordinal of `Header.Eviction.Reason`.
        255 pub reason_ordinal: u8,
    }
    invalid_header(|eviction| {
        if eviction.size != SIZE as u32 {
            return Some("size != @sizeOf(Header)");
        }
        if eviction.checksum_body != checksum_body_empty() {
            return Some("checksum_body != expected");
        }
        if eviction.release.value == 0 {
            return Some("release == 0");
        }
        if eviction.client == 0 {
            return Some("client == 0");
        }
        if !stdx::zeroed(&eviction.reserved) {
            return Some("reserved != 0");
        }
        match eviction.reason() {
            Some(reason) if reason != Reason::Reserved => {}
            Some(_) => return Some("reason == reserved"),
            None => return Some("reason invalid"),
        }
        None
    })
}

impl Eviction {
    /// The typed `reason` (`None` if the stored ordinal is not a valid variant).
    #[must_use]
    pub fn reason(&self) -> Option<Reason> {
        let value = self.reason_ordinal;
        REASONS_ALL.iter().copied().find(|reason| *reason as u8 == value)
    }
}

typed_header! {
    pub struct GetBlocks : Command::GetBlocks,
    {
        128 pub reserved: [u8; 128],
    }
    invalid_header(|get_blocks| {
        if get_blocks.view != 0 {
            return Some("view != 0");
        }
        if get_blocks.size == SIZE as u32 {
            return Some("size == @sizeOf(Header)");
        }
        if !(get_blocks.size as usize - SIZE).is_multiple_of(size_of::<crate::BlockRequest>()) {
            return Some("size multiple invalid");
        }
        if get_blocks.release.value != 0 {
            return Some("release != 0");
        }
        if !stdx::zeroed(&get_blocks.reserved) {
            return Some("reserved != 0");
        }
        None
    })
}

/// Port of `lsm/schema.BlockType`.
///
/// DEVIATION: upstream defines this in `src/lsm/schema.zig`; it lives here (vsr) because
/// `Header.Block` embeds it and the crate dependency direction is core ← lsm ← vsr.
/// TODO(port): move to tigerbeetle-lsm when the schema module lands, keeping the re-export.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlockType {
    /// Unused; verifies that no block is written with a default 0 block type.
    Reserved = 0,
    FreeSet = 1,
    ClientSessions = 2,
    Manifest = 3,
    Index = 4,
    Value = 5,
}

impl BlockType {
    /// Upstream `BlockType.valid` — for a raw on-disk ordinal, since a decoded `BlockType`
    /// is always representable.
    #[must_use]
    pub fn valid_ordinal(value: u8) -> bool {
        matches!(value, 0..=5)
    }

    /// Decodes a raw on-disk ordinal (upstream `std.meta.intToEnum`).
    #[must_use]
    pub fn from_ordinal(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Reserved),
            1 => Some(Self::FreeSet),
            2 => Some(Self::ClientSessions),
            3 => Some(Self::Manifest),
            4 => Some(Self::Index),
            5 => Some(Self::Value),
            _ => None,
        }
    }
}

typed_header! {
    #[allow(clippy::struct_field_names)] // upstream field name `block_type`
    pub struct Block : Command::Block,
    {
        // Schema is determined by `block_type`.
        128 pub metadata_bytes: [u8; 96],
        // Fields shared by all block types:
        224 pub address: u64,
        232 pub snapshot: u64,
        240 pub block_type_ordinal: u8,
        241 reserved_block: [u8; 15],
    }
    invalid_header(|block| {
        if block.size > tigerbeetle_core::constants::BLOCK_SIZE as u32 {
            return Some("size > block_size");
        }
        if block.size == SIZE as u32 {
            return Some("size = @sizeOf(Header)");
        }
        if block.view != 0 {
            return Some("view != 0");
        }
        if block.release.value == 0 {
            return Some("release == 0");
        }
        if block.replica != 0 {
            return Some("replica != 0");
        }
        if block.address == 0 {
            // address ≠ 0
            return Some("address == 0");
        }
        if !BlockType::valid_ordinal(block.block_type_ordinal) {
            return Some("block_type invalid");
        }
        if block.block_type_ordinal == BlockType::Reserved as u8 {
            return Some("block_type == .reserved");
        }
        // TODO When manifest blocks include a snapshot, verify that snapshot≠0.
        None
    })
}

impl Block {
    /// The size of the per-block-type metadata area in the header tail.
    pub const METADATA_SIZE: usize = 96;

    /// The typed `block_type` (`None` if the stored ordinal is not a valid variant).
    #[must_use]
    pub fn block_type(&self) -> Option<BlockType> {
        match self.block_type_ordinal {
            0 => Some(BlockType::Reserved),
            1 => Some(BlockType::FreeSet),
            2 => Some(BlockType::ClientSessions),
            3 => Some(BlockType::Manifest),
            4 => Some(BlockType::Index),
            5 => Some(BlockType::Value),
            _ => None,
        }
    }
}

impl Header {
    /// Returns whether the immediate sender is a replica or client (if this can be determined).
    /// Some commands such as .request or .prepare may be forwarded on to other replicas so that
    /// Header.replica or Header.client only identifies the ultimate origin, not the latest peer.
    ///
    /// # Panics
    /// Panics for `command == .reserved` (upstream `unreachable`), which cannot be decoded via
    /// [`Header::from_wire`] but may be constructed directly.
    // Arms intentionally mirror upstream's exhaustive switch, several of which return `.unknown`.
    #[allow(clippy::match_same_arms)]
    #[must_use]
    pub fn peer_type(&self) -> crate::Peer {
        match self.command {
            Command::Reserved => unreachable!("reserved command has no peer type"),
            // reply/prepare/block hide the origin entirely, as do deprecated commands:
            Command::Reply | Command::Prepare | Command::Block => crate::Peer::Unknown,

            // The peer may be a replica or a client, since replicas forward request messages.
            // However, we return the client ID, as it is useful for the MessageBus. Specifically,
            // a replica that receives a request from a client can immediately cache the connection
            // in its client map, instead of waiting for an infrequent PingClient message to do so.
            Command::Request => match self.into_typed::<Request>() {
                Some(request) => crate::Peer::ClientLikely { id: request.client },
                None => unreachable!("command decodes"),
            },

            // The peer is certainly a client:
            Command::PingClient => match self.into_typed::<PingClient>() {
                Some(ping) => crate::Peer::Client { id: ping.client },
                None => unreachable!("command decodes"),
            },

            // The peer is certainly a replica:
            Command::Ping
            | Command::Pong
            | Command::PongClient
            | Command::PrepareOk
            | Command::Commit
            | Command::ExitView
            | Command::JoinView
            | Command::View
            | Command::GetView
            | Command::GetHeaders
            | Command::GetPrepare
            | Command::GetReply
            | Command::Headers
            | Command::Eviction
            | Command::GetBlocks => crate::Peer::Replica { replica: self.replica },

            Command::Deprecated12
            | Command::Deprecated21
            | Command::Deprecated22
            | Command::Deprecated23 => crate::Peer::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiversion::ReleaseTriple;

    /// Pins our explicit `to_wire()` encoding to the upstream `extern struct` memory layout.
    #[test]
    fn header_wire_layout() {
        let header = Header {
            cluster: 1,
            epoch: 3,
            view: 2,
            release: Release::from_triple(ReleaseTriple { major: 1, minor: 2, patch: 3 }),
            command: Command::Prepare,
            replica: 4,
            ..Header::empty()
        };
        assert_eq!(header.size, SIZE as u32);
        assert_eq!(header.protocol, VERSION);

        let w = header.to_wire();
        assert!(w[0..16].iter().all(|&b| b == 0), "checksum starts zeroed");
        assert_eq!(&w[80..96], &u128::to_le_bytes(1), "cluster @80");
        assert_eq!(&w[96..100], &u32::to_le_bytes(256), "size @96");
        assert_eq!(&w[100..104], &u32::to_le_bytes(3), "epoch @100");
        assert_eq!(&w[104..108], &u32::to_le_bytes(2), "view @104");
        assert_eq!(&w[108..112], &u32::to_le_bytes(0x0001_0203), "release @108");
        assert_eq!(&w[112..114], &u16::to_le_bytes(VERSION), "protocol @112");
        assert_eq!(w[114], 6, "command Prepare @114");
        assert_eq!(w[115], 4, "replica @115");
        assert!(w[116..128].iter().all(|&b| b == 0), "reserved_frame @116..128");
        assert!(w[128..256].iter().all(|&b| b == 0), "reserved_command @128..256");

        // Upstream: reserved_command must be aligned so that it can be cast to any type.
        assert_eq!(128 % 32, 0);
    }

    #[test]
    fn header_wire_round_trip() {
        let header = Header {
            checksum: 0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10,
            checksum_padding: 17,
            checksum_body: 0x2122_2324_2526_2728_292a_2b2c_2d2e_2f30,
            checksum_body_padding: 19,
            nonce_reserved: 18,
            cluster: 0x1112_1314_1516_1718_191a_1b1c_1d1e_1f20,
            size: SIZE as u32,
            epoch: 5,
            view: 6,
            release: Release::MINIMUM,
            protocol: 42,
            command: Command::PingClient,
            replica: 7,
            reserved_frame: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            reserved_command: core::array::from_fn(|i| (i % 251) as u8),
        };
        let wire = header.to_wire();
        assert_eq!(Header::from_wire(&wire), Some(header));

        // An unknown command ordinal cannot decode into the frame:
        let mut corrupt = wire;
        corrupt[114] = 99;
        assert_eq!(Header::from_wire(&corrupt), None);
    }

    #[test]
    fn header_checksum() {
        const BODY: [u8; 64] = [7; 64];

        let mut header = Header {
            size: SIZE as u32 + BODY.len() as u32,
            command: Command::Prepare,
            ..Header::empty()
        };

        assert!(!header.valid_checksum());
        header.set_checksum_body(&BODY);
        assert!(!header.valid_checksum(), "set_checksum must run after set_checksum_body");
        assert!(header.valid_checksum_body(&BODY));

        header.set_checksum();
        assert!(header.valid_checksum());

        // Any change to the header invalidates its checksum:
        let tampered = Header { view: header.view + 1, ..header };
        assert_ne!(tampered.checksum, tampered.calculate_checksum());

        // Any change to the body invalidates its checksum:
        let mut other_body = BODY;
        other_body[0] ^= 1;
        assert!(!header.valid_checksum_body(&other_body));
    }

    #[test]
    fn checksum_body_empty_is_the_empty_input_tag() {
        assert_eq!(checksum_body_empty(), tigerbeetle_core::checksum::checksum(&[]));
    }
    /// Pins the command-tail offsets of a typed header against upstream's extern struct layout,
    /// and verifies the frame prefix matches the base `Header` encoding exactly.
    #[test]
    fn ping_wire_layout_and_frame_prefix() {
        let mut ping = Ping {
            cluster: 0xaaaa,
            view: 7, // pings carry the on-disk view
            replica: 2,
            release: Release::MINIMUM,
            checkpoint_id: u128::from_le_bytes(*b"checkpoint-id-16"),
            checkpoint_op: 1 << 20,
            ping_timestamp_monotonic: 123_456_789,
            release_count: 3,
            ..Ping::default()
        };
        let wire = ping.to_wire();

        // Command-tail offsets:
        assert_eq!(&wire[128..144], &u128::to_le_bytes(u128::from_le_bytes(*b"checkpoint-id-16")));
        assert_eq!(&wire[144..152], &u64::to_le_bytes(1 << 20));
        assert_eq!(&wire[152..160], &u64::to_le_bytes(123_456_789));
        assert_eq!(&wire[160..162], &u16::to_le_bytes(3));
        assert!(wire[162..256].iter().all(|&b| b == 0), "reserved @162..256");

        // Frame prefix identical to base Header with same fields:
        let frame = Header {
            cluster: ping.cluster,
            size: ping.size,
            epoch: ping.epoch,
            view: ping.view,
            release: ping.release,
            protocol: ping.protocol,
            command: ping.command,
            replica: ping.replica,
            reserved_frame: ping.reserved_frame,
            checksum: ping.checksum,
            checksum_padding: 0,
            checksum_body: ping.checksum_body,
            checksum_body_padding: 0,
            nonce_reserved: 0,
            reserved_command: [0; 128],
        };
        assert_eq!(wire[..128], frame.to_wire()[..128]);

        // Round trip:
        assert_eq!(Ping::from_wire(&wire), Some(ping));
        let parsed = Header::from_wire(&wire).and_then(|h| h.into_typed::<Ping>());
        assert_eq!(parsed, Some(ping));

        // set_checksum flows through and validates:
        ping.set_checksum();
        assert!(ping.valid_checksum());

        // Cross-type rejection: a pong wire frame does not decode as a ping.
        assert_eq!(Ping::from_wire(&Pong::default().to_wire()), None);
    }

    fn valid_ping() -> Ping {
        Ping {
            cluster: 1,
            size: SIZE as u32 + size_of::<Release>() as u32 * constants::VSR_RELEASES_MAX,
            view: 9,
            release: Release::MINIMUM,
            checkpoint_op: 0,
            ping_timestamp_monotonic: 42,
            release_count: 1,
            ..Ping::default()
        }
    }

    #[test]
    fn ping_invalid_header_checks() {
        assert_eq!(valid_ping().invalid_header(), None);

        let mut ping = valid_ping();
        ping.size -= 1;
        assert_eq!(
            ping.invalid_header(),
            Some("size != @sizeOf(Header) + @sizeOf(vsr.Release) * constants.vsr_releases_max")
        );

        let mut ping = valid_ping();
        ping.release = Release::ZERO;
        assert_eq!(ping.invalid_header(), Some("release == 0"));

        let mut ping = valid_ping();
        ping.checkpoint_op = 1;
        assert_eq!(ping.invalid_header(), Some("checkpoint_op invalid"));

        let mut ping = valid_ping();
        ping.ping_timestamp_monotonic = 0;
        assert_eq!(ping.invalid_header(), Some("ping_timestamp_monotonic != expected"));

        let mut ping = valid_ping();
        ping.release_count = 0;
        assert_eq!(ping.invalid_header(), Some("release_count == 0"));

        let mut ping = valid_ping();
        ping.release_count = constants::VSR_RELEASES_MAX as u16 + 1;
        assert_eq!(ping.invalid_header(), Some("release_count > vsr_releases_max"));

        let mut ping = valid_ping();
        ping.reserved[0] = 1;
        assert_eq!(ping.invalid_header(), Some("reserved != 0"));
    }

    // `ping_client`/`pong_client` are the upstream message names:
    #[allow(clippy::similar_names)]
    #[test]
    fn pong_client_family_invalid_header_checks() {
        let mut pong = Pong {
            cluster: 1,
            view: 4,
            replica: 0,
            release: Release::MINIMUM,
            ping_timestamp_monotonic: 11,
            pong_timestamp_wall: 22,
            ..Pong::default()
        };
        pong.set_checksum_body(&[]);
        assert_eq!(pong.invalid_header(), None);
        let tampered = Pong { pong_timestamp_wall: 0, ..pong };
        assert_eq!(tampered.invalid_header(), Some("pong_timestamp_wall == 0"));
        let no_body_checksum = Pong { checksum_body: 1, ..pong };
        assert_eq!(no_body_checksum.invalid_header(), Some("checksum_body != expected"));

        let mut ping_client = PingClient {
            client: 0xd00d,
            ping_timestamp_monotonic: 33,
            session: 0,
            release: Release::MINIMUM,
            ..PingClient::default()
        };
        ping_client.set_checksum_body(&[]);
        assert_eq!(ping_client.invalid_header(), None);
        let anonymous = PingClient { client: 0, ..ping_client };
        assert_eq!(anonymous.invalid_header(), Some("client == 0"));

        let mut pong_client = PongClient {
            ping_timestamp_monotonic: 44,
            release: Release::MINIMUM,
            ..PongClient::default()
        };
        // Empty-body commands carry the empty-input body checksum:
        pong_client.set_checksum_body(&[]);
        assert_eq!(pong_client.invalid_header(), None);
    }

    /// Upstream `Header.invalid()` runs the shared frame checks before the per-command ones.
    #[test]
    fn header_invalid_frame_checks() {
        let ping = valid_ping();
        let header = ping.frame();

        assert_eq!(
            Header { checksum_body_padding: 1, ..header }.invalid(),
            Some("checksum_body_padding != 0")
        );
        assert_eq!(Header { epoch: 1, ..header }.invalid(), Some("epoch != 0"));
        assert_eq!(
            Header { protocol: VERSION + 1, ..header }.invalid(),
            Some("protocol != Version")
        );
        assert_eq!(header.invalid(), None);

        let mut dirty_reserved_frame = header;
        dirty_reserved_frame.reserved_frame[3] = 9;
        assert_eq!(dirty_reserved_frame.invalid(), Some("reserved_frame != 0"));

        // Deprecated commands are rejected outright:
        let deprecated = Header { command: Command::Deprecated12, ..header };
        assert_eq!(deprecated.invalid(), Some("deprecated message type"));
    }
    /// Pins the tail layouts of the client-path headers (Request/Prepare/Reply) against the
    /// upstream extern struct offsets, and round-trips them through the wire encoding.
    // Upstream message names (`request`/`prepare`/`reply`) shadow nothing here but are similar:
    #[test]
    fn request_prepare_reply_wire_layout() {
        let mut request = Request {
            cluster: 9,
            size: SIZE as u32 + 64,
            release: Release::MINIMUM,
            parent: 0x1111,
            client: 0xd00d,
            session: 5,
            timestamp: 777,
            request: 6,
            operation: crate::Operation(200), // state-machine op
            previous_request_latency: 1234,
            ..Request::default()
        };
        let body = [7u8; 64];
        request.set_checksum_body(&body);
        request.set_checksum();

        let wire = request.to_wire();
        assert_eq!(&wire[128..144], &u128::to_le_bytes(0x1111), "parent @128");
        assert_eq!(&wire[160..176], &u128::to_le_bytes(0xd00d), "client @160");
        assert_eq!(&wire[176..184], &u64::to_le_bytes(5), "session @176");
        assert_eq!(wire[196], 200, "operation @196");
        assert_eq!(&wire[200..204], &u32::to_le_bytes(1234));
        assert_eq!(Request::from_wire(&wire), Some(request));
        assert_eq!(Header::from_wire(&wire).and_then(|h| h.into_typed::<Request>()), Some(request));

        let prepare = Prepare {
            cluster: 9,
            view: 3,
            release: Release::MINIMUM,
            parent: 0x2222,
            request_checksum: 0x3333,
            checkpoint_id: 0x4444,
            client: 0xd00d,
            op: 10,
            commit: 8,
            timestamp: 99_999,
            request: 6,
            operation: crate::Operation(200),
            ..Prepare::default()
        };
        let wire = prepare.to_wire();
        assert_eq!(&wire[128..144], &u128::to_le_bytes(0x2222), "parent @128");
        assert_eq!(&wire[160..176], &u128::to_le_bytes(0x3333), "request_checksum @160");
        assert_eq!(&wire[192..208], &u128::to_le_bytes(0x4444), "checkpoint_id @192");
        assert_eq!(&wire[224..232], &u64::to_le_bytes(10), "op @224");
        assert_eq!(&wire[248..252], &u32::to_le_bytes(6), "request @248");
        assert_eq!(wire[252], 200, "operation @252");
        assert_eq!(Prepare::from_wire(&wire), Some(prepare));

        let reply = Reply {
            cluster: 9,
            view: 3,
            replica: 1,
            release: Release::MINIMUM,
            request_checksum: 0x5555,
            context: 0x6666,
            client: 0xd00d,
            op: 8,
            commit: 8,
            timestamp: 99_999,
            request: 6,
            operation: crate::Operation(200),
            ..Reply::default()
        };
        let wire = reply.to_wire();
        assert_eq!(&wire[128..144], &u128::to_le_bytes(0x5555), "request_checksum @128");
        assert_eq!(&wire[160..176], &u128::to_le_bytes(0x6666), "context @160");
        assert_eq!(&wire[208..216], &u64::to_le_bytes(8), "op @208");
        assert_eq!(wire[236], 200, "operation @236");
        assert_eq!(Reply::from_wire(&wire), Some(reply));
    }

    #[test]
    fn request_invalid_header_checks() {
        // A state-machine-op request (the common case):
        let mut request = Request {
            cluster: 1,
            size: SIZE as u32 + 16,
            release: Release::MINIMUM,
            client: 0xd00d,
            session: 5,
            request: 6,
            parent: 0x9999,
            operation: crate::Operation(200),
            previous_request_latency: 1234,
            ..Request::default()
        };
        request.set_checksum_body(&[0xab; 16]);
        assert_eq!(request.invalid_header(), None);

        // register (client!=0, everything else zero):
        let mut register = Request {
            cluster: 1,
            size: SIZE as u32 + size_of::<crate::RegisterRequest>() as u32,
            release: Release::MINIMUM,
            client: 0xd00d,
            operation: crate::Operation::REGISTER,
            ..Request::default()
        };
        register.set_checksum_body(&[0u8; 256]);
        assert_eq!(register.invalid_header(), None);
        let wrong_size = Request { size: SIZE as u32 + 17, ..register };
        assert_eq!(
            wrong_size.invalid_header(),
            Some("register: size != @sizeOf(Header) [+ @sizeOf(vsr.RegisterRequest)]")
        );
        let reused_session = Request { session: 1, ..register };
        assert_eq!(reused_session.invalid_header(), Some("register: session != 0"));

        // pulse (everything zero, no body):
        let mut pulse = Request {
            cluster: 1,
            size: SIZE as u32,
            release: Release::MINIMUM,
            operation: crate::Operation::PULSE,
            ..Request::default()
        };
        pulse.set_checksum_body(&[]);
        assert_eq!(pulse.invalid_header(), None);
        let stray_client = Request { client: 1, ..pulse };
        assert_eq!(stray_client.invalid_header(), Some("pulse: client != 0"));

        // upgrade carries an UpgradeRequest body:
        let upgrade = Request {
            cluster: 1,
            size: SIZE as u32 + size_of::<crate::UpgradeRequest>() as u32,
            release: Release::MINIMUM,
            operation: crate::Operation::UPGRADE,
            ..Request::default()
        };
        assert_eq!(upgrade.invalid_header(), None);
        let undersized = Request { size: SIZE as u32 + 8, ..upgrade };
        assert_eq!(
            undersized.invalid_header(),
            Some("upgrade: size != @sizeOf(Header) + @sizeOf(vsr.UpgradeRequest)")
        );

        // Ordinals inside the reserved gap are rejected:
        let gap = Request { operation: crate::Operation(100), ..request };
        assert_eq!(gap.invalid_header(), Some("operation is reserved"));

        // State-machine ops pass through (Replica validates against the StateMachine):
        let sm_op = Request { operation: crate::Operation(128), ..request };
        assert_eq!(sm_op.invalid_header(), None);
    }

    #[test]
    fn prepare_reserve_root_and_invalid() {
        let cluster = 0xbeef;
        let reserved = Prepare::reserve(cluster, 3);
        assert_eq!(reserved.op, 3);
        assert_eq!(reserved.operation, crate::Operation::RESERVED);
        assert!(reserved.valid_checksum());
        assert!(reserved.valid_checksum_body(&[]));
        // Full validation through the base frame:
        let Some(header) = Header::from_wire(&reserved.to_wire()) else {
            unreachable!("decodable")
        };
        assert!(header.into_typed::<Prepare>().is_some());
        assert_eq!(header.invalid(), None);

        let root = Prepare::root(cluster);
        assert_eq!(root.operation, crate::Operation::ROOT);
        assert_eq!(
            root.checksum,
            Prepare::root(cluster).checksum,
            "root checksum is deterministic"
        );

        // A state-machine-op prepare must carry a body checksum, nonzero op > commit, etc.:
        let mut prepare = Prepare {
            cluster,
            size: SIZE as u32 + 14,
            view: 1,
            release: Release::MINIMUM,
            client: 0xd00d,
            op: 5,
            commit: 3,
            timestamp: 42_000,
            request: 1,
            operation: crate::Operation(200),
            ..Prepare::default()
        };
        prepare.set_checksum_body(b"transfer batch");
        assert_eq!(prepare.invalid_header(), None);

        let mut bad_order = prepare;
        bad_order.commit = 5; // commit == op
        assert_eq!(bad_order.invalid_header(), Some("op <= commit"));
    }

    /// `PrepareOk.root` cross-checks its `prepare_checksum` against `Prepare::root`.
    #[test]
    fn prepare_ok_root_consistency() {
        let cluster = 0xf00d;
        let mut ok = PrepareOk {
            cluster,
            view: 2,
            replica: 3,
            operation: crate::Operation::ROOT,
            prepare_checksum: Prepare::root(cluster).checksum,
            ..PrepareOk::default()
        };
        ok.set_checksum_body(&[]);
        assert_eq!(ok.invalid_header(), None);

        let mut wrong = ok;
        wrong.prepare_checksum ^= 1;
        assert_eq!(wrong.invalid_header(), Some("root: prepare_checksum != expected"));
    }

    #[test]
    fn eviction_reason_validation() {
        let mut eviction = Eviction {
            cluster: 1,
            view: 2,
            release: Release::MINIMUM,
            client: 0xcafe,
            reason_ordinal: Reason::SessionTooLow as u8,
            ..Eviction::default()
        };
        eviction.set_checksum_body(&[]);
        assert_eq!(eviction.invalid_header(), None);

        let reserved = Eviction { reason_ordinal: 0, ..eviction };
        assert_eq!(reserved.invalid_header(), Some("reason == reserved"));

        // An out-of-range ordinal survives decoding and is rejected:
        let corrupted = Eviction { reason_ordinal: 200, ..eviction };
        assert_eq!(corrupted.reason(), None);
        assert_eq!(corrupted.to_wire()[255], 200, "ordinal round-trips");
        assert_eq!(Eviction::from_wire(&corrupted.to_wire()), Some(corrupted));
        assert_eq!(corrupted.invalid_header(), Some("reason invalid"));

        // Every valid reason variant encodes to its ordinal and validates:
        for reason in REASONS_ALL {
            if reason == Reason::Reserved {
                continue;
            }
            let encoded = Eviction { reason_ordinal: reason as u8, ..eviction };
            assert_eq!(encoded.reason(), Some(reason));
            assert_eq!(encoded.invalid_header(), None);
        }
    }

    /// Headers/JoinView/GetBlocks validate their body-size multiples.
    #[test]
    fn body_multiple_size_checks() {
        // headers: one or more embedded headers
        let mut headers =
            Headers { cluster: 1, size: SIZE as u32 * 3, view: 1, ..Headers::default() };
        assert_eq!(headers.invalid_header(), None);
        headers.size = SIZE as u32;
        assert_eq!(headers.invalid_header(), Some("size == @sizeOf(Header)"));
        headers.size = SIZE as u32 * 3 + 4;
        assert_eq!(headers.invalid_header(), Some("size multiple invalid"));

        // get_blocks: multiples of BlockRequest (32)
        let mut get_blocks = GetBlocks { size: SIZE as u32 + 64, ..GetBlocks::default() };
        assert_eq!(get_blocks.invalid_header(), None);
        get_blocks.size = SIZE as u32 + 33;
        assert_eq!(get_blocks.invalid_header(), Some("size multiple invalid"));

        // join_view: body is whole headers
        let join_view = JoinView {
            size: SIZE as u32 * 2,
            cluster: 1,
            view: 1,
            op: 10,
            commit_min: 5,
            checkpoint_op: 0,
            ..JoinView::default()
        };
        assert_eq!(join_view.invalid_header(), None);

        // view: body starts with CheckpointState then whole headers
        let mut view = View {
            cluster: 1,
            view: 1,
            op: 20,
            commit_max: 15,
            checkpoint_op: 0,
            ..View::default()
        };
        view.size = (SIZE + tigerbeetle_core::constants::CHECKPOINT_STATE_SIZE) as u32;
        assert_eq!(view.invalid_header(), None);
        view.size -= 1;
        assert_eq!(view.invalid_header(), Some("checkpointstate missing"));
    }
    #[test]
    fn block_wire_layout_and_invalid() {
        let mut block = Block {
            cluster: 1,
            size: tigerbeetle_core::constants::BLOCK_SIZE as u32,
            release: Release::MINIMUM,
            metadata_bytes: core::array::from_fn(|i| (i % 97) as u8),
            address: 7,
            snapshot: 8,
            block_type_ordinal: BlockType::Value as u8,
            ..Block::default()
        };
        // Note: unlike most commands, Header.Block does not require an empty-body checksum;
        // its validity covers the metadata/address/type fields only.
        block.set_checksum();
        assert_eq!(block.block_type(), Some(BlockType::Value));
        assert!(block.invalid_header().is_none());

        // Tail layout pins:
        let wire = block.to_wire();
        assert_eq!(&wire[224..232], &u64::to_le_bytes(7), "address @224");
        assert_eq!(&wire[232..240], &u64::to_le_bytes(8), "snapshot @232");
        assert_eq!(wire[240], 5, "block_type @240");
        assert_eq!(wire[241..256], [0u8; 15], "reserved_block @241");
        assert_eq!(Block::from_wire(&wire), Some(block));

        // Validation table:
        let oversize = Block { size: tigerbeetle_core::constants::BLOCK_SIZE as u32 + 1, ..block };
        assert_eq!(oversize.invalid_header(), Some("size > block_size"));
        let empty = Block { size: SIZE as u32, ..block };
        assert_eq!(empty.invalid_header(), Some("size = @sizeOf(Header)"));
        let stale_view = Block { view: 1, ..block };
        assert_eq!(stale_view.invalid_header(), Some("view != 0"));
        let anonymous = Block { release: Release::ZERO, ..block };
        assert_eq!(anonymous.invalid_header(), Some("release == 0"));
        let replica_owned = Block { replica: 2, ..block };
        assert_eq!(replica_owned.invalid_header(), Some("replica != 0"));
        let null_address = Block { address: 0, ..block };
        assert_eq!(null_address.invalid_header(), Some("address == 0"));
        let unknown_type = Block { block_type_ordinal: 6, ..block };
        assert_eq!(unknown_type.invalid_header(), Some("block_type invalid"));
        assert_eq!(unknown_type.block_type(), None);
        let reserved_type = Block { block_type_ordinal: 0, ..block };
        assert_eq!(reserved_type.invalid_header(), Some("block_type == .reserved"));

        // Every valid ordinal round-trips:
        for ordinal in 0..=5u8 {
            let encoded = Block { block_type_ordinal: ordinal, ..block };
            assert!(BlockType::valid_ordinal(ordinal));
            assert!(encoded.block_type().is_some());
        }
        assert!(!BlockType::valid_ordinal(6));
    }

    /// Upstream `Header.peer_type()` mapping.
    // `ping_client`/`pong_client` are upstream message names:
    #[allow(clippy::similar_names)]
    #[test]
    fn header_peer_type() {
        let ping_client = PingClient {
            cluster: 1,
            client: 0xd00d,
            ping_timestamp_monotonic: 5,
            release: Release::MINIMUM,
            ..PingClient::default()
        };
        assert_eq!(ping_client.frame().peer_type(), crate::Peer::Client { id: 0xd00d });

        let request = Request {
            cluster: 1,
            client: 0xbeef,
            operation: crate::Operation(200),
            ..Request::default()
        };
        assert_eq!(request.frame().peer_type(), crate::Peer::ClientLikely { id: 0xbeef });

        let prepare_ok = PrepareOk {
            cluster: 1,
            view: 1,
            replica: 4,
            parent: 9,
            prepare_checksum: 10,
            checkpoint_id: 11,
            op: 3,
            commit_min: 3,
            timestamp: 12,
            request: 1,
            operation: crate::Operation(200),
            ..PrepareOk::default()
        };
        assert_eq!(prepare_ok.frame().peer_type(), crate::Peer::Replica { replica: 4 });

        // reply/prepare/block hide the origin:
        let prepare = Prepare {
            cluster: 1,
            replica: 4,
            operation: crate::Operation::RESERVED,
            op: 3,
            ..Prepare::default()
        };
        assert_eq!(prepare.frame().peer_type(), crate::Peer::Unknown);
    }

    /// Upstream `vsr.Peer.transition`.
    // Binding names mirror the upstream union tags:
    #[allow(clippy::similar_names)]
    #[test]
    fn peer_transition_matrix() {
        use crate::{Peer as P, PeerTransition as T};
        let replica_a = P::Replica { replica: 1 };
        let replica_b = P::Replica { replica: 2 };
        let client_x = P::Client { id: 100 };
        let likely_x = P::ClientLikely { id: 100 };
        let likely_y = P::ClientLikely { id: 200 };

        assert_eq!(P::transition(P::Unknown, client_x), T::Update);
        assert_eq!(P::transition(P::Unknown, replica_a), T::Update);
        assert_eq!(P::transition(P::Unknown, P::Unknown), T::Update);

        assert_eq!(P::transition(likely_x, likely_x), T::Retain);
        assert_eq!(P::transition(likely_x, likely_y), T::Retain);
        assert_eq!(P::transition(likely_x, client_x), T::Update);
        assert_eq!(P::transition(likely_x, P::Client { id: 101 }), T::Reject);
        assert_eq!(P::transition(likely_x, replica_a), T::Update);
        assert_eq!(P::transition(likely_x, P::Unknown), T::Retain);

        assert_eq!(P::transition(replica_a, replica_a), T::Retain);
        assert_eq!(P::transition(replica_a, replica_b), T::Reject);
        assert_eq!(P::transition(replica_a, client_x), T::Reject);
        assert_eq!(P::transition(replica_a, likely_x), T::Retain);

        assert_eq!(P::transition(client_x, client_x), T::Retain);
        assert_eq!(P::transition(client_x, P::Client { id: 101 }), T::Reject);
        assert_eq!(P::transition(client_x, likely_x), T::Retain);
        assert_eq!(P::transition(client_x, likely_y), T::Reject);
        assert_eq!(P::transition(client_x, replica_a), T::Reject);
        assert_eq!(P::transition(client_x, P::Unknown), T::Retain);
    }
}
