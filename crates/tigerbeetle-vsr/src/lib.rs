//! Port of `src/vsr/` — Viewstamped Replication consensus.
//! Bottom-up, one subsystem at a time; the Zig source is the specification.

#![allow(clippy::doc_markdown)] // doc comments are ported verbatim from upstream

pub use tigerbeetle_core::checksum::{ChecksumStream, checksum};

use crate::command::Command;
use tigerbeetle_core::constants::{PIPELINE_PREPARE_QUEUE_MAX, VIEW_HEADERS_MAX};

pub mod checkpoint_trailer;
pub mod client_replies;
pub mod client_sessions;
pub mod clock;
pub mod command;
pub mod fault_detector;
pub mod grid;
pub mod journal;
pub mod marzullo;
pub mod message;
pub mod message_buffer;
pub mod message_header;
pub mod message_pool;
pub mod multi_batch;
pub mod multiversion;
pub mod repair_budget;
pub mod schema;
pub mod storage;
pub mod superblock;
pub mod superblock_quorums;
pub mod testing;
pub mod time;

/// The version of our Viewstamped Replication protocol in use, including customizations.
/// For backwards compatibility through breaking changes (e.g. upgrading checksums/ciphers).
pub const VERSION: u16 = 0;

/// Port of `vsr.Operation`.
///
/// This type exists to avoid making the Header type dependent on the state
/// machine used, which would cause awkward circular type dependencies.
///
/// DEVIATION: upstream is a Zig non-exhaustive enum (`_` catch-all), so an `Operation` may hold
/// any `u8` — including state-machine operations (≥`vsr_operations_reserved`). Rust enums cannot
/// carry such gaps without a payload, so this port is a `#[repr(transparent)]` newtype over
/// `u8` with the known variants as associated constants; pattern matching becomes comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Operation(pub u8);

impl Operation {
    /// The value 0 is reserved to prevent a spurious zero from being interpreted as an operation.
    pub const RESERVED: Self = Self(0);
    /// The value 1 is reserved to initialize the cluster.
    pub const ROOT: Self = Self(1);
    /// The value 2 is reserved to register a client session with the cluster.
    pub const REGISTER: Self = Self(2);
    /// The value 3 is reserved for reconfiguration request.
    pub const RECONFIGURE: Self = Self(3);
    /// The value 4 is reserved for pulse request.
    pub const PULSE: Self = Self(4);
    /// The value 5 is reserved for release-upgrade requests.
    pub const UPGRADE: Self = Self(5);
    /// The value 6 is reserved for noop requests.
    pub const NOOP: Self = Self(6);

    // Operations <vsr_operations_reserved are reserved for the control plane.
    // Operations ≥vsr_operations_reserved are available for the state machine.

    #[must_use]
    pub fn vsr_reserved(self) -> bool {
        self.0 < tigerbeetle_core::constants::VSR_OPERATIONS_RESERVED
    }

    /// Port of `Operation.valid(StateMachineOperation)` for the vsr-reserved half only:
    /// whether this ordinal names one of the control-plane operations above.
    #[must_use]
    pub fn vsr_known(self) -> bool {
        matches!(
            self,
            Self::RESERVED
                | Self::ROOT
                | Self::REGISTER
                | Self::RECONFIGURE
                | Self::PULSE
                | Self::UPGRADE
                | Self::NOOP
        )
    }
}

/// Port of `vsr.RegisterRequest`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct RegisterRequest {
    /// When command=request, batch_size_limit = 0.
    /// When command=prepare, batch_size_limit > 0 and batch_size_limit ≤ message_body_size_max.
    /// (Note that this does *not* include the `@sizeOf(Header)`.)
    pub batch_size_limit: u32,
    pub reserved: [u8; 252],
}

const _: () = assert!(size_of::<RegisterRequest>() == 256);

/// Port of `vsr.RegisterResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct RegisterResult {
    pub batch_size_limit: u32,
    pub reserved: [u8; 60],
}

const _: () = assert!(size_of::<RegisterResult>() == 64);

/// Port of `vsr.UpgradeRequest`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct UpgradeRequest {
    pub release: crate::multiversion::Release,
    pub reserved: [u8; 12],
}

const _: () = assert!(size_of::<UpgradeRequest>() == 16);

/// Port of `vsr.BlockRequest`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BlockRequest {
    pub block_checksum: u128,
    pub block_address: u64,
    pub reserved: [u8; 8],
}

const _: () = assert!(size_of::<BlockRequest>() == 32);

/// Body of the builtin operation=.reconfigure request.
///
/// TODO(port): src/vsr.zig ReconfigurationRequest.validate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ReconfigurationRequest {
    /// The new list of members.
    ///
    /// Request is rejected if it is not a permutation of an existing list of members.
    /// This is done to separate different failure modes of physically adding a new machine to the
    /// cluster as opposed to logically changing the set of machines participating in quorums.
    pub members: Members,
    /// The new epoch.
    ///
    /// Request is rejected if it isn't exactly current epoch + 1, to protect from operator errors.
    /// Although there's already an `epoch` field in vsr.Header, we don't want to rely on that for
    /// reconfiguration itself, as it is updated automatically by the clients, and here we need
    /// a manual confirmation from the operator.
    pub epoch: u32,
    /// The new replica count.
    ///
    /// At the moment, we require this to be equal to the old count.
    pub replica_count: u8,
    /// The new standby count.
    ///
    /// At the moment, we require this to be equal to the old count.
    pub standby_count: u8,
    pub reserved: [u8; 54],
    /// The result of this request. Set to zero by the client and filled-in by the primary when it
    /// accepts a reconfiguration request.
    pub result: ReconfigurationResult,
}

/// Port of `vsr.Members`.
pub type Members = [u128; tigerbeetle_core::constants::MEMBERS_MAX];

/// Port of `vsr.ReconfigurationResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ReconfigurationResult {
    Reserved = 0,
    /// Reconfiguration request is valid.
    /// The cluster is guaranteed to transition to the new epoch with the specified configuration.
    Ok = 1,

    /// replica_count must be at least 1.
    ReplicaCountZero = 2,
    ReplicaCountMaxExceeded = 3,
    StandbyCountMaxExceeded = 4,
    MembersInvalid = 5,
    /// The number of non-zero entries in Members array does not match the sum of replica_count
    /// and standby_count.
    MembersCountInvalid = 6,
    /// A reserved field is non-zero.
    ReservedField = 7,
    /// result must be set to zero (.reserved).
    ResultMustBeReserved = 8,
    /// epoch is in the past (smaller than the current epoch).
    EpochInThePast = 9,
    /// epoch is too far in the future (larger than current epoch + 1).
    EpochInTheFuture = 10,
    /// Reconfiguration changes the number of replicas, that is not currently supported.
    DifferentReplicaCount = 11,
    /// Reconfiguration changes the number of standbys, that is not currently supported.
    DifferentStandbyCount = 12,
    /// members must be a permutation of the current set of cluster members.
    DifferentMemberSet = 13,
    /// epoch is equal to the current epoch and configuration is the same.
    ConfigurationApplied = 14,
    /// A conflicting reconfiguration request was accepted.
    ConfigurationConflict = 15,
    /// The request is valid, but there's no need to advance the epoch, because / configuration
    /// exactly matches the current one.
    ConfigurationIsNoOp = 16,
}

const _: () = assert!(size_of::<ReconfigurationRequest>() == 256);

/// Port of `vsr.Peer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Peer {
    Unknown,
    Replica { replica: u8 },
    Client { id: u128 },
    ClientLikely { id: u128 },
}

/// The result of [`Peer::transition`] (upstream anonymous enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PeerTransition {
    Retain,
    Update,
    Reject,
}

impl Peer {
    /// Port of `vsr.Peer.transition`.
    #[must_use]
    pub fn transition(old: Self, new: Self) -> PeerTransition {
        match old {
            // An unknown connection can be claimed by any peer type.
            Self::Unknown => PeerTransition::Update,

            // Receiving requests from two different clients on the same connection implies
            // that we are talking to a replica. However, as we don't know which one, we
            // retain this as a connection to a client, for simplicity.
            Self::ClientLikely { id } => match new {
                // Receiving requests from two different clients on the same connection implies
                // that we are talking to a replica. However, as we don't know which one, we
                // retain this as a connection to a client, for simplicity.
                Self::ClientLikely { .. } | Self::Unknown => PeerTransition::Retain,
                Self::Client { id: new_id } => {
                    if id == new_id {
                        PeerTransition::Update
                    } else {
                        PeerTransition::Reject
                    }
                }
                Self::Replica { .. } => PeerTransition::Update,
            },

            Self::Replica { replica } => match new {
                Self::Replica { replica: new_replica } => {
                    if replica == new_replica {
                        PeerTransition::Retain
                    } else {
                        PeerTransition::Reject
                    }
                }
                Self::Client { .. } => PeerTransition::Reject,
                Self::ClientLikely { .. } | Self::Unknown => PeerTransition::Retain,
            },

            Self::Client { id } => {
                match new {
                    // Both client tags identify the peer by its id:
                    Self::Client { id: new_id } | Self::ClientLikely { id: new_id } => {
                        if id == new_id { PeerTransition::Retain } else { PeerTransition::Reject }
                    }
                    Self::Replica { .. } => PeerTransition::Reject,
                    Self::Unknown => PeerTransition::Retain,
                }
            }
        }
    }
}

/// Deterministically assigns replica_ids for the initial configuration.
///
/// Eventually, we want to identify replicas using random u128 ids to prevent operator errors.
/// However, that requires unergonomic two-step process for spinning a new cluster up. To avoid
/// needlessly compromising the experience until reconfiguration is fully implemented, derive
/// replica ids for the initial cluster deterministically.
///
/// Port of `vsr.root_members` (src/vsr.zig).
///
/// DEVIATION: upstream packs an `extern struct { u128 align(1), u128 align(1), u8 }` (33 bytes,
/// no padding) and hashes its memory. Rust padding would differ, so the seed is serialized
/// manually as little-endian fields in declaration order.
///
/// # Panics
/// Panics if the derived member ids are not valid (upstream asserts; a checksum collision
/// would be required).
#[must_use]
pub fn root_members(cluster: u128) -> Members {
    const SEED_SIZE: usize = 16 + 16 + 1;
    assert_eq!(SEED_SIZE, 33); // upstream comptime: @sizeOf(IdSeed) == 33

    let mut result: Members = [0; tigerbeetle_core::constants::MEMBERS_MAX];
    // IdSeed { cluster_config_checksum, cluster, replica }, little-endian, no padding.
    let mut seed = [0u8; SEED_SIZE];
    seed[..16]
        .copy_from_slice(&tigerbeetle_core::constants::CONFIG.cluster.checksum().to_le_bytes());
    seed[16..32].copy_from_slice(&cluster.to_le_bytes());
    for (replica, slot) in result.iter_mut().enumerate() {
        seed[32] = u8::try_from(replica).unwrap_or_else(|_| unreachable!("members_max fits u8"));
        *slot = checksum(&seed);
    }

    assert!(valid_members(&result));
    result
}

/// Port of `vsr.valid_members` (src/vsr.zig).
#[must_use]
pub fn valid_members(members: &Members) -> bool {
    for (i, &replica_i) in members.iter().enumerate() {
        for &replica_j in &members[..i] {
            if replica_j == 0 && replica_i != 0 {
                return false;
            }
            if replica_j != 0 && replica_j == replica_i {
                return false;
            }
        }
    }
    true
}

/// Port of `vsr.member_index` (src/vsr.zig).
///
/// # Panics
/// Panics if `replica_id` is zero or `members` is not valid (upstream asserts).
#[must_use]
pub fn member_index(members: &Members, replica_id: u128) -> Option<u8> {
    assert!(replica_id != 0);
    assert!(valid_members(members));
    members.iter().position(|&member| member == replica_id).map(|index| match u8::try_from(index) {
        Ok(replica_index) => replica_index,
        Err(_) => unreachable!("members_max fits u8"),
    })
}

/// Port of `vsr.Checkpoint` (operation-space helpers).
///
/// TODO(port): src/vsr.zig Checkpoint — ops diagram snapshot test.
/// Port of `vsr.BlockReference` (src/vsr.zig) — identifies a block by checksum and address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockReference {
    pub checksum: u128,
    pub address: u64,
}

pub mod checkpoint {
    use tigerbeetle_core::constants::{
        LSM_COMPACTION_OPS, PIPELINE_PREPARE_QUEUE_MAX, VSR_CHECKPOINT_OPS,
    };

    /// Port of `vsr.Checkpoint.valid`.
    #[must_use]
    pub fn valid(op: u64) -> bool {
        // Divide by `lsm_compaction_ops` instead of `vsr_checkpoint_ops`:
        // although today in practice checkpoints are evenly spaced, the LSM layer doesn't assume
        // that. LSM allows any bar boundary to become a checkpoint which happens, e.g., in the
        // tree fuzzer.
        op == 0 || (op + 1).is_multiple_of(LSM_COMPACTION_OPS as u64)
    }

    /// Port of `vsr.Checkpoint.checkpoint_after`.
    ///
    /// # Panics
    /// Panics if `checkpoint` is not a valid checkpoint op (upstream asserts).
    #[must_use]
    pub fn checkpoint_after(checkpoint: u64) -> u64 {
        assert!(valid(checkpoint));

        let result = if checkpoint == 0 {
            // First wrap: op_checkpoint_next = 6-1 = 5
            // -1: vsr_checkpoint_ops is a count, result is an inclusive index.
            VSR_CHECKPOINT_OPS as u64 - 1
        } else {
            // Second wrap: op_checkpoint_next = 5+6 = 11
            // Third wrap: op_checkpoint_next = 11+6 = 17
            checkpoint + VSR_CHECKPOINT_OPS as u64
        };

        assert!((result + 1).is_multiple_of(LSM_COMPACTION_OPS as u64));
        assert!(valid(result));

        result
    }

    /// Port of `vsr.Checkpoint.trigger_for_checkpoint`.
    ///
    /// # Panics
    /// Panics if `checkpoint` is not a valid checkpoint op (upstream asserts).
    #[must_use]
    pub fn trigger_for_checkpoint(checkpoint: u64) -> Option<u64> {
        assert!(valid(checkpoint));

        if checkpoint == 0 { None } else { Some(checkpoint + LSM_COMPACTION_OPS as u64) }
    }

    /// Port of `vsr.Checkpoint.prepare_max_for_checkpoint`.
    ///
    /// # Panics
    /// Panics if `checkpoint` is not a valid checkpoint op (upstream asserts).
    #[must_use]
    pub fn prepare_max_for_checkpoint(checkpoint: u64) -> Option<u64> {
        assert!(valid(checkpoint));

        trigger_for_checkpoint(checkpoint)
            .map(|trigger| trigger + u64::from(PIPELINE_PREPARE_QUEUE_MAX) * 2)
    }

    /// Port of `vsr.Checkpoint.durable`.
    ///
    /// # Panics
    /// Panics if `checkpoint` is not a valid checkpoint op (upstream asserts).
    #[must_use]
    pub fn durable(checkpoint: u64, commit: u64) -> bool {
        assert!(valid(checkpoint));

        match trigger_for_checkpoint(checkpoint) {
            Some(trigger) => commit > trigger + u64::from(PIPELINE_PREPARE_QUEUE_MAX),
            None => true,
        }
    }
}

// Pin core's header-size placeholder to the real struct now that it exists:
const _: () = assert!(tigerbeetle_core::constants::HEADER_SIZE == message_header::SIZE);

/// Port of `vsr.Zone` (src/vsr.zig): layout of the data file's storage zones.
///
/// DEVIATION: an enum with methods rather than upstream's Zig enum-with-comptime-table; the
/// zone order and size formulas match upstream exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Zone {
    Superblock,
    WalHeaders,
    WalPrepares,
    ClientReplies,
    // Add padding between `client_replies` and `grid`, to make sure grid blocks are aligned to
    // block size and not just to sector size. Aligning blocks this way makes it more likely that
    // they are aligned to the underlying physical sector size. This padding is zeroed during
    // format, but isn't used otherwise.
    GridPadding,
    Grid,
}

/// Add padding between `client_replies` and `grid` so that grid blocks start aligned to the
/// block size (upstream: `Zone.size_grid_padding`).
#[must_use]
pub const fn zone_size_grid_padding() -> usize {
    let grid_start_unaligned = tigerbeetle_core::constants::CLIENT_REPLIES_SIZE
        + tigerbeetle_core::constants::JOURNAL_SIZE_PREPARES
        + tigerbeetle_core::constants::JOURNAL_SIZE_HEADERS
        + superblock::SUPERBLOCK_ZONE_SIZE;
    let grid_start_aligned = tigerbeetle_core::stdx::align_forward(
        grid_start_unaligned,
        tigerbeetle_core::constants::BLOCK_SIZE,
    );
    grid_start_aligned - grid_start_unaligned
}

impl Zone {
    /// Size of the zone, or `None` for the open-ended [`Zone::Grid`] (upstream: `Zone.size`).
    #[must_use]
    pub fn size(self) -> Option<u64> {
        use tigerbeetle_core::constants::{
            CLIENT_REPLIES_SIZE, JOURNAL_SIZE_HEADERS, JOURNAL_SIZE_PREPARES,
        };
        match self {
            Self::Superblock => Some(superblock::SUPERBLOCK_ZONE_SIZE as u64),
            Self::WalHeaders => Some(JOURNAL_SIZE_HEADERS as u64),
            Self::WalPrepares => Some(JOURNAL_SIZE_PREPARES as u64),
            Self::ClientReplies => Some(CLIENT_REPLIES_SIZE as u64),
            Self::GridPadding => Some(zone_size_grid_padding() as u64),
            Self::Grid => None,
        }
    }

    /// Start offset of the zone within the data file (upstream: `Zone.start`).
    #[must_use]
    pub fn start(self) -> u64 {
        use tigerbeetle_core::constants::{
            CLIENT_REPLIES_SIZE, JOURNAL_SIZE_HEADERS, JOURNAL_SIZE_PREPARES,
        };
        match self {
            Self::Superblock => 0,
            Self::WalHeaders => superblock::SUPERBLOCK_ZONE_SIZE as u64,
            Self::WalPrepares => Self::WalHeaders.start() + JOURNAL_SIZE_HEADERS as u64,
            Self::ClientReplies => Self::WalPrepares.start() + JOURNAL_SIZE_PREPARES as u64,
            Self::GridPadding => Self::ClientReplies.start() + CLIENT_REPLIES_SIZE as u64,
            Self::Grid => Self::GridPadding.start() + zone_size_grid_padding() as u64,
        }
    }

    /// Translates a logical offset within a zone into a file offset (upstream: `Zone.offset`).
    ///
    /// # Panics
    /// Panics if the zone has a fixed size and `offset_logical` exceeds it (upstream asserts).
    #[must_use]
    pub fn offset(self, offset_logical: u64) -> u64 {
        if let Some(zone_size) = self.size() {
            assert!(offset_logical < zone_size);
        }

        self.start() + offset_logical
    }

    /// Ensures that the read or write is aligned correctly for Direct I/O.
    /// If this is not the case, then the underlying syscall will return EINVAL.
    /// We check this only at the start of a read or write because the physical sector size may be
    /// less than our logical sector size so that partial IOs then leave us no longer aligned.
    /// (upstream: `Zone.verify_iop`)
    ///
    /// # Panics
    /// Panics if any of the alignment requirements are violated.
    pub fn verify_iop(self, buffer: &[u8], offset_in_zone: u64) {
        if let Some(zone_size) = self.size() {
            assert!(u64::try_from(buffer.len()).is_ok_and(|len| offset_in_zone + len <= zone_size));
        }
        assert!(
            (buffer.as_ptr() as usize).is_multiple_of(tigerbeetle_core::constants::SECTOR_SIZE),
            "buffer must be sector-aligned"
        );
        assert!(buffer.len().is_multiple_of(tigerbeetle_core::constants::SECTOR_SIZE));
        assert!(!buffer.is_empty());
        let offset_in_storage = self.offset(offset_in_zone);
        assert!(offset_in_storage.is_multiple_of(tigerbeetle_core::constants::SECTOR_SIZE as u64));
        if self == Self::Grid {
            assert!(
                offset_in_storage.is_multiple_of(tigerbeetle_core::constants::BLOCK_SIZE as u64)
            );
        }
    }
}

#[cfg(test)]
mod zone_tests {
    use crate::{Zone, zone_size_grid_padding};
    use tigerbeetle_core::constants::{BLOCK_SIZE, SECTOR_SIZE};

    #[test]
    fn zone_starts_are_sector_aligned_and_ordered() {
        let zones = [
            Zone::Superblock,
            Zone::WalHeaders,
            Zone::WalPrepares,
            Zone::ClientReplies,
            Zone::GridPadding,
            Zone::Grid,
        ];

        let mut previous_end = 0;
        for &zone in &zones {
            let start = zone.start();
            assert!(start >= previous_end);
            assert_eq!(start % SECTOR_SIZE as u64, 0, "{zone:?} start not sector-aligned");
            if let Some(size) = zone.size() {
                assert_eq!(size % SECTOR_SIZE as u64, 0, "{zone:?} size not sector-aligned");
                previous_end = start + size;
            } else {
                previous_end = start;
            }
        }

        assert_eq!(Zone::Grid.start() % BLOCK_SIZE as u64, 0);
    }

    #[test]
    fn zone_offset_bounds_check() {
        assert_eq!(Zone::Superblock.offset(0), 0);
        let wal_headers_start = Zone::WalHeaders.start();
        assert_eq!(Zone::WalHeaders.offset(16), wal_headers_start + 16,);
        let grid_padding = zone_size_grid_padding();
        let client_replies_size =
            Zone::ClientReplies.size().unwrap_or_else(|| unreachable!("sized"));
        assert_eq!(Zone::GridPadding.size(), Some(grid_padding as u64));
        assert_eq!(Zone::Grid.start(), Zone::GridPadding.start() + grid_padding as u64);
        assert_eq!(Zone::Grid.size(), None);
        let _ = client_replies_size;
    }
}

#[cfg(test)]
mod members_tests {
    use super::{Members, member_index, root_members, valid_members};
    use tigerbeetle_core::constants::MEMBERS_MAX;

    /// Port of upstream test "vsr: root_members valid" (src/vsr.zig).
    #[test]
    fn root_members_are_valid_and_distinct() {
        let big = u128::from(u64::MAX);
        for cluster in [0u128, 1, 2, big, big << 64] {
            let members = root_members(cluster);
            assert!(valid_members(&members));

            let non_zero = members.iter().filter(|&&m| m != 0).count();
            assert_eq!(non_zero, MEMBERS_MAX);

            // All ids distinct.
            for i in 0..MEMBERS_MAX {
                for j in 0..i {
                    assert_ne!(members[i], members[j]);
                }
            }

            // Deterministic.
            assert_eq!(root_members(cluster), members);
        }
    }

    #[test]
    fn root_members_depend_on_cluster() {
        assert_ne!(root_members(0), root_members(1));
    }

    /// Golden values produced by running upstream `vsr.root_members()` (Zig 0.14.1,
    /// test_min config) via `reference/tigerbeetle/src/tbcross_main.zig`.
    /// Pins the AEGIS checksum, IdSeed byte layout, and endianness against upstream.
    #[test]
    fn root_members_match_upstream_zig() {
        let expected_zero = [
            "459d6840872cd4709b6b08975dc2a6bb",
            "98bd1945b69568664d59ea1827288a22",
            "2db1caea465eae11497ac773dc80fc09",
            "378bd9dfed0df6c5996f2b9f4b64df1b",
            "cc1c24b3d56ea1fdb8a5dc65f425bab7",
            "7d13f0e77af77c389c8fc7c5e6773085",
            "90689d9dd8f712be32ef0a4d0de013b7",
            "8070f283e793e1a592bf4407cf1b18cc",
            "a3f9b3795c747ab772e63a335c94cefc",
            "4fe89fb4498beae769732d5bc5f91874",
            "49aa0b4b9ca1306d1030be36f4a436de",
            "8515e6f59ba6ce02f5cea27fa04e87b0",
        ];
        let members = root_members(0);
        for (member, expected) in members.iter().zip(expected_zero) {
            assert_eq!(format!("{member:032x}"), expected);
        }

        let expected_a1b2c3 = [
            "4caed0935a541ea4cabd2cb09675414e",
            "125f138234740dc6ae256afc0ef123ea",
            "035eb83da3734f8f626aef5a74652408",
            "88defe97db4dc4faacffb4e278cd1fd1",
            "427f6aaef304ea3eb8ee974f1ced77ea",
            "d770df1931473aba661bff188aa3b3f4",
            "ebef947b2ecb5eb6cb699625209a148a",
            "6aa376781ffef044d775b23bb10984ec",
            "6ef9b37849603c83f0043ea3a134cb53",
            "2985c5add5d5adeace28f84209b5f2e8",
            "420b3771ef9b4bdb4c23e3c19b305ee3",
            "00bf7bb43a94357bd899ba6aa433d891",
        ];
        let members_a1b2c3 = root_members(0x00A1_B2C3);
        for (member, expected) in members_a1b2c3.iter().zip(expected_a1b2c3) {
            assert_eq!(format!("{member:032x}"), expected);
        }
    }

    /// Port of upstream test "vsr: member_index" (src/vsr.zig).
    #[test]
    fn finds_member_index() {
        let mut members: Members = [0; MEMBERS_MAX];
        for (i, slot) in members.iter_mut().enumerate() {
            *slot = u128::try_from(i + 1).unwrap_or_else(|_| unreachable!("fits"));
        }
        assert_eq!(member_index(&members, 1), Some(0));
        assert_eq!(
            member_index(&members, u128::try_from(MEMBERS_MAX).unwrap_or_else(|_| unreachable!())),
            Some(u8::try_from(MEMBERS_MAX - 1).unwrap_or_else(|_| unreachable!("fits")))
        );
        assert_eq!(member_index(&members, MEMBERS_MAX as u128 + 1), None);
    }
}

/// Port of `vsr.Headers` (src/vsr.zig).
pub mod headers {
    use crate::Operation;
    use crate::command::Command;
    use crate::message_header::{self, TypedHeader as _};
    use crate::multiversion::Release;
    use tigerbeetle_core::constants::VIEW_HEADERS_MAX;
    use tigerbeetle_core::stdx::bounded_array::BoundedArray;

    /// Port of `Headers.Array`.
    pub type Array = BoundedArray<message_header::Prepare, { VIEW_HEADERS_MAX as usize }>;

    /// Port of `Headers.jv_header_type`'s anonymous result enum.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum JvHeaderType {
        Blank,
        Valid,
    }

    /// Port of `Headers.jv_blank`.
    #[must_use]
    pub fn jv_blank(op: u64) -> message_header::Prepare {
        message_header::Prepare {
            command: Command::Prepare,
            release: Release::ZERO,
            operation: Operation::RESERVED,
            op,
            ..Default::default()
        }
    }

    /// Port of `Headers.jv_header_type`.
    ///
    /// # Panics
    /// Panics if a non-blank header is invalid (upstream asserts).
    #[must_use]
    pub fn jv_header_type(header: &message_header::Prepare) -> JvHeaderType {
        if *header == jv_blank(header.op) {
            return JvHeaderType::Blank;
        }

        assert!(header.valid_checksum());
        assert_eq!(header.command, Command::Prepare);
        assert_ne!(header.operation, Operation::RESERVED);
        assert!(header.invalid().is_none());
        JvHeaderType::Valid
    }
}

/// Port of `vsr.ViewChangeCommand`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewChangeCommand {
    JoinView,
    View,
}

/// Port of `vsr.ViewChangeHeadersSlice`.
#[derive(Clone, Copy, Debug)]
pub struct ViewChangeHeadersSlice<'a> {
    pub command: ViewChangeCommand,
    /// Headers are ordered from high-to-low op.
    pub slice: &'a [message_header::Prepare],
}

impl<'a> ViewChangeHeadersSlice<'a> {
    /// # Panics
    /// Panics if the headers do not satisfy [`Self::verify`] (upstream asserts in `init`).
    #[must_use]
    pub fn init(command: ViewChangeCommand, slice: &'a [message_header::Prepare]) -> Self {
        let headers = Self { command, slice };
        headers.verify();
        headers
    }

    /// # Panics
    /// Panics if the header chain is malformed for the given command (upstream asserts).
    pub fn verify(&self) {
        assert!(!self.slice.is_empty());
        assert!(self.slice.len() <= VIEW_HEADERS_MAX as usize);

        let head = &self.slice[0];
        // A JV's head op is never a gap or faulty.
        // A View never includes gaps or faulty headers.
        assert_eq!(headers::jv_header_type(head), headers::JvHeaderType::Valid);

        let mut child = head;
        for (index, header) in self.slice.iter().enumerate().skip(1) {
            assert_eq!(header.command, Command::Prepare);
            // maybe(header.operation == .reserved): upstream documents that only the head must be
            // non-reserved; gaps elsewhere are allowed.
            assert!(header.op < child.op);

            // JV: Ops are consecutive (with explicit blank headers).
            // View: The first "pipeline + 1" ops of the View are consecutive.
            if self.command == ViewChangeCommand::JoinView
                || (self.command == ViewChangeCommand::View
                    && index < PIPELINE_PREPARE_QUEUE_MAX as usize + 1)
            {
                assert_eq!(header.op, head.op - index as u64);
            }

            match headers::jv_header_type(header) {
                headers::JvHeaderType::Blank => {
                    // We can't verify that View headers contain no gaps headers here:
                    // superblock.checkpoint could make .join_view headers durable instead of
                    // .view headers when view == log_view (see `commit_checkpoint_superblock`
                    // in `replica.zig`). When these headers are loaded from the superblock on
                    // startup, they are considered to be .view headers (see `view_headers` in
                    // `superblock.zig`).
                    continue; // Don't update "child".
                }
                headers::JvHeaderType::Valid => {
                    assert!(header.view <= child.view);
                    assert!(header.timestamp < child.timestamp);
                    if header.op + 1 == child.op {
                        assert_eq!(header.checksum, child.parent);
                    }
                }
            }
            child = header;
        }
    }

    /// Returns the range of possible views (of prepare, not commit) for a message that is part of
    /// the same log_view as these headers.
    ///
    /// - When these are JV headers for a log_view=V, we must be in view_change status working to
    ///   transition to a view beyond V. So we will never prepare anything else as part of view V.
    /// - When these are View headers for a log_view=V, we can continue to add to them (by preparing
    ///   more ops), but those ops will always be part of the log_view. If they were prepared during
    ///   a view prior to the log_view, they would already be part of the headers.
    ///
    /// # Panics
    /// Panics if `op` falls inside a gap not bounded by two valid headers (upstream: unreachable).
    #[must_use]
    pub fn view_for_op(&self, op: u64, log_view: u32) -> ViewRange {
        let header_newest = &self.slice[0];
        let oldest_index = {
            let mut oldest: Option<usize> = None;
            for (index, header) in self.slice.iter().enumerate() {
                match headers::jv_header_type(header) {
                    headers::JvHeaderType::Blank => assert!(index > 0),
                    headers::JvHeaderType::Valid => oldest = Some(index),
                }
            }
            oldest.unwrap_or_else(|| unreachable!("head is always valid"))
        };
        let header_oldest = &self.slice[oldest_index];
        assert!(header_newest.view <= log_view);
        assert!(header_newest.view >= header_oldest.view);
        assert!(header_newest.op >= header_oldest.op);

        if op < header_oldest.op {
            return ViewRange { min: 0, max: header_oldest.view };
        }
        if op > header_newest.op {
            return ViewRange { min: log_view, max: log_view };
        }

        for header in self.slice {
            if headers::jv_header_type(header) == headers::JvHeaderType::Valid && header.op == op {
                return ViewRange { min: header.view, max: header.view };
            }
        }

        let mut header_next = &self.slice[0];
        assert_eq!(headers::jv_header_type(header_next), headers::JvHeaderType::Valid);

        for header_prev in &self.slice[1..] {
            if headers::jv_header_type(header_prev) == headers::JvHeaderType::Valid {
                if header_prev.op < op && op < header_next.op {
                    return ViewRange { min: header_prev.view, max: header_next.view };
                }
                header_next = header_prev;
            }
        }
        unreachable!("op is between the oldest and newest ops");
    }
}

/// Port of `ViewChangeHeadersSlice.ViewRange`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewRange {
    /// Inclusive.
    pub min: u32,
    /// Inclusive.
    pub max: u32,
}

impl ViewRange {
    #[must_use]
    pub fn contains(self, view: u32) -> bool {
        self.min <= view && view <= self.max
    }
}

/// Port of `vsr.ViewChangeHeadersArray`: the headers of a View or JV message.
#[derive(Clone, Debug)]
pub struct ViewChangeHeadersArray {
    pub command: ViewChangeCommand,
    pub array: headers::Array,
}

impl ViewChangeHeadersArray {
    /// # Panics
    /// Panics if the constructed headers do not satisfy [`headers`-level verification]
    /// (upstream asserts).
    #[must_use]
    pub fn root(cluster: u128) -> Self {
        Self::init(ViewChangeCommand::View, &[message_header::Prepare::root(cluster)])
    }

    /// # Panics
    /// Panics if `slice` exceeds [`VIEW_HEADERS_MAX`] or fails verification (upstream asserts).
    #[must_use]
    pub fn init(command: ViewChangeCommand, slice: &[message_header::Prepare]) -> Self {
        let Ok(array) = headers::Array::from_slice(slice) else {
            unreachable!("slice fits view_headers_max")
        };
        let headers = Self { command, array };
        headers.verify();
        headers
    }

    /// # Panics
    /// Panics if the replaced headers fail verification (upstream asserts).
    pub fn replace(&mut self, command: ViewChangeCommand, slice: &[message_header::Prepare]) {
        self.command = command;
        self.array.clear();
        for header in slice {
            self.array.push(*header);
        }
        self.verify();
    }

    /// We don't do comprehensive validation here — assume that verify() will be called
    /// after any series of appends.
    ///
    /// # Panics
    /// Panics if the array is already full (upstream asserts).
    pub fn append(&mut self, header: &message_header::Prepare) {
        self.array.push(*header);
    }

    /// # Panics
    /// Panics unless appending join-view blanks (upstream asserts).
    pub fn append_blank(&mut self, op: u64) {
        assert_eq!(self.command, ViewChangeCommand::JoinView);
        assert!(self.array.count() > 0);
        self.array.push(headers::jv_blank(op));
    }

    /// # Panics
    /// Panics if the header chain is malformed for the given command (upstream asserts).
    pub fn verify(&self) {
        ViewChangeHeadersSlice { command: self.command, slice: self.array.slice() }.verify();
    }
}

/// Port of `vsr.Snapshot`.
pub mod snapshot {
    /// A table with TableInfo.snapshot_min=S was written during some commit with op<S.
    /// A block with snapshot_min=S is definitely readable at op=S.
    #[must_use]
    pub fn readable_at_commit(op: u64) -> u64 {
        // TODO: This is going to become more complicated when snapshot numbers match the op
        // acquiring the snapshot.
        op + 1
    }
}

#[cfg(test)]
mod view_change_headers_tests {
    #![allow(clippy::unwrap_used)]

    use super::{ViewChangeCommand, ViewChangeHeadersSlice, ViewRange, headers};
    use crate::Operation;
    use crate::command::Command;
    use crate::message_header::{self, TypedHeader as _};
    use crate::multiversion::Release;
    use tigerbeetle_core::constants::VSR_OPERATIONS_RESERVED;

    /// Port of upstream test "Headers.ViewChangeSlice.view_for_op".
    #[test]
    fn headers_view_change_slice_view_for_op() {
        let mut headers_array = [
            message_header::Prepare {
                checksum: 0,
                client: 6,
                request: 7,
                command: Command::Prepare,
                release: Release::MINIMUM,
                operation: Operation(VSR_OPERATIONS_RESERVED + 8),
                op: 9,
                view: 10,
                timestamp: 11,
                ..Default::default()
            },
            headers::jv_blank(8),
            headers::jv_blank(7),
            message_header::Prepare {
                checksum: 0,
                client: 3,
                request: 4,
                command: Command::Prepare,
                release: Release::MINIMUM,
                operation: Operation(VSR_OPERATIONS_RESERVED + 5),
                op: 6,
                view: 7,
                timestamp: 8,
                ..Default::default()
            },
            headers::jv_blank(5),
        ];

        headers_array[0].set_checksum();
        headers_array[3].set_checksum();

        let headers = ViewChangeHeadersSlice::init(ViewChangeCommand::JoinView, &headers_array);
        assert_eq!(headers.view_for_op(11, 12), ViewRange { min: 12, max: 12 });
        assert_eq!(headers.view_for_op(10, 12), ViewRange { min: 12, max: 12 });
        assert_eq!(headers.view_for_op(9, 12), ViewRange { min: 10, max: 10 });
        assert_eq!(headers.view_for_op(8, 12), ViewRange { min: 7, max: 10 });
        assert_eq!(headers.view_for_op(7, 12), ViewRange { min: 7, max: 10 });
        assert_eq!(headers.view_for_op(6, 12), ViewRange { min: 7, max: 7 });
        assert_eq!(headers.view_for_op(5, 12), ViewRange { min: 0, max: 7 });
        assert_eq!(headers.view_for_op(0, 12), ViewRange { min: 0, max: 7 });
    }
}
