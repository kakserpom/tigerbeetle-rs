//! Port of `src/vsr/` — Viewstamped Replication consensus.
//! Bottom-up, one subsystem at a time; the Zig source is the specification.

#![allow(clippy::doc_markdown)] // doc comments are ported verbatim from upstream

pub use tigerbeetle_core::checksum::{ChecksumStream, checksum};

pub mod checkpoint_trailer;
pub mod clock;
pub mod command;
pub mod fault_detector;
pub mod marzullo;
pub mod message;
pub mod message_buffer;
pub mod message_header;
pub mod message_pool;
pub mod multi_batch;
pub mod multiversion;
pub mod repair_budget;
pub mod schema;
pub mod superblock;
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
