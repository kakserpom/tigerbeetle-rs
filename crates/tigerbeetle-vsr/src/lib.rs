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
pub mod schema;
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
/// TODO(port): src/vsr.zig Checkpoint — trigger_for_checkpoint, durable, ops diagram test.
/// Port of `vsr.BlockReference` (src/vsr.zig) — identifies a block by checksum and address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockReference {
    pub checksum: u128,
    pub address: u64,
}

pub mod checkpoint {
    /// Port of `vsr.Checkpoint.valid`.
    #[must_use]
    pub fn valid(op: u64) -> bool {
        // Divide by `lsm_compaction_ops` instead of `vsr_checkpoint_ops`:
        // although today in practice checkpoints are evenly spaced, the LSM layer doesn't assume
        // that. LSM allows any bar boundary to become a checkpoint which happens, e.g., in the
        // tree fuzzer.
        op == 0 || (op + 1).is_multiple_of(tigerbeetle_core::constants::LSM_COMPACTION_OPS as u64)
    }
}

// Pin core's header-size placeholder to the real struct now that it exists:
const _: () = assert!(tigerbeetle_core::constants::HEADER_SIZE == message_header::SIZE);
