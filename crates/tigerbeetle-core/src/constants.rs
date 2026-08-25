//! Constants are the configuration that the code actually imports — they include:
//! - all of the configuration values (flattened)
//! - derived configuration values,
//!
//! Port of `src/constants.zig`. Upstream `comptime { assert(...) }` blocks become compile-time
//! assertions (`const _: () = assert!(...)`), evaluated on every build.
//!
//! Deferred, pending their subsystem ports:
//! - `semver` / release metadata: TODO(port) src/constants.zig semver, needs multiversion.
//! - `vsr_releases_max * @sizeOf(vsr.Release) <= message_body_size_max`: TODO(port) vsr.Release.
//! - `message_body_size_max >= @sizeOf(vsr.ReconfigurationRequest)`: TODO(port) src/vsr.zig.
//! - comptime parse of `ADDRESS`: TODO(port) src/vsr.zig parse_address.

#![allow(clippy::cast_possible_truncation)] // widths mirror upstream's declared integer types
#![allow(clippy::doc_markdown)] // doc comments are ported verbatim from upstream
#![allow(clippy::cast_possible_wrap)] // narrow casts mirror upstream's declared types

use crate::config;
use crate::stdx::{Duration, div_ceil};

/// The size of the vsr `Header` frame (upstream asserts `@sizeOf(Header) == 256`).
///
/// Kept as a literal here (core must not depend on vsr); `tigerbeetle-vsr::message_header`
/// statically asserts this equals the real `size_of::<Header>()`.
pub const HEADER_SIZE: usize = 256;
/// The size of the vsr `CheckpointState` struct (asserted 1024 upstream).
///
/// Kept as a literal here (core must not depend on vsr); TODO(port): pin statically once
/// `superblock.CheckpointState` is ported.
pub const CHECKPOINT_STATE_SIZE: usize = 1024;
/// TODO(port): src/vsr.zig BlockRequest (asserted 32 upstream).
const BLOCK_REQUEST_SIZE: usize = 32;

/// Const-context `min` (`Ord::min` is not const-callable yet).
const fn umin(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}

/// Const-context `max`.
const fn umax(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

pub const CONFIG: config::Config = config::configs::CURRENT;

/// The maximum number of replicas allowed in a cluster.
pub const REPLICAS_MAX: usize = 6;
/// The maximum number of standbys allowed in a cluster.
pub const STANDBYS_MAX: usize = 6;
/// The maximum number of cluster members (either standbys or active replicas).
pub const MEMBERS_MAX: usize = REPLICAS_MAX + STANDBYS_MAX;

/// All operations <vsr_operations_reserved are reserved for the control protocol.
/// All operations ≥vsr_operations_reserved are available for the state machine.
pub const VSR_OPERATIONS_RESERVED: u8 = 128;
// Upstream asserts `vsr_operations_reserved <= maxInt(u8)`; here the type system enforces that.

/// The checkpoint interval is chosen to be the highest possible value that satisfies the
/// constraints described below.
pub const VSR_CHECKPOINT_OPS: usize = JOURNAL_SLOT_COUNT as usize
    - LSM_COMPACTION_OPS
    - LSM_COMPACTION_OPS * div_ceil(PIPELINE_PREPARE_QUEUE_MAX as usize * 2, LSM_COMPACTION_OPS);

const _: () = {
    // Invariant: to guarantee durability, a log entry from a previous checkpoint can be overwritten
    // only when there is a quorum of replicas at the next checkpoint.
    //
    // This assert guarantees that when a prepare gets bumped from the log, there is a prepare
    // _committed_ on top of the next checkpoint, which in turn guarantees the existence of a
    // checkpoint quorum.
    //
    // More specifically, the checkpoint interval must be less than the WAL length by (at least) the
    // sum of:
    // - `LSM_COMPACTION_OPS`: Ensure that the final batch of entries immediately preceding a
    //   checkpoint trigger is not overwritten by the following checkpoint's entries.
    // - `2 * PIPELINE_PREPARE_QUEUE_MAX` (rounded up to the nearest lsm_compaction_ops multiple):
    //    This margin ensures that the entries prepared immediately following a checkpoint's prepare
    //    max never overwrite an entry from the previous WAL wrap until a quorum of replicas has
    //    reached that checkpoint.
    assert!(
        VSR_CHECKPOINT_OPS + LSM_COMPACTION_OPS + PIPELINE_PREPARE_QUEUE_MAX as usize * 2
            <= JOURNAL_SLOT_COUNT as usize
    );
    assert!(VSR_CHECKPOINT_OPS >= PIPELINE_PREPARE_QUEUE_MAX as usize);
    assert!(VSR_CHECKPOINT_OPS >= LSM_COMPACTION_OPS);
    assert!(VSR_CHECKPOINT_OPS.is_multiple_of(LSM_COMPACTION_OPS));
};

/// The maximum number of clients allowed per cluster, where each client has a unique 128-bit ID.
/// This impacts the amount of memory allocated at initialization by the server.
/// This determines the size of the VR client table used to cache replies to clients by client ID.
/// Each client has one entry in the VR client table to store the latest `message_size_max` reply.
/// Client ID 0 which is used by primary for pulse and upgrade request, is not counted.
pub const CLIENTS_MAX: u32 = CONFIG.cluster.clients_max;

const _: () = assert!(CLIENTS_MAX as usize >= config::ConfigCluster::CLIENTS_MAX_MIN);

/// The maximum number of release versions (upgrade candidates) that can be advertised by a replica
/// in each ping message body.
pub const VSR_RELEASES_MAX: u32 = CONFIG.cluster.vsr_releases_max;

/// The maximum cumulative size of a final TigerBeetle output binary - including potential past
/// releases and metadata.
///
/// DEVIATION: upstream takes `{macos, debug}` options; identical semantics here as a `const fn`.
#[must_use]
pub const fn multiversion_binary_platform_size_max(macos: bool, debug: bool) -> u64 {
    // {Linux, Windows} get the base value. macOS gets 2x since it has universal binaries. All cases
    // get a further 2x in debug.
    let mut size_max = CONFIG.process.multiversion_binary_platform_size_max as u64;
    if macos {
        size_max *= 2;
    }
    if debug {
        size_max *= 2;
    }
    size_max
}

/// The maximum size, like above, but for any platform.
pub const MULTIVERSION_BINARY_SIZE_MAX: u64 =
    CONFIG.process.multiversion_binary_platform_size_max as u64 * 2 * 2;

const _: () =
    assert!(multiversion_binary_platform_size_max(true, true) <= MULTIVERSION_BINARY_SIZE_MAX);

pub const MULTIVERSION_POLL_INTERVAL: Duration = CONFIG.process.multiversion_poll_interval;

const _: () = {
    assert!(VSR_RELEASES_MAX >= 2);
    // TODO(port): src/constants.zig — assert(vsr_releases_max * @sizeOf(vsr.Release) <=
    // MESSAGE_BODY_SIZE_MAX).
    // The number of releases is encoded into ping headers as a u16.
    assert!(VSR_RELEASES_MAX <= u16::MAX as u32);
};

/// The maximum number of nodes required to form a quorum for replication.
/// Majority quorums are only required across view change and replication phases (not within).
/// As per Flexible Paxos, provided `quorum_replication + quorum_view_change > replicas`:
/// 1. you may increase `quorum_view_change` above a majority, so that
/// 2. you can decrease `quorum_replication` below a majority, to optimize the common case.
///
/// This improves latency by reducing the number of nodes required for synchronous replication.
/// This reduces redundancy only in the short term, asynchronous replication will still continue.
/// The size of the replication quorum is limited to the minimum of this value and ⌈replicas/2⌉.
/// The size of the view change quorum will then be automatically inferred from quorum_replication.
pub const QUORUM_REPLICATION_MAX: u8 = CONFIG.cluster.quorum_replication_max;

/// The default server port to listen on if not specified in `--addresses`:
pub const PORT: u16 = CONFIG.process.port;

/// The default network interface address to listen on if not specified in `--addresses`:
/// WARNING: Binding to all interfaces with "0.0.0.0" is dangerous and opens the server to anyone.
/// Bind to the "127.0.0.1" loopback address to accept local connections as a safe default only.
pub const ADDRESS: &str = CONFIG.process.address;

// TODO(port): src/constants.zig — comptime parse_ip4(ADDRESS) check once net code exists.

/// The default maximum amount of memory to use.
pub const MEMORY_SIZE_MAX_DEFAULT: usize = CONFIG.process.memory_size_max_default;

/// At a high level, priority for object caching is (in descending order):
///
/// 1. Accounts.
///   - 2 lookups per created transfer
///   - high temporal locality
///   - positive expected result
/// 2. Posted transfers.
///   - high temporal locality
///   - positive expected result
/// 3. Transfers. Generally don't cache these because of:
///   - low temporal locality
///   - negative expected result
///
/// The default size of the accounts in-memory cache:
/// This impacts the amount of memory allocated at initialization by the server.
pub const CACHE_ACCOUNTS_SIZE_DEFAULT: usize = CONFIG.process.cache_accounts_size_default;

/// The default size of the transfers in-memory cache:
/// This impacts the amount of memory allocated at initialization by the server.
/// We allocate more capacity than the number of transfers for a safe hash table load factor.
pub const CACHE_TRANSFERS_SIZE_DEFAULT: usize = CONFIG.process.cache_transfers_size_default;

/// The default size of the two-phase transfers in-memory cache:
/// This impacts the amount of memory allocated at initialization by the server.
pub const CACHE_TRANSFERS_PENDING_SIZE_DEFAULT: usize =
    CONFIG.process.cache_transfers_pending_size_default;

/// The size of the client replies zone.
pub const CLIENT_REPLIES_SIZE: usize = CLIENTS_MAX as usize * MESSAGE_SIZE_MAX as usize;

const _: () = {
    assert!(CLIENT_REPLIES_SIZE > 0);
    assert!(CLIENT_REPLIES_SIZE.is_multiple_of(SECTOR_SIZE));
};

/// The maximum number of batch entries in the journal file:
/// A batch entry may contain many transfers, so this is not a limit on the number of transfers.
/// We need this limit to allocate space for copies of batch headers at the start of the journal.
/// These header copies enable us to disentangle corruption from crashes and recover accordingly.
pub const JOURNAL_SLOT_COUNT: u32 = CONFIG.cluster.journal_slot_count;

/// The maximum size of the WAL zone:
/// This is pre-allocated and zeroed for performance when initialized.
/// Writes within this file never extend the filesystem inode size reducing the cost of fdatasync().
/// This enables static allocation of disk space so that appends cannot fail with ENOSPC.
/// This also enables us to detect filesystem inode corruption that would change the journal size.
pub const JOURNAL_SIZE: usize = JOURNAL_SIZE_HEADERS + JOURNAL_SIZE_PREPARES;
pub const JOURNAL_SIZE_HEADERS: usize = JOURNAL_SLOT_COUNT as usize * HEADER_SIZE;
pub const JOURNAL_SIZE_PREPARES: usize = JOURNAL_SLOT_COUNT as usize * MESSAGE_SIZE_MAX as usize;

const _: () = {
    // For the given WAL (lsm_compaction_ops=4):
    //
    //   A    B    C    D    E
    //   |····|····|····|····|
    //
    // - ("|" delineates bars, where a bar is a multiple of prepare batches.)
    // - ("·" is a prepare in the WAL.)
    // - The Replica triggers a checkpoint at "E".
    // - The entries between "A" and "D" are on-disk in level 0.
    // - The entries between "D" and "E" are in-memory in the immutable table.
    // - So the checkpoint only includes "A…D".
    //
    // The journal must have at least two bars to ensure at least one is checkpointed.
    assert!(JOURNAL_SLOT_COUNT as usize >= config::ConfigCluster::JOURNAL_SLOT_COUNT_MIN);
    assert!(JOURNAL_SLOT_COUNT as usize >= LSM_COMPACTION_OPS * 2);
    assert!((JOURNAL_SLOT_COUNT as usize).is_multiple_of(LSM_COMPACTION_OPS));
    // The journal must have at least two pipelines of messages to ensure that a new,
    // fully-repaired primary has enough headers for a complete View message, even if the
    // view-change just truncated another pipeline of messages. (See op_repair_min()).
    assert!(JOURNAL_SLOT_COUNT as usize >= PIPELINE_PREPARE_QUEUE_MAX as usize * 2);

    assert!(JOURNAL_SIZE == JOURNAL_SIZE_HEADERS + JOURNAL_SIZE_PREPARES);
};

/// The maximum size of a message in bytes:
/// This is also the limit of all inflight data across multiple pipelined requests per connection.
/// We may have one request of up to 2 MiB inflight or 2 pipelined requests of up to 1 MiB inflight.
/// This impacts sequential disk write throughput, the larger the buffer the better.
/// 2 MiB is 16,384 transfers, and a reasonable choice for sequential disk write throughput.
/// However, this impacts bufferbloat and head-of-line blocking latency for pipelined requests.
/// For a 1 Gbps NIC = 125 MiB/s throughput: 2 MiB / 125 * 1000ms = 16ms for the next request.
/// This impacts the amount of memory allocated at initialization by the server.
pub const MESSAGE_SIZE_MAX: u32 = CONFIG.cluster.message_size_max;
pub const MESSAGE_BODY_SIZE_MAX: usize = MESSAGE_SIZE_MAX as usize - HEADER_SIZE;

const _: () = {
    // The WAL format requires messages to be a multiple of the sector size.
    assert!((MESSAGE_SIZE_MAX as usize).is_multiple_of(SECTOR_SIZE));
    assert!(MESSAGE_SIZE_MAX as usize >= HEADER_SIZE);
    assert!(MESSAGE_SIZE_MAX as usize >= SECTOR_SIZE);
    assert!(MESSAGE_SIZE_MAX >= config::ConfigCluster::message_size_max_min(CLIENTS_MAX));

    // Ensure that JV/View messages can fit all necessary headers.
    assert!(MESSAGE_BODY_SIZE_MAX >= VIEW_HEADERS_MAX as usize * HEADER_SIZE);

    // TODO(port): src/constants.zig — body size checks for vsr.ReconfigurationRequest,
    // vsr.BlockRequest, vsr.CheckpointState once those types are ported.
};

/// The maximum number of Viewstamped Replication prepare messages that can be inflight at a time.
/// This is immutable once assigned per cluster, as replicas need to know how many operations might
/// possibly be uncommitted during a view change, and this must be constant for all replicas.
pub const PIPELINE_PREPARE_QUEUE_MAX: u32 = CONFIG.cluster.pipeline_prepare_queue_max;

/// The maximum number of Viewstamped Replication request messages that can be queued at a primary,
/// waiting to prepare. Each client has at most one request in flight, and a primary can send a
/// pulse or request upgrade.
pub const PIPELINE_REQUEST_QUEUE_MAX: u32 =
    (CLIENTS_MAX + 1).saturating_sub(PIPELINE_PREPARE_QUEUE_MAX);

const _: () = {
    // A prepare-queue capacity larger than (clients_max + 1) is wasted.
    assert!(PIPELINE_PREPARE_QUEUE_MAX <= CLIENTS_MAX + 1);
    // A total queue capacity larger than (clients_max + 1) is wasted.
    assert!(PIPELINE_PREPARE_QUEUE_MAX + PIPELINE_REQUEST_QUEUE_MAX <= CLIENTS_MAX + 1);
    assert!(PIPELINE_PREPARE_QUEUE_MAX > 0);

    // A JV message uses the `header.context` (u128) field as a bitset to mark whether it has
    // prepared the corresponding header's message.
    assert!((PIPELINE_PREPARE_QUEUE_MAX as usize) < u128::BITS as usize);
};

/// Maximum number of headers from the WAL suffix to include in a View message.
/// Must at least cover the full pipeline.
/// Increasing this reduces likelihood that backups will need to repair their suffix's headers.
///
/// CRITICAL:
/// - We must provide enough headers to cover all uncommitted headers so that the new
///   primary (if we are in a view change) can decide whether to discard uncommitted headers
///   that cannot be repaired because they are gaps. See JVQuorum for more detail.
/// - +1 to leave room for commit_max, in case a backup converts the View to a JV.
pub const VIEW_CHANGE_HEADERS_SUFFIX_MAX: u32 = CONFIG.cluster.view_change_headers_suffix_max;

/// The number of prepare headers to include in the body of a JV/View.
///
/// View:
///
/// - We must include all uncommitted headers.
/// - +1 We must include the highest cluster-committed header (in case the View is converted to a JV
///   by the backup). (This is part of view_change_headers_suffix_max).
/// - +2: We must provide the header corresponding to each checkpoint-trigger in the intact
///   suffix of our journal.
///   - These help a lagging replica catch up when its `op < commit_max`.
///   - There are at most two of these in the journal.
///     (There are 2 immediately after we checkpoint, until we prepare enough to overwrite one).
///
/// JoinView:
///
/// - We must include all uncommitted headers.
/// - +1 We must include the highest cluster-committed header, so that the new primary still has a
///   head op if it truncates the entire pipeline.
pub const VIEW_HEADERS_MAX: u32 = VIEW_CHANGE_HEADERS_SUFFIX_MAX + 2;

const _: () = {
    assert!(VIEW_CHANGE_HEADERS_SUFFIX_MAX > PIPELINE_PREPARE_QUEUE_MAX);

    assert!(VIEW_HEADERS_MAX > 0);
    assert!(VIEW_HEADERS_MAX >= PIPELINE_PREPARE_QUEUE_MAX + 3);
    assert!(VIEW_HEADERS_MAX <= JOURNAL_SLOT_COUNT);
    assert!(
        VIEW_HEADERS_MAX as usize <= (MESSAGE_BODY_SIZE_MAX - CHECKPOINT_STATE_SIZE) / HEADER_SIZE
    );
    assert!(VIEW_HEADERS_MAX > VIEW_CHANGE_HEADERS_SUFFIX_MAX);
};

/// The maximum number of headers to include with a response to a command=get_headers message.
pub const GET_HEADERS_MAX: usize = umin(MESSAGE_BODY_SIZE_MAX / HEADER_SIZE, 64);

const _: () = assert!(GET_HEADERS_MAX > 0);

/// The maximum number of block addresses/checksums requested by a single command=get_blocks.
pub const GRID_REPAIR_REQUEST_MAX: u16 = CONFIG.process.grid_repair_request_max;

/// The number of grid reads allocated to handle incoming command=get_blocks messages.
pub const GRID_REPAIR_READS_MAX: u16 = CONFIG.process.grid_repair_reads_max;

/// Immediately after state sync we want access to all of the grid's write bandwidth to rapidly sync
/// table blocks.
pub const GRID_REPAIR_WRITES_MAX: u16 = GRID_IOPS_WRITE_MAX;

/// The default sizing of the grid cache. It's expected for operators to override this on the CLI.
pub const GRID_CACHE_SIZE_DEFAULT: usize = CONFIG.process.grid_cache_size_default;

/// The maximum capacity (in *single* blocks – not counting syncing tables) of the
/// GridBlocksMissing.
///
/// As this increases:
/// - GridBlocksMissing allocates more memory.
/// - The "period" of GridBlocksMissing's requests increases.
///   This makes the repair protocol more tolerant of network latency.
/// - (Repair protocol is used to repair manifest log blocks immediately after state sync).
pub const GRID_MISSING_BLOCKS_MAX: u32 = CONFIG.process.grid_missing_blocks_max;

/// The number of tables that can be synced simultaneously.
/// "Table" in this context is the number of table index blocks to hold in memory while syncing
/// their content.
///
/// As this increases:
/// - GridBlocksMissing allocates more memory (~2 blocks for each).
/// - Syncing is more efficient, as more blocks can be fetched concurrently.
pub const GRID_MISSING_TABLES_MAX: u32 = CONFIG.process.grid_missing_tables_max;

const _: () = {
    assert!(GRID_REPAIR_REQUEST_MAX > 0);
    assert!(GRID_REPAIR_REQUEST_MAX as usize <= MESSAGE_BODY_SIZE_MAX / BLOCK_REQUEST_SIZE);
    assert!(GRID_REPAIR_REQUEST_MAX <= GRID_REPAIR_READS_MAX);

    assert!(GRID_REPAIR_READS_MAX > 0);
    assert!(GRID_REPAIR_WRITES_MAX > 0);
    assert!(
        GRID_REPAIR_WRITES_MAX as usize
            <= GRID_MISSING_BLOCKS_MAX as usize
                + GRID_MISSING_TABLES_MAX as usize * LSM_TABLE_VALUE_BLOCKS_MAX
    );

    assert!(GRID_MISSING_BLOCKS_MAX > 0);
    assert!(GRID_MISSING_TABLES_MAX > 0);
};

/// The maximum number of concurrent scrubber reads.
///
/// Unless the scrubber cycle is extremely short and the data file very large there is no need to
/// set this higher than 1.
pub const GRID_SCRUBBER_READS_MAX: u16 = CONFIG.process.grid_scrubber_reads_max;

/// `grid_scrubber_cycle` is the (approximate, target) total duration per scrub of each
/// replica's entire grid. Scrubbing work is spread evenly across this duration.
///
/// Napkin math for the "worst case" scrubber read overhead as a function of cycle duration
/// (assuming a fully-loaded data file – maximum size and 100% acquired):
///
/// ```text
///   storage_size_limit          = 64TiB
///   grid_scrubber_cycle_seconds = 180 days * 24 hr/day * 60 min/hr * 60 s/min (2 cycles/year)
///   read_bytes_per_second       = storage_size_limit / grid_scrubber_cycle_seconds ≈ 4.32 MiB/s
/// ```
pub const GRID_SCRUBBER_CYCLE_TICKS: u64 = CONFIG.process.grid_scrubber_cycle.to_ms() / TICK_MS;

/// Accelerate/throttle scrubber reads if they are less/more frequent than this range.
/// (This is to keep the timeouts from being too extreme when the grid is tiny or huge.)
pub const GRID_SCRUBBER_INTERVAL_TICKS_MIN: u64 =
    CONFIG.process.grid_scrubber_interval_min.to_ms() / TICK_MS;
pub const GRID_SCRUBBER_INTERVAL_TICKS_MAX: u64 =
    CONFIG.process.grid_scrubber_interval_max.to_ms() / TICK_MS;

const MS_PER_MINUTE: u64 = 60 * 1000;

const _: () = {
    assert!(GRID_SCRUBBER_READS_MAX > 0);
    assert!(GRID_SCRUBBER_READS_MAX <= GRID_IOPS_READ_MAX);
    assert!(GRID_SCRUBBER_CYCLE_TICKS > 0);
    assert!(GRID_SCRUBBER_CYCLE_TICKS > MS_PER_MINUTE / TICK_MS); // Sanity-check.
    assert!(GRID_SCRUBBER_INTERVAL_TICKS_MIN > 0);
    assert!(GRID_SCRUBBER_INTERVAL_TICKS_MIN <= GRID_SCRUBBER_INTERVAL_TICKS_MAX);
    assert!(GRID_SCRUBBER_INTERVAL_TICKS_MAX > 0);
};

/// The minimum and maximum amount of time to wait before initiating a connection.
/// Exponential backoff and jitter are applied within this range.
pub const CONNECTION_DELAY_MIN: Duration = CONFIG.process.connection_delay_min;
pub const CONNECTION_DELAY_MAX: Duration = CONFIG.process.connection_delay_max;

/// The maximum number of outgoing messages that may be queued on a replica connection.
pub const CONNECTION_SEND_QUEUE_MAX_REPLICA: usize = umax(umin(CLIENTS_MAX as usize, 4), 2);

/// The maximum number of outgoing messages that may be queued on a client connection.
/// The client has one in-flight request, and occasionally a ping.
pub const CONNECTION_SEND_QUEUE_MAX_CLIENT: usize = 2;

/// The maximum number of outgoing requests that may be queued on a client (including the in-flight
/// request).
pub const CLIENT_REQUEST_QUEUE_MAX: u32 = CONFIG.process.client_request_queue_max;

/// The maximum number of connections in the kernel's complete connection queue pending an accept():
/// If the backlog argument is greater than the value in `/proc/sys/net/core/somaxconn`, then it is
/// silently truncated to that value. Since Linux 5.4, the default in this file is 4096.
pub const TCP_BACKLOG: u32 = CONFIG.process.tcp_backlog;

/// The maximum size of a kernel socket receive buffer in bytes (or 0 to use the system default):
/// This sets SO_RCVBUF as an alternative to the auto-tuning range in /proc/sys/net/ipv4/tcp_rmem.
/// The value is limited by /proc/sys/net/core/rmem_max, unless the CAP_NET_ADMIN privilege exists.
/// The kernel doubles this value to allow space for packet bookkeeping overhead.
/// The receive buffer should ideally exceed the Bandwidth-Delay Product for maximum throughput.
/// At the same time, be careful going beyond 4 MiB as the kernel may merge many small TCP packets,
/// causing considerable latency spikes for large buffer sizes:
/// <https://blog.cloudflare.com/the-story-of-one-latency-spike/>
pub const TCP_RCVBUF: i32 = CONFIG.process.tcp_rcvbuf;

/// The maximum size of a kernel socket send buffer in bytes (or 0 to use the system default):
/// This sets SO_SNDBUF as an alternative to the auto-tuning range in /proc/sys/net/ipv4/tcp_wmem.
/// The value is limited by /proc/sys/net/core/wmem_max, unless the CAP_NET_ADMIN privilege exists.
/// The kernel doubles this value to allow space for packet bookkeeping overhead.
pub const TCP_SNDBUF_REPLICA: i32 =
    umin(CONNECTION_SEND_QUEUE_MAX_REPLICA * MESSAGE_SIZE_MAX as usize, i32::MAX as usize) as i32;
pub const TCP_SNDBUF_CLIENT: i32 =
    umin(CONNECTION_SEND_QUEUE_MAX_CLIENT * MESSAGE_SIZE_MAX as usize, i32::MAX as usize) as i32;

const _: () = {
    // Avoid latency issues from setting sndbuf too high:
    assert!(TCP_SNDBUF_REPLICA as i64 <= 16 * MIB64);
    assert!(TCP_SNDBUF_CLIENT as i64 <= 16 * MIB64);
};

/// Whether to enable TCP keepalive:
pub const TCP_KEEPALIVE: bool = CONFIG.process.tcp_keepalive;

/// The time (in seconds) the connection needs to be idle before sending TCP keepalive probes:
/// Probes are not sent when the send buffer has data or the congestion window size is zero,
/// for these cases we also need tcp_user_timeout_ms below.
pub const TCP_KEEPIDLE: i32 = CONFIG.process.tcp_keepidle;

/// The time (in seconds) between individual keepalive probes:
pub const TCP_KEEPINTVL: i32 = CONFIG.process.tcp_keepintvl;

/// The maximum number of keepalive probes to send before dropping the connection:
pub const TCP_KEEPCNT: i32 = CONFIG.process.tcp_keepcnt;

/// The time (in milliseconds) to timeout an idle connection or unacknowledged send:
/// This timer rides on the granularity of the keepalive or retransmission timers.
/// For example, if keepalive will only send a probe after 10s then this becomes the lower bound
/// for tcp_user_timeout_ms to fire, even if tcp_user_timeout_ms is 2s. Nevertheless, this would
/// timeout the connection at 10s rather than wait for tcp_keepcnt probes to be sent. At the same
/// time, if tcp_user_timeout_ms is larger than the max keepalive time then tcp_keepcnt will be
/// ignored and more keepalive probes will be sent until tcp_user_timeout_ms fires.
/// For a thorough overview of how these settings interact:
/// <https://blog.cloudflare.com/when-tcp-sockets-refuse-to-die/>
pub const TCP_USER_TIMEOUT_MS: i32 = (TCP_KEEPIDLE + TCP_KEEPINTVL * TCP_KEEPCNT) * 1000;

/// Whether to disable Nagle's algorithm to eliminate send buffering delays:
pub const TCP_NODELAY: bool = CONFIG.process.tcp_nodelay;

/// Size of a CPU cache line in bytes
pub const CACHE_LINE_SIZE: usize = CONFIG.cluster.cache_line_size;

/// The minimum size of an aligned kernel page and an Advanced Format disk sector:
/// This is necessary for direct I/O without the kernel having to fix unaligned pages with a copy.
/// The new Advanced Format sector size is backwards compatible with the old 512 byte sector size.
/// This should therefore never be less than 4 KiB to be future-proof when server disks are swapped.
pub const SECTOR_SIZE: usize = 4096;

/// Whether to perform direct I/O to the underlying disk device:
/// This enables several performance optimizations:
/// * A memory copy to the kernel's page cache can be eliminated for reduced CPU utilization.
/// * I/O can be issued immediately to the disk device without buffering delay for improved latency.
///
/// This also enables several safety features:
/// * Disk data can be scrubbed to repair latent sector errors and checksum errors proactively.
/// * Fsync failures can be recovered from correctly.
///
/// WARNING: Disabling direct I/O is unsafe; the page cache cannot be trusted after an fsync error,
/// even after an application panic, since the kernel will mark dirty pages as clean, even
/// when they were never written to disk.
pub const DIRECT_IO: bool = CONFIG.process.direct_io;

pub const IOPS_READ_MAX: usize = JOURNAL_IOPS_READ_MAX as usize
    + CLIENT_REPLIES_IOPS_READ_MAX as usize
    + GRID_IOPS_READ_MAX as usize
    + SUPERBLOCK_IOPS_READ_MAX;
pub const IOPS_WRITE_MAX: usize = JOURNAL_IOPS_WRITE_MAX as usize
    + CLIENT_REPLIES_IOPS_WRITE_MAX as usize
    + GRID_IOPS_WRITE_MAX as usize
    + SUPERBLOCK_IOPS_WRITE_MAX;

/// Superblock has at most one write in flight.
const SUPERBLOCK_IOPS_READ_MAX: usize = 1;
const SUPERBLOCK_IOPS_WRITE_MAX: usize = 1;

/// The maximum number of concurrent WAL read I/O operations to allow at once.
pub const JOURNAL_IOPS_READ_MAX: u16 = CONFIG.process.journal_iops_read_max;
/// The maximum number of concurrent WAL write I/O operations to allow at once.
/// Ideally this is at least as high as pipeline_prepare_queue_max, but it is safe to be lower.
pub const JOURNAL_IOPS_WRITE_MAX: u16 = CONFIG.process.journal_iops_write_max;

/// The maximum number of concurrent reads to the client-replies zone.
/// Client replies are read when the client misses their original reply and retries a request.
pub const CLIENT_REPLIES_IOPS_READ_MAX: u16 = CONFIG.process.client_replies_iops_read_max;
/// The maximum number of concurrent writes to the client-replies zone.
/// Client replies are written after every commit.
pub const CLIENT_REPLIES_IOPS_WRITE_MAX: u16 = CONFIG.process.client_replies_iops_write_max;

/// The maximum number of concurrent grid read I/O operations to allow at once.
pub const GRID_IOPS_READ_MAX: u16 = CONFIG.process.grid_iops_read_max;
/// The maximum number of concurrent grid write I/O operations to allow at once.
pub const GRID_IOPS_WRITE_MAX: u16 = CONFIG.process.grid_iops_write_max;

const _: () = {
    assert!(JOURNAL_IOPS_READ_MAX > 0);
    assert!(JOURNAL_IOPS_WRITE_MAX > 0);
    assert!(CLIENT_REPLIES_IOPS_READ_MAX > 0);
    assert!(CLIENT_REPLIES_IOPS_WRITE_MAX > 0);
    assert!(CLIENT_REPLIES_IOPS_WRITE_MAX as u32 <= CLIENTS_MAX);
    assert!(GRID_IOPS_READ_MAX > 0);
    assert!(GRID_IOPS_WRITE_MAX > 0);
};

/// The number of redundant copies of the superblock in the superblock storage zone.
/// This must be either { 4, 6, 8 }, i.e. an even number, for more efficient flexible quorums.
///
/// The superblock contains local state for the replica and therefore cannot be replicated remotely.
/// Loss of the superblock would represent loss of the replica and so it must be protected.
///
/// This can mean checkpointing latencies in the rare extreme worst-case of at most 264ms, although
/// this would require EWAH compression of our block free set to have zero effective compression.
/// In practice, checkpointing latency should be an order of magnitude better due to compression,
/// because our block free set will fill holes when allocating.
///
/// The superblock only needs to be checkpointed every now and then, before the WAL wraps around,
/// or when a view change needs to take place to elect a new primary.
pub const SUPERBLOCK_COPIES: usize = CONFIG.cluster.superblock_copies;

const _: () = {
    assert!(SUPERBLOCK_COPIES.is_multiple_of(2));
    assert!(SUPERBLOCK_COPIES >= 4);
    assert!(SUPERBLOCK_COPIES <= 8);
};

/// The default maximum size of a local data file. This can be override, up to
/// storage_size_limit_max, by a CLI flag.
pub const STORAGE_SIZE_LIMIT_DEFAULT: usize = CONFIG.process.storage_size_limit_default;

/// The maximum size of a local data file.
/// This should not be much larger than several TiB to limit:
/// * blast radius and recovery time when a whole replica is lost,
/// * replicated storage overhead, since all data files are mirrored, and
/// * the static memory allocation required for tracking LSM forest metadata in memory.
///
/// This is a "firm" limit --- while it is a compile-time constant, it does not affect data file
/// layout and can be safely changed for an existing cluster.
pub const STORAGE_SIZE_LIMIT_MAX: usize = CONFIG.process.storage_size_limit_max;

const _: () = assert!(STORAGE_SIZE_LIMIT_MAX >= STORAGE_SIZE_LIMIT_DEFAULT);

/// The unit of read/write access to LSM manifest and LSM table blocks in the block storage zone.
///
/// - A lower block size increases the memory overhead of table metadata, due to smaller/more
///   tables.
/// - A higher block size increases space amplification due to partially-filled blocks.
pub const BLOCK_SIZE: usize = CONFIG.cluster.block_size;

const _: () = {
    assert!(BLOCK_SIZE.is_multiple_of(SECTOR_SIZE));
    assert!(BLOCK_SIZE > HEADER_SIZE);
    // Blocks are sent over the network as messages during grid repair and state sync.
    assert!(BLOCK_SIZE <= MESSAGE_SIZE_MAX as usize);
};

/// The number of levels in an LSM tree.
/// A higher number of levels increases read amplification, as well as total storage capacity.
pub const LSM_LEVELS: u8 = CONFIG.cluster.lsm_levels;

const _: () = {
    // ManifestLog serializes the level as a u6.
    assert!(LSM_LEVELS > 0);
    assert!(LSM_LEVELS <= 63); // maxInt(u6)
};

/// The number of tables at level i (0 ≤ i < lsm_levels) is `pow(lsm_growth_factor, i+1)`.
/// A higher growth factor increases write amplification (by increasing the number of tables in
/// level B that overlap a table in level A in a compaction), but decreases read amplification (by
/// reducing the height of the tree and thus the number of levels that must be probed). Since read
/// amplification can be optimized more easily (with caching), we target a growth
/// factor of 8 for lower write amplification rather than the more typical growth factor of 10.
pub const LSM_GROWTH_FACTOR: u32 = CONFIG.cluster.lsm_growth_factor;

const _: () = assert!(LSM_GROWTH_FACTOR > 1);

/// Size of nodes used by the LSM tree manifest implementation.
/// TODO Double-check this with our "LSM Manifest" spreadsheet.
pub const LSM_MANIFEST_NODE_SIZE: usize = CONFIG.process.lsm_manifest_node_size;

/// The number of manifest blocks to compact *beyond the minimum*, per half-bar.
///
/// In the worst case, we still compact entries faster than we produce them (by a margin of
/// "extra" blocks). This is necessary to ensure that the manifest has a bounded number of entries.
/// (Or in other words, that Pace's recurrence relation converges.)
///
/// This specific choice of value is somewhat arbitrary, but yields a decent balance between
/// "compaction work performed" and "total manifest size".
///
/// As this value increases, the manifest must perform more compaction work, but the manifest
/// upper-bound shrinks (and therefore manifest recovery time decreases).
///
/// See ManifestLog.Pace for more detail.
pub const LSM_MANIFEST_COMPACT_EXTRA_BLOCKS: usize =
    CONFIG.cluster.lsm_manifest_compact_extra_blocks;

const _: () = assert!(LSM_MANIFEST_COMPACT_EXTRA_BLOCKS > 0);

/// Number of prepares accumulated in the in-memory table before flushing to disk.
///
/// This is a batch of batches. Each prepare can contain at most 8_190 transfers. With
/// lsm_compaction_ops=32, 32 prepares are processed to fill the in-memory table with 262_080
/// transfers. During processing of the next 32 prepares, this in-memory table is flushed to disk.
/// Simultaneously, compaction is run to free up enough space to flush the in-memory table from the
/// next batch of lsm_compaction_ops prepares.
///
/// Together with message_body_size_max, lsm_compaction_ops determines the size a table on disk.
pub const LSM_COMPACTION_OPS: usize = CONFIG.cluster.lsm_compaction_ops;

const _: () = {
    // The LSM tree uses half-measures to balance compaction.
    assert!(LSM_COMPACTION_OPS.is_multiple_of(2));
};

// Limits for the number of value blocks that a single compaction can queue up for IO and for the
// number of IO operations themselves. The number of index blocks is always one per level.
// This is a comptime upper bound. The actual number of concurrency is also limited by the
// runtime-known number of free blocks.
//
// For simplicity for now, size IOPS to always be available.
pub const LSM_COMPACTION_QUEUE_READ_MAX: usize = 16;
pub const LSM_COMPACTION_QUEUE_WRITE_MAX: usize = 16;
pub const LSM_COMPACTION_IOPS_READ_MAX: usize = LSM_COMPACTION_QUEUE_READ_MAX + 2; // + two index blocks.
pub const LSM_COMPACTION_IOPS_WRITE_MAX: usize = LSM_COMPACTION_QUEUE_WRITE_MAX + 1; // + one index block.

pub const LSM_SNAPSHOTS_MAX: u32 = CONFIG.cluster.lsm_snapshots_max;

/// The maximum number of blocks that can possibly be referenced by any table index block.
///
/// - This is a very conservative (upper-bound) calculation that doesn't rely on the StateMachine's
///   tree configuration. (To prevent Grid from depending on StateMachine).
/// - This counts value blocks, but does not count the index block itself.
pub const LSM_TABLE_VALUE_BLOCKS_MAX: usize = (BLOCK_SIZE - HEADER_SIZE) / (U256_SIZE + U64_SIZE);

const U256_SIZE: usize = 32;
const U64_SIZE: usize = 8;

/// The default size in bytes of the NodePool used for the LSM forest's manifests.
pub const LSM_MANIFEST_MEMORY_SIZE_DEFAULT: usize = {
    // TODO Tune this better.
    let lsm_forest_node_count: usize = 8192;
    lsm_forest_node_count * LSM_MANIFEST_NODE_SIZE
};

/// The maximum size in bytes of the NodePool used for the LSM forest's manifests.
pub const LSM_MANIFEST_MEMORY_SIZE_MAX: usize =
    (u32::MAX as usize / LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER) * LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER;

/// The minimum size in bytes of the NodePool used for the LSM forest's manifests.
pub const LSM_MANIFEST_MEMORY_SIZE_MIN: usize = LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER;

/// The lsm memory size must be a multiple of this value.
///
/// While technically this could be equal to lsm_manifest_node_size, we set it
/// to 1MiB so it is a more obvious increment for users.
pub const LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER: usize = {
    let multiplier = 64 * LSM_MANIFEST_NODE_SIZE;
    assert!(multiplier == crate::stdx::MIB);
    multiplier
};

/// The LSM will attempt to coalesce a table if it is less full than this threshold.
pub const LSM_TABLE_COALESCING_THRESHOLD_PERCENT: usize =
    CONFIG.cluster.lsm_table_coalescing_threshold_percent;

const _: () = {
    assert!(LSM_TABLE_COALESCING_THRESHOLD_PERCENT > 0); // Ensure that coalescing is possible.
    assert!(LSM_TABLE_COALESCING_THRESHOLD_PERCENT < 100); // Don't coalesce full tables.
};

/// The number of milliseconds between each replica tick, the basic unit of time in TigerBeetle.
/// Used to regulate heartbeats, retries and timeouts, all specified as multiples of a tick.
pub const TICK_MS: u64 = CONFIG.process.tick_ms;

/// The conservative round-trip time at startup when there is no network knowledge.
/// Adjusted dynamically thereafter for RTT-sensitive timeouts according to network congestion.
/// This should be set higher rather than lower to avoid flooding the network at startup.
pub const RTT_TICKS: u64 = CONFIG.process.rtt.to_ms() / TICK_MS;

/// Maximum RTT, to prevent too-long timeouts.
pub const RTT_MAX_TICKS: u64 = CONFIG.process.rtt_max.to_ms() / TICK_MS;

/// The multiple of round-trip time for RTT-sensitive timeouts.
pub const RTT_MULTIPLE: usize = 2;

/// The min/max bounds of exponential backoff (and jitter) to add to RTT-sensitive timeouts.
pub const BACKOFF_MIN_TICKS: u64 = CONFIG.process.backoff_min.to_ms() / TICK_MS;
pub const BACKOFF_MAX_TICKS: u64 = CONFIG.process.backoff_max.to_ms() / TICK_MS;

/// The maximum amount of time we allow a peer to remain in unknown status,
/// after which it is terminated in favour of unconnected replicas (see
/// `reclaim_connection` in message_bus.zig). Peers must transition from unknown
/// to client or replica status in a reasonable amount of time using the
/// messages exchanged (see `peer_type` in message_header.zig). Pessimistically
/// set TTL to 45 seconds, since `ping_client` is sent by clients every 30 seconds.
pub const MESSAGE_BUS_UNKNOWN_TIME_TO_LIVE: Duration = Duration::seconds(45);

/// The maximum skew between two clocks to allow when considering them to be in agreement.
/// The principle is that no two clocks tick exactly alike but some clocks more or less agree.
/// The maximum skew across the cluster as a whole is this value times the total number of clocks.
/// The cluster will be unavailable if the majority of clocks are all further than this value apart.
/// Decreasing this reduces the probability of reaching agreement on synchronized time.
/// Increasing this reduces the accuracy of synchronized time.
pub const CLOCK_OFFSET_TOLERANCE_MAX: Duration = CONFIG.process.clock_offset_tolerance_max;

/// The amount of time before the clock's synchronized epoch is expired.
/// If the epoch is expired before it can be replaced with a new synchronized epoch, then this most
/// likely indicates either a network partition or else too many clock faults across the cluster.
/// A new synchronized epoch will be installed as soon as these conditions resolve.
pub const CLOCK_EPOCH_MAX: Duration = CONFIG.process.clock_epoch_max;

/// The amount of time to wait for enough accurate samples before synchronizing the clock.
/// The more samples we can take per remote clock source, the more accurate our estimation becomes.
/// This impacts cluster startup time as the primary must first wait for synchronization to
/// complete.
pub const CLOCK_SYNCHRONIZATION_WINDOW_MIN: Duration =
    CONFIG.process.clock_synchronization_window_min;

/// The amount of time without agreement before the clock window is expired and a new window opened.
/// This happens where some samples have been collected but not enough to reach agreement.
/// The quality of samples degrades as they age so at some point we throw them away and start over.
/// This eliminates the impact of gradual clock drift on our clock offset (clock skew) measurements.
/// If a window expires because of this then it is likely that the clock epoch will also be expired.
pub const CLOCK_SYNCHRONIZATION_WINDOW_MAX: Duration =
    CONFIG.process.clock_synchronization_window_max;

/// TigerBeetle uses asserts proactively, unless they severely degrade performance. For production,
/// 5% slow down might be deemed critical, tests tolerate slowdowns up to 5x. Tests should be
/// reasonably fast to make deterministic simulation effective. `constants.verify` disambiguates the
/// two cases.
///
/// In the control plane (eg, vsr proper) assert unconditionally. Due to batching, control plane
/// overhead is negligible. It is acceptable to spend O(N) time to verify O(1) computation.
///
/// In the data plane (eg, lsm tree), finer grained judgement is required. Do an unconditional O(1)
/// assert before an O(N) loop (e.g, a bounds check). Inside the loop, it might or might not be
/// feasible to add an extra assert per iteration. In the latter case, guard the assert with `if
/// VERIFY`, but prefer an unconditional assert unless benchmarks prove it to be costly.
///
/// In the data plane, never use O(N) asserts for O(1) computations --- due to randomized testing
/// the overall coverage is proportional to the number of tests run. Slow thorough assertions
/// decrease the overall test coverage.
///
/// Specific data structures might use a comptime parameter, to enable extra costly verification
/// only during unit tests of the data structure.
pub const VERIFY: bool = CONFIG.process.verify;

/// The maximum number of bytes to use for compaction blocks.
pub const COMPACTION_BLOCK_MEMORY_SIZE_MAX: u64 = u32::MAX as u64 * BLOCK_SIZE as u64;

/// Maximum number of tree scans that can be performed by a single query.
/// NOTE: Each condition in a query is a scan, for example `WHERE a=0 AND b=1` needs 2 scans.
pub const LSM_SCANS_MAX: usize = CONFIG.cluster.lsm_scans_max;

/// Processing more than this amount of messages in a single event loop turn issues a warning.
pub const BUS_MESSAGE_BURST_WARN_MIN: usize = 8;

const MIB64: i64 = 1 << 20;

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream switches to the `test_min` config under `zig build test`; `cfg!(test)` mirrors
    /// that. Verify the switch actually happens and derived constants follow suit.
    #[test]
    fn test_min_config_selected_under_test() {
        assert!(!CONFIG.is_production());
        assert_eq!(CONFIG.cluster.clients_max, 7);
        assert_eq!(JOURNAL_SLOT_COUNT, 32);
        assert_eq!(MESSAGE_SIZE_MAX, 4096);
        assert_eq!(BLOCK_SIZE, SECTOR_SIZE);
        assert_eq!(LSM_COMPACTION_OPS, 4);
    }

    #[test]
    fn derived_values_match_test_min_config() {
        // journal_slot_count_min = 2 * (sector_size / header_size):
        assert_eq!(config::ConfigCluster::JOURNAL_SLOT_COUNT_MIN, 2 * (SECTOR_SIZE / HEADER_SIZE));
        assert_eq!(MESSAGE_BODY_SIZE_MAX, MESSAGE_SIZE_MAX as usize - HEADER_SIZE);
        // vsr_checkpoint_ops = 32 - 4 - 4 * div_ceil(8, 4) = 20:
        assert_eq!(VSR_CHECKPOINT_OPS, 20);
        // (clients + 1) -| pipeline_prepare_queue_max = 8 -| 4:
        assert_eq!(PIPELINE_REQUEST_QUEUE_MAX, 4);
        assert_eq!(VIEW_CHANGE_HEADERS_SUFFIX_MAX, 5);
        assert_eq!(VIEW_HEADERS_MAX, 7);
        assert_eq!(GET_HEADERS_MAX, (4096 - 256) / 256); // 15
        assert_eq!(CLIENT_REPLIES_SIZE, 7 * 4096);
        assert_eq!(LSM_TABLE_VALUE_BLOCKS_MAX, (4096 - 256) / 40); // 96
        assert_eq!(RTT_TICKS, 30);
        assert_eq!(RTT_MAX_TICKS, 6000);
        assert_eq!(BACKOFF_MIN_TICKS, 1);
        assert_eq!(BACKOFF_MAX_TICKS, 1000);
        assert_eq!(GRID_SCRUBBER_INTERVAL_TICKS_MIN, 5);
        assert_eq!(GRID_SCRUBBER_INTERVAL_TICKS_MAX, 1000);
        // One hour cycle:
        assert_eq!(GRID_SCRUBBER_CYCLE_TICKS, (60 * 60 * 1000) / 10);
        assert_eq!(TCP_USER_TIMEOUT_MS, (5 + 4 * 3) * 1000);
        assert_eq!(LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER, crate::stdx::MIB);
    }
}
