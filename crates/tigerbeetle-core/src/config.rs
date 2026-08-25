//! Raw configuration values.
//!
//! Port of `src/config.zig`. Code which needs these values should use `constants` instead.
//! Configuration values are set from a combination of:
//! - default values (struct field defaults, mirroring upstream),
//! - the selected base config (`default_production` or `test_min`),
//! - build-time options.
//!
//! DEVIATION: Upstream injects build metadata (`release`, `release_client_min`, `git_commit`,
//! `config_verify`) via Zig build options and `log_level` via std.log. Those are deferred until
//! the multiversion/logging subsystems are ported; `verify` is always `true` for now.
//! TODO(port): src/config.zig BuildOptions / vsr.Release.

// Doc comments are ported verbatim from upstream; numeric casts mirror upstream's exact widths.
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

use crate::constants::{HEADER_SIZE, SECTOR_SIZE};
use crate::stdx::{Duration, GIB, KIB, MIB, TIB, align_forward};

/// TODO(port): src/tigerbeetle.zig Account — replace with `size_of::<crate::vsr::Account>()`.
pub(crate) const ACCOUNT_SIZE: usize = 128;

const MS_PER_HOUR: u64 = 60 * 60 * 1000;
const MS_PER_DAY: u64 = 24 * MS_PER_HOUR;

/// Configurations which are tunable per-replica (or per-client).
/// - Replica configs need not equal each other.
/// - Client configs need not equal each other.
/// - Client configs need not equal replica configs.
/// - Replica configs can change between restarts.
///
/// Fields are documented within constants.rs (upstream keeps docs there; we mirror that).
#[allow(clippy::struct_excessive_bools)] // mirrors upstream field set
#[derive(Clone, Copy, Debug)]
pub struct ConfigProcess {
    pub verify: bool,
    pub port: u16,
    pub address: &'static str,
    pub storage_size_limit_default: usize,
    pub storage_size_limit_max: usize,
    pub memory_size_max_default: usize,
    pub cache_accounts_size_default: usize,
    pub cache_transfers_size_default: usize,
    pub cache_transfers_pending_size_default: usize,
    pub client_request_queue_max: u32,
    pub lsm_manifest_node_size: usize,
    pub connection_delay_min: Duration,
    pub connection_delay_max: Duration,
    /// DEVIATION: upstream uses u31; Rust has no u31.
    pub tcp_backlog: u32,
    pub tcp_rcvbuf: i32,
    pub tcp_keepalive: bool,
    pub tcp_keepidle: i32,
    pub tcp_keepintvl: i32,
    pub tcp_keepcnt: i32,
    pub tcp_nodelay: bool,
    pub direct_io: bool,
    pub journal_iops_read_max: u16,
    pub journal_iops_write_max: u16,
    pub client_replies_iops_read_max: u16,
    pub client_replies_iops_write_max: u16,
    /// DEVIATION: upstream uses u63; Rust has no u63.
    pub tick_ms: u64,
    pub rtt: Duration,
    pub rtt_max: Duration,
    pub backoff_min: Duration,
    pub backoff_max: Duration,
    pub clock_offset_tolerance_max: Duration,
    pub clock_epoch_max: Duration,
    pub clock_synchronization_window_min: Duration,
    pub clock_synchronization_window_max: Duration,
    pub grid_iops_read_max: u16,
    pub grid_iops_write_max: u16,
    pub grid_cache_size_default: usize,
    pub grid_repair_request_max: u16,
    pub grid_repair_reads_max: u16,
    pub grid_missing_blocks_max: u32,
    pub grid_missing_tables_max: u32,
    pub grid_scrubber_reads_max: u16,
    pub grid_scrubber_cycle: Duration,
    pub grid_scrubber_interval_min: Duration,
    pub grid_scrubber_interval_max: Duration,
    pub multiversion_binary_platform_size_max: usize,
    pub multiversion_poll_interval: Duration,
}

impl ConfigProcess {
    /// Field defaults, mirroring the defaults in the upstream struct definition.
    /// (A const item rather than `Default::default()` so it stays usable in `const` contexts.)
    pub const DEFAULT: Self = Self {
        verify: false,
        port: 3001,
        address: "127.0.0.1",
        storage_size_limit_default: 16 * TIB,
        storage_size_limit_max: 64 * TIB,
        memory_size_max_default: GIB,
        // Required upstream; both base configs set these explicitly.
        cache_accounts_size_default: 0,
        cache_transfers_size_default: 0,
        cache_transfers_pending_size_default: 0,
        client_request_queue_max: 2,
        lsm_manifest_node_size: 16 * KIB,
        connection_delay_min: Duration::ms(50),
        connection_delay_max: Duration::ms(1000),
        tcp_backlog: 64,
        tcp_rcvbuf: (4 * MIB) as i32,
        tcp_keepalive: true,
        tcp_keepidle: 5,
        tcp_keepintvl: 4,
        tcp_keepcnt: 3,
        tcp_nodelay: true,
        direct_io: false,
        journal_iops_read_max: 8,
        journal_iops_write_max: 32,
        client_replies_iops_read_max: 1,
        client_replies_iops_write_max: 2,
        tick_ms: 10,
        rtt: Duration::ms(300),
        rtt_max: Duration::seconds(60),
        backoff_min: Duration::ms(10),
        backoff_max: Duration::ms(10000),
        clock_offset_tolerance_max: Duration::ms(10000),
        clock_epoch_max: Duration::ms(60000),
        clock_synchronization_window_min: Duration::ms(2000),
        clock_synchronization_window_max: Duration::ms(20000),
        grid_iops_read_max: 32,
        grid_iops_write_max: 32,
        grid_cache_size_default: GIB,
        grid_repair_request_max: 4,
        grid_repair_reads_max: 4,
        grid_missing_blocks_max: 30,
        grid_missing_tables_max: 6,
        grid_scrubber_reads_max: 1,
        grid_scrubber_cycle: Duration::ms(MS_PER_DAY * 180),
        grid_scrubber_interval_min: Duration::ms(50),
        grid_scrubber_interval_max: Duration::seconds(10),
        multiversion_binary_platform_size_max: 64 * MIB,
        multiversion_poll_interval: Duration::ms(1000),
    };
}

impl Default for ConfigProcess {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Configurations which are tunable per-cluster.
/// - All replicas within a cluster must have the same configuration.
/// - Replicas must reuse the same configuration when the binary is upgraded — they do not change
///   over the cluster lifetime.
/// - The storage formats generated by different ConfigClusters are incompatible.
#[derive(Clone, Copy, Debug)]
pub struct ConfigCluster {
    pub cache_line_size: usize,
    pub clients_max: u32,
    pub pipeline_prepare_queue_max: u32,
    pub view_change_headers_suffix_max: u32,
    pub quorum_replication_max: u8,
    pub journal_slot_count: u32,
    pub message_size_max: u32,
    pub superblock_copies: usize,
    pub block_size: usize,
    pub lsm_levels: u8,
    pub lsm_growth_factor: u32,
    pub lsm_compaction_ops: usize,
    pub lsm_snapshots_max: u32,
    pub lsm_manifest_compact_extra_blocks: usize,
    pub lsm_table_coalescing_threshold_percent: usize,
    pub vsr_releases_max: u32,
    pub lsm_scans_max: usize,
}

impl ConfigCluster {
    /// Field defaults, mirroring the defaults in the upstream struct definition.
    pub const DEFAULT: Self = Self {
        cache_line_size: 64,
        clients_max: 0,
        pipeline_prepare_queue_max: 8,
        view_change_headers_suffix_max: 8 + 1,
        quorum_replication_max: 3,
        journal_slot_count: 1024,
        message_size_max: MIB as u32,
        superblock_copies: 4,
        block_size: 512 * KIB,
        lsm_levels: 7,
        lsm_growth_factor: 8,
        lsm_compaction_ops: 32,
        lsm_snapshots_max: 32,
        lsm_manifest_compact_extra_blocks: 1,
        lsm_table_coalescing_threshold_percent: 50,
        vsr_releases_max: 64,
        lsm_scans_max: 6,
    };

    /// The WAL requires at least two sectors of redundant headers — otherwise we could lose them
    /// all to a single torn write. A replica needs at least one valid redundant header to
    /// determine an (untrusted) maximum op in recover_torn_prepare(), without which it cannot
    /// truncate a torn prepare.
    pub const JOURNAL_SLOT_COUNT_MIN: usize = 2 * (SECTOR_SIZE / HEADER_SIZE);

    pub const CLIENTS_MAX_MIN: usize = 1;

    /// The smallest possible message_size_max (for use in the simulator to improve performance).
    /// The message body must have room for pipeline_prepare_queue_max headers in the JV.
    #[must_use]
    pub const fn message_size_max_min(clients_max: u32) -> u32 {
        let body_and_headers = HEADER_SIZE + clients_max as usize * HEADER_SIZE;
        let aligned = align_forward(body_and_headers, SECTOR_SIZE);
        let min = if aligned > SECTOR_SIZE { aligned } else { SECTOR_SIZE };
        min as u32
    }

    /// Fingerprint of the cluster-wide configuration.
    /// It is used to assert that all cluster members share the same config.
    ///
    /// Port of `ConfigCluster.checksum()`: every field is widened to `u64` and serialized
    /// little-endian in declaration order, then run through the vsr checksum.
    #[must_use]
    pub fn checksum(&self) -> u128 {
        let values: [u64; 17] = [
            self.cache_line_size as u64,
            u64::from(self.clients_max),
            u64::from(self.pipeline_prepare_queue_max),
            u64::from(self.view_change_headers_suffix_max),
            u64::from(self.quorum_replication_max),
            u64::from(self.journal_slot_count),
            u64::from(self.message_size_max),
            self.superblock_copies as u64,
            self.block_size as u64,
            u64::from(self.lsm_levels),
            u64::from(self.lsm_growth_factor),
            self.lsm_compaction_ops as u64,
            u64::from(self.lsm_snapshots_max),
            self.lsm_manifest_compact_extra_blocks as u64,
            self.lsm_table_coalescing_threshold_percent as u64,
            u64::from(self.vsr_releases_max),
            self.lsm_scans_max as u64,
        ];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        crate::checksum::checksum(&bytes)
    }
}

impl Default for ConfigCluster {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Upstream: `src/config.zig` `Config`.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub cluster: ConfigCluster,
    pub process: ConfigProcess,
}

impl Config {
    /// Returns true if the configuration is intended for "production".
    /// Intended solely for extra sanity-checks: all meaningful decisions should be driven by
    /// specific fields of the config.
    #[must_use]
    pub const fn is_production(&self) -> bool {
        self.cluster.journal_slot_count > ConfigCluster::JOURNAL_SLOT_COUNT_MIN as u32
    }
}

/// Upstream: `src/config.zig` `configs`.
pub mod configs {
    use super::{ACCOUNT_SIZE, Config, ConfigCluster, ConfigProcess, Duration, GIB, MIB};
    use super::{MS_PER_HOUR, SECTOR_SIZE};

    /// A good default config for production.
    #[must_use]
    pub const fn default_production() -> Config {
        let process = ConfigProcess {
            direct_io: true,
            cache_accounts_size_default: ACCOUNT_SIZE * MIB,
            cache_transfers_size_default: 0,
            cache_transfers_pending_size_default: 0,
            verify: true,
            ..ConfigProcess::DEFAULT
        };
        let cluster = ConfigCluster { clients_max: 64, ..ConfigCluster::DEFAULT };
        Config { cluster, process }
    }

    /// Minimal test configuration — small WAL, small grid block size, etc.
    /// Not suitable for production, but good for testing code that would be otherwise hard to
    /// reach.
    #[must_use]
    pub const fn test_min() -> Config {
        let process = ConfigProcess {
            storage_size_limit_default: GIB,
            storage_size_limit_max: GIB,
            direct_io: false,
            cache_accounts_size_default: ACCOUNT_SIZE * 256,
            cache_transfers_size_default: 0,
            cache_transfers_pending_size_default: 0,
            journal_iops_read_max: 3,
            journal_iops_write_max: 2,
            grid_iops_read_max: 8,
            grid_iops_write_max: 8,
            grid_repair_request_max: 4,
            grid_repair_reads_max: 4,
            grid_missing_blocks_max: 3,
            grid_missing_tables_max: 2,
            grid_scrubber_reads_max: 2,
            grid_scrubber_cycle: Duration::ms(MS_PER_HOUR),
            verify: true,
            ..ConfigProcess::DEFAULT
        };
        let cluster = ConfigCluster {
            clients_max: 4 + 3,
            pipeline_prepare_queue_max: 4,
            view_change_headers_suffix_max: 4 + 1,
            journal_slot_count: ConfigCluster::JOURNAL_SLOT_COUNT_MIN as u32,
            message_size_max: ConfigCluster::message_size_max_min(4),

            block_size: SECTOR_SIZE,
            lsm_compaction_ops: 4,
            lsm_growth_factor: 4,
            // (This is higher than the production default value because the block size is
            // smaller.)
            lsm_manifest_compact_extra_blocks: 5,
            // (We need to fuzz more scans merge than in production.)
            lsm_scans_max: 12,

            ..ConfigCluster::DEFAULT
        };
        Config { cluster, process }
    }

    /// Upstream picks `test_min` when `builtin.is_test`, which recompiles constants for each
    /// consumer build. Rust has no cross-crate `cfg(test)`: `cfg!(test)` is only set when this
    /// crate itself is unit-tested, so downstream crates also select `test_min` via the
    /// `test-min` feature (wired through dev-dependencies). Resolved at compile time either
    /// way, keeping `CONFIG` a constant.
    pub const CURRENT: Config =
        if cfg!(test) || cfg!(feature = "test-min") { test_min() } else { default_production() };
}

#[cfg(test)]
mod config_tests {
    use crate::config::configs::{default_production, test_min};

    /// Golden values generated by running upstream `ConfigCluster.checksum()` (Zig 0.14.1)
    /// via `reference/tigerbeetle/src/tbcross_main.zig`. Pins field order, widths, and
    /// endianness of the serialization against upstream.
    #[test]
    fn cluster_checksum_matches_upstream_zig() {
        assert_eq!(
            format!("{:032x}", test_min().cluster.checksum()),
            "3abdeae77c8d7c45d643ff94f4c0701e"
        );
        assert_eq!(
            format!("{:032x}", default_production().cluster.checksum()),
            "790755030145d92aa1dcfec99f79affd"
        );
    }
}
