//! Parse and validate command-line arguments for the `tigerbeetle` binary.
//!
//! Port of `src/tigerbeetle/cli.zig` (parse + validate layer) and the runnable `version` /
//! `format` commands of `src/tigerbeetle/main.zig`. Everything that can be validated without
//! reading the data file is validated here, mirroring upstream semantics one-for-one.
//!
//! DEVIATION: upstream `vsr.fatal(.cli, ...)` prints and exits; here parse failures are
//! surfaced as [`ParseFailure`] (or a plain `Err(String)` from the desugar helpers), and
//! `main.rs` prints and exits non-zero. The async event loop is deferred, so `start`,
//! `recover`, `repl`, `benchmark`, `inspect`, `multiversion` and `amqp` are parsed and
//! validated but have no executor yet.

#![allow(clippy::cast_lossless)] // port arithmetic mirrors upstream `@intCast`/`@divExact`

use std::mem::size_of;

use tigerbeetle_core::constants::{
    CACHE_ACCOUNTS_SIZE_DEFAULT, CACHE_TRANSFERS_PENDING_SIZE_DEFAULT,
    CACHE_TRANSFERS_SIZE_DEFAULT, CLIENTS_MAX, GRID_CACHE_SIZE_DEFAULT,
    LSM_COMPACTION_QUEUE_READ_MAX, LSM_MANIFEST_MEMORY_SIZE_DEFAULT, LSM_MANIFEST_MEMORY_SIZE_MAX,
    LSM_MANIFEST_MEMORY_SIZE_MIN, LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER, LSM_MANIFEST_NODE_SIZE,
    MESSAGE_BODY_SIZE_MAX, MESSAGE_SIZE_MAX, PIPELINE_PREPARE_QUEUE_MAX,
    PIPELINE_REQUEST_QUEUE_MAX, REPLICAS_MAX, SECTOR_SIZE, STANDBYS_MAX,
    STORAGE_SIZE_LIMIT_DEFAULT, STORAGE_SIZE_LIMIT_MAX, SUPERBLOCK_COPIES, TICK_MS,
};
use tigerbeetle_core::net::SocketAddress;
use tigerbeetle_core::stdx::{
    ByteSize, ByteSizeUnit, ParseByteSizeError, ParseIntError, ParseIntOptions, parse_int_u8,
    parse_int_u16, parse_int_u32, parse_int_u64, parse_int_u128,
};
use tigerbeetle_core::types::{Account, Transfer, TransferPending};
use tigerbeetle_lsm::set_associative_cache::SetAssociativeCache;
use tigerbeetle_vsr::groove::{
    AccountObjectsCacheSpec, TransferObjectsCacheSpec, TransferPendingObjectsCacheSpec,
};
use tigerbeetle_vsr::socket::{ClusterAddress, ParseAddressesError, parse_address_and_port};
use tigerbeetle_vsr::storage::Storage;

/// The build version printed by `version` (upstream `constants.semver`).
///
/// DEVIATION: upstream stamps a semantic version; this port uses the crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// POSIX `PATH_MAX`; upstream sizes `Command.Path` by `std.fs.max_path_bytes`.
///
/// DEVIATION: Rust `std` has no portable path-max constant, so the upstream bound is fixed
/// to 4096 here.
const PATH_MAX_BYTES: usize = 4096;

// -----------------------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------------------

/// Top-level parse result: a command, a request to print help (exit 0), or a fatal message.
#[derive(Debug, PartialEq, Eq)]
pub enum ParseFailure {
    /// `-h`/`--help`: print the given text and exit 0 (upstream `parse_commands`).
    Help(&'static str),
    /// Fatal CLI error: print the message and exit non-zero (upstream `vsr.fatal(.cli, ...)`).
    Fatal(String),
}

/// A run-time error while executing a command (`main.zig` fatal paths).
pub type CliResult<T> = Result<T, String>;

// -----------------------------------------------------------------------------------
// Command (desugared, validated arguments) — mirrors upstream `cli.zig` `Command`.
// -----------------------------------------------------------------------------------

/// Upstream `Command` union.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Format(CommandFormat),
    Recover(CommandRecover),
    Start(CommandStart),
    Version(CommandVersion),
    Repl(CommandRepl),
    Benchmark(CommandBenchmark),
    Inspect(CommandInspect),
    Multiversion(CommandMultiversion),
    Amqp(CommandAmqp),
}

/// Upstream `Command.Format`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommandFormat {
    pub cluster: u128,
    pub replica: u8,
    pub replica_count: u8,
    pub development: bool,
    pub path: String,
    pub log_debug: bool,
}

/// Upstream `Command.Recover`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommandRecover {
    pub cluster: u128,
    pub addresses: ClusterAddress,
    pub replica: u8,
    pub replica_count: u8,
    pub development: bool,
    pub path: String,
    pub log_debug: bool,
}

/// Upstream `Command.Start`.
// Mirrors upstream's flat bool fields (development/experimental/replicate_star/aof_recovery/...).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, PartialEq, Eq)]
pub struct CommandStart {
    pub addresses: ClusterAddress,
    pub cache_accounts: u32,
    pub cache_transfers: u32,
    pub cache_transfers_pending: u32,
    pub storage_size_limit: u64,
    pub pipeline_requests_limit: u32,
    pub request_size_limit: u32,
    pub cache_grid_blocks: u32,
    pub lsm_forest_compaction_block_count: u32,
    pub lsm_forest_node_count: u32,
    pub timeout_prepare_ticks: Option<u64>,
    pub timeout_grid_repair_message_ticks: Option<u64>,
    pub commit_stall_probability: Option<Ratio>,
    pub commit_stall_lag_min: Option<u32>,
    pub commit_stall_lag_max: Option<u32>,
    pub commit_stall_multiple_max: Option<u16>,
    pub trace: Option<String>,
    pub development: bool,
    pub experimental: bool,
    pub replicate_star: bool,
    /// `Some` only when `--aof`/`--aof-file` was given (upstream `Command.Path`).
    pub aof_file: Option<String>,
    pub aof_recovery: bool,
    pub path: String,
    pub log_debug: bool,
    pub log_trace: bool,
    pub statsd: Option<SocketAddress>,
}

/// Upstream `Command.Version`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommandVersion {
    pub verbose: bool,
}

/// Upstream `Command.Repl`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommandRepl {
    pub addresses: ClusterAddress,
    pub cluster: u128,
    pub verbose: bool,
    pub statements: String,
    pub log_debug: bool,
}

/// A probability ratio (upstream `stdx.PRNG.Ratio`, default 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio {
    /// Invariant: `numerator <= denominator`.
    pub numerator: u64,
    /// Invariant: `denominator != 0`.
    pub denominator: u64,
}

impl Ratio {
    const fn zero() -> Self {
        Self { numerator: 0, denominator: 1 }
    }
}

/// Upstream `Command.Benchmark`.
// Mirrors upstream's flat bool fields (log_debug/no_history/imported/...).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, PartialEq, Eq)]
pub struct CommandBenchmark {
    pub cache_accounts: Option<String>,
    pub cache_transfers: Option<String>,
    pub cache_transfers_pending: Option<String>,
    pub cache_grid: Option<String>,
    pub memory: Option<String>,
    pub log_debug: bool,
    pub log_debug_replica: bool,
    pub account_count: u64,
    pub account_count_hot: u32,
    pub account_distribution: BenchmarkDistribution,
    pub no_history: bool,
    pub imported: bool,
    pub account_batch_count: u32,
    pub transfer_count: u64,
    pub transfer_hot_percent: u32,
    pub transfer_pending: bool,
    pub transfer_batch_count: u32,
    /// Nanoseconds (upstream `Duration`).
    pub transfer_batch_delay: u64,
    pub validate: bool,
    pub checksum_performance: bool,
    pub query_count: u32,
    pub print_batch_timings: bool,
    pub id_order: BenchmarkIdOrder,
    pub clients: u32,
    pub statsd: Option<String>,
    pub trace: Option<String>,
    pub file: Option<String>,
    pub addresses: Option<ClusterAddress>,
    pub seed: Option<String>,
}

/// Upstream `Command.Benchmark.IdOrder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkIdOrder {
    Tbid,
    Sequential,
    Random,
    Reversed,
}

/// Upstream `Command.Benchmark.Distribution`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkDistribution {
    Zipfian,
    Latest,
    Uniform,
}

/// Upstream `Command.Inspect`.
#[derive(Debug, PartialEq, Eq)]
pub enum CommandInspect {
    Constants,
    Metrics,
    Op(u64),
    DataFile(CommandInspectDataFile),
    Integrity(CommandInspectIntegrity),
}

/// Upstream `Command.Inspect.DataFile`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommandInspectDataFile {
    pub path: String,
    pub query: InspectQuery,
}

/// Upstream `Command.Inspect.DataFile.query`.
#[derive(Debug, PartialEq, Eq)]
pub enum InspectQuery {
    Superblock,
    Wal { slot: Option<usize> },
    Replies { slot: Option<usize>, superblock_copy: Option<u8> },
    Grid { block: Option<u64>, superblock_copy: Option<u8> },
    Manifest { superblock_copy: Option<u8> },
    Tables { superblock_copy: Option<u8>, tree: String, level: Option<u8> },
}

/// Upstream `Command.Inspect.Integrity`.
// Mirrors upstream's flat bool fields (log_debug/skip_wal/skip_client_replies/skip_grid).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, PartialEq, Eq)]
pub struct CommandInspectIntegrity {
    pub path: String,
    pub log_debug: bool,
    pub seed: Option<String>,
    pub lsm_forest_node_count: u32,
    pub skip_wal: bool,
    pub skip_client_replies: bool,
    pub skip_grid: bool,
}

/// Upstream `Command.Multiversion`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommandMultiversion {
    pub path: String,
    pub log_debug: bool,
}

/// Upstream `Command.AMQP`.
#[derive(Debug, PartialEq, Eq)]
pub struct CommandAmqp {
    pub addresses: ClusterAddress,
    pub cluster: u128,
    pub host: SocketAddress,
    pub user: String,
    pub password: String,
    pub vhost: String,
    pub publish_exchange: Option<String>,
    pub publish_routing_key: Option<String>,
    pub event_count_max: Option<u32>,
    pub idle_interval_ms: Option<u32>,
    pub requests_per_second_limit: Option<u32>,
    pub amqp_timeout_seconds: Option<u32>,
    pub tigerbeetle_timeout_seconds: Option<u32>,
    pub timestamp_last: Option<u64>,
    pub log_debug: bool,
}

impl Command {
    /// The subcommand name as typed on the CLI (used in deferred-executor messages).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Format(_) => "format",
            Self::Recover(_) => "recover",
            Self::Start(_) => "start",
            Self::Version(_) => "version",
            Self::Repl(_) => "repl",
            Self::Benchmark(_) => "benchmark",
            Self::Inspect(_) => "inspect",
            Self::Multiversion(_) => "multiversion",
            Self::Amqp(_) => "amqp",
        }
    }
}

// -----------------------------------------------------------------------------------
// Flag parsing engine — port of `stdx/flags.zig` `parse_flags`/`parse_commands`.
// -----------------------------------------------------------------------------------

/// The value kinds a named flag can carry, driving typed parsing (upstream comptime types).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NamedKind {
    /// Bare `--flag` sets true; `--flag=true|false` needs the explicit value.
    Bool,
    U8,
    U16,
    U32,
    U64,
    U128,
    /// `u6` (inspect `--level`): values above 63 report a 6-bit overflow.
    U6,
    /// Boilerplate `usize` (inspect `--slot`).
    Usize,
    /// `stdx.ByteSize` (e.g. `--cache-grid=1GiB`).
    Size,
    /// Any string.
    String_,
    /// `vsr.ClusterAddress`.
    Cluster,
    /// `host[:port]` socket with a default port.
    Socket {
        default_port: u16,
    },
    /// `stdx.PRNG.Ratio`.
    Ratio,
    /// `stdx.Duration`, stored as nanoseconds.
    DurationNs,
    /// An exhaustive enum plus its tag list (benchmark `--id-order` / `--account-distribution`).
    Enum(&'static [&'static str]),
}

/// A declared named flag plus its value kind (upstream struct fields, longest-first).
#[derive(Clone, Copy)]
struct Named {
    flag: &'static str,
    kind: NamedKind,
}

/// A typed flag value (upstream `parse_value`).
#[derive(Clone, Debug)]
enum Value {
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Usize(usize),
    Size(ByteSize),
    String(String),
    Cluster(ClusterAddress),
    Socket(SocketAddress),
    Ratio(Ratio),
    DurationNs(u64),
    Enum(usize),
}

impl Value {
    fn into_bool(self) -> bool {
        match self {
            Self::Bool(b) => b,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_u8(self) -> u8 {
        match self {
            Self::U8(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_u16(self) -> u16 {
        match self {
            Self::U16(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_u32(self) -> u32 {
        match self {
            Self::U32(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_u64(self) -> u64 {
        match self {
            Self::U64(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_u128(self) -> u128 {
        match self {
            Self::U128(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_usize(self) -> usize {
        match self {
            Self::Usize(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_size(self) -> ByteSize {
        match self {
            Self::Size(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_string(self) -> String {
        match self {
            Self::String(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_cluster(self) -> ClusterAddress {
        match self {
            Self::Cluster(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_socket(self) -> SocketAddress {
        match self {
            Self::Socket(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_ratio(self) -> Ratio {
        match self {
            Self::Ratio(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_duration_ns(self) -> u64 {
        match self {
            Self::DurationNs(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }

    fn into_enum_index(self) -> usize {
        match self {
            Self::Enum(v) => v,
            _ => panic!("flag kind mismatch"),
        }
    }
}

/// Result of scanning one subcommand's arguments (upstream `parse_flags` during its loop).
struct ParsedFlags {
    named: Vec<(Named, Value)>,
    positionals: Vec<String>,
}

impl ParsedFlags {
    /// The single positional argument of the subcommand.
    fn positional(&self) -> &str {
        &self.positionals[0]
    }
}

/// Split `arg` (matching `flag`) into its value: everything after `=`. Mirrors upstream
/// `parse_flag_split_value` diagnostics exactly.
fn split_flag_equals<'a>(flag: &str, arg: &'a str) -> Result<&'a str, String> {
    let rest = &arg[flag.len()..];
    if rest.is_empty() {
        return Err(format!("{flag}: expected value separator '='"));
    }
    if rest.as_bytes()[0] != b'=' {
        return Err(format!(
            "{flag}: expected value separator '=', but found '{}' in '{arg}'",
            rest.as_bytes()[0] as char
        ));
    }
    if rest.len() == 1 {
        return Err(format!("{flag}: argument requires a value"));
    }
    Ok(&rest[1..])
}

/// Parse a single named argument into a typed [`Value`] (upstream `parse_flag` + `parse_value`).
fn parse_flag_value(def: Named, arg: &str) -> Result<Value, String> {
    let flag = def.flag;
    if def.kind == NamedKind::Bool && arg == flag {
        return Ok(Value::Bool(true));
    }
    let value = split_flag_equals(flag, arg)?;
    match def.kind {
        NamedKind::Bool => {
            // `--flag=<value>` — only reachable for bool flags with an explicit value.
            Err(format!("{flag}: expected one of 'true' or 'false', but found '{value}'"))
        }
        NamedKind::U8 => parse_int_flag(flag, value, parse_int_u8).map(Value::U8),
        NamedKind::U16 => parse_int_flag(flag, value, parse_int_u16).map(Value::U16),
        NamedKind::U32 => parse_int_flag(flag, value, parse_int_u32).map(Value::U32),
        NamedKind::U64 => parse_int_flag(flag, value, parse_int_u64).map(Value::U64),
        NamedKind::U128 => parse_int_flag(flag, value, parse_int_u128).map(Value::U128),
        NamedKind::U6 => {
            let value = parse_int_flag_bits(flag, value, 6)?;
            u8::try_from(value)
                .map(Value::U8)
                .map_err(|_| format!("{flag}: value exceeds 6-bit unsigned integer: '{value}'"))
        }
        NamedKind::Usize => parse_int_flag(flag, value, parse_int_u64).and_then(|v| {
            usize::try_from(v).map(Value::Usize).map_err(|_| {
                format!("{flag}: value exceeds {}-bit unsigned integer: '{value}'", usize::BITS)
            })
        }),
        NamedKind::Size => parse_size_flag(flag, value).map(Value::Size),
        NamedKind::String_ => Ok(Value::String(value.to_owned())),
        NamedKind::Cluster => parse_cluster_flag(flag, value).map(Value::Cluster),
        NamedKind::Socket { default_port } => {
            parse_socket_flag(flag, value, default_port).map(Value::Socket)
        }
        NamedKind::Ratio => parse_ratio_flag(flag, value).map(Value::Ratio),
        NamedKind::DurationNs => parse_duration_flag(flag, value).map(Value::DurationNs),
        NamedKind::Enum(tags) => Ok(Value::Enum(parse_enum_flag(flag, value, tags)?)),
    }
}

/// Parse an unsigned integer flag with a custom bit-width in the overflow diagnostic
/// (`u6` and similar comptime types upstream).
fn parse_int_flag_bits(flag: &str, value: &str, bits: usize) -> Result<u64, String> {
    parse_int_u64(
        value,
        ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: true },
    )
    .map_err(|err| match err {
        ParseIntError::Overflow => {
            format!("{flag}: value exceeds {bits}-bit unsigned integer: '{value}'")
        }
        ParseIntError::InvalidCharacter => {
            format!("{flag}: expected an integer value, but found '{value}' (invalid digit)")
        }
        ParseIntError::LeadingZero => format!("{flag}: leading zero disallowed: '{value}'"),
    })
}

/// Parse an unsigned integer flag (upstream `parse_value_int`. Options:
/// base 10, `_` separators allowed, leading zeros rejected).
#[allow(clippy::cast_possible_wrap)]
fn parse_int_flag<T>(
    flag: &str,
    value: &str,
    parse: fn(&str, ParseIntOptions) -> Result<T, ParseIntError>,
) -> Result<T, String> {
    parse(value, ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: true })
        .map_err(|err| match err {
            ParseIntError::Overflow => {
                format!(
                    "{flag}: value exceeds {}-bit unsigned integer: '{value}'",
                    size_of::<T>() * 8
                )
            }
            ParseIntError::InvalidCharacter => {
                format!("{flag}: expected an integer value, but found '{value}' (invalid digit)")
            }
            ParseIntError::LeadingZero => format!("{flag}: leading zero disallowed: '{value}'"),
        })
}

/// Upstream diagnostic strings for `stdx.ByteSize.parse_flag_value`.
fn byte_size_diagnostic(err: ParseByteSizeError) -> &'static str {
    match err {
        ParseByteSizeError::OverflowValue => "value exceeds 64-bit unsigned integer:",
        ParseByteSizeError::InvalidCharacter => "expected a size, but found:",
        ParseByteSizeError::LeadingZero => "leading zero disallowed:",
        ParseByteSizeError::InvalidUnit => "invalid unit in size, needed KiB, MiB, GiB or TiB:",
        ParseByteSizeError::OverflowBytes => "size in bytes exceeds 64-bit unsigned integer:",
    }
}

fn parse_size_flag(flag: &str, value: &str) -> Result<ByteSize, String> {
    ByteSize::parse_flag_value(value)
        .map_err(|err| format!("{flag}: {} '{value}'", byte_size_diagnostic(err)))
}

fn parse_cluster_flag(flag: &str, value: &str) -> Result<ClusterAddress, String> {
    ClusterAddress::parse_flag_value(value)
        .map_err(|err| format!("{flag}: {} '{value}'", err.message()))
}

fn parse_socket_flag(flag: &str, value: &str, default_port: u16) -> Result<SocketAddress, String> {
    parse_address_and_port(value, default_port).map_err(|err| match err {
        ParseAddressesError::AddressHasMoreThanOneColon => {
            format!("{flag}: invalid address with more than one colon")
        }
        ParseAddressesError::PortInvalid => format!("{flag}: invalid port"),
        ParseAddressesError::AddressInvalid => format!("{flag}: invalid IPv4 or IPv6 address"),
        // A single-address parse cannot produce these, but keep the exhaustiveness honest.
        ParseAddressesError::AddressHasTrailingComma
        | ParseAddressesError::AddressLimitExceeded => {
            format!("{flag}: {}", err.message().trim_end_matches(':'))
        }
    })
}

/// `stdx.PRNG.Ratio.parse_flag_value` (`src/stdx/prng.zig`).
fn parse_ratio_flag(flag: &str, value: &str) -> Result<Ratio, String> {
    if value.len() == 1 && value.as_bytes()[0] == b'0' {
        return Ok(Ratio::zero());
    }
    let Some((numerator, denominator)) = value.split_once('/') else {
        return Err(format!("{flag}: expected 'a/b' ratio, but found: '{value}'"));
    };
    let numerator = parse_int_u64(numerator, ParseIntOptions::default())
        .map_err(|_| format!("{flag}: invalid numerator: '{value}'"))?;
    let denominator = parse_int_u64(denominator, ParseIntOptions::default())
        .map_err(|_| format!("{flag}: invalid denominator: '{value}'"))?;
    if denominator == 0 {
        return Err(format!("{flag}: denominator is zero: '{value}'"));
    }
    if numerator > denominator {
        return Err(format!("{flag}: ratio greater than 1: '{value}'"));
    }
    Ok(Ratio { numerator, denominator })
}

const NS_PER_US: u64 = 1_000;
const NS_PER_MS: u64 = 1_000_000;
const NS_PER_S: u64 = 1_000_000_000;
const NS_PER_MIN: u64 = 60 * NS_PER_S;
const NS_PER_HOUR: u64 = 60 * NS_PER_MIN;
const NS_PER_DAY: u64 = 24 * NS_PER_HOUR;

/// `stdx.Duration.parse_flag_value` (`src/stdx/time_units.zig`): a sequence of `<n><unit>`
/// components with units from `d/h/m/s/ms/us/ns`.
fn parse_duration_flag(flag: &str, value: &str) -> Result<u64, String> {
    const UNITS: &[(&str, u64)] = &[
        ("ns", 1),
        ("us", NS_PER_US),
        ("ms", NS_PER_MS),
        ("s", NS_PER_S),
        ("m", NS_PER_MIN),
        ("h", NS_PER_HOUR),
        ("d", NS_PER_DAY),
    ];

    let mut remaining = value;
    let mut total = 0u64;
    while !remaining.is_empty() {
        let split = remaining
            .as_bytes()
            .iter()
            .position(|c| !c.is_ascii_digit())
            .unwrap_or(remaining.len());
        if split == remaining.len() {
            return Err(format!(
                "{flag}: missing unit; must be one of: d/h/m/s/ms/us/ns: '{value}'"
            ));
        }
        if split == 0 {
            return Err(format!("{flag}: missing value: '{value}'"));
        }
        let amount = parse_int_u64(
            &remaining[..split],
            ParseIntOptions { base: 10, allow_leading_zero: false, allow_separators: true },
        )
        .map_err(|err| match err {
            ParseIntError::Overflow => format!("{flag}: integer overflow: '{value}'"),
            ParseIntError::LeadingZero => format!("{flag}: leading zero disallowed: '{value}'"),
            ParseIntError::InvalidCharacter => {
                // Only ASCII digits reach `parse_int`, so this cannot happen.
                format!("{flag}: integer overflow: '{value}'")
            }
        })?;
        remaining = &remaining[split..];

        let mut matched = None;
        for (unit, ns) in UNITS.iter().copied() {
            if remaining.starts_with(unit) {
                matched = Some((unit, ns));
                break;
            }
        }
        let Some((unit, ns)) = matched else {
            return Err(format!(
                "{flag}: unknown unit; must be one of: d/h/m/s/ms/us/ns: '{value}'"
            ));
        };
        remaining = &remaining[unit.len()..];
        total = total.saturating_add(amount.saturating_mul(ns));
    }

    if total >= 1_000 * NS_PER_DAY {
        return Err(format!("{flag}: duration too large: '{value}'"));
    }
    Ok(total)
}

/// `stdx.flags.parse_value_enum`: `{flag}: expected one of {list}, but found '{value}'`.
/// The list renders like `'a', 'b', or 'c'` (upstream `fields_to_comma_list`).
fn parse_enum_flag(flag: &str, value: &str, tags: &[&str]) -> Result<usize, String> {
    if let Some(index) = tags.iter().position(|tag| *tag == value) {
        return Ok(index);
    }
    let list = tags.iter().enumerate().fold(String::new(), |acc, (i, tag)| {
        let sep = match (i, tags.len()) {
            (0, _) => "",
            (i, n) if i + 1 == n && n == 2 => " or ",
            (i, n) if i + 1 == n => ", or ",
            _ => ", ",
        };
        format!("{acc}{sep}'{tag}'")
    });
    Err(format!("{flag}: expected one of {list}, but found '{value}'"))
}

/// Parse one subcommand's argument list (upstream `parse_flags`):
/// named flags (longest-first match, duplicates rejected), then up to one positional.
fn parse_flags(
    args: &[String],
    named: &[Named],
    positional: Option<&'static str>,
) -> Result<ParsedFlags, String> {
    let mut seen = std::collections::HashSet::new();
    let mut values = Vec::new();
    let mut positionals = Vec::new();
    let mut positional_seen = false;

    for arg in args {
        // Longest-first flag match (upstream sorts declared fields by name length so that
        // e.g. `--aof-file` is not confused for `--aof`).
        let mut matched = None;
        for def in named {
            if arg.starts_with(def.flag) {
                matched = Some(*def);
                break;
            }
        }
        let Some(def) = matched else {
            // No named flag matched: positional or unexpected argument.
            let Some(positional_name) = positional else {
                return Err(format!("unexpected argument: '{arg}'"));
            };
            if arg.is_empty() {
                return Err(format!("{positional_name}: empty argument"));
            }
            if arg.starts_with('-') {
                return Err(format!("unexpected argument: '{arg}'"));
            }
            positional_seen = true;
            positionals.push(arg.clone());
            continue;
        };
        if positional_seen {
            return Err(format!("unexpected trailing option: '{arg}'"));
        }
        if !seen.insert(def.flag) {
            return Err(format!("{}: duplicate argument", def.flag));
        }
        let value = parse_flag_value(def, arg)?;
        values.push((def, value));
    }

    if let Some(name) = positional {
        if positionals.is_empty() {
            return Err(format!("{name}: argument is required"));
        }
        if positionals.len() > 1 {
            return Err(format!("unexpected argument: '{}'", positionals[1]));
        }
    }

    Ok(ParsedFlags { named: values, positionals })
}

fn parse_flags_as(args: &[String], named: &[Named]) -> Result<ParsedFlags, String> {
    parse_flags(args, named, None)
}

// -----------------------------------------------------------------------------------
// Top-level dispatch (upstream `parse_args` + `parse_commands`)
// -----------------------------------------------------------------------------------

const SUBCOMMANDS: &str =
    "format, recover, start, version, repl, benchmark, inspect, multiversion, amqp";

/// Parse `argv` (not including the program name) into a validated `Command`.
///
/// On `-h`/`--help` returns [`ParseFailure::Help`] to make the caller print usage and exit 0.
pub fn parse_args(argv: &[String]) -> Result<Command, ParseFailure> {
    let Some(first) = argv.first() else {
        return Err(ParseFailure::Fatal(format!("subcommand required, expected {SUBCOMMANDS}")));
    };
    if first == "-h" || first == "--help" {
        return Err(ParseFailure::Help(HELP));
    }
    let rest = &argv[1..];
    let parsed = match first.as_str() {
        "format" => parse_args_format(rest).map(Command::Format),
        "recover" => parse_args_recover(rest).map(Command::Recover),
        "start" => parse_args_start(rest).map(Command::Start),
        "version" => parse_args_version(rest).map(Command::Version),
        "repl" => parse_args_repl(rest).map(Command::Repl),
        "benchmark" => parse_args_benchmark(rest).map(Command::Benchmark),
        "inspect" => match parse_args_inspect(rest) {
            Ok(cmd) => Ok(Command::Inspect(cmd)),
            Err(ParseFailure::Help(help)) => return Err(ParseFailure::Help(help)),
            Err(ParseFailure::Fatal(message)) => Err(message),
        },
        "multiversion" => parse_args_multiversion(rest).map(Command::Multiversion),
        "amqp" => parse_args_amqp(rest).map(Command::Amqp),
        other => Err(format!("unknown subcommand: '{other}'")),
    };
    parsed.map_err(ParseFailure::Fatal)
}

// -----------------------------------------------------------------------------------
// Subcommand desugaring — port of `parse_args_{format,recover,start,...}`.
// -----------------------------------------------------------------------------------

/// `parse_args_format`: replica/standby validation and the random-or-supplied cluster id.
// `REPLICAS_MAX`/`STANDBYS_MAX` are small `usize` constants (≤7), cast to `u8` exactly as
// upstream's comptime ints.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::too_many_lines)] // TODO(port): split flag parsing from validation.
fn parse_args_format(args: &[String]) -> Result<CommandFormat, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--replica-count", kind: NamedKind::U8 },
        Named { flag: "--replica", kind: NamedKind::U8 },
        Named { flag: "--standby", kind: NamedKind::U8 },
        Named { flag: "--cluster", kind: NamedKind::U128 },
        Named { flag: "--development", kind: NamedKind::Bool },
        Named { flag: "--log-debug", kind: NamedKind::Bool },
    ];
    let flags = parse_flags(args, NAMED, Some("path"))?;

    let mut cluster = None;
    let mut replica = None;
    let mut standby = None;
    let mut replica_count = None;
    let mut development = false;
    let mut log_debug = false;
    let path = flags.positional().to_owned();
    for (def, value) in flags.named {
        match def.flag {
            "--cluster" => cluster = Some(value.into_u128()),
            "--replica" => replica = Some(value.into_u8()),
            "--standby" => standby = Some(value.into_u8()),
            "--replica-count" => replica_count = Some(value.into_u8()),
            "--development" => development = value.into_bool(),
            "--log-debug" => log_debug = value.into_bool(),
            _ => panic!("unexpected named flag for format"),
        }
    }

    let replica_count =
        replica_count.ok_or_else(|| "--replica-count: argument is required".to_owned())?;
    // Upstream validates `replica_count` in parse_args_format (the flag is required, so its
    // value is present by now).
    if replica_count == 0 {
        return Err("--replica-count: value needs to be greater than zero".to_owned());
    }
    if replica_count > REPLICAS_MAX as u8 {
        return Err(format!(
            "--replica-count: value is too large ({replica_count}), at most {REPLICAS_MAX} is allowed"
        ));
    }

    if replica.is_none() && standby.is_none() {
        return Err("--replica: argument is required".to_owned());
    }
    if replica.is_some() && standby.is_some() {
        return Err("--standby: conflicts with '--replica'".to_owned());
    }
    if let Some(replica) = replica
        && replica >= replica_count
    {
        return Err(format!(
            "--replica: value is too large ({replica}), at most {} is allowed",
            replica_count - 1
        ));
    }
    if let Some(standby) = standby {
        if standby < replica_count {
            return Err(format!(
                "--standby: value is too small ({standby}), at least {replica_count} is required"
            ));
        }
        if standby >= replica_count + STANDBYS_MAX as u8 {
            return Err(format!(
                "--standby: value is too large ({standby}), at most {} is allowed",
                replica_count + STANDBYS_MAX as u8 - 1
            ));
        }
    }

    let Some(replica) = replica.or(standby) else {
        // One of `--replica`/`--standby` was validated to be present above.
        unreachable!();
    };
    assert!(replica < (REPLICAS_MAX + STANDBYS_MAX) as u8);
    assert!(replica < replica_count + STANDBYS_MAX as u8);

    let cluster = match cluster {
        Some(0) => {
            eprintln!(
                "a cluster id of 0 is reserved for testing and benchmarking, do not use in \
                 production"
            );
            eprintln!("omit --cluster=0 to randomly generate a suitable id");
            0
        }
        Some(cluster) => cluster,
        None => {
            // DEVIATION: upstream draws from `std.crypto.random`; Rust `std` has no RNG, so
            // the id is derived from the process id and the clock. State transitions are not
            // affected (this is only a data-file identifier).
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            {
                std::process::id().hash(&mut hasher);
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
                    .hash(&mut hasher);
            }
            let cluster = (hasher.finish() as u128) | 1;
            eprintln!("generated random cluster id: {cluster}");
            cluster
        }
    };

    Ok(CommandFormat { cluster, replica, replica_count, development, path, log_debug })
}

/// `parse_args_recover`.
// `REPLICAS_MAX`/`STANDBYS_MAX` are small `usize` constants (≤7), cast to `u8` exactly as
// upstream's comptime ints.
#[allow(clippy::cast_possible_truncation)]
fn parse_args_recover(args: &[String]) -> Result<CommandRecover, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--replica-count", kind: NamedKind::U8 },
        Named { flag: "--replica", kind: NamedKind::U8 },
        Named { flag: "--addresses", kind: NamedKind::Cluster },
        Named { flag: "--cluster", kind: NamedKind::U128 },
        Named { flag: "--development", kind: NamedKind::Bool },
        Named { flag: "--log-debug", kind: NamedKind::Bool },
    ];
    let flags = parse_flags(args, NAMED, Some("path"))?;

    let mut cluster = None;
    let mut addresses = None;
    let mut replica = None;
    let mut replica_count = None;
    let mut development = false;
    let mut log_debug = false;
    let path = flags.positional().to_owned();
    for (def, value) in flags.named {
        match def.flag {
            "--cluster" => cluster = Some(value.into_u128()),
            "--addresses" => addresses = Some(value.into_cluster()),
            "--replica" => replica = Some(value.into_u8()),
            "--replica-count" => replica_count = Some(value.into_u8()),
            "--development" => development = value.into_bool(),
            "--log-debug" => log_debug = value.into_bool(),
            _ => panic!("unexpected named flag for recover"),
        }
    }

    let cluster = cluster.ok_or_else(|| "--cluster: argument is required".to_owned())?;
    let addresses = addresses.ok_or_else(|| "--addresses: argument is required".to_owned())?;
    let replica = replica.ok_or_else(|| "--replica: argument is required".to_owned())?;
    let replica_count =
        replica_count.ok_or_else(|| "--replica-count: argument is required".to_owned())?;

    if replica_count == 0 {
        return Err("--replica-count: value needs to be greater than zero".to_owned());
    }
    if replica_count > REPLICAS_MAX as u8 {
        return Err(format!(
            "--replica-count: value is too large ({replica_count}), at most {REPLICAS_MAX} is allowed"
        ));
    }
    if replica >= replica_count {
        return Err(format!(
            "--replica: value is too large ({replica}), at most {} is allowed",
            replica_count - 1
        ));
    }
    if replica_count <= 2 {
        return Err("--replica-count: 1- or 2- replica clusters don't support 'recover'".to_owned());
    }

    assert!(replica < (REPLICAS_MAX + STANDBYS_MAX) as u8);
    assert!(replica < replica_count);

    Ok(CommandRecover { cluster, addresses, replica, replica_count, development, path, log_debug })
}

/// `start_defaults_production`/`start_defaults_development` (cli.zig rows 465-502).
struct StartDefaults {
    limit_pipeline_requests: u32,
    limit_request: u32,
    cache_accounts: u64,
    cache_transfers: u64,
    cache_transfers_pending: u64,
    cache_grid: u64,
    memory_lsm_compaction: u64,
}

/// `lsm_compaction_block_count_min` = `compaction_block_count_beat_min`
/// (upstream `src/lsm/compaction.zig:73`): one index + one value block for the output table,
/// one index block for level A, two index blocks for level B, and
/// `lsm_compaction_queue_read_max/2` value blocks for the two input tables.
// Upstream computes these with comptime ints; the `usize` constants are small (≤16), so the
// cast to `u32` cannot truncate.
#[allow(clippy::cast_possible_truncation)]
const LSM_COMPACTION_BLOCK_COUNT_MIN: u32 =
    (1 + 1) + (1 + 2) + LSM_COMPACTION_QUEUE_READ_MAX as u32;

const fn block_memory(blocks: u32) -> u64 {
    blocks as u64 * tigerbeetle_core::constants::BLOCK_SIZE as u64
}

fn start_defaults_production() -> StartDefaults {
    StartDefaults {
        // Upstream: `div_ceil(clients_max, 2) - pipeline_prepare_queue_max`:
        limit_pipeline_requests: CLIENTS_MAX.div_ceil(2) - PIPELINE_PREPARE_QUEUE_MAX,
        limit_request: MESSAGE_SIZE_MAX,
        cache_accounts: CACHE_ACCOUNTS_SIZE_DEFAULT as u64,
        cache_transfers: CACHE_TRANSFERS_SIZE_DEFAULT as u64,
        cache_transfers_pending: CACHE_TRANSFERS_PENDING_SIZE_DEFAULT as u64,
        cache_grid: GRID_CACHE_SIZE_DEFAULT as u64,
        // Upstream: `(lsm_compaction_block_count_min + lsm_compaction_iops_write_max) * block_size`.
        // Upstream computes this with comptime ints; `LSM_COMPACTION_IOPS_WRITE_MAX` is a small
        // `usize` constant, so the cast to `u32` cannot truncate.
        #[allow(clippy::cast_possible_truncation)]
        memory_lsm_compaction: block_memory(
            LSM_COMPACTION_BLOCK_COUNT_MIN
                + tigerbeetle_core::constants::LSM_COMPACTION_IOPS_WRITE_MAX as u32,
        ),
    }
}

fn start_defaults_development() -> StartDefaults {
    StartDefaults {
        limit_pipeline_requests: 0,
        limit_request: 32 * 1024, // 32KiB, the upstream development default.
        cache_accounts: 0,
        cache_transfers: 0,
        cache_transfers_pending: 0,
        cache_grid: block_memory(1) * cache_value_count_max_multiple(),
        memory_lsm_compaction: block_memory(LSM_COMPACTION_BLOCK_COUNT_MIN),
    }
}

/// `parse_cache_size_to_count`: the largest `value_count_max` (a multiple of
/// `value_count_max_multiple`) that fits `size` for `value_size`-byte values.
fn parse_cache_size_to_count(
    value_size: u64,
    value_count_max_multiple: u64,
    size: ByteSize,
    cli_flag: &str,
) -> Result<u32, String> {
    let count_limit = size.bytes() / value_size;
    let count_rounded = (count_limit / value_count_max_multiple) * value_count_max_multiple;
    let result =
        u32::try_from(count_rounded).map_err(|_| format!("{cli_flag}: exceeds the limit"))?;
    assert!(result as u64 * value_size <= size.bytes());
    Ok(result)
}

fn cache_value_count_max_multiple() -> u64 {
    tigerbeetle_vsr::grid::cache_value_count_max_multiple()
}

/// `memory_split_bytes` (upstream `memory_split_bytes(memory_bytes, percent)`).
fn memory_split_bytes(memory_bytes: u64, percent: u64) -> u64 {
    assert!(percent <= 100);
    memory_bytes.saturating_mul(percent) / 100
}

/// Upstream `MemorySplit.default`: fields must sum to 100. Field names mirror upstream.
#[allow(clippy::struct_field_names)]
struct MemorySplit {
    cache_grid: u64,
    cache_accounts: u64,
    cache_transfers: u64,
    cache_transfers_pending: u64,
}

const MEMORY_SPLIT_DEFAULT: MemorySplit = MemorySplit {
    cache_grid: 64,
    cache_accounts: 32,
    cache_transfers: 0,
    cache_transfers_pending: 4,
};

/// Field names mirror upstream.
#[allow(clippy::struct_field_names)]
struct CacheSizes {
    cache_grid: ByteSize,
    cache_accounts: ByteSize,
    cache_transfers: ByteSize,
    cache_transfers_pending: ByteSize,
}

/// `parse_timeout_to_ticks`: `ms` must be a non-zero multiple of `tick_ms`.
fn parse_timeout_to_ticks(timeout_ms: Option<u64>, cli_flag: &str) -> Result<Option<u64>, String> {
    let Some(ms) = timeout_ms else { return Ok(None) };
    if ms == 0 {
        return Err(format!("{cli_flag}: timeout {ms}ms be nonzero"));
    }
    if ms % TICK_MS != 0 {
        return Err(format!("{cli_flag}: timeout {ms}ms must be a multiple of {TICK_MS}ms"));
    }
    Ok(Some(ms / TICK_MS))
}

/// `parse_args_start` — the largest desugar: experimental gating, cache sizes, storage/
/// pipeline/request/manifest/compaction bounds, AOF, timeouts and the statsd socket.
#[allow(clippy::too_many_lines)] // mirrors upstream's straight-line validation
fn parse_args_start(args: &[String]) -> Result<CommandStart, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--timeout-grid-repair-message-ms", kind: NamedKind::U64 },
        Named { flag: "--commit-stall-multiple-max", kind: NamedKind::U16 },
        Named { flag: "--commit-stall-probability", kind: NamedKind::Ratio },
        Named { flag: "--commit-stall-lag-max", kind: NamedKind::U32 },
        Named { flag: "--commit-stall-lag-min", kind: NamedKind::U32 },
        Named { flag: "--timeout-prepare-ms", kind: NamedKind::U64 },
        Named { flag: "--limit-pipeline-requests", kind: NamedKind::U32 },
        Named { flag: "--cache-transfers-pending", kind: NamedKind::Size },
        Named { flag: "--memory-lsm-compaction", kind: NamedKind::Size },
        Named { flag: "--memory-lsm-manifest", kind: NamedKind::Size },
        Named { flag: "--aof-recovery", kind: NamedKind::Bool },
        Named { flag: "--aof-file", kind: NamedKind::String_ },
        Named { flag: "--cache-transfers", kind: NamedKind::Size },
        Named { flag: "--cache-accounts", kind: NamedKind::Size },
        Named { flag: "--limit-request", kind: NamedKind::Size },
        Named { flag: "--limit-storage", kind: NamedKind::Size },
        Named { flag: "--cache-grid", kind: NamedKind::Size },
        Named { flag: "--replicate-star", kind: NamedKind::Bool },
        Named { flag: "--experimental", kind: NamedKind::Bool },
        Named { flag: "--development", kind: NamedKind::Bool },
        Named { flag: "--log-trace", kind: NamedKind::Bool },
        Named { flag: "--log-debug", kind: NamedKind::Bool },
        Named { flag: "--addresses", kind: NamedKind::Cluster },
        Named { flag: "--memory", kind: NamedKind::Size },
        Named { flag: "--trace", kind: NamedKind::String_ },
        Named { flag: "--statsd", kind: NamedKind::Socket { default_port: 8125 } },
        Named { flag: "--aof", kind: NamedKind::Bool },
    ];

    let flags = parse_flags(args, NAMED, Some("path"))?;

    // Collect raw CLI arguments (upstream `CLIArgs.Start` fields).
    let mut addresses = None;
    let mut cache_grid = None;
    let mut development = false;
    let mut experimental = false;
    let mut limit_storage = None;
    let mut limit_pipeline_requests = None;
    let mut limit_request = None;
    let mut memory = None;
    let mut cache_accounts = None;
    let mut cache_transfers = None;
    let mut cache_transfers_pending = None;
    let mut memory_lsm_manifest = None;
    let mut memory_lsm_compaction = None;
    let mut trace = None;
    let mut log_debug = false;
    let mut log_trace = false;
    let mut timeout_prepare_ms = None;
    let mut timeout_grid_repair_message_ms = None;
    let mut commit_stall_probability = None;
    let mut commit_stall_lag_min = None;
    let mut commit_stall_lag_max = None;
    let mut commit_stall_multiple_max = None;
    let mut replicate_star = false;
    let mut aof_file = None;
    let mut aof = false;
    let mut aof_recovery = false;
    let mut statsd = None;
    let path = flags.positional().to_owned();
    for (def, value) in flags.named {
        match def.flag {
            "--addresses" => addresses = Some(value.into_cluster()),
            "--cache-grid" => cache_grid = Some(value.into_size()),
            "--development" => development = value.into_bool(),
            "--experimental" => experimental = value.into_bool(),
            "--limit-storage" => limit_storage = Some(value.into_size()),
            "--limit-pipeline-requests" => limit_pipeline_requests = Some(value.into_u32()),
            "--limit-request" => limit_request = Some(value.into_size()),
            "--memory" => memory = Some(value.into_size()),
            "--cache-accounts" => cache_accounts = Some(value.into_size()),
            "--cache-transfers" => cache_transfers = Some(value.into_size()),
            "--cache-transfers-pending" => cache_transfers_pending = Some(value.into_size()),
            "--memory-lsm-manifest" => memory_lsm_manifest = Some(value.into_size()),
            "--memory-lsm-compaction" => memory_lsm_compaction = Some(value.into_size()),
            "--trace" => trace = Some(value.into_string()),
            "--log-debug" => log_debug = value.into_bool(),
            "--log-trace" => log_trace = value.into_bool(),
            "--timeout-prepare-ms" => timeout_prepare_ms = Some(value.into_u64()),
            "--timeout-grid-repair-message-ms" => {
                timeout_grid_repair_message_ms = Some(value.into_u64());
            }
            "--commit-stall-probability" => commit_stall_probability = Some(value.into_ratio()),
            "--commit-stall-lag-min" => commit_stall_lag_min = Some(value.into_u32()),
            "--commit-stall-lag-max" => commit_stall_lag_max = Some(value.into_u32()),
            "--commit-stall-multiple-max" => commit_stall_multiple_max = Some(value.into_u16()),
            "--replicate-star" => replicate_star = value.into_bool(),
            "--aof-file" => aof_file = Some(value.into_string()),
            "--aof" => aof = value.into_bool(),
            "--aof-recovery" => aof_recovery = value.into_bool(),
            "--statsd" => statsd = Some(value.into_socket()),
            _ => panic!("unexpected named flag for start"),
        }
    }
    let addresses = addresses.ok_or_else(|| "--addresses: argument is required".to_owned())?;

    // Experimental flags require `--experimental` (stable allowlist: addresses, cache-grid,
    // development, experimental).
    let experimental_flags: &[(&str, bool)] = &[
        ("--limit-storage", limit_storage.is_some()),
        ("--limit-pipeline-requests", limit_pipeline_requests.is_some()),
        ("--limit-request", limit_request.is_some()),
        ("--memory", memory.is_some()),
        ("--cache-accounts", cache_accounts.is_some()),
        ("--cache-transfers", cache_transfers.is_some()),
        ("--cache-transfers-pending", cache_transfers_pending.is_some()),
        ("--memory-lsm-manifest", memory_lsm_manifest.is_some()),
        ("--memory-lsm-compaction", memory_lsm_compaction.is_some()),
        ("--trace", trace.is_some()),
        ("--log-debug", log_debug),
        ("--log-trace", log_trace),
        ("--timeout-prepare-ms", timeout_prepare_ms.is_some()),
        ("--timeout-grid-repair-message-ms", timeout_grid_repair_message_ms.is_some()),
        ("--commit-stall-probability", commit_stall_probability.is_some()),
        ("--commit-stall-lag-min", commit_stall_lag_min.is_some()),
        ("--commit-stall-lag-max", commit_stall_lag_max.is_some()),
        ("--commit-stall-multiple-max", commit_stall_multiple_max.is_some()),
        ("--replicate-star", replicate_star),
        ("--aof-file", aof_file.is_some()),
        ("--aof", aof),
        ("--aof-recovery", aof_recovery),
        ("--statsd", statsd.is_some()),
    ];
    if !experimental {
        for (flag, set) in experimental_flags {
            if *set {
                return Err(format!(
                    "{flag} is marked experimental, add `--experimental` to continue."
                ));
            }
        }
    }

    let defaults =
        if development { start_defaults_development() } else { start_defaults_production() };

    if memory.is_some() {
        for (flag, arg) in [
            (cache_grid.is_some(), "--cache-grid"),
            (cache_accounts.is_some(), "--cache-accounts"),
            (cache_transfers.is_some(), "--cache-transfers"),
            (cache_transfers_pending.is_some(), "--cache-transfers-pending"),
        ] {
            if flag {
                return Err(format!("--memory is mutually exclusive with {arg}"));
            }
        }
    }

    let cache_sizes = if let Some(memory) = memory {
        let memory_bytes = memory.bytes();
        CacheSizes {
            cache_grid: ByteSize {
                value: memory_split_bytes(memory_bytes, MEMORY_SPLIT_DEFAULT.cache_grid),
                unit: ByteSizeUnit::Bytes,
            },
            cache_accounts: ByteSize {
                value: memory_split_bytes(memory_bytes, MEMORY_SPLIT_DEFAULT.cache_accounts),
                unit: ByteSizeUnit::Bytes,
            },
            cache_transfers: ByteSize {
                value: memory_split_bytes(memory_bytes, MEMORY_SPLIT_DEFAULT.cache_transfers),
                unit: ByteSizeUnit::Bytes,
            },
            cache_transfers_pending: ByteSize {
                value: memory_split_bytes(
                    memory_bytes,
                    MEMORY_SPLIT_DEFAULT.cache_transfers_pending,
                ),
                unit: ByteSizeUnit::Bytes,
            },
        }
    } else {
        CacheSizes {
            cache_grid: cache_grid
                .unwrap_or(ByteSize { value: defaults.cache_grid, unit: ByteSizeUnit::Bytes }),
            cache_accounts: cache_accounts
                .unwrap_or(ByteSize { value: defaults.cache_accounts, unit: ByteSizeUnit::Bytes }),
            cache_transfers: cache_transfers
                .unwrap_or(ByteSize { value: defaults.cache_transfers, unit: ByteSizeUnit::Bytes }),
            cache_transfers_pending: cache_transfers_pending.unwrap_or(ByteSize {
                value: defaults.cache_transfers_pending,
                unit: ByteSizeUnit::Bytes,
            }),
        }
    };

    let start_limit_storage = limit_storage.unwrap_or(ByteSize {
        value: STORAGE_SIZE_LIMIT_DEFAULT as u64,
        unit: ByteSizeUnit::Bytes,
    });
    let start_memory_lsm_manifest = memory_lsm_manifest.unwrap_or(ByteSize {
        value: LSM_MANIFEST_MEMORY_SIZE_DEFAULT as u64,
        unit: ByteSizeUnit::Bytes,
    });

    let storage_size_limit = start_limit_storage.bytes();
    let storage_size_limit_min = tigerbeetle_vsr::superblock::DATA_FILE_SIZE_MIN as u64;
    let storage_size_limit_max = STORAGE_SIZE_LIMIT_MAX as u64;
    if storage_size_limit > storage_size_limit_max {
        return Err(format!(
            "--limit-storage: size {}{} exceeds maximum: {}",
            start_limit_storage.value,
            start_limit_storage.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(storage_size_limit_max)
        ));
    }
    if storage_size_limit < storage_size_limit_min {
        return Err(format!(
            "--limit-storage: size {}{} is below minimum: {}",
            start_limit_storage.value,
            start_limit_storage.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(storage_size_limit_min)
        ));
    }
    if storage_size_limit % SECTOR_SIZE as u64 != 0 {
        return Err(format!(
            "--limit-storage: size {}{} must be a multiple of sector size ({})",
            start_limit_storage.value,
            start_limit_storage.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(SECTOR_SIZE as u64)
        ));
    }

    let pipeline_limit = limit_pipeline_requests.unwrap_or(defaults.limit_pipeline_requests);
    let pipeline_limit_min = 0u32;
    let pipeline_limit_max = PIPELINE_REQUEST_QUEUE_MAX;
    if pipeline_limit > pipeline_limit_max {
        return Err(format!(
            "--limit-pipeline-requests: count {pipeline_limit} exceeds maximum: {pipeline_limit_max}"
        ));
    }
    if pipeline_limit < pipeline_limit_min {
        return Err(format!(
            "--limit-pipeline-requests: count {pipeline_limit} is below minimum: {pipeline_limit_min}"
        ));
    }

    let request_size_limit = limit_request
        .unwrap_or(ByteSize { value: defaults.limit_request as u64, unit: ByteSizeUnit::Bytes });
    let request_size_limit_min = 4096u64;
    let request_size_limit_max = MESSAGE_SIZE_MAX as u64;
    if request_size_limit.bytes() > request_size_limit_max {
        return Err(format!(
            "--limit-request: size {}{} exceeds maximum: {}",
            request_size_limit.value,
            request_size_limit.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(request_size_limit_max)
        ));
    }
    if request_size_limit.bytes() < request_size_limit_min {
        return Err(format!(
            "--limit-request: size {}{} is below minimum: {}",
            request_size_limit.value,
            request_size_limit.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(request_size_limit_min)
        ));
    }

    let lsm_manifest_memory = start_memory_lsm_manifest.bytes();
    let lsm_manifest_memory_max = LSM_MANIFEST_MEMORY_SIZE_MAX as u64;
    let lsm_manifest_memory_min = LSM_MANIFEST_MEMORY_SIZE_MIN as u64;
    let lsm_manifest_memory_multiplier = LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER as u64;
    if lsm_manifest_memory > lsm_manifest_memory_max {
        return Err(format!(
            "--memory-lsm-manifest: size {}{} exceeds maximum: {}",
            start_memory_lsm_manifest.value,
            start_memory_lsm_manifest.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_manifest_memory_max)
        ));
    }
    if lsm_manifest_memory < lsm_manifest_memory_min {
        return Err(format!(
            "--memory-lsm-manifest: size {}{} is below minimum: {}",
            start_memory_lsm_manifest.value,
            start_memory_lsm_manifest.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_manifest_memory_min)
        ));
    }
    if lsm_manifest_memory % lsm_manifest_memory_multiplier != 0 {
        return Err(format!(
            "--memory-lsm-manifest: size {}{} must be a multiple of {}",
            start_memory_lsm_manifest.value,
            start_memory_lsm_manifest.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_manifest_memory_multiplier)
        ));
    }

    let lsm_compaction_block_memory = memory_lsm_compaction
        .unwrap_or(ByteSize { value: defaults.memory_lsm_compaction, unit: ByteSizeUnit::Bytes });
    let lsm_compaction_block_memory_max =
        tigerbeetle_core::constants::COMPACTION_BLOCK_MEMORY_SIZE_MAX;
    let lsm_compaction_block_memory_min = block_memory(LSM_COMPACTION_BLOCK_COUNT_MIN);
    if lsm_compaction_block_memory.bytes() > lsm_compaction_block_memory_max {
        return Err(format!(
            "--memory-lsm-compaction: size {}{} exceeds maximum: {}",
            lsm_compaction_block_memory.value,
            lsm_compaction_block_memory.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_compaction_block_memory_max)
        ));
    }
    if lsm_compaction_block_memory.bytes() < lsm_compaction_block_memory_min {
        return Err(format!(
            "--memory-lsm-compaction: size {}{} is below minimum: {}",
            lsm_compaction_block_memory.value,
            lsm_compaction_block_memory.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_compaction_block_memory_min)
        ));
    }
    if lsm_compaction_block_memory.bytes() % tigerbeetle_core::constants::BLOCK_SIZE as u64 != 0 {
        return Err(format!(
            "--memory-lsm-compaction: size {}{} must be a multiple of {}",
            lsm_compaction_block_memory.value,
            lsm_compaction_block_memory.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(
                tigerbeetle_core::constants::BLOCK_SIZE as u64
            )
        ));
    }

    // The user gears memory within `--memory`; the derived block/node counts stay far below
    // `u32::MAX` (a grid of >17TiB, a manifest of >4EiB). Cap instead of panicking, per the
    // "no panics on invalid input" rule.
    let lsm_forest_compaction_block_count = u32::try_from(
        lsm_compaction_block_memory.bytes() / tigerbeetle_core::constants::BLOCK_SIZE as u64,
    )
    .unwrap_or(u32::MAX);
    let lsm_forest_node_count =
        u32::try_from(lsm_manifest_memory / LSM_MANIFEST_NODE_SIZE as u64).unwrap_or(u32::MAX);

    let aof_file = if aof {
        if aof_file.is_some() {
            return Err("--aof is mutually exclusive with --aof-file".to_owned());
        }
        if PATH_MAX_BYTES < path.len() + 4 {
            return Err("data file path is too long for --aof. use --aof-file".to_owned());
        }
        let aof_file = format!("{path}.aof");
        eprintln!("--aof is deprecated. consider switching to '--aof-file={aof_file}'");
        Some(aof_file)
    } else if let Some(aof_file) = aof_file {
        // Upstream compares `.aof` case-sensitively.
        #[allow(clippy::case_sensitive_file_extension_comparisons)]
        if !aof_file.ends_with(".aof") {
            return Err(format!("--aof-file must end with .aof: '{aof_file}'"));
        }
        if PATH_MAX_BYTES < aof_file.len() {
            return Err("--aof-file path is too long".to_owned());
        }
        Some(aof_file)
    } else {
        None
    };

    if log_trace && !log_debug {
        return Err("--log-debug must be provided when using --log-trace".to_owned());
    }

    if replicate_star {
        eprintln!("--replicate-star is deprecated; star replication is now the default.");
    }

    let request_size_limit = u32::try_from(request_size_limit.bytes())
        .map_err(|_| "--limit-request: size exceeds u32".to_owned())?;

    Ok(CommandStart {
        addresses,
        cache_accounts: parse_cache_size_to_count(
            size_of::<Account>() as u64,
            SetAssociativeCache::<AccountObjectsCacheSpec>::value_count_max_multiple(),
            cache_sizes.cache_accounts,
            "--cache-accounts",
        )?,
        cache_transfers: parse_cache_size_to_count(
            size_of::<Transfer>() as u64,
            SetAssociativeCache::<TransferObjectsCacheSpec>::value_count_max_multiple(),
            cache_sizes.cache_transfers,
            "--cache-transfers",
        )?,
        cache_transfers_pending: parse_cache_size_to_count(
            size_of::<TransferPending>() as u64,
            SetAssociativeCache::<TransferPendingObjectsCacheSpec>::value_count_max_multiple(),
            cache_sizes.cache_transfers_pending,
            "--cache-transfers-pending",
        )?,
        cache_grid_blocks: parse_cache_size_to_count(
            tigerbeetle_core::constants::BLOCK_SIZE as u64,
            cache_value_count_max_multiple(),
            cache_sizes.cache_grid,
            "--cache-grid",
        )?,
        storage_size_limit,
        pipeline_requests_limit: pipeline_limit,
        request_size_limit,
        lsm_forest_compaction_block_count,
        lsm_forest_node_count,
        timeout_prepare_ticks: parse_timeout_to_ticks(timeout_prepare_ms, "--timeout-prepare-ms")?,
        timeout_grid_repair_message_ticks: parse_timeout_to_ticks(
            timeout_grid_repair_message_ms,
            "--timeout-grid-repair-message-ms",
        )?,
        commit_stall_probability,
        commit_stall_lag_min,
        commit_stall_lag_max,
        commit_stall_multiple_max,
        trace,
        development,
        experimental,
        replicate_star,
        aof_file,
        aof_recovery,
        path,
        log_debug,
        log_trace,
        statsd,
    })
}

/// `parse_args_version`.
fn parse_args_version(args: &[String]) -> Result<CommandVersion, String> {
    const NAMED: &[Named] = &[Named { flag: "--verbose", kind: NamedKind::Bool }];
    let flags = parse_flags_as(args, NAMED)?;
    let mut verbose = false;
    for (def, value) in flags.named {
        match def.flag {
            "--verbose" => verbose = value.into_bool(),
            _ => panic!("unexpected named flag for version"),
        }
    }
    Ok(CommandVersion { verbose })
}

/// `parse_args_repl`.
fn parse_args_repl(args: &[String]) -> Result<CommandRepl, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--addresses", kind: NamedKind::Cluster },
        Named { flag: "--cluster", kind: NamedKind::U128 },
        Named { flag: "--verbose", kind: NamedKind::Bool },
        Named { flag: "--command", kind: NamedKind::String_ },
        Named { flag: "--log-debug", kind: NamedKind::Bool },
    ];
    let flags = parse_flags_as(args, NAMED)?;

    let mut addresses = None;
    let mut cluster = None;
    let mut verbose = false;
    let mut command = String::new();
    let mut log_debug = false;
    for (def, value) in flags.named {
        match def.flag {
            "--addresses" => addresses = Some(value.into_cluster()),
            "--cluster" => cluster = Some(value.into_u128()),
            "--verbose" => verbose = value.into_bool(),
            "--command" => command = value.into_string(),
            "--log-debug" => log_debug = value.into_bool(),
            _ => panic!("unexpected named flag for repl"),
        }
    }

    let addresses = addresses.ok_or_else(|| "--addresses: argument is required".to_owned())?;
    let cluster = cluster.ok_or_else(|| "--cluster: argument is required".to_owned())?;

    Ok(CommandRepl { addresses, cluster, verbose, statements: command, log_debug })
}

fn account_batch_count_max() -> u32 {
    u32::try_from(MESSAGE_BODY_SIZE_MAX / size_of::<Account>()).unwrap_or(u32::MAX)
}

fn transfer_batch_count_max() -> u32 {
    u32::try_from(MESSAGE_BODY_SIZE_MAX / size_of::<Transfer>()).unwrap_or(u32::MAX)
}

/// `parse_args_benchmark`.
#[allow(clippy::too_many_lines)] // TODO(port): split flag parsing from validation.
fn parse_args_benchmark(args: &[String]) -> Result<CommandBenchmark, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--cache-transfers-pending", kind: NamedKind::String_ },
        Named { flag: "--checksum-performance", kind: NamedKind::Bool },
        Named { flag: "--print-batch-timings", kind: NamedKind::Bool },
        Named { flag: "--transfer-batch-count", kind: NamedKind::U32 },
        Named { flag: "--transfer-hot-percent", kind: NamedKind::U32 },
        Named { flag: "--account-batch-count", kind: NamedKind::U32 },
        Named {
            flag: "--account-distribution",
            kind: NamedKind::Enum(&["zipfian", "latest", "uniform"]),
        },
        Named { flag: "--account-count-hot", kind: NamedKind::U32 },
        Named { flag: "--transfer-batch-delay", kind: NamedKind::DurationNs },
        Named { flag: "--log-debug-replica", kind: NamedKind::Bool },
        Named { flag: "--cache-transfers", kind: NamedKind::String_ },
        Named { flag: "--cache-accounts", kind: NamedKind::String_ },
        Named { flag: "--account-count", kind: NamedKind::U64 },
        Named { flag: "--transfer-count", kind: NamedKind::U64 },
        Named { flag: "--no-history", kind: NamedKind::Bool },
        Named { flag: "--transfer-pending", kind: NamedKind::Bool },
        Named { flag: "--query-count", kind: NamedKind::U32 },
        Named { flag: "--cache-grid", kind: NamedKind::String_ },
        Named { flag: "--log-debug", kind: NamedKind::Bool },
        Named {
            flag: "--id-order",
            kind: NamedKind::Enum(&["tbid", "sequential", "random", "reversed"]),
        },
        Named { flag: "--validate", kind: NamedKind::Bool },
        Named { flag: "--imported", kind: NamedKind::Bool },
        Named { flag: "--clients", kind: NamedKind::U32 },
        Named { flag: "--memory", kind: NamedKind::String_ },
        Named { flag: "--statsd", kind: NamedKind::String_ },
        Named { flag: "--trace", kind: NamedKind::String_ },
        Named { flag: "--seed", kind: NamedKind::String_ },
        Named { flag: "--file", kind: NamedKind::String_ },
        Named { flag: "--addresses", kind: NamedKind::Cluster },
    ];

    let flags = parse_flags_as(args, NAMED)?;

    let mut cache_accounts = None;
    let mut cache_transfers = None;
    let mut cache_transfers_pending = None;
    let mut cache_grid = None;
    let mut memory = None;
    let mut log_debug = false;
    let mut log_debug_replica = false;
    let mut account_count = 10_000u64;
    let mut account_count_hot = 0u32;
    let mut account_distribution = 2usize; // uniform
    let mut no_history = false;
    let mut imported = false;
    let mut account_batch_count = account_batch_count_max();
    let mut transfer_count = 10_000_000u64;
    let mut transfer_hot_percent = 100u32;
    let mut transfer_pending = false;
    let mut transfer_batch_count = transfer_batch_count_max();
    let mut transfer_batch_delay = 0u64;
    let mut validate = false;
    let mut checksum_performance = false;
    let mut query_count = 100u32;
    let mut print_batch_timings = false;
    let mut id_order = 0usize; // tbid
    let mut clients = 1u32;
    let mut statsd = None;
    let mut trace = None;
    let mut file = None;
    let mut addresses = None;
    let mut seed = None;
    for (def, value) in flags.named {
        match def.flag {
            "--cache-transfers-pending" => cache_transfers_pending = Some(value.into_string()),
            "--checksum-performance" => checksum_performance = value.into_bool(),
            "--print-batch-timings" => print_batch_timings = value.into_bool(),
            "--transfer-batch-count" => transfer_batch_count = value.into_u32(),
            "--transfer-hot-percent" => transfer_hot_percent = value.into_u32(),
            "--account-batch-count" => account_batch_count = value.into_u32(),
            "--account-distribution" => account_distribution = value.into_enum_index(),
            "--account-count-hot" => account_count_hot = value.into_u32(),
            "--transfer-batch-delay" => transfer_batch_delay = value.into_duration_ns(),
            "--log-debug-replica" => log_debug_replica = value.into_bool(),
            "--cache-transfers" => cache_transfers = Some(value.into_string()),
            "--cache-accounts" => cache_accounts = Some(value.into_string()),
            "--account-count" => account_count = value.into_u64(),
            "--transfer-count" => transfer_count = value.into_u64(),
            "--no-history" => no_history = value.into_bool(),
            "--transfer-pending" => transfer_pending = value.into_bool(),
            "--query-count" => query_count = value.into_u32(),
            "--cache-grid" => cache_grid = Some(value.into_string()),
            "--log-debug" => log_debug = value.into_bool(),
            "--id-order" => id_order = value.into_enum_index(),
            "--validate" => validate = value.into_bool(),
            "--imported" => imported = value.into_bool(),
            "--clients" => clients = value.into_u32(),
            "--memory" => memory = Some(value.into_string()),
            "--statsd" => statsd = Some(value.into_string()),
            "--trace" => trace = Some(value.into_string()),
            "--seed" => seed = Some(value.into_string()),
            "--file" => file = Some(value.into_string()),
            "--addresses" => addresses = Some(value.into_cluster()),
            _ => panic!("unexpected named flag for benchmark"),
        }
    }

    if addresses.is_some() && file.is_some() {
        return Err("--file: --addresses and --file are mutually exclusive".to_owned());
    }
    if account_batch_count == 0 {
        return Err("--account-batch-count must be greater than 0".to_owned());
    }
    let account_batch_count_max = account_batch_count_max();
    if account_batch_count > account_batch_count_max {
        return Err(format!(
            "--account-batch-count must be less than or equal to {account_batch_count_max}"
        ));
    }
    if transfer_batch_count == 0 {
        return Err("--transfer-batch-count must be greater than 0".to_owned());
    }
    let transfer_batch_count_max = transfer_batch_count_max();
    if transfer_batch_count > transfer_batch_count_max {
        return Err(format!(
            "--transfer-batch-count must be less than or equal to {transfer_batch_count_max}"
        ));
    }

    Ok(CommandBenchmark {
        cache_accounts,
        cache_transfers,
        cache_transfers_pending,
        cache_grid,
        memory,
        log_debug,
        log_debug_replica,
        account_count,
        account_count_hot,
        account_distribution: match account_distribution {
            0 => BenchmarkDistribution::Zipfian,
            1 => BenchmarkDistribution::Latest,
            2 => BenchmarkDistribution::Uniform,
            _ => unreachable!(),
        },
        no_history,
        imported,
        account_batch_count,
        transfer_count,
        transfer_hot_percent,
        transfer_pending,
        transfer_batch_count,
        transfer_batch_delay,
        validate,
        checksum_performance,
        query_count,
        print_batch_timings,
        id_order: match id_order {
            0 => BenchmarkIdOrder::Tbid,
            1 => BenchmarkIdOrder::Sequential,
            2 => BenchmarkIdOrder::Random,
            3 => BenchmarkIdOrder::Reversed,
            _ => unreachable!(),
        },
        clients,
        statsd,
        trace,
        file,
        addresses,
        seed,
    })
}

/// The `inspect` sub-subcommand list (upstream `CLIArgs.Inspect` union fields).
const INSPECT_SUBCOMMANDS: &str =
    "constants, metrics, op, superblock, wal, replies, grid, manifest, tables, integrity";

/// `parse_args` for the `inspect` union (upstream `parse_commands` of `CLIArgs.Inspect`).
///
/// Help must propagate out of the `inspect` layer, so this returns [`ParseFailure`] directly
/// (the parent `parse_args` unwraps the `Help` case before mapping the rest to `Fatal`).
fn parse_args_inspect(args: &[String]) -> Result<CommandInspect, ParseFailure> {
    let Some(first) = args.first() else {
        return Err(ParseFailure::Fatal(format!(
            "subcommand required, expected {INSPECT_SUBCOMMANDS}"
        )));
    };
    if first == "-h" || first == "--help" {
        return Err(ParseFailure::Help(INSPECT_HELP));
    }
    let rest = &args[1..];
    let parsed = match first.as_str() {
        "constants" => {
            if !rest.is_empty() {
                return Err(ParseFailure::Fatal(format!("unexpected argument: '{}'", rest[0])));
            }
            Ok(CommandInspect::Constants)
        }
        "metrics" => {
            if !rest.is_empty() {
                return Err(ParseFailure::Fatal(format!("unexpected argument: '{}'", rest[0])));
            }
            Ok(CommandInspect::Metrics)
        }
        "op" => parse_args_inspect_op(rest),
        "superblock" => parse_args_inspect_data_file(rest, |_| InspectQuery::Superblock),
        "wal" => parse_args_inspect_data_file(rest, |f| InspectQuery::Wal { slot: f.slot }),
        "replies" => parse_args_inspect_data_file(rest, |f| InspectQuery::Replies {
            slot: f.slot,
            superblock_copy: f.superblock_copy,
        }),
        "grid" => parse_args_inspect_data_file(rest, |f| InspectQuery::Grid {
            block: f.block,
            superblock_copy: f.superblock_copy,
        }),
        "manifest" => parse_args_inspect_data_file(rest, |f| InspectQuery::Manifest {
            superblock_copy: f.superblock_copy,
        }),
        "tables" => parse_args_inspect_data_file(rest, |f| InspectQuery::Tables {
            superblock_copy: f.superblock_copy,
            tree: f.tree.unwrap_or_default(),
            level: f.level,
        }),
        "integrity" => parse_args_inspect_integrity(rest),
        other => Err(format!("unknown subcommand: '{other}'")),
    };
    parsed.map_err(ParseFailure::Fatal)
}

/// The optional named flags shared by the `inspect` data-file subcommands.
struct InspectDataFileArgs {
    slot: Option<usize>,
    superblock_copy: Option<u8>,
    block: Option<u64>,
    tree: Option<String>,
    level: Option<u8>,
}

/// `parse_args_inspect` for the `op` subcommand: a single positional `u64`.
fn parse_args_inspect_op(args: &[String]) -> Result<CommandInspect, String> {
    let flags = parse_flags(args, &[], Some("op"))?;
    if flags.positionals.is_empty() {
        // parse_flags already reports `{name}: argument is required`.
        unreachable!();
    }
    let op = parse_int_flag("op", flags.positional(), parse_int_u64)?;
    Ok(CommandInspect::Op(op))
}

/// `parse_args_inspect` for the data-file subcommands (superblock, wal, replies, grid,
/// manifest, tables): one positional path plus the optional named flags.
fn parse_args_inspect_data_file(
    args: &[String],
    query: impl FnOnce(InspectDataFileArgs) -> InspectQuery,
) -> Result<CommandInspect, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--superblock-copy", kind: NamedKind::U8 },
        Named { flag: "--slot", kind: NamedKind::Usize },
        Named { flag: "--block", kind: NamedKind::U64 },
        Named { flag: "--tree", kind: NamedKind::String_ },
        // `--level` is a `u6` upstream; values outside 0..=63 must report an overflow.
        Named { flag: "--level", kind: NamedKind::U6 },
    ];
    let flags = parse_flags(args, NAMED, Some("path"))?;

    let mut slot = None;
    let mut superblock_copy = None;
    let mut block = None;
    let mut tree = None;
    let mut level = None;
    let path = flags.positional().to_owned();
    for (def, value) in flags.named {
        match def.flag {
            "--slot" => slot = Some(value.into_usize()),
            "--superblock-copy" => superblock_copy = Some(value.into_u8()),
            "--block" => block = Some(value.into_u64()),
            "--tree" => tree = Some(value.into_string()),
            "--level" => level = Some(parse_level_flag(&value)),
            _ => panic!("unexpected named flag for inspect"),
        }
    }

    Ok(CommandInspect::DataFile(CommandInspectDataFile {
        path,
        query: query(InspectDataFileArgs { slot, superblock_copy, block, tree, level }),
    }))
}

fn parse_level_flag(value: &Value) -> u8 {
    let &Value::U8(v) = value else { panic!("flag kind mismatch") };
    assert!(v <= 63, "--level: value exceeds 6-bit unsigned integer: '{v}'");
    v
}

/// `parse_args_inspect_integrity`.
fn parse_args_inspect_integrity(args: &[String]) -> Result<CommandInspect, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--memory-lsm-manifest", kind: NamedKind::Size },
        Named { flag: "--skip-client-replies", kind: NamedKind::Bool },
        Named { flag: "--log-debug", kind: NamedKind::Bool },
        Named { flag: "--skip-grid", kind: NamedKind::Bool },
        Named { flag: "--skip-wal", kind: NamedKind::Bool },
        Named { flag: "--seed", kind: NamedKind::String_ },
    ];
    let flags = parse_flags(args, NAMED, Some("path"))?;

    let mut log_debug = false;
    let mut seed = None;
    let mut memory_lsm_manifest = None;
    let mut skip_wal = false;
    let mut skip_client_replies = false;
    let mut skip_grid = false;
    let path = flags.positional().to_owned();
    for (def, value) in flags.named {
        match def.flag {
            "--log-debug" => log_debug = value.into_bool(),
            "--seed" => seed = Some(value.into_string()),
            "--memory-lsm-manifest" => memory_lsm_manifest = Some(value.into_size()),
            "--skip-wal" => skip_wal = value.into_bool(),
            "--skip-client-replies" => skip_client_replies = value.into_bool(),
            "--skip-grid" => skip_grid = value.into_bool(),
            _ => panic!("unexpected named flag for inspect integrity"),
        }
    }

    let scrub_memory_lsm_manifest = memory_lsm_manifest.unwrap_or(ByteSize {
        value: LSM_MANIFEST_MEMORY_SIZE_DEFAULT as u64,
        unit: ByteSizeUnit::Bytes,
    });
    let lsm_manifest_memory = scrub_memory_lsm_manifest.bytes();
    let lsm_manifest_memory_max = LSM_MANIFEST_MEMORY_SIZE_MAX as u64;
    let lsm_manifest_memory_min = LSM_MANIFEST_MEMORY_SIZE_MIN as u64;
    let lsm_manifest_memory_multiplier = LSM_MANIFEST_MEMORY_SIZE_MULTIPLIER as u64;
    if lsm_manifest_memory > lsm_manifest_memory_max {
        return Err(format!(
            "--memory-lsm-manifest: size {}{} exceeds maximum: {}",
            scrub_memory_lsm_manifest.value,
            scrub_memory_lsm_manifest.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_manifest_memory_max)
        ));
    }
    if lsm_manifest_memory < lsm_manifest_memory_min {
        return Err(format!(
            "--memory-lsm-manifest: size {}{} is below minimum: {}",
            scrub_memory_lsm_manifest.value,
            scrub_memory_lsm_manifest.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_manifest_memory_min)
        ));
    }
    if lsm_manifest_memory % lsm_manifest_memory_multiplier != 0 {
        return Err(format!(
            "--memory-lsm-manifest: size {}{} must be a multiple of {}",
            scrub_memory_lsm_manifest.value,
            scrub_memory_lsm_manifest.suffix(),
            tigerbeetle_core::stdx::fmt_int_size_bin_exact(lsm_manifest_memory_multiplier)
        ));
    }

    let lsm_forest_node_count =
        u32::try_from(lsm_manifest_memory / LSM_MANIFEST_NODE_SIZE as u64).unwrap_or(u32::MAX);

    Ok(CommandInspect::Integrity(CommandInspectIntegrity {
        path,
        log_debug,
        seed,
        lsm_forest_node_count,
        skip_wal,
        skip_client_replies,
        skip_grid,
    }))
}

/// `parse_args_multiversion`.
fn parse_args_multiversion(args: &[String]) -> Result<CommandMultiversion, String> {
    const NAMED: &[Named] = &[Named { flag: "--log-debug", kind: NamedKind::Bool }];
    let flags = parse_flags(args, NAMED, Some("path"))?;
    let path = flags.positional().to_owned();
    let mut log_debug = false;
    for (def, value) in flags.named {
        match def.flag {
            "--log-debug" => log_debug = value.into_bool(),
            _ => panic!("unexpected named flag for multiversion"),
        }
    }
    Ok(CommandMultiversion { path, log_debug })
}

/// `parse_args_amqp`.
#[allow(clippy::too_many_lines)] // TODO(port): split flag parsing from validation.
fn parse_args_amqp(args: &[String]) -> Result<CommandAmqp, String> {
    const NAMED: &[Named] = &[
        Named { flag: "--requests-per-second-limit", kind: NamedKind::U32 },
        Named { flag: "--publish-routing-key", kind: NamedKind::String_ },
        Named { flag: "--tigerbeetle-timeout-seconds", kind: NamedKind::U32 },
        Named { flag: "--amqp-timeout-seconds", kind: NamedKind::U32 },
        Named { flag: "--event-count-max", kind: NamedKind::U32 },
        Named { flag: "--idle-interval-ms", kind: NamedKind::U32 },
        Named { flag: "--publish-exchange", kind: NamedKind::String_ },
        Named { flag: "--timestamp-last", kind: NamedKind::U64 },
        Named { flag: "--addresses", kind: NamedKind::Cluster },
        Named { flag: "--cluster", kind: NamedKind::U128 },
        Named { flag: "--verbose", kind: NamedKind::Bool },
        Named { flag: "--password", kind: NamedKind::String_ },
        Named { flag: "--host", kind: NamedKind::Socket { default_port: 5672 } },
        Named { flag: "--vhost", kind: NamedKind::String_ },
        Named { flag: "--user", kind: NamedKind::String_ },
    ];
    let flags = parse_flags_as(args, NAMED)?;

    let mut addresses = None;
    let mut cluster = None;
    let mut host = None;
    let mut user = None;
    let mut password = None;
    let mut vhost = None;
    let mut publish_exchange = None;
    let mut publish_routing_key = None;
    let mut event_count_max = None;
    let mut idle_interval_ms = None;
    let mut requests_per_second_limit = None;
    let mut amqp_timeout_seconds = None;
    let mut tigerbeetle_timeout_seconds = None;
    let mut timestamp_last = None;
    let mut verbose = false;
    for (def, value) in flags.named {
        match def.flag {
            "--addresses" => addresses = Some(value.into_cluster()),
            "--cluster" => cluster = Some(value.into_u128()),
            "--host" => host = Some(value.into_socket()),
            "--user" => user = Some(value.into_string()),
            "--password" => password = Some(value.into_string()),
            "--vhost" => vhost = Some(value.into_string()),
            "--publish-exchange" => publish_exchange = Some(value.into_string()),
            "--publish-routing-key" => publish_routing_key = Some(value.into_string()),
            "--event-count-max" => event_count_max = Some(value.into_u32()),
            "--idle-interval-ms" => idle_interval_ms = Some(value.into_u32()),
            "--requests-per-second-limit" => requests_per_second_limit = Some(value.into_u32()),
            "--amqp-timeout-seconds" => amqp_timeout_seconds = Some(value.into_u32()),
            "--tigerbeetle-timeout-seconds" => {
                tigerbeetle_timeout_seconds = Some(value.into_u32());
            }
            "--timestamp-last" => timestamp_last = Some(value.into_u64()),
            "--verbose" => verbose = value.into_bool(),
            _ => panic!("unexpected named flag for amqp"),
        }
    }

    let addresses = addresses.ok_or_else(|| "--addresses: argument is required".to_owned())?;
    let cluster = cluster.ok_or_else(|| "--cluster: argument is required".to_owned())?;
    let host = host.ok_or_else(|| "--host: argument is required".to_owned())?;
    let user = user.ok_or_else(|| "--user: argument is required".to_owned())?;
    let password = password.ok_or_else(|| "--password: argument is required".to_owned())?;
    let vhost = vhost.ok_or_else(|| "--vhost: argument is required".to_owned())?;

    if publish_exchange.is_none() && publish_routing_key.is_none() {
        return Err("--publish-exchange and --publish-routing-key cannot both be empty.".to_owned());
    }
    if let Some(requests_per_second_limit) = requests_per_second_limit
        && requests_per_second_limit == 0
    {
        return Err("--requests-per-second-limit must not be zero.".to_owned());
    }
    if let Some(idle_interval_ms) = idle_interval_ms
        && idle_interval_ms == 0
    {
        return Err("--idle-interval-ms must not be zero.".to_owned());
    }
    if let Some(amqp_timeout_seconds) = amqp_timeout_seconds
        && amqp_timeout_seconds == 0
    {
        return Err("--amqp-timeout-seconds must not be zero.".to_owned());
    }
    if let Some(tigerbeetle_timeout_seconds) = tigerbeetle_timeout_seconds
        && tigerbeetle_timeout_seconds == 0
    {
        return Err("--tigerbeetle-timeout-seconds must not be zero.".to_owned());
    }

    Ok(CommandAmqp {
        addresses,
        cluster,
        host,
        user,
        password,
        vhost,
        publish_exchange,
        publish_routing_key,
        event_count_max,
        idle_interval_ms,
        requests_per_second_limit,
        amqp_timeout_seconds,
        tigerbeetle_timeout_seconds,
        timestamp_last,
        log_debug: verbose,
    })
}

// -----------------------------------------------------------------------------------
// Runnable commands (port of `main.zig` subcommands that map onto in-scope infrastructure)
// -----------------------------------------------------------------------------------

/// Execute a parsed command. `format` (over a real data file), `inspect superblock`, and
/// `version` run; all other subcommands parse and validate but the executor is deferred to the
/// async-loop slice.
pub fn run_command(command: &Command) -> CliResult<()> {
    match command {
        Command::Version(cmd) => {
            command_version(cmd);
            Ok(())
        }
        Command::Format(cmd) => command_format(cmd),
        Command::Inspect(cmd) => command_inspect(cmd),
        other => Err(format!(
            "the '{name}' subcommand is parsed and validated but not yet implemented in this \
             port (the async event loop is deferred)",
            name = other.name()
        )),
    }
}

/// `main.zig command_inspect`: runs the read-only `inspect superblock` data-file query; the
/// remaining inspect subcommands are parsed but deferred to the async-loop slice.
pub fn command_inspect(cmd: &CommandInspect) -> CliResult<()> {
    match cmd {
        CommandInspect::DataFile(datafile) => match &datafile.query {
            InspectQuery::Superblock => command_inspect_superblock(datafile),
            // Upstream inspects these over spread/blocked grid blocks in the async loop.
            InspectQuery::Wal { .. }
            | InspectQuery::Replies { .. }
            | InspectQuery::Grid { .. }
            | InspectQuery::Manifest { .. }
            | InspectQuery::Tables { .. } => Err(format!(
                "inspect data-file query '{}' is parsed and validated but not yet implemented in \
                 this port (the async event loop is deferred)",
                inspect_query_name(&datafile.query)
            )),
        },
        CommandInspect::Constants
        | CommandInspect::Metrics
        | CommandInspect::Op(_)
        | CommandInspect::Integrity(_) => Err(format!(
            "inspect subcommand '{}' is parsed and validated but not yet implemented in this \
             port (the async event loop is deferred)",
            inspect_subcommand_name(cmd)
        )),
    }
}

/// Human name of an [`InspectQuery`] variant (upstream switches on the Zig union).
fn inspect_query_name(query: &InspectQuery) -> &'static str {
    match query {
        InspectQuery::Superblock => "superblock",
        InspectQuery::Wal { .. } => "wal",
        InspectQuery::Replies { .. } => "replies",
        InspectQuery::Grid { .. } => "grid",
        InspectQuery::Manifest { .. } => "manifest",
        InspectQuery::Tables { .. } => "tables",
    }
}

/// Human name of a non-data-file [`CommandInspect`] variant.
fn inspect_subcommand_name(cmd: &CommandInspect) -> &'static str {
    match cmd {
        CommandInspect::Constants => "constants",
        CommandInspect::Metrics => "metrics",
        CommandInspect::Op(_) => "op",
        CommandInspect::DataFile(_) => "data-file",
        CommandInspect::Integrity(_) => "integrity",
    }
}

/// `main.zig command_version`: print the build version, and with `--verbose` the config.
pub fn command_version(cmd: &CommandVersion) {
    println!("TigerBeetle version {VERSION}");

    if cmd.verbose {
        // DEVIATION: `main.zig` prints every `config.cluster`/`config.process` field; the
        // port installs the crate version and a subset of the pinned constants instead.
        println!();
        println!(
            "cluster.checksum = 0x{:032x}",
            tigerbeetle_core::config::ConfigCluster::default().checksum()
        );
        println!("cluster.members_count = {}", REPLICAS_MAX + STANDBYS_MAX);
        println!("process.block_size = {}", tigerbeetle_core::constants::BLOCK_SIZE);
        println!("process.message_size_max = {MESSAGE_SIZE_MAX}");
        println!("process.grid_cache_size_default = {GRID_CACHE_SIZE_DEFAULT}");
        println!("process.storage_size_limit_max = {STORAGE_SIZE_LIMIT_MAX}");
    }
}

/// Format a replica data file over a [`dyn Storage`](tigerbeetle_vsr::storage::Storage):
/// the WAL then the superblock, flushed and verified (port of `vsr.replica_format.format`).
///
/// Matches `tbcross_format`'s parameters when `cluster=0, replica=0, replica_count=6,
/// view=None, release=MINIMUM`, so the emitted superblock can be cross-checked against the
/// upstream golden values.
///
/// DEVIATION: upstream `command_format` also fans the format I/O through a real grid; the
/// sans-IO port drives the same [`SuperBlock`] through the caller's storage.
pub fn drive_format(
    storage: &mut dyn tigerbeetle_vsr::storage::Storage,
    cmd: &CommandFormat,
) -> tigerbeetle_vsr::superblock::SuperBlock {
    use tigerbeetle_vsr::multiversion::Release;
    use tigerbeetle_vsr::superblock::FormatOptions;

    tigerbeetle_vsr::replica_format::format(
        storage,
        FormatOptions {
            cluster: cmd.cluster,
            release: Release::MINIMUM,
            replica: cmd.replica,
            replica_count: cmd.replica_count,
            view: None,
        },
    )
}

/// The data-file size chosen by `format` (upstream: `data_file_size_min`).
///
/// DEVIATION: `cmd_cluster` is ignored (upstream's grid is not attached during format).
pub fn cmd_storage_size(_cmd: &CommandFormat) -> u64 {
    tigerbeetle_vsr::superblock::DATA_FILE_SIZE_MIN as u64
}

/// `main.zig command_format`: create (exclusively) the data file and format it for `cmd`.
///
/// Mirroring upstream's `open_data_file(.format)` (`O_CREAT|O_EXCL`), an existing path fails
/// with `PathAlreadyExists` so a data file is never accidentally reformatted.
pub fn command_format(cmd: &CommandFormat) -> CliResult<()> {
    let size = cmd_storage_size(cmd);
    let mut storage = match tigerbeetle_vsr::storage::FileStorage::open_format(&cmd.path, size) {
        Ok(storage) => storage,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err("PathAlreadyExists".into());
        }
        Err(err) => return Err(err.to_string()),
    };
    let sb = drive_format(&mut storage, cmd);

    println!("info(data_file): checking '{}'...", cmd.path);
    println!(
        "info(format): cluster = {}\ninfo(format): replica = {}\ninfo(format): replica_count = {}",
        cmd.cluster, cmd.replica, cmd.replica_count
    );
    println!("info(format): data_file_size = {size}");
    println!("info(format): superblock.sequence = {}", sb.working().sequence);
    Ok(())
}

/// `main.zig command_inspect_superblock`: read the superblock zone and print every field of
/// the (grouped) copies.
///
/// Port of upstream `Inspector.create` + `MainType.inspect_superblock`: the zone is read
/// raw, a working superblock quorum gates the version check, and the whole zone is then
/// printed without ever decoding structurally (corrupt copies are shown, not rejected).
pub fn command_inspect_superblock(datafile: &CommandInspectDataFile) -> CliResult<()> {
    use tigerbeetle_vsr::storage::FileStorage;
    use tigerbeetle_vsr::superblock::DATA_FILE_SIZE_MIN;

    let mut storage = FileStorage::open_read_only(&datafile.path).map_err(|err| err.to_string())?;
    // Upstream `open_data_file(.inspect)` panics if the file is shorter than
    // `data_file_size_min`; per the no-panics-on-disk-input porting rule we return an error
    // with the same wording.
    if storage.size() < DATA_FILE_SIZE_MIN as u64 {
        return Err("data file inode size was truncated or corrupted".into());
    }

    let output = drive_inspect_superblock(&mut storage, &datafile.path)?;
    print!("{output}");
    Ok(())
}

/// Feed the superblock zone of `storage` through the inspect exporter, returning the printed
/// output (port of `MainType.inspect_superblock` with the `Inspector` read/verify prologue).
///
/// Mirrors upstream's sequencing: read the whole superblock zone, select the working quorum
/// (failing the inspect if none reaches the open threshold), verify the version, then print.
pub fn drive_inspect_superblock(
    storage: &mut dyn tigerbeetle_vsr::storage::Storage,
    path: &str,
) -> CliResult<String> {
    use tigerbeetle_vsr::Zone;
    use tigerbeetle_vsr::inspect::inspect_superblock;
    use tigerbeetle_vsr::storage::{Completion, ReadRequest};
    use tigerbeetle_vsr::superblock::{
        SUPERBLOCK_COPY_SIZE, SUPERBLOCK_HEADER_SIZE, SUPERBLOCK_VERSION, SUPERBLOCK_ZONE_SIZE,
    };
    use tigerbeetle_vsr::superblock_quorums::{SuperBlockQuorums, Threshold};

    storage.read_sectors(ReadRequest {
        zone: Zone::Superblock,
        offset_in_zone: 0,
        buffer: vec![0u8; SUPERBLOCK_ZONE_SIZE],
    });
    let buffer = match storage
        .next_completion()
        .unwrap_or_else(|| unreachable!("read completes synchronously"))
    {
        Completion::Read(request) => request.buffer,
        Completion::Write(_) => unreachable!("only a read was submitted"),
    };

    // Upstream `bytesAsValue`s each copy directly out of the zone buffer.
    let copies: Vec<[u8; SUPERBLOCK_HEADER_SIZE]> = (0..SUPERBLOCK_COPIES)
        .map(|i| {
            let offset = i * SUPERBLOCK_COPY_SIZE;
            buffer[offset..offset + SUPERBLOCK_HEADER_SIZE]
                .try_into()
                .unwrap_or_else(|_| unreachable!("slice length checked"))
        })
        .collect();

    // The working-quorum check + version gate replicate `Inspector.create`'s
    // `read_superblock(null)`; printing then reads the raw copies back.
    let headers: Vec<tigerbeetle_vsr::superblock::SuperBlockHeader> = copies
        .iter()
        .map(|copy| {
            tigerbeetle_vsr::inspect::decode_lenient(copy)
                .unwrap_or_else(|| unreachable!("lenient decode never fails"))
        })
        .collect();
    let mut quorums = SuperBlockQuorums::default();
    let quorum = quorums.working(&headers, Threshold::Open).map_err(|err| err.to_string())?;
    if !quorum.valid() {
        return Err("SuperBlockQuorumInvalid".into());
    }
    let version = quorum.header().version;
    if version != SUPERBLOCK_VERSION {
        return Err(format!(
            "invalid superblock version; inspector supports version={SUPERBLOCK_VERSION}, \
             version in {path}={version}"
        ));
    }

    let mut output = String::new();
    inspect_superblock(&mut output, &copies).map_err(|err| err.to_string())?;
    Ok(output)
}

// -----------------------------------------------------------------------------------
// Help text (ported verbatim from upstream `cli.zig`; placeholder values inlined)
// -----------------------------------------------------------------------------------

/// Top-level usage (upstream `CLIArgs.help`, with `default_address`/`default_port`/
/// `default_cache_grid_gb` inlined as 127.0.0.1/3000/3).
pub const HELP: &str = concat!(
    "Usage:\n",
    "\n",
    "  tigerbeetle [-h | --help]\n",
    "\n",
    "  tigerbeetle format [--cluster=<integer>] --replica=<index> --replica-count=<integer> <path>\n",
    "\n",
    "  tigerbeetle start --addresses=<addresses> [--cache-grid=<size><KiB|MiB|GiB>] <path>\n",
    "\n",
    "  tigerbeetle recover --cluster=<integer> --addresses=<addresses>\n",
    "                      --replica=<index> --replica-count=<integer> <path>\n",
    "\n",
    "  tigerbeetle version [--verbose]\n",
    "\n",
    "  tigerbeetle repl --cluster=<integer> --addresses=<addresses>\n",
    "\n",
    "Commands:\n",
    "\n",
    "  format     Create a TigerBeetle replica data file at <path>.\n",
    "             The --replica and --replica-count arguments are required.\n",
    "             Each TigerBeetle replica must have its own data file.\n",
    "\n",
    "  start      Run a TigerBeetle replica from the data file at <path>.\n",
    "\n",
    "  recover    Create a TigerBeetle replica data file at <path> for recovery.\n",
    "             Used when a replica's data file is completely lost.\n",
    "             Replicas with recovered data files must sync with the cluster before\n",
    "             they can participate in consensus.\n",
    "\n",
    "  version    Print the TigerBeetle build version and the compile-time config values.\n",
    "\n",
    "  repl       Enter the TigerBeetle client REPL.\n",
    "\n",
    "  amqp       CDC connector for AMQP targets.\n",
    "\n",
    "Options:\n",
    "\n",
    "  -h, --help\n",
    "        Print this help message and exit.\n",
    "\n",
    "  --cluster=<integer>\n",
    "        Set the cluster ID to the provided 128-bit unsigned decimal integer.\n",
    "        Defaults to generating a random cluster ID.\n",
    "\n",
    "  --replica=<index>\n",
    "        Set the zero-based index that will be used for the replica process.\n",
    "        An index greater than or equal to \"replica-count\" makes the replica a standby.\n",
    "        The value of this argument will be interpreted as an index into the --addresses array.\n",
    "\n",
    "  --replica-count=<integer>\n",
    "        Set the number of replicas participating in replication.\n",
    "\n",
    "  --addresses=<addresses>\n",
    "        The addresses of all replicas in the cluster.\n",
    "        Accepts a comma-separated list of IPv4/IPv6 addresses with port numbers.\n",
    "        The order is significant and must match across all replicas and clients.\n",
    "        Either the address or port number (but not both) may be omitted,\n",
    "        in which case a default of 127.0.0.1 or 3000 will be used.\n",
    "        \"addresses[i]\" corresponds to replica \"i\".\n",
    "\n",
    "  --cache-grid=<size><KiB|MiB|GiB>\n",
    "        Set the grid cache size. The grid cache acts like a page cache for TigerBeetle,\n",
    "        and should be set as large as possible.\n",
    "        On a machine running only TigerBeetle, this is somewhere around\n",
    "        (Total RAM) - 3GiB (TigerBeetle) - 1GiB (System), eg 12GiB for a 16GiB machine.\n",
    "        Defaults to 3GiB.\n",
    "\n",
    "  --verbose\n",
    "        Print compile-time configuration along with the build version.\n",
    "\n",
    "  --development\n",
    "        Allow the replica to format/start/recover even when Direct IO is unavailable.\n",
    "        Additionally, use smaller cache sizes and batch size by default.\n",
    "\n",
    "        Since this shrinks the batch size, note that:\n",
    "        * All replicas should use the same batch size. That is, if any replica in the cluster has\n",
    "          \"--development\", then all replicas should have \"--development\".\n",
    "        * It is always possible to increase the batch size by restarting without \"--development\".\n",
    "        * Shrinking the batch size of an existing cluster is possible, but not recommended.\n",
    "\n",
    "        For safety, production replicas should always enforce Direct IO -- this flag should only be\n",
    "        used for testing and development. It should not be used for production or benchmarks.\n",
    "\n",
    "Examples:\n",
    "\n",
    "  tigerbeetle format --cluster=0 --replica=0 --replica-count=3 0_0.tigerbeetle\n",
    "  tigerbeetle format --cluster=0 --replica=1 --replica-count=3 0_1.tigerbeetle\n",
    "  tigerbeetle format --cluster=0 --replica=2 --replica-count=3 0_2.tigerbeetle\n",
    "\n",
    "  tigerbeetle start --addresses=127.0.0.1:3000,127.0.0.1:3001,127.0.0.1:3002 0_0.tigerbeetle\n",
    "  tigerbeetle start --addresses=3000,3001,3002 0_1.tigerbeetle\n",
    "  tigerbeetle start --addresses=3000,3001,3002 0_2.tigerbeetle\n",
    "\n",
    "  tigerbeetle start --addresses=192.168.0.1,192.168.0.2,192.168.0.3 0_0.tigerbeetle\n",
    "\n",
    "  tigerbeetle start --addresses='[::1]:3000,[::1]:3001,[::1]:3002' 0_0.tigerbeetle\n",
    "\n",
    "  tigerbeetle recover --cluster=0 --addresses=3003,3001,3002 \\\n",
    "                      --replica=1 --replica-count=3 0_1.tigerbeetle\n",
    "\n",
    "  tigerbeetle version --verbose\n",
    "\n",
    "  tigerbeetle repl --addresses=3000,3001,3002 --cluster=0\n",
    "\n",
    "  tigerbeetle amqp --addresses=3000,3001,3002 --cluster=0 \\\n",
    "      --host=127.0.0.1 --vhost=/ --user=guest --password=guest \\\n",
    "      --publish-exchange=my_exchange_name\n",
);

/// `inspect` usage (upstream `CLIArgs.Inspect.help`).
pub const INSPECT_HELP: &str = concat!(
    "Usage:\n",
    "\n",
    "  tigerbeetle inspect [-h | --help]\n",
    "\n",
    "  tigerbeetle inspect constants\n",
    "\n",
    "  tigerbeetle inspect metrics\n",
    "\n",
    "  tigerbeetle inspect op <op>\n",
    "\n",
    "  tigerbeetle inspect superblock <path>\n",
    "\n",
    "  tigerbeetle inspect wal [--slot=<slot>] <path>\n",
    "\n",
    "  tigerbeetle inspect replies [--slot=<slot>] <path>\n",
    "\n",
    "  tigerbeetle inspect grid [--block=<address>] <path>\n",
    "\n",
    "  tigerbeetle inspect manifest <path>\n",
    "\n",
    "  tigerbeetle inspect tables --tree=<name|id> [--level=<integer>] <path>\n",
    "\n",
    "  tigerbeetle inspect integrity [--log-debug] [--seed=<seed>]\n",
    "                                [--memory-lsm-manifest=<size>]\n",
    "                                [--skip-wal] [--skip-client-replies] [--skip-grid]\n",
    "                                <path>\n",
    "\n",
    "Options:\n",
    "\n",
    "  When `--superblock-copy` is set, use the trailer referenced by that superblock copy.\n",
    "  Otherwise, the current quorum will be used by default.\n",
    "\n",
    "  -h, --help\n",
    "        Print this help message and exit.\n",
);

// -----------------------------------------------------------------------------------
// Tests (upstream `cli.zig` parse-vector coverage + golden `tbcross_format` pin)
// -----------------------------------------------------------------------------------

// Upstream test vectors use `unwrap()` freely, and the test-min config keeps the derived
// block/node counts well inside `u32`.
#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use tigerbeetle_core::constants::BLOCK_SIZE;
    use tigerbeetle_vsr::storage::MemoryStorage;
    use tigerbeetle_vsr::superblock::DATA_FILE_SIZE_MIN;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| String::from(*s)).collect()
    }

    #[test]
    fn format_full() {
        let cmd = parse_args(&args(&[
            "format",
            "--cluster=0",
            "--replica=0",
            "--replica-count=6",
            "0_0.tigerbeetle",
        ]))
        .unwrap();
        assert_eq!(
            cmd,
            Command::Format(CommandFormat {
                cluster: 0,
                replica: 0,
                replica_count: 6,
                development: false,
                path: "0_0.tigerbeetle".into(),
                log_debug: false,
            })
        );
    }

    #[test]
    fn format_standby() {
        let cmd =
            parse_args(&args(&["format", "--standby=7", "--replica-count=6", "0_6.tigerbeetle"]))
                .unwrap();
        let Command::Format(cmd) = cmd else { unreachable!() };
        assert_eq!(cmd.replica, 7);
        assert_eq!(cmd.replica_count, 6);
    }

    #[test]
    fn format_errors() {
        // Missing subcommand.
        assert_eq!(
            parse_args(&args(&[])),
            Err(ParseFailure::Fatal(
                "subcommand required, expected format, recover, start, version, repl, \
                 benchmark, inspect, multiversion, amqp"
                    .into()
            ))
        );
        // Unknown subcommand.
        assert_eq!(
            parse_args(&args(&["formattt", "0_0.tigerbeetle"])),
            Err(ParseFailure::Fatal("unknown subcommand: 'formattt'".into()))
        );
        // Missing required flags.
        assert_eq!(
            parse_args(&args(&["format", "--replica-count=6", "path"])),
            Err(ParseFailure::Fatal("--replica: argument is required".into()))
        );
        assert_eq!(
            parse_args(&args(&["format", "--replica=0", "path"])),
            Err(ParseFailure::Fatal("--replica-count: argument is required".into()))
        );
        assert_eq!(
            parse_args(&args(&["format", "--replica=0", "--replica-count=6"])),
            Err(ParseFailure::Fatal("path: argument is required".into()))
        );
        // replica-count bounds.
        assert_eq!(
            parse_args(&args(&["format", "--replica=0", "--replica-count=0", "path"])),
            Err(ParseFailure::Fatal("--replica-count: value needs to be greater than zero".into()))
        );
        assert_eq!(
            parse_args(&args(&["format", "--replica=0", "--replica-count=7", "path"])),
            Err(ParseFailure::Fatal(
                "--replica-count: value is too large (7), at most 6 is allowed".into()
            ))
        );
        // replica / standby conflicts.
        assert_eq!(
            parse_args(&args(&[
                "format",
                "--replica=0",
                "--standby=7",
                "--replica-count=6",
                "path"
            ])),
            Err(ParseFailure::Fatal("--standby: conflicts with '--replica'".into()))
        );
        assert_eq!(
            parse_args(&args(&["format", "--replica=6", "--replica-count=6", "path"])),
            Err(ParseFailure::Fatal(
                "--replica: value is too large (6), at most 5 is allowed".into()
            ))
        );
        // Duplicate argument.
        assert_eq!(
            parse_args(&args(&[
                "format",
                "--replica=0",
                "--replica=1",
                "--replica-count=6",
                "path"
            ])),
            Err(ParseFailure::Fatal("--replica: duplicate argument".into()))
        );
        // Int overflow / invalid digit / leading zero.
        assert_eq!(
            parse_args(&args(&["format", "--replica=256", "--replica-count=6", "path"])),
            Err(ParseFailure::Fatal(
                "--replica: value exceeds 8-bit unsigned integer: '256'".into()
            ))
        );
        assert_eq!(
            parse_args(&args(&["format", "--replica=0a", "--replica-count=6", "path"])),
            Err(ParseFailure::Fatal("--replica: leading zero disallowed: '0a'".into()))
        );
        assert_eq!(
            parse_args(&args(&["format", "--replica=00", "--replica-count=6", "path"])),
            Err(ParseFailure::Fatal("--replica: leading zero disallowed: '00'".into()))
        );
    }
    #[test]
    fn flag_separator_errors() {
        // A bare non-boolean flag without `=` reports a missing separator.
        assert_eq!(
            parse_args(&args(&["format", "--replica-count"])),
            Err(ParseFailure::Fatal("--replica-count: expected value separator '='".into()))
        );
        // An empty `--flag=` value reports a missing argument.
        assert_eq!(
            parse_args(&args(&["format", "--replica-count=", "path"])),
            Err(ParseFailure::Fatal("--replica-count: argument requires a value".into()))
        );
    }

    #[test]
    fn format_cluster_zero_warns() {
        // cluster=0 is accepted (with a warning) and preserved.
        let cmd = parse_args(&args(&[
            "format",
            "--cluster=0",
            "--replica=0",
            "--replica-count=3",
            "0_0.tigerbeetle",
        ]))
        .unwrap();
        let Command::Format(cmd) = cmd else { unreachable!() };
        assert_eq!(cmd.cluster, 0);
    }

    #[test]
    fn recover_root() {
        let cmd = parse_args(&args(&[
            "recover",
            "--cluster=0",
            "--addresses=3003,3001,3002",
            "--replica=1",
            "--replica-count=3",
            "0_1.tigerbeetle",
        ]))
        .unwrap();
        let Command::Recover(cmd) = cmd else { unreachable!() };
        assert_eq!(cmd.cluster, 0);
        assert_eq!(cmd.replica, 1);
        assert_eq!(cmd.replica_count, 3);
        assert_eq!(cmd.addresses.members_count(), 3);
    }

    #[test]
    fn recover_errors() {
        assert_eq!(
            parse_args(&args(&[
                "recover",
                "--cluster=0",
                "--addresses=3003",
                "--replica=1",
                "--replica-count=1",
                "path",
            ])),
            Err(ParseFailure::Fatal(
                "--replica: value is too large (1), at most 0 is allowed".into()
            ))
        );
        assert_eq!(
            parse_args(&args(&[
                "recover",
                "--cluster=0",
                "--addresses=3003",
                "--replica=0",
                "--replica-count=2",
                "path",
            ])),
            Err(ParseFailure::Fatal(
                "--replica-count: 1- or 2- replica clusters don't support 'recover'".into()
            ))
        );
        assert_eq!(
            parse_args(&args(&[
                "recover",
                "--addresses=3003",
                "--replica=0",
                "--replica-count=3",
                "path",
            ])),
            Err(ParseFailure::Fatal("--cluster: argument is required".into()))
        );
    }

    #[test]
    fn start_development_defaults() {
        // The dev `limit_request` default (32KiB) exceeds `message_size_max` under test_min
        // (4KiB), exactly as upstream validates against its production constants. Pin the
        // defaults directly, then exercise the CLI with an explicit, test_min-valid request
        // size so the rest of the dev-default behavior can be observed.
        let defaults = start_defaults_development();
        assert_eq!(defaults.limit_pipeline_requests, 0);
        assert_eq!(defaults.limit_request, 32 * tigerbeetle_core::stdx::KIB as u32);
        assert_eq!(defaults.cache_accounts, 0);

        let cmd = parse_args(&args(&[
            "start",
            "--addresses=3000,3001,3002",
            "--experimental",
            "--development",
            "--limit-request=4KiB",
            "0_0.tigerbeetle",
        ]))
        .unwrap();
        let Command::Start(cmd) = cmd else { unreachable!() };
        assert_eq!(cmd.addresses.members_count(), 3);
        assert!(cmd.development);
        // `data_file_size_min` is the minimum storage size, and it is satisfiable.
        assert!(cmd.storage_size_limit >= DATA_FILE_SIZE_MIN as u64);
        assert_eq!(cmd.pipeline_requests_limit, 0);
        assert_eq!(cmd.request_size_limit, 4096);
        assert_eq!(cmd.cache_accounts, 0);
        assert!(!cmd.log_debug);
    }

    #[test]
    fn start_experimental_gate() {
        assert_eq!(
            parse_args(&args(&[
                "start",
                "--addresses=3000",
                "--cache-accounts=1GiB",
                "0_0.tigerbeetle",
            ])),
            Err(ParseFailure::Fatal(
                "--cache-accounts is marked experimental, add `--experimental` to continue.".into()
            ))
        );
        // With --experimental the same flag is accepted.
        let _ = parse_args(&args(&[
            "start",
            "--addresses=3000",
            "--experimental",
            "--cache-accounts=4MiB",
            "0_0.tigerbeetle",
        ]))
        .unwrap();
    }

    #[test]
    fn start_memory_split() {
        // Value caches must be rounded to the SetAssociativeCache multiple; the grid cache
        // to `Grid.Cache`'s multiple.
        let cmd = parse_args(&args(&[
            "start",
            "--addresses=3000",
            "--experimental",
            "--memory=64MiB",
            "0_0.tigerbeetle",
        ]))
        .unwrap();
        let Command::Start(cmd) = cmd else { unreachable!() };
        let grid_bytes = memory_split_bytes(64 * tigerbeetle_core::stdx::MIB as u64, 64);
        let multiple = cache_value_count_max_multiple() as u32;
        // The grid cache rounds the block count down to a multiple of the cache-value count.
        assert_eq!(
            cmd.cache_grid_blocks,
            (grid_bytes / BLOCK_SIZE as u64) as u32 / multiple * multiple
        );
        assert_eq!(cmd.cache_grid_blocks % multiple, 0);
        assert!(cmd.cache_grid_blocks as u64 * BLOCK_SIZE as u64 <= grid_bytes);
    }

    #[test]
    fn start_errors() {
        assert_eq!(
            parse_args(&args(&["start", "0_0.tigerbeetle"])),
            Err(ParseFailure::Fatal("--addresses: argument is required".into()))
        );
        // Missing positional.
        assert_eq!(
            parse_args(&args(&["start", "--addresses=3000"])),
            Err(ParseFailure::Fatal("path: argument is required".into()))
        );
        // trailing option after positional.
        assert_eq!(
            parse_args(&args(&["start", "--addresses=3000", "path", "--development"])),
            Err(ParseFailure::Fatal("unexpected trailing option: '--development'".into()))
        );
        // empty positional.
        assert_eq!(
            parse_args(&args(&["start", "--addresses=3000", ""])),
            Err(ParseFailure::Fatal("path: empty argument".into()))
        );
    }

    #[test]
    fn version_verbose() {
        let cmd = parse_args(&args(&["version", "--verbose"])).unwrap();
        let Command::Version(cmd) = cmd else { unreachable!() };
        assert!(cmd.verbose);
    }

    #[test]
    fn repl_requires_addresses_and_cluster() {
        assert_eq!(
            parse_args(&args(&["repl"])),
            Err(ParseFailure::Fatal("--addresses: argument is required".into()))
        );
        let cmd = parse_args(&args(&["repl", "--addresses=3000", "--cluster=0"])).unwrap();
        let Command::Repl(cmd) = cmd else { unreachable!() };
        assert_eq!(cmd.cluster, 0);
        assert_eq!(cmd.statements, "");
    }

    #[test]
    fn benchmark_enums_and_vectors() {
        // Longest-prefix: `--transfer-batch-count` vs `--transfer-count`.
        let cmd = parse_args(&args(&[
            "benchmark",
            "--account-distribution=zipfian",
            "--id-order=sequential",
            "--transfer-count=77",
            "--transfer-batch-count=22",
        ]))
        .unwrap();
        let Command::Benchmark(cmd) = cmd else { unreachable!() };
        assert_eq!(cmd.account_distribution, BenchmarkDistribution::Zipfian);
        assert_eq!(cmd.id_order, BenchmarkIdOrder::Sequential);
        assert_eq!(cmd.transfer_count, 77);
        assert_eq!(cmd.transfer_batch_count, 22);

        assert_eq!(
            parse_args(&args(&["benchmark", "--account-distribution=bogus"])),
            Err(ParseFailure::Fatal(
                "--account-distribution: expected one of 'zipfian', 'latest', or 'uniform', \
                 but found 'bogus'"
                    .into()
            ))
        );
        assert_eq!(
            parse_args(&args(&["benchmark", "--id-order=bogus"])),
            Err(ParseFailure::Fatal(
                "--id-order: expected one of 'tbid', 'sequential', 'random', or 'reversed', \
                 but found 'bogus'"
                    .into()
            ))
        );
    }

    #[test]
    fn duration_flag_value() {
        // Ported from upstream `Duration.parse_flag_value` fuzz vectors.
        assert_eq!(parse_duration_flag("--transfer-batch-delay", "1h").unwrap(), NS_PER_HOUR);
        assert_eq!(
            parse_duration_flag("--transfer-batch-delay", "1h2m").unwrap(),
            NS_PER_HOUR + 2 * NS_PER_MIN
        );
        assert_eq!(
            parse_duration_flag("--transfer-batch-delay", "1ms2us3ns").unwrap(),
            NS_PER_MS + 2 * NS_PER_US + 3
        );
        for (input, expected) in [
            ("h", "missing value"),
            ("1", "missing unit; must be one of: d/h/m/s/ms/us/ns"),
            ("1H", "unknown unit; must be one of: d/h/m/s/ms/us/ns"),
            ("1h2x", "unknown unit; must be one of: d/h/m/s/ms/us/ns"),
            ("1_0h", "unknown unit; must be one of: d/h/m/s/ms/us/ns"),
            ("1h 2m", "missing value"),
            ("18446744073709551616ns", "integer overflow"),
            ("1844674407370955161s", "duration too large"),
            ("0024h", "leading zero disallowed"),
        ] {
            assert_eq!(
                parse_duration_flag("--transfer-batch-delay", input),
                Err(format!("--transfer-batch-delay: {expected}: '{input}'")),
                "{input}"
            );
        }
    }

    #[test]
    fn ratio_flag_value() {
        assert_eq!(parse_ratio_flag("--commit-stall-probability", "0").unwrap(), Ratio::zero());
        assert_eq!(
            parse_ratio_flag("--commit-stall-probability", "3/4").unwrap(),
            Ratio { numerator: 3, denominator: 4 }
        );
        assert_eq!(
            parse_ratio_flag("--commit-stall-probability", "1/0"),
            Err("--commit-stall-probability: denominator is zero: '1/0'".into())
        );
        assert_eq!(
            parse_ratio_flag("--commit-stall-probability", "2/1"),
            Err("--commit-stall-probability: ratio greater than 1: '2/1'".into())
        );
        assert_eq!(
            parse_ratio_flag("--commit-stall-probability", "1"),
            Err("--commit-stall-probability: expected 'a/b' ratio, but found: '1'".into())
        );
    }

    #[test]
    fn cluster_address_flag_value() {
        // Ported from upstream `ClusterAddress.parse_flag_value` positive/negative vectors.
        assert_eq!(
            parse_cluster_flag("--addresses", "127.0.0.1:3000,127.0.0.1:3001")
                .unwrap()
                .members_count(),
            2
        );
        assert_eq!(parse_cluster_flag("--addresses", "3000,3001,3002").unwrap().members_count(), 3);
        assert!(parse_cluster_flag("--addresses", "0").unwrap().zero);
        assert_eq!(parse_cluster_flag("--addresses", "[::1]:3000").unwrap().members_count(), 1);

        assert_eq!(
            parse_cluster_flag("--addresses", "1.2.3.4:567,1.2.3.4:"),
            Err("--addresses: invalid port: '1.2.3.4:567,1.2.3.4:'".into())
        );
        // A bare "3000:3000" parses the port but fails on the "3000" host.
        assert_eq!(
            parse_cluster_flag("--addresses", "3000:3000"),
            Err("--addresses: invalid IPv4 or IPv6 address: '3000:3000'".into())
        );
        // Bare ports (including "1") are valid against the default address.
        assert_eq!(parse_cluster_flag("--addresses", "3000,3001,1").unwrap().members_count(), 3);
    }

    #[test]
    fn inspect_commands() {
        assert_eq!(
            parse_args(&args(&["inspect", "constants"])).unwrap(),
            Command::Inspect(CommandInspect::Constants)
        );
        assert_eq!(
            parse_args(&args(&["inspect", "op", "19"])).unwrap(),
            Command::Inspect(CommandInspect::Op(19))
        );
        assert_eq!(
            parse_args(&args(&["inspect", "superblock", "0_0.tigerbeetle"])).unwrap(),
            Command::Inspect(CommandInspect::DataFile(CommandInspectDataFile {
                path: "0_0.tigerbeetle".into(),
                query: InspectQuery::Superblock,
            }))
        );
        assert_eq!(
            parse_args(&args(&["inspect", "wal", "--slot=4", "0_0.tigerbeetle"])).unwrap(),
            Command::Inspect(CommandInspect::DataFile(CommandInspectDataFile {
                path: "0_0.tigerbeetle".into(),
                query: InspectQuery::Wal { slot: Some(4) },
            }))
        );
        assert_eq!(
            parse_args(&args(&[
                "inspect",
                "tables",
                "--tree=transfers",
                "--level=2",
                "0_0.tigerbeetle"
            ]))
            .unwrap(),
            Command::Inspect(CommandInspect::DataFile(CommandInspectDataFile {
                path: "0_0.tigerbeetle".into(),
                query: InspectQuery::Tables {
                    superblock_copy: None,
                    tree: "transfers".into(),
                    level: Some(2),
                },
            }))
        );
        assert_eq!(
            parse_args(&args(&["inspect", "integrity", "0_0.tigerbeetle"])).unwrap(),
            Command::Inspect(CommandInspect::Integrity(CommandInspectIntegrity {
                path: "0_0.tigerbeetle".into(),
                log_debug: false,
                seed: None,
                lsm_forest_node_count: (LSM_MANIFEST_MEMORY_SIZE_DEFAULT / LSM_MANIFEST_NODE_SIZE)
                    as u32,
                skip_wal: false,
                skip_client_replies: false,
                skip_grid: false,
            }))
        );
    }

    #[test]
    fn amqp_validation() {
        assert_eq!(
            parse_args(&args(&[
                "amqp",
                "--addresses=3000",
                "--cluster=0",
                "--host=127.0.0.1",
                "--user=u",
                "--password=p",
                "--vhost=/",
            ])),
            Err(ParseFailure::Fatal(
                "--publish-exchange and --publish-routing-key cannot both be empty.".into()
            ))
        );
        let cmd = parse_args(&args(&[
            "amqp",
            "--addresses=3000",
            "--cluster=0",
            "--host=127.0.0.1",
            "--user=u",
            "--password=p",
            "--vhost=/",
            "--publish-exchange=x",
        ]))
        .unwrap();
        let Command::Amqp(cmd) = cmd else { unreachable!() };
        assert!(cmd.publish_exchange.is_some());
        // Default AMQP port 5672 fills in for a bare host.
        assert_eq!(cmd.host.port, 5672);
    }

    #[test]
    fn bytesize_flag_value() {
        let size = parse_size_flag("--cache-grid", "4MiB").unwrap();
        assert_eq!(size.bytes(), 4 * 1024 * 1024);
        assert_eq!(size.unit, ByteSizeUnit::Mib);
        assert_eq!(
            parse_size_flag("--cache-grid", "4xb"),
            Err("--cache-grid: invalid unit in size, needed KiB, MiB, GiB or TiB: '4xb'".into())
        );
        // A bare magnitude is a valid byte size.
        assert_eq!(parse_size_flag("--cache-grid", "4").unwrap().bytes(), 4);
        assert_eq!(
            parse_size_flag("--cache-grid", "04MiB"),
            Err("--cache-grid: leading zero disallowed: '04MiB'".into())
        );
        assert_eq!(
            parse_size_flag("--cache-grid", "0000000000000000000001KiB"),
            Err("--cache-grid: leading zero disallowed: '0000000000000000000001KiB'".into())
        );
    }

    /// Pins `drive_format`'s output to the `tbcross_format` golden values (cluster=0,
    /// `replica=0`, `replica_count=6`, release minimum, view=None), replicating the vsr
    /// `superblock_tests::format_matches_upstream_zig_golden` assertions (including the
    /// 0xAA-poisoned padding) so the CLI's format path diverges from upstream with golden
    /// tests still green.
    #[test]
    fn format_matches_upstream_zig_golden() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        storage.poison_image();

        let cmd = CommandFormat {
            cluster: 0,
            replica: 0,
            replica_count: 6,
            development: false,
            path: "0_0.tigerbeetle".into(),
            log_debug: false,
        };
        let sb = drive_format(&mut storage, &cmd);

        assert_eq!(sb.working().sequence, 1);
        assert_eq!(sb.working().cluster, 0);
        assert_eq!(sb.working().vsr_state.replica_id, tigerbeetle_vsr::root_members(0)[0]);
        assert_eq!(sb.working().checksum, 0xe741_49b8_992b_5101_1bc1_5348_f1b5_dd77);
        assert_eq!(
            tigerbeetle_core::checksum::checksum(&sb.working().vsr_state.to_wire()),
            0x3037_99b6_add2_362c_af2c_353a_f793_f683
        );
        assert_eq!(
            sb.working().vsr_state.checkpoint.header.checksum,
            0x5146_b8d0_e1f6_9ca2_e686_7c42_bb82_63b7
        );
    }

    #[test]
    fn inspect_superblock_matches_format_golden() {
        let mut storage = MemoryStorage::new(DATA_FILE_SIZE_MIN as u64);
        storage.poison_image();
        let cmd = CommandFormat {
            cluster: 0,
            replica: 0,
            replica_count: 6,
            development: false,
            path: "0_0.tigerbeetle".into(),
            log_debug: false,
        };
        drive_format(&mut storage, &cmd);

        let output = drive_inspect_superblock(&mut storage, "0_0.tigerbeetle").unwrap();
        let lines: Vec<&str> = output.lines().collect();

        // All four copies are fresh-identical, so every non-`copy` field is a single `||||` group.
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("|||| checksum=0xe74149b8992b51011bc15348f1b5dd77"))
        );
        assert!(lines.iter().any(|line| line.starts_with("|||| version=2")));
        assert!(lines.iter().any(|line| line.starts_with("|||| release_format=0.0.1")));
        assert!(lines.iter().any(|line| line.starts_with("|||| sequence=1")));
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("|||| cluster=0x00000000000000000000000000000000"))
        );
        assert!(lines.iter().any(|line| line.ends_with(" view_headers_count=1")));

        // Only the `copy` field disagrees across copies → four single-member groups.
        assert!(lines.contains(&"|___ copy=0"));
        assert!(lines.contains(&"_|__ copy=1"));
        assert!(lines.contains(&"__|_ copy=2"));
        assert!(lines.contains(&"___| copy=3"));

        // No group splits anywhere else, and the copy field uses its own mask.
        assert_eq!(
            lines.iter().filter(|line| !line.starts_with('|') && !line.starts_with('_')).count(),
            0
        );

        // `vsr_state.members` = 12 members (test-min config), each printed as hex.
        let member_lines: Vec<&&str> =
            lines.iter().filter(|line| line.contains("vsr_state.members[")).collect();
        assert_eq!(member_lines.len(), 12);
        // replica_id and members[0] are root_members(0)[0] = 0x459d... (upstream golden,
        // `tbcross_main.zig` root_members(0)).
        assert!(lines.iter().any(|line| {
            line.starts_with("|||| vsr_state.replica_id=0x459d6840872cd4709b6b08975dc2a6bb")
        }));
        assert!(lines.iter().any(|line| {
            line.starts_with("|||| vsr_state.members[0]=0x459d6840872cd4709b6b08975dc2a6bb")
        }));

        // `view_headers_all` = 7 view headers, printed field-by-field as `Prepare`.
        let view_header_lines: Vec<&&str> =
            lines.iter().filter(|line| line.contains("view_headers_all[")).collect();
        assert_eq!(view_header_lines.len(), 7);

        // The checkpoint header is the root `Prepare` (command=prepare, operation=root) with
        // checksum 0x5146... (upstream golden, `tbcross_format.zig`).
        assert!(lines.iter().any(|line| line.starts_with(
            "|||| vsr_state.checkpoint.header=Prepare{ .checksum=5146b8d0e1f69ca2e6867c42bb8263b7"
        )));
        assert!(lines.iter().any(|line| line.contains("vsr_state.checkpoint.header=Prepare{")
            && line.ends_with(".operation=vsr.Operation.root }")));
    }

    /// `inspect superblock` over a real data file produced by `command_format`: the printed
    /// checksum and headlines match the golden values, and `run_command` routes the parsed
    /// `inspect superblock` subcommand here.
    #[test]
    fn inspect_superblock_real_data_file() {
        let path = std::env::temp_dir()
            .join(format!("tigerbeetle-rs-cli-inspect-{}.tigerbeetle", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let cmd = CommandFormat {
            cluster: 0,
            replica: 0,
            replica_count: 6,
            development: false,
            path: path.to_str().unwrap().to_owned(),
            log_debug: false,
        };
        command_format(&cmd).unwrap();

        let mut storage = tigerbeetle_vsr::storage::FileStorage::open_read_only(&cmd.path).unwrap();
        let output = drive_inspect_superblock(&mut storage, &cmd.path).unwrap();
        assert!(output.contains("|||| checksum=0xe74149b8992b51011bc15348f1b5dd77"));
        assert!(output.contains("|||| version=2"));
        assert!(output.contains("_|__ copy=1"));

        // The end-to-end `run_command` path also succeeds and prints.
        let parsed = parse_args(&args(&["inspect", "superblock", &cmd.path])).unwrap();
        run_command(&parsed).unwrap();
        // Deferred inspect subcommands still report the deferred executor.
        let wal = parse_args(&args(&["inspect", "wal", &cmd.path])).unwrap();
        let err = run_command(&wal).unwrap_err();
        assert!(err.contains("not yet implemented"), "unexpected: {err}");

        let _ = std::fs::remove_file(&cmd.path);
    }

    /// Protocol deviations that abort `inspect superblock` (upstream `vsr.fatal`/errors) keep
    /// the CLI quiet with an explicit error.
    #[test]
    fn inspect_superblock_errors() {
        use tigerbeetle_core::checksum::checksum;
        use tigerbeetle_vsr::Zone;
        use tigerbeetle_vsr::storage::{Completion, ReadRequest, WriteRequest};
        use tigerbeetle_vsr::superblock::{SUPERBLOCK_COPY_SIZE, SUPERBLOCK_HEADER_SIZE};

        // Missing file → filesystem error.
        let missing = std::env::temp_dir()
            .join(format!("tigerbeetle-rs-cli-inspect-missing-{}.tigerbeetle", std::process::id()));
        let _ = std::fs::remove_file(&missing);
        assert!(
            command_inspect_superblock(&CommandInspectDataFile {
                path: missing.to_str().unwrap().to_owned(),
                query: InspectQuery::Superblock,
            })
            .is_err()
        );

        // Truncated file → upstream's panic wording, surfaced as an error.
        let truncated = std::env::temp_dir().join(format!(
            "tigerbeetle-rs-cli-inspect-truncated-{}.tigerbeetle",
            std::process::id()
        ));
        std::fs::write(&truncated, [0u8; SECTOR_SIZE]).unwrap();
        let err = command_inspect_superblock(&CommandInspectDataFile {
            path: truncated.to_str().unwrap().to_owned(),
            query: InspectQuery::Superblock,
        })
        .unwrap_err();
        assert_eq!(err, "data file inode size was truncated or corrupted");
        let _ = std::fs::remove_file(&truncated);

        // Bumped version (re-verified checksum) → upstream's fatal wording, no output.
        let path = std::env::temp_dir()
            .join(format!("tigerbeetle-rs-cli-inspect-version-{}.tigerbeetle", std::process::id()));
        let _ = std::fs::remove_file(&path);
        command_format(&CommandFormat {
            cluster: 0,
            replica: 0,
            replica_count: 6,
            development: false,
            path: path.to_str().unwrap().to_owned(),
            log_debug: false,
        })
        .unwrap();

        let mut storage = tigerbeetle_vsr::storage::FileStorage::open(&path, 0).unwrap();
        storage.read_sectors(ReadRequest {
            zone: Zone::Superblock,
            offset_in_zone: 0,
            buffer: vec![0u8; tigerbeetle_vsr::superblock::SUPERBLOCK_ZONE_SIZE],
        });
        let mut zone = match storage
            .next_completion()
            .unwrap_or_else(|| unreachable!("read completes synchronously"))
        {
            Completion::Read(request) => request.buffer,
            Completion::Write(_) => unreachable!("only a read was submitted"),
        };
        for copy in 0..SUPERBLOCK_COPIES {
            let base = copy * SUPERBLOCK_COPY_SIZE;
            zone[base + 34..base + 36].copy_from_slice(&3u16.to_le_bytes());
            let sum = checksum(&zone[base + 34..base + SUPERBLOCK_HEADER_SIZE]);
            zone[base..base + 16].copy_from_slice(&sum.to_le_bytes());
        }
        storage.write_sectors(WriteRequest {
            zone: Zone::Superblock,
            offset_in_zone: 0,
            buffer: zone,
        });
        let _ = storage
            .next_completion()
            .unwrap_or_else(|| unreachable!("write completes synchronously"));

        let mut storage = tigerbeetle_vsr::storage::FileStorage::open_read_only(&path).unwrap();
        let err = drive_inspect_superblock(&mut storage, path.to_str().unwrap()).unwrap_err();
        assert!(
            err.contains("invalid superblock version; inspector supports version=2, version in")
        );

        let _ = std::fs::remove_file(&path);
    }

    /// `command_format` creates a real data file whose superblock reopens to the golden
    /// values (cluster=0, replica=0, `replica_count=6`).
    #[test]
    fn format_creates_a_real_data_file() {
        let path = std::env::temp_dir()
            .join(format!("tigerbeetle-rs-cli-format-{}.tigerbeetle", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let cmd = CommandFormat {
            cluster: 0,
            replica: 0,
            replica_count: 6,
            development: false,
            path: path.to_str().unwrap().to_owned(),
            log_debug: false,
        };

        command_format(&cmd).unwrap();
        assert_eq!(std::fs::metadata(&cmd.path).unwrap().len(), DATA_FILE_SIZE_MIN as u64);

        let mut storage = tigerbeetle_vsr::storage::FileStorage::open(&cmd.path, 0).unwrap();
        let mut sb = tigerbeetle_vsr::superblock::SuperBlock::new(DATA_FILE_SIZE_MIN as u64);
        sb.open(&mut storage);
        sb.poll(&mut storage);
        assert!(sb.opened());
        assert_eq!(sb.working().sequence, 1);
        assert_eq!(sb.working().cluster, 0);
        assert_eq!(sb.working().vsr_state.replica_id, tigerbeetle_vsr::root_members(0)[0]);
        assert_eq!(sb.working().checksum, 0xe741_49b8_992b_5101_1bc1_5348_f1b5_dd77);

        let _ = std::fs::remove_file(&cmd.path);
    }

    /// Upstream `open_data_file(.format)` uses `O_CREAT|O_EXCL`: a second format of the same
    /// path must fail with `error.PathAlreadyExists` before touching the file.
    #[test]
    fn format_refuses_to_reformat_an_existing_file() {
        let path = std::env::temp_dir()
            .join(format!("tigerbeetle-rs-cli-reformat-{}.tigerbeetle", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let cmd = CommandFormat {
            cluster: 0,
            replica: 0,
            replica_count: 1,
            development: false,
            path: path.to_str().unwrap().to_owned(),
            log_debug: false,
        };

        command_format(&cmd).unwrap();
        assert_eq!(command_format(&cmd).unwrap_err(), "PathAlreadyExists");

        let _ = std::fs::remove_file(&cmd.path);
    }

    #[test]
    fn help_texts_present() {
        assert!(HELP.contains("tigerbeetle format"));
        assert!(INSPECT_HELP.contains("inspect integrity"));
        assert_eq!(parse_args(&args(&["-h"])), Err(ParseFailure::Help(HELP)));
        assert_eq!(parse_args(&args(&["--help"])), Err(ParseFailure::Help(HELP)));
        assert_eq!(
            parse_args(&args(&["inspect", "--help"])),
            Err(ParseFailure::Help(INSPECT_HELP))
        );
    }
}
