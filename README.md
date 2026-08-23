# tigerbeetle-rs

A Rust rewrite of [TigerBeetle](https://github.com/tigerbeetle/tigerbeetle) — a distributed
financial accounting database designed for mission-critical financial infrastructure.

**Status:** project bootstrap. Cargo workspace (`crates/tigerbeetle-core`,
`crates/tigerbeetle-server`; more to come). The upstream Zig implementation
(shallow-cloned to `reference/tigerbeetle/`, gitignored) is the specification;
porting proceeds bottom-up:
constants → checksum → storage → LSM forest → state machine → VSR consensus → I/O → binary.

See [AGENTS.md](AGENTS.md) for the porting rules, module map, and workflow.
