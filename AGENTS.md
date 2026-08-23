# AGENTS.md — tigerbeetle-rs

Rust rewrite of [TigerBeetle](https://github.com/tigerbeetle/tigerbeetle), the financial
accounting database. The upstream Zig implementation is the **specification**; this repo is a
faithful port, not a redesign.

## Reference checkout

`reference/tigerbeetle/` is a shallow clone of upstream (gitignored). Refresh it with:

```sh
git -C reference/tigerbeetle fetch --depth 1 origin master && git -C reference/tigerbeetle reset --hard FETCH_HEAD
```

Key upstream reading (under `reference/tigerbeetle/docs/`):

- `ARCHITECTURE.md` — read this first; system overview and design decisions.
- `TIGER_STYLE.md` — engineering style that this port inherits.
- `internals/vsr.md`, `internals/lsm.md`, `internals/data_file.md` — subsystem deep dives.

## Workspace layout

Cargo workspace; members live under `crates/` and share lints via `[workspace.lints]`:

| Crate                 | Role                                            | Upstream origin |
| --------------------- | ----------------------------------------------- | --------------- |
| `tigerbeetle-core`    | constants, config, stdx shims, checksums        | `src/constants.zig`, `config.zig`, `src/stdx/`, `src/vsr/checksum.zig` |
| `tigerbeetle-server`  | server binary (bin name: `tigerbeetle`)         | `src/tigerbeetle/main.zig`, `cli.zig` |

Future crates: `tigerbeetle-vsr`, `tigerbeetle-lsm`, `tigerbeetle-client`. Keep the dependency
graph acyclic bottom-up (`core` ← `lsm` ← `vsr` ← `server`); do not add reverse edges.

## Upstream → Rust module map

| Upstream (Zig)                     | Role                                        | Planned Rust location |
| ---------------------------------- | ------------------------------------------- | --------------------- |
| `src/constants.zig`, `config.zig`  | Sizes, limits, configuration                | `tigerbeetle-core/src/{constants,config}.rs` |
| `src/vsr.zig`, `src/vsr/`          | Viewstamped Replication consensus, checksums, clock, replica/client | `tigerbeetle-vsr/src/` |
| `src/lsm/`                         | Forest of LSM trees (grooves, tables, manifests, compaction, scans) | `tigerbeetle-lsm/src/` |
| `src/state_machine.zig`            | Accounting logic: accounts/transfers/balances | `tigerbeetle-lsm/src/state_machine.rs` |
| `src/storage.zig`, `src/io*`       | Data file, direct I/O per platform          | `tigerbeetle-core/src/{storage,io}/` |
| `src/message_pool.zig`, `message_buffer.zig`, `message_bus.zig` | Message memory + networking | `tigerbeetle-vsr/src/message*.rs` |
| `src/tigerbeetle/main.zig`, `cli.zig` | Server binary                            | `tigerbeetle-server/src/` |
| `src/clients/rust/`                | Existing official Rust client (do not duplicate blindly) | reference only |
| `src/stdx/`, `src/time.zig`, `src/queue.zig` | std shims — usually map to `std` or small helpers | `tigerbeetle-core/src/stdx.rs` as needed |

Port one subsystem at a time, bottom-up: constants → checksum → storage primitives → lsm →
state machine → vsr → io/networking → binary.

## Porting rules

1. **The Zig source is the spec.** Port semantics 1:1 first. Do not "improve" algorithms,
   rename protocol concepts (`prepare`, `commit`, `superblock`, `groove`, `checkpoint`,
   `view`, `head`), or change on-disk/wire formats while porting.
2. **Determinism is sacred.** The state machine must be bit-for-bit reproducible:
   - no floating point in state transitions;
   - no wall-clock time in logic — use the VSR synthetic clock;
   - iteration order over hashed collections must never influence state or output.
3. **No `unsafe`.** Enforced by `cargo` lint config. If a subsystem seems to need it
   (e.g. direct I/O), isolate it behind a safe abstraction and document why.
4. **Assert liberally.** TigerBeetle keeps assertions in release builds; use `assert!`
   (not `debug_assert!`) for invariants, matching upstream's `assert`.
5. **No panics on invalid input** from clients/disk: return explicit errors like upstream
   returns error codes. `unwrap`/`expect` are denied by clippy; use `assert!` for internal
   invariants instead.
6. **Zero runtime dependencies** unless justified in writing here first. Upstream is
   deliberately dependency-free; mirror that. Dev-only deps (e.g. test harnesses) also
   need justification.
7. **Deviations must be documented**: when Rust forces a structural difference, add a doc
   comment starting with `DEVIATION:` explaining what upstream does and why we differ.
8. **Tests travel with code.** Port upstream unit tests alongside each module, keeping
   upstream test names where practical. Fuzzers (`*_fuzz.zig`) become `#[test]`-driven
   deterministic seeds until we adopt a fuzzing setup.

## Commands

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt && cargo fmt --check
```

All four must pass before finishing any task. Lints (`unsafe_code = forbid`,
full clippy::pedantic) live in `Cargo.toml`; do not weaken them to make code compile.

## Style

- 100-column hard limit (`rustfmt.toml`), trailing commas, rustfmt decides wrapping.
- Mirror Zig file layout inside modules so diffing against upstream stays easy.
- Prefer plain data structs with explicit padding/layout (`#[repr(C)]`) where the struct
  maps to disk or wire format; add static size assertions (`assert_eq!(size_of::<T>(), N)`).
- Keep TODOs referencing upstream file/function, e.g. `TODO(port): src/lsm/table.zig:210`.

## Open decisions (do not silently pick)

- Async model: upstream hand-rolls its event loop. Default plan is a hand-rolled reactor
  too; introducing tokio et al. requires updating this section first.
- Crate layout: **decided** — workspace crates from the start (`tigerbeetle-core`,
  `tigerbeetle-lsm`, `tigerbeetle-vsr`, `tigerbeetle-server`, `tigerbeetle-client`).
