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

## Config selection (`test-min` feature)

Upstream recompiles constants per consumer via `builtin.is_test` (tests get `test_min`,
binaries `default_production`). Rust has no cross-crate `cfg(test)`, so `tigerbeetle-core`
has a `test-min` cargo feature selecting the config at compile time; every crate enables it
through its `[dev-dependencies]` edge on core. Consequence: all workspace tests run under
`test_min` (like upstream), release binaries under `default_production`. Never read
`CONFIG` from a plain dependency edge in tests and expect production values.

## Cross-validation against upstream

Golden values are generated by running real upstream code with the vendored Zig
(`sh reference/tigerbeetle/zig/download.sh` — upstream pins 0.14.1; system Zig is too new):

- `reference/tigerbeetle/src/tbcross_main.zig` — prints `ConfigCluster.checksum()` +
  `vsr.root_members()` for known clusters (pinned in core `config_tests` / vsr `members_tests`).
- `reference/tigerbeetle/src/tbcross_format.zig` — drives upstream `SuperBlock.format()` +
  `open()` over the test Storage and prints region checksums of the formatted superblock
  zone (pinned in `superblock_tests::format_matches_upstream_zig_golden`). Note: upstream's
  test Storage memory is 0xAA-poisoned by the debug allocator; unwritten padding reads as
  `0xAA` there, so the harness fills our `MemoryStorage` first.

Both files are untracked helpers living inside the gitignored reference checkout; they
declare the root-module escape hatches (`vsr_options`, `tigerbeetle_config`) that force the
`test_min` config in an executable build. Recompile pattern:

```sh
cd reference/tigerbeetle/src && ../zig/zig build-exe \
  --dep stdx -Mroot=tbcross_format.zig \
  -Mstdx=$PWD/stdx/stdx.zig -femit-bin=/tmp/tbformat && /tmp/tbformat
```

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

## Compaction porting notes

- Pure helper functions (`snapshot_max_for_table_input`, `snapshot_min_for_table_output`,
  `compaction_op_min`) live in `tigerbeetle-lsm/src/compaction.rs`.
- `Compaction` struct lives in `tigerbeetle-vsr/src/compaction.rs`, generic over `S: TreeSpec`.
- **No ResourcePool:** block allocation/IOPS management deferred to Phase 3.
- **No I/O dispatch:** `compaction_dispatch`, read/write callbacks deferred to Phase 3.
- **Lifecycle stubs:** `half_bar_commence`, `half_bar_complete`, `beat_commence` deferred to
  Phase 2 (need Manifest operations).
- **`TableInfoA` simplified:** upstream stores `ImmutableTableIterator` in a union; Rust
  ownership makes this self-referential. Instead, we track `Immutable`/`Disk` variant and
  re-create the iterator when needed during the dispatch loop.

## State machine porting notes

- State machine lives in `tigerbeetle-vsr/src/state_machine.rs` (not core) — it depends on
  grooves which are in vsr.
- **Batch orchestrator implemented:** `execute_create_accounts` and `execute_create_transfers`
  handle linked chains, imported batch validation, timestamp assignment, and scope management
  (scope_open/close deferred to prefetch integration).
- **Post/void pending transfers implemented:** `post_or_void_pending_transfer` validates flags,
  pending_id, account/ledger/code matching, amount bounds, pending status, closed accounts.
  Returns `PostVoidPendingResult` with amount and is_post flag.
- **No imported timestamp validation:** Key-range checks (`objects.key_range`) and
  `indirect_lookup` calls deferred.
- **No `account_event` CDC:** The `AccountEvent` groove (history/CDC) does not exist yet.
- **No expiry scheduling:** `expire_pending_transfers.pulse_next_timestamp` scheduling
  deferred.
- **Standalone functions:** `create_account` and `create_transfer` are pure functions taking
  references + lookup results, not methods on a `StateMachine` struct.

## Groove porting notes

- Groove lives in `tigerbeetle-vsr/src/groove.rs` (not `lsm`) — it owns `Tree<S>` which
  is in vsr; putting groove in lsm would create circular dep.
- **No comptime code generation:** upstream uses `GrooveType()` comptime function to
  auto-generate `IndexTrees`, `UniqueKey`, `PrefetchKeys`, `IndexHelperType`. This port
  manually defines concrete structs for Account (9 trees) and Transfer (14 trees).
- **Index tree spec types** (`AccountObjectSpec`, `TransferObjectSpec`,
  `CompositeKey128Spec`, `CompositeKey64Spec`, `CompositeKeyUnitSpec`, `UniqueKey128Spec`)
  implement both `table_memory::Table` and `tree::TreeSpec`.
- **`U256` traits:** `TournamentKey`, `RadixKey`, `manifest::TableKey` impls live in
  `composite_key.rs` (lsm crate). `vsr::table::TableKey` impl lives in `vsr/table.rs`.
- **`TableLayout::compute` is `const fn`** — allows `const LAYOUT` on tree spec types.
  `TableIndex::init` and `TableValue::init` in schema.rs are also `const fn`.
- **No `Grid` stored on `Tree`:** methods needing Grid take `grid: &mut Grid` param.
- **No `ScratchMemory` stored on `Tree`:** passed per-call to `compact()` and
  `swap_mutable_and_immutable()`.
- **`ObjectsCache` deferred:** CacheMap is ported but not yet wired into groove.
  Temporary `HashMap<u128, V>` (`id_map`) provides `get()` for primary-key lookups.
- **Prefetch deferred:** ~480 lines, needs async I/O; stubs only for now.
- **`cached_value_block_search` is a static fn** (no `&mut self`) — avoids borrow-checker
  tension in `lookup_from_levels_cache`.
- **`open_complete` takes `checkpoint_op: u64`** instead of reaching into
  `grid.superblock`.
- **`open_table` bypasses ManifestLog:** Direct `manifest.levels[l].insert_table()`.
- **No Tracer:** `grid.trace.start/stop` calls deferred (TODO).
