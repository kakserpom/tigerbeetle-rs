//! Fuzzer for the CacheMap, checking it against a straightforward model.
//!
//! Port of `src/lsm/cache_map_fuzz.zig`. The whole module is compiled only under
//! `cfg(test)`, standing in for upstream's standalone fuzz binary (`main`).
//!
//! DEVIATIONS:
//! - upstream runs `min(1e9, exponential(avg=1e8))` operations per run; the port scales
//!   this down (seeded deterministically) so the fuzz passes as an ordinary unit test;
//! - `fuzz.random_id`/`random_int_exponential` (upstream `src/testing/fuzz.zig`) are
//!   re-implemented locally until a second consumer justifies centralizing them.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)] // exponential samples are clamped non-negative
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::collections::HashMap;

use tigerbeetle_core::constants::LSM_COMPACTION_OPS;
use tigerbeetle_core::stdx::prng::Prng;

use crate::cache_map::{CacheMap, CacheMapOptions, CacheMapSpec};
use crate::set_associative_cache::{Layout, SetAssociativeCacheSpec};
use crate::tree::ScopeCloseMode;

/// Upstream `TestTable`/`TestValue` from `cache_map.zig`; duplicated here because Rust
/// privacy does not let one test module import another's test-only items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TestValue {
    key: u32,
    value: u32,
    tombstone: bool,
    padding: [u8; 7],
}

impl TestValue {
    const fn new(key: u32, value: u32) -> Self {
        Self { key, value, tombstone: false, padding: [0; 7] }
    }
}

const _: () = assert!(core::mem::size_of::<TestValue>() == 16);

#[derive(Clone, Copy, Debug, Default)]
struct TestTable;

impl Layout for TestTable {}

impl SetAssociativeCacheSpec for TestTable {
    type Key = u32;
    type Value = TestValue;

    fn key_from_value(value: &TestValue) -> u32 {
        value.key
    }

    // DEVIATION: identity hash, mirroring cache_map.rs unit tests (see there).
    fn hash(key: u32) -> u64 {
        u64::from(key)
    }
}

impl CacheMapSpec for TestTable {
    fn tombstone_from_key(key: u32) -> TestValue {
        TestValue { key, value: 0, tombstone: true, padding: [0; 7] }
    }

    fn tombstone(value: &TestValue) -> bool {
        value.tombstone
    }
}

type TestCacheMap = CacheMap<TestTable>;
type Key = <TestTable as SetAssociativeCacheSpec>::Key;
type Value = <TestTable as SetAssociativeCacheSpec>::Value;

const STASH_VALUE_COUNT_MAX: u32 = 1024;
// Use a large scope (relative to stash_value_count_max) to increase the chances of
// (SetAssociativeCache) hash collisions.
const SCOPE_VALUE_COUNT_MAX: u32 = STASH_VALUE_COUNT_MAX;

#[derive(Clone, Copy, Debug)]
struct OpValue {
    op: u32,
    value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScopeMode {
    Open,
    Persist,
    Discard,
}

#[derive(Clone, Copy, Debug)]
enum FuzzOp {
    Compact,
    Get(Key),
    Upsert(Value),
    Remove(Key),
    Scope(ScopeMode),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FuzzOpTag {
    Compact,
    Get,
    Upsert,
    Remove,
    Scope,
}

struct Environment {
    cache_map: TestCacheMap,
    model: Model,
}

impl Environment {
    fn new(options: CacheMapOptions) -> Self {
        Self { cache_map: CacheMap::new(options), model: Model::default() }
    }

    fn apply(&mut self, fuzz_ops: &[FuzzOp]) {
        // The cache_map should behave exactly like a hash map, with some exceptions:
        // * .compact() removes values added more than one .compact() ago.
        // * .scope_close(.discard) rolls back all operations done from the corresponding
        //   .scope_open()

        for fuzz_op in fuzz_ops {
            match *fuzz_op {
                FuzzOp::Compact => {
                    self.cache_map.compact();
                    self.model.compact();
                }
                FuzzOp::Upsert(value) => {
                    self.cache_map.upsert(&value);
                    self.model.upsert(&value);
                }
                FuzzOp::Remove(key) => {
                    let model_value = self.model.get(key);
                    if let Some(cache_map_value) = self.cache_map.get(key).copied() {
                        assert!(model_value.is_some());
                        assert_eq!(cache_map_value, model_value.expect("present").value);
                    } else {
                        let Some(model_value) = model_value else {
                            continue; // The key doesn't exist.
                        };

                        // If the entry has an op from one or more compactions ago, it
                        // may have been evicted from the cache.
                        // It must be loaded into the cache before removal, though.
                        assert!(self.model.compacts > model_value.op);
                        self.cache_map.upsert(&model_value.value);
                    }

                    self.cache_map.remove(key);
                    self.model.remove(key);
                }
                FuzzOp::Get(key) => {
                    // Get account from cache_map.
                    let cache_map_value = self.cache_map.get(key).copied();

                    // Compare result to model.
                    let model_value = self.model.get(key);
                    match model_value {
                        None => {
                            assert!(cache_map_value.is_none());
                        }
                        Some(model_value) if self.model.compacts > model_value.op => {
                            // .compact() support; if the entry has an op 1 or more compacts
                            // ago, it doesn't have to exist in the cache_map. It may still be
                            // served from the cache layer, however.
                            if let Some(cache_map_value) = cache_map_value {
                                assert_eq!(cache_map_value, model_value.value);
                            }
                        }
                        Some(model_value) => {
                            assert_eq!(
                                model_value.value,
                                cache_map_value.expect("recently written values are cached")
                            );
                        }
                    }
                }
                FuzzOp::Scope(mode) => match mode {
                    ScopeMode::Open => {
                        self.cache_map.scope_open();
                        self.model.scope_open();
                    }
                    ScopeMode::Persist => {
                        self.cache_map.scope_close(ScopeCloseMode::Persist);
                        self.model.scope_close(true);
                    }
                    ScopeMode::Discard => {
                        self.cache_map.scope_close(ScopeCloseMode::Discard);
                        self.model.scope_close(false);
                    }
                },
            }
        }
    }

    /// Verifies both the positive and negative spaces, as both are equally important. We verify
    /// the positive space by iterating over our model, and ensuring everything exists and is
    /// equal in the cache_map.
    ///
    /// We verify the negative space by iterating over the cache_map's cache and maps directly,
    /// ensuring that:
    /// 1. The values in the cache all exist and are equal in the model.
    /// 2. The values in stash either exists and are equal in the model, or there's the same key
    ///    in the cache.
    fn verify(&mut self) {
        for (&key, entry) in &self.model.map {
            // Compare from cache_map, if found:
            let cache_map_value = self.cache_map.get(key).copied();
            if let Some(cache_map_value) = cache_map_value {
                assert_eq!(entry.value, cache_map_value);
            } else {
                // .compact() support:
                assert!(self.model.compacts > entry.op);
            }
        }

        // It's fine for the cache_map to have values older than .compact() in it; good, in fact,
        // but they _MUST NOT_ be stale.
        if let Some(cache) = self.cache_map.cache.as_ref() {
            for (_index, cache_value) in cache.occupied_slots() {
                let model_val = self
                    .model
                    .get(TestTable::key_from_value(cache_value))
                    .expect("cached value must exist in the model");
                assert_eq!(*cache_value, model_val.value);
            }
        }

        // The stash can have stale values, but in that case the real value _must_ exist
        // in the cache. It should be impossible for the stash to have a value that isn't in the
        // model, since cache_map.remove() removes from both the cache and stash.
        for &stash_value in self.cache_map.stash.values() {
            // Get account from model.
            let model_value = self.model.get(TestTable::key_from_value(&stash_value));

            // Even if the stash has stale values, the key must still exist in the model.
            if TestTable::tombstone(&stash_value) {
                continue; // Model may or may not hold the key.
            }
            assert!(model_value.is_some());

            let stash_value_equal =
                stash_value == model_value.expect("non-tombstoned stash key is modeled").value;

            if !stash_value_equal && let Some(cache) = self.cache_map.cache.as_mut() {
                // We verified all cache entries were equal and correct above, so if it
                // exists, it must be right.
                assert!(cache.get(TestTable::key_from_value(&stash_value)).is_some());
            }
        }
    }
}

#[derive(Default)]
struct Model {
    map: HashMap<Key, OpValue>,
    undo_log: Vec<(Key, Option<OpValue>)>,
    scope_active: bool,
    compacts: u32,
}

impl Model {
    fn get(&self, key: Key) -> Option<&OpValue> {
        self.map.get(&key)
    }

    fn upsert(&mut self, value: &Value) {
        let key = TestTable::key_from_value(value);
        let kv_old = self.map.insert(key, OpValue { op: self.compacts, value: *value });
        if self.scope_active {
            self.undo_log.push((key, kv_old));
        }
    }

    fn remove(&mut self, key: Key) {
        let kv_old = self.map.remove(&key);
        if self.scope_active {
            self.undo_log.push((key, kv_old));
        }
    }

    fn compact(&mut self) {
        assert!(!self.scope_active);
        self.compacts += 1;
    }

    fn scope_open(&mut self) {
        assert!(!self.scope_active);
        assert!(self.undo_log.is_empty());
        self.scope_active = true;
    }

    fn scope_close(&mut self, persist: bool) {
        assert!(self.scope_active);
        self.scope_active = false;

        if persist {
            self.undo_log.clear();
        } else {
            while let Some(undo_entry) = self.undo_log.pop() {
                if let Some(value) = undo_entry.1 {
                    self.map.insert(undo_entry.0, value);
                } else {
                    self.map.remove(&undo_entry.0);
                }
            }
        }
        assert!(self.undo_log.is_empty());
    }
}

/// Returns an integer with an exponential distribution of rate `avg`.
///
/// TODO(port): src/testing/fuzz.zig:16 — centralize if more fuzzers need it.
// Note: upstream also uses floats here and relies on the std implementation, noting they
// "should do neither"; test-only code keeps that trade-off.
fn random_int_exponential(prng: &mut Prng, avg: u64) -> u64 {
    let uniform = (prng.int_u64() >> 11) as f64 * f64::powi(2.0, -53);
    let exp = -(1.0 - uniform).ln();
    let value = exp * avg as f64;
    if value.is_sign_negative() {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

/// We have two opposing desires for prng ids:
/// 1. We want to cause many collisions.
/// 2. We want to generate enough ids that various caches can't hold them all.
///
/// So, flip a coin and pick an ID either from a small, or from a large set.
fn random_id(prng: &mut Prng) -> Key {
    let average_cold = SCOPE_VALUE_COUNT_MAX
        + STASH_VALUE_COUNT_MAX
        + u32::try_from(CacheMap::<TestTable>::cache_value_count_max_multiple())
            .expect("cache multiple fits u32");
    let average = if prng.boolean() { 8 } else { average_cold };
    u32::try_from(random_int_exponential(prng, u64::from(average)))
        .expect("exponential sample fits u32")
}

/// Upstream `prng.enum_weighted` over a fixed weight table.
fn enum_weighted(prng: &mut Prng, weights: &[(FuzzOpTag, u64)]) -> FuzzOpTag {
    let total: u64 = weights.iter().map(|&(_, weight)| weight).sum();
    assert!(total > 0);
    let mut pick = prng.int_inclusive_usize(total as usize - 1);
    for &(tag, weight) in weights {
        let weight = weight as usize;
        if pick < weight {
            return tag;
        }
        pick -= weight;
    }
    unreachable!("weighted pick exhausted the total")
}

fn generate_fuzz_ops(prng: &mut Prng, fuzz_op_count: usize) -> Vec<FuzzOp> {
    let fuzz_op_weights: Vec<(FuzzOpTag, u64)> = vec![
        // Always do puts, and always more puts than removes.
        (FuzzOpTag::Upsert, (LSM_COMPACTION_OPS * 2) as u64),
        // Maybe do some removes.
        (FuzzOpTag::Remove, if prng.boolean() { 0 } else { LSM_COMPACTION_OPS as u64 }),
        // Maybe do some gets.
        (FuzzOpTag::Get, if prng.boolean() { 0 } else { LSM_COMPACTION_OPS as u64 }),
        // Maybe do some extra compacts.
        (FuzzOpTag::Compact, if prng.boolean() { 0 } else { 2 }),
        // Maybe use scopes.
        (FuzzOpTag::Scope, if prng.boolean() { 0 } else { (LSM_COMPACTION_OPS / 4) as u64 }),
    ];

    // TODO(upstream): Is there a point to making _max random (both here and in init) and
    // anything less than the maximum capacity...?
    let mut operations_since_scope_open: usize = 0;
    let operations_since_scope_open_max: usize = SCOPE_VALUE_COUNT_MAX as usize;
    let mut upserts_since_compact: usize = 0;
    let upserts_since_compact_max: usize = STASH_VALUE_COUNT_MAX as usize;
    let mut scope_is_open = false;

    let mut fuzz_ops = Vec::with_capacity(fuzz_op_count);
    for i in 0..fuzz_op_count {
        let fuzz_op_tag = if upserts_since_compact >= upserts_since_compact_max {
            // We have to compact before doing any other operations, but the scope must be
            // closed.
            if scope_is_open { FuzzOpTag::Scope } else { FuzzOpTag::Compact }
        } else if operations_since_scope_open >= operations_since_scope_open_max {
            // We have to close our scope before doing anything else.
            FuzzOpTag::Scope
        } else if i == fuzz_op_count - 1 && scope_is_open {
            // Ensure we close scope before ending.
            FuzzOpTag::Scope
        } else {
            let tag = enum_weighted(prng, &fuzz_op_weights);
            if scope_is_open && tag == FuzzOpTag::Compact {
                // We can't compact while a scope is open.
                FuzzOpTag::Scope
            } else if !scope_is_open && i == fuzz_op_count - 1 && tag == FuzzOpTag::Scope {
                // We can't let our final operation be a scope open.
                FuzzOpTag::Get
            } else {
                tag
            }
        };

        let fuzz_op = match fuzz_op_tag {
            FuzzOpTag::Upsert => {
                upserts_since_compact += 1;
                if scope_is_open {
                    operations_since_scope_open += 1;
                }
                FuzzOp::Upsert(TestValue::new(random_id(prng), prng.int_u32()))
            }
            FuzzOpTag::Remove => {
                upserts_since_compact += 1; // remove() adds a tombstone to the stash.
                if scope_is_open {
                    operations_since_scope_open += 1;
                }
                FuzzOp::Remove(random_id(prng))
            }
            FuzzOpTag::Get => FuzzOp::Get(random_id(prng)),
            FuzzOpTag::Compact => {
                upserts_since_compact = 0;
                FuzzOp::Compact
            }
            FuzzOpTag::Scope => {
                operations_since_scope_open = 0;
                let closing = scope_is_open;
                scope_is_open = !scope_is_open;
                if closing {
                    if prng.boolean() {
                        FuzzOp::Scope(ScopeMode::Persist)
                    } else {
                        FuzzOp::Scope(ScopeMode::Discard)
                    }
                } else {
                    FuzzOp::Scope(ScopeMode::Open)
                }
            }
        };
        fuzz_ops.push(fuzz_op);
    }

    fuzz_ops
}

#[test]
fn cache_map_fuzz() {
    let mut prng = Prng::from_seed(42);

    // Upstream: min(events_max orelse 1e9, exponential(avg=1e8)) — reduced scale for CI.
    let fuzz_op_count = 100_000.min(random_int_exponential(&mut prng, 20_000) as usize);
    let fuzz_ops = generate_fuzz_ops(&mut prng, fuzz_op_count);

    // Running the same fuzz with and without cache enabled.
    for cache_value_count_max in [
        u32::try_from(CacheMap::<TestTable>::cache_value_count_max_multiple())
            .expect("cache multiple fits u32"),
        0,
    ] {
        let options = CacheMapOptions {
            cache_value_count_max,
            stash_value_count_max: STASH_VALUE_COUNT_MAX,
            scope_value_count_max: SCOPE_VALUE_COUNT_MAX,
            name: "fuzz map",
        };

        let mut env = Environment::new(options);
        env.apply(&fuzz_ops);
        env.verify();
    }
}
