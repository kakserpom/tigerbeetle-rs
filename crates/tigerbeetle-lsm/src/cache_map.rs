//! A CacheMap is a hybrid between our [`SetAssociativeCache`] and a hash map (stash).
//! The SetAssociativeCache sits on top and absorbs the majority of get / put requests.
//! Below that lives a hash map. Should an `upsert()` cause an eviction (which can happen
//! either because the Key is the same, or because our Way is full), the evicted value is
//! caught and put in the stash.
//!
//! This allows for a potentially huge cache, with all the advantages of CLOCK Nth-Chance,
//! while still being able to give hard guarantees that values will be present. The stash
//! will often be significantly smaller, as the amount of values we're required to guarantee
//! is less than what we'd like to optimistically keep in memory.
//!
//! Within our LSM, the CacheMap is the backing for the combined Groove prefetch + cache.
//! The cache part fills the use case of an object cache, while the stash ensures that
//! prefetched values are available in memory during their respective commit.
//!
//! Cache invalidation for the stash is handled by `compact`.
//!
//! Upstream: `src/lsm/cache_map.zig`.
//!
//! DEVIATION: upstream's stash is `std.HashMapUnmanaged(Value, void, …)` keyed by the whole
//! Value with equality/hashing derived from `key_from_value`; this port uses
//! `std::collections::HashMap<Key, Value>`, which is operation-for-operation equivalent
//! (all stash access is single-key; the map is never iterated). Consequently `Key`
//! additionally requires `Eq + Hash`, and hashing uses std's hasher rather than
//! `stdx.hash_inline(key_from_value(…))`.
//!
//! DEVIATION: upstream stores the rollback log as an allocator-backed ArrayList sized at
//! `options.scope_value_count_max`; this port uses a plain `Vec` guarded by the same
//! capacity assertion on every append (`appendAssumeCapacity`).
//!
//! DEVIATION: lookups take `&mut self` because the underlying [`SetAssociativeCache`] bumps
//! reference counters on hits (see its port notes).

#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;

use tigerbeetle_core::constants::VERIFY;

use crate::set_associative_cache::{SetAssociativeCache, SetAssociativeCacheSpec};
use crate::tree::ScopeCloseMode;

/// Static description of a cache-map instantiation (upstream `CacheMapType` comptime
/// function parameters).
pub trait CacheMapSpec: SetAssociativeCacheSpec<Key: Eq + std::hash::Hash> + 'static {
    /// Builds the tombstone representation of `key` (upstream `tombstone_from_key`).
    fn tombstone_from_key(key: Self::Key) -> Self::Value;
    /// Whether `value` is a tombstone (upstream `tombstone`).
    fn tombstone(value: &Self::Value) -> bool;
}

/// The hierarchy for lookups is cache (if present) -> stash -> immutable table -> lsm.
/// Lower levels _may_ have stale values, provided the correct value exists in one of the
/// levels above. Evictions from the cache first flow into stash, with `.compact()` clearing
/// it. When cache is null, the stash mirrors the mutable table.
pub struct CacheMap<S: CacheMapSpec> {
    /// Crate-visible so that `cache_map_fuzz` can verify internal consistency
    /// (upstream's fuzzer reaches into these directly).
    pub(crate) cache: Option<SetAssociativeCache<S>>,
    pub(crate) stash: HashMap<S::Key, S::Value>,

    /// Scopes allow performing operations on the CacheMap before either persisting or
    /// discarding them.
    scope_is_active: bool,
    scope_rollback_log: RollbackLog<S>,

    options: CacheMapOptions,
}

#[derive(Clone, Debug)]
pub struct CacheMapOptions {
    pub cache_value_count_max: u32,
    pub stash_value_count_max: u32,
    pub scope_value_count_max: u32,
    pub name: &'static str,
}

#[derive(Clone, Copy)]
enum RollbackLogAction<S: CacheMapSpec> {
    /// The operation updated or deleted a value that needs to be restored on rollback.
    Restore(S::Value),
    /// The operation inserted a value over a _tombstone_ that needs to be restored on
    /// rollback.
    RestoreTombstone(S::Key),
    /// The operation inserted a value that did not previously exist and must be removed on
    /// rollback.
    Remove(S::Key),
}

struct RollbackLog<S: CacheMapSpec> {
    actions: Vec<RollbackLogAction<S>>,
    capacity: usize,
}

impl<S: CacheMapSpec> RollbackLog<S> {
    fn new(capacity: usize) -> Self {
        Self { actions: Vec::new(), capacity }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }

    fn append(&mut self, action: RollbackLogAction<S>) {
        assert!(self.actions.len() < self.capacity, "scope rollback log overflow");
        self.actions.push(action);
    }

    fn clear(&mut self) {
        self.actions.clear();
    }
}

/// Result of [`CacheMap::get_or_tombstone`] (upstream anonymous union).
#[derive(Clone, Copy, Debug)]
pub enum GetOrTombstone<'a, S: CacheMapSpec> {
    Found(&'a S::Value),
    NotFound,
    Tombstone,
}

impl<S: CacheMapSpec> CacheMap<S> {
    /// Upstream `Cache.value_count_max_multiple`, for sizing `cache_value_count_max`.
    #[must_use]
    pub fn cache_value_count_max_multiple() -> u64 {
        SetAssociativeCache::<S>::value_count_max_multiple()
    }

    /// Upstream `init`. See the module docs for the stash/log storage deviations.
    ///
    /// # Panics
    /// Panics if `stash_value_count_max` is zero (upstream assert).
    #[must_use]
    pub fn new(options: CacheMapOptions) -> Self {
        assert!(options.stash_value_count_max > 0);

        // TODO(port): upstream soft-asserts (stdx.maybe) that these may legitimately be zero.
        let cache = (options.cache_value_count_max != 0).then(|| {
            SetAssociativeCache::<S>::new(u64::from(options.cache_value_count_max), options.name)
        });

        let stash = HashMap::with_capacity(options.stash_value_count_max as usize);

        Self {
            cache,
            stash,
            scope_is_active: false,
            scope_rollback_log: RollbackLog::new(options.scope_value_count_max as usize),
            options,
        }
    }

    /// # Panics
    /// Panics if called while a scope is active, with a non-empty rollback log, or if the
    /// stash exceeds its configured maximum (upstream asserts).
    pub fn reset(&mut self) {
        assert!(!self.scope_is_active);
        assert_eq!(self.scope_rollback_log.len(), 0);
        assert!(self.stash.len() as u64 <= u64::from(self.options.stash_value_count_max));

        if let Some(cache) = self.cache.as_mut() {
            cache.reset();
        }
        self.stash.clear();

        self.scope_is_active = false;
        self.scope_rollback_log.clear();
    }

    #[must_use]
    pub fn has(&mut self, key: S::Key) -> bool {
        self.get(key).is_some()
    }

    #[must_use]
    pub fn get(&mut self, key: S::Key) -> Option<&S::Value> {
        let from_cache = self.cache.as_mut().and_then(|cache| cache.get(key));

        if from_cache.is_some() {
            return from_cache;
        }

        // Deleted keys are represented as tombstones in the stash.
        self.stash.get(&key).filter(|object| !S::tombstone(object))
    }

    #[must_use]
    pub fn get_or_tombstone(&mut self, key: S::Key) -> GetOrTombstone<'_, S> {
        if let Some(object) = self.cache.as_mut().and_then(|cache| cache.get(key)) {
            return GetOrTombstone::Found(object);
        }
        if let Some(object) = self.stash.get(&key) {
            // Deleted keys are represented as tombstones in the stash.
            return if S::tombstone(object) {
                GetOrTombstone::Tombstone
            } else {
                GetOrTombstone::Found(object)
            };
        }

        GetOrTombstone::NotFound
    }

    #[must_use]
    pub fn cache_entries(&self) -> u64 {
        self.cache.as_ref().map_or(0, SetAssociativeCache::value_count)
    }

    #[must_use]
    pub const fn cache_entries_max(&self) -> u64 {
        self.options.cache_value_count_max as u64
    }

    /// # Panics
    /// Panics under `constants.verify` when upserting over an existing tombstone inside a
    /// scope (upstream asserts this only happens in tests), or if a scope rollback log
    /// overflows its capacity.
    pub fn upsert(&mut self, value: &S::Value) {
        let updated = self.fetch_upsert(value);

        // When upserting into a scope:
        if self.scope_is_active {
            let rollback_action = match updated {
                None => RollbackLogAction::Remove(S::key_from_value(value)),
                Some(old_value) => {
                    if S::tombstone(&old_value) {
                        // Only unit tests and fuzzers call `remove`.
                        // Tombstones should never be present in production code.
                        #[allow(clippy::assertions_on_constants)]
                        {
                            assert!(VERIFY);
                        }
                        RollbackLogAction::RestoreTombstone(S::key_from_value(value))
                    } else {
                        RollbackLogAction::Restore(old_value)
                    }
                }
            };
            self.scope_rollback_log.append(rollback_action);
        }
    }

    /// Upserts the cache and stash and returns the old value in case of an update.
    fn fetch_upsert(&mut self, value: &S::Value) -> Option<S::Value> {
        let Some(cache) = self.cache.as_mut() else {
            // No cache. Upserting the stash directly.
            return self.stash_upsert(value);
        };

        let key = S::key_from_value(value);
        let result = cache.upsert(value);

        if let Some(evicted) = result.evicted {
            match result.updated {
                crate::set_associative_cache::UpdateOrInsert::Update => {
                    assert_eq!(S::key_from_value(&evicted), key);
                    if VERIFY {
                        let stash = self.stash.get(&key);
                        assert!(stash.is_none_or(|stash_value| S::tombstone(stash_value)));
                    }

                    // There was an eviction because an item was updated,
                    // the evicted item is always its previous version.
                    return Some(evicted);
                }
                crate::set_associative_cache::UpdateOrInsert::Insert => {
                    assert_ne!(S::key_from_value(&evicted), key);

                    // There was an eviction because a new item was inserted,
                    // the evicted item will be added to the stash.
                    let stash_updated = self.stash_upsert(&evicted);

                    // We don't expect stale values on the stash.
                    assert!(
                        stash_updated.as_ref().is_none_or(|stash_value| S::tombstone(stash_value))
                    );
                }
            }
        } else {
            // It must be an insert without eviction,
            // since updates always evict the old version.
            assert_eq!(result.updated, crate::set_associative_cache::UpdateOrInsert::Insert);
        }

        // The stash may have the old value if nothing was evicted.
        self.stash_remove(key)
    }

    fn stash_upsert(&mut self, value: &S::Value) -> Option<S::Value> {
        let old = self.stash.insert(S::key_from_value(value), *value);
        assert!(self.stash.len() as u64 <= u64::from(self.options.stash_value_count_max));
        old
    }

    /// Removes a key from cache, adding a tombstone to record the action.
    /// Invariant: The key must be present in cache.
    ///
    /// Only unit tests and fuzzers call this function; upstream guards it with a comptime
    /// `assert(constants.verify)`.
    ///
    /// # Panics
    /// Panics unless built with `verify`, or when removing a key absent from both cache and
    /// stash, or when removing an already-tombstoned value (upstream asserts).
    pub fn remove(&mut self, key: S::Key) {
        // Upstream asserts this at compile time; ours is a true constant today.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(VERIFY);
        }

        let cache_removed = self.cache.as_mut().and_then(|cache| cache.remove(key));

        // The stash and cache can contain different versions of the same key,
        // so we also need to remove it from the stash, leaving a tombstone in
        // its place. The tombstone indicates that, although the object is not
        // present in the cache, a deletion occurred and the key must not be
        // looked up in the immutable table or LSM tree.
        let tombstone_object = S::tombstone_from_key(key);

        // If the key exists in the stash, it will be replaced with a tombstone without
        // increasing the stash count.
        assert!(self.stash.len() as u64 <= u64::from(self.options.stash_value_count_max));
        let stash_removed = self.stash.insert(key, tombstone_object);

        // Does not allow removing a key that is not in the cache.
        assert!(cache_removed.is_some() || stash_removed.is_some());

        let Some(old_value) = cache_removed.or(stash_removed) else {
            unreachable!("key must exist in cache or stash");
        };
        // Cannot remove a value that has already been removed.
        assert!(!S::tombstone(&old_value));
        if self.scope_is_active {
            self.scope_rollback_log.append(RollbackLogAction::Restore(old_value));
        }
    }

    fn stash_remove(&mut self, key: S::Key) -> Option<S::Value> {
        assert!(self.stash.len() as u64 <= u64::from(self.options.stash_value_count_max));
        self.stash.remove(&key)
    }

    /// Start a new scope. Within a scope, changes can be persisted or discarded. At most one
    /// scope can be active at a time.
    ///
    /// # Panics
    /// Panics if a scope is already active or the rollback log is non-empty.
    pub fn scope_open(&mut self) {
        assert!(!self.scope_is_active);
        assert_eq!(self.scope_rollback_log.len(), 0);
        self.scope_is_active = true;
    }

    /// # Panics
    /// Panics if no scope is active, or — when discarding — under replay of a tombstone
    /// restore without `verify`, or on any invariant violation during replay (upstream
    /// asserts).
    pub fn scope_close(&mut self, mode: ScopeCloseMode) {
        assert!(self.scope_is_active);
        self.scope_is_active = false;

        // We don't need to do anything to persist a scope.
        if let ScopeCloseMode::Persist = mode {
            self.scope_rollback_log.clear();
            return;
        }

        // The scope_rollback_log stores the operations we need to reverse the changes a
        // scope made. They get replayed in reverse order. The log is taken out of `self`
        // wholesale so that replays can mutate the cache map freely.
        let actions = std::mem::take(&mut self.scope_rollback_log.actions);
        for action in actions.into_iter().rev() {
            self.replay(&action);
        }

        self.scope_rollback_log.clear();
    }

    fn replay(&mut self, action: &RollbackLogAction<S>) {
        match *action {
            RollbackLogAction::Restore(rollback_value) => {
                // Reverting an update or delete consists of an insert of the original value.
                assert!(!S::tombstone(&rollback_value));
                self.upsert(&rollback_value);
            }
            RollbackLogAction::RestoreTombstone(key) => {
                // Reverting an insert that overwrote a tombstone consists of restoring the
                // tombstone to the stash.
                if VERIFY {
                    // Only unit tests and fuzzers call `remove`.
                    self.remove(key);
                } else {
                    unreachable!("restore_tombstone requires verify");
                }
            }
            RollbackLogAction::Remove(key) => {
                // Reverting an insert consists of removing the value.
                let cache_removed =
                    self.cache.as_mut().and_then(|cache| cache.remove(key)).is_some();

                // The key should be in the stash iff it wasn't in the cache.
                if let Some(stash_value) = self.stash_remove(key) {
                    assert!(!cache_removed || S::tombstone(&stash_value));
                } else {
                    assert!(cache_removed);
                }
            }
        }
    }

    /// # Panics
    /// Panics if called while a scope is active, with a non-empty rollback log, or if the
    /// stash exceeds its configured maximum (upstream asserts).
    pub fn compact(&mut self) {
        assert!(!self.scope_is_active);
        assert_eq!(self.scope_rollback_log.len(), 0);
        assert!(self.stash.len() as u64 <= u64::from(self.options.stash_value_count_max));

        self.stash.clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::{CacheMap, CacheMapOptions, CacheMapSpec, GetOrTombstone};
    use crate::set_associative_cache::{Layout, SetAssociativeCacheSpec};
    use crate::tree::ScopeCloseMode;

    /// Upstream `TestTable`. The `padding` field exists upstream to round the value size up
    /// to a power of two (16 bytes), as required by the set-associative cache layout.
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

    // Upstream relies on this size when computing cache geometry.
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

        fn hash(key: u32) -> u64 {
            // DEVIATION: upstream uses stdx.hash_inline(key); identity suffices for these
            // unit tests and keeps the port dependency-free.
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

    #[test]
    fn cache_map_unit() {
        let mut cache_map = CacheMap::<TestTable>::new(CacheMapOptions {
            cache_value_count_max: CacheMap::<TestTable>::cache_value_count_max_multiple() as u32,
            scope_value_count_max: 32,
            stash_value_count_max: 32,
            name: "test map",
        });

        assert_eq!(cache_map.cache_entries(), 0);
        assert_eq!(
            cache_map.cache_entries_max(),
            u64::from(cache_map.options.cache_value_count_max)
        );

        cache_map.upsert(&TestValue::new(1, 1));
        assert_eq!(cache_map.cache_entries(), 1);
        assert_eq!(*cache_map.get(1).expect("present"), TestValue::new(1, 1));

        // Test scope persisting.
        cache_map.scope_open();
        cache_map.upsert(&TestValue::new(2, 2));
        assert_eq!(*cache_map.get(2).expect("present"), TestValue::new(2, 2));
        cache_map.scope_close(ScopeCloseMode::Persist);
        assert_eq!(*cache_map.get(2).expect("present"), TestValue::new(2, 2));

        // Test scope discard on updates.
        cache_map.scope_open();
        cache_map.upsert(&TestValue::new(2, 22));
        cache_map.upsert(&TestValue::new(2, 222));
        cache_map.upsert(&TestValue::new(2, 2222));
        assert_eq!(*cache_map.get(2).expect("present"), TestValue::new(2, 2222));
        cache_map.scope_close(ScopeCloseMode::Discard);
        assert_eq!(*cache_map.get(2).expect("present"), TestValue::new(2, 2));

        // Test scope discard on inserts.
        cache_map.scope_open();
        cache_map.upsert(&TestValue::new(3, 3));
        assert_eq!(*cache_map.get(3).expect("present"), TestValue::new(3, 3));
        cache_map.upsert(&TestValue::new(3, 33));
        assert_eq!(*cache_map.get(3).expect("present"), TestValue::new(3, 33));
        cache_map.scope_close(ScopeCloseMode::Discard);
        assert!(!cache_map.has(3));
        assert!(cache_map.get(3).is_none());

        // Test scope discard on removes.
        cache_map.scope_open();
        cache_map.remove(2);
        assert!(!cache_map.has(2));
        assert!(cache_map.get(2).is_none());
        cache_map.scope_close(ScopeCloseMode::Discard);
        assert_eq!(*cache_map.get(2).expect("present"), TestValue::new(2, 2));

        // Test scope discard on a sequence of insert->remove->insert.
        cache_map.upsert(&TestValue::new(4, 4));
        assert_eq!(*cache_map.get(4).expect("present"), TestValue::new(4, 4));

        cache_map.remove(4);
        assert!(!cache_map.has(4));
        assert!(cache_map.get(4).is_none());
        assert!(matches!(cache_map.get_or_tombstone(4), GetOrTombstone::Tombstone));

        cache_map.scope_open();
        cache_map.upsert(&TestValue::new(4, 4));
        cache_map.scope_close(ScopeCloseMode::Discard);

        assert!(!cache_map.has(4));
        assert!(cache_map.get(4).is_none());
        assert!(matches!(cache_map.get_or_tombstone(4), GetOrTombstone::Tombstone));
    }
}
