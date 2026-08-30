//! Per-session object cache with chain-wide scope rollback.
//!
//! Upstream: `src/lsm/cache_map.zig` (`CacheMapType`), instantiated as the
//! `ObjectsCache` of each groove (`src/lsm/groove.zig`).
//!
//! Upstream's CacheMap is a SetAssociativeCache with a HashMap stash below it.
//! The stash carries the *guarantee* that holds within a commit: every object
//! that was prefetched, plus every object inserted in-session, is present for
//! the batch's reads to see. This port keeps only that stash layer — there is
//! no tree/immutable-table backing to absorb lookups sans-IO, so the
//! associative cache on top would cache nothing.
//!
//! Scopes implement chain semantics: `scope_open` starts a scope,
//! `scope_close(.persist)` commits it, and `scope_close(.discard)` replays a
//! rollback log (inserts removed, updates restored to their prior value) to
//! undo every mutation made since the scope opened.
//!
//! DEVIATION: upstream keys the stash by a `u128`-bit identifying the object's
//! primary key and marks deleted keys with tombstones (`remove` +
//! `restore_tombstone`). Tombstones only matter when a deleted key could
//! otherwise be resurrected from the immutable table, and `remove` is only
//! reachable under `constants.verify` (unit tests/fuzzers) — the port never
//! removes an object, so neither tombstones nor `remove` are ported. Deleting
//! a key created within a scope is a bare rollback-log `remove`.

use std::collections::HashMap;
use std::hash::Hash;

use tigerbeetle_lsm::tree::ScopeCloseMode;

/// The primary key of an object, used as the cache key.
///
/// Upstream reads the cache key out of the value via the groove's
/// `key_from_value` comptime fn; the same `Top-K` identity is expressed as a
/// trait here.
pub trait ObjectKey<K> {
    fn object_key(&self) -> K;
}

/// The actions a scope must reverse if it is discarded, recorded in insertion
/// order and replayed in reverse (`cache_map.zig:353-385`).
#[derive(Clone, Copy)]
enum RollbackAction<K, V> {
    /// The value at `key` was updated; restore the pre-scope value.
    Restore(K, V),
    /// The value at `key` was inserted and did not previously exist; remove it.
    Remove(K),
}

/// A batch-scoped object cache keyed by an object's primary key.
///
/// Upstream: `CacheMapType` with `cache_value_count_max = 0` (stash only).
pub struct ObjectsCache<K, V> {
    stash: HashMap<K, V>,
    scope_active: bool,
    scope_rollback_log: Vec<RollbackAction<K, V>>,
}

impl<K, V> Default for ObjectsCache<K, V> {
    fn default() -> Self {
        Self { stash: HashMap::new(), scope_active: false, scope_rollback_log: Vec::new() }
    }
}

impl<K, V> ObjectsCache<K, V>
where
    K: Eq + Hash + Copy,
    V: ObjectKey<K> + Copy,
{
    /// Returns the object at `key`, or `None` if the key was not prefetched
    /// and has not been inserted in this session (`cache_map.zig:157`).
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.stash.get(key)
    }

    /// Whether the objects cache holds `key`.
    #[must_use]
    pub fn has(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Inserts or updates `value`, recording a rollback action if a scope is
    /// active (`cache_map.zig:196-216`).
    pub fn upsert(&mut self, value: V) {
        let key = value.object_key();
        if self.scope_active {
            let rollback_action = match self.stash.get(&key) {
                Some(previous) => RollbackAction::Restore(key, *previous),
                None => RollbackAction::Remove(key),
            };
            self.scope_rollback_log.push(rollback_action);
        }
        self.stash.insert(key, value);
    }

    /// Start a new scope. At most one scope can be active at a time
    /// (`cache_map.zig:331-335`).
    ///
    /// # Panics
    ///
    /// Panics if a scope is already active or the rollback log is non-empty
    /// (both mirror upstream's `assert`s).
    pub fn scope_open(&mut self) {
        assert!(!self.scope_active);
        assert!(self.scope_rollback_log.is_empty());
        self.scope_active = true;
    }

    /// Close the scope. `.persist` commits its changes (the rollback log is
    /// dropped); `.discard` reverts them in reverse order
    /// (`cache_map.zig:337-385`).
    ///
    /// # Panics
    ///
    /// Panics if no scope is active, mirroring upstream's `assert`.
    pub fn scope_close(&mut self, mode: ScopeCloseMode) {
        assert!(self.scope_active);
        self.scope_active = false;

        if mode == ScopeCloseMode::Persist {
            self.scope_rollback_log.clear();
            return;
        }

        while let Some(action) = self.scope_rollback_log.pop() {
            match action {
                RollbackAction::Restore(key, value) => {
                    self.stash.insert(key, value);
                }
                RollbackAction::Remove(key) => {
                    self.stash.remove(&key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestValue {
        key: u32,
        value: u32,
    }

    impl ObjectKey<u32> for TestValue {
        fn object_key(&self) -> u32 {
            self.key
        }
    }

    const fn tv(key: u32, value: u32) -> TestValue {
        TestValue { key, value }
    }

    #[test]
    fn get_returns_inserted_value() {
        let mut cache = ObjectsCache::<u32, TestValue>::default();
        assert!(!cache.has(&1));
        cache.upsert(tv(1, 1));
        assert!(cache.has(&1));
        assert_eq!(cache.get(&1), Some(&tv(1, 1)));
    }

    #[test]
    fn scope_persist_commits_changes() {
        let mut cache = ObjectsCache::<u32, TestValue>::default();
        cache.upsert(tv(1, 1));

        cache.scope_open();
        cache.upsert(tv(2, 2));
        assert_eq!(cache.get(&2), Some(&tv(2, 2)));
        cache.scope_close(ScopeCloseMode::Persist);

        assert_eq!(cache.get(&2), Some(&tv(2, 2)));
    }

    #[test]
    fn scope_discard_restores_updates() {
        let mut cache = ObjectsCache::<u32, TestValue>::default();
        cache.upsert(tv(2, 2));

        cache.scope_open();
        cache.upsert(tv(2, 22));
        cache.upsert(tv(2, 222));
        cache.upsert(tv(2, 2222));
        assert_eq!(cache.get(&2), Some(&tv(2, 2222)));
        cache.scope_close(ScopeCloseMode::Discard);

        assert_eq!(cache.get(&2), Some(&tv(2, 2)));
    }

    #[test]
    fn scope_discard_removes_inserts() {
        let mut cache = ObjectsCache::<u32, TestValue>::default();

        cache.scope_open();
        cache.upsert(tv(3, 3));
        cache.upsert(tv(3, 33));
        assert_eq!(cache.get(&3), Some(&tv(3, 33)));
        cache.scope_close(ScopeCloseMode::Discard);

        assert!(!cache.has(&3));
        assert!(cache.get(&3).is_none());
    }

    #[test]
    fn scope_discard_rolls_back_only_touched_keys() {
        let mut cache = ObjectsCache::<u32, TestValue>::default();
        cache.upsert(tv(1, 1));

        cache.scope_open();
        cache.upsert(tv(2, 2));
        cache.upsert(tv(1, 11));
        cache.upsert(tv(3, 3));
        cache.scope_close(ScopeCloseMode::Discard);

        // Untouched key survives; updated key restored; inserted keys removed.
        cache.upsert(tv(4, 4));
        assert_eq!(cache.get(&1), Some(&tv(1, 1)));
        assert!(!cache.has(&2));
        assert!(!cache.has(&3));
        assert_eq!(cache.get(&4), Some(&tv(4, 4)));
    }

    #[test]
    fn two_scopes_cannot_be_open_at_once() {
        let mut cache = ObjectsCache::<u32, TestValue>::default();
        cache.scope_open();
        cache.scope_close(ScopeCloseMode::Persist);
        // After close the log is empty and a new scope may open.
        cache.scope_open();
        cache.scope_close(ScopeCloseMode::Discard);

        let mut cache = ObjectsCache::<u32, TestValue>::default();
        cache.scope_open();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cache.scope_open())).is_err()
        );
    }
}
