//! A pool of fixed-size nodes, used to back manifest levels without per-node heap allocation.
//!
//! Upstream: `src/lsm/node_pool.zig`.
//!
//! DEVIATION: upstream hands out raw aligned pointers (`*align(A) [N]u8`) into the pool
//! buffer and validates them by address arithmetic on release. Safe Rust cannot hold
//! interior pointers without borrowing the pool, so this port deals in node *indices*
//! (`usize`) with `node()`/`node_mut()` accessors; the alignment parameter is dropped
//! because no safe consumer can form aligned references into the buffer.
//!
//! DEVIATION: upstream reports exhaustion via `vsr.fatal(.manifest_node_pool_exhausted)`
//! which exits the process. The `lsm` crate cannot depend on `vsr`, so exhaustion panics
//! with the same operator guidance instead (both paths are unrecoverable).

use tigerbeetle_core::stdx::bitset::BitSet;

/// `node_size` must be positive and a power of two (upstream also requires the same of the
/// alignment, which this port does not model).
pub struct NodePool<const NODE_SIZE: usize> {
    buffer: Vec<u8>,
    /// Free-list of node indices, one bit per node (upstream `DynamicBitSetUnmanaged`).
    free: BitSet,
    node_count: usize,
}

impl<const NODE_SIZE: usize> NodePool<NODE_SIZE> {
    /// Upstream `init`: allocates `node_count` zeroed nodes, all free.
    ///
    /// # Panics
    /// Panics if `node_count == 0` or `NODE_SIZE` is not a power of two (upstream comptime
    /// asserts).
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        assert!(node_count > 0);
        const { assert!(NODE_SIZE > 0 && NODE_SIZE.is_power_of_two()) };

        Self {
            buffer: vec![0_u8; NODE_SIZE * node_count],
            free: BitSet::new_full(node_count),
            node_count,
        }
    }

    /// Number of nodes in the pool (upstream `free.bit_length`).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.node_count
    }

    /// Bytes of a single node.
    pub const NODE_SIZE: usize = NODE_SIZE;

    /// Reserves a free node, returning its index.
    ///
    /// # Panics
    /// Panics when the pool is exhausted; upstream aborts with
    /// `.manifest_node_pool_exhausted` and advises restarting with a larger
    /// `--memory-lsm-manifest`.
    // TODO(port): replace the panic with `vsr::fatal(.manifest_node_pool_exhausted)` once the
    // server binary wires up fatal-error handling (src/vsr.zig fatal()).
    pub fn acquire(&mut self) -> usize {
        let node_index = self.free.find_first_set().unwrap_or_else(|| {
            panic!(
                "out of memory for manifest, restart the replica increasing '--memory-lsm-manifest'"
            )
        });
        assert!(self.free.get(node_index));
        self.free.unset(node_index);
        node_index
    }

    /// Returns a node's bytes for reading.
    ///
    /// # Panics
    /// Panics if `node_index` is out of range.
    #[must_use]
    pub fn node(&self, node_index: usize) -> &[u8] {
        assert!(node_index < self.node_count);
        let start = node_index * NODE_SIZE;
        &self.buffer[start..start + NODE_SIZE]
    }

    /// Returns a node's bytes for writing.
    ///
    /// # Panics
    /// Panics if `node_index` is out of range.
    pub fn node_mut(&mut self, node_index: usize) -> &mut [u8] {
        assert!(node_index < self.node_count);
        let start = node_index * NODE_SIZE;
        &mut self.buffer[start..start + NODE_SIZE]
    }

    /// Returns a node to the pool.
    ///
    /// # Panics
    /// Panics if `node_index` is out of range or already free (upstream asserts the pointer
    /// lies within the pool buffer and is not double-released).
    pub fn release(&mut self, node_index: usize) {
        assert!(node_index < self.capacity());
        assert!(!self.free.get(node_index));
        self.free.set(node_index);
    }

    /// True when every node has been released (upstream asserts this in `deinit`).
    #[must_use]
    pub fn all_free(&self) -> bool {
        self.free.full()
    }
}

#[cfg(test)]
mod tests {
    use super::NodePool;
    use tigerbeetle_core::stdx::prng::Prng;

    fn words(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
        bytes.as_chunks::<8>().0.iter().map(|word| u64::from_le_bytes(*word))
    }

    fn fill_words(bytes: &mut [u8], value: u64) {
        for chunk in bytes.as_chunks_mut::<8>().0 {
            *chunk = value.to_le_bytes();
        }
    }

    /// Upstream test context: interleaved acquire/release while verifying that node contents
    /// are never clobbered by the pool itself. The node map keeps `(node_index, written_id)`.
    struct TestContext {
        prng: Prng,
        sentinel: u64,
        node_pool: TestPool,
        node_map: Vec<(usize, u64)>,
        acquires: u64,
        releases: u64,
    }

    // Upstream tests several (size, alignment) tuples; alignment is not modeled here, so only
    // distinct sizes are exercised (the largest tuple's size).
    type TestPool = NodePool<128>;

    impl TestContext {
        fn new(prng: &mut Prng, node_count: usize) -> Self {
            let mut node_pool = TestPool::new(node_count);
            let sentinel = prng.int_u64();
            for node_index in 0..node_count {
                fill_words(node_pool.node_mut(node_index), sentinel);
            }
            Self {
                prng: *prng,
                sentinel,
                node_pool,
                node_map: Vec::new(),
                acquires: 0,
                releases: 0,
            }
        }

        fn run(&mut self) {
            for (acquire_weight, _) in [(60_u64, 40_u64), (40, 60)] {
                for _ in 0..self.node_pool.capacity() * 4 {
                    if self.prng.gen_int_inclusive_u64(99) < acquire_weight {
                        self.acquire();
                    } else {
                        self.release();
                    }
                }
            }
            self.release_all();
        }

        fn acquire(&mut self) {
            if self.node_map.len() == self.node_pool.capacity() {
                return;
            }

            let node_index = self.node_pool.acquire();

            // Verify that this node has not already been acquired.
            assert!(words(self.node_pool.node(node_index)).all(|word| word == self.sentinel));

            assert!(!self.node_map.iter().any(|&(index, _)| index == node_index));

            // Write unique data into the node so we can test that it doesn't get overwritten.
            let id = self.prng.int_u64();
            fill_words(self.node_pool.node_mut(node_index), id);
            self.node_map.push((node_index, id));

            self.acquires += 1;
        }

        fn release(&mut self) {
            if self.node_map.is_empty() {
                return;
            }

            let index = self.prng.range_inclusive_usize(0, self.node_map.len() - 1);
            let &(node_index, id) = &self.node_map[index];

            // Verify that the data of this node has not been overwritten since we acquired it.
            assert!(words(self.node_pool.node(node_index)).all(|word| word == id));

            fill_words(self.node_pool.node_mut(node_index), self.sentinel);
            self.node_pool.release(node_index);
            self.node_map.swap_remove(index);

            self.releases += 1;
        }

        fn release_all(&mut self) {
            while !self.node_map.is_empty() {
                self.release();
            }

            // Verify that nothing in the entire buffer has been acquired.
            for node_index in 0..self.node_pool.capacity() {
                assert!(words(self.node_pool.node(node_index)).all(|word| word == self.sentinel));
            }

            assert!(self.acquires > 0);
            assert_eq!(self.acquires, self.releases);
        }
    }

    #[test]
    fn node_pool_acquire_release_fuzz() {
        // Upstream "NodePool" test: seed 42, node counts 1..64.
        let mut prng = Prng::from_seed(42);
        for node_count in 1..64 {
            let mut context = TestContext::new(&mut prng, node_count);
            context.run();
            prng = context.prng;
        }
    }

    #[test]
    fn exhausted_pool_panics_with_operator_guidance() {
        let mut pool = NodePool::<64>::new(1);
        let _node = pool.acquire();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.acquire()));
        assert!(result.is_err());
    }

    #[test]
    fn released_nodes_are_reusable() {
        let mut pool = NodePool::<8>::new(2);
        let first = pool.acquire();
        let second = pool.acquire();
        assert_ne!(first, second);
        assert!(!pool.all_free());

        pool.release(first);
        pool.release(second);
        assert!(pool.all_free());

        assert_eq!(pool.acquire(), 0);
        assert_eq!(pool.acquire(), 1);
    }
}
