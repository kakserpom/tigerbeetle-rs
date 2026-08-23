//! A growable/shrinkable array of fixed-capacity segments ("nodes"), with O(log N) absolute
//! indexing and optional key-ordering, used to back LSM manifest levels.
//!
//! Upstream: `src/lsm/segmented_array.zig`.
//!
//! DEVIATION: upstream allocates node storage from a shared [`NodePool`](crate::node_pool)
//! handed to every method, and its `nodes` array stores raw aligned pointers. Safe Rust
//! cannot hold interior pointers, so this port owns its node buffers
//! (`Box<[Value]>` of exactly `NODE_CAPACITY` elements) and the pool parameter disappears
//! from the API. Capacity accounting (`node_capacity`, `node_count_max`) is preserved via
//! [`SegmentedArray::node_capacity_for`] so callers can size nodes exactly like upstream.
//!
//! DEVIATION: upstream performs overlapping in-place moves (`stdx.copy_right/copy_left`,
//! `mem.copyBackwards`). Where source and destination may alias, this port snapshots the
//! source range (values are `Copy`) and writes it out afterwards; placement semantics are
//! identical, at the cost of one extra pass over the moved range.
//!
//! DEVIATION: upstream has two constructors (`SegmentedArrayType` unsorted /
//! `SortedSegmentedArrayType`) selected by a nullable comptime `Key` parameter. Here a single
//! generic array is parameterized by the [`SegmentedArraySpec`] trait; `SORTED == false`
//! corresponds to `Key == null` (sorted-only methods then assert like the upstream comptime
//! checks).

// Indexes and counts mirror upstream's u32 layout; every value is bounded by
// NODE_CAPACITY / ELEMENT_COUNT_MAX <= u32::MAX, so truncation cannot occur.
#![allow(clippy::cast_possible_truncation)]

use crate::binary_search::{Config, binary_search_values_upsert_index};
use core::fmt::Debug;

/// Static description of a segmented array instantiation (upstream comptime parameters).
pub trait SegmentedArraySpec {
    type Value: Copy + Default + Debug + PartialEq;
    /// Unused when [`SORTED`](Self::SORTED) is `false`; pick `Value` itself.
    type Key: Ord + Copy + Debug;
    /// Whether elements are kept ordered by `key_from_value`.
    const SORTED: bool;
    /// Maximum number of live elements (upstream `element_count_max`).
    const ELEMENT_COUNT_MAX: u32;
    /// Elements per node; must be even and `> 2`-friendly — see
    /// [`SegmentedArray::node_capacity_for`].
    const NODE_CAPACITY: usize;
    /// Smallest possible key (upstream `minInt(Key)` in tests/search edge cases).
    const KEY_MIN: Self::Key;
    /// Greatest possible key (upstream `maxInt(Key)`).
    const KEY_MAX: Self::Key;
    /// Extracts the sort key (identity when unsorted).
    fn key_from_value(value: &Self::Value) -> Self::Key;
}

/// Position of an element: which node, and the index within that node
/// (upstream `SegmentedArray.Cursor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cursor {
    pub node: u32,
    pub relative_index: u32,
}

/// Computes the number of whole elements that fit in a node of `node_size` bytes, rounded
/// down to an even capacity so that nodes split/join at the midpoint (upstream
/// `SegmentedArrayBaseType.node_capacity`).
///
/// # Panics
/// Panics if the resulting capacity is smaller than 2 (upstream asserts).
#[must_use]
pub const fn node_capacity_for(node_size: usize, element_size: usize) -> usize {
    let max = node_size / element_size;
    let capacity = if max.is_multiple_of(2) { max } else { max - 1 };
    assert!(capacity >= 2);
    capacity
}

/// Number of nodes needed in the worst case, where every node is half full
/// (upstream `node_count_max_naive`/`node_count_max` pair; see upstream comments for why the
/// naive bound is unreachable in some configurations).
#[must_use]
pub const fn node_count_max(element_count_max: u32, node_capacity: usize) -> usize {
    let elements_per_node_min = node_capacity / 2;
    // Upstream works in usize; ELEMENT_COUNT_MAX <= u32::MAX always fits.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let element_count_max_usize = element_count_max as usize;
    let naive = element_count_max_usize.div_ceil(elements_per_node_min);

    // To split one more node we need: the node being split full, at least one element in the
    // last node, every other node at least half full, plus the incoming element.
    let split_budget = node_capacity + 1 + (naive.saturating_sub(3) * (node_capacity / 2)) + 1;
    if element_count_max_usize >= split_budget { naive } else { naive - 1 }
}

pub struct SegmentedArray<S: SegmentedArraySpec> {
    node_count: u32,
    /// Live nodes, `self.nodes[i].len() == NODE_CAPACITY` for all `i < node_count`.
    nodes: Vec<Box<[S::Value]>>,
    /// Prefix sums: `indexes[i]` is the absolute index of node `i`'s first element.
    /// The extra last slot holds the total element count.
    indexes: Vec<u32>,
    node_count_max: usize,
}

impl<S: SegmentedArraySpec> SegmentedArray<S> {
    /// Upstream `init`.
    ///
    /// # Panics
    /// Panics if `ELEMENT_COUNT_MAX <= NODE_CAPACITY` ("we should be using a non-segmented
    /// array").
    #[must_use]
    pub fn new() -> Self {
        // (Upstream also asserts ELEMENT_COUNT_MAX <= maxInt(u32), which is the type's bound.)
        assert!(S::ELEMENT_COUNT_MAX > S::NODE_CAPACITY as u32);

        let node_count_max = node_count_max(S::ELEMENT_COUNT_MAX, S::NODE_CAPACITY);
        Self {
            node_count: 0,
            nodes: Vec::new(),
            indexes: vec![0; node_count_max + 1],
            node_count_max,
        }
    }

    /// Highest number of nodes this array can ever hold.
    #[must_use]
    pub const fn node_count_max(&self) -> usize {
        self.node_count_max
    }

    /// Empties the array, releasing all nodes (upstream `reset`).
    pub fn reset(&mut self) {
        self.verify();

        self.nodes.clear();
        self.node_count = 0;
        self.indexes.fill(0);

        self.verify();
    }

    /// Checks structural invariants (upstream `verify`; upstream gates this behind an option,
    /// this port always verifies — matching upstream's `constants.verify` policy).
    ///
    /// # Panics
    /// Panics on invariant violation.
    pub fn verify(&self) {
        assert!(self.node_count as usize <= self.node_count_max);
        assert_eq!(self.nodes.len(), self.node_count as usize);
        for node_index in 0..self.node_count as usize {
            let c = self.count(node_index as u32);
            // Every node is at most full.
            assert!(c as usize <= S::NODE_CAPACITY);
            // Every node is at least half-full, except the last.
            if node_index < self.node_count as usize - 1 {
                assert!(c as usize >= S::NODE_CAPACITY / 2);
            }
        }
        if S::SORTED {
            // Elements must be sorted by key_from_value (but not necessarily unique).
            let mut key_prior_or_none: Option<S::Key> = None;
            for node_index in 0..self.node_count {
                for value in self.node_elements(node_index) {
                    let key = S::key_from_value(value);
                    if let Some(key_prior) = key_prior_or_none {
                        assert!(key_prior <= key);
                    }
                    key_prior_or_none = Some(key);
                }
            }
        }
    }

    /// Elements in node `node` (derived from the prefix-sum indexes).
    ///
    /// # Panics
    /// Panics if `node >= node_count`.
    #[must_use]
    pub(crate) fn count(&self, node: u32) -> u32 {
        let result = self.indexes[node as usize + 1] - self.indexes[node as usize];
        assert!(result as usize <= S::NODE_CAPACITY);
        result
    }

    fn increment_indexes_after(&mut self, node: u32, delta: u32) {
        for index in &mut self.indexes[node as usize + 1..=self.node_count as usize] {
            *index += delta;
        }
    }

    fn decrement_indexes_after(&mut self, node: u32, delta: u32) {
        for index in &mut self.indexes[node as usize + 1..=self.node_count as usize] {
            *index -= delta;
        }
    }

    /// Inserts one element in key order, returning its absolute index
    /// (sorted arrays only, upstream `insert_element`).
    ///
    /// # Panics
    /// Panics if the spec is unsorted, or on invariant violation.
    pub fn insert_element(&mut self, element: S::Value) -> u32 {
        assert!(S::SORTED);
        self.verify();

        let count_before = self.len();

        let cursor = self.search(S::key_from_value(&element));
        let absolute_index = self.absolute_index_for_cursor(cursor);
        self.insert_elements_at_absolute_index(absolute_index, &[element]);

        self.verify();

        assert_eq!(self.len(), count_before + 1);

        absolute_index
    }

    /// Inserts `elements` at `absolute_index` (unsorted arrays only,
    /// upstream `insert_elements`).
    ///
    /// # Panics
    /// Panics if the spec is sorted, or on invariant violation.
    pub fn insert_elements(&mut self, absolute_index: u32, elements: &[S::Value]) {
        assert!(!S::SORTED);
        self.verify();

        let count_before = self.len();
        self.insert_elements_at_absolute_index(absolute_index, elements);

        assert_eq!(self.len(), count_before + elements.len() as u32);

        self.verify();
    }

    fn insert_elements_at_absolute_index(&mut self, absolute_index: u32, elements: &[S::Value]) {
        assert!(!elements.is_empty());
        assert!(absolute_index + elements.len() as u32 <= S::ELEMENT_COUNT_MAX);

        let mut i = 0_usize;
        while i < elements.len() {
            let batch = S::NODE_CAPACITY.min(elements.len() - i);
            self.insert_elements_batch(absolute_index + i as u32, &elements[i..i + batch]);
            i += batch;
        }
        assert_eq!(i, elements.len());
    }

    fn insert_elements_batch(&mut self, absolute_index: u32, elements: &[S::Value]) {
        assert!(!elements.is_empty());
        assert!(elements.len() <= S::NODE_CAPACITY);
        assert!(absolute_index + elements.len() as u32 <= S::ELEMENT_COUNT_MAX);

        if self.node_count == 0 {
            assert_eq!(absolute_index, 0);

            self.insert_empty_node_at(0);

            assert_eq!(self.node_count, 1);
            assert_eq!(self.indexes[0], 0);
            assert_eq!(self.indexes[1], 0);
        }

        let cursor = self.cursor_for_absolute_index(absolute_index);
        assert!(cursor.node < self.node_count);

        let a = cursor.node as usize;
        let a_count = self.count(cursor.node) as usize;
        assert!(cursor.relative_index as usize <= a_count);

        let total = a_count + elements.len();
        if total <= S::NODE_CAPACITY {
            let node = &mut self.nodes[a];
            node.copy_within(
                cursor.relative_index as usize..a_count,
                cursor.relative_index as usize + elements.len(),
            );
            node[cursor.relative_index as usize..cursor.relative_index as usize + elements.len()]
                .copy_from_slice(elements);

            self.increment_indexes_after(cursor.node, elements.len() as u32);
            return;
        }

        // Insert a new node after the node being split.
        let b = a + 1;
        self.insert_empty_node_at(b);

        let a_half = total.div_ceil(2);
        let b_half = total - a_half;
        assert!(a_half >= b_half);
        assert_eq!(a_half + b_half, total);

        // See upstream for the exhaustive case commentary; the three moves below place:
        // [a_prefix | elements | a_suffix] into the virtual buffer [a_half | b_half].
        let rel = cursor.relative_index as usize;

        // Move part of `a` forwards (past the insertion point) to make space for elements.
        let suffix: Vec<S::Value> = self.nodes[a][rel..a_count].to_vec();
        {
            let (a_buf, b_buf) = Self::two_nodes(&mut self.nodes, a, b);
            let (a_half_buf, _) = a_buf.split_at_mut(a_half);
            let (b_half_buf, _) = b_buf.split_at_mut(b_half);
            copy_backwards_two(a_half_buf, b_half_buf, rel + elements.len(), &suffix);
        }

        // Move the part of `a` before the insertion point but past the halfway mark into `b`.
        if a_half < rel {
            let head_tail: Vec<S::Value> = self.nodes[a][a_half..rel].to_vec();
            self.nodes[b][..head_tail.len()].copy_from_slice(&head_tail);
        }

        // Place the inserted elements.
        {
            let (a_buf, b_buf) = Self::two_nodes(&mut self.nodes, a, b);
            let (a_half_buf, _) = a_buf.split_at_mut(a_half);
            let (b_half_buf, _) = b_buf.split_at_mut(b_half);
            copy_backwards_two(a_half_buf, b_half_buf, rel, elements);
        }

        self.indexes[b] = self.indexes[a] + a_half as u32;
        self.increment_indexes_after(b as u32, elements.len() as u32);
    }

    /// Borrows two distinct nodes simultaneously (they are separate heap allocations).
    fn two_nodes(
        nodes: &mut [Box<[S::Value]>],
        a: usize,
        b: usize,
    ) -> (&mut [S::Value], &mut [S::Value]) {
        assert!(a < b);
        let (left, right) = nodes.split_at_mut(b);
        (&mut left[a], &mut right[0])
    }

    /// Insert an empty node at index `node`.
    ///
    /// # Panics
    /// Panics if the node budget is exhausted (upstream asserts).
    fn insert_empty_node_at(&mut self, node: usize) {
        assert!(node <= self.node_count as usize);
        assert!(self.node_count < self.node_count_max as u32);

        self.indexes.push(0);
        self.indexes.copy_within(node..=self.node_count as usize, node + 1);

        self.nodes.insert(node, vec![S::Value::default(); S::NODE_CAPACITY].into_boxed_slice());

        self.node_count += 1;
        assert_eq!(self.indexes[node], self.indexes[node + 1]);
    }

    /// Removes `remove_count` elements starting at `absolute_index`
    /// (upstream `remove_elements`).
    ///
    /// # Panics
    /// Panics if the array is empty or the range is out of bounds (upstream asserts).
    pub fn remove_elements(&mut self, absolute_index: u32, remove_count: u32) {
        self.verify();

        assert!(self.node_count > 0);
        assert!(remove_count > 0);
        assert!(absolute_index + remove_count <= S::ELEMENT_COUNT_MAX);
        assert!(absolute_index + remove_count <= self.indexes[self.node_count as usize]);

        let half = S::NODE_CAPACITY / 2;

        let mut i = remove_count;
        while i > 0 {
            let batch = (half as u32).min(i);
            self.remove_elements_batch(absolute_index, batch);
            i -= batch;
        }

        self.verify();
    }

    fn remove_elements_batch(&mut self, absolute_index: u32, remove_count: u32) {
        assert!(self.node_count > 0);

        // Restricting the batch size to half node capacity ensures that elements
        // are removed from at most two nodes.
        let half = S::NODE_CAPACITY / 2;
        assert!(remove_count as usize <= half);
        assert!(remove_count > 0);

        assert!(absolute_index + remove_count <= S::ELEMENT_COUNT_MAX);
        assert!(
            absolute_index + remove_count <= self.indexes[self.node_count as usize],
            "removal past the end of the array"
        );

        let cursor = self.cursor_for_absolute_index(absolute_index);
        assert!(cursor.node < self.node_count);

        let a = cursor.node as usize;
        let a_count = self.count(cursor.node) as usize;
        let a_remaining = cursor.relative_index as usize;

        // Remove elements from exactly one node:
        if a_remaining + remove_count as usize <= a_count {
            self.nodes[a].copy_within(a_remaining + remove_count as usize..a_count, a_remaining);

            self.decrement_indexes_after(cursor.node, remove_count);

            self.maybe_remove_or_merge_node_with_next(a);
            return;
        }

        // Remove elements from exactly two nodes:
        let b = a + 1;
        let b_count = self.count(b as u32) as usize;
        let offset_in_b = remove_count as usize - (a_count - a_remaining);
        // The removal crosses the a/b boundary, so it must consume part of `a`.
        assert!(offset_in_b > 0);
        // Logical contents of `b` after the removal.
        let b_remaining: Vec<S::Value> = self.nodes[b][offset_in_b..b_count].to_vec();

        // Only one of these nodes may become empty, as we limit batch size to
        // half node capacity.
        assert!(a_remaining > 0 || !b_remaining.is_empty());

        if a_remaining >= half {
            self.nodes[b][..b_remaining.len()].copy_from_slice(&b_remaining);

            self.indexes[b] = self.indexes[a] + a_remaining as u32;
            self.decrement_indexes_after(b as u32, remove_count);

            self.maybe_remove_or_merge_node_with_next(b);
        } else if b_remaining.len() >= half {
            assert!(a_remaining < half);

            self.indexes[b] = self.indexes[a] + a_remaining as u32;
            self.decrement_indexes_after(b as u32, remove_count);

            self.maybe_merge_nodes(a, &b_remaining, false);
        } else {
            assert!(a_remaining < half && b_remaining.len() < half);
            assert!(a_remaining + b_remaining.len() <= S::NODE_CAPACITY);

            self.nodes[a][a_remaining..a_remaining + b_remaining.len()]
                .copy_from_slice(&b_remaining);

            self.indexes[b] = self.indexes[a] + (a_remaining + b_remaining.len()) as u32;
            self.decrement_indexes_after(b as u32, remove_count);

            self.remove_empty_node_at(b);

            // Either:
            // * `b` was the last node so now `a` is the last node
            // * both `a` and `b` were at least half-full so now `a` is at least half-full
            assert!(b as u32 == self.node_count || self.count(a as u32) as usize >= half);
        }
    }

    fn maybe_remove_or_merge_node_with_next(&mut self, node: usize) {
        assert!(node < self.node_count as usize);

        if self.count(node as u32) == 0 {
            self.remove_empty_node_at(node);
            return;
        }

        if node == self.node_count as usize - 1 {
            return;
        }

        let next_count = self.count(node as u32 + 1) as usize;
        let next_elements: Vec<S::Value> = self.nodes[node + 1][..next_count].to_vec();
        self.maybe_merge_nodes(node, &next_elements, true);
    }

    /// Attempts to join node `a` with the (logical) contents of node `a + 1`.
    ///
    /// `in_place` mirrors upstream's `b_pointer == b_elements.ptr` pointer check: `true` when
    /// the passed contents already sit at the front of node `a + 1` (the do-nothing fast path
    /// relies on it).
    fn maybe_merge_nodes(&mut self, a: usize, b_elements: &[S::Value], in_place: bool) {
        let half = S::NODE_CAPACITY / 2;

        let a_count = self.count(a as u32) as usize;
        assert!(a_count <= S::NODE_CAPACITY);

        let b = a + 1;
        let b_count = self.count(b as u32) as usize;
        assert_eq!(b_elements.len(), b_count);
        assert!(!b_elements.is_empty());
        assert!(b_elements.len() >= half || b == self.node_count as usize - 1);
        assert!(b_elements.len() <= S::NODE_CAPACITY);

        // Our function would still be correct if this assert fails, but we would
        // unnecessarily copy all elements of b to node a and then delete b
        // instead of simply deleting a.
        assert!(!(a_count == 0 && in_place));

        let total = a_count + b_elements.len();
        if total <= S::NODE_CAPACITY {
            self.nodes[a][a_count..total].copy_from_slice(b_elements);

            self.indexes[b] = self.indexes[b + 1];
            self.remove_empty_node_at(b);

            assert!(self.count(a as u32) as usize >= half || a == self.node_count as usize - 1);
        } else if a_count < half {
            let a_half = total.div_ceil(2);
            let b_half = total - a_half;
            assert!(a_half >= b_half);
            assert_eq!(a_half + b_half, total);

            // Fill the rest of `a` from the front of `b_elements`...
            self.nodes[a][a_count..a_half].copy_from_slice(&b_elements[..a_half - a_count]);
            // ...and shift what remains of `b_elements` to the front of node `b`.
            self.nodes[b][..b_half].copy_from_slice(&b_elements[a_half - a_count..]);

            self.indexes[b] = self.indexes[a] + a_half as u32;

            assert!(self.count(a as u32) as usize >= half);
            assert!(self.count(b as u32) as usize >= half);
        } else {
            assert!(in_place);
            assert_eq!(self.indexes[b] + b_elements.len() as u32, self.indexes[b + 1]);
        }
    }

    /// Remove an empty node at index `node`.
    fn remove_empty_node_at(&mut self, node: usize) {
        assert!(self.node_count > 0);
        assert!(node < self.node_count as usize);
        assert_eq!(self.count(node as u32), 0);

        self.nodes.remove(node);
        self.indexes.copy_within(node + 1..=self.node_count as usize, node);
        self.indexes.pop();

        self.node_count -= 1;
    }

    /// The elements stored in node `node`.
    ///
    /// # Panics
    /// Panics if `node >= node_count` (upstream asserts).
    #[must_use]
    pub fn node_elements(&self, node: u32) -> &[S::Value] {
        assert!(node < self.node_count);
        &self.nodes[node as usize][..self.count(node) as usize]
    }

    /// The last element stored in node `node`.
    #[must_use]
    pub fn node_last_element(&self, node: u32) -> S::Value {
        self.node_elements(node)[self.count(node) as usize - 1]
    }

    /// The element at `cursor`.
    #[must_use]
    pub fn element_at_cursor(&self, cursor: Cursor) -> S::Value {
        self.node_elements(cursor.node)[cursor.relative_index as usize]
    }

    /// Cursor to the very first element.
    #[must_use]
    pub const fn first(&self) -> Cursor {
        Cursor { node: 0, relative_index: 0 }
    }

    /// Cursor to the very last element (equals [`Self::first`] when empty).
    #[must_use]
    pub fn last(&self) -> Cursor {
        if self.node_count == 0 {
            return self.first();
        }

        Cursor { node: self.node_count - 1, relative_index: self.count(self.node_count - 1) - 1 }
    }

    /// Number of live nodes.
    #[must_use]
    pub const fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Total number of live elements.
    ///
    /// # Panics
    /// Panics if the stored total exceeds `ELEMENT_COUNT_MAX` (upstream asserts).
    #[must_use]
    pub fn len(&self) -> u32 {
        let result = self.indexes[self.node_count as usize];
        assert!(result <= S::ELEMENT_COUNT_MAX);
        result
    }

    /// True when the array holds no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// Absolute index of the element (or insertion point) at `cursor`.
    ///
    /// # Panics
    /// Panics on out-of-range cursors (upstream asserts).
    #[must_use]
    pub fn absolute_index_for_cursor(&self, cursor: Cursor) -> u32 {
        if self.node_count == 0 {
            assert_eq!(cursor.node, 0);
            assert_eq!(cursor.relative_index, 0);
            return 0;
        }
        assert!(cursor.node < self.node_count);
        if cursor.node == self.node_count - 1 {
            // Insertion may target the index one past the end of the array.
            assert!(cursor.relative_index <= self.count(cursor.node));
        } else {
            assert!(cursor.relative_index < self.count(cursor.node));
        }
        self.indexes[cursor.node as usize] + cursor.relative_index
    }

    /// Inverse of [`Self::absolute_index_for_cursor`] (internal; exposed for tests).
    ///
    /// # Panics
    /// Panics if the array is empty or `absolute_index` is out of range (upstream asserts).
    pub(crate) fn cursor_for_absolute_index(&self, absolute_index: u32) -> Cursor {
        // This function could handle node_count == 0 by returning a zero Cursor.
        // However, this is an internal function and we don't require this behavior.
        assert!(self.node_count > 0);

        assert!(absolute_index < S::ELEMENT_COUNT_MAX);
        assert!(absolute_index <= self.len());

        // Find the first node whose start index is >= absolute_index.
        let starts = &self.indexes[0..self.node_count as usize];
        let index = starts.partition_point(|&start| start < absolute_index);

        if starts.get(index) == Some(&absolute_index) {
            Cursor { node: index as u32, relative_index: 0 }
        } else {
            let node = index - 1;
            let relative_index = absolute_index - self.indexes[node];
            if node == self.node_count as usize - 1 {
                // Insertion may target the index one past the end of the array.
                assert!(relative_index <= self.count(node as u32));
            } else {
                assert!(relative_index < self.count(node as u32));
            }
            Cursor { node: node as u32, relative_index }
        }
    }

    /// Iterates elements starting at `cursor`, in `direction`
    /// (upstream `iterator_from_cursor`).
    ///
    /// # Panics
    /// Panics if `cursor` is out of range for a non-empty array (upstream asserts).
    #[must_use]
    pub fn iterator_from_cursor(
        &self,
        cursor: Cursor,
        direction: Direction,
    ) -> SegmentedArrayIterator<'_, S> {
        if self.node_count == 0 {
            assert_eq!(cursor.node, 0);
            assert_eq!(cursor.relative_index, 0);

            SegmentedArrayIterator {
                array: self,
                direction,
                cursor: Cursor { node: 0, relative_index: 0 },
                done: true,
            }
        } else if cursor.node == self.node_count - 1
            && cursor.relative_index == self.count(cursor.node)
        {
            match direction {
                Direction::Ascending => {
                    SegmentedArrayIterator { array: self, direction, cursor, done: true }
                }
                Direction::Descending => SegmentedArrayIterator {
                    array: self,
                    direction,
                    cursor: Cursor { node: cursor.node, relative_index: cursor.relative_index - 1 },
                    done: false,
                },
            }
        } else {
            assert!(cursor.node < self.node_count);
            assert!(cursor.relative_index < self.count(cursor.node));

            SegmentedArrayIterator { array: self, direction, cursor, done: false }
        }
    }

    /// Iterates elements starting at `absolute_index`, in `direction`
    /// (upstream `iterator_from_index`).
    ///
    /// # Panics
    /// Panics if `absolute_index >= ELEMENT_COUNT_MAX`, or if it is out of range for a
    /// non-empty array (upstream asserts).
    #[must_use]
    pub fn iterator_from_index(
        &self,
        absolute_index: u32,
        direction: Direction,
    ) -> SegmentedArrayIterator<'_, S> {
        assert!(absolute_index < S::ELEMENT_COUNT_MAX);

        if self.node_count == 0 {
            assert_eq!(absolute_index, 0);

            SegmentedArrayIterator {
                array: self,
                direction,
                cursor: Cursor { node: 0, relative_index: 0 },
                done: true,
            }
        } else {
            assert!(absolute_index < self.len());

            SegmentedArrayIterator {
                array: self,
                direction,
                cursor: self.cursor_for_absolute_index(absolute_index),
                done: false,
            }
        }
    }

    /// Returns a cursor to the index of the key either exactly equal to the target key or,
    /// if there is no exact match, the next greatest key (sorted arrays only).
    ///
    /// # Panics
    /// Panics if the spec is unsorted (upstream comptime assert).
    #[must_use]
    pub fn search(&self, key: S::Key) -> Cursor {
        assert!(S::SORTED);
        if self.node_count == 0 {
            return Cursor { node: 0, relative_index: 0 };
        }

        // Binary search over the first element of each node; "round down" to the previous
        // node when the key falls between two nodes.
        let mut offset = 0_usize;
        let mut length = self.node_count as usize;
        while length > 1 {
            let half = length / 2;
            let mid = offset + half;

            let node_first = self.node_elements(mid as u32)[0];
            if S::key_from_value(&node_first) < key {
                offset = mid;
            }

            length -= half;
        }

        // Unlike a normal binary search, don't increment the offset when "key" is higher
        // than the element — "round down" to the previous node.
        // This guarantees that the node result is never "== node_count".
        //
        // (If there are two adjacent nodes starting with keys A and C, and we search B,
        // we want to pick the A node.)
        let node = offset as u32;
        assert!(node < self.node_count);

        let relative_index = binary_search_values_upsert_index(
            &S::key_from_value,
            self.node_elements(node),
            key,
            Config::default(),
        );

        // Follow the same rule as absolute_index_for_cursor:
        // only return relative_index==count() at the last node.
        if node + 1 < self.node_count && relative_index == self.count(node) {
            Cursor { node: node + 1, relative_index: 0 }
        } else {
            Cursor { node, relative_index }
        }
    }
}

impl<S: SegmentedArraySpec> Default for SegmentedArray<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Behaves like `mem.copyBackwards` over two destination slices treated as one contiguous
/// virtual buffer: writes `source` at `target` (an offset into `a ++ b`), moving data towards
/// higher indices. Source must not overlap either destination (the port snapshots sources
/// before calling).
fn copy_backwards_two<T: Copy>(a: &mut [T], b: &mut [T], mut cursor: usize, source: &[T]) {
    assert!(cursor + source.len() <= a.len() + b.len());

    // Fill whatever fits into `a`, then spill the rest into `b`.
    let mut remaining = source;
    if cursor < a.len() {
        let take = remaining.len().min(a.len() - cursor);
        a[cursor..cursor + take].copy_from_slice(&remaining[..take]);
        remaining = &remaining[take..];
        cursor += take;
    }
    if !remaining.is_empty() {
        let offset = cursor - a.len();
        b[offset..offset + remaining.len()].copy_from_slice(remaining);
    }
}

/// Iteration direction for segmented-array iterators (upstream `Direction` restricted to the
/// iterator API; the full enum lives in [`crate::direction`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Ascending,
    Descending,
}

pub struct SegmentedArrayIterator<'a, S: SegmentedArraySpec> {
    array: &'a SegmentedArray<S>,
    direction: Direction,
    cursor: Cursor,

    /// The user may set this early to stop iteration. For example,
    /// if the returned table info is outside the key range.
    done: bool,
}

impl<'a, S: SegmentedArraySpec> SegmentedArrayIterator<'a, S> {
    /// Returns the current element and advances, or `None` when finished
    /// (upstream `Iterator.next`).
    ///
    /// # Panics
    /// Panics if the iterator's cursor is out of range (upstream asserts).
    // DEVIATION: upstream's inherent `next` could become a std Iterator impl, but the manual
    // form keeps the port diffable against upstream.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&'a S::Value> {
        if self.done {
            return None;
        }

        assert!(self.cursor.node < self.array.node_count);

        let elements = self.array.node_elements(self.cursor.node);
        let element = &elements[self.cursor.relative_index as usize];

        match self.direction {
            Direction::Ascending => {
                if self.cursor.relative_index as usize == elements.len() - 1 {
                    if self.cursor.node == self.array.node_count - 1 {
                        self.done = true;
                    } else {
                        self.cursor.node += 1;
                        self.cursor.relative_index = 0;
                    }
                } else {
                    self.cursor.relative_index += 1;
                }
            }
            Direction::Descending => {
                if self.cursor.relative_index == 0 {
                    if self.cursor.node == 0 {
                        self.done = true;
                    } else {
                        self.cursor.node -= 1;
                        self.cursor.relative_index = self.array.count(self.cursor.node) - 1;
                    }
                } else {
                    self.cursor.relative_index -= 1;
                }
            }
        }

        Some(element)
    }

    /// Stops iteration early (upstream sets `done` directly).
    pub fn stop(&mut self) {
        self.done = true;
    }
}

#[cfg(test)]
mod tests {
    use super::{Direction::*, SegmentedArray, SegmentedArraySpec, node_capacity_for};
    use crate::binary_search::{Config, binary_search_values_upsert_index};
    use tigerbeetle_core::stdx::prng::Prng;

    /// Upstream test pool: `NodePoolType(128 * @sizeOf(u32), ...)` => capacity 128.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct DupSpec;

    impl SegmentedArraySpec for DupSpec {
        type Value = u32;
        type Key = u32;
        const SORTED: bool = true;
        const ELEMENT_COUNT_MAX: u32 = 1024;
        const NODE_CAPACITY: usize = node_capacity_for(128 * size_of::<u32>(), size_of::<u32>());
        const KEY_MIN: u32 = 0;
        const KEY_MAX: u32 = u32::MAX;
        fn key_from_value(value: &u32) -> u32 {
            *value
        }
    }

    const fn size_of<T>() -> usize {
        core::mem::size_of::<T>()
    }

    /// Upstream test "SortedSegmentedArray duplicate elements":
    /// Create [0, 0, 0, 100, 100, 100, ~0, ~0, ~0]; verify that search is left-biased.
    #[test]
    fn sorted_segmented_array_duplicate_elements() {
        let mut array = SegmentedArray::<DupSpec>::new();

        for index in 0..3_usize {
            // Elements are inserted to the left of a row of duplicates.
            let mut inserted_at = array.insert_element(0);
            assert_eq!(inserted_at, 0);

            inserted_at = array.insert_element(100);
            assert_eq!(inserted_at, index as u32 + 1);

            inserted_at = array.insert_element(u32::MAX);
            assert_eq!(inserted_at, ((index + 1) * 2) as u32);
        }
        assert_eq!(array.len(), 9);

        // Search finds the leftmost element.
        assert_eq!(array.absolute_index_for_cursor(array.search(0)), 0);
        assert_eq!(array.absolute_index_for_cursor(array.search(100)), 3);
        assert_eq!(array.absolute_index_for_cursor(array.search(u32::MAX)), 6);

        // Ascending iterators pick the leftmost element.
        // Descending iterators are weird --- they _also_ pick the leftmost element, although
        // the rightmost would make more sense.
        {
            let target = 0_u32;
            let mut it = array.iterator_from_cursor(array.search(target), Ascending);
            assert_eq!(it.next().copied(), Some(0));
            assert_eq!(it.next().copied(), Some(0));
            assert_eq!(it.next().copied(), Some(0));
            assert_eq!(it.next().copied(), Some(100));

            let mut it = array.iterator_from_cursor(array.search(target), Descending);
            assert_eq!(it.next().copied(), Some(0));
            assert_eq!(it.next().copied(), None);
        }

        {
            let target = 100_u32;
            let mut it = array.iterator_from_cursor(array.search(target), Ascending);
            assert_eq!(it.next().copied(), Some(100));
            assert_eq!(it.next().copied(), Some(100));
            assert_eq!(it.next().copied(), Some(100));
            assert_eq!(it.next().copied(), Some(u32::MAX));

            let mut it = array.iterator_from_cursor(array.search(target), Descending);
            assert_eq!(it.next().copied(), Some(100));
            assert_eq!(it.next().copied(), Some(0));
        }

        {
            let target = u32::MAX;
            let mut it = array.iterator_from_cursor(array.search(target), Ascending);
            assert_eq!(it.next().copied(), Some(u32::MAX));
            assert_eq!(it.next().copied(), Some(u32::MAX));
            assert_eq!(it.next().copied(), Some(u32::MAX));
            assert_eq!(it.next().copied(), None);

            let mut it = array.iterator_from_cursor(array.search(target), Descending);
            assert_eq!(it.next().copied(), Some(u32::MAX));
            assert_eq!(it.next().copied(), Some(100));
        }
    }

    /// Upstream `FuzzContextType`/`run_fuzz`, restricted to `u32` elements
    /// (`TableInfo` configurations depend on `table.zig`/`manifest.zig`, not yet ported).
    /// TODO(port): src/lsm/segmented_array.zig run_fuzz TableInfo configurations.
    struct FuzzContext<S: SegmentedArraySpec> {
        prng: Prng,
        array: SegmentedArray<S>,
        reference: Vec<S::Value>,
        inserts: u64,
        removes: u64,
    }

    const LOG: bool = false;

    /// Widens a raw u32 draw into the spec's value type (`u32 -> u32` here; upstream fuzz also
    /// exercises larger structs once TableInfo exists).
    fn from_u32<V: From<u32>>(value: u32) -> V {
        V::from(value)
    }

    impl<S> FuzzContext<S>
    where
        S: SegmentedArraySpec,
        S::Value: From<u32>,
        S::Key: From<u32>,
    {
        fn new(seed: &mut Prng) -> Self {
            Self {
                prng: *seed,
                array: SegmentedArray::new(),
                reference: Vec::with_capacity(S::ELEMENT_COUNT_MAX as usize),
                inserts: 0,
                removes: 0,
            }
        }

        fn finish(self) -> Prng {
            self.array.verify();
            self.prng
        }

        fn run(&mut self) {
            {
                let mut i = 0;
                while i < S::ELEMENT_COUNT_MAX as usize * 2 {
                    if self.prng.gen_int_inclusive_u64(99) < 60 {
                        self.insert();
                    } else {
                        self.remove();
                    }
                    i += 1;
                }
            }

            {
                let mut i = 0;
                while i < S::ELEMENT_COUNT_MAX as usize * 2 {
                    if self.prng.gen_int_inclusive_u64(99) < 40 {
                        self.insert();
                    } else {
                        self.remove();
                    }
                    i += 1;
                }
            }

            // Rarely, the code above won't generate an insert at all.
            if self.inserts > 0 {
                self.remove_all();
            }

            if !S::SORTED {
                // Insert at the beginning of the array until the array is full.
                while self.array.len() < S::ELEMENT_COUNT_MAX {
                    self.insert_before_first();
                }
                assert!(self.array.node_count() as usize >= self.array.node_count_max() - 1);

                // Remove all-but-one elements from the last node and insert them into the
                // first node.
                let last_node = self.array.node_count() - 1;
                let element_count_last = self.array.count(last_node);
                let mut element_index = 0;
                while element_index < element_count_last - 1 {
                    self.remove_last();
                    self.insert_before_first();
                    element_index += 1;
                }

                // We should now have maxed out our node count.
                assert_eq!(self.array.node_count() as usize, self.array.node_count_max());

                self.remove_all();
            }
        }

        fn random_values(&mut self, count_max: usize) -> Vec<S::Value> {
            // DEVIATION: upstream fills raw bytes (`prng.fill(sliceAsBytes)`); here each value
            // comes from one PRNG draw instead.
            let count = self.prng.range_inclusive_usize(1, count_max);
            (0..count)
                .map(|_| {
                    let mut bytes = [0_u8; 4];
                    self.prng.fill(&mut bytes);
                    from_u32(u32::from_le_bytes(bytes))
                })
                .collect()
        }

        fn insert(&mut self) {
            let reference_len = self.reference.len() as u32;
            let count_free = S::ELEMENT_COUNT_MAX - reference_len;

            if count_free == 0 {
                return;
            }

            let count_max = (count_free as usize).min(S::NODE_CAPACITY * 3);
            let values = self.random_values(count_max);

            let inserted_count = values.len();
            if S::SORTED {
                for value in values {
                    let index_actual = self.array.insert_element(value);
                    let index_expect = self.reference_index(S::key_from_value(&value)) as usize;
                    self.reference.insert(index_expect, value);
                    assert_eq!(index_expect as u32, index_actual);
                }
            } else {
                let index = self.prng.gen_int_inclusive_u32(reference_len) as usize;

                let snapshot = values.clone();
                self.array.insert_elements(index as u32, &snapshot);
                let tail: Vec<S::Value> = self.reference.drain(index..).collect();
                self.reference.extend_from_slice(&snapshot);
                self.reference.extend(tail);
            }
            self.inserts += inserted_count as u64;

            self.verify();
        }

        fn remove(&mut self) {
            let reference_len = self.reference.len() as u32;
            if reference_len == 0 {
                return;
            }

            let count_max = (reference_len as usize).min(S::NODE_CAPACITY * 3);
            let count = 1 + self.prng.gen_int_inclusive_u64(count_max as u64 - 1) as u32;

            assert!(self.reference.len() <= S::ELEMENT_COUNT_MAX as usize);
            let index = self.prng.gen_int_inclusive_u32(reference_len - count);

            self.array.remove_elements(index, count);

            self.reference.drain(index as usize..index as usize + count as usize);

            self.removes += u64::from(count);

            self.verify();
        }

        fn insert_before_first(&mut self) {
            assert!(!S::SORTED);

            let insert_index = self.array.absolute_index_for_cursor(self.array.first());

            let mut bytes = [0_u8; 4];
            self.prng.fill(&mut bytes);
            let element: S::Value = from_u32(u32::from_le_bytes(bytes));

            self.array.insert_elements(insert_index, &[element]);
            self.reference.insert(insert_index as usize, element);

            self.inserts += 1;

            self.verify();
        }

        fn remove_last(&mut self) {
            assert!(!S::SORTED);

            let remove_index = self.array.absolute_index_for_cursor(self.array.last());

            self.array.remove_elements(remove_index, 1);
            self.reference.remove(remove_index as usize);

            self.removes += 1;

            self.verify();
        }

        fn remove_all(&mut self) {
            while !self.reference.is_empty() {
                self.remove();
            }

            assert_eq!(self.array.len(), 0);
            assert!(self.inserts > 0);
            assert_eq!(self.inserts, self.removes);

            self.verify();
        }

        fn verify(&mut self) {
            if LOG {
                println!("expect: {:?}", self.reference);
                print!("actual: ");
                let mut it = self.array.iterator_from_index(0, Ascending);
                while let Some(i) = it.next() {
                    print!("{i:?}, ");
                }
                println!();
            }

            assert_eq!(self.reference.len(), self.array.len() as usize);

            {
                let mut it = self.array.iterator_from_index(0, Ascending);

                for expect in &self.reference {
                    let actual =
                        it.next().unwrap_or_else(|| panic!("array shorter than reference"));
                    assert_eq!(*expect, *actual);
                }
                assert!(it.next().is_none());
            }

            {
                let start = (self.reference.len() as u32).saturating_sub(1);
                let mut it = self.array.iterator_from_index(start, Descending);

                for expect in self.reference.iter().rev() {
                    let actual =
                        it.next().unwrap_or_else(|| panic!("array shorter than reference"));
                    assert_eq!(*expect, *actual);
                }
                assert!(it.next().is_none());
            }

            {
                for i in 0..self.reference.len() as u32 {
                    assert_eq!(
                        i,
                        self.array
                            .absolute_index_for_cursor(self.array.cursor_for_absolute_index(i)),
                    );
                }
            }

            if S::SORTED {
                for i in 1..self.reference.len() {
                    assert!(
                        S::key_from_value(&self.reference[i - 1])
                            <= S::key_from_value(&self.reference[i])
                    );
                }
            }

            if self.array.is_empty() {
                assert_eq!(self.array.node_count(), 0);
            }

            {
                let mut i = 0;
                let node_count = self.array.node_count();
                while i + 1 < node_count {
                    assert!(self.array.count(i) as usize >= S::NODE_CAPACITY / 2);
                    i += 1;
                }
            }
            if S::SORTED {
                self.verify_search();
            }
        }

        fn verify_search(&mut self) {
            // DEVIATION: upstream fills raw query bytes; one draw per query here.
            let mut queries = [S::KEY_MIN; 20];
            for query in &mut queries {
                let mut bytes = [0_u8; 4];
                self.prng.fill(&mut bytes);
                *query = from_u32(u32::from_le_bytes(bytes));
            }

            // Test min/max exceptional values on different SegmentedArray shapes.
            queries[0] = S::KEY_MIN;
            queries[1] = S::KEY_MAX;

            for &query in &queries {
                assert_eq!(
                    self.reference_index(query),
                    self.array.absolute_index_for_cursor(self.array.search(query)),
                );
            }

            {
                let mut iterator_end =
                    self.array.iterator_from_cursor(self.array.search(S::KEY_MAX), Ascending);
                while let Some(item) = iterator_end.next() {
                    assert_eq!(S::key_from_value(item), S::KEY_MAX);
                }
            }

            {
                // 0 is not symmetric with maxInt, because `search` doesn't take direction into
                // account.
                let mut iterator_start =
                    self.array.iterator_from_cursor(self.array.search(S::KEY_MIN), Descending);
                if self.reference.is_empty() {
                    assert!(iterator_start.next().is_none());
                } else {
                    assert!(iterator_start.next().is_some());
                    assert!(iterator_start.next().is_none());
                }
            }
        }

        fn reference_index(&self, key: S::Key) -> u32 {
            binary_search_values_upsert_index(
                &S::key_from_value,
                &self.reference,
                key,
                Config::default(),
            )
        }
    }

    macro_rules! fuzz_spec {
        ($name:ident, $sorted:expr, $count_max:expr, $capacity:expr) => {
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            struct $name;

            impl SegmentedArraySpec for $name {
                type Value = u32;
                type Key = u32;
                const SORTED: bool = $sorted;
                const ELEMENT_COUNT_MAX: u32 = $count_max;
                const NODE_CAPACITY: usize = $capacity;
                const KEY_MIN: u32 = 0;
                const KEY_MAX: u32 = u32::MAX;
                fn key_from_value(value: &u32) -> u32 {
                    *value
                }
            }
        };
    }

    fuzz_spec!(SortedCap2N3, true, 3, 2);
    fuzz_spec!(UnsortedCap2N3, false, 3, 2);
    fuzz_spec!(SortedCap2N4, true, 4, 2);
    fuzz_spec!(UnsortedCap2N4, false, 4, 2);
    fuzz_spec!(SortedCap2N5, true, 5, 2);
    fuzz_spec!(UnsortedCap2N5, false, 5, 2);
    fuzz_spec!(SortedCap2N6, true, 6, 2);
    fuzz_spec!(UnsortedCap2N6, false, 6, 2);
    fuzz_spec!(SortedCap2N1024, true, 1024, 2);
    fuzz_spec!(UnsortedCap2N1024, false, 1024, 2);
    fuzz_spec!(SortedCap4N1024, true, 1024, node_capacity_for(16 * 4, 4));
    fuzz_spec!(UnsortedCap4N1024, false, 1024, node_capacity_for(16 * 4, 4));
    fuzz_spec!(SortedCap8N1024, true, 1024, node_capacity_for(32 * 4, 4));
    fuzz_spec!(UnsortedCap8N1024, false, 1024, node_capacity_for(32 * 4, 4));
    fuzz_spec!(SortedCap16N1024, true, 1024, node_capacity_for(64 * 4, 4));
    fuzz_spec!(UnsortedCap16N1024, false, 1024, node_capacity_for(64 * 4, 4));

    /// Upstream `segmented_array_fuzz.zig` driver.
    #[test]
    #[allow(unused_assignments)] // prng threading through macro arms reads as dead on the last one
    fn segmented_array_fuzz() {
        let mut prng = Prng::from_seed(42);

        macro_rules! run_fuzz {
            ($spec:ty) => {{
                println!("run_fuzz: {}", core::any::type_name::<$spec>());
                let mut context = FuzzContext::<$spec>::new(&mut prng);
                context.run();
                prng = context.finish();
            }};
        }

        run_fuzz!(SortedCap2N3);
        run_fuzz!(UnsortedCap2N3);
        run_fuzz!(SortedCap2N4);
        run_fuzz!(UnsortedCap2N4);
        run_fuzz!(SortedCap2N5);
        run_fuzz!(UnsortedCap2N5);
        run_fuzz!(SortedCap2N6);
        run_fuzz!(UnsortedCap2N6);
        run_fuzz!(SortedCap2N1024);
        run_fuzz!(UnsortedCap2N1024);
        run_fuzz!(SortedCap4N1024);
        run_fuzz!(UnsortedCap4N1024);
        run_fuzz!(SortedCap8N1024);
        run_fuzz!(UnsortedCap8N1024);
        run_fuzz!(SortedCap16N1024);
        run_fuzz!(UnsortedCap16N1024);
    }
}
