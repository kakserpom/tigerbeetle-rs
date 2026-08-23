//! An intrusive last in/first out linked list (LIFO).
//!
//! Port of `src/stack.zig`.
//!
//! DEVIATION: upstream links elements by raw pointers through an embedded `link` field; safe
//! Rust forbids that aliasing, so this stack links elements by caller-chosen `u32` indices
//! instead (e.g., a slot index into a pool). Behavior — LIFO order, `count`/`capacity`
//! bookkeeping, optional duplicate-push verification — matches upstream's non-generic
//! `StackAny`.

/// A LIFO stack of element indices.
#[derive(Clone, Debug)]
pub struct Stack {
    entries: Vec<u32>,
    capacity: u32,

    /// If the number of elements is large, the verify check in [`Self::push`] can be too
    /// expensive. Allow the user to gate it.
    verify_push: bool,
}

impl Stack {
    /// Upstream `init(.{ .capacity, .verify_push })`.
    #[must_use]
    pub fn new(capacity: u32, verify_push: bool) -> Self {
        Self { entries: Vec::with_capacity(capacity as usize), capacity, verify_push }
    }

    #[must_use]
    pub fn count(&self) -> u32 {
        u32::try_from(self.entries.len()).unwrap_or_else(|_| unreachable!("len <= capacity"))
    }

    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Pushes a new node index onto the top of the stack.
    ///
    /// # Panics
    /// Panics on overflow or (when enabled) on pushing an index already contained.
    pub fn push(&mut self, node: u32) {
        if self.verify_push {
            assert!(!self.contains(node));
        }

        assert!(self.count() < self.capacity);
        self.entries.push(node);
    }

    /// Returns the top element index, and removes it.
    pub fn pop(&mut self) -> Option<u32> {
        self.entries.pop()
    }

    /// Returns the top element index, but does not remove it.
    #[must_use]
    pub fn peek(&self) -> Option<u32> {
        self.entries.last().copied()
    }

    /// Checks if the stack is empty.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns whether the stack contains the given element index.
    ///
    /// # Panics
    /// Panics if `count > capacity` (upstream asserts the same invariant).
    #[must_use]
    pub fn contains(&self, needle: u32) -> bool {
        assert!(self.count() <= self.capacity);
        self.entries.contains(&needle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdx::prng::Prng;

    /// Port of upstream test "Stack: push/pop/peek/empty".
    #[test]
    fn stack_push_pop_peek_empty() {
        let mut stack = Stack::new(3, true);

        assert!(stack.empty());

        // Push one element and verify:
        stack.push(1);
        assert!(!stack.empty());
        assert_eq!(Some(1), stack.peek());
        assert!(stack.contains(1));
        assert!(!stack.contains(2));
        assert!(!stack.contains(3));

        // Push two more elements:
        stack.push(2);
        stack.push(3);
        assert!(!stack.empty());
        assert_eq!(Some(3), stack.peek());
        assert!(stack.contains(1));
        assert!(stack.contains(2));
        assert!(stack.contains(3));

        // Pop elements and check stack order:
        assert_eq!(Some(3), stack.pop());
        assert_eq!(Some(2), stack.pop());
        assert_eq!(Some(1), stack.pop());
        assert!(stack.empty());
        assert_eq!(None, stack.pop());
    }

    /// Port of upstream test "Stack: fuzz" — compare behavior against a reference model.
    ///
    /// DEVIATION: elements are indices into a pool rather than pointers into it.
    #[test]
    fn stack_fuzz() {
        let mut prng = Prng::from_seed(0);

        let item_count_max = 1024_u32;
        let events_max = 1 << 10;

        // A bit set that tracks which nodes are available:
        let mut items_free = vec![true; item_count_max as usize];

        let mut stack = Stack::new(item_count_max, true);

        // Reference model: node IDs in stack order (last is the top).
        let mut model: Vec<u32> = Vec::with_capacity(item_count_max as usize);

        for _ in 0..events_max {
            assert!(model.len() <= item_count_max as usize);
            assert_eq!(model.len(), stack.count() as usize);
            if !model.is_empty() {
                assert!(!stack.empty());
            }

            // Upstream weights: push=2 pop=1.
            let event_push = prng.range_inclusive_usize(0, 2) < 2;
            if event_push {
                // Only push if a free node is available:
                let Some(free_index) = items_free.iter().position(|free| *free) else {
                    continue;
                };
                let free_id = u32::try_from(free_index).unwrap_or(u32::MAX);
                stack.push(free_id);
                model.push(free_id);
                items_free[free_index] = false;
            } else if let Some(item) = stack.pop() {
                // The reference model should have the same node at the top:
                let expected = model.pop();
                assert_eq!(expected, Some(item));
                items_free[item as usize] = true;
            } else {
                assert!(model.is_empty());
                assert!(stack.empty());
                assert_eq!(stack.count(), 0);
                assert_eq!(stack.peek(), None);
            }

            // Verify that peek() returns the same as the last element in our model:
            if let Some(top_ref) = model.last().copied() {
                assert_eq!(stack.peek(), Some(top_ref));
            } else {
                assert!(stack.empty());
                assert_eq!(stack.count(), 0);
                assert_eq!(stack.peek(), None);
            }
        }

        // Finally, empty the stack and ensure our reference model agrees:
        while let Some(item) = stack.pop() {
            let expected = model.pop();
            assert_eq!(expected, Some(item));
            items_free[item as usize] = true;
        }
        assert!(model.is_empty());
        assert!(stack.empty());
        assert_eq!(stack.count(), 0);
        assert_eq!(stack.peek(), None);
    }
}
