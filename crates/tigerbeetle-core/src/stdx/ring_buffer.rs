//! A First In, First Out ring buffer.
//!
//! Port of `src/stdx/ring_buffer.zig`.
//!
//! DEVIATION: upstream is generic over storage (compile-time array vs runtime slice); this port
//! uses a single `Vec<Option<T>>`-backed buffer with runtime capacity for both cases, since safe
//! Rust cannot leave slots uninitialized. Slots hold `Option<T>`; the capacity is fixed after
//! construction (`with_capacity`). Pointer-returning upstream functions map to reference
//! accessors, and the manual iterators map to `iter()`/`iter_mut()`.

use std::fmt;

/// Upstream `error.NoSpaceLeft`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoSpaceLeft;

impl fmt::Display for NoSpaceLeft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NoSpaceLeft")
    }
}

impl std::error::Error for NoSpaceLeft {}

/// A First In, First Out ring buffer with a fixed capacity.
#[derive(Clone)]
pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,

    /// The index of the slot with the first item, if any.
    index: usize,

    /// The number of items in the buffer.
    count: usize,
}

impl<T> RingBuffer<T> {
    /// Port of `init_slice`; the array-backed variant collapses into this as well.
    ///
    /// # Panics
    /// Panics if `capacity == 0` (upstream asserts `capacity > 0`).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self { buffer: (0..capacity).map(|_| None).collect(), index: 0, count: 0 }
    }

    /// The maximum number of items in the buffer (upstream `count_max`/`buffer.len`).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub fn clear(&mut self) {
        // Drop all items so that non-Copy values are released deterministically:
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.index = 0;
        self.count = 0;
    }

    #[must_use]
    pub fn head(&self) -> Option<&T> {
        if self.empty() {
            return None;
        }
        self.buffer[self.index].as_ref()
    }

    #[must_use]
    pub fn head_mut(&mut self) -> Option<&mut T> {
        if self.empty() {
            return None;
        }
        self.buffer[self.index].as_mut()
    }

    #[must_use]
    pub fn tail(&self) -> Option<&T> {
        if self.empty() {
            return None;
        }
        let slot = (self.index + self.count - 1) % self.capacity();
        self.buffer[slot].as_ref()
    }

    #[must_use]
    pub fn tail_mut(&mut self) -> Option<&mut T> {
        if self.empty() {
            return None;
        }
        let slot = (self.index + self.count - 1) % self.capacity();
        self.buffer[slot].as_mut()
    }

    /// Returns the `index`th item in the buffer (from the head), or `None` past the end.
    /// Unlike upstream, an out-of-bounds `index >= capacity` also returns `None` instead of
    /// asserting.
    ///
    /// # Panics
    /// Panics if the buffer has zero capacity (impossible via [`Self::with_capacity`]).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if self.capacity() == 0 {
            unreachable!("zero-capacity ring buffer");
        }

        if index < self.count {
            let slot = (self.index + index) % self.capacity();
            self.buffer[slot].as_ref()
        } else {
            assert!(index < self.capacity());
            None
        }
    }

    /// # Panics
    /// Panics if the buffer has zero capacity (impossible via [`Self::with_capacity`]).
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.capacity() == 0 {
            unreachable!("zero-capacity ring buffer");
        }

        if index < self.count {
            let slot = (self.index + index) % self.capacity();
            self.buffer[slot].as_mut()
        } else {
            assert!(index < self.capacity());
            None
        }
    }

    /// Mutable access to the slot that the next `push()` would fill.
    /// Returns `None` only when the buffer is full — unlike a plain `Option<&mut T>` view of an
    /// empty slot, this distinguishes "full" from "slot currently vacant" (DEVIATION: upstream's
    /// `next_tail_ptr` hands out a pointer into uninitialized memory; here the slot is
    /// `Option<T>`).
    #[must_use]
    pub fn next_tail_slot_mut(&mut self) -> Option<&mut Option<T>> {
        if self.full() {
            return None;
        }
        let slot = (self.index + self.count) % self.capacity();
        self.buffer.get_mut(slot)
    }

    pub fn advance_head(&mut self) {
        self.index += 1;
        self.index %= self.capacity();
        self.count -= 1;
    }

    /// # Panics
    /// Panics if `discard > count` (upstream asserts the same).
    pub fn advance_head_many(&mut self, discard: usize) {
        assert!(discard <= self.count);

        self.index += discard;
        self.index %= self.capacity();
        self.count -= discard;
    }

    /// # Panics
    /// Panics if the buffer is full.
    pub fn retreat_head(&mut self) {
        assert!(self.count < self.capacity());

        self.index += self.capacity() - 1;
        self.index %= self.capacity();
        self.count += 1;
    }

    /// # Panics
    /// Panics if the buffer is full.
    pub fn advance_tail(&mut self) {
        assert!(self.count < self.capacity());
        self.count += 1;
    }

    /// # Panics
    /// Panics if the buffer is empty.
    pub fn retreat_tail(&mut self) {
        self.count -= 1;
    }

    /// Returns whether the ring buffer is completely full.
    #[must_use]
    pub fn full(&self) -> bool {
        self.count == self.capacity()
    }

    #[must_use]
    pub fn spare_capacity(&self) -> usize {
        self.capacity() - self.count
    }

    /// Returns whether the ring buffer is completely empty.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.count == 0
    }

    // Higher level, less error-prone wrappers:

    /// Add an element at the head. Returns [`NoSpaceLeft`] if the buffer is already full.
    ///
    /// # Errors
    /// Returns [`NoSpaceLeft`] when `count == capacity`.
    pub fn push_head(&mut self, item: T) -> Result<(), NoSpaceLeft> {
        if self.count == self.capacity() {
            return Err(NoSpaceLeft);
        }
        self.push_head_assume_capacity(item);
        Ok(())
    }

    /// # Panics
    /// Panics if the buffer is full.
    pub fn push_head_assume_capacity(&mut self, item: T) {
        assert!(self.count < self.capacity());

        self.retreat_head();
        let slot = self.index;
        self.buffer[slot] = Some(item);
    }

    /// Add an element to the ring buffer. Returns [`NoSpaceLeft`] if the buffer
    /// is already full and the element could not be added.
    ///
    /// # Errors
    /// Returns [`NoSpaceLeft`] when the buffer is full.
    pub fn push(&mut self, item: T) -> Result<(), NoSpaceLeft> {
        match self.next_tail_slot_mut() {
            Some(slot) => {
                *slot = Some(item);
                self.advance_tail();
                Ok(())
            }
            None => Err(NoSpaceLeft),
        }
    }

    /// Add an element to the ring buffer, asserting that the capacity is sufficient.
    ///
    /// # Panics
    /// Panics if the buffer is full.
    pub fn push_assume_capacity(&mut self, item: T) {
        assert!(self.count < self.capacity());
        let pushed = self.push(item);
        assert!(pushed.is_ok());
    }

    /// Append elements, wrapping across the internal buffer boundary.
    /// Requires `T: Clone` in place of upstream's disjoint copy.
    ///
    /// # Errors
    /// Returns [`NoSpaceLeft`] when all of `items` do not fit in the spare capacity.
    pub fn push_slice(&mut self, items: &[T]) -> Result<(), NoSpaceLeft>
    where
        T: Clone,
    {
        if self.capacity() == 0 {
            return Err(NoSpaceLeft);
        }
        if self.count + items.len() > self.capacity() {
            return Err(NoSpaceLeft);
        }

        for item in items {
            self.push_assume_capacity(item.clone());
        }
        Ok(())
    }

    /// Remove and return the next item, if any.
    pub fn pop(&mut self) -> Option<T> {
        if self.empty() {
            return None;
        }
        let slot = self.index;
        self.advance_head();
        self.buffer[slot].take()
    }

    /// Remove and return the last item, if any.
    pub fn pop_tail(&mut self) -> Option<T> {
        if self.empty() {
            return None;
        }
        let slot = (self.index + self.count - 1) % self.capacity();
        self.retreat_tail();
        self.buffer[slot].take()
    }

    /// Returns an iterator through all `count` items.
    /// The iterator is invalidated if the ring buffer is advanced.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { ring: self, pos: 0 }
    }
}

impl<T: fmt::Debug> fmt::Debug for RingBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Iterator over the live items of a [`RingBuffer`] (upstream `Iterator`).
pub struct Iter<'a, T> {
    ring: &'a RingBuffer<T>,
    pos: usize,
}

impl<'a, T> IntoIterator for &'a RingBuffer<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(self.pos <= self.ring.count);
        if self.pos == self.ring.count {
            return None;
        }
        let slot = (self.ring.index + self.pos) % self.ring.capacity();
        self.pos += 1;
        self.ring.buffer[slot].as_ref()
    }
}

// Upstream also has `IteratorMutable`; a safe Rust equivalent cannot implement
// `std::iter::Iterator` (yielding `&mut T`) without `unsafe`. Mutate while iterating by index:
//
//   let mut position = 0;
//   while let Some(item) = ring.get_mut(position) { /* … */ position += 1; }

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of upstream `test_iterator` (mutable half drives mutation by index).
    fn test_iterator(ring: &mut RingBuffer<u32>, values: &[u32]) {
        let ring_index = ring.index;

        for _ in 0..2 {
            let collected: Vec<u32> = ring.iter().copied().collect();
            assert_eq!(&collected, values);
        }

        // Upstream mutates through `IteratorMutable.next_ptr()`; here we mutate by index.
        let permutation = u32::MAX / 2;
        for ((position, value), delta) in values.iter().enumerate().zip(0u32..) {
            let Some(slot) = ring.get_mut(position) else {
                panic!("live item");
            };
            assert_eq!(*value, *slot);
            *slot += permutation + delta;
        }
        for ((check_index, value), delta) in values.iter().enumerate().zip(0u32..) {
            let Some(slot) = ring.get_mut(check_index) else {
                panic!("live item");
            };
            assert_eq!(value + permutation + delta, *slot);
            *slot -= permutation + delta;
        }

        assert_eq!(ring_index, ring.index);
    }

    /// Port of upstream `test_low_level_interface`.
    fn test_low_level_interface(ring: &mut RingBuffer<u32>) {
        assert_eq!(ring.push_slice(&[]), Ok(()));
        test_iterator(ring, &[]);

        assert_eq!(Err(NoSpaceLeft), ring.push_slice(&[1, 2, 3]));

        assert_eq!(ring.push_slice(&[1]), Ok(()));
        assert_eq!(Some(&1), ring.tail());
        ring.advance_head();

        assert_eq!(1, ring.index);
        assert_eq!(0, ring.count);
        assert_eq!(ring.push_slice(&[1, 2]), Ok(()));
        test_iterator(ring, &[1, 2]);
        ring.advance_head();
        ring.advance_head();

        assert_eq!(1, ring.index);
        assert_eq!(0, ring.count);
        assert_eq!(ring.push_slice(&[1]), Ok(()));
        assert_eq!(Some(&1), ring.tail());
        ring.advance_head();

        assert_eq!(None, ring.head());
        assert_eq!(None, ring.tail());

        let Some(slot) = ring.next_tail_slot_mut() else { panic!("not full") };
        *slot = Some(0);
        ring.advance_tail();
        assert_eq!(Some(&0), ring.tail());
        test_iterator(ring, &[0]);

        let Some(slot) = ring.next_tail_slot_mut() else { panic!("not full") };
        *slot = Some(1);
        ring.advance_tail();
        assert_eq!(Some(&1), ring.tail());
        test_iterator(ring, &[0, 1]);

        assert_eq!(Some(&0), ring.head());
        ring.advance_head();
        test_iterator(ring, &[1]);

        let Some(slot) = ring.next_tail_slot_mut() else { panic!("not full") };
        *slot = Some(2);
        ring.advance_tail();
        assert_eq!(Some(&2), ring.tail());
        test_iterator(ring, &[1, 2]);

        ring.advance_head();
        test_iterator(ring, &[2]);

        let Some(slot) = ring.next_tail_slot_mut() else { panic!("not full") };
        *slot = Some(3);
        ring.advance_tail();
        assert_eq!(Some(&3), ring.tail());
        test_iterator(ring, &[2, 3]);

        assert_eq!(Some(&2), ring.head());
        ring.advance_head();
        test_iterator(ring, &[3]);

        assert_eq!(Some(&3), ring.head());
        ring.advance_head();
        test_iterator(ring, &[]);

        assert_eq!(None, ring.head());
        assert_eq!(None, ring.tail());
    }

    /// Port of upstream test `RingBuffer: low level interface`.
    #[test]
    fn ring_buffer_low_level_interface() {
        let mut ring = RingBuffer::<u32>::with_capacity(2);
        test_low_level_interface(&mut ring);
    }

    /// Port of upstream test `RingBuffer: push/pop high level interface`.
    #[test]
    fn ring_buffer_push_pop_high_level_interface() {
        let mut fifo = RingBuffer::<u32>::with_capacity(3);

        assert!(!fifo.full());
        assert!(fifo.empty());
        assert_eq!(None, fifo.get(0));
        assert_eq!(None, fifo.get(1));
        assert_eq!(None, fifo.get(2));

        assert_eq!(fifo.push(1), Ok(()));
        assert_eq!(Some(&1), fifo.head());
        assert_eq!(Some(&1), fifo.get(0));
        assert_eq!(None, fifo.get(1));

        assert!(!fifo.full());
        assert!(!fifo.empty());

        assert_eq!(fifo.push(2), Ok(()));
        assert_eq!(Some(&1), fifo.head());
        assert_eq!(Some(&2), fifo.get(1));

        assert_eq!(fifo.push(3), Ok(()));
        assert_eq!(Err(NoSpaceLeft), fifo.push(4));

        assert!(fifo.full());
        assert!(!fifo.empty());

        assert_eq!(Some(&1), fifo.head());
        assert_eq!(Some(1), fifo.pop());
        assert_eq!(Some(&2), fifo.get(0));
        assert_eq!(Some(&3), fifo.get(1));
        assert_eq!(None, fifo.get(2));

        assert!(!fifo.full());
        assert!(!fifo.empty());

        assert_eq!(fifo.push(4), Ok(()));

        assert_eq!(Some(2), fifo.pop());
        assert_eq!(Some(3), fifo.pop());
        assert_eq!(Some(4), fifo.pop());
        assert_eq!(None, fifo.pop());

        assert!(!fifo.full());
        assert!(fifo.empty());
    }

    /// Port of upstream test `RingBuffer: pop_tail`.
    #[test]
    fn ring_buffer_pop_tail() {
        let mut lifo = RingBuffer::<u32>::with_capacity(3);
        assert_eq!(lifo.push(1), Ok(()));
        assert_eq!(lifo.push(2), Ok(()));
        assert_eq!(lifo.push(3), Ok(()));
        assert!(lifo.full());

        assert_eq!(Some(3), lifo.pop_tail());
        assert_eq!(Some(&1), lifo.head());
        assert_eq!(Some(2), lifo.pop_tail());
        assert_eq!(Some(&1), lifo.head());
        assert_eq!(Some(1), lifo.pop_tail());
        assert_eq!(None, lifo.pop_tail());
        assert!(lifo.empty());
    }

    /// Port of upstream test `RingBuffer: push_head`.
    #[test]
    fn ring_buffer_push_head() {
        let mut ring = RingBuffer::<u32>::with_capacity(3);
        assert_eq!(ring.push_head(1), Ok(()));
        assert_eq!(ring.push(2), Ok(()));
        assert_eq!(ring.push_head(3), Ok(()));
        assert!(ring.full());

        assert_eq!(Some(3), ring.pop());
        assert_eq!(Some(1), ring.pop());
        assert_eq!(Some(2), ring.pop());
        assert!(ring.empty());
    }

    /// DEVIATION: upstream instantiates a zero-capacity array-backed buffer (`refAllDecls`);
    /// here zero capacity panics in `with_capacity`, matching upstream's slice-init assertion.
    #[test]
    #[should_panic(expected = "capacity > 0")]
    fn ring_buffer_count_max_zero() {
        let _ = RingBuffer::<u32>::with_capacity(0);
    }
}
