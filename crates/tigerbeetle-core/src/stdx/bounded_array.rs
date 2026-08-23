//! A version of standard `BoundedArray` with TigerBeetle-idiomatic APIs.
//!
//! Port of `src/stdx/bounded_array.zig`.
//!
//! DEVIATION: upstream stores a fixed `[T; N]` plus a count, leaving unused slots uninitialized.
//! Safe Rust cannot leave array elements uninitialized, so this port is backed by a `Vec<T>`
//! whose length never exceeds `N`. Consequently `unused_capacity_slice` has no equivalent
//! (upstream uses it for zero-copy fills; use [`Self::push`] here), and `get` returns a
//! reference rather than copying.

/// Upstream's `error.Overflow`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overflow;

/// A fixed-capacity array of at most `N` elements.
#[derive(Clone, Debug)]
pub struct BoundedArray<T, const N: usize> {
    buffer: Vec<T>,
}

impl<T, const N: usize> BoundedArray<T, N> {
    #[must_use]
    pub fn new() -> Self {
        Self { buffer: Vec::with_capacity(N) }
    }

    /// # Errors
    /// Returns [`Overflow`] if `items` do not fit within the capacity.
    pub fn from_slice(items: &[T]) -> Result<Self, Overflow>
    where
        T: Clone,
    {
        if items.len() <= N {
            let mut result = Self::new();
            result.push_slice(items);
            Ok(result)
        } else {
            Err(Overflow)
        }
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.buffer.len()
    }

    /// Returns count of elements in this bounded array in the specified integer type,
    /// checking at runtime that it can represent the length.
    ///
    /// # Panics
    /// Panics if the count does not fit in `U` (upstream checks this at compile time).
    #[must_use]
    pub fn count_as<U: TryFrom<usize>>(&self) -> U {
        if let Ok(count) = U::try_from(self.count()) {
            count
        } else {
            panic!("BoundedArray count does not fit in the requested integer type");
        }
    }

    #[must_use]
    pub fn full(&self) -> bool {
        self.count() == N
    }

    #[must_use]
    pub fn empty(&self) -> bool {
        self.count() == 0
    }

    /// # Panics
    /// Panics if `index >= count` (upstream asserts the same).
    #[must_use]
    pub fn get(&self, index: usize) -> &T {
        assert!(index < self.count());
        &self.buffer[index]
    }

    /// # Panics
    /// Panics if `index >= count`.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.count());
        &mut self.buffer[index]
    }

    #[must_use]
    pub fn slice(&self) -> &[T] {
        &self.buffer[..self.count()]
    }

    #[must_use]
    pub fn slice_mut(&mut self) -> &mut [T] {
        let count = self.count();
        &mut self.buffer[..count]
    }

    /// # Panics
    /// Panics if full or if `index > count` (upstream asserts the same conditions).
    pub fn insert_at(&mut self, index: usize, item: T) {
        assert!(!self.full());
        assert!(index <= self.count());
        self.buffer.insert(index, item);
    }

    /// # Panics
    /// Panics if full.
    pub fn push(&mut self, item: T) {
        assert!(!self.full());
        self.buffer.push(item);
    }

    /// # Panics
    /// Panics if the items exceed the remaining capacity.
    pub fn push_slice(&mut self, items: &[T])
    where
        T: Clone,
    {
        assert!(self.count() + items.len() <= N);
        self.buffer.extend_from_slice(items);
    }

    /// Removes and returns the element at `index`, replacing it with the last element.
    ///
    /// # Panics
    /// Panics if empty or `index >= count`.
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(self.count() > 0);
        assert!(index < self.count());
        self.buffer.swap_remove(index)
    }

    /// Removes and returns the element at `index`, shifting subsequent elements left.
    ///
    /// # Panics
    /// Panics if empty or `index >= count`.
    pub fn ordered_remove(&mut self, index: usize) -> T {
        assert!(self.count() > 0);
        assert!(index < self.count());
        self.buffer.remove(index)
    }

    /// # Errors
    /// Returns [`Overflow`] if `count_new > N`. New slots take `T::default()` when growing.
    pub fn resize(&mut self, count_new: usize) -> Result<(), Overflow>
    where
        T: Default,
    {
        if count_new > N {
            return Err(Overflow);
        }
        while self.count() < count_new {
            self.push(T::default());
        }
        Ok(())
    }

    /// # Panics
    /// Panics if `count_new > count`.
    pub fn truncate(&mut self, count_new: usize) {
        assert!(count_new <= self.count());
        self.buffer.truncate(count_new);
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop()
    }
}

impl<T, const N: usize> Default for BoundedArray<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of upstream test `BoundedArray.insert_at`.
    #[test]
    fn bounded_array_insert_at() {
        let mut array = BoundedArray::<u32, 4>::new();
        array.insert_at(0, 3);
        array.insert_at(0, 1);
        array.insert_at(1, 2);
        assert_eq!(array.slice(), &[1, 2, 3]);
        assert_eq!(array.pop(), Some(3));
        assert_eq!(array.pop(), Some(2));
        assert_eq!(array.pop(), Some(1));
        assert_eq!(array.pop(), None);

        let mut array = BoundedArray::<u32, 4>::new();
        for i in 0..4_u32 {
            array.push(i);
        }
        assert!(array.full());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            array.push(4);
        }));
        assert!(result.is_err(), "push to a full array must panic");
    }

    /// Port of upstream test `BoundedArrayType` core behaviors.
    #[test]
    fn bounded_array_core() {
        let array = BoundedArray::<u32, 4>::from_slice(&[10, 11, 12]);
        assert!(array.is_ok());

        let mut array = BoundedArray::<u32, 4>::new();
        array.push_slice(&[10, 11, 12]);
        assert_eq!(3, array.count());
        assert_eq!(3_u32, array.count_as::<u32>());
        assert!(!array.full());
        assert!(!array.empty());

        assert_eq!(12, *array.get(2));
        assert_eq!(&mut 11, array.get_mut(1));

        assert_eq!(12, array.swap_remove(2));
        assert_eq!(array.slice(), &[10, 11]);
        assert_eq!(10, array.ordered_remove(0));
        assert_eq!(array.slice(), &[11]);

        assert_eq!(Ok(()), array.resize(4));
        assert_eq!(array.slice(), &[11, 0, 0, 0]);
        array.truncate(1);
        assert_eq!(array.slice(), &[11]);
        array.clear();
        assert!(array.empty());

        let overflow = BoundedArray::<u32, 2>::from_slice(&[1, 2, 3]);
        assert_eq!(Err(Overflow), overflow.map(|_| ()));

        let mut small = BoundedArray::<u32, 2>::new();
        assert_eq!(Err(Overflow), small.resize(3));
    }
}
