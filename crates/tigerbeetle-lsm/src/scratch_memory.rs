//! ScratchMemory is a scratch buffer meant for situations where a buffer is required
//! (e.g., radix sort) and can be shared between components.
//!
//! Port of `src/lsm/scratch_memory.zig`.
//!
//! DEVIATION: upstream holds a page-aligned `[]u8` and reinterprets it as `[]T` on
//! `acquire(T, count)` (`stdx.bytes_as_slice`). Safe Rust cannot reinterpret borrowed
//! bytes, so the scratch is generic over its element type `T` (`ScratchMemory<Value>`)
//! and sized in elements instead of bytes; `acquire` hands out `&mut [T]` directly.
//! The page-alignment trick (a max-aligned pointer satisfies smaller alignments) is
//! subsumed by `Vec`'s natural `align_of::<T>()` alignment.

/// Upstream: `state: enum { free, busy }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Free,
    Busy,
}

/// A single-acquirer-at-a-time scratch buffer shared between components.
pub struct ScratchMemory<T> {
    values: Box<[T]>,
    state: State,
}

impl<T: Copy + Default> ScratchMemory<T> {
    /// Upstream `init(gpa, size_bytes)`; here `capacity` counts elements of `T`.
    ///
    /// # Panics
    /// Panics if `capacity == 0` (upstream asserts).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self { values: vec![T::default(); capacity].into_boxed_slice(), state: State::Free }
    }

    /// Lends out `count` elements. The borrow ends before [`Self::release`] is called
    /// (non-lexical lifetimes replace upstream's pointer-identity assertions).
    ///
    /// # Panics
    /// Panics if the scratch is not free, or if `count` exceeds the capacity
    /// (upstream asserts).
    pub fn acquire(&mut self, count: usize) -> &mut [T] {
        assert_eq!(self.state, State::Free);
        assert!(count <= self.values.len());
        self.state = State::Busy;
        &mut self.values[..count]
    }

    /// Returns the scratch to the free state.
    ///
    /// # Panics
    /// Panics if the scratch is not busy (upstream asserts).
    pub fn release(&mut self) {
        assert_eq!(self.state, State::Busy);
        self.state = State::Free;
    }

    /// Upstream exposes `state` for assertions at call sites (e.g., `TableMemory.init`).
    #[must_use]
    pub const fn is_free(&self) -> bool {
        matches!(self.state, State::Free)
    }
}

#[cfg(test)]
mod tests {
    use super::ScratchMemory;

    #[test]
    fn scratch_memory_basic() {
        let mut scratch: ScratchMemory<u64> = ScratchMemory::new(10);

        let slice = scratch.acquire(10);
        for (n, slot) in slice.iter_mut().enumerate() {
            *slot = n as u64;
        }
        // Slice borrow ends here; release asserts the busy->free transition.
        scratch.release();

        // Re-acquire sees the previous contents (buffer is reused, like upstream):
        let slice = scratch.acquire(5);
        assert_eq!(&slice[..3], &[0_u64, 1, 2]);
    }
}
