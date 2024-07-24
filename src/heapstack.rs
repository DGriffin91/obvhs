//! A stack data structure implemented on the heap with adjustable capacity.

/// A stack data structure implemented on the heap with adjustable capacity.
///
/// This structure allows pushing and popping elements and will never automatically
/// allocate or deallocate. The only functions that will result in allocation are
/// `HeapStack::new_with_capacity` and `HeapStack::reserve`.
///
/// The elements must implement the `Clone` and `Default` traits.
#[derive(Default)]
pub struct HeapStack<T: Clone + Default> {
    data: Vec<T>,
    index: usize,
}

impl<T: Clone + Default> HeapStack<T> {
    /// Creates a new `HeapStack` with the specified initial capacity.
    ///
    /// # Arguments
    /// * `cap` - The initial capacity of the stack. Must be greater than zero.
    ///
    /// # Returns
    /// A `HeapStack` with pre-allocated space for `cap` elements.
    ///
    /// # Panics
    /// This function will panic if `cap` is zero.
    #[inline(always)]
    pub fn new_with_capacity(cap: usize) -> Self {
        assert!(cap > 0);
        HeapStack {
            data: vec![Default::default(); cap],
            index: 0,
        }
    }

    /// Pushes a value onto the stack.
    ///
    /// # Arguments
    /// * `v` - The value to be pushed onto the stack.
    ///
    /// # Panics
    /// This function will panic if the stack is full.
    #[inline(always)]
    pub fn push(&mut self, v: T) {
        if self.index < self.data.len() {
            *unsafe { self.data.get_unchecked_mut(self.index) } = v;
            self.index += 1;
        } else {
            panic!("Index out of bounds: the HeapStack is full (length: {}) and cannot accommodate more elements", self.data.len());
        }
    }

    /// Pops a value from the stack.
    ///
    /// # Returns
    /// `Some(T)` if the stack is not empty, otherwise `None`.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<&T> {
        if self.index > 0 {
            self.index = self.index.saturating_sub(1);
            Some(&self.data[self.index])
        } else {
            None
        }
    }

    /// Pops a value from the stack without checking bounds.
    ///
    /// This function is safe to call because a `HeapStack` cannot have a capacity of zero.
    /// However, if the stack is empty when this function is called, it will access what was previously
    /// the first value in the stack, which may not be the expected behavior.
    ///
    /// # Returns
    /// The value at the top of the stack.
    #[inline(always)]
    pub fn pop_fast(&mut self) -> &T {
        self.index = self.index.saturating_sub(1);
        let v = unsafe { self.data.get_unchecked(self.index) };
        v
    }

    /// Returns the number of elements in the stack.
    ///
    /// # Returns
    /// The length of the stack.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.index
    }

    /// Returns true if the stack is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.index == 0
    }

    /// Clears the stack, removing all elements.
    ///
    /// This operation does not deallocate the stack's capacity.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.index = 0;
    }

    /// Reserves capacity for at least `cap` elements.
    ///
    /// # Arguments
    /// * `cap` - The desired capacity.
    /// If the new capacity is smaller than the current capacity, this function does nothing.
    #[inline(always)]
    pub fn reserve(&mut self, cap: usize) {
        if cap < self.data.len() {
            return;
        }
        self.data.resize(cap, Default::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_capacity() {
        let stack: HeapStack<i32> = HeapStack::new_with_capacity(10);
        assert_eq!(stack.len(), 0);
        assert_eq!(stack.data.len(), 10);
    }

    #[test]
    fn test_push_and_pop() {
        let mut stack: HeapStack<i32> = HeapStack::new_with_capacity(10);
        stack.push(1);
        stack.push(2);
        stack.push(3);

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.pop(), Some(&3));
        assert_eq!(stack.pop(), Some(&2));
        assert_eq!(stack.pop(), Some(&1));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds: the HeapStack is full")]
    fn test_push_panic() {
        let mut stack: HeapStack<i32> = HeapStack::new_with_capacity(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // This should panic
    }

    #[test]
    fn test_pop_fast() {
        let mut stack: HeapStack<i32> = HeapStack::new_with_capacity(10);
        stack.push(1);
        stack.push(2);
        stack.push(3);

        assert_eq!(*stack.pop_fast(), 3);
        assert_eq!(*stack.pop_fast(), 2);
        assert_eq!(*stack.pop_fast(), 1);
    }

    #[test]
    fn test_clear() {
        let mut stack: HeapStack<i32> = HeapStack::new_with_capacity(10);
        stack.push(1);
        stack.push(2);
        stack.push(3);

        stack.clear();
        assert_eq!(stack.len(), 0);
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_reserve() {
        let mut stack: HeapStack<i32> = HeapStack::new_with_capacity(5);
        assert_eq!(stack.data.len(), 5);

        stack.reserve(10);
        assert_eq!(stack.data.len(), 10);

        stack.reserve(3); // Should not shrink the capacity
        assert_eq!(stack.data.len(), 10);
    }
}
