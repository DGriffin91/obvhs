//! Stack data structures implemented on the stack or heap
use core::ops::{Deref, DerefMut};

// TODO could allow Clone + Zeroable instead of Copy
pub trait FastStack<T: Copy + Default> {
    fn push(&mut self, v: T);
    fn pop_fast(&mut self) -> T;
    fn pop(&mut self) -> Option<T>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn clear(&mut self);
}

/// Creates a stack- or heap-allocated stack based on runtime size.
///
/// This macro chooses between `StackStack<T, N>` (stack-allocated) and `HeapStack<T>`
/// (heap-allocated) depending on the provided size and threshold values.
///
/// # Usage
/// fast_stack!(
///     u32,             // type
///     (64, 128, 256),  // thresholds
///     size,            // runtime size
///     stack,           // variable name
///     {
///         stack.push(42);
///     }
/// );
///
/// - Falls back to `HeapStack<T>::new_with_capacity(size)` if `size` exceeds thresholds.
#[macro_export]
macro_rules! fast_stack {
    ( $ty:ty,
      ($first:expr $(, $rest:expr)* $(,)?),
      $size:expr,
      $stack_ident:ident,
      $body:block
    ) => {{
        match $size {
            s if s <= $first => {
                let mut $stack_ident = $crate::faststack::StackStack::<$ty, $first>::default();
                $body
            }
            $(
                s if s <= $rest => {
                    let mut $stack_ident = $crate::faststack::StackStack::<$ty, $rest>::default();
                    $body
                }
            )*
            _ => {
                let mut $stack_ident = $crate::faststack::HeapStack::<$ty>::new_with_capacity($size);
                $body
            }
        }
    }};
}

/// A stack data structure implemented on the heap with adjustable capacity.
///
/// This structure allows pushing and popping elements and will never automatically
/// allocate or deallocate. The only functions that will result in allocation are
/// `HeapStack::new_with_capacity` and `HeapStack::reserve`.
///
/// The elements must implement the `Clone` and `Default` traits.
#[derive(Clone)]
pub struct HeapStack<T: Copy + Default> {
    data: Vec<T>,
    index: usize,
}

impl<T: Copy + Default> Default for HeapStack<T> {
    // For safety, HeapStack cannot have a capacity of zero.
    fn default() -> Self {
        Self::new_with_capacity(1)
    }
}

impl<T: Copy + Default> HeapStack<T> {
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

    /// Returns the capacity of the stack.
    ///
    /// # Returns
    /// The capacity of the stack.
    #[inline(always)]
    pub fn cap(&self) -> usize {
        self.data.len()
    }

    /// Reserves capacity for at least `cap` elements.
    ///
    /// # Arguments
    /// * `cap` - The desired capacity.
    ///
    /// If the new capacity is smaller than the current capacity, this function does nothing.
    #[inline(always)]
    pub fn reserve(&mut self, cap: usize) {
        if cap < self.data.len() {
            return;
        }
        self.data.resize(cap, Default::default());
    }
}

impl<T: Copy + Default> Deref for HeapStack<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data[..self.index]
    }
}

impl<T: Copy + Default> DerefMut for HeapStack<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data[..self.index]
    }
}

impl<T: Copy + Default> FastStack<T> for HeapStack<T> {
    /// Pushes a value onto the stack.
    ///
    /// # Arguments
    /// * `v` - The value to be pushed onto the stack.
    ///
    /// # Panics
    /// This function will panic if the stack is full.
    #[inline(always)]
    fn push(&mut self, v: T) {
        if self.index < self.data.len() {
            *unsafe { self.data.get_unchecked_mut(self.index) } = v;
            self.index += 1;
        } else {
            panic!(
                "Index out of bounds: the HeapStack is full (length: {}) and cannot accommodate more elements",
                self.data.len()
            );
        }
    }

    /// Pops a value from the stack.
    ///
    /// # Returns
    /// `Some(T)` if the stack is not empty, otherwise `None`.
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        if self.index > 0 {
            self.index = self.index.saturating_sub(1);
            Some(self.data[self.index])
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
    fn pop_fast(&mut self) -> T {
        self.index = self.index.saturating_sub(1);
        let v = unsafe { self.data.get_unchecked(self.index) };
        *v
    }

    /// Returns the number of elements in the stack.
    ///
    /// # Returns
    /// The length of the stack.
    #[inline(always)]
    fn len(&self) -> usize {
        self.index
    }

    /// Returns true if the stack is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.index == 0
    }

    /// Clears the stack, removing all elements.
    ///
    /// This operation does not deallocate the stack's capacity.
    #[inline(always)]
    fn clear(&mut self) {
        self.index = 0;
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
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
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

        assert_eq!(stack.pop_fast(), 3);
        assert_eq!(stack.pop_fast(), 2);
        assert_eq!(stack.pop_fast(), 1);
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

/// A stack data structure implemented on the stack with fixed capacity.
pub struct StackStack<T: Copy + Default, const STACK_SIZE: usize> {
    data: [T; STACK_SIZE],
    index: usize,
}

impl<T: Copy + Default, const STACK_SIZE: usize> Default for StackStack<T, STACK_SIZE> {
    fn default() -> Self {
        Self {
            data: [Default::default(); STACK_SIZE],
            index: Default::default(),
        }
    }
}

impl<T: Copy + Default, const STACK_SIZE: usize> FastStack<T> for StackStack<T, STACK_SIZE> {
    /// Pushes a value onto the stack. If the stack is full it will overwrite the value in the last position.
    #[inline(always)]
    fn push(&mut self, v: T) {
        // TODO: possibly check bounds in debug or make a push_fast()
        *unsafe { self.data.get_unchecked_mut(self.index) } = v;
        self.index = (self.index + 1).min(STACK_SIZE - 1);
    }
    /// Pops a value from the stack without checking bounds. If the stack is empty it will return the value in the first position.
    #[inline(always)]
    fn pop_fast(&mut self) -> T {
        self.index = self.index.saturating_sub(1);
        let v = unsafe { self.data.get_unchecked(self.index) };
        *v
    }
    /// Pops a value from the stack.
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        if self.index > 0 {
            self.index = self.index.saturating_sub(1);
            Some(self.data[self.index])
        } else {
            None
        }
    }
    /// Returns the number of elements in the stack.
    #[inline(always)]
    fn len(&self) -> usize {
        self.index
    }
    /// Returns true if the stack is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.index == 0
    }
    /// Clears the stack, removing all elements.
    #[inline(always)]
    fn clear(&mut self) {
        self.index = 0;
    }
}

impl<T: Copy + Default> FastStack<T> for Vec<T> {
    fn push(&mut self, v: T) {
        self.push(v);
    }
    fn pop_fast(&mut self) -> T {
        self.pop().unwrap()
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    fn clear(&mut self) {
        self.clear();
    }
}
