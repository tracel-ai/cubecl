use std::{
    alloc::{self, Layout},
    mem,
    ops::{Index, IndexMut},
    ptr::{self, NonNull},
};

/// Circular buffer with stable index without reallocations.
pub struct CircularBuffer<T> {
    ptr: NonNull<T>,
    cap: usize,
    front: usize,
    len: usize,
}

unsafe impl<T: Send> Send for CircularBuffer<T> {}
unsafe impl<T: Sync> Sync for CircularBuffer<T> {}

impl<T> Drop for CircularBuffer<T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop() {}
        let layout = Layout::array::<T>(self.cap).unwrap();
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T> IndexMut<usize> for CircularBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

impl<T> Index<usize> for CircularBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl<T> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        debug_assert!(
            mem::size_of::<T>() != 0,
            "This data structure doesn't support ZST"
        );
        let front = 0;
        let len = 0;
        let layout = Layout::array::<T>(capacity).expect("Allocation too large");

        let ptr = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr as *mut T) else {
            alloc::handle_alloc_error(layout);
        };
        Self {
            ptr,
            front,
            len,
            cap: capacity,
        }
    }

    pub fn front(&self) -> usize {
        self.front
    }

    pub fn back(&self) -> usize {
        let mut index = self.front + self.len;
        if index >= self.cap {
            index -= self.cap;
        }
        index
    }

    pub fn push(&mut self, elem: T) {
        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.back()), elem);
        }
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        let value = unsafe { Some(ptr::read(self.ptr.as_ptr().add(self.front))) };
        self.len -= 1;
        self.front += 1;

        if self.front >= self.cap {
            self.front -= self.cap;
        }
        value
    }
}
