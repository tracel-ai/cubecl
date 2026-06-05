use std::{
    alloc::{self, Layout},
    fmt::Debug,
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

impl<T: Debug> Debug for CircularBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let internal_slice = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap) };
        f.debug_struct("CircularBuffer")
            .field("ptr", &internal_slice)
            .field("front", &self.front)
            .field("len", &self.len)
            .finish()
    }
}

unsafe impl<T: Send> Send for CircularBuffer<T> {}
unsafe impl<T: Sync> Sync for CircularBuffer<T> {}

impl<T> Drop for CircularBuffer<T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop_back() {}
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

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
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

    pub fn push_back(&mut self, elem: T) {
        debug_assert!(self.len <= self.cap);
        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.back()), elem);
        }
        self.len += 1;
    }

    pub fn pop_front(&mut self) -> Option<T> {
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

    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let back = self.back();
        let back = if back > 0 { back - 1 } else { self.cap - 1 };
        let value = unsafe { Some(ptr::read(self.ptr.as_ptr().add(back))) };
        self.len -= 1;
        value
    }
}

#[cfg(test)]
mod tests {
    use crate::compute::threadpool::circular_buffer::CircularBuffer;

    #[test]
    fn test_order() {
        let capacity = 16;
        let mut circular_buffer = CircularBuffer::new(capacity);
        for i in 0..capacity {
            circular_buffer.push_back(i);
        }
        for i in 0..capacity {
            assert_eq!(Some(i), circular_buffer.pop_front());
        }
    }

    #[test]
    fn test_pop_front_empty() {
        let capacity = 16;
        let mut circular_buffer = CircularBuffer::new(capacity);
        for i in 0..4 {
            circular_buffer.push_back(i);
        }
        for i in 0..6 {
            let expected = if i < 4 { Some(i) } else { None };
            assert_eq!(expected, circular_buffer.pop_front());
        }
    }

    #[test]
    fn test_pop_back_empty() {
        let capacity = 16;
        let mut circular_buffer = CircularBuffer::new(capacity);
        for i in 0..4 {
            circular_buffer.push_back(i);
        }
        for i in 0..6 {
            let expected = if i < 4 { Some(3 - i) } else { None };
            assert_eq!(expected, circular_buffer.pop_back());
        }
    }

    #[test]
    fn test_pop_back_around() {
        let capacity = 16;
        let mut circular_buffer = CircularBuffer::new(capacity);
        for i in 0..capacity / 2 {
            circular_buffer.push_back(i);
            circular_buffer.pop_front();
        }
        for i in 0..capacity {
            circular_buffer.push_back(i);
        }
        for i in 0..capacity {
            let expected = Some(capacity - i - 1);
            assert_eq!(expected, circular_buffer.pop_back());
        }
    }

    #[test]
    fn test_pop_front_around() {
        let capacity = 16;
        let mut circular_buffer = CircularBuffer::new(capacity);
        for i in 0..capacity / 2 {
            circular_buffer.push_back(i);
            circular_buffer.pop_front();
        }
        for i in 0..capacity {
            circular_buffer.push_back(i);
        }
        for i in 0..capacity {
            let expected = Some(i);
            assert_eq!(expected, circular_buffer.pop_front());
        }
    }
}
