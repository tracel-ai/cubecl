use alloc::rc::Rc;
use core::sync::atomic::{AtomicUsize, Ordering};

/// An atomic counter with a simplified interface.
#[derive(Clone, Debug, Default)]
pub struct AtomicCounter {
    inner: Rc<AtomicUsize>,
}

impl AtomicCounter {
    /// Creates a new counter with `val` as its initial value.
    pub fn new(val: usize) -> Self {
        Self {
            inner: Rc::new(AtomicUsize::new(val)),
        }
    }

    /// Increments the counter and returns the last count.
    pub fn inc(&self) -> usize {
        self.inner.fetch_add(1, Ordering::SeqCst)
    }

    /// Gets the value of the counter without incrementing it.
    pub fn get(&self) -> usize {
        self.inner.load(Ordering::SeqCst)
    }

    pub fn get_and_reset(&self) -> usize {
        self.inner.swap(0, Ordering::SeqCst)
    }
}
