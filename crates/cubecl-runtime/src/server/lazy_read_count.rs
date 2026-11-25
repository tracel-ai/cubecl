use crate::memory_management::ManagedMemoryDescriptor;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

/// How many lazy reads a device keeps that a write may have to start or wait for: per
/// allocation, which a write to that memory asks, and in all, which spares every write the
/// asking while there are none.
///
/// Raised on the device thread and lowered from any, so a count read late costs a write a
/// lookup and never misses a read.
#[derive(Clone, Debug, Default)]
pub struct LazyReadCount {
    total: Arc<AtomicUsize>,
    /// Only ever raised, so a server that saw it unchanged knows no read was let go since.
    let_go: Arc<AtomicUsize>,
}

impl LazyReadCount {
    /// Whether the device keeps no lazy read a write may have to start or wait for.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.total.load(Ordering::Relaxed) == 0
    }

    /// How many lazy reads were let go of so far.
    pub fn let_go_so_far(&self) -> usize {
        self.let_go.load(Ordering::Acquire)
    }

    /// One more lazy read of the memory `descriptor` names.
    pub fn add(&self, descriptor: &ManagedMemoryDescriptor) {
        descriptor.lazy_read_kept();
        self.total.fetch_add(1, Ordering::Relaxed);
    }

    /// One lazy read of the memory `descriptor` names fewer.
    ///
    /// # Assumptions
    ///
    /// - It follows an [`add`](Self::add) of the same memory, which nothing else lowered.
    pub fn remove(&self, descriptor: &ManagedMemoryDescriptor) {
        descriptor.lazy_read_let_go();
        self.total.fetch_sub(1, Ordering::Relaxed);
        self.let_go.fetch_add(1, Ordering::Release);
    }
}
