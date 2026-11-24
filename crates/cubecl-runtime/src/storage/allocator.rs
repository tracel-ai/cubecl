use super::{StorageHandle, StorageId};
use crate::server::IoError;

/// What a runtime allocates device memory with: the pools carve their pages
/// from it and hand its allocations back to it.
pub trait StorageAllocator: Send {
    /// What a kernel or a copy is given to reach an allocation, such as a
    /// device pointer or a buffer.
    type Resource: Send;

    /// The resource of the allocation `handle` names, covering at least the
    /// part of it the handle covers.
    ///
    /// # Assumptions
    ///
    /// `handle` names an allocation of this storage that has not been
    /// [deallocated](Self::deallocate), and lies within it.
    ///
    /// # Panics
    ///
    /// An implementation may panic when `handle` names no live allocation, or
    /// reaches past the end of the one it names.
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    /// A new allocation of `size` bytes, held until it is
    /// [deallocated](Self::deallocate).
    ///
    /// # Errors
    ///
    /// Every implementation returns an error when the device cannot hold
    /// `size` more bytes, and may return one for a size it does not accept.
    fn allocate_bytes(&mut self, size: u64) -> Result<StorageHandle, IoError>;

    /// Gives the allocation `id` back. An implementation may defer the
    /// release to the next [`flush`](Self::flush), but never reuses `id`.
    fn deallocate(&mut self, id: StorageId);

    /// Releases every deallocation still deferred.
    fn flush(&mut self);

    /// The bytes this storage holds from the device right now: allocated and
    /// not yet returned. A deallocation still waiting on [`flush`](Self::flush)
    /// is still held.
    fn bytes_allocated(&self) -> u64;
}
