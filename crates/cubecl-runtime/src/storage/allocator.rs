use super::{StorageHandle, StorageId};
use crate::server::IoError;

/// Storage types are responsible for allocating and deallocating memory.
pub trait ComputeStorage: Send {
    /// The resource associated type determines the way data is implemented and how
    /// it can be accessed by kernels.
    type Resource: Send;

    /// The alignment memory is allocated with in this storage.
    fn alignment(&self) -> usize;

    /// Returns the underlying resource for a specified storage handle.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageHandleNotFound`] when this storage holds no
    /// allocation for the handle's id — a handle that outlived its allocation,
    /// or one whose id a deallocation retired.
    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError>;

    /// Allocates `size` units of memory and returns a handle to it
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError>;

    /// Deallocates the memory pointed by the given storage id.
    ///
    /// These deallocations might need to be flushed with [`Self::flush`].
    fn dealloc(&mut self, id: StorageId);

    /// Flush deallocations when required.
    fn flush(&mut self);

    /// The bytes this storage holds from the device right now: allocated and
    /// not yet returned. A deallocation still waiting on [`flush`](Self::flush)
    /// is still held.
    fn bytes_allocated(&self) -> u64;
}
