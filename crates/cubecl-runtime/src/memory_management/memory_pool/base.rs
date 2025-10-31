use super::{SliceBinding, SliceHandle, SliceId};
use crate::{
    memory_management::MemoryUsage,
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};

/// Declares how memory is allocated in a reusable pool.
pub trait MemoryPool {
    /// Whether the memory pool accepts the given size.
    fn accept(&self, size: u64) -> bool;

    /// Retrieves the [storage handle](StorageHandle) using the [slice binding](SliceBinding).
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle>;

    /// Try to reserve a memory slice of the given size.
    ///
    /// # Notes
    ///
    /// It is not guaranteed the `try_reserve` function will reapply the accept function.
    /// Therefore it is a good idea to call [MemoryUsage::accept()] before using `try_reserve`.
    ///
    /// # Returns
    ///
    /// A [slice handle](StorageHandle) if the current memory pool has enough memory, otherwise it
    /// will returns [None]. You can then call [MemoryPool::alloc()] to increase the amount of
    /// memory the pool has.
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle>;

    /// Increases the amount of memory the pool has and returns a [slice handle](StorageHandle)
    /// corresponding to the requested size.
    ///
    /// # Notes
    ///
    /// The function uses a [ComputeStorage] to perform the allocation. It might return an error
    /// if the allocation fails or if the requested size is bigger than the memory pool is
    /// configured to handle.
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError>;

    /// Computes the [MemoryUsage] for this pool.
    fn get_memory_usage(&self) -> MemoryUsage;

    /// Cleanup the memory pool, maybe freeing some memory using the [ComputeStorage].
    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    );
}

#[derive(new, Debug)]
/// Slice of data with its associated storage.
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: u64,
}

impl Slice {
    /// If the slice is free to be reused.
    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    /// The total size of the slice including padding.
    pub(crate) fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding
    }

    /// The id of the slice.
    pub(crate) fn id(&self) -> SliceId {
        *self.handle.id()
    }
}

/// Calculates the padding required to store the given size in a buffer given the memory alignment.
pub(crate) fn calculate_padding(size: u64, memory_alignment: u64) -> u64 {
    let remainder = size % memory_alignment;
    if remainder != 0 {
        memory_alignment - remainder
    } else {
        0
    }
}
