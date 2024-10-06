use super::{SliceBinding, SliceHandle, SliceId};
use crate::{memory_management::MemoryUsage, storage::{ComputeStorage, StorageHandle, StorageId}};

#[derive(new, Debug)]
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: usize,
}

impl Slice {
    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    pub(crate) fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }

    pub(crate) fn id(&self) -> SliceId {
        *self.handle.id()
    }
}

pub(crate) fn calculate_padding(size: usize, buffer_alignment: usize) -> usize {
    let remainder = size % buffer_alignment;
    if remainder != 0 {
        buffer_alignment - remainder
    } else {
        0
    }
}

pub trait MemoryPool {
    fn max_alloc_size(&self) -> usize;

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle>;

    fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        exclude: &[StorageId],
    ) -> SliceHandle;

    fn alloc<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: usize)
        -> SliceHandle;

    fn get_memory_usage(&self) -> MemoryUsage;
}

#[derive(Debug, Clone)]
pub enum PoolType {
    ExclusivePages,
    SlicedPages { max_slice_size: usize },
}

/// Options to create a memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolOptions {
    /// What kind of pool to use.
    pub pool_type: PoolType,
    /// The amount of bytes used for each chunk in the memory pool.
    pub page_size: usize,
    /// The number of chunks allocated directly at creation.
    ///
    /// Useful when you know in advance how much memory you'll need.
    pub chunk_num_prealloc: usize,
}
