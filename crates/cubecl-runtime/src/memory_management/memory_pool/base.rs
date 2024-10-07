use super::{SliceBinding, SliceHandle, SliceId};
use crate::{
    memory_management::{MemoryLock, MemoryUsage},
    storage::{ComputeStorage, StorageHandle}
};

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
        locked: Option<&MemoryLock>,
    ) -> SliceHandle;

    fn alloc<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: usize)
        -> SliceHandle;

    fn get_memory_usage(&self) -> MemoryUsage;
}
