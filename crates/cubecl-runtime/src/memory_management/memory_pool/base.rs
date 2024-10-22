use super::{SliceBinding, SliceHandle, SliceId};
use crate::memory_management::MemoryLock;
use crate::{
    memory_management::MemoryUsage,
    storage::{ComputeStorage, StorageHandle},
};

#[derive(new, Debug)]
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: u64,
}

impl Slice {
    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    pub(crate) fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding
    }

    pub(crate) fn id(&self) -> SliceId {
        *self.handle.id()
    }
}

pub(crate) fn calculate_padding(size: u64, buffer_alignment: u64) -> u64 {
    let remainder = size % buffer_alignment;
    if remainder != 0 {
        buffer_alignment - remainder
    } else {
        0
    }
}

pub trait MemoryPool {
    fn max_alloc_size(&self) -> u64;

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle>;

    fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        locked: Option<&MemoryLock>,
    ) -> SliceHandle;

    fn alloc<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: u64) -> SliceHandle;

    fn get_memory_usage(&self) -> MemoryUsage;

    fn cleanup<Storage: ComputeStorage>(&mut self, storage: &mut Storage, alloc_nr: u64);
}
