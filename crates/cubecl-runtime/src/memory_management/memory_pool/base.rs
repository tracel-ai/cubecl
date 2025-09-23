use super::{SliceBinding, SliceHandle, SliceId};
use crate::{
    memory_management::MemoryUsage,
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};
use hashbrown::HashMap;
 use crate::storage::VirtualStorage;

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

    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle>;

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError>;

    fn get_memory_usage(&self) -> MemoryUsage;

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    );
}


pub(crate) trait MemoryChunk {



    /// merge slice at first_slice_address with the next slice (if there is one and if it's free)
    /// return a boolean representing if a merge happened
    fn merge_with_next_slice(
        &mut self,
        first_slice_address: u64,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> bool;
    fn find_slice(&self, address: u64) -> Option<SliceId>;
    fn insert_slice(&mut self, address: u64, slice: SliceId);
}



pub trait VirtualMemoryPool {
    fn max_alloc_size(&self) -> u64;

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle>;

    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle>;

    fn alloc<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError>;

    fn get_memory_usage(&self) -> MemoryUsage;

    fn cleanup<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    );
}
