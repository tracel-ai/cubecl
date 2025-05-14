use cubecl_runtime::storage::ComputeStorage;

pub struct CPUStorage {}

pub struct SharedPointer {}

impl ComputeStorage for CPUStorage {
    type Resource = SharedPointer;

    // TODO Find what alignment is needed
    const ALIGNMENT: u64 = 32;

    fn get(&mut self, handle: &cubecl_runtime::storage::StorageHandle) -> Self::Resource {
        todo!()
    }

    fn alloc(&mut self, size: u64) -> cubecl_runtime::storage::StorageHandle {
        todo!()
    }

    fn dealloc(&mut self, id: cubecl_runtime::storage::StorageId) {
        todo!()
    }
}
