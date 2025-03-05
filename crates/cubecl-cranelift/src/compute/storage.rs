use cranelift_jit::JITModule;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use hashbrown::HashMap;
use std::{
    alloc::{alloc, Layout},
    num::NonZeroU64,
};

pub struct CraneliftStorage {
    memory: HashMap<StorageId, *mut u8>,
    deallocations: Vec<StorageId>,
}

unsafe impl Send for CraneliftStorage {}

pub struct CraneliftResource {
    pub ptr: *mut u8,
    offset: u64,
    size: u64,
}

impl From<&CraneliftResource> for Vec<u8> {
    fn from(resource: &CraneliftResource) -> Self {
        unsafe {
            Vec::from_raw_parts(
                resource.ptr.add(resource.offset as usize),
                resource.size as usize,
                resource.size as usize,
            )
        }
    }
}

unsafe impl Send for CraneliftResource {}

impl ComputeStorage for CraneliftStorage {
    type Resource = CraneliftResource;

    const ALIGNMENT: u64 = 8; //FIXME

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = self.memory.get(&handle.id).unwrap();
        let offset = handle.offset();
        let size = handle.size();

        CraneliftResource {
            ptr: *ptr,
            offset,
            size,
        }
    }

    fn alloc(&mut self, size: u64) -> StorageHandle {
        let id = StorageId::new();
        let ptr = unsafe {
            alloc(Layout::from_size_align_unchecked(
                (size as usize).next_power_of_two(),
                Self::ALIGNMENT as usize,
            ))
        };
        self.memory.insert(id, ptr);
        StorageHandle::new(id, StorageUtilization { offset: 0, size })
    }

    fn dealloc(&mut self, id: StorageId) {
        (self.memory.remove(&id).unwrap());
    }
}
