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
    ptr: *mut u8,
    offset: u64,
    size: u64,
}

unsafe impl Send for CraneliftResource {}

impl ComputeStorage for CraneliftStorage {
    type Resource = CraneliftResource;

    const ALIGNMENT: u64 = 8;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = self.memory.get(&handle.id).unwrap();
        let offset = handle.offset();
        let size = handle.size();

        todo!()
    }

    fn alloc(&mut self, size: u64) -> StorageHandle {
        let id = StorageId::new();
        let ptr = unsafe {
            alloc(Layout::from_size_align_unchecked(
                next_2_power(size),
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

fn next_2_power(n: u64) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p as usize
}
