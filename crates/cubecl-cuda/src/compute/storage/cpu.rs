use std::{collections::HashMap, ffi::c_void};

use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};

pub const PINNED_MEMORY_ALIGNMENT: usize = core::mem::size_of::<u128>();

pub struct PinnedMemoryStorage {
    memory: HashMap<StorageId, PinnedMemory>,
    mem_alignment: usize,
}

impl PinnedMemoryStorage {
    /// Create a new storage.
    pub fn new() -> Self {
        Self {
            memory: Default::default(),
            mem_alignment: PINNED_MEMORY_ALIGNMENT,
        }
    }
}

struct PinnedMemory {
    ptr: *mut c_void,
    /// Keep the pointer alive for async allocation.
    #[allow(unused)]
    ptr2ptr: *mut *mut c_void,
}

pub struct PinnedMemoryResource {
    pub ptr: *mut u8,
    pub size: usize,
}

impl ComputeStorage for PinnedMemoryStorage {
    type Resource = PinnedMemoryResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let memory = self.memory.get(&handle.id).unwrap();

        let offset = handle.offset() as usize;
        let size = handle.size() as usize;

        unsafe {
            PinnedMemoryResource {
                ptr: memory.ptr.add(offset).cast(),
                size,
            }
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let resource = unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let ptr2ptr: *mut *mut c_void = &mut ptr;

            // Call cuMemAllocHost_v2 to allocate pinned host memory
            let result = cudarc::driver::sys::cuMemAllocHost_v2(ptr2ptr, size as usize);

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!("cuMemAllocHost_v2 failed with error code: {:?}", result);
            }

            PinnedMemory { ptr, ptr2ptr }
        };

        let id = StorageId::new();
        self.memory.insert(id.clone(), resource);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        if let Some(resource) = self.memory.remove(&id) {
            unsafe {
                cudarc::driver::sys::cuMemFreeHost(resource.ptr);
            };
        }
    }
}

unsafe impl Send for PinnedMemoryResource {}
unsafe impl Send for PinnedMemoryStorage {}
