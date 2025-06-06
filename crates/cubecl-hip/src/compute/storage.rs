use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::collections::HashMap;

/// Buffer storage for HIP.
pub struct HipStorage {
    mem_alignment: usize,
    memory: HashMap<StorageId, cubecl_hip_sys::hipDeviceptr_t>,
    deallocations: Vec<StorageId>,
    stream: cubecl_hip_sys::hipStream_t,
    activate_slices: HashMap<ActiveResource, cubecl_hip_sys::hipDeviceptr_t>,
}

#[derive(new, Debug, Hash, PartialEq, Eq, Clone)]
struct ActiveResource {
    ptr: u64,
}

unsafe impl Send for HipStorage {}

impl core::fmt::Debug for HipStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("HipStorage {{ device: {:?} }}", self.stream).as_str())
    }
}

/// Keeps actual HIP buffer references in a hashmap with ids as key.
impl HipStorage {
    /// Create a new storage on the given stream.
    pub fn new(mem_alignment: usize, stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self {
            mem_alignment,
            memory: HashMap::new(),
            deallocations: Vec::new(),
            stream,
            activate_slices: HashMap::new(),
        }
    }

    /// Actually deallocates buffers tagged to be deallocated.
    pub fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(ptr) = self.memory.remove(&id) {
                unsafe {
                    cubecl_hip_sys::hipFreeAsync(ptr, self.stream);
                }
            }
        }
    }

    pub fn flush(&mut self) {
        self.activate_slices.clear();
    }
}

pub type Binding = cubecl_hip_sys::hipDeviceptr_t;

/// The memory resource that can be allocated for the device.
#[derive(new, Debug)]
pub struct HipResource {
    /// The buffer.
    pub ptr: cubecl_hip_sys::hipDeviceptr_t,
    pub binding: Binding,
    pub offset: u64,
    pub size: u64,
}

unsafe impl Send for HipResource {}

impl ComputeStorage for HipStorage {
    type Resource = HipResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = (*self.memory.get(&handle.id).unwrap()) as u64;

        let offset = handle.offset();
        let size = handle.size();

        let ptr = ptr + offset;
        let key = ActiveResource::new(ptr);

        self.activate_slices
            .insert(key.clone(), ptr as cubecl_hip_sys::hipDeviceptr_t);

        // The ptr needs to stay alive until we send the task to the server.
        let ptr = self.activate_slices.get(&key).unwrap();

        HipResource::new(
            *ptr,
            ptr as *const cubecl_hip_sys::hipDeviceptr_t as *mut std::ffi::c_void,
            offset,
            size,
        )
    }

    fn alloc(&mut self, size: u64) -> StorageHandle {
        let id = StorageId::new();
        unsafe {
            let mut dptr: *mut ::std::os::raw::c_void = std::ptr::null_mut();
            let status = cubecl_hip_sys::hipMallocAsync(&mut dptr, size as usize, self.stream);
            assert_eq!(status, HIP_SUCCESS, "Should allocate memory");
            self.memory.insert(id, dptr);
        };
        StorageHandle::new(id, StorageUtilization { offset: 0, size })
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }
}
