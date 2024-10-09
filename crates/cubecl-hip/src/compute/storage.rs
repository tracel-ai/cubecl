use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::collections::HashMap;

/// Buffer storage for HIP.
pub struct HipStorage {
    memory: HashMap<StorageId, cubecl_hip_sys::hipDeviceptr_t>,
    deallocations: Vec<StorageId>,
    stream: cubecl_hip_sys::hipStream_t,
    activate_slices: HashMap<ActiveResource, cubecl_hip_sys::hipDeviceptr_t>,
}

#[derive(new, Debug, Hash, PartialEq, Eq, Clone)]
struct ActiveResource {
    ptr: u64,
    kind: HipResourceKind,
}

unsafe impl Send for HipStorage {}

impl core::fmt::Debug for HipStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("HipStorage {{ device: {:?} }}", self.stream).as_str())
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl HipStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self {
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

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct HipResource {
    /// The buffer.
    pub ptr: cubecl_hip_sys::hipDeviceptr_t,
    pub binding: Binding,
    /// How the resource is used.
    pub kind: HipResourceKind,
}

unsafe impl Send for HipResource {}


impl HipResource {
    /// Return the binding view of the buffer.
    pub fn as_binding(&self) -> Binding {
        match self.kind {
            HipResourceKind::Full { .. } => self.binding,
            HipResourceKind::Slice { .. } => self.binding,
        }
    }

    /// Return the buffer size.
    pub fn size(&self) -> usize {
        match self.kind {
            HipResourceKind::Full { size } => size,
            HipResourceKind::Slice { size, offset: _ } => size,
        }
    }

    /// Return the buffer offset.
    pub fn offset(&self) -> usize {
        match self.kind {
            HipResourceKind::Full { size: _ } => 0,
            HipResourceKind::Slice { size: _, offset } => offset,
        }
    }
}

/// How the resource is used, either as a slice or fully.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum HipResourceKind {
    /// Represents an entire buffer.
    Full { size: usize },
    /// A slice over a buffer.
    Slice { size: usize, offset: usize },
}

impl ComputeStorage for HipStorage {
    type Resource = HipResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = *self.memory.get(&handle.id).unwrap();

        match handle.utilization {
            StorageUtilization::Full(size) => HipResource::new(
                ptr,
                ptr,
                HipResourceKind::Full { size },
            ),
            StorageUtilization::Slice { offset, size } => {
                let ptr: u64 = ptr as u64 + offset as u64;
                let kind = HipResourceKind::Slice { size, offset };
                let key = ActiveResource::new(ptr, kind.clone());
                self.activate_slices.insert(key.clone(), ptr as *mut _);

                // The ptr needs to stay alive until we send the task to the server.
                let ptr = *self.activate_slices.get(&key).unwrap();
                HipResource::new(
                    ptr,
                    ptr,
                    kind,
                )
            }
        }
    }

    fn alloc(&mut self, size: usize) -> StorageHandle {
        let id = StorageId::new();
        unsafe {
            let mut dptr: *mut ::std::os::raw::c_void = std::ptr::null_mut();
            let status = cubecl_hip_sys::hipMallocAsync(&mut dptr, size, self.stream);
            assert_eq!(status, HIP_SUCCESS, "Should allocate memory");
            self.memory.insert(id, dptr);
        };
        StorageHandle::new(id, StorageUtilization::Full(size))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }
}
