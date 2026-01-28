use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;

/// Wrapper for MTLBuffer that is Send + Sync
/// Safety: Metal objects are thread-safe by design
#[derive(Debug, Clone)]
pub struct MetalBufferHandle(Retained<ProtocolObject<dyn MTLBuffer>>);

unsafe impl Send for MetalBufferHandle {}
unsafe impl Sync for MetalBufferHandle {}

impl MetalBufferHandle {
    pub fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        Self(buffer)
    }

    pub fn as_ref(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.0
    }
}

/// Metal buffer storage
#[derive(Debug)]
pub struct MetalStorage {
    buffers: HashMap<StorageId, MetalBufferHandle>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    counter: u64,
}

impl MetalStorage {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            buffers: HashMap::new(),
            device,
            counter: 0,
        }
    }

    pub fn get_buffer(&self, handle: &StorageHandle) -> Option<&MetalBufferHandle> {
        self.buffers.get(&handle.id)
    }
}

impl ComputeStorage for MetalStorage {
    type Resource = MetalBufferHandle;

    fn alignment(&self) -> usize {
        256 // Metal standard alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        self.get_buffer(handle).expect("Buffer not found").clone()
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, cubecl_core::server::IoError> {
        use objc2_metal::MTLDevice;

        let id = StorageId::new();

        // Create buffer with shared storage mode for CPU-GPU access
        // MTLResourceStorageModeShared allows both CPU and GPU to access the buffer
        let buffer = (*self.device)
            .newBufferWithLength_options(size as usize, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| cubecl_core::server::IoError::Unknown {
                description: format!("Failed to allocate Metal buffer of size {}", size),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            })?;

        self.buffers.insert(id, MetalBufferHandle::new(buffer));
        self.counter += 1;

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.buffers.remove(&id);
    }
}

// Helper method to insert buffers (used by server)
impl MetalStorage {
    pub fn insert(&mut self, resource: Retained<ProtocolObject<dyn MTLBuffer>>) -> StorageHandle {
        let id = StorageId::new();
        self.counter += 1;
        self.buffers.insert(id, MetalBufferHandle::new(resource));
        StorageHandle {
            id,
            utilization: StorageUtilization { offset: 0, size: 0 },
        }
    }
}
