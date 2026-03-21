use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;

/// Wrapper for `MTLBuffer` that is Send + Sync.
#[derive(Debug, Clone)]
pub struct MetalBufferHandle(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: GPU memory access is synchronized via command buffer ordering.
unsafe impl Send for MetalBufferHandle {}
unsafe impl Sync for MetalBufferHandle {}

impl MetalBufferHandle {
    pub fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        Self(buffer)
    }

    pub fn inner(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.0
    }
}

/// Metal buffer storage
#[derive(Debug)]
pub struct MetalStorage {
    buffers: HashMap<StorageId, MetalBufferHandle>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalStorage {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            buffers: HashMap::new(),
            device,
        }
    }
}

impl ComputeStorage for MetalStorage {
    type Resource = MetalBufferHandle;

    fn alignment(&self) -> usize {
        256 // Metal standard alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        self.buffers
            .get(&handle.id)
            .expect("Buffer not found")
            .clone()
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, cubecl_core::server::IoError> {
        use objc2_metal::MTLDevice;

        let id = StorageId::new();

        // MTLResourceStorageModeShared allows both CPU and GPU to access the buffer.
        let buffer = (*self.device)
            .newBufferWithLength_options(size as usize, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| cubecl_core::server::IoError::Unknown {
                description: format!("Failed to allocate Metal buffer of size {}", size),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            })?;

        self.buffers.insert(id, MetalBufferHandle::new(buffer));

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.buffers.remove(&id);
    }
}
