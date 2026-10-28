use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;

/// A slice of a Metal buffer: what a storage handle resolves to.
///
/// The buffer and the slice travel together because Metal binds a buffer at
/// an offset, and every use of one — a dispatch, a blit, a host copy — needs
/// both.
#[derive(Debug, Clone)]
pub struct MetalResource {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: u64,
    size: u64,
}

// SAFETY: GPU memory access is synchronized via command buffer ordering.
unsafe impl Send for MetalResource {}
unsafe impl Sync for MetalResource {}

impl MetalResource {
    /// The buffer the slice is carved from.
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// The buffer the slice is carved from, retained.
    pub fn retained(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.buffer.clone()
    }

    /// Where the slice starts in its buffer, in bytes.
    pub fn offset(&self) -> usize {
        self.offset as usize
    }

    /// The bytes in the slice.
    pub fn size(&self) -> usize {
        self.size as usize
    }

    /// The slice's first byte, as the host sees it: Metal buffers are shared
    /// with the host, so this is the memory the device reads and writes.
    pub fn host_ptr(&self) -> *mut u8 {
        // SAFETY: the slice lies inside the buffer, so the offset stays in
        // the allocation `contents` points at.
        unsafe { (self.buffer.contents().as_ptr() as *mut u8).add(self.offset()) }
    }
}

/// Metal buffer storage
#[derive(Debug)]
pub struct MetalStorage {
    buffers: HashMap<StorageId, Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// The bytes `buffers` holds.
    allocated: u64,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

// SAFETY: GPU memory access is synchronized via command buffer ordering.
unsafe impl Send for MetalStorage {}

impl MetalStorage {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            buffers: HashMap::new(),
            allocated: 0,
            device,
        }
    }
}

impl ComputeStorage for MetalStorage {
    type Resource = MetalResource;

    fn alignment(&self) -> usize {
        256 // Metal standard alignment
    }

    fn get(
        &mut self,
        handle: &StorageHandle,
    ) -> Result<Self::Resource, cubecl_core::server::IoError> {
        let buffer = self.buffers.get(&handle.id).ok_or_else(|| {
            cubecl_core::server::IoError::StorageHandleNotFound {
                reason: format!("{} in the Metal buffer storage", handle.id).into(),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            }
        })?;
        Ok(MetalResource {
            buffer: buffer.clone(),
            offset: handle.offset(),
            size: handle.size(),
        })
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, cubecl_core::server::IoError> {
        use objc2_metal::MTLDevice;

        let id = StorageId::new();

        // MTLResourceStorageModeShared allows both CPU and GPU to access the buffer. Metal returns
        // nil for a zero-byte buffer, so an empty tensor still gets one byte behind its handle.
        let length = size.max(1) as usize;
        let buffer = (*self.device)
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| cubecl_core::server::IoError::Unknown {
                description: format!("Failed to allocate Metal buffer of size {}", size),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            })?;

        // The device may round the length up; that is what it holds.
        self.allocated += buffer.length() as u64;
        self.buffers.insert(id, buffer);

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        if let Some(buffer) = self.buffers.remove(&id) {
            self.allocated -= buffer.length() as u64;
        }
    }

    fn flush(&mut self) {
        // No deferred deallocations to flush.
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}
