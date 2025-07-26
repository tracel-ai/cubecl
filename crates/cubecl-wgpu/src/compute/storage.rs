use cubecl_runtime::storage::{
    AllocError, ComputeStorage, StorageHandle, StorageId, StorageUtilization,
};
use hashbrown::HashMap;
use std::num::NonZeroU64;
use wgpu::BufferUsages;

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, wgpu::Buffer>,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuResource {
    /// The wgpu buffer.
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
}

impl WgpuResource {
    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource {
        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: Some(
                NonZeroU64::new(self.size).expect("0 size resources are not yet supported."),
            ),
        };
        wgpu::BindingResource::Buffer(binding)
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Return the buffer size.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Return the buffer offset.
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(mem_alignment: usize, device: wgpu::Device, usages: BufferUsages) -> Self {
        Self {
            memory: HashMap::new(),
            device,
            buffer_usages: usages,
            mem_alignment,
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let buffer = self.memory.get(&handle.id).unwrap();
        WgpuResource::new(buffer.clone(), handle.offset(), handle.size())
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, AllocError> {
        let id = StorageId::new();

        // TODO: Test whether memor fits.
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        });

        self.memory.insert(id, buffer);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.memory.remove(&id);
    }
}
