use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
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
    pub buffer: wgpu::Buffer,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
}

impl WgpuResource {
    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource<'_> {
        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: Some(
                NonZeroU64::new(self.size).expect("0 size resources are not yet supported."),
            ),
        };
        wgpu::BindingResource::Buffer(binding)
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

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

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

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.memory.remove(&id);
    }
}
