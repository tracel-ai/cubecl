use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use hashbrown::HashMap;
use std::{num::NonZeroU64, sync::Arc};

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, Arc<wgpu::Buffer>>,
    deallocations: Vec<StorageId>,
    device: Arc<wgpu::Device>,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
#[derive(new)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: Arc<wgpu::Buffer>,

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
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            device,
        }
    }

    /// Actually deallocates buffers tagged to be deallocated.
    pub fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(buffer) = self.memory.remove(&id) {
                buffer.destroy()
            }
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    // 32 bytes is enough to handle a double4 worth of alignment.
    // See: https://github.com/gfx-rs/wgpu/issues/3508
    // NB: cudamalloc and co. actually align to _256_ bytes. Worth
    // trying this in the future to see if it reduces memory coalescing.
    const ALIGNMENT: u64 = 32;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let buffer = self.memory.get(&handle.id).unwrap();
        WgpuResource::new(buffer.clone(), handle.offset(), handle.size())
    }

    fn alloc(&mut self, size: u64) -> StorageHandle {
        let id = StorageId::new();
        let buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));

        self.memory.insert(id, buffer);
        StorageHandle::new(id, StorageUtilization { offset: 0, size })
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }
}
