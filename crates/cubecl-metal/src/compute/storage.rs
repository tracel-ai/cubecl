use crate::METAL_DISPATCH_LIMIT;
use cubecl_runtime::memory_management::MemoryDeviceProperties;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCPUCacheMode, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLResourceOptions, MTLStorageMode,
};
use std::collections::HashMap;

#[derive(Debug)]
pub struct MetalResource {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    size: usize,
}

unsafe impl Send for MetalResource {}
unsafe impl Sync for MetalResource {}

impl MetalResource {
    pub fn new(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        size: usize,
    ) -> Self {
        Self {
            buffer,
            offset,
            size,
        }
    }
    pub fn buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

pub struct MetalStorage {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    memory: HashMap<StorageId, MetalResource>,
    mem_alignment: usize,
}

unsafe impl Send for MetalStorage {}
unsafe impl Sync for MetalStorage {}

impl MetalStorage {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        memory_properties: MemoryDeviceProperties,
    ) -> Self {
        let heap = autoreleasepool(|_| unsafe {
            let heap_descriptor = MTLHeapDescriptor::new();
            let heap_size = memory_properties.max_page_size.next_multiple_of(4096);

            heap_descriptor.setSize(heap_size as usize);
            heap_descriptor.setStorageMode(MTLStorageMode::Shared);
            heap_descriptor.setCpuCacheMode(MTLCPUCacheMode::WriteCombined);
            heap_descriptor.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);

            device.newHeapWithDescriptor(&heap_descriptor)
        })
        .expect("Failed to create heap");
        let mem_alignment = memory_properties.alignment as usize;

        Self {
            device,
            heap,
            mem_alignment,
            memory: HashMap::with_capacity(METAL_DISPATCH_LIMIT),
        }
    }
}

impl ComputeStorage for MetalStorage {
    type Resource = MetalResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let resource = self.memory.get(&handle.id).expect("Failed to get buffer");
        MetalResource::new(
            resource.buffer().clone(),
            handle.offset() as usize,
            handle.size() as usize,
        )
    }

    fn alloc(&mut self, size: u64) -> StorageHandle {
        let size = size as usize;
        let available_size = self.heap.maxAvailableSizeWithAlignment(self.alignment());

        let resource_options = MTLResourceOptions::StorageModeShared
            | MTLResourceOptions::HazardTrackingModeUntracked
            | MTLResourceOptions::CPUCacheModeWriteCombined;

        // Try heap allocation first if there's enough space
        let buffer = if available_size >= size {
            // Attempt heap allocation
            match self
                .heap
                .newBufferWithLength_options(size, resource_options)
            {
                Some(buf) => buf,
                None => {
                    // Heap allocation failed despite available space, fallback to device
                    self.device
                        .newBufferWithLength_options(size, resource_options)
                        .expect("Failed to allocate buffer from device")
                }
            }
        } else {
            // Not enough heap space, allocate directly from device
            self.device
                .newBufferWithLength_options(size, resource_options)
                .expect("Failed to allocate buffer from device")
        };

        // Create storage handle and resource
        let storage_id = StorageId::new();
        let resource = MetalResource {
            buffer,
            offset: 0,
            size,
        };

        self.memory.insert(storage_id, resource);

        StorageHandle::new(
            storage_id,
            StorageUtilization {
                offset: 0,
                size: size as u64,
            },
        )
    }

    fn dealloc(&mut self, id: StorageId) {
        self.memory.remove(&id);
    }
}
