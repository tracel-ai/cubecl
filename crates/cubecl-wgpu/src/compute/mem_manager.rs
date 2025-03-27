use cubecl_core::{
    MemoryConfiguration,
    server::{Binding, Handle},
};
use cubecl_runtime::{
    memory_management::{
        MemoryDeviceProperties, MemoryManagement, MemoryPoolOptions, PoolType, StorageExclude,
    },
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

use super::{WgpuResource, WgpuStorage};

#[derive(Debug)]
pub(crate) struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    memory_pool_queries: MemoryManagement<WgpuStorage>,
    pending_operations: StorageExclude,
}

impl WgpuMemManager {
    pub(crate) fn new(
        device: wgpu::Device,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
    ) -> Self {
        // Allocate storage & memory management for the main memory buffers. Any calls
        // to empty() or create() with a small enough size will be allocated from this
        // main memory pool.
        #[allow(unused_mut)]
        let mut memory_main = MemoryManagement::from_configuration(
            WgpuStorage::new(
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT
                    | BufferUsages::QUERY_RESOLVE,
            ),
            &memory_properties,
            memory_config.clone(),
        );

        // Memory pool for timing queries.
        let memory_pool_queries = MemoryManagement::from_configuration(
            WgpuStorage::new(
                device.clone(),
                BufferUsages::COPY_SRC | BufferUsages::QUERY_RESOLVE,
            ),
            &memory_properties,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        // Size only needs to be 2 u64, but at leas alignment size.
                        // Assume alignment is enough.
                        max_alloc_size: memory_properties.alignment,
                    },
                    dealloc_period: None,
                }],
            },
        );
        Self {
            memory_pool: memory_main,
            memory_pool_queries,
            pending_operations: StorageExclude::default(),
        }
    }

    pub(crate) fn query(&mut self) -> (Handle, WgpuResource) {
        let size = 2 * size_of::<u64>() as u64;

        let handle = Handle::new(
            self.memory_pool_queries.reserve(size, None),
            None,
            None,
            size,
        );

        let binding = handle.clone().binding();
        let resource = self
            .memory_pool_queries
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .unwrap();

        (handle, resource)
    }

    pub(crate) fn reserve(&mut self, size: u64, exclude_pending_operations: bool) -> Handle {
        let exclude = if exclude_pending_operations {
            Some(&self.pending_operations)
        } else {
            None
        };
        Handle::new(self.memory_pool.reserve(size, exclude), None, None, size)
    }

    pub(crate) fn get_resource(&mut self, binding: Binding) -> WgpuResource {
        match self.memory_pool.get(binding.memory.clone()) {
            Some(handle) => {
                let handle = match binding.offset_start {
                    Some(offset) => handle.offset_start(offset),
                    None => handle,
                };
                let handle = match binding.offset_end {
                    Some(offset) => handle.offset_end(offset),
                    None => handle,
                };
                // Assume this resource is now used for something. That means we can't copy to it anymore,
                // as any copy will get ordered first.
                self.pending_operations.exclude_storage(handle.id);
                self.memory_pool.storage().get(&handle)
            }
            None => {
                let handle = self
                    .memory_pool_queries
                    .get(binding.memory.clone())
                    .expect("Failed to find storage!");

                let handle = match binding.offset_start {
                    Some(offset) => handle.offset_start(offset),
                    None => handle,
                };
                let handle = match binding.offset_end {
                    Some(offset) => handle.offset_end(offset),
                    None => handle,
                };
                self.memory_pool_queries.storage().get(&handle)
            }
        }
    }

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
    }

    pub(crate) fn clear_pending(&mut self) {
        self.pending_operations.clear();
    }

    pub(crate) fn needs_flush(&self, max_pending: usize) -> bool {
        self.pending_operations.count() > max_pending
    }
}
