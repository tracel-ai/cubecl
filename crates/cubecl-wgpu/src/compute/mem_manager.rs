use cubecl_common::stream_id::StreamId;
use cubecl_core::{
    MemoryConfiguration,
    server::{Binding, Handle, IoError},
};
use cubecl_runtime::{
    memory_management::{MemoryDeviceProperties, MemoryHandle, MemoryManagement, SliceHandle},
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

use super::{WgpuResource, WgpuStorage};

#[derive(Debug)]
pub(crate) struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    memory_uniforms: MemoryManagement<WgpuStorage>,
    uniforms: Vec<SliceHandle>,
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
        let memory_main = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT,
            ),
            &memory_properties,
            memory_config.clone(),
        );

        // TODO: In the future this should not need STORAGE, if cube writes out all
        // uniforms as having <uniform> usage.
        let memory_uniforms = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
        );

        Self {
            memory_pool: memory_main,
            memory_uniforms,
            uniforms: vec![],
        }
    }

    pub(crate) fn reserve(&mut self, size: u64, stream_id: StreamId) -> Result<Handle, IoError> {
        Ok(Handle::new(
            self.memory_pool.reserve(size)?,
            None,
            None,
            stream_id,
            size,
        ))
    }

    pub(crate) fn get_resource(&mut self, binding: Binding) -> WgpuResource {
        let handle = self
            .memory_pool
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
        self.memory_pool.storage().get(&handle)
    }

    pub(crate) fn reserve_uniform(&mut self, size: u64) -> WgpuResource {
        let slice = self
            .memory_uniforms
            .reserve(size)
            .expect("Must have enough memory for a uniform");
        // Keep track of this uniform until it is released.
        self.uniforms.push(slice.clone());
        let handle = self
            .memory_uniforms
            .get(slice.binding())
            .expect("Failed to find storage!");
        self.memory_uniforms.storage().get(&handle)
    }

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
    }

    pub(crate) fn mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }

    pub(crate) fn release_uniforms(&mut self) {
        self.uniforms.clear();
    }
}
