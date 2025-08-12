use cubecl_core::{
    MemoryConfiguration,
    server::{Binding, Handle, IoError},
};
use cubecl_runtime::{
    memory_management::{MemoryDeviceProperties, MemoryHandle, MemoryManagement},
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

use super::{WgpuResource, WgpuStorage};

#[derive(Debug)]
pub(crate) struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    memory_uniforms: MemoryManagement<WgpuStorage>,
}

impl WgpuMemManager {
    pub(crate) fn new(
        device: wgpu::Device,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
    ) -> Self {
        // Allocate storage & memory management for the main memory buffers.
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
        // uniforms as <uniform> usage.
        let memory_uniforms = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::UNIFORM
                    | BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT,
            ),
            &memory_properties,
            // Want simple pages with no offsets etc.
            MemoryConfiguration::ExclusivePages,
        );

        Self {
            memory_pool: memory_main,
            memory_uniforms,
        }
    }

    pub(crate) fn reserve_uniform(&mut self, size: u64) -> Result<WgpuResource, IoError> {
        let slice = self.memory_uniforms.reserve(size)?;
        let handle = self.memory_uniforms.get(slice.binding()).unwrap();
        Ok(self.memory_uniforms.storage().get(&handle))
    }

    pub(crate) fn reserve(&mut self, size: u64) -> Result<Handle, IoError> {
        Ok(Handle::new(
            self.memory_pool.reserve(size)?,
            None,
            None,
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

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
    }

    pub(crate) fn mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }
}
