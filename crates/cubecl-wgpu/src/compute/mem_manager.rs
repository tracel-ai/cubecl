use crate::{WgpuResource, WgpuStorage};
use cubecl_common::{backtrace::BackTrace, stub::Arc};
use cubecl_core::{
    MemoryConfiguration,
    server::{Binding, HandleId, IoError, MemorySlot},
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryBinding, ManagedMemoryHandle, MemoryAllocationMode, MemoryHandle,
        MemoryManagement, MemoryManagementOptions,
    },
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

#[derive(Debug)]
pub(crate) struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    memory_uniforms: MemoryManagement<WgpuStorage>,
    memory_pool_staging: MemoryManagement<WgpuStorage>,
    uniforms: Vec<ManagedMemoryHandle>,
}

impl WgpuMemManager {
    pub(crate) fn new(
        device: wgpu::Device,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
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
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );

        let memory_staging = MemoryManagement::from_configuration(
            WgpuStorage::new(
                wgpu::COPY_BUFFER_ALIGNMENT as usize,
                device.clone(),
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            &memory_properties,
            // Unfortunately, we can't reuse a different part of a buffer for different reads, so we
            // can't have a single binding with multiple slices allocated.
            MemoryConfiguration::ExclusivePages,
            logger.clone(),
            MemoryManagementOptions::new("Staging CPU Memory").mode(MemoryAllocationMode::Auto),
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
            logger,
            MemoryManagementOptions::new("Uniform GPU Memory").mode(MemoryAllocationMode::Auto),
        );

        Self {
            memory_pool: memory_main,
            memory_pool_staging: memory_staging,
            memory_uniforms,
            uniforms: vec![],
        }
    }

    pub(crate) fn bind(&mut self, buffers: Vec<MemorySlot>, handles: Vec<Binding>) {
        for (buffer, handle) in buffers.into_iter().zip(handles.into_iter()) {
            self.memory_pool.bind(handle.id, buffer);
        }
    }

    pub(crate) fn free(&mut self, handle: HandleId) {
        self.memory_pool.free(handle)
    }

    pub(crate) fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.memory_pool.reserve(size)
    }

    pub(crate) fn reserve_staging(
        &mut self,
        size: u64,
    ) -> Result<(WgpuResource, ManagedMemoryBinding), IoError> {
        let handle = self.memory_pool_staging.reserve(size)?;
        let binding = MemoryHandle::binding(handle);
        let resource = self
            .memory_pool_staging
            .get_resource(binding.clone(), None, None)
            .unwrap();

        Ok((resource, binding))
    }

    pub(crate) fn get_resource(
        &mut self,
        handle: Binding,
    ) -> Result<(WgpuResource, ManagedMemoryHandle), IoError> {
        let buffer = self.memory_pool.get_slot(handle)?;
        let handle = self
            .memory_pool
            .get_storage(buffer.memory.clone().binding())
            .ok_or_else(|| IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })?;
        let handle = match buffer.offset_start {
            Some(offset) => handle.offset_start(offset),
            None => handle,
        };
        let handle = match buffer.offset_end {
            Some(offset) => handle.offset_end(offset),
            None => handle,
        };
        let resource = self.memory_pool.storage().get(&handle);
        Ok((resource, buffer.memory))
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
            .get_storage(slice.binding())
            .expect("Failed to find storage!");
        self.memory_uniforms.storage().get(&handle)
    }

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
    }

    pub(crate) fn mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }

    pub(crate) fn release_uniforms(&mut self) {
        self.uniforms.clear();
    }
}
