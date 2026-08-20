use crate::{WgpuResource, WgpuStorage};
use cubecl_core::{
    MemoryConfiguration,
    server::{BufferBinding, IoError},
};
use cubecl_environment::sync::Arc;
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
        use_vulkan_compiler: bool,
    ) -> Self {
        // Allocate storage & memory management for the main memory buffers. Any calls
        // to empty() or create() with a small enough size will be allocated from this
        // main memory pool.
        //
        // `memory_config` (which honors any programmatic pool override) shapes
        // the main pool only; the staging and uniforms pools below have
        // deliberate configurations that must not be overridden.
        let memory_main = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT,
                use_vulkan_compiler,
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
                false,
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
                use_vulkan_compiler,
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

    pub(crate) fn bind(&mut self, old: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        self.memory_pool.bind(old, new, 0).unwrap();
    }

    pub(crate) fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        match self.memory_pool.reserve(size) {
            Ok(handle) => Ok(handle),
            Err(err) => Err(err),
        }
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

    pub(crate) fn get_resource(&mut self, binding: BufferBinding) -> Result<WgpuResource, IoError> {
        self.memory_pool
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// Reserve a uniform slice and resolve its resource. The returned
    /// [`ManagedMemoryHandle`] owns the slice: the uniform stays reserved as
    /// long as a clone of it is held (the info cache holds one for cached
    /// metadata buffers), on top of the per-flush retention in `self.uniforms`.
    pub(crate) fn reserve_uniform(&mut self, size: u64) -> (ManagedMemoryHandle, WgpuResource) {
        let slice = self
            .memory_uniforms
            .reserve(size)
            .expect("Must have enough memory for a uniform");
        // Keep track of this uniform until it is released.
        self.uniforms.push(slice.clone());
        let retained = slice.clone();
        let handle = self
            .memory_uniforms
            .get_storage(slice.binding())
            .expect("Failed to find storage!");
        let resource = self
            .memory_uniforms
            .storage()
            .get(&handle)
            .expect("Failed to get the uniform's storage!");
        (retained, resource)
    }

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_report(&self) -> cubecl_runtime::memory_management::MemoryReport {
        self.memory_pool.memory_report()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
        // An explicit cleanup also reclaims the uniforms pool: the info cache
        // holds uniform slices across flushes, so this is where the pages of
        // just-released entries (see `MetadataInfoCache::clear_unpinned`) are
        // actually returned to the driver.
        if explicit {
            self.memory_uniforms.cleanup(explicit);
        }
    }

    pub(crate) fn mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }

    /// Rebuild the main pool with a new layout, keeping the old one when
    /// something is still live in it. The staging and uniforms pools keep
    /// their deliberate configurations.
    ///
    /// # Errors
    ///
    /// [`InstallMemoryPoolsError::PoolsInUse`] when the rebuild was refused.
    pub(crate) fn install_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        props: &MemoryDeviceProperties,
    ) -> Result<(), cubecl_runtime::memory_management::InstallMemoryPoolsError> {
        self.memory_pool.install_pools(config, props)
    }

    pub(crate) fn release_uniforms(&mut self) {
        self.uniforms.clear();
    }

    /// Begin a graph capture on the pools a recorded launch allocates from:
    /// the main pool (kernel buffers, intermediates) and the uniforms pool
    /// (info uniforms, Vulkan address buffers). The staging pool is left
    /// alone — reads are rejected while recording, and warmup-phase staging is
    /// transient. See [`MemoryManagement::capture_begin`].
    pub(crate) fn capture_begin(&mut self) {
        self.memory_pool.capture_begin();
        self.memory_uniforms.capture_begin();
    }

    /// End the warmup priming phase on the captured pools; call immediately
    /// before recording starts. See [`MemoryManagement::capture_priming_end`].
    pub(crate) fn capture_priming_end(&mut self) {
        self.memory_pool.capture_priming_end();
        self.memory_uniforms.capture_priming_end();
    }

    /// End the capture on both pools, returning the retained handles that pin
    /// every slice the window touched for the graph's lifetime. See
    /// [`MemoryManagement::capture_end`].
    pub(crate) fn capture_end(&mut self) -> Vec<ManagedMemoryHandle> {
        let mut retained = self.memory_pool.capture_end();
        retained.extend(self.memory_uniforms.capture_end());
        retained
    }
}
