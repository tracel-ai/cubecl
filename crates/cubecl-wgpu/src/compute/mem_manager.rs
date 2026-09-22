use crate::compute::copies::WgpuCopies;
use crate::{WgpuResource, WgpuStorage};
use cubecl_core::{
    MemoryConfiguration,
    server::{BufferBinding, IoError},
};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::memory_management::Cleanup;
use cubecl_server::memory_management::relocation::Relocate;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, FailureId, ManagedMemoryBinding, ManagedMemoryHandle, MemoryAllocationMode,
        MemoryHandle, MemoryLocation, MemoryManagement, MemoryManagementOptions, PageGuard,
    },
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

#[derive(Debug)]
pub struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    memory_uniforms: MemoryManagement<WgpuStorage>,
    memory_pool_staging: MemoryManagement<WgpuStorage>,
    uniforms: Vec<ManagedMemoryHandle>,
    /// The failure store the staging and uniforms pools shed into.
    ///
    /// Only the main pool's allocations back [`BufferBinding`]s, so only they
    /// can ever carry a failure — the device-wide store is threaded into the
    /// main pool's operations for that reason. The auxiliary pools still need
    /// a store to shed into (their signatures are the same), and every
    /// decrement they make is of `None`, so this one stays empty forever.
    aux: ErrorGraph,
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
            aux: ErrorGraph::default(),
        }
    }

    pub(crate) fn bind(
        &mut self,
        old: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        failures: &mut ErrorGraph,
    ) {
        self.memory_pool.bind(old, new, 0, failures).unwrap();
    }

    /// Reserve `size` bytes on the main pool without cleaning it up first —
    /// see [`MemoryManagement::reserve_keeping_pages`].
    pub(crate) fn reserve_keeping_pages(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.memory_pool.reserve_keeping_pages(size, failures)
    }

    /// Whether the main pool has anything outdated a relocation could move
    /// — see [`MemoryManagement::relocatable`].
    pub(crate) fn relocatable(&self) -> bool {
        self.memory_pool.relocatable()
    }

    /// The bytes every pool of this stream holds from the device.
    pub(crate) fn bytes_allocated(&self) -> u64 {
        self.memory_pool.bytes_allocated()
            + self.memory_uniforms.bytes_allocated()
            + self.memory_pool_staging.bytes_allocated()
    }

    /// Whether the main pool's outdated pools are worth emptying now, on a
    /// device whose streams hold `allocated` bytes — see
    /// [`MemoryManagement::relocation`].
    pub(crate) fn relocation(&self, allocated: u64) -> Option<Relocate> {
        self.memory_pool.relocation(allocated)
    }

    /// Empty the main pool's outdated pools into its current pages, with
    /// `copier` carrying the bytes.
    pub(crate) fn relocate(
        &mut self,
        copier: &mut WgpuCopies,
        reason: Relocate,
        failures: &mut ErrorGraph,
    ) {
        self.memory_pool.relocate(copier, reason, failures);
    }

    /// The failure carried by the allocation behind `binding`, if any — see
    /// [`MemoryManagement::failure`]. Main pool only: the auxiliary pools'
    /// allocations never back a [`BufferBinding`].
    pub(crate) fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory_pool.failure(&binding.memory, binding.range())
    }

    /// Point the bytes `binding` names at `failure` — see
    /// [`MemoryManagement::taint`].
    pub(crate) fn taint(
        &mut self,
        binding: &BufferBinding,
        failure: FailureId,
        failures: &mut ErrorGraph,
    ) {
        self.memory_pool
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    /// The bytes `binding` names have a writer again — see
    /// [`MemoryManagement::written`].
    pub(crate) fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory_pool
            .written(&binding.memory, binding.range(), failures)
    }

    pub(crate) fn reserve_staging(
        &mut self,
        size: u64,
    ) -> Result<(WgpuResource, ManagedMemoryBinding), IoError> {
        let handle = self.memory_pool_staging.reserve(size, &mut self.aux)?;
        let binding = MemoryHandle::binding(handle);
        let resource = self
            .memory_pool_staging
            .get_resource(binding.clone(), None, None)
            .unwrap();

        Ok((resource, binding))
    }

    /// Reserve `size` bytes in the room the main pool already holds — see
    /// [`MemoryManagement::try_reserve`].
    pub(crate) fn try_reserve(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        self.memory_pool.try_reserve(size, failures)
    }

    /// Keep the main-pool page `location` names as it is while the guard
    /// lives — see [`MemoryManagement::guard`].
    pub(crate) fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        self.memory_pool.guard(location)
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
            .reserve(size, &mut self.aux)
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

    pub(crate) fn memory_usage(&self) -> cubecl_server::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_report(&self) -> cubecl_server::memory_management::MemoryReport {
        self.memory_pool.memory_report()
    }

    pub(crate) fn memory_cleanup(&mut self, cleanup: Cleanup, failures: &mut ErrorGraph) {
        self.memory_pool.cleanup(cleanup, failures);
        // An explicit cleanup also reclaims the uniforms pool: the info cache
        // holds uniform slices across flushes, so this is where the pages of
        // just-released entries (see `MetadataInfoCache::clear_unpinned`) are
        // actually returned to the driver.
        if cleanup == Cleanup::Explicit {
            self.memory_uniforms.cleanup(cleanup, &mut self.aux);
        }
    }

    pub(crate) fn mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }

    pub(crate) fn release_uniforms(&mut self) {
        self.uniforms.clear();
    }
}
