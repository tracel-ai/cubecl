//! A wgpu stream's memories: the main one every buffer a user allocates lives
//! in, and the ones the stream keeps for its own reads and launches.

use crate::{WgpuResource, WgpuStorage};
use cubecl_core::{MemoryConfiguration, server::IoError};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{
        AuxiliaryMemoryReport, Cleanup, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle,
        MemoryAllocationMode, MemoryHandle, MemoryManagement, MemoryManagementOptions, PageUpdate,
    },
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

const STAGING: &str = "Staging CPU Memory";
const UNIFORMS: &str = "Uniform GPU Memory";

/// The memory every buffer a user allocates lives in, laid out as `config`
/// says.
pub(crate) fn main_memory(
    device: &wgpu::Device,
    properties: &MemoryDeviceProperties,
    config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
    use_vulkan_compiler: bool,
) -> MemoryManagement<WgpuStorage> {
    MemoryManagement::from_configuration(
        WgpuStorage::new(
            properties.alignment as usize,
            device.clone(),
            BufferUsages::STORAGE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::INDIRECT,
            use_vulkan_compiler,
        ),
        properties,
        config,
        logger,
        MemoryManagementOptions::new("Main GPU Memory"),
    )
}

/// The memories a stream keeps for itself: staging buffers its reads map, and
/// the uniform buffers its launches read their info from.
///
/// Neither ever backs a buffer a user holds, so neither is relocated or takes
/// the main memory's layout: each has one page per allocation.
#[derive(Debug)]
pub struct AuxiliaryMemory {
    staging: MemoryManagement<WgpuStorage>,
    uniforms: MemoryManagement<WgpuStorage>,
    /// The uniforms the queued launches read, released once they are
    /// submitted.
    queued_uniforms: Vec<ManagedMemoryHandle>,
    /// The failure store these memories shed into.
    ///
    /// Only the main memory's allocations back a user's buffer, so only they
    /// can ever carry a failure. These memories still need a store to shed
    /// into (their signatures are the same), and every decrement they make is
    /// of `None`, so this one stays empty forever.
    failures: ErrorGraph,
}

impl AuxiliaryMemory {
    pub(crate) fn new(
        device: &wgpu::Device,
        properties: &MemoryDeviceProperties,
        logger: Arc<ServerLogger>,
        use_vulkan_compiler: bool,
    ) -> Self {
        let staging = MemoryManagement::from_configuration(
            WgpuStorage::new(
                wgpu::COPY_BUFFER_ALIGNMENT as usize,
                device.clone(),
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                false,
            ),
            properties,
            // A mapped buffer is mapped whole, so two reads cannot share one.
            MemoryConfiguration::ExclusivePages,
            logger.clone(),
            MemoryManagementOptions::new(STAGING).mode(MemoryAllocationMode::Auto),
        );

        // TODO: In the future this should not need STORAGE, if cube writes out all
        // uniforms as having <uniform> usage.
        let uniforms = MemoryManagement::from_configuration(
            WgpuStorage::new(
                properties.alignment as usize,
                device.clone(),
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
                use_vulkan_compiler,
            ),
            properties,
            MemoryConfiguration::ExclusivePages,
            logger,
            MemoryManagementOptions::new(UNIFORMS).mode(MemoryAllocationMode::Auto),
        );

        Self {
            staging,
            uniforms,
            queued_uniforms: Vec::new(),
            failures: ErrorGraph::default(),
        }
    }

    /// The bytes these memories hold from the device.
    pub(crate) fn bytes_allocated(&self) -> u64 {
        self.staging.bytes_allocated() + self.uniforms.bytes_allocated()
    }

    /// What each of these memories holds.
    pub(crate) fn report(&self) -> Vec<AuxiliaryMemoryReport> {
        [(STAGING, &self.staging), (UNIFORMS, &self.uniforms)]
            .into_iter()
            .map(|(name, memory)| AuxiliaryMemoryReport {
                name: name.to_string(),
                pools: memory.memory_report(),
            })
            .collect()
    }

    pub(crate) fn reserve_staging(
        &mut self,
        size: u64,
    ) -> Result<(WgpuResource, ManagedMemoryBinding), IoError> {
        let handle = self
            .staging
            .reserve(size, PageUpdate::Allow, &mut self.failures)?;
        let binding = MemoryHandle::binding(handle);
        let resource = self
            .staging
            .get_resource(binding.clone(), None, None)
            .unwrap();

        Ok((resource, binding))
    }

    /// Reserve a uniform slice and resolve its resource. The returned
    /// [`ManagedMemoryHandle`] owns the slice: the uniform stays reserved as
    /// long as a clone of it is held (the info cache holds one for cached
    /// metadata buffers), on top of the per-flush retention in
    /// `queued_uniforms`.
    pub(crate) fn reserve_uniform(&mut self, size: u64) -> (ManagedMemoryHandle, WgpuResource) {
        let slice = self
            .uniforms
            .reserve(size, PageUpdate::Allow, &mut self.failures)
            .expect("Must have enough memory for a uniform");
        self.queued_uniforms.push(slice.clone());
        let retained = slice.clone();
        let handle = self
            .uniforms
            .get_storage(slice.binding())
            .expect("Failed to find storage!");
        let resource = self
            .uniforms
            .storage()
            .get(&handle)
            .expect("Failed to get the uniform's storage!");
        (retained, resource)
    }

    /// Give back every uniform page nothing uses. The info cache holds
    /// uniform slices across flushes, so this is where the pages of released
    /// entries (see `MetadataInfoCache::clear_unpinned`) go back to the driver.
    pub(crate) fn cleanup_uniforms(&mut self) {
        self.uniforms.cleanup(Cleanup::Explicit, &mut self.failures);
    }

    /// The queued launches were submitted: their uniforms may be reused.
    pub(crate) fn release_uniforms(&mut self) {
        self.queued_uniforms.clear();
    }
}
