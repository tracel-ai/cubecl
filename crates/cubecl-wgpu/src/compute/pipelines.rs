//! The half of wgpu's compilation that needs the server: the SPIR-V store,
//! and the pipelines it hands out.

#[cfg(feature = "spirv")]
use alloc::sync::Arc;

use crate::{PipelineEntry, WgpuCompiler};
use cubecl_core::server::MetadataBindingInfo;
#[cfg(feature = "spirv")]
use cubecl_environment::persistence::Store;
use cubecl_server::compiler::{ArtifactId, CompilationError, CompilationTarget};
#[cfg(feature = "spirv")]
use cubecl_server::compiler::{KernelCacheKey, store_compiled};

use super::artifact::{WgpuArtifact, WgpuArtifactCompiler};

/// What a launch's metadata binding is, as far as the pipeline layout that
/// receives it is concerned: the part of a launch, beyond its kernel, that a
/// pipeline is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataLayout {
    /// The launch passes no metadata, so the layout has no binding for it.
    Absent,
    /// Every value is known when the kernel is compiled; a backend that can
    /// may bind it as a uniform.
    Static,
    /// Part of it is sized at launch, so it is bound as a storage buffer.
    Dynamic,
}

/// The layout `info` is bound with.
impl From<&MetadataBindingInfo> for MetadataLayout {
    fn from(info: &MetadataBindingInfo) -> Self {
        if info.data.is_empty() {
            Self::Absent
        } else if info.dynamic_metadata_offset >= info.data.len() {
            Self::Static
        } else {
            Self::Dynamic
        }
    }
}

/// Builds wgpu pipelines for one device, from the SPIR-V store when it holds
/// the kernel and from [`WgpuArtifactCompiler`] when it does not.
#[derive(Debug)]
pub(crate) struct WgpuPipelines<C> {
    compiler: WgpuArtifactCompiler<C>,
    #[cfg(feature = "spirv")]
    spirv_store: Option<Store<(u64, KernelCacheKey), cubecl_spirv::SpirvCacheEntry>>,
    /// What the SPIR-V store's keys are scoped by beside the kernel: SPIR-V
    /// compiled for one set of device properties is not valid for another.
    #[cfg(feature = "spirv")]
    properties_hash: u64,
    #[cfg(feature = "spirv")]
    build_id: cubecl_common::hash::StableHash,
}

impl<C> WgpuPipelines<C> {
    /// Pipelines built by `compiler`'s device, with nothing built yet.
    pub fn new(
        compiler: WgpuArtifactCompiler<C>,
        #[cfg(feature = "spirv")] spirv_store: Option<
            Store<(u64, KernelCacheKey), cubecl_spirv::SpirvCacheEntry>,
        >,
        #[cfg(feature = "spirv")] properties_hash: u64,
    ) -> Self {
        Self {
            compiler,
            #[cfg(feature = "spirv")]
            spirv_store,
            #[cfg(feature = "spirv")]
            properties_hash,
            #[cfg(feature = "spirv")]
            build_id: cubecl_server::compiler::build_id_hash(),
        }
    }
}

impl<C: WgpuCompiler> CompilationTarget for WgpuPipelines<C> {
    type Compiler = WgpuArtifactCompiler<C>;
    type Loaded = PipelineEntry;

    fn compiler(&self) -> &WgpuArtifactCompiler<C> {
        &self.compiler
    }

    /// WGSL is compiled by the driver on every run, so without the SPIR-V
    /// store there is nothing persisted for a switch to invalidate.
    fn persists(&self) -> bool {
        #[cfg(feature = "spirv")]
        return self.spirv_store.is_some();
        #[cfg(not(feature = "spirv"))]
        return false;
    }

    #[allow(unused_variables)]
    fn stored(&mut self, id: &ArtifactId<MetadataLayout>) -> Option<WgpuArtifact> {
        #[cfg(not(feature = "spirv"))]
        return None;

        #[cfg(feature = "spirv")]
        {
            use crate::{AutoRepresentation, CompilerInfo, ParamsTransfer};

            let key = (
                self.properties_hash,
                KernelCacheKey::new(&id.kernel, self.build_id),
            );
            let entry = self.spirv_store.as_mut()?.remove(&key)?;
            log::trace!("Using SPIR-V cache");

            let params_transfer = match entry.kernel.immediate_size {
                Some(_) => ParamsTransfer::Immediate,
                None => ParamsTransfer::Uniform,
            };
            Some(WgpuArtifact {
                entrypoint_name: entry.entrypoint_name,
                // SPIR-V reaches the device as an assembled module, never as text.
                source: String::new(),
                lang: "spv",
                io: entry.kernel.io.clone().map(Arc::from),
                repr: Some(AutoRepresentation::SpirV(entry.kernel)),
                compiler_info: CompilerInfo::Vulkan { params_transfer },
                pipeline: None,
            })
        }
    }

    /// The pipeline finalizing built, or, for an artifact read from the
    /// store, the one built now.
    fn load(
        &mut self,
        id: &ArtifactId<MetadataLayout>,
        artifact: &WgpuArtifact,
    ) -> Result<PipelineEntry, CompilationError> {
        match &artifact.pipeline {
            Some(pipeline) => Ok(pipeline.clone()),
            None => self.compiler.build(id, artifact),
        }
    }

    /// Only a SPIR-V kernel is stored: any other build changes nothing.
    #[allow(unused_variables)]
    fn store(
        &mut self,
        id: &ArtifactId<MetadataLayout>,
        artifact: WgpuArtifact,
        _source: Option<&str>,
    ) -> bool {
        #[cfg(not(feature = "spirv"))]
        return false;

        #[cfg(feature = "spirv")]
        {
            let (Some(store), Some(crate::AutoRepresentation::SpirV(kernel))) =
                (self.spirv_store.as_mut(), artifact.repr)
            else {
                return false;
            };
            let key = (
                self.properties_hash,
                KernelCacheKey::new(&id.kernel, self.build_id),
            );
            store_compiled(
                store,
                key,
                cubecl_spirv::SpirvCacheEntry::new(artifact.entrypoint_name, kernel),
            )
        }
    }
}
