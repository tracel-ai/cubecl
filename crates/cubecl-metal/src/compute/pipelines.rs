//! The half of Metal's compilation that needs the server: the MSL store, and
//! the pipelines it hands out.

use crate::compute::artifact::{MetalArtifact, MetalArtifactCompiler, MslCacheEntry};
use cubecl_core::prelude::*;
use cubecl_server::compiler::{ArtifactId, ArtifactStore, CompilationError, CompilationTarget};
use cubecl_server::kernel::BufferIOAttr;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLDevice};

/// A compute pipeline and what launching it needs.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) cube_dim: CubeDim,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the compiler's answer, carried here because on a cache hit nothing
    /// else of the compilation survives. `None` for entries persisted before
    /// the answer existed, which the launch path reads as every buffer both
    /// read and written.
    pub(crate) io: Option<std::sync::Arc<[BufferIOAttr]>>,
}

/// Builds Metal pipelines for one device, from the MSL store when it holds the
/// kernel and from [`MetalArtifactCompiler`] when it does not.
#[derive(Debug)]
pub(crate) struct MetalPipelines {
    compiler: MetalArtifactCompiler,
    /// MSL kept between runs, which saves the compilation to MSL; the
    /// driver still builds the library from it.
    msl_store: ArtifactStore<MslCacheEntry>,
}

impl MetalPipelines {
    pub fn new(compiler: MetalArtifactCompiler) -> Self {
        // The MSL is emitted from device-derived compilation options and
        // validated against the device's shared-memory limit, so the device
        // name is what keeps a bundle shipped across machines from serving
        // sources built for another GPU.
        let device_key = compiler.device().name().to_string();

        Self {
            compiler,
            msl_store: ArtifactStore::new("metal", format!("msl_{device_key}")),
        }
    }
}

impl CompilationTarget for MetalPipelines {
    type Compiler = MetalArtifactCompiler;
    type Loaded = CompiledKernel;

    fn compiler(&self) -> &MetalArtifactCompiler {
        &self.compiler
    }

    fn persists(&self) -> bool {
        self.msl_store.persists()
    }

    fn stored(&mut self, id: &ArtifactId<()>) -> Option<MetalArtifact> {
        let entry = self.msl_store.take(&id.kernel)?;
        Some(MetalArtifact {
            entry,
            pipeline: None,
        })
    }

    /// The pipeline finalizing built, or, for MSL read from the store, the
    /// one the driver builds now.
    fn load(
        &mut self,
        _id: &ArtifactId<()>,
        artifact: &MetalArtifact,
    ) -> Result<CompiledKernel, CompilationError> {
        match &artifact.pipeline {
            Some(pipeline) => Ok(pipeline.clone()),
            None => self.compiler.build(&artifact.entry),
        }
    }

    fn store(
        &mut self,
        id: &ArtifactId<()>,
        artifact: MetalArtifact,
        _source: Option<&str>,
    ) -> bool {
        self.msl_store.keep(&id.kernel, artifact.entry, None)
    }
}
