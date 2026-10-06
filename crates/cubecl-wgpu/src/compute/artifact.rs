//! The half of wgpu's compilation any thread can do: from a kernel to the
//! shader a pipeline is built from.

use alloc::sync::Arc;
use core::marker::PhantomData;

use crate::{AutoRepresentation, CompilerInfo, WgpuCompiler};
use cubecl_core::{WgpuCompilationOptions, prelude::*};
use cubecl_ir::DeviceProperties;
use cubecl_server::compiler::{ArtifactCompiler, ArtifactId, CompilationRecording};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_units};

use super::pipelines::MetadataLayout;

/// Compiles kernels for one wgpu device with `C`, the shader language that
/// device was set up for.
#[derive(Debug)]
pub(crate) struct WgpuArtifactCompiler<C> {
    pub properties: DeviceProperties,
    pub options: WgpuCompilationOptions,
    pub backend: wgpu::Backend,
    pub _compiler: PhantomData<fn() -> C>,
}

/// A compiled shader and what building and launching its pipeline needs.
pub struct WgpuArtifact {
    pub entrypoint_name: String,
    /// The shader's text: what a module is built from unless `repr` says
    /// the device takes something else.
    pub source: String,
    /// The tag of the language `source` is written in.
    pub lang: &'static str,
    pub repr: Option<AutoRepresentation>,
    pub compiler_info: CompilerInfo,
    pub io: Option<Arc<[BufferIOAttr]>>,
}

impl<C: WgpuCompiler> ArtifactCompiler for WgpuArtifactCompiler<C> {
    type Variant = MetadataLayout;
    type Lowered = CompiledKernel<C>;
    type Artifact = WgpuArtifact;

    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<MetadataLayout>,
        recording: &mut CompilationRecording,
        logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError> {
        validate_cube_dim(&self.properties, &id.kernel)?;
        validate_units(&self.properties, &id.kernel)?;

        let mut compiler = C::init(self.backend, &self.options);
        let mut lowered = CompiledKernel::lower(kernel, recording, |definition| {
            compiler.compile_kernel(kernel, definition, &self.options)
        })?;

        if logger.compilation_source_activated() {
            lowered.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                id.kernel.clone(),
            ));
        }
        logger.log_compilation(&lowered);

        compiler.validate_ir(&lowered.repr, &self.properties)?;

        Ok(lowered)
    }

    fn finalize(
        &self,
        _id: &ArtifactId<MetadataLayout>,
        mut lowered: Self::Lowered,
    ) -> Result<WgpuArtifact, LaunchError> {
        let compiler = C::init(self.backend, &self.options);
        // The compiled kernel's per-buffer answer, before the repr is
        // consumed: what the write scope stages from.
        let io = lowered.io.take().map(Arc::from);
        let (compiler_info, repr) = compiler.normalize_repr(lowered.repr);

        Ok(WgpuArtifact {
            entrypoint_name: lowered.entrypoint_name,
            source: lowered.source,
            lang: compiler.lang_tag(),
            repr,
            compiler_info,
            io,
        })
    }
}
