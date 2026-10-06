//! The half of Metal's compilation any thread can do: from a kernel to the
//! MSL a library is built from.

use crate::MetalCompiler;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_server::compiler::{ArtifactCompiler, ArtifactId, CompilationRecording};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_shared_memory, validate_units};

/// Compiles kernels to MSL for one Metal device.
#[derive(Debug)]
pub(crate) struct MetalArtifactCompiler {
    pub properties: DeviceProperties,
    pub options: cubecl_cpp::shared::CompilationOptions,
}

/// MSL ready for the driver to build a library from, with what launching it
/// needs: what the compilation store keeps for a kernel.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct MslCacheEntry {
    pub entrypoint_name: String,
    pub cube_dim: (u32, u32, u32),
    pub source: String,
    /// See [`CompiledKernel::io`](super::pipelines::CompiledKernel::io);
    /// defaulted for entries persisted before the field existed.
    #[serde(default)]
    pub io: Option<Vec<BufferIOAttr>>,
}

impl ArtifactCompiler for MetalArtifactCompiler {
    type Variant = ();
    type Lowered = CompiledKernel<MetalCompiler>;
    type Artifact = MslCacheEntry;

    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        recording: &mut CompilationRecording,
        logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError> {
        validate_cube_dim(&self.properties, &id.kernel)?;
        validate_units(&self.properties, &id.kernel)?;

        log::trace!("Compiling kernel to MSL");

        let mut lowered = CompiledKernel::lower(kernel, recording, |definition| {
            CompiledKernel::compile(
                kernel,
                definition,
                &mut MetalCompiler::default(),
                &self.options,
            )
        })?;

        if logger.compilation_source_activated() {
            lowered.debug_info = Some(DebugInformation::new("msl", id.kernel.clone()));
        }
        logger.log_compilation(&lowered);

        // Checked before the driver builds a pipeline: Metal would reject the
        // kernel there anyway, but with an opaque compilation error instead of
        // a resource limit error.
        validate_shared_memory(
            &self.properties,
            lowered.repr.as_ref().map(|repr| repr.shared_memory_size),
        )?;

        Ok(lowered)
    }

    fn finalize(
        &self,
        _id: &ArtifactId<()>,
        mut lowered: Self::Lowered,
    ) -> Result<MslCacheEntry, LaunchError> {
        let cube_dim = lowered.cube_dim;
        Ok(MslCacheEntry {
            io: lowered.io.take(),
            entrypoint_name: lowered.entrypoint_name,
            cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
            source: lowered.source,
        })
    }
}
