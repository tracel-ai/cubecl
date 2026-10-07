//! The half of Metal's compilation any thread can do: from a kernel to the
//! MSL, and the library and pipeline the device builds from it.

use crate::MetalCompiler;
use crate::compute::pipelines::CompiledKernel as MetalKernel;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::CompilationError;
use cubecl_server::compiler::{ArtifactCompiler, ArtifactId, CompilationRecording};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_shared_memory, validate_units};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode,
};

/// Compiles kernels to MSL for one Metal device, and has the device build
/// their pipelines.
///
/// The device is shared by every thread compiling at once: Metal's device
/// and the pipeline states it creates are thread-safe.
#[derive(Debug)]
pub(crate) struct MetalArtifactCompiler {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    properties: DeviceProperties,
    options: cubecl_cpp::shared::CompilationOptions,
}

impl MetalArtifactCompiler {
    pub(crate) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        properties: DeviceProperties,
        options: cubecl_cpp::shared::CompilationOptions,
    ) -> Self {
        Self {
            device,
            properties,
            options,
        }
    }

    /// The device kernels are compiled and loaded for.
    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
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

/// MSL, and the pipeline finalizing built from it.
pub struct MetalArtifact {
    /// What the store keeps.
    pub entry: MslCacheEntry,
    /// `None` for MSL read from the store, which is built when it is loaded.
    pub pipeline: Option<MetalKernel>,
}

impl ArtifactCompiler for MetalArtifactCompiler {
    type Variant = ();
    type Lowered = CompiledKernel<MetalCompiler>;
    type Artifact = MetalArtifact;

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

    /// Has the device build the library and the pipeline too: the driver's
    /// compilation from MSL is most of what a Metal kernel costs.
    fn finalize(
        &self,
        _id: &ArtifactId<()>,
        mut lowered: Self::Lowered,
    ) -> Result<MetalArtifact, LaunchError> {
        let cube_dim = lowered.cube_dim;
        let entry = MslCacheEntry {
            io: lowered.io.take(),
            entrypoint_name: lowered.entrypoint_name,
            cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
            source: lowered.source,
        };
        let pipeline = self.build(&entry)?;
        Ok(MetalArtifact {
            entry,
            pipeline: Some(pipeline),
        })
    }
}

impl MetalArtifactCompiler {
    /// Has the device build `entry`'s library and pipeline.
    pub fn build(&self, entry: &MslCacheEntry) -> Result<MetalKernel, CompilationError> {
        let cube_dim = entry.cube_dim.into();
        let pipeline = self.create_pipeline(&entry.source, &entry.entrypoint_name, cube_dim)?;
        Ok(MetalKernel {
            pipeline,
            cube_dim,
            io: entry.io.clone().map(std::sync::Arc::from),
        })
    }

    /// Creates a compute pipeline from MSL source code.
    fn create_pipeline(
        &self,
        source: &str,
        entrypoint_name: &str,
        cube_dim: CubeDim,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, CompilationError> {
        use objc2_metal::MTLDevice;

        let source_ns = NSString::from_str(source);

        let library = self
            .device
            .newLibraryWithSource_options_error(&source_ns, Some(&msl_compile_options()))
            .map_err(|err| CompilationError::Generic {
                reason: format!("Failed to compile MSL: {:?}", err.localizedDescription()),
                backtrace: BackTrace::capture(),
            })?;

        let entrypoint_ns = NSString::from_str(entrypoint_name);
        let function = library.newFunctionWithName(&entrypoint_ns).ok_or_else(|| {
            CompilationError::Generic {
                reason: format!("Function '{}' not found in library", entrypoint_name),
                backtrace: BackTrace::capture(),
            }
        })?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| CompilationError::Generic {
                reason: format!(
                    "Failed to create compute pipeline: {:?}",
                    err.localizedDescription()
                ),
                backtrace: BackTrace::capture(),
            })?;

        // A kernel's register and shared-memory use can cap its threadgroup size below the
        // device limit; exceeding it fails the dispatch on the GPU, so reject it at compile time.
        let max_units = pipeline.maxTotalThreadsPerThreadgroup();
        let requested = (cube_dim.x as usize) * (cube_dim.y as usize) * (cube_dim.z as usize);
        if requested > max_units {
            return Err(CompilationError::Generic {
                reason: format!(
                    "Cube dim {}x{}x{} ({requested} units) exceeds this kernel's limit of \
                     {max_units} threads per threadgroup",
                    cube_dim.x, cube_dim.y, cube_dim.z
                ),
                backtrace: BackTrace::capture(),
            });
        }

        Ok(pipeline)
    }
}

/// The options the driver compiles MSL with. Made per compilation: they are
/// a few setters, and an options object is not shared across threads.
fn msl_compile_options() -> Retained<MTLCompileOptions> {
    let msl_compile_options = MTLCompileOptions::new();
    // MSL 3.2 for lambdas.
    msl_compile_options.setLanguageVersion(MTLLanguageVersion::Version3_2);
    // Compile with IEEE-safe math by default; per-op fast math is opted into separately.
    // `mathMode` disables FP reassociation/contraction, `mathFloatingPointFunctions`
    // keeps math functions precise.
    msl_compile_options.setMathMode(MTLMathMode::Safe);
    msl_compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    msl_compile_options
}
