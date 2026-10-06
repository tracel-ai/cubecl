//! The half of Metal's compilation that runs where the server does: the MSL
//! store, and the pipelines the driver builds from what it and the compiler
//! produce.

use crate::compute::artifact::{MetalArtifactCompiler, MslCacheEntry};
use cubecl_core::prelude::*;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{ArtifactId, ArtifactStore, CompilationError, CompilationTarget};
use cubecl_server::kernel::BufferIOAttr;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode,
};

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
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// MSL kept between runs, which saves the compilation to MSL; the
    /// driver still builds the library from it.
    msl_store: ArtifactStore<MslCacheEntry>,
    msl_compile_options: Retained<MTLCompileOptions>,
}

impl MetalPipelines {
    pub fn new(
        compiler: MetalArtifactCompiler,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Self {
        let msl_compile_options = MTLCompileOptions::new();
        // MSL 3.2 for lambdas.
        msl_compile_options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        // Compile with IEEE-safe math by default; per-op fast math is opted into separately.
        // `mathMode` disables FP reassociation/contraction, `mathFloatingPointFunctions`
        // keeps math functions precise.
        msl_compile_options.setMathMode(MTLMathMode::Safe);
        msl_compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);

        // The MSL is emitted from device-derived compilation options and
        // validated against the device's shared-memory limit, so the device
        // name is what keeps a bundle shipped across machines from serving
        // sources built for another GPU.
        let device_key = device.name().to_string();

        Self {
            compiler,
            msl_store: ArtifactStore::new("metal", format!("msl_{device_key}")),
            device,
            msl_compile_options,
        }
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
            .newLibraryWithSource_options_error(&source_ns, Some(&self.msl_compile_options))
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

impl CompilationTarget for MetalPipelines {
    type Compiler = MetalArtifactCompiler;
    type Loaded = CompiledKernel;

    fn compiler(&self) -> &MetalArtifactCompiler {
        &self.compiler
    }

    fn persists(&self) -> bool {
        self.msl_store.persists()
    }

    fn stored(&mut self, id: &ArtifactId<()>) -> Option<MslCacheEntry> {
        self.msl_store.take(&id.kernel)
    }

    fn load(
        &mut self,
        _id: &ArtifactId<()>,
        artifact: &MslCacheEntry,
    ) -> Result<CompiledKernel, CompilationError> {
        let cube_dim = artifact.cube_dim.into();
        let pipeline =
            self.create_pipeline(&artifact.source, &artifact.entrypoint_name, cube_dim)?;
        Ok(CompiledKernel {
            pipeline,
            cube_dim,
            io: artifact.io.clone().map(std::sync::Arc::from),
        })
    }

    fn store(
        &mut self,
        id: &ArtifactId<()>,
        artifact: MslCacheEntry,
        _source: Option<&str>,
    ) -> bool {
        self.msl_store.keep(&id.kernel, artifact, None)
    }
}
