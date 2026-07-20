use crate::MetalCompiler;
use cubecl_core::prelude::*;
use cubecl_core::server::{LaunchError, ResourceLimitError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode,
};
use std::sync::Arc;

use cubecl_environment::persistence::{KvStore, KvStoreOptions};

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) cube_dim: CubeDim,
    /// Shared memory usage in bytes, validated against the device limit.
    pub(crate) shared_memory_bytes: usize,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct MslCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    source: String,
}

/// Compiles `CubeCL` IR to MSL and on to `MTLComputePipelineState`, caching results.
#[derive(Debug)]
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    compiled_kernels: HashMap<KernelId, CompiledKernel>,
    /// On-disk MSL source cache for faster recompilation across runs.
    msl_cache: Option<KvStore<String, MslCacheEntry>>,
    compilation_options: cubecl_cpp::shared::CompilationOptions,
    msl_compile_options: Retained<MTLCompileOptions>,
}

impl MetalContext {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        compilation_options: cubecl_cpp::shared::CompilationOptions,
    ) -> Self {
        let msl_compile_options = MTLCompileOptions::new();
        // MSL 3.1 for native `bfloat`.
        msl_compile_options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        // Compile with IEEE-safe math by default; per-op fast math is opted into separately.
        // `mathMode` disables FP reassociation/contraction, `mathFloatingPointFunctions`
        // keeps math functions precise.
        msl_compile_options.setMathMode(MTLMathMode::Safe);
        msl_compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);

        Self {
            device,
            compiled_kernels: HashMap::new(),
            msl_cache: {
                use cubecl_runtime::config::RuntimeConfig;
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if config.compilation.cache {
                    Some(KvStore::open(
                        "msl",
                        KvStoreOptions::default().name("metal"),
                    ))
                } else {
                    None
                }
            },
            compilation_options,
            msl_compile_options,
        }
    }

    /// Compiles a kernel and caches the result.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<MetalCompiler>>,
        mode: ExecutionMode,
        max_shared_memory_size: usize,
        logger: Arc<ServerLogger>,
    ) -> Result<CompiledKernel, LaunchError> {
        if let Some(compiled) = self.compiled_kernels.get(kernel_id) {
            return Ok(compiled.clone());
        }

        if let Some(cache) = &self.msl_cache {
            let cache_key = kernel_id.stable_format();
            if let Some(entry) = cache.get(&cache_key) {
                log::trace!("Using MSL cache");

                let compiled = self.create_pipeline_from_source(
                    &entry.source,
                    &entry.entrypoint_name,
                    CubeDim {
                        x: entry.cube_dim.0,
                        y: entry.cube_dim.1,
                        z: entry.cube_dim.2,
                    },
                )?;

                self.compiled_kernels
                    .insert(kernel_id.clone(), compiled.clone());
                return Ok(compiled);
            }
        }

        log::trace!("Compiling kernel to MSL");

        let mut kernel_compiled = kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            kernel.address_type(),
        )?;

        if logger.compilation_source_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("msl", kernel_id.clone()));
        }

        logger.log_compilation(&kernel_compiled);

        let entrypoint_name = kernel_compiled.entrypoint_name.clone();
        let cube_dim = kernel_compiled.cube_dim;
        let source = kernel_compiled.source.clone();
        let shared_memory_bytes = kernel_compiled
            .repr
            .as_ref()
            .map(|r| r.shared_memory_size())
            .unwrap_or(0);

        // Check before creating the pipeline: Metal would reject the kernel there anyway,
        // but with an opaque compilation error instead of a resource limit error.
        if shared_memory_bytes > max_shared_memory_size {
            return Err(LaunchError::TooManyResources(
                ResourceLimitError::SharedMemory {
                    requested: shared_memory_bytes,
                    max: max_shared_memory_size,
                    backtrace: BackTrace::capture(),
                },
            ));
        }

        let mut compiled = self.create_pipeline_from_source(&source, &entrypoint_name, cube_dim)?;
        compiled.shared_memory_bytes = shared_memory_bytes;

        if let Some(cache) = &mut self.msl_cache {
            let cache_key = kernel_id.stable_format();
            let result = cache.insert(
                cache_key,
                MslCacheEntry {
                    entrypoint_name,
                    cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
                    source,
                },
            );
            if let Err(err) = result {
                log::warn!("Unable to save MSL to cache: {err:?}");
            }
        }

        self.compiled_kernels
            .insert(kernel_id.clone(), compiled.clone());
        Ok(compiled)
    }

    /// Creates a compute pipeline from MSL source code.
    fn create_pipeline_from_source(
        &self,
        source: &str,
        entrypoint_name: &str,
        cube_dim: CubeDim,
    ) -> Result<CompiledKernel, cubecl_runtime::compiler::CompilationError> {
        use objc2_metal::MTLDevice;

        let source_ns = NSString::from_str(source);

        let library = self
            .device
            .newLibraryWithSource_options_error(&source_ns, Some(&self.msl_compile_options))
            .map_err(|err| cubecl_runtime::compiler::CompilationError::Generic {
                reason: format!("Failed to compile MSL: {:?}", err.localizedDescription()),
                backtrace: BackTrace::capture(),
            })?;

        let entrypoint_ns = NSString::from_str(entrypoint_name);
        let function = library.newFunctionWithName(&entrypoint_ns).ok_or_else(|| {
            cubecl_runtime::compiler::CompilationError::Generic {
                reason: format!("Function '{}' not found in library", entrypoint_name),
                backtrace: BackTrace::capture(),
            }
        })?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| cubecl_runtime::compiler::CompilationError::Generic {
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
            return Err(cubecl_runtime::compiler::CompilationError::Generic {
                reason: format!(
                    "Cube dim {}x{}x{} ({requested} units) exceeds this kernel's limit of \
                     {max_units} threads per threadgroup",
                    cube_dim.x, cube_dim.y, cube_dim.z
                ),
                backtrace: BackTrace::capture(),
            });
        }

        Ok(CompiledKernel {
            pipeline,
            cube_dim,
            shared_memory_bytes: 0,
        })
    }

    /// Returns the compiled kernel for `kernel_id`, if present.
    pub fn get_kernel(&self, kernel_id: &KernelId) -> Option<&CompiledKernel> {
        self.compiled_kernels.get(kernel_id)
    }
}

// SAFETY: Only accessed from the server thread. Pipeline states are immutable once created.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}
