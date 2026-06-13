use crate::MetalCompiler;
use cubecl_common::backtrace::BackTrace;
use cubecl_core::prelude::*;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use hashbrown::HashMap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLMathMode,
};
use std::sync::Arc;

use cubecl_common::cache::{Cache, CacheOption};

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
    msl_cache: Option<Cache<String, MslCacheEntry>>,
    compilation_options: cubecl_cpp::shared::CompilationOptions,
    msl_compile_options: Retained<MTLCompileOptions>,
}

impl MetalContext {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        compilation_options: cubecl_cpp::shared::CompilationOptions,
    ) -> Self {
        let msl_compile_options = MTLCompileOptions::new();
        // We rely on Metal 3 features (e.g. atomic_float, simdgroup_matrix).
        msl_compile_options.setLanguageVersion(MTLLanguageVersion::Version3_0);
        // Default `Fast` math lets the compiler reassociate/contract FP arithmetic
        // (e.g. folding `log(exp(x))` back to `x`), breaking lowerings like `expm1`.
        // Use IEEE-safe math. (Per-kernel fast-math isn't wired through yet; the
        // `Fast*` codegen variants are CUDA-only, so all Metal math is precise.)
        msl_compile_options.setMathMode(MTLMathMode::Safe);
        // `mathMode` is only honored from MSL 3.1; on 3.0 the deprecated
        // `fastMathEnabled` flag governs and defaults to `true`, so disable it too.
        #[allow(deprecated)]
        msl_compile_options.setFastMathEnabled(false);

        Self {
            device,
            compiled_kernels: HashMap::new(),
            msl_cache: {
                use cubecl_runtime::config::RuntimeConfig;
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(Cache::new(
                        "msl",
                        CacheOption::default().name("metal").root(root),
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
        logger: Arc<ServerLogger>,
    ) -> Result<CompiledKernel, cubecl_runtime::compiler::CompilationError> {
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

        if logger.compilation_activated() {
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
