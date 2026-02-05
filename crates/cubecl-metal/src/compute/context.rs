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
};
use std::sync::Arc;

use cubecl_common::cache::{Cache, CacheOption};

/// Compiled Metal kernel with pipeline state
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) cube_dim: CubeDim,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct MslCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    source: String,
}

/// Metal compilation context
///
/// Handles kernel compilation and caching for Metal compute pipelines.
/// Manages compilation from `CubeCL` IR to MSL source, then to `MTLLibrary`
/// and `MTLComputePipelineState`.
#[derive(Debug)]
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Cache of compiled kernels by kernel ID
    compiled_kernels: HashMap<KernelId, CompiledKernel>,
    /// Optional MSL source cache for faster recompilation
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

        Self {
            device,
            compiled_kernels: HashMap::new(),
            msl_cache: {
                let config = cubecl_runtime::config::GlobalConfig::get();
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

    /// Compiles a kernel and caches the result
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<MetalCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<CompiledKernel, cubecl_runtime::compiler::CompilationError> {
        // Check if already compiled
        if let Some(compiled) = self.compiled_kernels.get(kernel_id) {
            return Ok(compiled.clone());
        }

        // Try loading from cache
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

        // Compile kernel using cubecl-cpp
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

        // Create pipeline from source
        let compiled = self.create_pipeline_from_source(&source, &entrypoint_name, cube_dim)?;

        // Cache the MSL source
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

    /// Creates a compute pipeline from MSL source code
    fn create_pipeline_from_source(
        &self,
        source: &str,
        entrypoint_name: &str,
        cube_dim: CubeDim,
    ) -> Result<CompiledKernel, cubecl_runtime::compiler::CompilationError> {
        use objc2_metal::MTLDevice;

        // Convert source to NSString
        let source_ns = NSString::from_str(source);

        // Compile source to library
        let library = self
            .device
            .newLibraryWithSource_options_error(&source_ns, Some(&self.msl_compile_options))
            .map_err(|err| cubecl_runtime::compiler::CompilationError::Generic {
                reason: format!("Failed to compile MSL: {:?}", err.localizedDescription()),
                backtrace: BackTrace::capture(),
            })?;

        // Get the compute function
        let entrypoint_ns = NSString::from_str(entrypoint_name);
        let function = library.newFunctionWithName(&entrypoint_ns).ok_or_else(|| {
            cubecl_runtime::compiler::CompilationError::Generic {
                reason: format!("Function '{}' not found in library", entrypoint_name),
                backtrace: BackTrace::capture(),
            }
        })?;

        // Create compute pipeline state
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

        Ok(CompiledKernel { pipeline, cube_dim })
    }

    /// Gets a compiled kernel by ID
    pub fn get_kernel(&self, kernel_id: &KernelId) -> Option<&CompiledKernel> {
        self.compiled_kernels.get(kernel_id)
    }
}

// SAFETY: Metal objects are thread-safe and can be safely sent across threads
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}
