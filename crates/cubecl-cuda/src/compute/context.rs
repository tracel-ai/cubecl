use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};
use cubecl_environment::backtrace::BackTrace;
use cubecl_runtime::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};

use crate::{CudaCompiler, compute::stream::Stream};
use crate::{
    CudaComputeKernel,
    install::{cccl_include_path, include_path},
};
use cubecl_core::{
    hash::{StableHash, StableHasher},
    ir::DeviceProperties,
    prelude::*,
    server::ResourceLimitError,
};
use cubecl_environment::persistence::Store;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{
    compiler::{CubeTask, KernelCacheKey},
    logging::ServerLogger,
};
use cudarc::driver::DriverError;
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute};
use std::ffi::CString;
use std::ffi::c_char;
use std::str::FromStr;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};

use cubecl_runtime::compiler::{CompilationCache, compilation_store, store_compiled};

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    /// The modules loaded on the device, in front of [`Self::ptx_cache`].
    ///
    /// An environment switch drops these, and nothing unloads the modules they
    /// name: see [`CudaContext::is_loaded`].
    modules: CompilationCache<KernelId, CompiledKernel>,
    ptx_cache: Option<Store<KernelCacheKey, PtxCacheEntry>>,
    /// Cache mapping C++ code hashes to the key that first compiled them. We can skip the slow CUDA
    /// compiler if we already have a compiled artifact for the same code.
    second_line_ptx_cache: Option<Store<StableHash, KernelCacheKey>>,
    pub timestamps: TimestampProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
}

#[derive(Debug)]
pub struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    ptx: Vec<std::ffi::c_char>,
}

impl CudaContext {
    pub fn new(
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        let ptx_cache = compilation_store("cuda", format!("ptx_sm{}", arch.version));
        let second_line_ptx_cache =
            compilation_store("cuda-second-line", format!("ptx_sm{}", arch.version));

        Self {
            context,
            modules: CompilationCache::mirroring(&ptx_cache),
            ptx_cache,
            second_line_ptx_cache,
            arch,
            timestamps: TimestampProfiler::default(),
            compilation_options,
            properties,
        }
    }

    /// Whether `kernel_id` is already loaded on the device.
    ///
    /// An environment switch drops the whole cache, so the new environment's
    /// PTX store is filled rather than bypassed. The modules those entries
    /// named stay resident: nothing calls `cuModuleUnload`, here or anywhere
    /// else in this context, and unloading one a stream still has queued work
    /// against would be unsound. A process that switches environments a handful
    /// of times at startup pays a bounded price; one that switches repeatedly
    /// grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn is_loaded(&mut self, kernel_id: &KernelId) -> bool {
        self.modules.contains(kernel_id)
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }

    fn try_load_cached(
        &mut self,
        kernel_id: &KernelId,
    ) -> Result<Result<(), Option<KernelCacheKey>>, CompilationError> {
        let key = if let Some(cache) = self.ptx_cache.as_mut() {
            let key = KernelCacheKey::new(kernel_id);

            if let Some(entry) = cache.remove(&key) {
                log::trace!("Using PTX cache");

                self.load_ptx(
                    entry.ptx,
                    kernel_id.clone(),
                    entry.entrypoint_name,
                    kernel_id.cube_dim.into(),
                    entry.shared_mem_bytes,
                )?;
                return Ok(Ok(()));
            }
            Some(key)
        } else {
            None
        };
        Ok(Err(key))
    }

    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let definition = kernel.define();

        let key = match self.try_load_cached(kernel_id)? {
            Ok(()) => return Ok(()),
            Err(key) => key,
        };

        log::trace!("Compiling kernel");

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        let mut kernel_compiled = kernel.compile(
            definition,
            &mut Default::default(),
            &self.compilation_options,
        )?;

        self.validate_shared(&kernel_compiled.repr)?;

        if logger.compilation_source_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let cpp_hash = if let Some(cache) = self.ptx_cache.as_mut() {
            let key = key.unwrap();
            let second_line_cache = self.second_line_ptx_cache.as_mut().unwrap();
            let cpp_hash = StableHasher::hash_one(&kernel_compiled.source);

            if let Some(old_key) = second_line_cache.purge_key(&cpp_hash)
                && let Some(entry) = cache.purge_key(&old_key)
            {
                log::trace!("Using second-line PTX cache");
                store_compiled(cache, key, entry);
                store_compiled(second_line_cache, cpp_hash, key);
                self.try_load_cached(kernel_id)?
                    .expect("Should be cached now");
                return Ok(());
            }

            Some(cpp_hash)
        } else {
            None
        };

        let cube_dim = kernel_compiled.cube_dim;
        let arch = if self.arch.version >= 90 {
            format!("--gpu-architecture=sm_{}a", self.arch)
        } else {
            format!("--gpu-architecture=sm_{}", self.arch)
        };

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_path = cccl_include_path();
        let cccl_include_option = format!("--include-path={}", cccl_include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if cccl_include_path.exists() {
            options.push(&cccl_include_option);
        }

        logger.log_compilation(&kernel_compiled);

        // SAFETY: Calling NVRTC FFI to create, compile, and extract PTX from a program.
        // The `CString` source is null-terminated and outlives the program. On compilation
        // failure, the error log is retrieved and reported before returning.
        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::from_str(&kernel_compiled.source).unwrap();
            let program =
                cudarc::nvrtc::result::create_program(source.as_c_str(), None).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;
            if cudarc::nvrtc::result::compile_program(program, &options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;

                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{}", kernel_compiled.source),
                    backtrace: BackTrace::capture(),
                })?;
            };
            cudarc::nvrtc::result::get_ptx(program).map_err(|err| CompilationError::Generic {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?
        };

        let repr = kernel_compiled.repr.unwrap();

        if let Some(cache) = &mut self.ptx_cache {
            let second_line_cache = self.second_line_ptx_cache.as_mut().unwrap();
            let key = key.unwrap();
            store_compiled(
                cache,
                key,
                PtxCacheEntry {
                    entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                    shared_mem_bytes: repr.shared_memory_size,
                    ptx: ptx.clone(),
                },
            );
            store_compiled(second_line_cache, cpp_hash.unwrap(), key);
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            repr.shared_memory_size,
        )?;
        Ok(())
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name).unwrap();
        // SAFETY: `ptx` is a valid null-terminated PTX binary from NVRTC. `func_name` is a
        // null-terminated `CString` matching the kernel entry point in the compiled module.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the PTX: {err}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        self.modules.insert(
            kernel_id.clone(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );

        Ok(())
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &mut [*mut c_void],
    ) -> Result<(), LaunchError> {
        let kernel = self.modules.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                resources,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?;
        };

        Ok(())
    }

    fn validate_shared(&self, repr: &Option<CudaComputeKernel>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size);
        let max = self.properties.hardware.max_shared_memory_size;
        if let Some(requested) = requested
            && requested > max
        {
            Err(ResourceLimitError::SharedMemory {
                requested,
                max,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}
