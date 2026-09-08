use cubecl_cpp::cuda::arch::CudaArchitecture;
use cubecl_cpp::formatter::format_cpp;
use cubecl_environment::backtrace::BackTrace;
use cubecl_runtime::kernel::BufferIOAttr;
use cubecl_runtime::{
    compiler::{CompilationError, build_id_hash},
    validation::{validate_cube_dim, validate_units},
};

use crate::compiler::{CudaBackend, CudaCompilationOptions, CudaCompiler, CudaRepresentation};
use crate::compute::events::EventProfiler;
use crate::compute::stream::Stream;
use crate::install::{cccl_include_path, include_path};
use cubecl_core::{
    hash::{StableHash, StableHasher},
    ir::DeviceProperties,
    prelude::*,
    server::ResourceLimitError,
};
use cubecl_environment::persistence::Store;
use cubecl_runtime::{
    compiler::KernelCacheKey,
    kernel::{CompiledKernel, CubeKernel},
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
    modules: CompilationCache<KernelId, CudaCompiledKernel>,
    ptx_cache: Option<Store<KernelCacheKey, PtxCacheEntry>>,
    /// Cache mapping C++ code hashes to the key that first compiled them. We can skip the slow CUDA
    /// compiler if we already have a compiled artifact for the same code.
    ///
    /// C++ only: the LLVM backend emits PTX directly, so there is no intermediate source to
    /// key a second line on.
    second_line_ptx_cache: Option<Store<StableHash, KernelCacheKey>>,
    pub profiler: EventProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CudaCompilationOptions,
    pub properties: DeviceProperties,
    build_id: StableHash,
}

#[derive(Debug)]
pub struct CudaCompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the compiler's answer, carried here because on a cache hit nothing
    /// else of the compilation survives. `None` for entries persisted before
    /// the answer existed, which the launch path reads as every buffer both
    /// read and written.
    io: Option<Arc<[BufferIOAttr]>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    ptx: Vec<std::ffi::c_char>,
    /// See [`CudaCompiledKernel::io`]; defaulted for entries persisted before the
    /// field existed.
    #[serde(default)]
    io: Option<Vec<BufferIOAttr>>,
}

/// The namespace a backend's compiled artifacts live under.
///
/// Both backends emit PTX, so without the backend in the key a stale artifact from one would
/// load and run happily under the other -- tests passing while measuring nothing.
fn cache_namespace(fingerprint: &str, backend: CudaBackend) -> String {
    let backend = match backend {
        CudaBackend::Cpp => "cpp",
        CudaBackend::Llvm => "llvm",
    };
    format!("{fingerprint}-{backend}")
}

impl CudaContext {
    /// `backend` is which one compiles here; see [`cache_namespace`].
    pub fn new(
        compilation_options: CudaCompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
        backend: CudaBackend,
    ) -> Self {
        let fingerprint = cache_namespace(&format!("ptx_sm{}", arch.version), backend);
        let ptx_cache = compilation_store("cuda", &fingerprint);
        let second_line_ptx_cache = compilation_store("cuda-second-line", fingerprint);

        Self {
            context,
            modules: CompilationCache::mirroring(&ptx_cache),
            ptx_cache,
            second_line_ptx_cache,
            arch,
            profiler: EventProfiler::default(),
            compilation_options,
            properties,
            build_id: build_id_hash(),
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
            let key = KernelCacheKey::new(kernel_id, self.build_id);

            if let Some(entry) = cache.remove(&key) {
                log::trace!("Using PTX cache");

                self.load_ptx(
                    entry.ptx,
                    kernel_id.clone(),
                    entry.entrypoint_name,
                    kernel_id.cube_dim.into(),
                    entry.shared_mem_bytes,
                    entry.io.map(Arc::from),
                )?;
                return Ok(Ok(()));
            }
            Some(key)
        } else {
            None
        };
        Ok(Err(key))
    }

    /// Compiles a kernel.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeKernel>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let key = match self.try_load_cached(kernel_id)? {
            Ok(()) => return Ok(()),
            Err(key) => key,
        };

        log::trace!("Compiling kernel");

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        let definition = kernel.define();
        let jitc_kernel = CompiledKernel::compile(
            &*kernel,
            definition,
            &mut CudaCompiler::default(),
            &self.compilation_options,
        )?;

        self.validate_shared(&jitc_kernel.repr)?;

        self.load_jit_kernel(kernel_id, key, jitc_kernel, logger)
    }

    /// Loads what the compiler produced, by the route that backend's output takes.
    fn load_jit_kernel(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        jitc_kernel: CompiledKernel<CudaCompiler>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        match &jitc_kernel.repr {
            Some(CudaRepresentation::Cpp(_)) => {
                self.load_transpiled(kernel_id, key, jitc_kernel, logger)
            }
            Some(CudaRepresentation::Llvm(_)) => {
                self.load_emitted_ptx(kernel_id, key, jitc_kernel, logger)
            }
            // A precompiled kernel: its text passed the language check in
            // `CompiledKernel::compile`, so it is whatever the default backend reads. CUDA
            // C++ goes through NVRTC like a transpiled kernel; the LLVM backend produces PTX
            // from the dialect and has no route for text.
            None => match CudaBackend::default() {
                CudaBackend::Cpp => self.load_transpiled(kernel_id, key, jitc_kernel, logger),
                CudaBackend::Llvm => Err(CompilationError::Generic {
                    reason: "the LLVM backend cannot load a precompiled kernel: it has no text to \
                         compile from"
                        .to_string(),
                    backtrace: BackTrace::capture(),
                }
                .into()),
            },
        }
    }

    /// Turns a kernel the LLVM backend just compiled into a loaded module.
    ///
    /// What it hands back is already PTX, so no `nvrtc*` call belongs here: the bytes go
    /// straight to [`Self::load_ptx`], exactly as a cache hit would. The driver still JITs
    /// them when the module loads, which is what it does with NVRTC's output too.
    fn load_emitted_ptx(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<CudaCompiler>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let Some(CudaRepresentation::Llvm(module)) = &jitc_kernel.repr else {
            unreachable!("dispatched on the representation");
        };

        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("ll", kernel_id.clone()));
        }
        logger.log_compilation(&jitc_kernel);

        let ptx = module.ptx.clone();
        let shared_mem_bytes = module.shared_memory_size;
        let io = jitc_kernel.io.take();
        let entrypoint_name = jitc_kernel.entrypoint_name.clone();

        self.load_ptx(
            ptx.clone(),
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            shared_mem_bytes,
            io.clone().map(Arc::from),
        )?;

        // Cached after the load, so PTX the driver rejects is not handed back on the next run.
        // `try_load_cached` hands back a key exactly when there is a cache to put it in. No
        // second-line entry: that cache is keyed on generated C++ source, which this backend
        // never produces.
        if let Some((cache, key)) = self.ptx_cache.as_mut().zip(key) {
            store_compiled(
                cache,
                key,
                PtxCacheEntry {
                    entrypoint_name,
                    shared_mem_bytes,
                    ptx,
                    io,
                },
            );
        }
        Ok(())
    }

    /// Turns a kernel the C++ backend just transpiled into a loaded module, by running its
    /// source through NVRTC first.
    fn load_transpiled(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<CudaCompiler>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }

        let cpp_hash = if let Some(cache) = self.ptx_cache.as_mut() {
            let key = key.unwrap();
            let second_line_cache = self.second_line_ptx_cache.as_mut().unwrap();
            let cpp_hash = StableHasher::hash_one(&jitc_kernel.source);

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

        logger.log_compilation(&jitc_kernel);

        let ptx = self.compile_to_ptx(&jitc_kernel.source)?;

        let cube_dim = jitc_kernel.cube_dim;
        let io = jitc_kernel.io.take();
        // A precompiled kernel has no representation to read the size from: it declares its
        // shared memory statically, so the launch reserves none.
        let shared_mem_bytes = jitc_kernel
            .repr
            .as_ref()
            .map(|repr| repr.shared_memory_size())
            .unwrap_or(0);

        if let Some(cache) = &mut self.ptx_cache {
            let second_line_cache = self.second_line_ptx_cache.as_mut().unwrap();
            let key = key.unwrap();
            store_compiled(
                cache,
                key,
                PtxCacheEntry {
                    entrypoint_name: jitc_kernel.entrypoint_name.clone(),
                    shared_mem_bytes,
                    ptx: ptx.clone(),
                    io: io.clone(),
                },
            );
            store_compiled(second_line_cache, cpp_hash.unwrap(), key);
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            cube_dim,
            shared_mem_bytes,
            io.map(Arc::from),
        )?;
        Ok(())
    }

    /// Compiles `source` to PTX with NVRTC.
    ///
    /// # Errors
    ///
    /// [`CompilationError::Generic`] carrying the compiler's own log, and the source that
    /// produced it, so a kernel the compiler refuses says why.
    fn compile_to_ptx(&self, source: &str) -> Result<Vec<c_char>, CompilationError> {
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

        // SAFETY: Calling NVRTC FFI to create, compile, and extract PTX from a program.
        // The `CString` source is null-terminated and outlives the program. On compilation
        // failure, the error log is retrieved and reported before returning.
        unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let c_source = CString::from_str(source).unwrap();
            let program = cudarc::nvrtc::result::create_program(c_source.as_c_str(), None)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("{err}"),
                    backtrace: BackTrace::capture(),
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
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                })?;
            };
            cudarc::nvrtc::result::get_ptx(program).map_err(|err| CompilationError::Generic {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })
        }
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
        io: Option<Arc<[BufferIOAttr]>>,
    ) -> Result<(), CompilationError> {
        dump_ptx(&kernel_id, &ptx);

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
            CudaCompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
                io,
            },
        );

        Ok(())
    }

    /// What the compiled kernel does with each buffer binding, by buffer
    /// position — `None` when the kernel is not loaded or predates the
    /// answer, which the launch path reads as every buffer both read and
    /// written.
    pub fn kernel_io(&mut self, kernel_id: &KernelId) -> Option<Arc<[BufferIOAttr]>> {
        self.modules
            .get(kernel_id)
            .and_then(|kernel| kernel.io.clone())
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

    fn validate_shared(&self, repr: &Option<CudaRepresentation>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size());
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

#[cfg(test)]
mod tests {
    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    fn cache_namespace_separates_backends() {
        assert_ne!(
            super::cache_namespace("ptx_sm86", crate::compiler::CudaBackend::Cpp),
            super::cache_namespace("ptx_sm86", crate::compiler::CudaBackend::Llvm),
        );
    }
}

/// Writes the PTX for `kernel_id` under the directory named by `CUBECL_CUDA_DUMP_PTX`, if that
/// variable is set.
///
/// Both backends end up here, so the two can be diffed instruction for instruction. That is the
/// comparison worth making: the PTX is deterministic, where a wall-clock measurement on a laptop
/// GPU is not.
fn dump_ptx(kernel_id: &KernelId, ptx: &[c_char]) {
    let Some(dir) = std::env::var_os("CUBECL_CUDA_DUMP_PTX") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    // The id is a type path plus its settings, so it carries every character a path cannot.
    let name: String = kernel_id
        .to_string()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    // ...and is long enough to blow past NAME_MAX on its own.
    let name = &name[name.len().saturating_sub(180)..];

    // SAFETY: the PTX handed to the driver is a null-terminated C string.
    let text = unsafe { CStr::from_ptr(ptx.as_ptr()) };
    let _ = std::fs::write(dir.join(format!("{name}.ptx")), text.to_bytes());
}
