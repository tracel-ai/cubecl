use super::storage::gpu::GpuResource;
use crate::runtime::HipCompiler;
use crate::{compute::stream::Stream, runtime::HipComputeKernel};
use cubecl_core::{
    hash::{StableHash, StableHasher},
    ir::DeviceProperties,
    prelude::*,
    server::ResourceLimitError,
};
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::persistence::Store;
use cubecl_hip_sys::{HIP_SUCCESS, get_hip_include_path, hiprtcResult_HIPRTC_SUCCESS};
use cubecl_runtime::compiler::{
    CompilationCache, build_id_hash, compilation_store, store_compiled,
};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};
use cubecl_runtime::{
    compiler::{CubeTask, KernelCacheKey},
    logging::ServerLogger,
};
use serde::Deserialize;
use serde::Serialize;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct HipContext {
    /// The modules loaded on the device, in front of
    /// [`Self::compilation_cache`].
    ///
    /// An environment switch drops these, and nothing unloads the modules they
    /// name: see [`HipContext::is_loaded`].
    modules: CompilationCache<KernelId, HipCompiledKernel>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<Store<KernelCacheKey, CompilationCacheEntry>>,
    /// Cache mapping C++ code hashes to the key that first compiled them. We can skip the slow HIP
    /// compiler if we already have a compiled artifact for the same code.
    pub second_line_compilation_cache: Option<Store<StableHash, KernelCacheKey>>,
    build_id: StableHash,
}

#[derive(Debug)]
pub struct HipCompiledKernel {
    _module: cubecl_hip_sys::hipModule_t,
    func: cubecl_hip_sys::hipFunction_t,
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    binary: Vec<i8>,
}

impl HipContext {
    /// `fingerprint` is the one the runtime already published on
    /// [`DeviceProperties::identity`], rather than one rebuilt here: the
    /// namespace a kernel is cached under and the identity a bundle is stamped
    /// with have to be the same string, and the only way to guarantee that is
    /// for there to be one string.
    pub fn new(
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
        fingerprint: String,
    ) -> Self {
        let compilation_cache = compilation_store("hip", &fingerprint);
        let second_line_compilation_cache = compilation_store("hip-second-line", fingerprint);

        Self {
            modules: CompilationCache::mirroring(&compilation_cache),
            timestamps: TimestampProfiler::default(),
            compilation_options,
            compilation_cache,
            second_line_compilation_cache,
            properties,
            build_id: build_id_hash(),
        }
    }

    /// Whether `kernel_id` is already loaded on the device.
    ///
    /// An environment switch drops the whole cache, so the new environment's
    /// store is filled rather than bypassed. The modules those entries named
    /// stay resident: nothing calls `hipModuleUnload`, here or anywhere else in
    /// this context, and unloading one a stream still has queued work against
    /// would be unsound. A process that switches environments a handful of
    /// times at startup pays a bounded price; one that switches repeatedly
    /// grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn is_loaded(&mut self, kernel_id: &KernelId) -> bool {
        self.modules.contains(kernel_id)
    }

    fn try_load_cached(
        &mut self,
        kernel_id: &KernelId,
    ) -> Result<Result<(), Option<KernelCacheKey>>, CompilationError> {
        let key = if let Some(cache) = self.compilation_cache.as_mut() {
            let key = KernelCacheKey::new(kernel_id, self.build_id);

            if let Some(entry) = cache.remove(&key) {
                log::trace!("Using compilation cache");

                self.load_compiled_binary(
                    entry.binary,
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

    /// Compiles a kernel.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        cube_kernel: Box<dyn CubeTask<HipCompiler>>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let key = match self.try_load_cached(kernel_id)? {
            Ok(()) => return Ok(()),
            Err(key) => key,
        };

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        // CubeCL compilation
        // jitc = just-in-time compiled
        let definition = cube_kernel.define();
        let mut jitc_kernel = cube_kernel.compile(
            definition,
            &mut Default::default(),
            &self.compilation_options,
        )?;

        self.validate_shared(&jitc_kernel.repr)?;

        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }
        logger.log_compilation(&jitc_kernel);

        let cpp_hash = if let Some(cache) = self.compilation_cache.as_mut() {
            let key = key.unwrap();
            let second_line_cache = self.second_line_compilation_cache.as_mut().unwrap();
            let cpp_hash = StableHasher::hash_one(&jitc_kernel.source);

            if let Some(old_key) = second_line_cache.purge_key(&cpp_hash)
                && let Some(entry) = cache.purge_key(&old_key)
            {
                log::trace!("Using second-line compilation cache");
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

        // Create HIP Program
        // SAFETY: Calling HIP RTC FFI to create a program from source. The `CString` ensures
        // the source is null-terminated. The returned `program` handle is valid on success.
        let program = unsafe {
            let source = CString::new(jitc_kernel.source.clone()).unwrap();
            let mut program: cubecl_hip_sys::hiprtcProgram = std::ptr::null_mut();

            let status = cubecl_hip_sys::hiprtcCreateProgram(
                &mut program,
                source.as_ptr(),
                std::ptr::null(), // program name seems unnecessary
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );

            if status != hiprtcResult_HIPRTC_SUCCESS {
                Err(CompilationError::Generic {
                    reason: format!(
                        "Unable to create the program from the source: HIP STATUS: {status}"
                    ),
                    backtrace: BackTrace::capture(),
                })?;
            }

            program
        };
        // Compile HIP program
        // options
        let include_path = get_hip_include_path().unwrap();
        let include_option = format!("-I{include_path}");
        let include_option_cstr = CString::new(include_option).unwrap();
        // needed for rocWMMA extension to compile
        let cpp_std_option_cstr = CString::new("--std=c++17").unwrap();
        let optimization_level = CString::new("-O3").unwrap();
        let mut options = vec![
            cpp_std_option_cstr.as_ptr(),
            include_option_cstr.as_ptr(),
            optimization_level.as_ptr(),
        ];
        // SAFETY: `program` is a valid RTC program handle created above. The `options` vector
        // contains valid null-terminated `CString` pointers that outlive this call. On failure,
        // we retrieve and report the compilation log before returning an error.
        unsafe {
            let options_ptr = options.as_mut_ptr();
            let status =
                cubecl_hip_sys::hiprtcCompileProgram(program, options.len() as i32, options_ptr);

            if status != hiprtcResult_HIPRTC_SUCCESS {
                let mut log_size: usize = 0;
                let status =
                    cubecl_hip_sys::hiprtcGetProgramLogSize(program, &mut log_size as *mut usize);

                if status != hiprtcResult_HIPRTC_SUCCESS {
                    Err(CompilationError::Generic {
                        reason: format!(
                            "An error during compilation happened, but we're unable to fetch the error log size. STATUS: {status}"
                        ),
                        backtrace: BackTrace::capture(),
                    })?;
                }

                let mut log_buffer = vec![0; log_size];
                let status = cubecl_hip_sys::hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());

                if status != hiprtcResult_HIPRTC_SUCCESS {
                    Err(CompilationError::Generic {
                        reason: format!(
                            "An error during compilation happened, but we're unable to fetch the error log content. STATUS: {status}"
                        ),
                        backtrace: BackTrace::capture(),
                    })?;
                }

                let log = CStr::from_ptr(log_buffer.as_ptr());
                let mut message = "[Compilation Error] ".to_string();
                if log_size > 0 {
                    for line in log.to_string_lossy().split('\n') {
                        if !line.is_empty() {
                            message += format!("\n    {line}").as_str();
                        }
                    }
                } else {
                    message += "\n No compilation logs found!";
                }
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{}", jitc_kernel.source),
                    backtrace: BackTrace::capture(),
                })?;
            }
        };

        // Get HIP compiled code from program
        let mut code_size: usize = 0;
        // SAFETY: `program` was successfully compiled above. `code_size` is a valid mutable
        // pointer to receive the size of the compiled binary.
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCodeSize(program, &mut code_size);
            if status != hiprtcResult_HIPRTC_SUCCESS {
                Err(CompilationError::Generic {
                    reason: format!(
                        "Unable to get the size of the compiled code. STATUS: {status}"
                    ),
                    backtrace: BackTrace::capture(),
                })?;
            }
        }
        let mut code = vec![0; code_size];
        // SAFETY: `code` is allocated with `code_size` bytes as reported by `hiprtcGetCodeSize`.
        // `program` is a valid compiled program handle.
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCode(program, code.as_mut_ptr());

            if status != hiprtcResult_HIPRTC_SUCCESS {
                Err(CompilationError::Generic {
                    reason: format!("Unable to get the compiled code. STATUS: {status}"),
                    backtrace: BackTrace::capture(),
                })?;
            }
        }

        let repr = jitc_kernel.repr.unwrap();

        if let Some(cache) = self.compilation_cache.as_mut() {
            let second_line_cache = self.second_line_compilation_cache.as_mut().unwrap();
            let key = key.unwrap();
            store_compiled(
                cache,
                key,
                CompilationCacheEntry {
                    entrypoint_name: jitc_kernel.entrypoint_name.clone(),
                    shared_mem_bytes: repr.shared_memory_size,
                    binary: code.clone(),
                },
            );
            store_compiled(second_line_cache, cpp_hash.unwrap(), key);
        }

        self.load_compiled_binary(
            code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            repr.shared_memory_size,
        )?;
        Ok(())
    }

    fn load_compiled_binary(
        &mut self,
        code: Vec<i8>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name.clone()).unwrap();

        // Create the HIP module
        let mut module: cubecl_hip_sys::hipModule_t = std::ptr::null_mut();
        // SAFETY: `code` contains a valid compiled binary obtained from `hiprtcGetCode`.
        // `module` receives the loaded module handle on success.
        unsafe {
            let codeptr = code.as_ptr();
            let status = cubecl_hip_sys::hipModuleLoadData(&mut module, codeptr as *const _);
            if status != hiprtcResult_HIPRTC_SUCCESS {
                return Err(CompilationError::Generic {
                    reason: format!("Unable to load the compiled module. STATUS: {status}"),
                    backtrace: BackTrace::capture(),
                });
            }
        }
        // Retrieve the HIP module function
        let mut func: cubecl_hip_sys::hipFunction_t = std::ptr::null_mut();
        // SAFETY: `module` is a valid loaded module from `hipModuleLoadData` above.
        // `func_name` is a null-terminated `CString` matching the kernel entry point.
        unsafe {
            let status =
                cubecl_hip_sys::hipModuleGetFunction(&mut func, module, func_name.as_ptr());
            if status != hiprtcResult_HIPRTC_SUCCESS {
                return Err(CompilationError::Generic {
                    reason: format!(
                        "Unable to load the function in the compiled module. STATUS: {status}"
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
        }

        // register module
        self.modules.insert(
            kernel_id.clone(),
            HipCompiledKernel {
                _module: module,
                func,
                cube_dim,
                shared_mem_bytes,
            },
        );

        Ok(())
    }

    /// Executes a task on the given stream.
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &[GpuResource],
    ) -> Result<(), LaunchError> {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.binding)
            .collect::<Vec<_>>();

        let kernel = self.modules.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;

        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid HIP stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            let status = cubecl_hip_sys::hipModuleLaunchKernel(
                kernel.func,
                dispatch_count.0,
                dispatch_count.1,
                dispatch_count.2,
                cube_dim.x,
                cube_dim.y,
                cube_dim.z,
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                bindings.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory {
                Err(LaunchError::OutOfMemory {
                    reason: format!("Out of memory when launching kernel: {kernel_id:?}"),
                    backtrace: BackTrace::capture(),
                })
            } else if status != HIP_SUCCESS {
                Err(LaunchError::Unknown {
                    reason: format!("Unable to launch kernel {kernel_id:?} with status {status:?}"),
                    backtrace: BackTrace::capture(),
                })
            } else {
                Ok(())
            }
        }
    }

    fn validate_shared(&self, repr: &Option<HipComputeKernel>) -> Result<(), LaunchError> {
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
