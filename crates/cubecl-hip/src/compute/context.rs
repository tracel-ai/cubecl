//! The device context: its compiled kernels, and its open profiles.
//!
//! Compilation is memoized here rather than per stream, because a module is
//! loaded into the context and every stream sharing it can launch from the
//! same one. What a compiled kernel answers for beyond its entry point is
//! which of its bindings it writes, which is what a launch stages its write
//! scope from.

use super::storage::gpu::GpuResource;
use crate::compiler::{HipBackend, HipCompilationOptions, HipCompiler, HipRepresentation};
use crate::compute::stream::Stream;
use cubecl_core::hash::StableHasher;
use cubecl_core::{hash::StableHash, ir::DeviceProperties, prelude::*, server::ResourceLimitError};
use cubecl_cpp::formatter::format_cpp;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::persistence::Store;
use cubecl_hip_sys::get_hip_include_path;
use cubecl_runtime::compiler::{
    CompilationCache, build_id_hash, compilation_store, store_compiled,
};
use cubecl_runtime::driver::checked;
use cubecl_runtime::kernel::BufferIOAttr;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};
use cubecl_runtime::{
    compiler::{CubeTask, KernelCacheKey},
    kernel::CompiledKernel,
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
    pub compilation_options: HipCompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<Store<KernelCacheKey, CompilationCacheEntry>>,
    /// Cache mapping C++ code hashes to the key that first compiled them. We can skip the slow HIP
    /// compiler if we already have a compiled artifact for the same code.
    ///
    /// C++ only: the LLVM backend emits a code object directly, so there is no
    /// intermediate source to key a second line on.
    pub second_line_compilation_cache: Option<Store<StableHash, KernelCacheKey>>,
    build_id: StableHash,
}

#[derive(Debug)]
pub struct HipCompiledKernel {
    _module: cubecl_hip_sys::hipModule_t,
    func: cubecl_hip_sys::hipFunction_t,
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the compiler's answer, carried here because on a cache hit nothing
    /// else of the compilation survives. `None` for entries persisted before
    /// the answer existed, which the launch path reads as every buffer both
    /// read and written.
    io: Option<Arc<[BufferIOAttr]>>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    binary: Vec<i8>,
    /// See [`HipCompiledKernel::io`]; defaulted for entries persisted before
    /// the field existed.
    #[serde(default)]
    io: Option<Vec<BufferIOAttr>>,
}

/// The namespace a backend's compiled artifacts live under.
///
/// Both backends emit AMD code objects, so without the backend in the key a stale
/// artifact from one would load and run happily under the other — tests passing
/// while measuring nothing.
fn cache_namespace(fingerprint: &str, backend: HipBackend) -> String {
    let backend = match backend {
        HipBackend::Cpp => "cpp",
        HipBackend::Llvm => "llvm",
    };
    format!("{fingerprint}-{backend}")
}

impl HipContext {
    /// `fingerprint` is the one the runtime already published on
    /// [`DeviceProperties::identity`], rather than one rebuilt here: the
    /// namespace a kernel is cached under and the identity a bundle is stamped
    /// with have to be the same string, and the only way to guarantee that is
    /// for there to be one string.
    ///
    /// `backend` is which one compiles here; see [`cache_namespace`].
    pub fn new(
        compilation_options: HipCompilationOptions,
        properties: DeviceProperties,
        fingerprint: String,
        backend: HipBackend,
    ) -> Self {
        let fingerprint = cache_namespace(&fingerprint, backend);
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
                    &entry.binary,
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
        let jitc_kernel = cube_kernel.compile(
            definition,
            &mut Default::default(),
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
        jitc_kernel: CompiledKernel<HipCompiler>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        match &jitc_kernel.repr {
            Some(HipRepresentation::Cpp(_)) => {
                self.load_transpiled(kernel_id, key, jitc_kernel, logger)
            }
            Some(HipRepresentation::Llvm(_)) => {
                self.load_code_object(kernel_id, key, jitc_kernel, logger)
            }
            None => Err(CompilationError::Generic {
                reason: "the compiler returned no kernel to load".to_string(),
                backtrace: BackTrace::capture(),
            }
            .into()),
        }
    }

    /// Turns a kernel the LLVM backend just compiled into a loaded module.
    ///
    /// What it hands back is already a linked `ET_DYN` code object, so no `hiprtc*`
    /// call belongs here: the bytes go straight to [`Self::load_compiled_binary`],
    /// exactly as a cache hit would.
    fn load_code_object(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<HipCompiler>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let Some(HipRepresentation::Llvm(module)) = &jitc_kernel.repr else {
            unreachable!("dispatched on the representation");
        };

        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("ll", kernel_id.clone()));
        }
        logger.log_compilation(&jitc_kernel);

        let code = to_signed(&module.code_object);
        let shared_mem_bytes = module.shared_memory_size;
        let io = jitc_kernel.io.take();
        let entrypoint_name = jitc_kernel.entrypoint_name.clone();

        self.load_compiled_binary(
            &code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            shared_mem_bytes,
            io.clone().map(Arc::from),
        )?;

        // Cached after the load, so a code object the driver rejects is not handed back on
        // the next run, and the bytes move rather than being copied a third time.
        // `try_load_cached` hands back a key exactly when there is a cache to put it in.
        if let Some((cache, key)) = self.compilation_cache.as_mut().zip(key) {
            store_compiled(
                cache,
                key,
                CompilationCacheEntry {
                    entrypoint_name,
                    shared_mem_bytes,
                    binary: code,
                    io,
                },
            );
        }
        Ok(())
    }

    /// Turns a kernel the C++ backend just transpiled into a loaded module, by
    /// running its source through HIP RTC first.
    fn load_transpiled(
        &mut self,
        kernel_id: &KernelId,
        key: Option<KernelCacheKey>,
        mut jitc_kernel: CompiledKernel<HipCompiler>,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        if logger.compilation_source_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }
        logger.log_compilation(&jitc_kernel);

        // `try_load_cached` hands back a key exactly when there is a cache to put it in, and
        // both stores are opened together in `HipContext::new`.
        let cpp_hash = if let Some(key) = key
            && let Some(cache) = self.compilation_cache.as_mut()
        {
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

        let code = compile_to_binary(&jitc_kernel.source)?;

        let io = jitc_kernel.io.take();
        let repr = jitc_kernel.repr.unwrap();
        let shared_mem_bytes = repr.shared_memory_size();
        let entrypoint_name = jitc_kernel.entrypoint_name.clone();

        self.load_compiled_binary(
            &code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            shared_mem_bytes,
            io.clone().map(Arc::from),
        )?;

        // Cached after the load, so a binary the driver rejects is not handed back on the
        // next run, and the bytes move rather than being copied.
        if let Some((cache, key)) = self.compilation_cache.as_mut().zip(key) {
            let second_line_cache = self.second_line_compilation_cache.as_mut().unwrap();
            store_compiled(
                cache,
                key,
                CompilationCacheEntry {
                    entrypoint_name,
                    shared_mem_bytes,
                    binary: code,
                    io,
                },
            );
            store_compiled(second_line_cache, cpp_hash.unwrap(), key);
        }
        Ok(())
    }

    fn load_compiled_binary(
        &mut self,
        code: &[i8],
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
        io: Option<Arc<[BufferIOAttr]>>,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name.clone()).unwrap();

        // Create the HIP module
        let mut module: cubecl_hip_sys::hipModule_t = std::ptr::null_mut();
        // SAFETY: `code` contains a valid compiled binary obtained from `hiprtcGetCode`.
        // `module` receives the loaded module handle on success.
        unsafe {
            let codeptr = code.as_ptr();
            let status = cubecl_hip_sys::hipModuleLoadData(&mut module, codeptr as *const _);
            checked("hipModuleLoadData", status)?;
        }
        // Retrieve the HIP module function
        let mut func: cubecl_hip_sys::hipFunction_t = std::ptr::null_mut();
        // SAFETY: `module` is a valid loaded module from `hipModuleLoadData` above.
        // `func_name` is a null-terminated `CString` matching the kernel entry point.
        unsafe {
            let status =
                cubecl_hip_sys::hipModuleGetFunction(&mut func, module, func_name.as_ptr());
            checked("hipModuleGetFunction", status)?;
        }

        // register module
        self.modules.insert(
            kernel_id.clone(),
            HipCompiledKernel {
                _module: module,
                func,
                cube_dim,
                shared_mem_bytes,
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

            // Out of memory is told apart from the rest because the caller
            // can act on it — reclaim and relaunch — where nothing else here
            // is worth retrying.
            match checked("hipModuleLaunchKernel", status) {
                Ok(()) => Ok(()),
                Err(_) if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory => {
                    Err(LaunchError::OutOfMemory {
                        reason: format!("out of memory launching kernel {kernel_id:?}"),
                        backtrace: BackTrace::capture(),
                    })
                }
                Err(err) => Err(LaunchError::Unknown {
                    reason: format!("{err}, launching kernel {kernel_id:?}"),
                    backtrace: BackTrace::capture(),
                }),
            }
        }
    }

    fn validate_shared(&self, repr: &Option<HipRepresentation>) -> Result<(), LaunchError> {
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

/// A HIP RTC program, destroyed on drop.
///
/// The handle owns the source, the compilation log and the compiled code
/// inside the RTC runtime, none of which the caller needs once the binary has
/// been copied out. A guard rather than a call at the end because every step
/// of the compilation below returns early on failure, and each one of those
/// paths used to leak the program.
struct RtcProgram(cubecl_hip_sys::hiprtcProgram);

impl Drop for RtcProgram {
    fn drop(&mut self) {
        // SAFETY: created by `hiprtcCreateProgram` below and destroyed exactly
        // once here, after the compiled code has been copied out.
        unsafe {
            cubecl_hip_sys::hiprtcDestroyProgram(&mut self.0 as *mut _);
        }
    }
}

/// Compile `source` to a device binary with HIP RTC.
///
/// # Errors
///
/// [`CompilationError::Generic`] carrying the compiler's own log, and the
/// source that produced it, so a kernel the driver refuses says why.
fn compile_to_binary(source: &str) -> Result<Vec<i8>, CompilationError> {
    let source = CString::new(source).map_err(|err| CompilationError::Generic {
        reason: format!("The generated source is not a valid C string: {err}"),
        backtrace: BackTrace::capture(),
    })?;

    // SAFETY: `source` is null-terminated and outlives the call. The returned
    // handle is valid on success and owned by the guard from here on.
    let program = unsafe {
        let mut program: cubecl_hip_sys::hiprtcProgram = std::ptr::null_mut();
        let status = cubecl_hip_sys::hiprtcCreateProgram(
            &mut program,
            source.as_ptr(),
            std::ptr::null(), // program name seems unnecessary
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        checked("hiprtcCreateProgram", status)?;
        RtcProgram(program)
    };

    let include_path = get_hip_include_path().map_err(|err| CompilationError::Generic {
        reason: format!("Unable to locate the HIP headers to compile against: {err}"),
        backtrace: BackTrace::capture(),
    })?;
    let include_option =
        CString::new(format!("-I{include_path}")).map_err(|err| CompilationError::Generic {
            reason: format!("The HIP include path is not a valid C string: {err}"),
            backtrace: BackTrace::capture(),
        })?;
    // needed for rocWMMA extension to compile
    let cpp_std_option = c"--std=c++17";
    let optimization_level = c"-O3";
    let mut options = [
        cpp_std_option.as_ptr(),
        include_option.as_ptr(),
        optimization_level.as_ptr(),
    ];

    // SAFETY: `program.0` is the handle created above, and `options` holds
    // null-terminated pointers that outlive the call.
    let status = unsafe {
        cubecl_hip_sys::hiprtcCompileProgram(program.0, options.len() as i32, options.as_mut_ptr())
    };
    if checked("hiprtcCompileProgram", status).is_err() {
        return Err(CompilationError::Generic {
            reason: format!(
                "{}\n[Source]  \n{}",
                compilation_log(&program),
                source.to_string_lossy()
            ),
            backtrace: BackTrace::capture(),
        });
    }

    // SAFETY: `program.0` compiled successfully above, so it has code to
    // report the size of and to copy out into a buffer of exactly that size.
    unsafe {
        let mut code_size: usize = 0;
        let status = cubecl_hip_sys::hiprtcGetCodeSize(program.0, &mut code_size);
        checked("hiprtcGetCodeSize", status)?;
        let mut code = vec![0; code_size];
        let status = cubecl_hip_sys::hiprtcGetCode(program.0, code.as_mut_ptr());
        checked("hiprtcGetCode", status)?;
        Ok(code)
    }
}

/// The compiler's log for a program it refused, indented under a heading.
///
/// Reports why the log itself is missing rather than failing on it: this runs
/// on a path that already has an error to report, and losing that error to a
/// second one would leave the caller with nothing.
fn compilation_log(program: &RtcProgram) -> String {
    let mut message = "[Compilation Error] ".to_string();
    // SAFETY: `program.0` is a valid handle; the log buffer is sized by the
    // call that reports its length, and read back as a C string.
    let log = unsafe {
        let mut log_size: usize = 0;
        let status =
            cubecl_hip_sys::hiprtcGetProgramLogSize(program.0, &mut log_size as *mut usize);
        if let Err(err) = checked("hiprtcGetProgramLogSize", status) {
            return message + &format!("\n the log's length is unavailable: {err}");
        }
        if log_size == 0 {
            return message + "\n No compilation logs found!";
        }
        let mut log_buffer = vec![0; log_size];
        let status = cubecl_hip_sys::hiprtcGetProgramLog(program.0, log_buffer.as_mut_ptr());
        if let Err(err) = checked("hiprtcGetProgramLog", status) {
            return message + &format!("\n the log itself is unavailable: {err}");
        }
        CStr::from_ptr(log_buffer.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    for line in log.split('\n').filter(|line| !line.is_empty()) {
        message += format!("\n    {line}").as_str();
    }
    message
}

/// A code object as the `c_char` slice the cache and `hipModuleLoadData` are written in.
///
/// `i8` and `u8` have the same layout, so this is the copy out of the compiler's buffer and
/// nothing more.
fn to_signed(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|byte| *byte as i8).collect()
}

#[cfg(test)]
mod tests {
    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    fn cache_namespace_separates_backends() {
        assert_ne!(
            super::cache_namespace("gfx1201-abc", crate::compiler::HipBackend::Cpp),
            super::cache_namespace("gfx1201-abc", crate::compiler::HipBackend::Llvm),
        );
    }
}
