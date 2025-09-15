use super::storage::gpu::GpuResource;
use super::storage::gpu::GpuStorage;
use crate::compute::LaunchError;
use crate::compute::cpu::PinnedMemoryStorage;
use crate::compute::stream::Stream;
use crate::runtime::HipCompiler;
use cubecl_core::compute::CubeTask;
use cubecl_core::compute::DebugInformation;
use cubecl_core::prelude::*;
use cubecl_core::server::Binding;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_hip_sys::hipMemcpyKind_hipMemcpyHostToDevice;
use cubecl_hip_sys::{HIP_SUCCESS, get_hip_include_path, hiprtcResult_HIPRTC_SUCCESS};
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::MemoryManagement;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct HipContext {
    pub memory_management_gpu: MemoryManagement<GpuStorage>,
    pub memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
    pub module_names: HashMap<KernelId, HipCompiledKernel>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    #[cfg(feature = "compilation-cache")]
    pub compilation_cache: Cache<String, CompilationCacheEntry>,
}

#[derive(Debug)]
pub struct HipCompiledKernel {
    _module: cubecl_hip_sys::hipModule_t,
    func: cubecl_hip_sys::hipFunction_t,
    cube_dim: CubeDim,
}

#[cfg(feature = "compilation-cache")]
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    binary: Vec<i8>,
}

impl HipContext {
    pub fn new(
        memory_management_gpu: MemoryManagement<GpuStorage>,
        memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
        compilation_options: CompilationOptions,
    ) -> Self {
        Self {
            memory_management_gpu,
            memory_management_cpu,
            module_names: HashMap::new(),
            timestamps: TimestampProfiler::default(),
            compilation_options,
            #[cfg(feature = "compilation-cache")]
            compilation_cache: Cache::new("hip/compilation", CacheOption::default()),
        }
    }

    /// Compiles a kernel.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        cube_kernel: Box<dyn CubeTask<HipCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) {
        #[cfg(feature = "compilation-cache")]
        let name = kernel_id.stable_format();
        #[cfg(feature = "compilation-cache")]
        if let Some(entry) = self.compilation_cache.get(&name) {
            log::trace!("Using compilation cache");
            self.load_compiled_binary(
                entry.binary.clone(),
                kernel_id.clone(),
                entry.entrypoint_name.clone(),
                CubeDim {
                    x: entry.cube_dim.0,
                    y: entry.cube_dim.1,
                    z: entry.cube_dim.2,
                },
            );
            return;
        }

        // CubeCL compilation
        // jitc = just-in-time compiled
        let mut jitc_kernel =
            cube_kernel.compile(&mut Default::default(), &self.compilation_options, mode);

        if logger.compilation_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }
        logger.log_compilation(&jitc_kernel);

        // Create HIP Program
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
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should create the program"
            );
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
        unsafe {
            let options_ptr = options.as_mut_ptr();
            let status =
                cubecl_hip_sys::hiprtcCompileProgram(program, options.len() as i32, options_ptr);
            if status != hiprtcResult_HIPRTC_SUCCESS {
                let mut log_size: usize = 0;
                let status =
                    cubecl_hip_sys::hiprtcGetProgramLogSize(program, &mut log_size as *mut usize);
                assert_eq!(
                    status, hiprtcResult_HIPRTC_SUCCESS,
                    "Should retrieve the compilation log size"
                );
                let mut log_buffer = vec![0; log_size];
                let status = cubecl_hip_sys::hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());
                assert_eq!(
                    status, hiprtcResult_HIPRTC_SUCCESS,
                    "Should retrieve the compilation log contents"
                );
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
                panic!("{message}\n[Source]  \n{}", jitc_kernel.source);
            }
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should compile the program"
            );
        };
        // Get HIP compiled code from program
        let mut code_size: usize = 0;
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCodeSize(program, &mut code_size);
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should get size of compiled code"
            );
        }
        let mut code = vec![0; code_size];
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCode(program, code.as_mut_ptr());
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should load compiled code"
            );
        }

        #[cfg(feature = "compilation-cache")]
        self.compilation_cache
            .insert(
                name,
                CompilationCacheEntry {
                    entrypoint_name: jitc_kernel.entrypoint_name.clone(),
                    cube_dim: (
                        jitc_kernel.cube_dim.x,
                        jitc_kernel.cube_dim.y,
                        jitc_kernel.cube_dim.z,
                    ),
                    binary: code.clone(),
                },
            )
            .unwrap();

        self.load_compiled_binary(
            code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
        );
    }

    fn load_compiled_binary(
        &mut self,
        code: Vec<i8>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
    ) {
        let func_name = CString::new(entrypoint_name.clone()).unwrap();

        // Create the HIP module
        let mut module: cubecl_hip_sys::hipModule_t = std::ptr::null_mut();
        unsafe {
            let codeptr = code.as_ptr();
            let status = cubecl_hip_sys::hipModuleLoadData(&mut module, codeptr as *const _);
            assert_eq!(status, HIP_SUCCESS, "Should load compiled code into module");
        }
        // Retrieve the HIP module function
        let mut func: cubecl_hip_sys::hipFunction_t = std::ptr::null_mut();
        unsafe {
            let status =
                cubecl_hip_sys::hipModuleGetFunction(&mut func, module, func_name.as_ptr());
            assert_eq!(status, HIP_SUCCESS, "Should return module function");
        }

        // register module
        self.module_names.insert(
            kernel_id.clone(),
            HipCompiledKernel {
                _module: module,
                func,
                cube_dim,
            },
        );
    }

    /// Executes a task on the given stream.
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: Vec<GpuResource>,
    ) {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.binding)
            .collect::<Vec<_>>();

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;

        let result = unsafe {
            let status = cubecl_hip_sys::hipModuleLaunchKernel(
                kernel.func,
                dispatch_count.0,
                dispatch_count.1,
                dispatch_count.2,
                cube_dim.x,
                cube_dim.y,
                cube_dim.z,
                // Shared memory is specified statically in the kernel, and no dynamic shared
                // memory is supported yet in the kernel, which would be that value for the
                // current kernel launch.
                0,
                stream.sys,
                bindings.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory {
                Err(LaunchError::OutOfMemory)
            } else if status != HIP_SUCCESS {
                Err(LaunchError::Unknown(format!(
                    "Unable to launch kernel {kernel_id:?} with status {status:?}"
                )))
            } else {
                Ok(())
            }
        };

        match result {
            Ok(_) => {}
            Err(err) => match self.timestamps.is_empty() {
                true => panic!("{err:?}"),
                false => self.timestamps.error(err.into()),
            },
        }
    }

    pub fn copy_to_binding(&mut self, stream: &mut Stream, binding: Binding, data: &[u8]) {
        let resource = self
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .unwrap();

        unsafe {
            let status = cubecl_hip_sys::hipMemcpyHtoDAsync(
                resource.ptr,
                data as *const _ as *mut _,
                data.len(),
                stream.sys,
            );
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
    }

    pub fn copy_to_binding_2d(
        &mut self,
        stream: &mut Stream,
        binding: Binding,
        data: &[u8],
        shape: &[usize],
        stride: usize,
        elem_size: usize,
    ) {
        let resource = self
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .unwrap();

        let width = *shape.last().unwrap_or(&1);
        let height: usize = shape.iter().rev().skip(1).product();
        let width_bytes = width * elem_size;
        let stride_bytes = stride * elem_size;

        unsafe {
            let status = cubecl_hip_sys::hipMemcpy2DAsync(
                resource.ptr,
                stride_bytes,
                data as *const _ as *mut _,
                width_bytes,
                width_bytes,
                height.max(1),
                hipMemcpyKind_hipMemcpyHostToDevice,
                stream.sys,
            );
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
    }
}
