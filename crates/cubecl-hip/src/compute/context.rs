use super::storage::gpu::GpuResource;
use crate::runtime::HipCompiler;
use crate::{compute::stream::Stream, runtime::HipComputeKernel};
use cubecl_common::backtrace::BackTrace;
use cubecl_common::cache::Cache;
use cubecl_common::cache::CacheOption;
use cubecl_core::{ir::DeviceProperties, prelude::*, server::ResourceLimitError};
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_hip_sys::{HIP_SUCCESS, get_hip_include_path, hiprtcResult_HIPRTC_SUCCESS};
use cubecl_runtime::compiler::CompilationError;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct HipContext {
    pub module_names: HashMap<KernelId, HipCompiledKernel>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<Cache<String, CompilationCacheEntry>>,
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
    cube_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    binary: Vec<i8>,
}

impl HipContext {
    pub fn new(compilation_options: CompilationOptions, properties: DeviceProperties) -> Self {
        Self {
            module_names: HashMap::new(),
            timestamps: TimestampProfiler::default(),
            compilation_options,
            compilation_cache: {
                let config = cubecl_runtime::config::GlobalConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(Cache::new(
                        "hip-kernel",
                        CacheOption::default().name("hip").root(root),
                    ))
                } else {
                    None
                }
            },
            properties,
        }
    }

    /// Compiles a kernel.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        cube_kernel: Box<dyn CubeTask<HipCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let name = if let Some(cache) = self.compilation_cache.as_ref() {
            let name = kernel_id.stable_format();
            if let Some(entry) = cache.get(&name) {
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
                    entry.shared_mem_bytes,
                )?;
                return Ok(());
            }
            Some(name)
        } else {
            None
        };

        self.validate_cube_dim(kernel_id)?;
        self.validate_units(kernel_id)?;

        // CubeCL compilation
        // jitc = just-in-time compiled
        let mut jitc_kernel = cube_kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            cube_kernel.address_type(),
        )?;

        self.validate_shared(&jitc_kernel.repr)?;

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
            cache
                .insert(
                    name.unwrap(),
                    CompilationCacheEntry {
                        entrypoint_name: jitc_kernel.entrypoint_name.clone(),
                        cube_dim: (
                            jitc_kernel.cube_dim.x,
                            jitc_kernel.cube_dim.y,
                            jitc_kernel.cube_dim.z,
                        ),
                        shared_mem_bytes: repr.shared_memory_size(),
                        binary: code.clone(),
                    },
                )
                .unwrap();
        }

        self.load_compiled_binary(
            code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.cube_dim,
            repr.shared_memory_size(),
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
        self.module_names.insert(
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

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;

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

    fn validate_cube_dim(&self, kernel_id: &KernelId) -> Result<(), LaunchError> {
        let requested = kernel_id.cube_dim;
        let max: CubeDim = self.properties.hardware.max_cube_dim.into();
        if !max.can_contain(requested) {
            Err(ResourceLimitError::CubeDim {
                requested: requested.into(),
                max: max.into(),
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }

    fn validate_units(&self, kernel_id: &KernelId) -> Result<(), LaunchError> {
        let requested = kernel_id.cube_dim.num_elems();
        let max = self.properties.hardware.max_units_per_cube;
        if requested > max {
            Err(ResourceLimitError::Units {
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
