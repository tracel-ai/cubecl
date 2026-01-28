use cubecl_common::backtrace::BackTrace;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};
use cubecl_runtime::compiler::CompilationError;

use super::storage::gpu::GpuResource;
use crate::install::{cccl_include_path, include_path};
use crate::{CudaCompiler, compute::stream::Stream};
use cubecl_core::{ir::DeviceProperties, prelude::*, server::ResourceLimitError};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUtensorMap};
use cudarc::driver::sys::{CUfunc_st, CUresult};
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::str::FromStr;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};

use cubecl_common::cache::{Cache, CacheOption};

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    pub module_names: HashMap<KernelId, CompiledKernel>,
    ptx_cache: Option<Cache<String, PtxCacheEntry>>,
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
    cube_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    cluster_dim: Option<(u32, u32, u32)>,
    ptx: Vec<std::ffi::c_char>,
}

impl CudaContext {
    pub fn new(
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            context,
            module_names: HashMap::new(),
            ptx_cache: {
                let config = cubecl_runtime::config::GlobalConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(Cache::new(
                        "ptx",
                        CacheOption::default().name("cuda").root(root),
                    ))
                } else {
                    None
                }
            },
            arch,
            timestamps: TimestampProfiler::default(),
            compilation_options,
            properties,
        }
    }

    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), CompilationError> {
        let name = if let Some(cache) = &self.ptx_cache {
            let name = kernel_id.stable_format();

            if let Some(entry) = cache.get(&name) {
                log::trace!("Using PTX cache");

                return self.load_ptx(
                    entry.ptx.clone(),
                    kernel_id.clone(),
                    entry.entrypoint_name.clone(),
                    CubeDim {
                        x: entry.cube_dim.0,
                        y: entry.cube_dim.1,
                        z: entry.cube_dim.2,
                    },
                    entry.shared_mem_bytes,
                );
            }
            Some(name)
        } else {
            None
        };

        log::trace!("Compiling kernel");

        let mut kernel_compiled = kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            kernel.address_type(),
        )?;

        if logger.compilation_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let compute_kernel = kernel_compiled.repr.as_ref().unwrap();
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

        let cluster_dim = compute_kernel.cluster_dim;

        logger.log_compilation(&kernel_compiled);

        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::from_str(&kernel_compiled.source).unwrap();
            let program =
                cudarc::nvrtc::result::create_program(source.as_c_str(), None).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;
            if cudarc::nvrtc::result::compile_program(program, &options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
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
                let source = kernel
                    .compile(
                        &mut Default::default(),
                        &self.compilation_options,
                        mode,
                        kernel.address_type(),
                    )?
                    .source;
                return Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                });
            };
            cudarc::nvrtc::result::get_ptx(program).map_err(|err| CompilationError::Generic {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?
        };

        let repr = kernel_compiled.repr.unwrap();

        if let Some(cache) = &mut self.ptx_cache {
            let result = cache.insert(
                name.unwrap(),
                PtxCacheEntry {
                    entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                    cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
                    shared_mem_bytes: repr.shared_memory_size(),
                    cluster_dim: cluster_dim.map(|cluster| (cluster.x, cluster.y, cluster.z)),
                    ptx: ptx.clone(),
                },
            );
            if let Err(err) = result {
                log::warn!("Unable to save the ptx {err:?}");
            }
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            repr.shared_memory_size(),
        )
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
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the PTX: {err:?}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        self.module_names.insert(
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
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        scalars: &[*mut c_void],
    ) -> Result<(), LaunchError> {
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.binding));
        bindings.extend(scalars);

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| match err.0 {
                CUresult::CUDA_ERROR_INVALID_VALUE => ResourceLimitError::SharedMemory {
                    requested: kernel.shared_mem_bytes,
                    max: self.properties.hardware.max_shared_memory_size,
                    backtrace: BackTrace::capture(),
                }
                .into(),
                _ => LaunchError::Unknown {
                    reason: format!("{err:?}"),
                    backtrace: BackTrace::capture(),
                },
            })?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                &mut bindings,
            )
            .map_err(|err| match err.0 {
                CUresult::CUDA_ERROR_INVALID_VALUE => {
                    let props = &self.properties.hardware;
                    let num_elems = cube_dim.num_elems();
                    if num_elems > props.max_units_per_cube {
                        LaunchError::TooManyResources(ResourceLimitError::Units {
                            requested: cube_dim.num_elems(),
                            max: self.properties.hardware.max_units_per_cube,
                            backtrace: BackTrace::capture(),
                        })
                    } else if cube_dim.x > props.max_cube_dim.0
                        || cube_dim.y > props.max_cube_dim.1
                        || cube_dim.z > props.max_cube_dim.2
                    {
                        LaunchError::TooManyResources(ResourceLimitError::CubeDim {
                            requested: (cube_dim.x, cube_dim.y, cube_dim.z),
                            max: props.max_cube_dim,
                            backtrace: BackTrace::capture(),
                        })
                    } else {
                        LaunchError::Unknown {
                            reason: format!("{err:?}"),
                            backtrace: BackTrace::capture(),
                        }
                    }
                }
                _ => LaunchError::Unknown {
                    reason: format!("{err:?}"),
                    backtrace: BackTrace::capture(),
                },
            })?;
        };

        Ok(())
    }
}
