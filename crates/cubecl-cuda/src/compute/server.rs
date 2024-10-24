use cubecl_cpp::{formatter::format_cpp, CudaCompiler};

use super::storage::CudaStorage;
use super::{uninit_vec, CudaResource};
use cubecl_core::compute::DebugInformation;
use cubecl_core::ir::CubeDim;
use cubecl_core::Feature;
use cubecl_core::{prelude::*, KernelId};
use cubecl_runtime::debug::{DebugLogger, ProfileLevel};
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use cubecl_runtime::{ExecutionMode, TimestampsError, TimestampsResult};
use cudarc::driver::sys::CUctx_st;
use cudarc::driver::sys::CUfunc_st;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::future::Future;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    logger: DebugLogger,
}

#[derive(Debug)]
pub(crate) struct CudaContext {
    context: *mut CUctx_st,
    stream: cudarc::driver::sys::CUstream,
    memory_management: MemoryManagement<CudaStorage>,
    module_names: HashMap<KernelId, CompiledKernel>,
    timestamps: KernelTimestamps,
    pub(crate) arch: u32,
}

#[derive(Debug)]
enum KernelTimestamps {
    Inferred { start_time: Instant },
    Disabled,
}

impl KernelTimestamps {
    fn enable(&mut self) {
        if !matches!(self, Self::Disabled) {
            return;
        }

        *self = Self::Inferred {
            start_time: Instant::now(),
        };
    }

    fn disable(&mut self) {
        *self = Self::Disabled;
    }
}

#[derive(Debug)]
struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

unsafe impl Send for CudaServer {}

impl CudaServer {
    pub(crate) fn arch_version(&mut self) -> u32 {
        let ctx = self.get_context();
        ctx.arch
    }

    fn read_sync(&mut self, binding: server::Binding) -> Vec<u8> {
        let ctx = self.get_context();
        let resource = ctx.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        );

        let mut data = uninit_vec(resource.size() as usize);

        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(&mut data, resource.ptr, ctx.stream).unwrap();
        };

        ctx.sync();

        data
    }
}

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn CubeTask<CudaCompiler>>;
    type Storage = CudaStorage;
    type Feature = Feature;

    fn read(&mut self, binding: server::Binding) -> impl Future<Output = Vec<u8>> + 'static {
        let value = self.read_sync(binding);
        async { value }
    }

    fn create(&mut self, data: &[u8]) -> server::Handle {
        let handle = self.empty(data.len());
        let ctx = self.get_context();

        let binding = handle.clone().binding();
        let resource = ctx.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        );

        unsafe {
            cudarc::driver::result::memcpy_htod_async(resource.ptr, data, ctx.stream).unwrap();
        }

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size as u64, None);
        server::Handle::new(handle, None, None)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Vec<server::Binding>,
        mode: ExecutionMode,
    ) {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        let profile_level = self.logger.profile_level();
        let profile_info = if profile_level.is_some() {
            Some((kernel.name(), kernel_id.clone()))
        } else {
            None
        };

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = self.read_sync(binding);
                let data = bytemuck::cast_slice(&data);
                assert!(
                    data.len() == 3,
                    "Dynamic cube count should contain 3 values"
                );
                (data[0], data[1], data[2])
            }
        };

        let (ctx, logger) = self.get_context_with_logger();

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, logger, mode);
        }

        let resources = bindings
            .into_iter()
            .map(|binding| {
                ctx.memory_management.get_resource(
                    binding.memory,
                    binding.offset_start,
                    binding.offset_end,
                )
            })
            .collect::<Vec<_>>();

        if let Some(level) = profile_level {
            ctx.sync();
            let start = std::time::SystemTime::now();
            ctx.execute_task(kernel_id, count, resources);
            ctx.sync();

            let (name, kernel_id) = profile_info.unwrap();
            let info = match level {
                ProfileLevel::Basic | ProfileLevel::Medium => {
                    if let Some(val) = name.split("<").next() {
                        val.split("::").last().unwrap_or(name).to_string()
                    } else {
                        name.to_string()
                    }
                }
                ProfileLevel::Full => {
                    format!("{name}: {kernel_id} CubeCount {count:?}")
                }
            };

            self.logger
                .register_profiled(info, start.elapsed().unwrap());
        } else {
            ctx.execute_task(kernel_id, count, resources);
        }
    }

    fn flush(&mut self) {}

    fn sync(&mut self) -> impl Future<Output = ()> + 'static {
        self.logger.profile_summary();

        let ctx = self.get_context();
        ctx.sync();
        async move {}
    }

    fn sync_elapsed(&mut self) -> impl Future<Output = TimestampsResult> + 'static {
        self.logger.profile_summary();

        let ctx = self.get_context();
        ctx.sync();

        let duration = match &mut ctx.timestamps {
            KernelTimestamps::Inferred { start_time } => {
                let duration = start_time.elapsed();
                *start_time = Instant::now();
                Ok(duration)
            }
            KernelTimestamps::Disabled => Err(TimestampsError::Disabled),
        };

        async move { duration }
    }

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<Self> {
        let ctx = self.get_context();
        BindingResource::new(
            binding.clone(),
            ctx.memory_management.get_resource(
                binding.memory,
                binding.offset_start,
                binding.offset_end,
            ),
        )
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_usage()
    }

    fn enable_timestamps(&mut self) {
        self.ctx.timestamps.enable();
    }

    fn disable_timestamps(&mut self) {
        if self.logger.profile_level().is_none() {
            self.ctx.timestamps.disable();
        }
    }
}

impl CudaContext {
    pub fn new(
        memory_management: MemoryManagement<CudaStorage>,
        stream: cudarc::driver::sys::CUstream,
        context: *mut CUctx_st,
        arch: u32,
    ) -> Self {
        Self {
            context,
            memory_management,
            module_names: HashMap::new(),
            stream,
            arch,
            timestamps: KernelTimestamps::Disabled,
        }
    }

    fn sync(&mut self) {
        unsafe {
            cudarc::driver::result::stream::synchronize(self.stream).unwrap();
        };
    }

    fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        logger: &mut DebugLogger,
        mode: ExecutionMode,
    ) {
        let mut kernel_compiled = kernel.compile(mode);

        if logger.is_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let shared_mem_bytes = kernel_compiled.shared_mem_bytes;
        let cube_dim = kernel_compiled.cube_dim;
        let arch = format!("--gpu-architecture=sm_{}", self.arch);

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let options = &[arch.as_str(), include_option.as_str()];

        let kernel_compiled = logger.debug(kernel_compiled);

        let ptx = unsafe {
            let program = cudarc::nvrtc::result::create_program(kernel_compiled.source).unwrap();
            if cudarc::nvrtc::result::compile_program(program, options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).unwrap();
                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                let source = kernel.compile(mode).source;
                panic!("{message}\n[Source]  \n{source}");
            };
            cudarc::nvrtc::result::get_ptx(program).unwrap()
        };

        let func_name = CString::new("kernel".to_string()).unwrap();
        let func = unsafe {
            let module =
                cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _).unwrap();
            cudarc::driver::result::module::get_function(module, func_name).unwrap()
        };

        self.module_names.insert(
            kernel_id.clone(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );
    }

    fn execute_task(
        &mut self,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: Vec<CudaResource>,
    ) {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.as_binding())
            .collect::<Vec<_>>();

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        unsafe {
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                kernel.shared_mem_bytes as u32,
                self.stream,
                &mut bindings,
            )
            .unwrap();
        };
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(mut ctx: CudaContext) -> Self {
        let logger = DebugLogger::default();
        if logger.profile_level().is_some() {
            ctx.timestamps.enable();
        }
        Self { ctx, logger }
    }

    fn get_context(&mut self) -> &mut CudaContext {
        self.get_context_with_logger().0
    }

    fn get_context_with_logger(&mut self) -> (&mut CudaContext, &mut DebugLogger) {
        unsafe {
            cudarc::driver::result::ctx::set_current(self.ctx.context).unwrap();
        };
        (&mut self.ctx, &mut self.logger)
    }
}

fn include_path() -> PathBuf {
    let mut path = cuda_path().expect("
        CUDA installation not found.
        Please ensure that CUDA is installed and the CUDA_PATH environment variable is set correctly.
        Note: Default paths are used for Linux (/usr/local/cuda) and Windows (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/), which may not be correct.
    ");
    path.push("include");
    path
}

fn cuda_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("CUDA_PATH") {
        return Some(PathBuf::from(path));
    }

    #[cfg(target_os = "linux")]
    {
        return Some(PathBuf::from("/usr/local/cuda"));
    }

    #[cfg(target_os = "windows")]
    {
        return Some(PathBuf::from(
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
        ));
    }

    #[allow(unreachable_code)]
    None
}
