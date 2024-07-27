use crate::compiler::format_cpp_code;

use super::storage::CudaStorage;
use super::CudaResource;
use cubecl_common::reader::{reader_from_concrete, Reader};
use cubecl_common::sync_type::SyncType;
use cubecl_core::compute::DebugInformation;
use cubecl_core::ir::CubeDim;
use cubecl_core::FeatureSet;
use cubecl_core::{prelude::*, KernelId};
use cubecl_runtime::debug::DebugLogger;
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use cudarc::driver::sys::CUctx_st;
use cudarc::driver::sys::CUfunc_st;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::path::PathBuf;

#[derive(Debug)]
pub struct CudaServer<MM: MemoryManagement<CudaStorage>> {
    state: CudaServerState<MM>,
    logger: DebugLogger,
    pub(crate) archs: Vec<i32>,
    pub(crate) minimum_arch_version: i32,
}

pub(crate) enum CudaServerState<MM: MemoryManagement<CudaStorage>> {
    Uninitialized {
        device_index: usize,
        init: Box<dyn Fn(usize) -> CudaContext<MM>>,
    },
    Initialized {
        ctx: CudaContext<MM>,
    },
}

impl<MM: MemoryManagement<CudaStorage>> core::fmt::Debug for CudaServerState<MM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Context")
    }
}

#[derive(Debug)]
pub(crate) struct CudaContext<MM: MemoryManagement<CudaStorage>> {
    context: *mut CUctx_st,
    stream: cudarc::driver::sys::CUstream,
    memory_management: MM,
    module_names: HashMap<KernelId, CompiledKernel>,
}

#[derive(Debug)]
struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

unsafe impl<MM: MemoryManagement<CudaStorage>> Send for CudaServer<MM> {}

impl<MM: MemoryManagement<CudaStorage>> CudaServer<MM> {
    fn read_sync(&mut self, binding: server::Binding<Self>) -> Vec<u8> {
        let ctx = self.get_context();
        let resource = ctx.memory_management.get(binding.memory);

        // TODO: Check if it is possible to make this faster
        let mut data = vec![0; resource.size() as usize];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(&mut data, resource.ptr, ctx.stream).unwrap();
        };
        ctx.sync();
        data
    }
}

impl<MM: MemoryManagement<CudaStorage>> ComputeServer for CudaServer<MM> {
    type Kernel = Box<dyn CubeTask>;
    type DispatchOptions = CubeCount<Self>;
    type Storage = CudaStorage;
    type MemoryManagement = MM;
    type FeatureSet = FeatureSet;

    fn read(&mut self, binding: server::Binding<Self>) -> Reader {
        reader_from_concrete(self.read_sync(binding))
    }

    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(data.len(), || unsafe {
            cudarc::driver::result::stream::synchronize(ctx.stream).unwrap();
        });
        let handle = server::Handle::new(handle);
        let binding = handle.clone().binding().memory;
        let resource = ctx.memory_management.get(binding);

        unsafe {
            cudarc::driver::result::memcpy_htod_async(resource.ptr, data, ctx.stream).unwrap();
        }

        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size, || unsafe {
            cudarc::driver::result::stream::synchronize(ctx.stream).unwrap();
        });
        server::Handle::new(handle)
    }

    fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: Self::DispatchOptions,
        bindings: Vec<server::Binding<Self>>,
    ) {
        let arch = self.minimum_arch_version;

        let kernel_id = kernel.id();

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
            ctx.compile_kernel(&kernel_id, kernel, arch, logger);
        }

        let resources = bindings
            .into_iter()
            .map(|binding| ctx.memory_management.get(binding.memory))
            .collect::<Vec<_>>();

        ctx.execute_task(kernel_id, count, resources);
    }

    fn sync(&mut self, sync_type: SyncType) {
        match sync_type {
            // Synchronize the stream if waiting.
            SyncType::Wait => {
                let ctx = self.get_context();
                ctx.sync();
            }
            // Nothing to do - all tasks are already submitted to the stream.
            SyncType::Flush => (),
        }
    }

    fn get_resource(
        &mut self,
        binding: server::Binding<Self>,
    ) -> <Self::Storage as cubecl_runtime::storage::ComputeStorage>::Resource {
        let ctx = self.get_context();
        ctx.memory_management.get(binding.memory)
    }
}

impl<MM: MemoryManagement<CudaStorage>> CudaContext<MM> {
    pub fn new(
        memory_management: MM,
        stream: cudarc::driver::sys::CUstream,
        context: *mut CUctx_st,
    ) -> Self {
        Self {
            context,
            memory_management,
            module_names: HashMap::new(),
            stream,
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
        kernel: Box<dyn CubeTask>,
        arch: i32,
        logger: &mut DebugLogger,
    ) {
        let mut kernel_compiled = kernel.compile();

        if logger.is_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp_code(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let shared_mem_bytes = kernel_compiled.shared_mem_bytes;
        let cube_dim = kernel_compiled.cube_dim;
        let arch = format!("--gpu-architecture=sm_{}", arch);

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
                let source = kernel.compile().source;
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

        self.memory_management.storage().flush(resources)
    }
}

impl<MM: MemoryManagement<CudaStorage>> CudaServer<MM> {
    /// Create a new cuda server.
    pub(crate) fn new(index: usize, init: Box<dyn Fn(usize) -> CudaContext<MM>>) -> Self {
        let archs = unsafe {
            let mut num_supported_arg: core::ffi::c_int = 0;
            cudarc::nvrtc::sys::lib()
                .nvrtcGetNumSupportedArchs(core::ptr::from_mut(&mut num_supported_arg));

            let mut archs: Vec<core::ffi::c_int> = vec![0; num_supported_arg as usize];
            cudarc::nvrtc::sys::lib().nvrtcGetSupportedArchs(core::ptr::from_mut(&mut archs[0]));
            archs
        };

        let minimum_arch_version = archs[0];

        Self {
            state: CudaServerState::Uninitialized {
                device_index: index,
                init,
            },
            logger: DebugLogger::new(),
            archs,
            minimum_arch_version,
        }
    }

    fn get_context(&mut self) -> &mut CudaContext<MM> {
        self.get_context_with_logger().0
    }

    fn get_context_with_logger(&mut self) -> (&mut CudaContext<MM>, &mut DebugLogger) {
        if let CudaServerState::Uninitialized { device_index, init } = &self.state {
            let ctx = init(*device_index);
            self.state = CudaServerState::Initialized { ctx };
        }
        if let CudaServerState::Initialized { ctx } = &mut self.state {
            unsafe {
                cudarc::driver::result::ctx::set_current(ctx.context).unwrap();
            };
            (ctx, &mut self.logger)
        } else {
            panic!("Context should be initialized");
        }
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
