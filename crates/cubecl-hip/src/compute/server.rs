use crate::compiler::format_cpp_code;

use super::storage::HipStorage;
use super::HipResource;
use cubecl_common::reader::{reader_from_concrete, Reader};
use cubecl_common::sync_type::SyncType;
use cubecl_core::compute::DebugInformation;
use cubecl_core::ir::CubeDim;
use cubecl_core::{prelude::*, KernelId};
use cubecl_core::{FeatureSet, Properties};
use cubecl_hip_sys::{hiprtcResult_HIPRTC_SUCCESS, HIP_SUCCESS};
use cubecl_runtime::debug::DebugLogger;
use cubecl_runtime::ExecutionMode;
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;

#[derive(Debug)]
pub struct HipServer<MM: MemoryManagement<HipStorage>> {
    state: HipServerState<MM>,
    logger: DebugLogger,
}

pub(crate) enum HipServerState<MM: MemoryManagement<HipStorage>> {
    Uninitialized {
        device_index: usize,
        init: Box<dyn Fn(usize) -> HipContext<MM>>,
    },
    Initialized {
        ctx: HipContext<MM>,
    },
}

impl<MM: MemoryManagement<HipStorage>> core::fmt::Debug for HipServerState<MM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Context")
    }
}

#[derive(Debug)]
pub(crate) struct HipContext<MM: MemoryManagement<HipStorage>> {
    context: *mut cubecl_hip_sys::hipCtx_t,
    stream: cubecl_hip_sys::hipStream_t,
    memory_management: MM,
    module_names: HashMap<KernelId, HipCompiledKernel>,
}

#[derive(Debug)]
struct HipCompiledKernel {
    _module: cubecl_hip_sys::hipModule_t,
    func: cubecl_hip_sys::hipFunction_t,
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
}

unsafe impl<MM: MemoryManagement<HipStorage>> Send for HipServer<MM> {}

impl<MM: MemoryManagement<HipStorage>> HipServer<MM> {
    fn read_sync(&mut self, binding: server::Binding<Self>) -> Vec<u8> {
        let ctx = self.get_context();
        let resource = ctx.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        );

        // TODO: Check if it is possible to make this faster
        let mut data = vec![0; resource.size() as usize];
        unsafe {
            let status = cubecl_hip_sys::hipMemcpyDtoHAsync(data.as_mut_ptr() as *mut _, resource.ptr, resource.size() as usize, ctx.stream);
            assert_eq!(status, HIP_SUCCESS, "Should copy data from device to host");
        };
        ctx.sync();
        data
    }
}

impl<MM: MemoryManagement<HipStorage>> ComputeServer for HipServer<MM> {
    type Kernel = Box<dyn CubeTask>;
    type DispatchOptions = CubeCount<Self>;
    type Storage = HipStorage;
    type MemoryManagement = MM;
    type FeatureSet = FeatureSet;
    type Properties = Properties;

    fn read(&mut self, binding: server::Binding<Self>) -> Reader {
        reader_from_concrete(self.read_sync(binding))
    }

    fn create(&mut self, data: &[u8]) -> server::Handle<Self> {
        let handle = self.empty(data.len());
        let ctx = self.get_context();

        let binding = handle.clone().binding();
        let resource = ctx.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        );

        unsafe {
            let status = cubecl_hip_sys::hipMemcpyHtoDAsync(resource.ptr, data as *const _ as *mut _, data.len(), ctx.stream);
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle<Self> {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size, &[]);
        server::Handle::new(handle, None, None)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: Self::DispatchOptions,
        bindings: Vec<server::Binding<Self>>,
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
                cubecl_runtime::debug::ProfileLevel::Basic => {
                    if let Some(val) = name.split("<").next() {
                        val.split("::").last().unwrap_or(name).to_string()
                    } else {
                        name.to_string()
                    }
                }
                cubecl_runtime::debug::ProfileLevel::Full => {
                    format!("{name}: {kernel_id} CubeCount {count:?}")
                }
            };

            self.logger
                .register_profiled(info, start.elapsed().unwrap());
        } else {
            ctx.execute_task(kernel_id, count, resources);
        }
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
        ctx.memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }
}

impl<MM: MemoryManagement<HipStorage>> HipContext<MM> {
    pub fn new(
        memory_management: MM,
        stream: cubecl_hip_sys::hipStream_t,
        context: *mut cubecl_hip_sys::hipCtx_t,
    ) -> Self {
        Self {
            memory_management,
            module_names: HashMap::new(),
            stream,
            context,
        }
    }

    fn sync(&mut self) {
        unsafe {
            let status = cubecl_hip_sys::hipStreamSynchronize(self.stream);
            assert_eq!(status, HIP_SUCCESS, "Should successfuly synchronize stream");
        };
        self.memory_management.storage().flush();
    }

    fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        cube_kernel: Box<dyn CubeTask>,
        logger: &mut DebugLogger,
        mode: ExecutionMode,
    ) {
        // CubeCL compilation
        // jitc = just-in-time compiled
        let mut jitc_kernel = cube_kernel.compile(mode);

        if logger.is_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp_code(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }
        let jitc_kernel = logger.debug(jitc_kernel);
        let kernel_name = CString::new(kernel_id.to_string()).unwrap();

        // Create HIP Program
        let program = unsafe {
            let source = CString::new(jitc_kernel.source.clone()).unwrap();
            let mut program: cubecl_hip_sys::hiprtcProgram = std::ptr::null_mut();
            let status = cubecl_hip_sys::hiprtcCreateProgram(
                &mut program,
                source.as_ptr(),
                kernel_name.as_ptr(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut());
            assert_eq!(status, hiprtcResult_HIPRTC_SUCCESS, "Should create the program");
            program
        };
        // Compile HIP program
        unsafe {
            let status = cubecl_hip_sys::hiprtcCompileProgram(program, 0, std::ptr::null_mut());
            if status != hiprtcResult_HIPRTC_SUCCESS {
                let mut log_size: usize = 0;
                let status = cubecl_hip_sys::hiprtcGetProgramLogSize(program, &mut log_size as *mut usize);
                assert_eq!(status, hiprtcResult_HIPRTC_SUCCESS, "Should retrieve the compilation log size");
                println!("Compilation log size: {log_size}");
                let mut log_buffer = vec![0i8;log_size];
                let status = cubecl_hip_sys::hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());
                assert_eq!(status, hiprtcResult_HIPRTC_SUCCESS, "Should retrieve the compilation log contents");
                let log = CStr::from_ptr(log_buffer.as_ptr());
                let mut message = "[Compilation Error] ".to_string();
                for line in log.to_string_lossy().split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                panic!("{message}\n[Source]  \n{}", jitc_kernel.source);
            }
            assert_eq!(status, hiprtcResult_HIPRTC_SUCCESS, "Should compile the program");
        };
        // Get HIP compiled code from program
        let mut code_size: usize = 0;
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCodeSize(program, &mut code_size);
            assert_eq!(status, hiprtcResult_HIPRTC_SUCCESS, "Should get size of compiled code");
        }
        let mut code = vec![0i8;code_size];
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCode(program, code.as_mut_ptr());
            assert_eq!(status, hiprtcResult_HIPRTC_SUCCESS, "Should load compiled code");
        }
        // Create the HIP module
        let mut module: cubecl_hip_sys::hipModule_t = std::ptr::null_mut();
        unsafe {
            let status = cubecl_hip_sys::hipModuleLoadData(&mut module, code.as_ptr() as *const _);
            assert_eq!(status, HIP_SUCCESS, "Should load compiled code into module");
        }
        // Retrieve the HIP module function
        let mut func: cubecl_hip_sys::hipFunction_t = std::ptr::null_mut();
        unsafe {
            let status = cubecl_hip_sys::hipModuleGetFunction(&mut func, module, kernel_name.as_ptr());
            assert_eq!(status, HIP_SUCCESS, "Should return module function");
        }

        // register module
        self.module_names.insert(
            kernel_id.clone(),
            HipCompiledKernel {
                _module: module,
                func,
                cube_dim: jitc_kernel.cube_dim,
                shared_mem_bytes: jitc_kernel.shared_mem_bytes,
            },
        );
    }

    fn execute_task(
        &mut self,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: Vec<HipResource>,
    ) {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.as_binding())
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
                kernel.shared_mem_bytes as u32,
                self.stream,
                bindings.as_mut_ptr(),
                std::ptr::null_mut());
            assert_eq!(status, HIP_SUCCESS, "Should launch the kernel");
        };
    }
}

impl<MM: MemoryManagement<HipStorage>> HipServer<MM> {
    /// Create a new cuda server.
    pub(crate) fn new(index: usize, init: Box<dyn Fn(usize) -> HipContext<MM>>) -> Self {
        Self {
            state: HipServerState::Uninitialized {
                device_index: index,
                init,
            },
            logger: DebugLogger::new(),
        }
    }

    fn get_context(&mut self) -> &mut HipContext<MM> {
        self.get_context_with_logger().0
    }

    fn get_context_with_logger(&mut self) -> (&mut HipContext<MM>, &mut DebugLogger) {
        if let HipServerState::Uninitialized { device_index, init } = &self.state {
            let ctx = init(*device_index);

            self.state = HipServerState::Initialized { ctx };
        }
        if let HipServerState::Initialized { ctx } = &mut self.state {
            unsafe {
                cubecl_hip_sys::hipCtxSetCurrent(*ctx.context);
            };
            (ctx, &mut self.logger)
        } else {
            panic!("Context should be initialized");
        }
    }
}

