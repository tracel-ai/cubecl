use cubecl_core::server::ProfilingToken;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::shared::CompilationOptions;

use super::fence::{Fence, SyncStream};
use super::storage::HipStorage;
use super::{HipResource, uninit_vec};
use crate::runtime::HipCompiler;
use cubecl_common::benchmark::ProfileDuration;
use cubecl_common::future::DynFut;
use cubecl_core::compute::CubeTask;
use cubecl_core::compute::DebugInformation;
use cubecl_core::prelude::*;
use cubecl_core::{Feature, server::Bindings};
use cubecl_hip_sys::{HIP_SUCCESS, get_hip_include_path, hiprtcResult_HIPRTC_SUCCESS};
use cubecl_runtime::kernel_timestamps::KernelTimestamps;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::memory_management::offset_handles;
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::storage::ComputeStorage;
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::future::Future;
use std::sync::Arc;

#[cfg(feature = "compilation-cache")]
use cubecl_common::cache::{Cache, CacheOption};

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
}

#[derive(Debug)]
pub(crate) struct HipContext {
    stream: cubecl_hip_sys::hipStream_t,
    memory_management: MemoryManagement<HipStorage>,
    module_names: HashMap<KernelId, HipCompiledKernel>,
    timestamps: KernelTimestamps,
    compilation_options: CompilationOptions,
    #[cfg(feature = "compilation-cache")]
    compilation_cache: Cache<String, CompilationCacheEntry>,
}

#[cfg(feature = "compilation-cache")]
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    binary: Vec<i8>,
}

#[derive(Debug)]
struct HipCompiledKernel {
    _module: cubecl_hip_sys::hipModule_t,
    func: cubecl_hip_sys::hipFunction_t,
    cube_dim: CubeDim,
}

unsafe impl Send for HipServer {}

impl HipServer {
    fn read_sync(&mut self, binding: server::Binding) -> Vec<u8> {
        let ctx = self.get_context();
        let resource = ctx
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .expect("Failed to find resource");

        let mut data = uninit_vec(resource.size as usize);
        unsafe {
            let status = cubecl_hip_sys::hipMemcpyDtoHAsync(
                data.as_mut_ptr() as *mut _,
                resource.ptr,
                resource.size as usize,
                ctx.stream,
            );
            assert_eq!(status, HIP_SUCCESS, "Should copy data from device to host");
        };
        ctx.sync();
        data
    }

    fn read_async(
        &mut self,
        bindings: Vec<server::Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + use<> {
        let ctx = self.get_context();
        let mut result = Vec::with_capacity(bindings.len());

        for binding in bindings {
            let resource = ctx
                .memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource");

            let mut data = uninit_vec(resource.size as usize);
            unsafe {
                let status = cubecl_hip_sys::hipMemcpyDtoHAsync(
                    data.as_mut_ptr() as *mut _,
                    resource.ptr,
                    resource.size as usize,
                    ctx.stream,
                );
                assert_eq!(status, HIP_SUCCESS, "Should copy data from device to host");
            };
            result.push(data);
        }

        let fence = ctx.fence();

        async move {
            fence.wait();
            result
        }
    }

    fn sync_stream_async(&mut self) -> impl Future<Output = ()> + Send + use<> {
        let ctx = self.get_context();
        // We can't use a fence here because no action has been recorded on the context.
        // We need at least one action to be recorded after the context is initialized
        // with `cudarc::driver::result::ctx::set_current(self.ctx.context)` for the fence
        // to have any effect. Otherwise, it seems to be ignored.
        let sync = ctx.lazy_sync_stream();

        async move {
            sync.wait();
        }
    }
}

impl ComputeServer for HipServer {
    type Kernel = Box<dyn CubeTask<HipCompiler>>;
    type Storage = HipStorage;
    type Feature = Feature;
    type Info = ();

    fn read(&mut self, bindings: Vec<server::Binding>) -> DynFut<Vec<Vec<u8>>> {
        Box::pin(self.read_async(bindings))
    }

    fn read_tensor(&mut self, bindings: Vec<server::BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let bindings = bindings.into_iter().map(|it| it.binding).collect();
        Box::pin(self.read_async(bindings))
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        let ctx = self.get_context();
        ctx.memory_management.cleanup(true);
    }

    fn create(&mut self, data: &[u8]) -> server::Handle {
        let handle = self.empty(data.len());

        let binding = handle.clone().binding();
        self.copy_to_binding(binding, data);
        handle
    }

    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(server::Handle, Vec<usize>)> {
        let handles_strides = self.empty_tensors(shapes.clone(), elem_sizes);
        for i in 0..data.len() {
            let data = data[i];
            let (handle, _) = &handles_strides[i];
            let binding = handle.clone().binding();
            self.copy_to_binding(binding, data);
        }
        handles_strides
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size as u64, None);
        server::Handle::new(handle, None, None, size as u64)
    }

    fn empty_tensors(
        &mut self,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(server::Handle, Vec<usize>)> {
        let align = <Self::Storage as ComputeStorage>::ALIGNMENT as usize;

        let mut total_size = 0;
        let mut strides = Vec::new();
        let mut sizes = Vec::new();

        for (shape, elem_size) in shapes.into_iter().zip(elem_sizes) {
            let size = (shape.iter().product::<usize>() * elem_size).next_multiple_of(align);
            strides.push(contiguous_strides(shape));
            sizes.push(size);
            total_size += size;
        }

        let mem_handle = self.empty(total_size);
        let handles = offset_handles(mem_handle, &sizes);

        handles.into_iter().zip(strides).collect()
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: HIP doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
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

        let Bindings {
            buffers,
            metadata,
            scalars,
            tensor_maps,
        } = bindings;

        debug_assert!(tensor_maps.is_empty(), "Can't use tensor maps on HIP");
        let info = self.create(bytemuck::cast_slice(&metadata.data));
        let scalars: Vec<_> = scalars.values().map(|s| self.create(s.data())).collect();

        let ctx = self.get_context();

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, mode, logger);
        }

        let mut resources: Vec<_> = buffers.into_iter().map(|b| find_resource(ctx, b)).collect();
        resources.push(find_resource(ctx, info.clone().binding()));
        resources.extend(scalars.into_iter().map(|s| find_resource(ctx, s.binding())));

        ctx.execute_task(kernel_id, count, resources);
    }

    fn flush(&mut self) {}

    fn sync(&mut self) -> DynFut<()> {
        Box::pin(self.sync_stream_async())
    }

    fn start_profile(&mut self) -> ProfilingToken {
        cubecl_common::future::block_on(self.sync());
        self.ctx.timestamps.start()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> ProfileDuration {
        cubecl_common::future::block_on(self.sync());
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<HipResource> {
        let ctx = self.get_context();
        BindingResource::new(
            binding.clone(),
            ctx.memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Can't find resource"),
        )
    }
}

fn find_resource(ctx: &mut HipContext, binding: server::Binding) -> HipResource {
    ctx.memory_management
        .get_resource(binding.memory, binding.offset_start, binding.offset_end)
        .expect("Failed to find resource")
}

impl HipContext {
    pub fn new(
        memory_management: MemoryManagement<HipStorage>,
        compilation_options: CompilationOptions,
        stream: cubecl_hip_sys::hipStream_t,
    ) -> Self {
        Self {
            memory_management,
            module_names: HashMap::new(),
            stream,
            timestamps: KernelTimestamps::default(),
            compilation_options,
            #[cfg(feature = "compilation-cache")]
            compilation_cache: Cache::new("hip/compilation", CacheOption::default()),
        }
    }

    fn fence(&mut self) -> Fence {
        Fence::new(self.stream)
    }

    fn lazy_sync_stream(&mut self) -> SyncStream {
        SyncStream::new(self.stream)
    }

    fn sync(&mut self) {
        unsafe {
            let status = cubecl_hip_sys::hipStreamSynchronize(self.stream);
            assert_eq!(
                status, HIP_SUCCESS,
                "Should successfully synchronize stream"
            );
        };
        self.memory_management.storage().flush();
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }

    fn compile_kernel(
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
        let optimization_level = CString::new("-o=3").unwrap();
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

    fn execute_task(
        &mut self,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: Vec<HipResource>,
    ) {
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
                // Shared memory is specified statically in the kernel, and no dynamic shared
                // memory is supported yet in the kernel, which would be that value for the
                // current kernel launch.
                0,
                self.stream,
                bindings.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory {
                panic!("Error: Cannot launch kernel (Out of memory)\n{kernel_id}")
            }
            assert_eq!(status, HIP_SUCCESS, "Should launch the kernel");
        };
    }
}

impl HipServer {
    /// Create a new hip server.
    pub(crate) fn new(ctx: HipContext) -> Self {
        Self { ctx }
    }

    fn get_context(&mut self) -> &mut HipContext {
        &mut self.ctx
    }

    fn copy_to_binding(&mut self, binding: server::Binding, data: &[u8]) {
        let ctx = self.get_context();
        let resource = ctx
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .unwrap();

        unsafe {
            let status = cubecl_hip_sys::hipMemcpyHtoDAsync(
                resource.ptr,
                data as *const _ as *mut _,
                data.len(),
                ctx.stream,
            );
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
    }
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
