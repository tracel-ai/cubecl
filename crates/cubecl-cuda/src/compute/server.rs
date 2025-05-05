use cubecl_core::benchmark::ProfileDuration;
use cubecl_core::future::{self, DynFut};
use cubecl_cpp::{
    CudaCompiler, cuda::arch::CudaArchitecture, formatter::format_cpp, shared::CompilationOptions,
};

use cubecl_runtime::kernel_timestamps::KernelTimestamps;
use serde::{Deserialize, Serialize};

use super::fence::{Fence, SyncStream};
use super::storage::CudaStorage;
use super::{CudaResource, uninit_vec};
use cubecl_core::{
    Feature,
    ir::FloatKind,
    server::{BindingWithMeta, Bindings, Handle, TensorMapBinding},
};
use cubecl_core::{KernelId, prelude::*};
use cubecl_core::{
    compute::DebugInformation,
    ir::{Elem, IntKind, UIntKind},
};
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::{
    logging::{ProfileLevel, ServerLogger},
    storage::ComputeStorage,
};
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use cudarc::driver::sys::{
    CUDA_MEMCPY2D_st, CUctx_st, CUfunction_attribute, CUmemorytype, CUtensorMap,
    CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapL2promotion, CUtensorMapSwizzle,
    cuMemcpy2DAsync_v2, cuTensorMapEncodeIm2col, cuTensorMapEncodeTiled,
};
use cudarc::driver::sys::{CUfunc_st, CUtensorMapInterleave};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::{ffi::CStr, os::raw::c_void};
use std::{ffi::CString, mem::MaybeUninit};

#[cfg(feature = "compilation-cache")]
use cubecl_common::cache::{Cache, CacheOption};

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    logger: ServerLogger,
}

#[derive(Debug)]
pub(crate) struct CudaContext {
    context: *mut CUctx_st,
    stream: cudarc::driver::sys::CUstream,
    memory_management: MemoryManagement<CudaStorage>,
    module_names: HashMap<KernelId, CompiledKernel>,
    #[cfg(feature = "compilation-cache")]
    ptx_cache: Option<Cache<String, PtxCacheEntry>>,
    timestamps: KernelTimestamps,
    pub(crate) arch: CudaArchitecture,
    compilation_options: CompilationOptions,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    cluster_dim: Option<(u32, u32, u32)>,
    ptx: Vec<i8>,
}

#[derive(Debug)]
struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

unsafe impl Send for CudaServer {}

impl CudaServer {
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

            let mut data = uninit_vec(resource.size() as usize);

            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(&mut data, resource.ptr, ctx.stream)
                    .unwrap();
            };
            result.push(data);
        }

        let fence = ctx.fence();
        async move {
            // Wait for the fence. This is still sync, so the future will still complete in a
            // single poll.
            fence.wait();
            result
        }
    }

    fn read_tensor_async(
        &mut self,
        handles: Vec<BindingWithMeta>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + use<> {
        let ctx = self.get_context();
        let mut result = Vec::with_capacity(handles.len());

        for handle in handles {
            let binding = handle.binding;
            let resource = ctx
                .memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource");
            let mut data = vec![0; handle.shape.iter().product::<usize>() * handle.elem_size];

            let rank = handle.shape.len();
            if rank <= 1 {
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_async(&mut data, resource.ptr, ctx.stream)
                        .unwrap();
                };
                result.push(data);
                continue;
            }

            let dim_x = handle.shape[rank - 1];
            let width_bytes = dim_x * handle.elem_size;
            let dim_y: usize = handle.shape.iter().rev().skip(1).product();
            let pitch = handle.strides[rank - 2] * handle.elem_size;

            let cpy = CUDA_MEMCPY2D_st {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcDevice: resource.ptr,
                srcPitch: pitch,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                dstHost: data.as_mut_ptr() as *mut c_void,
                dstPitch: width_bytes,
                WidthInBytes: width_bytes,
                Height: dim_y,
                ..Default::default()
            };

            unsafe {
                cuMemcpy2DAsync_v2(&cpy, ctx.stream).result().unwrap();
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

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn CubeTask<CudaCompiler>>;
    type Storage = CudaStorage;
    type Feature = Feature;
    type Info = ();

    fn read(&mut self, bindings: Vec<server::Binding>) -> DynFut<Vec<Vec<u8>>> {
        Box::pin(self.read_async(bindings))
    }

    fn read_tensor(&mut self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        Box::pin(self.read_tensor_async(bindings))
    }

    fn create(&mut self, data: &[u8]) -> server::Handle {
        let handle = self.empty(data.len());
        let ctx = self.get_context();

        let binding = handle.clone().binding();
        let resource = ctx
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .expect("Failed to find resource");

        unsafe {
            cudarc::driver::result::memcpy_htod_async(resource.ptr, data, ctx.stream).unwrap();
        }

        handle
    }

    fn create_tensor(
        &mut self,
        data: &[u8],
        shape: &[usize],
        elem_size: usize,
    ) -> (Handle, Vec<usize>) {
        let rank = shape.len();
        if rank <= 1 {
            let handle = self.create(data);
            return (handle, vec![1]);
        }
        let (handle, strides) = self.empty_tensor(shape, elem_size);
        let ctx = self.get_context();

        let binding = handle.clone().binding();
        let resource = ctx
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .expect("Failed to find resource");

        let dim_x = shape[rank - 1];
        let width_bytes = dim_x * elem_size;
        let dim_y: usize = shape.iter().rev().skip(1).product();
        let pitch = strides[rank - 2] * elem_size;

        let cpy = CUDA_MEMCPY2D_st {
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: data.as_ptr() as *const c_void,
            srcPitch: width_bytes,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstDevice: resource.ptr,
            dstPitch: pitch,
            WidthInBytes: width_bytes,
            Height: dim_y,
            ..Default::default()
        };

        unsafe {
            cuMemcpy2DAsync_v2(&cpy, ctx.stream).result().unwrap();
        }

        (handle, strides)
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size as u64, None);
        server::Handle::new(handle, None, None, size as u64)
    }

    fn empty_tensor(&mut self, shape: &[usize], elem_size: usize) -> (Handle, Vec<usize>) {
        let rank = shape.len();
        let ctx = self.get_context();
        let width = *shape.last().unwrap_or(&1);
        let height: usize = shape.iter().rev().skip(1).product();
        let height = height.max(1);
        let width_bytes = width * elem_size;
        // This should be done with cuMemAllocPitch, but that would propagate changes much deeper
        // through memory management. So just manually pitch to alignment for now.
        let pitch = width_bytes.next_multiple_of(Self::Storage::ALIGNMENT as usize);
        let size = height * pitch;
        let handle = ctx.memory_management.reserve(size as u64, None);
        let mut strides = vec![1; rank];
        if rank > 1 {
            strides[rank - 2] = pitch / elem_size;
        }
        if rank > 2 {
            for i in (0..rank - 2).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        let mem_handle = server::Handle::new(handle, None, None, size as u64);
        (mem_handle, strides)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
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
                let data = future::block_on(self.read_async(vec![binding]));
                let data = bytemuck::cast_slice(&data[0]);
                assert!(
                    data.len() == 3,
                    "Dynamic cube count should contain 3 values"
                );
                (data[0], data[1], data[2])
            }
        };

        let (scalars, scalar_bindings) = if self.ctx.compilation_options.grid_constants {
            let mut scalars = Vec::with_capacity(bindings.scalars.len() + 1);
            // We need to sort by largest first to have proper packed alignment. Assumes device
            // pointers are 64-bit aligned, which I believe is true on all cards that support grid
            // constants regardless. Metadata is inserted after the 8-aligned scalars to ensure proper
            // packing
            for binding in bindings.scalars.values().filter(|it| it.elem.size() == 8) {
                scalars.push(binding.data.as_ptr() as *const _ as *mut c_void);
            }
            if bindings.metadata.static_len > 0 {
                scalars.push(bindings.metadata.data.as_ptr() as *const _ as *mut c_void);
            }
            for size in [4, 2, 1] {
                for binding in bindings
                    .scalars
                    .values()
                    .filter(|it| it.elem.size() == size)
                {
                    scalars.push(binding.data.as_ptr() as *const _ as *mut c_void);
                }
            }

            let mut handles = Vec::new();
            if bindings.metadata.static_len > 0 {
                let dyn_meta = &bindings.metadata.data[bindings.metadata.static_len..];
                handles.push(self.create(bytemuck::cast_slice(dyn_meta)));
            }

            (scalars, handles)
        } else {
            let mut handles = Vec::new();
            if !bindings.metadata.data.is_empty() {
                handles.push(self.create(bytemuck::cast_slice(&bindings.metadata.data)))
            }
            handles.extend(
                bindings
                    .scalars
                    .values()
                    .map(|scalar| self.create(scalar.data())),
            );
            (Vec::new(), handles)
        };

        let (ctx, logger) = self.get_context_with_logger();

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, logger, mode);
        }

        let tensor_maps: Vec<_> = bindings
            .tensor_maps
            .into_iter()
            .map(|TensorMapBinding { map, binding }| {
                let resource = ctx
                    .memory_management
                    .get_resource(
                        binding.memory.clone(),
                        binding.offset_start,
                        binding.offset_end,
                    )
                    .expect("Failed to find resource");
                let device_ptr = resource.ptr as *mut c_void;
                debug_assert!(
                    device_ptr as usize % 16 == 0,
                    "Tensor pointer must be 16 byte aligned"
                );
                let mut map_ptr = MaybeUninit::zeroed();

                let shape: Vec<_> = map.shape.iter().rev().map(|s| *s as u64).collect();
                let strides: Vec<_> = map
                    .strides
                    .iter()
                    .rev()
                    .skip(1)
                    .map(|s| *s as u64 * map.elem.size() as u64)
                    .collect();
                let elem_stride: Vec<_> = map.elem_stride.iter().rev().map(|s| *s as u32).collect();

                debug_assert!(
                    strides.iter().all(|it| it % 16 == 0),
                    "Strides must be 16 byte aligned"
                );

                match &map.format {
                    TensorMapFormat::Tiled { tile_size } => unsafe {
                        debug_assert_eq!(tile_size.len(), map.rank, "Tile shape should match rank");
                        let tile_size: Vec<_> = tile_size.iter().rev().copied().collect();

                        cuTensorMapEncodeTiled(
                            map_ptr.as_mut_ptr(),
                            elem_to_tensor_map_type(map.elem),
                            map.rank as u32,
                            device_ptr,
                            shape.as_ptr(),
                            strides.as_ptr(),
                            tile_size.as_ptr(),
                            elem_stride.as_ptr(),
                            interleave_to_cuda(map.interleave),
                            swizzle_to_cuda(map.swizzle),
                            prefetch_to_cuda(map.prefetch),
                            oob_to_cuda(map.oob_fill),
                        )
                        .result()
                        .unwrap()
                    },
                    TensorMapFormat::Im2col {
                        pixel_box_lower_corner,
                        pixel_box_upper_corner,
                        channels_per_pixel,
                        pixels_per_column,
                    } => unsafe {
                        debug_assert_eq!(pixel_box_lower_corner.len(), map.rank - 2);
                        debug_assert_eq!(pixel_box_upper_corner.len(), map.rank - 2);

                        let lower_corner: Vec<_> =
                            pixel_box_lower_corner.iter().rev().copied().collect();
                        let upper_corner: Vec<_> =
                            pixel_box_upper_corner.iter().rev().copied().collect();

                        cuTensorMapEncodeIm2col(
                            map_ptr.as_mut_ptr(),
                            elem_to_tensor_map_type(map.elem),
                            map.rank as u32,
                            device_ptr,
                            shape.as_ptr(),
                            strides.as_ptr(),
                            lower_corner.as_ptr(),
                            upper_corner.as_ptr(),
                            *channels_per_pixel,
                            *pixels_per_column,
                            elem_stride.as_ptr(),
                            interleave_to_cuda(map.interleave),
                            swizzle_to_cuda(map.swizzle),
                            prefetch_to_cuda(map.prefetch),
                            oob_to_cuda(map.oob_fill),
                        )
                        .result()
                        .unwrap()
                    },
                    TensorMapFormat::Im2colWide { .. } => {
                        unimplemented!("Not yet implemented in cudarc")
                    }
                };
                unsafe { map_ptr.assume_init() }
            })
            .collect::<_>();

        let mut resources = bindings
            .buffers
            .into_iter()
            .map(|binding| find_resource(ctx, binding))
            .collect::<Vec<_>>();
        resources.extend(
            scalar_bindings
                .into_iter()
                .map(|s| find_resource(ctx, s.binding())),
        );

        if let Some(level) = profile_level {
            ctx.sync();
            let start = std::time::SystemTime::now();
            ctx.execute_task(kernel_id, count, &tensor_maps, &resources, &scalars);
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
            ctx.execute_task(kernel_id, count, &tensor_maps, &resources, &scalars);
        }
    }

    fn flush(&mut self) {}

    fn sync(&mut self) -> DynFut<()> {
        self.logger.profile_summary();
        Box::pin(self.sync_stream_async())
    }

    fn start_profile(&mut self) {
        // Wait for current work to be done.
        self.ctx.sync();
        self.ctx.timestamps.start();
    }

    fn end_profile(&mut self) -> ProfileDuration {
        self.logger.profile_summary();
        self.ctx.sync();
        self.ctx.timestamps.stop()
    }

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<CudaResource> {
        let ctx = self.get_context();
        BindingResource::new(
            binding.clone(),
            ctx.memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource"),
        )
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.ctx.memory_management.cleanup(true);
    }
}

fn find_resource(ctx: &mut CudaContext, binding: server::Binding) -> CudaResource {
    ctx.memory_management
        .get_resource(binding.memory, binding.offset_start, binding.offset_end)
        .expect("Failed to find resource")
}

impl CudaContext {
    pub fn new(
        memory_management: MemoryManagement<CudaStorage>,
        compilation_options: CompilationOptions,
        stream: cudarc::driver::sys::CUstream,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            context,
            memory_management,
            module_names: HashMap::new(),
            #[cfg(feature = "compilation-cache")]
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
            stream,
            arch,
            timestamps: KernelTimestamps::default(),
            compilation_options,
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
            cudarc::driver::result::stream::synchronize(self.stream).unwrap();
        };
    }

    fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        logger: &mut ServerLogger,
        mode: ExecutionMode,
    ) {
        #[cfg(feature = "compilation-cache")]
        let name = if let Some(cache) = &self.ptx_cache {
            let name = kernel_id.stable_format();

            if let Some(entry) = cache.get(&name) {
                log::trace!("Using PTX cache");
                self.load_ptx(
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
                return;
            }
            Some(name)
        } else {
            None
        };

        log::trace!("Compiling kernel");

        let mut kernel_compiled =
            kernel.compile(&mut Default::default(), &self.compilation_options, mode);

        if logger.compilation_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let compute_kernel = kernel_compiled.repr.as_ref().unwrap();
        let cube_dim = kernel_compiled.cube_dim;
        let fast_math = compute_kernel.flags.inst_fast_math;
        let arch = format!("--gpu-architecture=sm_{}", self.arch);

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if fast_math {
            options.push("--use_fast_math");
        }

        #[cfg(feature = "compilation-cache")]
        let cluster_dim = compute_kernel.cluster_dim;

        let kernel_compiled = logger.log_compilation(kernel_compiled);

        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::from_str(&kernel_compiled.source).unwrap();
            let program = cudarc::nvrtc::result::create_program(source.as_c_str(), None).unwrap();
            if cudarc::nvrtc::result::compile_program(program, &options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).unwrap();
                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                let source = kernel
                    .compile(&mut Default::default(), &self.compilation_options, mode)
                    .source;
                panic!("{message}\n[Source]  \n{source}");
            };
            cudarc::nvrtc::result::get_ptx(program).unwrap()
        };

        let repr: cubecl_cpp::ComputeKernel<cubecl_cpp::cuda::CudaDialect> =
            kernel_compiled.repr.unwrap();

        #[cfg(feature = "compilation-cache")]
        if let Some(cache) = &mut self.ptx_cache {
            cache
                .insert(
                    name.unwrap(),
                    PtxCacheEntry {
                        entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                        cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
                        shared_mem_bytes: repr.shared_memory_size(),
                        cluster_dim: cluster_dim.map(|cluster| (cluster.x, cluster.y, cluster.z)),
                        ptx: ptx.clone(),
                    },
                )
                .unwrap();
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            repr.shared_memory_size(),
        );
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<i8>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) {
        let func_name = CString::new(entrypoint_name).unwrap();
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
        tensor_maps: &[CUtensorMap],
        resources: &[CudaResource],
        scalars: &[*mut c_void],
    ) {
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.as_binding()));
        bindings.extend(scalars);

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .unwrap();
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                self.stream,
                &mut bindings,
            )
            .unwrap();
        };
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(ctx: CudaContext) -> Self {
        let logger = ServerLogger::default();
        Self { ctx, logger }
    }

    fn get_context(&mut self) -> &mut CudaContext {
        self.get_context_with_logger().0
    }

    fn get_context_with_logger(&mut self) -> (&mut CudaContext, &mut ServerLogger) {
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
        // If it is installed as part of the distribution
        return if std::fs::exists("/usr/local/cuda").is_ok_and(|exists| exists) {
            Some(PathBuf::from("/usr/local/cuda"))
        } else if std::fs::exists("/opt/cuda").is_ok_and(|exists| exists) {
            Some(PathBuf::from("/opt/cuda"))
        } else if std::fs::exists("/usr/bin/nvcc").is_ok_and(|exists| exists) {
            // Maybe the compiler was installed within the user path.
            Some(PathBuf::from("/usr"))
        } else {
            None
        };
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

fn elem_to_tensor_map_type(elem: Elem) -> CUtensorMapDataType {
    use cudarc::driver::sys::CUtensorMapDataType::*;
    match elem {
        Elem::Float(kind) => match kind {
            FloatKind::F16 => CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            FloatKind::BF16 => CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            FloatKind::Flex32 | FloatKind::F32 => CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            FloatKind::TF32 => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            FloatKind::F64 => CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        },
        Elem::Int(kind) => match kind {
            IntKind::I8 | IntKind::I16 => unimplemented!("Not supported for tensor map type"),
            IntKind::I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            IntKind::I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
        },
        Elem::UInt(kind) => match kind {
            UIntKind::U8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            UIntKind::U16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            UIntKind::U32 => CU_TENSOR_MAP_DATA_TYPE_UINT32,
            UIntKind::U64 => CU_TENSOR_MAP_DATA_TYPE_UINT64,
        },
        Elem::AtomicFloat(_) | Elem::AtomicInt(_) | Elem::AtomicUInt(_) => {
            unimplemented!("Not supported for tensor map type")
        }
        _ => unimplemented!(),
    }
}

fn interleave_to_cuda(interleave: TensorMapInterleave) -> CUtensorMapInterleave {
    use cudarc::driver::sys::CUtensorMapInterleave::*;
    match interleave {
        TensorMapInterleave::None => CU_TENSOR_MAP_INTERLEAVE_NONE,
        TensorMapInterleave::B16 => CU_TENSOR_MAP_INTERLEAVE_16B,
        TensorMapInterleave::B32 => CU_TENSOR_MAP_INTERLEAVE_32B,
    }
}

fn swizzle_to_cuda(swizzle: TensorMapSwizzle) -> CUtensorMapSwizzle {
    use cudarc::driver::sys::CUtensorMapSwizzle::*;
    match swizzle {
        TensorMapSwizzle::None => CU_TENSOR_MAP_SWIZZLE_NONE,
        TensorMapSwizzle::B32 => CU_TENSOR_MAP_SWIZZLE_32B,
        TensorMapSwizzle::B64 => CU_TENSOR_MAP_SWIZZLE_64B,
        TensorMapSwizzle::B128 => CU_TENSOR_MAP_SWIZZLE_128B,
    }
}

fn prefetch_to_cuda(prefetch: TensorMapPrefetch) -> CUtensorMapL2promotion {
    use cudarc::driver::sys::CUtensorMapL2promotion::*;
    match prefetch {
        TensorMapPrefetch::None => CU_TENSOR_MAP_L2_PROMOTION_NONE,
        TensorMapPrefetch::B64 => CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        TensorMapPrefetch::B128 => CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        TensorMapPrefetch::B256 => CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    }
}

fn oob_to_cuda(fill: OobFill) -> CUtensorMapFloatOOBfill {
    use cudarc::driver::sys::CUtensorMapFloatOOBfill::*;
    match fill {
        OobFill::Zero => CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        OobFill::NaN => CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
    }
}
