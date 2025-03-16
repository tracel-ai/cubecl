use cubecl_common::{
    OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle,
};
use cubecl_cpp::{
    cuda::arch::CudaArchitecture, formatter::format_cpp, shared::CompilationOptions, CudaCompiler,
};
use serde::{Deserialize, Serialize};

use super::fence::{Fence, SyncStream};
use super::storage::CudaStorage;
use super::{uninit_vec, CudaResource};
use cubecl_core::{
    compute::DebugInformation,
    ir::{Elem, IntKind, UIntKind},
};
use cubecl_core::{ir::FloatKind, Feature};
use cubecl_core::{prelude::*, KernelId};
use cubecl_runtime::debug::{DebugLogger, ProfileLevel};
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use cubecl_runtime::{TimestampsError, TimestampsResult};
use cudarc::driver::sys::{
    CUctx_st, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapL2promotion,
    CUtensorMapSwizzle,
};
use cudarc::driver::sys::{CUfunc_st, CUtensorMapInterleave};
use std::ffi::CString;
use std::future::Future;
use std::path::PathBuf;
use std::time::Instant;
use std::{collections::HashMap, mem::MaybeUninit};
use std::{ffi::CStr, os::raw::c_void};

#[cfg(feature = "cache-ptx")]
use cubecl_common::cache::{Cache, CacheOption};

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
    #[cfg(feature = "cache-ptx")]
    ptx_cache: Cache<String, PtxCacheEntry>,
    timestamps: KernelTimestamps,
    pub(crate) arch: CudaArchitecture,
    compilation_options: CompilationOptions,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    ptx: Vec<i8>,
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
    fn read_sync(&mut self, binding: server::Binding) -> Vec<u8> {
        let ctx = self.get_context();
        let resource = ctx
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .expect("Failed to find resource");

        let mut data = uninit_vec(resource.size() as usize);

        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(&mut data, resource.ptr, ctx.stream).unwrap();
        };

        ctx.sync();

        data
    }

    fn read_async(
        &mut self,
        bindings: Vec<server::Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + 'static + Send {
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
            fence.wait();
            result
        }
    }

    fn sync_stream_async(&mut self) -> impl Future<Output = ()> + 'static + Send {
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

    fn read(
        &mut self,
        bindings: Vec<server::Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + 'static {
        self.read_async(bindings)
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

    fn empty(&mut self, size: usize) -> server::Handle {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size as u64);
        server::Handle::new(handle, None, None, size as u64)
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
            .iter()
            .map(|binding| {
                let mut resource = ctx
                    .memory_management
                    .get_resource(
                        binding.memory.clone(),
                        binding.offset_start,
                        binding.offset_end,
                    )
                    .expect("Failed to find resource");
                if let Some(map) = &binding.tensor_map {
                    let lib = cudarc::driver::sys::lib();
                    let mut map_ptr = Box::new_uninit();
                    let data_ty = elem_to_tmap_type(map.elem);
                    match &map.format {
                        TensorMapFormat::Tiled { tile_size } => lib.cuTensorMapEncodeTiled(
                            map_ptr.as_mut_ptr(),
                            data_ty,
                            map.rank as u32,
                            resource.as_binding(),
                            map.shape.as_ptr(),
                            map.strides.as_ptr(),
                            tile_size.as_ptr(),
                            map.elem_stride.as_ptr(),
                            interleave_to_cuda(map.interleave),
                            swizzle_to_cuda(map.swizzle),
                            prefetch_to_cuda(map.prefetch),
                            oob_to_cuda(map.oob_fill),
                        ),
                        TensorMapFormat::Im2col {
                            pixel_box_lower_corner,
                            pixel_box_upper_corner,
                            channels_per_pixel,
                            pixels_per_column,
                        } => lib.cuTensorMapEncodeIm2col(
                            map_ptr.as_mut_ptr(),
                            data_ty,
                            map.rank as u32,
                            resource.as_binding(),
                            map.shape.as_ptr(),
                            map.strides.as_ptr(),
                            pixel_box_lower_corner.as_ptr(),
                            pixel_box_upper_corner.as_ptr(),
                            *channels_per_pixel,
                            *pixels_per_column,
                            map.elem_stride.as_ptr(),
                            interleave_to_cuda(map.interleave),
                            swizzle_to_cuda(map.swizzle),
                            prefetch_to_cuda(map.prefetch),
                            oob_to_cuda(map.oob_fill),
                        ),
                        TensorMapFormat::Im2colWide { .. } => {
                            unimplemented!("Not yet implemented in cudarc")
                        }
                    };
                    resource.binding = Box::into_raw(map_ptr.assume_init()) as *mut c_void;
                }
                resource
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
        self.sync_stream_async()
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
        compilation_options: CompilationOptions,
        stream: cudarc::driver::sys::CUstream,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            context,
            memory_management,
            module_names: HashMap::new(),
            #[cfg(feature = "cache-ptx")]
            ptx_cache: Cache::new("cuda/ptx", CacheOption::default()),
            stream,
            arch,
            timestamps: KernelTimestamps::Disabled,
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
        logger: &mut DebugLogger,
        mode: ExecutionMode,
    ) {
        #[cfg(feature = "cache-ptx")]
        let name = kernel_id.stable_format();

        #[cfg(feature = "cache-ptx")]
        if let Some(entry) = self.ptx_cache.get(&name) {
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
        log::trace!("Compiling kernel");

        let mut kernel_compiled =
            kernel.compile(&mut Default::default(), &self.compilation_options, mode);

        if logger.is_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let compute_kernel = kernel_compiled.repr.as_ref().unwrap();
        let shared_mem_bytes = compute_kernel.shared_memory_size();
        let cube_dim = kernel_compiled.cube_dim;
        let fast_math = compute_kernel.fast_math;
        let arch = format!("--gpu-architecture=sm_{}", self.arch);

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if fast_math {
            options.push("--use_fast_math");
        }

        let kernel_compiled = logger.debug(kernel_compiled);

        let ptx = unsafe {
            let program = cudarc::nvrtc::result::create_program(kernel_compiled.source).unwrap();
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

        #[cfg(feature = "cache-ptx")]
        self.ptx_cache
            .insert(
                name,
                PtxCacheEntry {
                    entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                    cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
                    shared_mem_bytes,
                    ptx: ptx.clone(),
                },
            )
            .unwrap();

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            shared_mem_bytes,
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

fn elem_to_tmap_type(elem: Elem) -> CUtensorMapDataType {
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
