use cubecl_core::{
    compute::{CubeTask, DebugInformation},
    server::{DataTransferService, IoError},
};
use cubecl_core::{
    future::{self, DynFut},
    server::AllocationKind,
};
use cubecl_core::{
    ir::StorageType,
    server::{Allocation, AllocationDescriptor, ProfileError, ProfilingToken},
};
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};

use super::storage::gpu::{GpuResource, GpuStorage};
use super::sync::{Fence, SyncStream};
use crate::compute::{
    DataTransferItem, DataTransferRuntime, io::register_copies_to_bytes,
    storage::cpu::PinnedMemoryStorage,
};
use crate::{CudaCompiler, WmmaCompiler};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::ir::{ElemType, IntKind, UIntKind};
use cubecl_core::prelude::*;
use cubecl_core::{
    ir::FloatKind,
    server::{Bindings, CopyDescriptor, TensorMapBinding},
};
use cubecl_runtime::data_service::DataTransferId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use cubecl_runtime::{memory_management::offset_handles, timestamp_profiler::TimestampProfiler};
use cudarc::driver::sys::{
    CUDA_MEMCPY2D_st, CUctx_st, CUfunction_attribute, CUmemorytype, CUtensorMap,
    CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapL2promotion, CUtensorMapSwizzle,
    cuMemcpy2DAsync_v2, cuTensorMapEncodeIm2col, cuTensorMapEncodeTiled,
};
use cudarc::driver::sys::{CUfunc_st, CUtensorMapInterleave};
#[cfg(feature = "cuda-12080")]
use cudarc::driver::sys::{CUtensorMapIm2ColWideMode, cuTensorMapEncodeIm2colWide};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::c_char;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};
use std::{ffi::CString, mem::MaybeUninit};

#[cfg(feature = "compilation-cache")]
use cubecl_common::cache::{Cache, CacheOption};

pub(crate) const MB: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    mem_alignment: usize,
}

#[derive(Debug)]
pub(crate) struct CudaContext {
    context: *mut CUctx_st,
    pub(crate) stream: cudarc::driver::sys::CUstream,
    pub(crate) memory_management_gpu: MemoryManagement<GpuStorage>,
    pub(crate) memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
    module_names: HashMap<KernelId, CompiledKernel>,
    #[cfg(feature = "compilation-cache")]
    ptx_cache: Option<Cache<String, PtxCacheEntry>>,
    timestamps: TimestampProfiler,
    pub(crate) arch: CudaArchitecture,
    compilation_options: CompilationOptions,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    cluster_dim: Option<(u32, u32, u32)>,
    ptx: Vec<c_char>,
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
        descriptors: Vec<CopyDescriptor<'_>>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        let ctx = self.get_context();
        let result = register_copies_to_bytes(ctx, descriptors);
        let fence = ctx.fence();

        async move {
            fence.wait_sync();
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
    type Storage = GpuStorage;
    type Info = ();

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        Box::pin(self.read_async(descriptors))
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut strides = Vec::new();
        let mut sizes = Vec::new();
        let mut total_size = 0;

        for descriptor in descriptors {
            let pitch_align = match descriptor.kind {
                AllocationKind::Contiguous => 1,
                AllocationKind::Optimized => self.mem_alignment,
            };

            let rank = descriptor.shape.len();
            let width = *descriptor.shape.last().unwrap_or(&1);
            let height: usize = descriptor.shape.iter().rev().skip(1).product();
            let height = height.max(1);
            let width_bytes = width * descriptor.elem_size;
            let pitch = width_bytes.next_multiple_of(pitch_align);
            let size = height * pitch;
            total_size += size.next_multiple_of(self.mem_alignment);
            let mut stride = vec![1; rank];
            if rank > 1 {
                stride[rank - 2] = pitch / descriptor.elem_size;
            }
            if rank > 2 {
                for i in (0..rank - 2).rev() {
                    stride[i] = stride[i + 1] * descriptor.shape[i + 1];
                }
            }

            strides.push(stride);
            sizes.push(size);
        }

        let ctx = self.get_context();
        let handle = ctx.memory_management_gpu.reserve(total_size as u64)?;
        let mem_handle = server::Handle::new(
            handle,
            None,
            None,
            cubecl_common::stream_id::StreamId::current(),
            total_size as u64,
            0,
        );

        let handles = offset_handles(mem_handle, &sizes, self.mem_alignment);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError> {
        let ctx = self.get_context();

        for (descriptor, data) in descriptors {
            let CopyDescriptor {
                binding,
                shape,
                strides,
                elem_size,
            } = descriptor;
            let rank = shape.len();

            if !valid_strides(shape, strides) {
                return Err(IoError::UnsupportedStrides);
            }

            let resource = ctx
                .memory_management_gpu
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .ok_or(IoError::InvalidHandle)?;

            if rank > 1 {
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
            } else {
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(resource.ptr, data, ctx.stream)
                        .unwrap();
                }
            }
        }

        Ok(())
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
        stream_id: StreamId,
    ) {
        let stream_id = StreamId::current();
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = future::block_on(self.read_async(vec![CopyDescriptor::new(
                    binding,
                    &[3],
                    &[1],
                    4,
                )]))
                .unwrap();
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
            for binding in bindings.scalars.values().filter(|it| it.ty.size() == 8) {
                scalars.push(binding.data.as_ptr() as *const _ as *mut c_void);
            }
            if bindings.metadata.static_len > 0 {
                scalars.push(bindings.metadata.data.as_ptr() as *const _ as *mut c_void);
            }
            for size in [4, 2, 1] {
                for binding in bindings.scalars.values().filter(|it| it.ty.size() == size) {
                    scalars.push(binding.data.as_ptr() as *const _ as *mut c_void);
                }
            }

            let mut handles = Vec::new();
            if bindings.metadata.static_len > 0 {
                let dyn_meta = &bindings.metadata.data[bindings.metadata.static_len..];
                handles.push(
                    self.create_with_data(bytemuck::cast_slice(dyn_meta), stream_id)
                        .unwrap(),
                );
            }

            (scalars, handles)
        } else {
            let mut handles = Vec::new();
            if !bindings.metadata.data.is_empty() {
                handles.push(
                    self.create_with_data(bytemuck::cast_slice(&bindings.metadata.data), stream_id)
                        .unwrap(),
                )
            }
            handles.extend(
                bindings
                    .scalars
                    .values()
                    .map(|scalar| self.create_with_data(scalar.data(), stream_id).unwrap()),
            );
            (Vec::new(), handles)
        };

        let ctx = self.get_context();

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, mode, logger);
        }

        let tensor_maps: Vec<_> = bindings
            .tensor_maps
            .into_iter()
            .map(|TensorMapBinding { map, binding }| {
                let resource = ctx
                    .memory_management_gpu
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
                    .map(|s| *s as u64 * map.storage_ty.size() as u64)
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
                            elem_to_tensor_map_type(map.storage_ty),
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
                            elem_to_tensor_map_type(map.storage_ty),
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
                    #[cfg(feature = "cuda-12080")]
                    TensorMapFormat::Im2colWide {
                        pixel_box_lower_corner_width,
                        pixel_box_upper_corner_width,
                        channels_per_pixel,
                        pixels_per_column,
                    } => unsafe {
                        cuTensorMapEncodeIm2colWide(
                            map_ptr.as_mut_ptr(),
                            elem_to_tensor_map_type(map.storage_ty),
                            map.rank as u32,
                            device_ptr,
                            shape.as_ptr(),
                            strides.as_ptr(),
                            *pixel_box_lower_corner_width,
                            *pixel_box_upper_corner_width,
                            *channels_per_pixel,
                            *pixels_per_column,
                            elem_stride.as_ptr(),
                            interleave_to_cuda(map.interleave),
                            CUtensorMapIm2ColWideMode::CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
                            swizzle_to_cuda(map.swizzle),
                            prefetch_to_cuda(map.prefetch),
                            oob_to_cuda(map.oob_fill),
                        )
                        .result()
                        .unwrap()
                    },
                    #[cfg(not(feature = "cuda-12080"))]
                    TensorMapFormat::Im2colWide {
                        pixel_box_lower_corner_width: _,
                        pixel_box_upper_corner_width: _,
                        channels_per_pixel: _,
                        pixels_per_column: _,
                    } => panic!("CUDA version 12.8 required for tensor map format Im2colWide"),
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

        let result = ctx.execute_task(kernel_id, count, &tensor_maps, &resources, &scalars);

        match result {
            Ok(_) => {}
            Err(err) => match ctx.timestamps.is_empty() {
                true => panic!("{err:?}"),
                false => ctx.timestamps.error(ProfileError::Unknown(err)),
            },
        }
    }

    fn flush(&mut self) {}

    fn sync(&mut self) -> DynFut<()> {
        Box::pin(self.sync_stream_async())
    }

    fn start_profile(&mut self) -> ProfilingToken {
        // Wait for current work to be done.
        self.ctx.sync();
        self.ctx.timestamps.start()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        self.ctx.sync();
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<GpuResource> {
        let ctx = self.get_context();
        BindingResource::new(
            binding.clone(),
            ctx.memory_management_gpu
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource"),
        )
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_management_gpu.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.ctx.memory_management_gpu.cleanup(true);
    }

    fn allocation_mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
        self.ctx.memory_management_gpu.mode(mode);
    }
}

impl DataTransferService for CudaServer {
    fn register_src(&mut self, id: DataTransferId, src: CopyDescriptor<'_>) {
        let src_ctx = self.get_context();

        let src_resource = src_ctx
            .memory_management_gpu
            .get_resource(
                src.binding.memory,
                src.binding.offset_start,
                src.binding.offset_end,
            )
            .ok_or(IoError::InvalidHandle)
            .unwrap();

        let client = DataTransferRuntime::client();

        let handle = DataTransferItem {
            context: self.ctx.context,
            stream: self.ctx.stream,
            resource: src_resource,
        };
        let fence = Fence::new(self.ctx.stream);

        client.register_src(id, handle, fence);
    }

    fn register_dest(&mut self, id: DataTransferId, dst: CopyDescriptor<'_>) {
        let dst_ctx = self.get_context();
        let dst_resource = dst_ctx
            .memory_management_gpu
            .get_resource(
                dst.binding.memory,
                dst.binding.offset_start,
                dst.binding.offset_end,
            )
            .ok_or(IoError::InvalidHandle)
            .unwrap();

        let client = DataTransferRuntime::client();

        let call = DataTransferItem {
            context: self.ctx.context,
            stream: self.ctx.stream,
            resource: dst_resource,
        };

        client.register_dest(id, call);
    }
}

fn find_resource(ctx: &mut CudaContext, binding: server::Binding) -> GpuResource {
    ctx.memory_management_gpu
        .get_resource(binding.memory, binding.offset_start, binding.offset_end)
        .expect("Failed to find resource")
}

impl CudaContext {
    pub fn new(
        memory_management_gpu: MemoryManagement<GpuStorage>,
        memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
        compilation_options: CompilationOptions,
        stream: cudarc::driver::sys::CUstream,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            context,
            memory_management_gpu,
            memory_management_cpu,
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
            timestamps: TimestampProfiler::default(),
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
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
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
        if fast_math {
            options.push("--use_fast_math");
        }
        if cccl_include_path.exists() {
            options.push(&cccl_include_option);
        }

        #[cfg(feature = "compilation-cache")]
        let cluster_dim = compute_kernel.cluster_dim;

        logger.log_compilation(&kernel_compiled);

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

        let repr: cubecl_cpp::ComputeKernel<cubecl_cpp::cuda::CudaDialect<WmmaCompiler>> =
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
        ptx: Vec<c_char>,
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
        resources: &[GpuResource],
        scalars: &[*mut c_void],
    ) -> Result<(), String> {
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
            .map_err(|err| format!("{err:?}"))?;
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
            .map_err(|err| format!("{err:?}"))?;
        };

        Ok(())
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(mem_alignment: usize, ctx: CudaContext) -> Self {
        Self { mem_alignment, ctx }
    }

    fn get_context(&mut self) -> &mut CudaContext {
        unsafe {
            cudarc::driver::result::ctx::set_current(self.ctx.context).unwrap();
        };
        &mut self.ctx
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

fn cccl_include_path() -> PathBuf {
    let mut path = include_path();
    path.push("cccl");
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

fn elem_to_tensor_map_type(ty: StorageType) -> CUtensorMapDataType {
    use cudarc::driver::sys::CUtensorMapDataType::*;
    match ty {
        // packed fp4 should be treated as single 4-bit values to simplify indexing/shape handling
        // So a tile of width 16 with fp4 elements is 8 x fp4x2 elements wide.
        #[cfg(feature = "cuda-12080")]
        StorageType::Packed(ty, 2) if ty.size_bits() == 4 => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        StorageType::Scalar(ElemType::Float(kind)) => match kind {
            // There's no special handling for FP8, so load as u8. `0u8 == 0.0` when reinterpreting.
            FloatKind::E2M1 // single fp4s are padded to a full byte
            | FloatKind::E4M3
            | FloatKind::E5M2
            | FloatKind::UE8M0
            | FloatKind::E2M3
            | FloatKind::E3M2 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            FloatKind::F16 => CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            FloatKind::BF16 => CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            FloatKind::Flex32 | FloatKind::F32 => CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            FloatKind::TF32 => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            FloatKind::F64 => CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        },
        StorageType::Scalar(ElemType::Int(kind)) => match kind {
            // UInt is fine because zero bits and size is the same between both
            IntKind::I8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            IntKind::I16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            IntKind::I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            IntKind::I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
        },
        StorageType::Scalar(ElemType::UInt(kind)) => match kind {
            UIntKind::U8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            UIntKind::U16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            UIntKind::U32 => CU_TENSOR_MAP_DATA_TYPE_UINT32,
            UIntKind::U64 => CU_TENSOR_MAP_DATA_TYPE_UINT64,
        },
        _ => unimplemented!("Not supported for tensor map type"),
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

pub fn valid_strides(shape: &[usize], strides: &[usize]) -> bool {
    let rank = shape.len();
    if strides[rank - 1] != 1 {
        return false;
    }
    if rank <= 1 {
        return true;
    }

    let mut sorted = strides.to_vec();
    sorted.sort();
    sorted.reverse();

    if sorted != strides {
        return false;
    }

    for i in 0..rank - 2 {
        if strides[i] != shape[i + 1] * strides[i + 1] {
            return false;
        }
    }
    true
}
