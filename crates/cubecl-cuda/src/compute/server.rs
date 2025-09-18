use super::storage::gpu::{GpuResource, GpuStorage};
use super::sync::Fence;
use crate::CudaCompiler;
use crate::compute::stream::CudaStreamBackend;
use crate::compute::{
    DataTransferItem, DataTransferRuntime, context::CudaContext, io::register_copies_to_bytes,
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::ir::{ElemType, IntKind, UIntKind};
use cubecl_core::server::Binding;
use cubecl_core::{MemoryConfiguration, prelude::*};
use cubecl_core::{
    compute::CubeTask,
    server::{DataTransferService, IoError},
};
use cubecl_core::{
    future::{self, DynFut},
    server::AllocationKind,
};
use cubecl_core::{
    ir::FloatKind,
    server::{Bindings, CopyDescriptor, TensorMapBinding},
};
use cubecl_core::{
    ir::StorageType,
    server::{Allocation, AllocationDescriptor, ProfileError, ProfilingToken},
};
use cubecl_runtime::data_service::DataTransferId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::offset_handles;
use cubecl_runtime::memory_management::{MemoryDeviceProperties, MemoryUsage};
use cubecl_runtime::server::{self, ComputeServer};
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::stream::{MultiStream, ResolvedStreams};
use cudarc::driver::sys::CUtensorMapInterleave;
use cudarc::driver::sys::{
    CUDA_MEMCPY2D_st, CUmemorytype, CUtensorMapDataType, CUtensorMapFloatOOBfill,
    CUtensorMapL2promotion, CUtensorMapSwizzle, cuMemcpy2DAsync_v2, cuTensorMapEncodeIm2col,
    cuTensorMapEncodeTiled,
};
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_void};
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) const MB: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    streams: MultiStream<CudaStreamBackend>,
    mem_alignment: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    cluster_dim: Option<(u32, u32, u32)>,
    ptx: Vec<c_char>,
}

unsafe impl Send for CudaServer {}

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn CubeTask<CudaCompiler>>;
    type Storage = GpuStorage;
    type Info = ();

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        Box::pin(self.read_async(descriptors, stream_id))
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

        let (_ctx, mut streams) = self.resolve_context_basic(stream_id);

        let handle = streams
            .get(&stream_id)
            .memory_management_gpu
            .reserve(total_size as u64)?;
        let mem_handle = server::Handle::new(
            handle,
            None,
            None,
            stream_id,
            streams.cursor,
            total_size as u64,
        );

        let handles = offset_handles(mem_handle, &sizes, self.mem_alignment);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let (_ctx, mut streams) = self
            .resolve_context_bindings(stream_id, descriptors.iter().map(|desc| &desc.0.binding));

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

            let resource = streams
                .get(&binding.stream)
                .memory_management_gpu
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .ok_or(IoError::InvalidHandle)?;

            let current = streams.get(&stream_id);

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
                    cuMemcpy2DAsync_v2(&cpy, current.sys).result().unwrap();
                }
            } else {
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(resource.ptr, data, current.sys)
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
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = future::block_on(
                    self.read_async(vec![CopyDescriptor::new(binding, &[3], &[1], 4)], stream_id),
                )
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

        let (ctx, mut streams) = self.resolve_context_bindings(stream_id, bindings.buffers.iter());

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, mode, logger);
        }

        let tensor_maps: Vec<_> = bindings
            .tensor_maps
            .into_iter()
            .map(|TensorMapBinding { map, binding }| {
                let resource = streams
                    .get(&binding.stream)
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
            .map(|binding| find_resource(&mut streams, binding))
            .collect::<Vec<_>>();
        resources.extend(
            scalar_bindings
                .into_iter()
                .map(|s| find_resource(&mut streams, s.binding())),
        );

        let current = streams.get(&stream_id);
        let result = ctx.execute_task(
            current,
            kernel_id,
            count,
            &tensor_maps,
            &resources,
            &scalars,
        );

        match result {
            Ok(_) => {}
            Err(err) => match ctx.timestamps.is_empty() {
                true => panic!("{err:?}"),
                false => ctx.timestamps.error(ProfileError::Unknown(err)),
            },
        }
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn sync(&mut self, stream_id: StreamId) -> DynFut<()> {
        let (_, mut streams) = self.resolve_context_basic(stream_id);
        let fence = streams.get(&stream_id).fence();

        Box::pin(async {
            fence.wait_sync();
        })
    }

    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken {
        cubecl_common::future::block_on(self.sync(stream_id));
        self.ctx.timestamps.start()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        cubecl_common::future::block_on(self.sync(stream_id));
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: server::Binding,
        stream_id: StreamId,
    ) -> BindingResource<GpuResource> {
        let (_, mut streams) = self.resolve_context_bindings(stream_id, [&binding].into_iter());

        BindingResource::new(
            binding.clone(),
            streams
                .get(&binding.stream)
                .memory_management_gpu
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource"),
        )
    }

    fn memory_usage(&self) -> MemoryUsage {
        todo!()
        // self.ctx.memory_management_gpu.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        todo!()
        // self.ctx.memory_management_gpu.cleanup(true);
    }

    fn allocation_mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
        todo!()
        // self.ctx.memory_management_gpu.mode(mode);
    }
}

impl DataTransferService for CudaServer {
    fn register_src(&mut self, stream_id: StreamId, id: DataTransferId, src: CopyDescriptor<'_>) {
        let (src_ctx, mut streams) =
            self.resolve_context_bindings(stream_id, [&src.binding].into_iter());

        let src_resource = streams
            .get(&src.binding.stream)
            .memory_management_gpu
            .get_resource(
                src.binding.memory,
                src.binding.offset_start,
                src.binding.offset_end,
            )
            .ok_or(IoError::InvalidHandle)
            .unwrap();

        let client = DataTransferRuntime::client();
        let current = streams.get(&stream_id);

        let handle = DataTransferItem {
            stream: current.sys,
            context: src_ctx.context,
            resource: src_resource,
        };
        let fence = Fence::new(current.sys);

        client.register_src(id, handle, fence);
    }

    fn register_dest(&mut self, stream_id: StreamId, id: DataTransferId, dst: CopyDescriptor<'_>) {
        let (dst_ctx, mut streams) =
            self.resolve_context_bindings(stream_id, [&dst.binding].into_iter());
        let dst_resource = streams
            .get(&dst.binding.stream)
            .memory_management_gpu
            .get_resource(
                dst.binding.memory,
                dst.binding.offset_start,
                dst.binding.offset_end,
            )
            .ok_or(IoError::InvalidHandle)
            .unwrap();

        let current = streams.get(&stream_id);
        let client = DataTransferRuntime::client();

        let call = DataTransferItem {
            context: dst_ctx.context,
            stream: current.sys,
            resource: dst_resource,
        };

        client.register_dest(id, call);
    }
}

fn find_resource(
    streams: &mut ResolvedStreams<CudaStreamBackend>,
    binding: server::Binding,
) -> GpuResource {
    streams
        .get(&binding.stream)
        .memory_management_gpu
        .get_resource(binding.memory, binding.offset_start, binding.offset_end)
        .expect("Failed to find resource")
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(
        ctx: CudaContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
    ) -> Self {
        Self {
            mem_alignment,
            ctx,
            streams: MultiStream::new(CudaStreamBackend::new(mem_props, mem_config, mem_alignment)),
        }
    }

    fn resolve_context_basic(
        &mut self,
        stream_id: StreamId,
    ) -> (&mut CudaContext, ResolvedStreams<'_, CudaStreamBackend>) {
        self.resolve_context_bindings(stream_id, [].into_iter())
    }

    fn resolve_context_bindings<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> (&mut CudaContext, ResolvedStreams<'_, CudaStreamBackend>) {
        unsafe {
            cudarc::driver::result::ctx::set_current(self.ctx.context).unwrap();
        };
        let streams = self.streams.resolve(stream_id, bindings);

        (&mut self.ctx, streams)
    }

    fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        let (_ctx, mut streams) =
            self.resolve_context_bindings(stream_id, descriptors.iter().map(|desc| &desc.binding));

        let result = register_copies_to_bytes(stream_id, &mut streams, descriptors);
        let fence = streams.get(&stream_id).fence();

        async move {
            fence.wait_sync();
            result
        }
    }
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
