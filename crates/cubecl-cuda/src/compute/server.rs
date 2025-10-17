use super::storage::gpu::{GpuResource, GpuStorage};
use crate::CudaCompiler;
use crate::compute::command::{Command, write_to_cpu};
use crate::compute::context::CudaContext;
use crate::compute::stream::CudaStreamBackend;
use crate::compute::sync::Fence;
use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::ir::{ElemType, IntKind, UIntKind};
use cubecl_core::server::{Binding, ServerCommunication, ServerUtilities};
use cubecl_core::{MemoryConfiguration, prelude::*};
use cubecl_core::{compute::CubeTask, server::IoError};
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
use cubecl_runtime::config::GlobalConfig;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{MemoryAllocationMode, offset_handles};
use cubecl_runtime::memory_management::{MemoryDeviceProperties, MemoryUsage};
use cubecl_runtime::server::{self, ComputeServer};
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::stream::MultiStream;
use cudarc::driver::sys::{CUcontext, CUresult, CUtensorMapInterleave, cuCtxEnablePeerAccess};
use cudarc::driver::sys::{
    CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapL2promotion, CUtensorMapSwizzle,
    cuTensorMapEncodeIm2col, cuTensorMapEncodeTiled,
};
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) const MB: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    streams: MultiStream<CudaStreamBackend>,
    peer_activated: bool,
    mem_alignment: usize,
    utilities: Arc<ServerUtilities<Self>>,
}

unsafe impl Send for CudaServer {}

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn CubeTask<CudaCompiler>>;
    type Storage = GpuStorage;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut command = self.command(stream_id, descriptors.iter().map(|d| &d.binding));

        Box::pin(command.read_async(descriptors))
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

        let mem_alignment = self.mem_alignment;
        let mut command = self.command_no_inputs(stream_id);

        let handle = command.reserve(total_size as u64)?;
        let handles = offset_handles(handle, &sizes, mem_alignment);

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
        let mut command = self.command(stream_id, descriptors.iter().map(|desc| &desc.0.binding));

        for (descriptor, data) in descriptors {
            command.write_to_gpu(descriptor, data)?;
        }

        Ok(())
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let grid_constants = self.ctx.compilation_options.grid_constants;
        let mut command = self.command(stream_id, bindings.buffers.iter());

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = future::block_on(command.read_async(vec![CopyDescriptor::new(
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

        let (scalars, scalar_bindings) = if grid_constants {
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
                    command
                        .create_with_data(bytemuck::cast_slice(dyn_meta))
                        .unwrap(),
                );
            }

            (scalars, handles)
        } else {
            let mut handles = Vec::new();
            if !bindings.metadata.data.is_empty() {
                handles.push(
                    command
                        .create_with_data(bytemuck::cast_slice(&bindings.metadata.data))
                        .unwrap(),
                )
            }
            handles.extend(
                bindings
                    .scalars
                    .values()
                    .map(|scalar| command.create_with_data(scalar.data()).unwrap()),
            );
            (Vec::new(), handles)
        };

        let mut resources = bindings
            .tensor_maps
            .iter()
            .map(|it| it.binding.clone())
            .chain(bindings.buffers)
            .map(|binding| command.resource(binding).expect("Resource to exist."))
            .collect::<Vec<_>>();

        let tensor_maps: Vec<_> = bindings
            .tensor_maps
            .into_iter()
            .map(|TensorMapBinding { map, binding }| {
                let resource = command
                    .resource(binding)
                    .expect("Tensor map resource exists.");
                let device_ptr = resource.ptr as *mut c_void;
                debug_assert!(
                    (device_ptr as usize).is_multiple_of(16),
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

        resources.extend(
            scalar_bindings
                .into_iter()
                .map(|s| command.resource(s.binding()).expect("Resource to exist")),
        );

        command.kernel(
            kernel_id,
            kernel,
            mode,
            count,
            &tensor_maps,
            &resources,
            &scalars,
            logger,
        )
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn sync(&mut self, stream_id: StreamId) -> DynFut<()> {
        let mut command = self.command_no_inputs(stream_id);
        command.sync()
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
        let mut command = self.command(stream_id, [&binding].into_iter());

        BindingResource::new(
            binding.clone(),
            command.resource(binding).expect("Failed to find resource"),
        )
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> MemoryUsage {
        let mut command = self.command_no_inputs(stream_id);
        command.memory_usage()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.allocation_mode(mode)
    }
}

impl ServerCommunication for CudaServer {
    const SERVER_COMM_ENABLED: bool = true;

    fn copy(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        if server_src.peer_activated {
            Self::change_server_peer(server_src, server_dst, src, stream_id_src, stream_id_dst)
        } else {
            Self::change_server_serialized(
                server_src,
                server_dst,
                src,
                stream_id_src,
                stream_id_dst,
            )
        }
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(
        ctx: CudaContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        device_id: i32,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = GlobalConfig::get();
        let max_streams = config.streaming.max_streams;

        unsafe {
            cudarc::driver::result::ctx::set_current(ctx.context).unwrap();
        };

        let peer_activated = enable_one_way_peer_access(ctx.context).is_ok();
        if peer_activated {
            log::info!("Peer data transfer activated for device {device_id}");
        } else {
            log::info!("Peer data transfer not available for device {device_id}");
        }

        Self {
            mem_alignment,
            ctx,
            peer_activated,
            streams: MultiStream::new(
                utilities.logger.clone(),
                CudaStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    utilities.logger.clone(),
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
        }
    }

    fn change_server_peer(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        let strides = src.strides.to_vec();
        let binding = src.binding.clone();

        let context_src = server_src.ctx.context;
        let context_dst = server_dst.ctx.context;

        // We create a command on the source server to retrieve the correct resource from the
        // source memory pools. We also make sure the current stream is aligned with the stream of
        // the binding, where the data was first allocated.
        let mut command_src = server_src.command(stream_id_src, [&src.binding].into_iter());
        let resource_src = command_src.resource(binding.clone())?;
        let stream_src = command_src.streams.current().sys;
        let fence_src = Fence::new(stream_src);

        // We need to free the command before creating another one.
        core::mem::drop(command_src);

        // We create a new command on the destination server to reserve the necessary GPU memory
        // and wait on the source server, making sure the execution is updated. Then, we perform
        // the peer memcpy on the destination server.
        let mut command_dst = server_dst.command_no_inputs(stream_id_dst);
        let stream_dst = command_dst.streams.current().sys;

        let handle = command_dst.reserve(binding.size())?;
        let resource_dst = command_dst.resource(handle.clone().binding())?;
        fence_src.wait_async(stream_dst);

        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                resource_dst.ptr,
                context_dst,
                resource_src.ptr,
                context_src,
                binding.size() as usize,
                stream_dst,
            )
            .result()
            .expect("Peer communication should be activated");
        }

        // We drop the last command.
        core::mem::drop(command_dst);

        Ok(Allocation { handle, strides })
    }

    fn change_server_serialized(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        let shape = src.shape.to_vec();
        let strides = src.strides.to_vec();
        let elem_size = src.elem_size;
        let binding = src.binding.clone();
        let num_bytes = shape.iter().product::<usize>() * elem_size;

        // We start by creating a command on the destination server.
        //
        // Here we allocate the necessary bytes using pinned memory managed by the destination
        // server along a new GPU handle. This way, the bytes could be reused later by that server,
        // and the lifetime of that handle is aligned with the execution order of the destination server,
        // removing the need to keep the bytes handle alive using synchronization, which would be the
        // case if we allocated the bytes using the source server.
        let mut command_dst = server_dst.command_no_inputs(stream_id_dst);
        let handle = command_dst.reserve(binding.size())?;
        let mut bytes = command_dst.reserve_cpu(num_bytes, true, None);
        let copy_desc = handle.copy_descriptor(&shape, &strides, elem_size);

        // We need to free the command before creating another one.
        core::mem::drop(command_dst);

        // We create a command on the source server to retrieve the correct resource from the
        // source memory pools. We also make sure the current stream is aligned with the stream of
        // the binding, where the data was first allocated.
        //
        // We use the source stream to copy the data from the source server into the allocated
        // bytes. This ensures that the source binding follows the correct execution order, meaning
        // that we don't have to keep the source handle alive using synchronization, which would be
        // the case if we performed the copy on the destination server.
        let mut command_src = server_src.command(stream_id_src, [&src.binding].into_iter());
        let resource_src = command_src.resource(binding.clone())?;
        let stream_src = command_src.streams.current().sys;

        unsafe {
            write_to_cpu(
                &shape,
                &strides,
                elem_size,
                &mut bytes,
                resource_src.ptr,
                stream_src,
            )?;
        }
        let fence_src = Fence::new(stream_src);

        // We need to free the command before creating another one.
        core::mem::drop(command_src);

        // Finally, we recreate a new command on the destination server to write the data stored in
        // pinned memory into the destination server. Here we need to wait for the initial copy
        // made by the source server using an event. The synchronization is done lazily on the
        // destination stream, which is very efficient.
        let mut command_dst = server_dst.command_no_inputs(stream_id_dst);
        let stream_dst = command_dst.streams.current().sys;

        fence_src.wait_async(stream_dst);
        command_dst.write_to_gpu(copy_desc, &bytes)?;

        // We drop the last command.
        core::mem::drop(command_dst);

        Ok(Allocation { handle, strides })
    }

    fn command_no_inputs(&mut self, stream_id: StreamId) -> Command<'_> {
        self.command(stream_id, [].into_iter())
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> Command<'_> {
        unsafe {
            cudarc::driver::result::ctx::set_current(self.ctx.context).unwrap();
        };
        let streams = self.streams.resolve(stream_id, bindings);

        Command::new(&mut self.ctx, streams)
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

use cudarc::driver::sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;

fn enable_one_way_peer_access(ctx_src: CUcontext) -> Result<(), CUresult> {
    unsafe {
        match cuCtxEnablePeerAccess(ctx_src, 0) {
            CUDA_SUCCESS | CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Ok(()),
            err => Err(err),
        }
    }
}
