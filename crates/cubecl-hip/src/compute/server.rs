use super::storage::gpu::GpuStorage;
use super::{storage::gpu::GpuResource, uninit_vec};
use crate::compute::context::HipContext;
use crate::compute::gpu::GpuStorageContext;
use crate::compute::io::register_copies_to_bytes;
use crate::compute::stream::{HipStreamBackend, Stream};
use crate::runtime::HipCompiler;
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_core::compute::CubeTask;
use cubecl_core::prelude::*;
use cubecl_core::server::{
    Allocation, AllocationKind, CopyDescriptor, DataTransferService, IoError, ProfileError,
    ProfilingToken,
};
use cubecl_core::server::{Binding, Bindings};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::memory_management::offset_handles;
use cubecl_runtime::server::{self, ComputeServer};
use cubecl_runtime::storage::{BindingResource, ComputeStorage};
use cubecl_runtime::stream::MultiStream;
use std::future::Future;
use std::sync::Arc;

#[cfg(feature = "compilation-cache")]
use cubecl_common::cache::{Cache, CacheOption};

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    streams: MultiStream<HipStreamBackend>,
    mem_alignment: usize,
}

unsafe impl Send for HipServer {}
impl DataTransferService for HipServer {}

impl ComputeServer for HipServer {
    type Kernel = Box<dyn CubeTask<HipCompiler>>;
    type Storage = GpuStorage;
    type Info = ();

    fn create(
        &mut self,
        descriptors: Vec<server::AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<server::Allocation>, IoError> {
        let mut total_size = 0;
        let mut strides = Vec::new();
        let mut sizes = Vec::new();

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

        let (ctx, _stream, cursor) = self.resolve_context_basic(stream_id);

        let handle = ctx.memory_management_gpu.reserve(total_size as u64)?;
        let mem_handle =
            server::Handle::new(handle, None, None, stream_id, cursor, total_size as u64);
        let handles = offset_handles(mem_handle, &sizes, self.mem_alignment);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn read(
        &mut self,
        descriptors: Vec<server::CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        Box::pin(self.read_async(descriptors, stream_id))
    }

    fn write(
        &mut self,
        descriptors: Vec<(server::CopyDescriptor<'_>, &[u8])>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let (ctx, stream, _cursor) = self
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

            if rank > 1 {
                let stride = strides[rank - 2];

                ctx.copy_to_binding_2d(stream, binding, data, shape, stride, elem_size);
            } else {
                ctx.copy_to_binding(stream, binding, data);
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_management_gpu.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.ctx.memory_management_gpu.cleanup(true);
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
            // TODO: HIP doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = self.read_sync(binding, stream_id);
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
        let info = self
            .create_with_data(bytemuck::cast_slice(&metadata.data), stream_id)
            .unwrap();
        let scalars: Vec<_> = scalars
            .values()
            .map(|s| self.create_with_data(s.data(), stream_id).unwrap())
            .collect();

        let (ctx, stream, _cursor) = self.resolve_context_bindings(stream_id, buffers.iter());

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, mode, logger);
        }

        let mut resources: Vec<_> = buffers.into_iter().map(|b| find_resource(ctx, b)).collect();
        resources.push(find_resource(ctx, info.clone().binding()));
        resources.extend(scalars.into_iter().map(|s| find_resource(ctx, s.binding())));

        ctx.execute_task(stream, kernel_id, count, resources);
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn sync(&mut self, stream_id: StreamId) -> DynFut<()> {
        let (_, stream, _) = self.resolve_context_basic(stream_id);
        let fence = stream.fence();

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

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<GpuResource> {
        BindingResource::new(
            binding.clone(),
            self.ctx
                .memory_management_gpu
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Can't find resource"),
        )
    }

    fn allocation_mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
        self.ctx.memory_management_gpu.mode(mode);
    }
}

fn find_resource(ctx: &mut HipContext, binding: server::Binding) -> GpuResource {
    ctx.memory_management_gpu
        .get_resource(binding.memory, binding.offset_start, binding.offset_end)
        .expect("Failed to find resource")
}

impl HipServer {
    /// Create a new hip server.
    pub(crate) fn new(mem_alignment: usize, ctx: HipContext) -> Self {
        Self {
            ctx,
            mem_alignment,
            streams: MultiStream::new(),
        }
    }

    fn resolve_context_basic(
        &mut self,
        stream_id: StreamId,
    ) -> (&mut HipContext, &mut Stream, u64) {
        let stream = self.streams.get(stream_id);

        self.ctx
            .memory_management_gpu
            .storage()
            .context(GpuStorageContext {
                stream: stream.stream.sys.clone(),
            });

        (&mut self.ctx, &mut stream.stream, stream.cursor)
    }

    fn resolve_context_bindings<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> (&mut HipContext, &mut Stream, u64) {
        let stream = self.streams.resolve(stream_id, bindings);

        self.ctx
            .memory_management_gpu
            .storage()
            .context(GpuStorageContext {
                stream: stream.stream.sys.clone(),
            });

        (&mut self.ctx, &mut stream.stream, stream.cursor)
    }

    fn read_sync(&mut self, binding: server::Binding, stream_id: StreamId) -> Vec<u8> {
        let (ctx, stream, _cursor) =
            self.resolve_context_bindings(stream_id, [&binding].into_iter());

        let resource = ctx
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .expect("Failed to find resource");

        let mut data = uninit_vec(resource.size as usize);
        unsafe {
            let status = cubecl_hip_sys::hipMemcpyDtoHAsync(
                data.as_mut_ptr() as *mut _,
                resource.ptr,
                resource.size as usize,
                stream.sys,
            );
            assert_eq!(status, HIP_SUCCESS, "Should copy data from device to host");
        };
        stream.sync();
        data
    }

    fn read_async(
        &mut self,
        descriptors: Vec<server::CopyDescriptor>,
        stream_id: StreamId,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        let (ctx, stream, _cursor) =
            self.resolve_context_bindings(stream_id, descriptors.iter().map(|desc| &desc.binding));

        let result = register_copies_to_bytes(ctx, stream, descriptors);
        let fence = stream.fence();

        async move {
            fence.wait_sync();
            result
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

#[derive(Debug)]
pub(crate) enum LaunchError {
    OutOfMemory,
    Unknown(String),
}

impl From<LaunchError> for ProfileError {
    fn from(val: LaunchError) -> Self {
        match val {
            LaunchError::OutOfMemory => ProfileError::Unknown("Out of memory".into()),
            LaunchError::Unknown(msg) => ProfileError::Unknown(msg),
        }
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
