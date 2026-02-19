use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    compute::{
        command::{Command, write_to_cpu},
        context::HipContext,
        fence::Fence,
        stream::HipStreamBackend,
    },
    runtime::HipCompiler,
};
use cubecl_common::{bytes::Bytes, future::DynFut, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    MemoryConfiguration, future,
    ir::MemoryDeviceProperties,
    prelude::*,
    server::{
        Allocation, AllocationKind, Binding, Bindings, CopyDescriptor, ExecutionError, IoError,
        LaunchError, ProfileError, ProfilingToken, ServerCommunication, ServerUtilities,
    },
    zspace::{Shape, Strides, strides},
};
use cubecl_runtime::{
    compiler::CubeTask,
    config::GlobalConfig,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryUsage, create_buffers, optimal_align},
    server::{self, ComputeServer},
    storage::BindingResource,
    stream::MultiStream,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    streams: MultiStream<HipStreamBackend>,
    mem_alignment: usize,
    utilities: Arc<ServerUtilities<Self>>,
}

unsafe impl Send for HipServer {}

impl ComputeServer for HipServer {
    type Kernel = Box<dyn CubeTask<HipCompiler>>;
    type Storage = GpuStorage;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, IoError> {
        let mut command = self.command_no_inputs(stream_id);

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, true, None))
            .collect())
    }

    fn create(
        &mut self,
        descriptors: Vec<server::AllocationDescriptor>,
        stream_id: StreamId,
    ) -> Result<Vec<server::Allocation>, IoError> {
        let mut total_size = 0;
        let mut strides = Vec::new();
        let mut sizes = Vec::new();

        for descriptor in descriptors {
            let last_dim = descriptor.shape.last().copied().unwrap_or(1);
            let pitch_align = match descriptor.kind {
                AllocationKind::Contiguous => 1,
                AllocationKind::Optimized => {
                    optimal_align(last_dim, descriptor.elem_size, self.mem_alignment)
                }
            };

            let rank = descriptor.shape.len();
            let width = *descriptor.shape.last().unwrap_or(&1);
            let height: usize = descriptor.shape.iter().rev().skip(1).product();
            let height = Ord::max(height, 1);
            let width_bytes = width * descriptor.elem_size;
            let pitch = width_bytes.next_multiple_of(pitch_align);
            let size = height * pitch;
            total_size += size.next_multiple_of(self.mem_alignment);

            let mut stride = strides![1; rank];
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
        let handles = create_buffers(handle, &sizes, mem_alignment);

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
        let mut command = self.command(stream_id, descriptors.iter().map(|d| &d.handle));

        Box::pin(command.read_async(descriptors))
    }

    fn write(
        &mut self,
        descriptors: Vec<(server::CopyDescriptor, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let mut command = self.command(stream_id, descriptors.iter().map(|desc| &desc.0.handle));

        let mut to_drop = Vec::with_capacity(descriptors.len());

        for (descriptor, data) in descriptors {
            command.write_to_gpu(descriptor, &data)?;
            to_drop.push(data);
        }

        command.gc(to_drop);

        Ok(())
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> MemoryUsage {
        let mut command = self.command_no_inputs(stream_id);
        command.memory_usage()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.memory_cleanup()
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let mut command = self.command(stream_id, bindings.buffers.iter());

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: HIP doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = future::block_on(command.read_async(vec![CopyDescriptor::new(
                    binding,
                    [3].into(),
                    [1].into(),
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

        let Bindings {
            buffers,
            metadata,
            scalars,
            tensor_maps,
        } = bindings;

        debug_assert!(tensor_maps.is_empty(), "Can't use tensor maps on HIP");

        let info = command
            .create_with_data(bytemuck::cast_slice(&metadata.data))
            .unwrap();
        let scalars: Vec<_> = scalars
            .values()
            .map(|s| command.create_with_data(s.data()).unwrap())
            .collect();

        let mut resources: Vec<_> = buffers
            .into_iter()
            .map(|b| command.resource(b).expect("Resource to exist."))
            .collect();
        resources.push(
            command
                .resource(info.clone().binding())
                .expect("Resource to exist."),
        );
        resources.extend(
            scalars
                .into_iter()
                .map(|s| command.resource(s.binding()).expect("Resource to exist.")),
        );

        command.kernel(kernel_id, kernel, mode, count, &resources, logger)?;

        Ok(())
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>> {
        let mut command = self.command_no_inputs(stream_id);
        command.sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            self.ctx.timestamps.error(err.into())
        }

        self.ctx.timestamps.start()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            self.ctx.timestamps.error(err.into())
        }
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

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.allocation_mode(mode)
    }
}

impl ServerCommunication for HipServer {
    const SERVER_COMM_ENABLED: bool = true;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(server_src, server_dst, src))
    )]
    fn copy(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        Self::change_server_serialized(server_src, server_dst, src, stream_id_src, stream_id_dst)
    }
}

impl HipServer {
    /// Create a new hip server.
    pub(crate) fn new(
        ctx: HipContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = GlobalConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            ctx,
            mem_alignment,
            streams: MultiStream::new(
                utilities.logger.clone(),
                HipStreamBackend::new(
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

    fn command_no_inputs(&mut self, stream_id: StreamId) -> Command<'_> {
        self.command(stream_id, [].into_iter())
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> Command<'_> {
        let streams = self.streams.resolve(stream_id, bindings);

        Command::new(&mut self.ctx, streams)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(server_src, server_dst, src))
    )]
    fn change_server_serialized(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        let shape: Shape = src.shape.into();
        let strides: Strides = src.strides.into();
        let elem_size = src.elem_size;
        let binding = src.handle.clone();
        let num_bytes = shape.iter().product::<usize>() * elem_size;

        // We start by creating a command on the destination server.
        //
        // Here we allocate the necessary bytes using pinned memory managed by the destination
        // server along a new GPU handle. This way, the bytes could be reused later by that server,
        // and the lifetime of that handle is aligned with the execution order of the destination server,
        // removing the need to keep the bytes handle alive using synchronization, which would be the
        // case if we allocated the bytes using the source server.
        let mut command_dst = server_dst.command_no_inputs(stream_id_dst);
        let handle = command_dst.reserve(binding.size_in_used())?;
        let mut bytes = command_dst.reserve_cpu(num_bytes, true, None);
        let copy_desc = handle.copy_descriptor(shape, strides, elem_size);

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
        let mut command_src = server_src.command(stream_id_src, [&src.handle].into_iter());
        let resource_src = command_src.resource(binding.clone())?;
        let stream_src = command_src.streams.current().sys;

        unsafe {
            write_to_cpu(
                &copy_desc.shape,
                &copy_desc.strides,
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
        let strides = copy_desc.strides.clone();
        command_dst.write_to_gpu(copy_desc, &bytes)?;
        command_dst.gc(bytes);

        // We drop the last command.
        core::mem::drop(command_dst);

        Ok(Allocation { handle, strides })
    }
}
