use super::storage::gpu::GpuResource;
use super::storage::gpu::GpuStorage;
use crate::compute::command::Command;
use crate::compute::context::HipContext;
use crate::compute::stream::HipStreamBackend;
use crate::runtime::HipCompiler;
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_core::compute::CubeTask;
use cubecl_core::server::{
    Allocation, AllocationKind, CopyDescriptor, DataTransferService, IoError, ProfileError,
    ProfilingToken,
};
use cubecl_core::server::{Binding, Bindings};
use cubecl_core::{MemoryConfiguration, future, prelude::*};
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{MemoryAllocationMode, MemoryUsage};
use cubecl_runtime::memory_management::{MemoryDeviceProperties, offset_handles};
use cubecl_runtime::server::{self, ComputeServer};
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::stream::MultiStream;
use std::sync::Arc;

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

    fn read(
        &mut self,
        descriptors: Vec<server::CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut command = self.command(stream_id, descriptors.iter().map(|d| &d.binding));

        Box::pin(command.read_async(descriptors))
    }

    fn write(
        &mut self,
        descriptors: Vec<(server::CopyDescriptor<'_>, &[u8])>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let mut command = self.command(stream_id, descriptors.iter().map(|desc| &desc.0.binding));

        for (descriptor, data) in descriptors {
            command.write_to_gpu(descriptor, data)?;
        }

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
        let mut command = self.command(stream_id, bindings.buffers.iter());

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: HIP doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
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

        command.kernel(kernel_id, kernel, mode, count, &resources, logger)
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

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.allocation_mode(mode)
    }
}

impl HipServer {
    /// Create a new hip server.
    pub(crate) fn new(
        ctx: HipContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
    ) -> Self {
        Self {
            ctx,
            mem_alignment,
            streams: MultiStream::new(
                HipStreamBackend::new(mem_props, mem_config, mem_alignment),
                8,
            ),
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
