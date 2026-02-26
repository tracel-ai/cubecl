use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    compute::{command::Command, context::HipContext, stream::HipStreamBackend},
    runtime::HipCompiler,
};
use cubecl_common::{bytes::Bytes, future::DynFut, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    MemoryConfiguration, future,
    ir::MemoryDeviceProperties,
    prelude::*,
    server::{
        Bindings, CopyDescriptor, HandleBinding, HandleId, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities,
    },
};
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::CubeTask,
    config::GlobalConfig,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    server::ComputeServer,
    storage::BindingResource,
    stream::MultiStream,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    streams: MultiStream<HipStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
}

unsafe impl Send for HipServer {}

impl ComputeServer for HipServer {
    type Kernel = Box<dyn CubeTask<HipCompiler>>;
    type Storage = GpuStorage;
    type MemoryLayoutPolicy = PitchedMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, ServerError> {
        let mut command = self.command_no_inputs(stream_id)?;

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, true, None))
            .collect())
    }

    fn bind(&mut self, handles: Vec<HandleBinding>, stream_id: StreamId) {
        let mut sizes = Vec::new();
        let mut total_size = 0;

        for handle in handles.iter() {
            let size = handle.size();
            total_size += size;
            sizes.push(size);
        }

        let mut command = match self.command_no_inputs(stream_id) {
            Ok(val) => val,
            // Server is in error.
            Err(_) => return,
        };

        let memory = command.reserve(total_size).unwrap();
        let slots = memory.partition(total_size, &handles, command.cursor(), stream_id);

        for (handle, slot) in handles.into_iter().zip(slots.into_iter()) {
            command.bind(handle, slot);
        }
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        match self.command(stream_id, descriptors.iter().map(|d| &d.handle)) {
            Ok(mut command) => Box::pin(command.read_async(descriptors)),
            Err(err) => Box::pin(async move { Err(err) }),
        }
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        let mut command =
            match self.command(stream_id, descriptors.iter().map(|desc| &desc.0.handle)) {
                Ok(val) => val,
                // Server is in error
                Err(_) => return,
            };

        for (descriptor, data) in descriptors {
            if let Err(err) = command.write_to_gpu(descriptor, &data) {
                command.error(err.into());
                return;
            }
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id) {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(_) => return,
            };
            stream.current().errors.push(err);
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let _command = self.command_no_inputs(stream_id)?;
        Ok(())
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let command = self.command_no_inputs(stream_id);

        match command {
            Ok(mut command) => command.sync(),
            Err(err) => Box::pin(async { Err(err) }),
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            log::warn!("{err}");
        }

        Ok(self.ctx.timestamps.start())
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            self.ctx
                .timestamps
                .error(ProfileError::Server(Box::new(err)));
        }
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        handle: HandleBinding,
        stream_id: StreamId,
    ) -> Result<BindingResource<GpuResource>, ServerError> {
        let mut command = self.command(stream_id, [&handle].into_iter())?;
        let (resource, handle) = command.resource(handle)?;

        Ok(BindingResource::new(handle, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let mut command = self.command_no_inputs(stream_id)?;
        Ok(command.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(stream_id) {
            Ok(val) => val,
            // Server is in error.
            Err(_) => return,
        };
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(stream_id) {
            Ok(val) => val,
            Err(_) => return,
        };
        command.allocation_mode(mode)
    }

    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError> {
        let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
            Ok(stream) => stream,
            Err(_) => return Vec::new(),
        };
        let errors = core::mem::take(&mut stream.current().errors);
        core::mem::drop(stream);
        self.memory_cleanup(stream_id);
        errors
    }

    fn free(&mut self, handle: HandleId, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(stream_id) {
            Ok(val) => val,
            // Server is in error.
            Err(_) => return,
        };
        command.free(handle);
    }
}

impl ServerCommunication for HipServer {
    const SERVER_COMM_ENABLED: bool = false;
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

    fn command_no_inputs(&mut self, stream_id: StreamId) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter())
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a HandleBinding>,
    ) -> Result<Command<'_>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles, true)?;

        Ok(Command::new(&mut self.ctx, streams))
    }

    fn launch_checked(
        &mut self,
        kernel: Box<dyn CubeTask<HipCompiler>>,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let mut command = self.command(stream_id, bindings.handles.iter())?;

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
            handles: buffers,
            metadata,
            scalars,
            tensor_maps,
        } = bindings;

        debug_assert!(tensor_maps.is_empty(), "Can't use tensor maps on HIP");

        let info = command
            .create_with_data(bytemuck::cast_slice(&metadata.data), true)
            .unwrap();

        let scalars: Vec<_> = scalars
            .values()
            .map(|s| command.create_with_data(s.data(), true).unwrap())
            .collect();

        let mut resources: Vec<_> = buffers
            .into_iter()
            .map(|b| command.resource(b).expect("Resource to exist.").0)
            .collect();

        resources.push({
            // Manual cleaning.
            command.resource(info).expect("Resource to exist.").0
        });
        resources.extend(
            scalars
                .into_iter()
                .map(|s| command.resource(s).expect("Resource to exist.").0),
        );

        command.kernel(kernel_id, kernel, mode, count, &resources, logger)?;

        Ok(())
    }
}
