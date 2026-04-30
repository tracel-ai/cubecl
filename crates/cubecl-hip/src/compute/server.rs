use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    compute::{command::Command, context::HipContext, fence::Fence, stream::HipStreamBackend},
    runtime::HipCompiler,
};
use cubecl_common::{bytes::Bytes, future::DynFut, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    future,
    ir::MemoryDeviceProperties,
    prelude::*,
    server::{
        Binding, CopyDescriptor, KernelArguments, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities, StreamErrorMode,
    },
};
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::MultiStream,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    streams: MultiStream<HipStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
}

// SAFETY: `HipServer` is only accessed from one thread at a time via the `DeviceHandle`
// (which serializes access through either a mutex or a dedicated runner thread depending
// on the selected channel feature). The HIP context and streams it manages are never
// shared across threads without synchronization.
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
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, true, None))
            .collect())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err}"),
        };

        let reserved = command.reserve(size).unwrap();
        command.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        match self.command(
            stream_id,
            descriptors.iter().map(|d| &d.handle),
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        ) {
            Ok(mut command) => Box::pin(command.read_async(descriptors)),
            Err(err) => Box::pin(async move { Err(err) }),
        }
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        let mut command = match self.command(
            stream_id,
            descriptors.iter().map(|desc| &desc.0.handle),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err}"),
        };

        for (descriptor, data) in descriptors {
            if let Err(err) = command.write_to_gpu(descriptor, data) {
                command.error(err.into());
                return;
            }
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id) {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err}"),
            };
            stream.current().errors.push(err);
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;

        let current = command.streams.current();
        current.drop_queue.flush(|| Fence::new(current.sys));
        current.memory_management_gpu.storage().flush();

        Ok(())
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        );

        match command {
            Ok(mut command) => command.sync(),
            Err(err) => Box::pin(async { Err(err) }),
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        cubecl_common::future::block_on(self.sync(stream_id))?;
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
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<GpuResource>, ServerError> {
        let mut command = self.command(
            stream_id,
            [&binding].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let memory = binding.memory.clone();
        let resource = command.resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        Ok(command.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            // Server is in error.
            Err(_) => return,
        };
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err}"),
        };
        command.allocation_mode(mode)
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
        is_integrated: bool,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            ctx,
            streams: MultiStream::new(
                utilities.logger.clone(),
                HipStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    is_integrated,
                    utilities.logger.clone(),
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
        }
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        if mode.flush {
            let errors = self.flush_errors(stream_id);

            if !mode.ignore && !errors.is_empty() {
                return Err(ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                });
            }
        }
        let streams = self.streams.resolve(stream_id, handles, !mode.ignore)?;

        Ok(Command::new(&mut self.ctx, streams))
    }

    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError> {
        let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
            Ok(stream) => stream,
            Err(_) => return Vec::new(),
        };
        let errors = core::mem::take(&mut stream.current().errors);

        // It is very important to tag current profiles as being wrong.
        if !errors.is_empty() {
            self.ctx.timestamps.error(ProfileError::Unknown {
                reason: alloc::format!("{errors:?}"),
                backtrace: BackTrace::capture(),
            });
            stream.current().memory_management_gpu.cleanup(false);
        }

        core::mem::drop(stream);
        errors
    }

    fn launch_checked(
        &mut self,
        kernel: Box<dyn CubeTask<HipCompiler>>,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let mut command = self.command(
            stream_id,
            bindings.buffers.iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

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

        let KernelArguments {
            buffers,
            info,
            tensor_maps,
        } = bindings;

        debug_assert!(tensor_maps.is_empty(), "Can't use tensor maps on HIP");

        let info = command
            .create_with_data(bytemuck::cast_slice(&info.data))
            .unwrap();

        let mut resources: Vec<_> = buffers
            .into_iter()
            .map(|b| command.resource(b).expect("Resource to exist."))
            .collect();

        resources.push(
            command
                .resource(info.binding())
                .expect("Resource to exist."),
        );

        command.kernel(kernel_id, kernel, mode, count, &resources, logger)?;

        Ok(())
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}
