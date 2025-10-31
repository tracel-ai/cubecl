use std::sync::{Arc, mpsc::SyncSender};

use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    CubeCount, ExecutionMode, MemoryConfiguration, MemoryUsage,
    compute::CubeTask,
    future::DynFut,
    server::{
        Allocation, AllocationKind, Binding, Bindings, Handle, IoError, ProfileError,
        ProfilingToken, ServerUtilities,
    },
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryDeviceProperties, MemoryManagement, MemoryManagementOptions,
        offset_handles,
    },
    storage::{BindingResource, BytesResource, BytesStorage},
};

use crate::{
    CpuCompiler,
    compute::{
        alloc_controller::CpuAllocController,
        runner::Runner,
        server::{CpuContext, CpuServer, contiguous_strides},
    },
};

pub(crate) struct CpuStream {
    ctx: CpuContext,
    runner: Runner,
    utilities: Arc<ServerUtilities<CpuServer>>,
}

impl CpuStream {
    pub fn new(
        config: MemoryConfiguration,
        properties: &MemoryDeviceProperties,
        logger: Arc<ServerLogger>,
        utilities: Arc<ServerUtilities<CpuServer>>,
    ) -> Self {
        Self {
            runner: Runner::default(),
            ctx: CpuContext::new(MemoryManagement::from_configuration(
                BytesStorage::default(),
                properties,
                config,
                logger,
                MemoryManagementOptions::new("CPU"),
            )),
            utilities,
        }
    }

    pub fn start(mut self, stream_id: StreamId) -> SyncSender<CpuTask> {
        let (sender, rec) = std::sync::mpsc::sync_channel(10);
        std::thread::spawn(move || {
            loop {
                let task = match rec.recv() {
                    Ok(task) => task,
                    Err(err) => panic!("Error in the channel {err:?}"),
                };

                match task {
                    CpuTask::Compute {
                        kernel,
                        count,
                        bindings,
                        kind,
                    } => {
                        self.execute(kernel, count, bindings, kind);
                    }
                    CpuTask::Read {
                        descriptors,
                        callback,
                    } => {
                        let fut = self.read(descriptors, stream_id);
                        callback.send(fut).unwrap();
                    }
                    CpuTask::Write {
                        descriptors,
                        callback,
                    } => {
                        let fut = self.write(descriptors);
                        callback.send(fut).unwrap();
                    }
                    CpuTask::Create {
                        descriptors,
                        callback,
                    } => {
                        let fut = self.create(descriptors, stream_id);
                        callback.send(fut).unwrap();
                    }
                    CpuTask::Flush => {}
                    CpuTask::Sync { callback } => callback.send(()).unwrap(),
                    CpuTask::MemoryUsage { callback } => callback
                        .send(self.ctx.memory_management.memory_usage())
                        .unwrap(),
                    CpuTask::MemoryCleanup => {
                        self.ctx.memory_management.cleanup(true);
                    }
                    CpuTask::StartProfile { callback } => {
                        callback.send(self.start_profile(stream_id)).unwrap()
                    }
                    CpuTask::EndProfile { token, callback } => {
                        callback.send(self.end_profile(stream_id, token)).unwrap()
                    }
                    CpuTask::AllocMode { mode } => {
                        self.ctx.memory_management.mode(mode);
                    }
                    CpuTask::GetResource { binding, callback } => callback
                        .send(BindingResource::new(
                            binding.clone(),
                            self.ctx
                                .memory_management
                                .get_resource(
                                    binding.memory,
                                    binding.offset_start,
                                    binding.offset_end,
                                )
                                .expect("Can't find resource"),
                        ))
                        .unwrap(),
                }
            }
        });

        sender
    }

    fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptorOwned>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        fn inner(
            ctx: &mut CpuContext,
            descriptors: Vec<CopyDescriptorOwned>,
        ) -> Result<Vec<Bytes>, IoError> {
            let mut result = Vec::with_capacity(descriptors.len());
            for desc in descriptors {
                let len = desc.binding.size() as usize;
                let controller = Box::new(CpuAllocController::init(
                    desc.binding,
                    &mut ctx.memory_management,
                )?);
                // SAFETY:
                // - The binding has initialized memory for at least `len` bytes.
                result.push(unsafe { Bytes::from_controller(controller, len) });
            }
            Ok(result)
        }

        let res = inner(&mut self.ctx, descriptors);

        async move { res }
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptorOwned>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        Box::pin(self.read_async(descriptors))
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptorOwned, Vec<u8>)>) -> Result<(), IoError> {
        for (desc, data) in descriptors {
            if desc.strides != contiguous_strides(&desc.shape) {
                return Err(IoError::UnsupportedStrides);
            }

            self.copy_to_binding(desc.binding, &data);
        }
        Ok(())
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptorOwned>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let align = 8;
        let strides = descriptors
            .iter()
            .map(|desc| contiguous_strides(&desc.shape))
            .collect::<Vec<_>>();
        let sizes = descriptors
            .iter()
            .map(|desc| desc.shape.iter().product::<usize>() * desc.elem_size)
            .collect::<Vec<_>>();
        let total_size = sizes
            .iter()
            .map(|it| it.next_multiple_of(align))
            .sum::<usize>();

        let handle = self.ctx.memory_management.reserve(total_size as u64)?;
        let mem_handle = Handle::new(handle, None, None, stream_id, 0, total_size as u64);
        let handles = offset_handles(mem_handle, &sizes, align);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn execute(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
    ) {
        let cube_count = match count {
            CubeCount::Static(x, y, z) => [x, y, z],
            CubeCount::Dynamic(binding) => {
                let handle = self
                    .ctx
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .expect("Failed to find resource");
                let bytes = handle.read();
                let x = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
                let y = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                let z = u32::from_ne_bytes(bytes[8..12].try_into().unwrap());
                [x, y, z]
            }
        };
        self.runner.dispatch_execute(
            kernel,
            cube_count,
            bindings,
            kind,
            &mut self.ctx.memory_management,
        );
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
        self.utilities.logger.profile_summary();
        cubecl_common::future::block_on(self.sync(stream_id));
        self.ctx.timestamps.stop(token)
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<()> {
        self.utilities.logger.profile_summary();
        Box::pin(async move {})
    }
    fn copy_to_binding(&mut self, binding: Binding, data: &[u8]) {
        let mut resource = self
            .ctx
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .unwrap();
        resource.write().copy_from_slice(data);
    }
}
#[derive(new, Debug, Clone)]
#[allow(unused)]
pub struct CopyDescriptorOwned {
    pub binding: Binding,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub elem_size: usize,
}

#[derive(new, Debug, Clone)]
#[allow(unused)]
pub struct AllocationDescriptorOwned {
    pub kind: AllocationKind,
    pub shape: Vec<usize>,
    pub elem_size: usize,
}

pub enum CpuTask {
    Compute {
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
    },
    Read {
        descriptors: Vec<CopyDescriptorOwned>,
        callback: SyncSender<DynFut<Result<Vec<Bytes>, IoError>>>,
    },
    Write {
        descriptors: Vec<(CopyDescriptorOwned, Vec<u8>)>,
        callback: SyncSender<Result<(), IoError>>,
    },
    Create {
        descriptors: Vec<AllocationDescriptorOwned>,
        callback: SyncSender<Result<Vec<Allocation>, IoError>>,
    },
    Flush,
    Sync {
        callback: SyncSender<()>,
    },
    MemoryUsage {
        callback: SyncSender<MemoryUsage>,
    },
    MemoryCleanup,
    AllocMode {
        mode: MemoryAllocationMode,
    },
    StartProfile {
        callback: SyncSender<ProfilingToken>,
    },
    EndProfile {
        token: ProfilingToken,
        callback: SyncSender<Result<ProfileDuration, ProfileError>>,
    },
    GetResource {
        binding: Binding,
        callback: SyncSender<BindingResource<BytesResource>>,
    },
}
