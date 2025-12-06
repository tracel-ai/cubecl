use crate::{
    CpuCompiler,
    compute::{
        alloc_controller::CpuAllocController,
        queue::CpuExecutionQueue,
        schedule::{BindingsResource, ScheduleTask},
        scheduler::KernelRunner,
        server::{CpuContext, contiguous_strides},
    },
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    CompilationError, CubeCount, CubeTask, ExecutionMode, MemoryConfiguration,
    server::{
        Allocation, AllocationDescriptor, CopyDescriptor, ExecutionError, Handle, IoError,
        ProfileError, ProfilingToken,
    },
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryDeviceProperties, MemoryManagement, MemoryManagementOptions,
        offset_handles,
    },
    storage::BytesStorage,
};
use std::sync::Arc;

pub struct CpuStream {
    pub(crate) ctx: CpuContext,
    runner: KernelRunner,
    queue: CpuExecutionQueue,
}

impl core::fmt::Debug for CpuStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuStream").finish()
    }
}

impl CpuStream {
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        let memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Main CPU"),
        );
        let memory_management_shared_memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
            logger,
            MemoryManagementOptions::new("Shared Memory"),
        );

        let ctx = CpuContext::new(memory_management, memory_management_shared_memory);

        Self {
            ctx,
            runner: KernelRunner::default(),
            queue: CpuExecutionQueue::get(),
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        self.queue.push(task);
    }

    pub fn flush(&mut self) {
        self.queue.flush();
    }
}

impl CpuStream {
    pub fn read_async(
        &mut self,
        descriptor: CopyDescriptor,
    ) -> impl Future<Output = Result<Bytes, IoError>> + Send + use<> {
        fn inner(ctx: &mut CpuContext, descriptor: CopyDescriptor) -> Result<Bytes, IoError> {
            let len = descriptor.binding.size() as usize;
            let controller = Box::new(CpuAllocController::init(
                descriptor.binding,
                &mut ctx.memory_management,
            )?);
            // SAFETY:
            // - The binding has initialized memory for at least `len` bytes.
            Ok(unsafe { Bytes::from_controller(controller, len) })
        }

        let res = inner(&mut self.ctx, descriptor);

        async move { res }
    }
    pub fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let align = 8;
        let strides = descriptors
            .iter()
            .map(|desc| contiguous_strides(desc.shape))
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
    pub fn prepare(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        count: CubeCount,
        bindings: BindingsResource,
        kind: ExecutionMode,
    ) -> Result<ScheduleTask, CompilationError> {
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

        self.runner.prepare(
            kernel,
            cube_count,
            bindings,
            kind,
            &mut self.ctx.memory_management_shared_memory,
        )
    }

    pub fn sync(&mut self) -> Result<(), ExecutionError> {
        self.queue.flush();

        Ok(())
    }

    pub fn start_profile(&mut self) -> ProfilingToken {
        if let Err(err) = self.sync() {
            self.ctx.timestamps.error(err.into());
        };
        self.ctx.timestamps.start()
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = self.sync() {
            self.ctx.timestamps.error(err.into());
        }

        self.ctx.timestamps.stop(token)
    }

    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.ctx.memory_management.mode(mode);
    }
}
