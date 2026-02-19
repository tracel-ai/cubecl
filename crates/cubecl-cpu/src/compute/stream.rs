use crate::compute::{
    alloc_controller::CpuAllocController, queue::CpuExecutionQueue, schedule::ScheduleTask,
    server::contiguous_strides,
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{
        Allocation, AllocationDescriptor, CopyDescriptor, Handle, IoError, ProfileError,
        ProfilingToken, ServerError,
    },
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryManagement, MemoryManagementOptions, create_buffers,
    },
    storage::BytesStorage,
    timestamp_profiler::TimestampProfiler,
};
use std::sync::Arc;

pub struct CpuStream {
    queue: CpuExecutionQueue,
    pub(crate) memory_management: MemoryManagement<BytesStorage>,
    pub(crate) timestamps: TimestampProfiler,
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

        Self {
            memory_management,
            timestamps: TimestampProfiler::default(),
            queue: CpuExecutionQueue::get(logger),
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        self.queue.add(task);
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
        fn inner(
            mem: &mut MemoryManagement<BytesStorage>,
            descriptor: CopyDescriptor,
        ) -> Result<Bytes, IoError> {
            let len = descriptor.handle.size_in_used() as usize;
            let controller = Box::new(CpuAllocController::init(descriptor.handle, mem)?);
            // SAFETY:
            // - The binding has initialized memory for at least `len` bytes.
            Ok(unsafe { Bytes::from_controller(controller, len) })
        }

        let res = inner(&mut self.memory_management, descriptor);

        async move { res }
    }
    pub fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor>,
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

        let handle = self.memory_management.reserve(total_size as u64)?;
        let mem_handle = Handle::new(handle, None, None, stream_id, 0, total_size as u64);
        let handles = create_buffers(mem_handle, &sizes, align);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    pub fn sync(&mut self) -> Result<(), ServerError> {
        self.queue.flush();

        Ok(())
    }

    pub fn start_profile(&mut self) -> ProfilingToken {
        if let Err(err) = self.sync() {
            log::warn!("{err}");
        };
        self.timestamps.start()
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = self.sync() {
            self.timestamps.error(err.into());
        }

        self.timestamps.stop(token)
    }

    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_management.mode(mode);
    }
}
