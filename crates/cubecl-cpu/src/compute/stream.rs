use crate::compute::{
    alloc_controller::CpuAllocController, queue::CpuExecutionQueue, schedule::ScheduleTask,
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    ir::MemoryDeviceProperties,
    server::{
        CopyDescriptor, Handle, IoError, MemorySlot, ProfileError, ProfilingToken, ServerError,
    },
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
    },
    storage::{BytesResource, BytesStorage, ComputeStorage},
    timestamp_profiler::TimestampProfiler,
};
use std::sync::Arc;

pub struct CpuStream {
    queue: CpuExecutionQueue,
    pub(crate) memory_management: MemoryManagement<BytesStorage>,
    pub(crate) timestamps: TimestampProfiler,
    errors: Vec<ServerError>,
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
            errors: Vec::new(),
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        self.queue.add(task);
    }

    pub fn flush(&mut self) {
        self.queue.flush();
    }

    /// Returns whether the stream can accept new tasks.
    pub fn is_healty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Registers a new error into the error sink.
    pub fn error(&mut self, error: ServerError) {
        self.errors.push(error);
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.memory_management.reserve(size)
    }

    /// Maps handles to their corresponding buffers.
    pub fn bind(&mut self, slots: Vec<MemorySlot>, handles: Vec<Handle>) {
        for (buffer, handle) in slots.into_iter().zip(handles.into_iter()) {
            self.memory_management.bind(handle.id, buffer);
        }
    }

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

    pub fn sync(&mut self) -> Result<(), ServerError> {
        self.queue.flush();

        Ok(())
    }

    pub fn flush_errors(&mut self) -> Vec<ServerError> {
        core::mem::take(&mut self.errors)
    }

    pub fn start_profile(&mut self) -> ProfilingToken {
        if let Err(err) = self.sync() {
            log::warn!("{err}");
        };
        self.timestamps.start()
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = self.sync() {
            self.timestamps.error(ProfileError::Server(Box::new(err)));
        }

        self.timestamps.stop(token)
    }

    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_management.mode(mode);
    }

    pub fn get_resource(&mut self, handle: Handle) -> Result<BytesResource, IoError> {
        let slot = self.memory_management.get_slot(handle)?;
        let handle = self
            .memory_management
            .get_storage(slot.memory.binding())
            .ok_or_else(|| IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })?;
        let handle = match slot.offset_start {
            Some(offset) => handle.offset_start(offset),
            None => handle,
        };
        let handle = match slot.offset_end {
            Some(offset) => handle.offset_end(offset),
            None => handle,
        };
        Ok(self.memory_management.storage().get(&handle))
    }
}
