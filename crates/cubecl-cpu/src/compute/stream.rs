use crate::compute::{
    alloc_controller::CpuAllocController, schedule::ScheduleTask, threadpool::Threadpool,
};
use crossbeam_utils::CachePadded;
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{
        BufferBinding, CopyDescriptor, IoError, ProfileError, ProfilingToken, ServerError,
        StreamErrorMode,
    },
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
    },
    storage::{BytesResource, BytesStorage},
    stream::StreamErrors,
    timestamp_profiler::TimestampProfiler,
};
use std::sync::{Arc, atomic::AtomicU64};

pub struct CpuStream {
    pub(crate) memory_management: MemoryManagement<BytesStorage>,
    /// Dedicated pool for per-launch shared memory.
    ///
    /// Shared memory MUST NOT be reserved from `memory_management`: kernel input/output
    /// bindings keep their allocation alive through a `ManagedMemoryBinding`, which does
    /// *not* hold the pool reservation. `reserve` would then hand a still-bound tensor's
    /// slice to shared memory, aliasing an input and corrupting it in place.
    pub(crate) shared_memory_management: MemoryManagement<BytesStorage>,
    pub(crate) timestamps: TimestampProfiler,
    errors: StreamErrors,
    threadpool: &'static spin::Mutex<Threadpool>,
    next_counter_step: u64,
    atomic_counter: Arc<CachePadded<AtomicU64>>,
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
        // `memory_config` shapes the main pool only; the shared pool below is
        // left alone, as it has a deliberate configuration that must not be
        // overridden. Pool layout overrides reach GPU runtimes through
        // `install_memory_pools`; the CPU runtime has no such override and
        // keeps the config it's handed.
        let memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            memory_config.clone(),
            logger.clone(),
            MemoryManagementOptions::new("Main CPU"),
        );
        let shared_memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Shared CPU"),
        );
        let threadpool = Threadpool::get();
        let next_counter_step = 0;
        let atomic_counter = Arc::new(CachePadded::new(AtomicU64::new(0)));
        Self {
            memory_management,
            shared_memory_management,
            timestamps: TimestampProfiler::default(),
            errors: StreamErrors::default(),
            threadpool,
            next_counter_step,
            atomic_counter,
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        // Launches pipeline: `ComputeTask::is_ready` orders tasks and the
        // launch's resources ride in `SharedData::keepalive`, so the client
        // only drains where that protocol does not cover:
        // * a host `Write`, which copies on this thread and would race a
        //   queued kernel reading the buffer;
        // * a shared-memory kernel, whose pool reservations are released at
        //   enqueue — sound only while one such launch has the pool to itself.
        match task {
            ScheduleTask::Write { data, mut buffer } => {
                self.flush_uncheck();
                buffer.resource_mut().write().copy_from_slice(&data);
            }
            ScheduleTask::Execute {
                pliron_engine,
                bindings,
                cube_dim,
                cube_count,
                ..
            } => {
                if !pliron_engine
                    .requirements()
                    .shared_memories
                    .blocks
                    .is_empty()
                {
                    self.flush_uncheck();
                }
                // No unit cap: the threadpool grows to fit any cube_dim, one
                // worker per unit for barrier kernels.
                let units = cube_dim.num_elems();
                self.threadpool.lock().execute_data(
                    pliron_engine,
                    bindings,
                    cube_dim,
                    cube_count,
                    &mut self.shared_memory_management,
                    self.next_counter_step,
                    &self.atomic_counter,
                );
                self.next_counter_step += units as u64;
            }
        }
    }

    fn flush_uncheck(&mut self) {
        // Spin briefly, then yield between polls: the client is not pinned,
        // and a pure spin parked on a worker's logical CPU keeps that worker
        // off it until the next timer tick (~3 ms unit-start stalls).
        const SPINS_BEFORE_YIELD: u32 = 1_000;
        let mut spins = 0u32;
        while self
            .atomic_counter
            .load(std::sync::atomic::Ordering::Acquire)
            != self.next_counter_step
        {
            spins += 1;
            if spins < SPINS_BEFORE_YIELD {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Wait for the queued work, then surface the errors `stream_id` owns (see
    /// [`StreamErrors`]). `None` is for the pooled paths that flush the stream
    /// without any logical stream asking, which never surface errors anyway.
    pub fn flush(
        &mut self,
        mode: StreamErrorMode,
        stream_id: Option<StreamId>,
    ) -> Result<(), ServerError> {
        self.flush_uncheck();
        self.flush_errors(mode, stream_id)
    }

    fn flush_errors(
        &mut self,
        mode: StreamErrorMode,
        stream_id: Option<StreamId>,
    ) -> Result<(), ServerError> {
        if mode.flush {
            let errors = self.flush_errors_queue(stream_id);

            if !mode.ignore && !errors.is_empty() {
                let error = ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                };
                return Err(error);
            }
        } else if !mode.ignore && self.errors.any(stream_id) {
            let error = ServerError::ServerUnhealthy {
                errors: self.errors.peek(stream_id),
                backtrace: BackTrace::capture(),
            };
            return Err(error);
        }

        Ok(())
    }

    pub(crate) fn flush_errors_queue(&mut self, stream_id: Option<StreamId>) -> Vec<ServerError> {
        let errors = self.errors.take(stream_id);

        if !errors.is_empty() {
            self.timestamps.error(ProfileError::Unknown {
                reason: alloc::format!("{:?}", errors),
                backtrace: BackTrace::capture(),
            });
        }

        errors
    }

    /// Whether the stream can accept new tasks from `stream_id`.
    ///
    /// Errors are queued per logical stream (see [`StreamErrors`]), so the
    /// backend stream is broken for the streams whose errors are still queued
    /// on it, not for every stream sharing it.
    pub fn is_healthy(&self, stream_id: StreamId) -> bool {
        !self.errors.any(Some(stream_id))
    }

    /// Registers a new error into the error sink, for `stream_id` to surface.
    pub fn error(&mut self, stream_id: StreamId, error: ServerError) {
        self.errors.push(stream_id, error);
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.memory_management.reserve(size)
    }

    /// Maps handles to their corresponding buffers.
    pub fn bind(&mut self, reserved: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        self.memory_management.bind(reserved, new, 0).unwrap();
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

    pub fn sync(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.flush(
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
            Some(stream_id),
        )
    }

    pub fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.sync(stream_id)?;

        Ok(self.timestamps.start())
    }

    pub fn end_profile(
        &mut self,
        token: ProfilingToken,
        stream_id: StreamId,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = self.sync(stream_id) {
            self.timestamps.error(ProfileError::Server(Box::new(err)));
        }

        self.timestamps.stop(token)
    }

    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_management.mode(mode);
    }

    pub fn get_resource(&mut self, binding: BufferBinding) -> Result<BytesResource, IoError> {
        self.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        )
    }
}
