use crate::compute::{
    alloc_controller::CpuAllocController, schedule::ScheduleTask, threadpool::Threadpool,
};
use crossbeam_utils::CachePadded;
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{
        Binding, CopyDescriptor, IoError, ProfileError, ProfilingToken, ServerError,
        StreamErrorMode,
    },
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
    },
    storage::{BytesResource, BytesStorage},
    timestamp_profiler::TimestampProfiler,
};
use std::sync::{Arc, atomic::AtomicU64};

pub struct CpuStream {
    // TEMP: only read by the disabled unit-limit check in `enqueue_task`. Kept so
    // the plumbing stays in place; remove the attribute when the check is restored.
    #[allow(dead_code)]
    pub(crate) max_units_per_cube: u32,
    pub(crate) memory_management: MemoryManagement<BytesStorage>,
    /// Dedicated pool for per-launch shared memory.
    ///
    /// Shared memory MUST NOT be reserved from `memory_management`: kernel input/output
    /// bindings keep their allocation alive through a `ManagedMemoryBinding`, which does
    /// *not* hold the pool reservation. `reserve` would then hand a still-bound tensor's
    /// slice to shared memory, aliasing an input and corrupting it in place.
    pub(crate) shared_memory_management: MemoryManagement<BytesStorage>,
    pub(crate) timestamps: TimestampProfiler,
    errors: Vec<ServerError>,
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
        max_units_per_cube: u32,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        // `memory_config` shapes the main pool only; the shared pool below is
        // left alone, as it has a deliberate configuration that must not be
        // overridden. Pool layout overrides reach GPU runtimes through
        // `configure_memory_pools`; the CPU runtime has no such override and
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
            max_units_per_cube,
            memory_management,
            shared_memory_management,
            timestamps: TimestampProfiler::default(),
            errors: Vec::new(),
            threadpool,
            next_counter_step,
            atomic_counter,
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        self.flush_uncheck();
        match task {
            ScheduleTask::Write { data, mut buffer } => {
                buffer.resource_mut().write().copy_from_slice(&data);
            }
            ScheduleTask::Execute {
                mlir_engine,
                bindings,
                cube_dim,
                cube_count,
            } => {
                let requested = cube_dim.num_elems();
                // TEMP: the threadpool now grows to fit any cube_dim (it spawns one
                // worker per unit for barrier kernels, see Threadpool::execute_data),
                // so the per-launch unit-count limit is disabled. Uncomment to restore
                // the hard cap (also re-add the LaunchError/ResourceLimitError imports).
                // let max = self.max_units_per_cube;
                // if requested > max {
                //     let launch_error: LaunchError = ResourceLimitError::MaxUnitPerCube {
                //         requested,
                //         max,
                //         backtrace: BackTrace::capture(),
                //     }
                //     .into();
                //     self.error(launch_error.into());
                //     return;
                // }

                self.threadpool.lock().execute_data(
                    mlir_engine,
                    bindings,
                    cube_dim,
                    cube_count,
                    &mut self.shared_memory_management,
                    self.next_counter_step,
                    &self.atomic_counter,
                );
                self.next_counter_step += requested as u64;
            }
        }
    }

    fn flush_uncheck(&mut self) {
        while self
            .atomic_counter
            .load(std::sync::atomic::Ordering::Acquire)
            != self.next_counter_step
        {
            std::hint::spin_loop();
        }
    }

    pub fn flush(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        self.flush_uncheck();
        self.flush_errors(mode)
    }

    fn flush_errors(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        if mode.flush {
            let errors = self.flush_errors_queue();

            if !mode.ignore && !errors.is_empty() {
                let error = ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                };
                return Err(error);
            }
        } else if !mode.ignore && !self.errors.is_empty() {
            let error = ServerError::ServerUnhealthy {
                errors: self.errors.clone(),
                backtrace: BackTrace::capture(),
            };
            return Err(error);
        }

        Ok(())
    }

    pub(crate) fn flush_errors_queue(&mut self) -> Vec<ServerError> {
        let errors = core::mem::take(&mut self.errors);

        if !errors.is_empty() {
            self.timestamps.error(ProfileError::Unknown {
                reason: alloc::format!("{:?}", errors),
                backtrace: BackTrace::capture(),
            });
        }

        errors
    }

    /// Returns whether the stream can accept new tasks.
    pub fn is_healthy(&self) -> bool {
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

    pub fn sync(&mut self) -> Result<(), ServerError> {
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        self.sync()?;

        Ok(self.timestamps.start())
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

    pub fn get_resource(&mut self, binding: Binding) -> Result<BytesResource, IoError> {
        self.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        )
    }
}
