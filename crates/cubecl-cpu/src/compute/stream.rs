use crate::compute::{
    alloc_controller::CpuAllocController, schedule::ScheduleTask, threadpool::Threadpool,
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    ir::MemoryDeviceProperties,
    server::{
        Binding, CopyDescriptor, IoError, LaunchError, ProfileError, ProfilingToken,
        ResourceLimitError, ServerError, StreamErrorMode,
    },
};
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
    pub(crate) max_units_per_cube: u32,
    pub(crate) memory_management: MemoryManagement<BytesStorage>,
    pub(crate) timestamps: TimestampProfiler,
    errors: Vec<ServerError>,
    threadpool: &'static spin::Mutex<Threadpool>,
    stream_id: usize,
    next_counter_step: u64,
    atomic_counter: Arc<AtomicU64>,
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
        stream_id: usize,
    ) -> Self {
        let memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Main CPU"),
        );
        let threadpool = Threadpool::get();
        let next_counter_step = 0;
        let atomic_counter = Arc::new(AtomicU64::new(0));
        Self {
            max_units_per_cube,
            memory_management,
            timestamps: TimestampProfiler::default(),
            errors: Vec::new(),
            stream_id,
            threadpool,
            next_counter_step,
            atomic_counter,
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        self.flush_uncheck();
        match task {
            ScheduleTask::Write { data, mut buffer } => {
                buffer.write().copy_from_slice(&data);
            }
            ScheduleTask::Execute {
                mlir_engine,
                bindings,
                cube_dim,
                cube_count,
            } => {
                let requested = cube_dim.num_elems();
                let max = self.max_units_per_cube;
                if requested > max {
                    let launch_error: LaunchError = ResourceLimitError::MaxUnitPerCube {
                        requested,
                        max,
                        backtrace: BackTrace::capture(),
                    }
                    .into();
                    self.error(launch_error.into());
                    return;
                }

                self.threadpool.lock().execute_data(
                    mlir_engine,
                    bindings,
                    cube_dim,
                    cube_count,
                    &mut self.memory_management,
                    self.stream_id,
                    self.next_counter_step,
                    &self.atomic_counter,
                );
                self.next_counter_step += requested as u64;
            }
        }
    }

    fn flush_uncheck(&mut self) {
        // println!(
        //     "{}",
        //     self.atomic_counter
        //         .load(std::sync::atomic::Ordering::Acquire)
        // );
        while self
            .atomic_counter
            .load(std::sync::atomic::Ordering::Acquire)
            != self.next_counter_step
        {
            // println!(
            //     "{}",
            //     self.atomic_counter
            //         .load(std::sync::atomic::Ordering::Acquire)
            // );
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
