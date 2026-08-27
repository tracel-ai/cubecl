use crate::compute::{
    alloc_controller::CpuAllocController, schedule::ScheduleTask, threadpool::Threadpool,
};
use crossbeam_utils::CachePadded;
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{BufferBinding, CopyDescriptor, IoError, ProfileError, ProfilingToken, ServerError},
};
use cubecl_environment::stream::StreamId;
use cubecl_runtime::{
    config::{CubeClRuntimeConfig, RuntimeConfig, streaming::BatchWait},
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, FailureId, ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement,
        MemoryManagementOptions,
    },
    storage::{BytesResource, BytesStorage},
    stream::StreamMemory,
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
    threadpool: &'static spin::Mutex<Threadpool>,
    batch_wait: BatchWaitChoice,
    launches: u32,
    dispatch_ns: u64,
    next_counter_step: u64,
    atomic_counter: Arc<CachePadded<AtomicU64>>,
}

impl StreamMemory for CpuStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory_management
            .failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.memory_management
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory_management
            .written(&binding.memory, binding.range(), failures)
    }
}

impl core::fmt::Debug for CpuStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuStream").finish()
    }
}

/// Decides once whether this stream waits for a filled batch, from what the
/// workload does rather than from what it is.
///
/// A batch is timed two ways: what the client spent dispatching it, and what
/// it then spent waiting for the pool to finish. A pool that drains faster
/// than the client can refill it leaves that wait small, and queuing behind it
/// is what keeps it busy. A pool the client cannot keep up with leaves the
/// wait dominant, and queuing only pins buffers for a gain that is not there.
///
/// The first batches are all kernel compilation, which inverts the reading, so
/// they are skipped, and only batches that dispatched something count towards
/// either window. The choice is then frozen: taking it changes the very
/// costs it was made from, since a hot worker is woken by a send and a parked
/// one is not, and a policy that kept adapting would chase its own tail.
#[derive(Default)]
struct BatchWaitChoice {
    batches: u32,
    dispatch_ns: u64,
    wait_ns: u64,
    decided: Option<bool>,
}

impl BatchWaitChoice {
    /// Batches given over to compilation before any of this means anything.
    const WARMUP: u32 = 32;
    /// Batches the choice is made from.
    const SAMPLE: u32 = 32;
    /// How far the wait may outweigh the dispatch and still be worth queuing.
    /// Measured across mobilenetv3-small and resnet50: 6 and 27 for the shapes
    /// that gain, 98 for the one that does not.
    const RATIO: u64 = 50;

    fn batch(&mut self, launches: u32, dispatch_ns: u64, wait_ns: u64) {
        // A drain with nothing dispatched into it teaches nothing, and there
        // are many: every host write and every shared-memory launch drains the
        // stream. Loading a model's weights alone would otherwise spend the
        // whole window before a single kernel has run.
        if self.decided.is_some() || launches == 0 {
            return;
        }
        self.batches += 1;
        if self.batches <= Self::WARMUP {
            return;
        }
        self.dispatch_ns += dispatch_ns;
        self.wait_ns += wait_ns;
        if self.batches >= Self::WARMUP + Self::SAMPLE {
            let decided = self.wait_ns < Self::RATIO * self.dispatch_ns.max(1);
            self.decided = Some(decided);
            if std::env::var("CUBECL_CPU_METER").is_ok() {
                eprintln!(
                    "BATCHWAIT\tskip {decided}\tratio {:.1}\tdispatch {} us\twait {} us\tbatches {}",
                    self.wait_ns as f64 / self.dispatch_ns.max(1) as f64,
                    self.dispatch_ns / 1000,
                    self.wait_ns / 1000,
                    self.batches
                );
            }
        }
    }

    /// Whether the wait may be skipped. Undecided still waits: the choice is
    /// made from what the waiting measures.
    fn skips(&self) -> bool {
        self.decided.unwrap_or(false)
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
            threadpool,
            batch_wait: BatchWaitChoice::default(),
            launches: 0,
            dispatch_ns: 0,
            next_counter_step,
            atomic_counter,
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask, failures: &mut ErrorGraph) {
        // Launches pipeline: `ComputeTask::is_ready` orders tasks and the
        // launch's resources ride in `SharedData::keepalive`, so the client
        // only drains where that protocol does not cover:
        // * a host `Write`, which copies on this thread and would race a
        //   queued kernel reading the buffer;
        // * a shared-memory kernel, whose pool reservations are released at
        //   enqueue — sound only while one such launch has the pool to itself.
        match task {
            ScheduleTask::Write { data, mut buffer } => {
                self.submit();
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
                    self.submit();
                }
                // No unit cap: the threadpool grows to fit any cube_dim, one
                // worker per unit for barrier kernels.
                let units = cube_dim.num_elems();
                let dispatch_start = std::time::Instant::now();
                self.threadpool.lock().execute_data(
                    pliron_engine,
                    bindings,
                    cube_dim,
                    cube_count,
                    &mut self.shared_memory_management,
                    failures,
                    self.next_counter_step,
                    &self.atomic_counter,
                );
                self.dispatch_ns += dispatch_start.elapsed().as_nanos() as u64;
                self.launches += 1;
                self.next_counter_step += units as u64;
            }
        }
    }

    /// Wait for the queued work and surface nothing.
    ///
    /// For the pooled paths that flush the stream without any logical stream
    /// asking — a full task queue, the ordering barrier before a write, the
    /// scheduler aligning streams. Whatever is queued stays queued, for the
    /// flush of the stream that owns it.
    pub fn submit(&mut self) {
        // Spin briefly, then yield between polls: the client is not pinned,
        // and a pure spin parked on a worker's logical CPU keeps that worker
        // off it until the next timer tick (~3 ms unit-start stalls).
        const SPINS_BEFORE_YIELD: u32 = 1_000;
        let wait_start = std::time::Instant::now();
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
        let launches = core::mem::take(&mut self.launches);
        let dispatch_ns = core::mem::take(&mut self.dispatch_ns);
        self.batch_wait.batch(
            launches,
            dispatch_ns,
            wait_start.elapsed().as_nanos() as u64,
        );
        // A worker waiting its turn should yield only once the client is
        // queueing past it; until then there is nobody to yield to.
        crate::compute::threadpool::scheduler::dispatcher::set_skipping_batch_wait(
            self.may_skip_wait(),
        );
    }

    /// Wait for the queued work. A launch failure is not the flush's to
    /// report: it lives on the buffers the launch left unwritten, and
    /// surfaces on any read, sync or check of them.
    pub fn flush(&mut self, _owner: StreamId) -> Result<(), ServerError> {
        self.submit();
        Ok(())
    }

    /// Whether a full batch may be handed to the pool without waiting for it.
    pub(crate) fn may_skip_wait(&self) -> bool {
        match CubeClRuntimeConfig::get().streaming.batch_wait {
            BatchWait::Always => false,
            BatchWait::Never => true,
            BatchWait::Auto => self.batch_wait.skips(),
        }
    }

    /// Mark every open profile invalid: a failure inside a profiling window
    /// invalidates the measurement. A no-op with no profile open.
    pub fn profile_failure(&mut self, error: &ServerError) {
        self.timestamps.failure(error);
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.memory_management.reserve(size, failures)
    }

    /// Maps handles to their corresponding buffers.
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        failures: &mut ErrorGraph,
    ) {
        self.memory_management
            .bind(reserved, new, 0, failures)
            .unwrap();
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

    pub fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.flush(stream_id)?;

        Ok(self.timestamps.start())
    }

    pub fn end_profile(
        &mut self,
        token: ProfilingToken,
        stream_id: StreamId,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = self.flush(stream_id) {
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
