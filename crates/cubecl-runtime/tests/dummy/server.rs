use super::DummyKernel;
use crate::dummy::DummyCompiler;
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_ir::{
    AddressType, DeviceIdentity, DeviceProperties, ElemType, HardwareProperties,
    MemoryDeviceProperties, UIntKind, VectorSize,
    features::Features,
    metadata::Info,
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement, MemoryUsage,
    },
    server::{
        BufferBinding, ComputeServer, CopyDescriptor, CubeCount, CubeDim, Handle, KernelArguments,
        KernelResource, ProfileError, ProfilingToken, ServerCommunication, ServerError,
        ServerUtilities,
    },
    storage::{BytesResource, BytesStorage, ComputeStorage, ManagedResource},
    timestamp_profiler::TimestampProfiler,
};
use cubecl_zspace::{Shape, Strides};
use std::sync::Arc;

/// Makes `start_profile` fail while set, the way a real server refuses one
/// inside a graph capture window. Process-wide, so tests that flip it run
/// `serial`.
pub static REFUSE_PROFILES: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// The dummy server is used to test the cubecl-runtime infrastructure.
/// It uses simple memory management with a bytes storage on CPU, without asynchronous tasks.
#[derive(Debug)]
pub struct DummyServer {
    memory_management: MemoryManagement<BytesStorage>,
    timestamps: TimestampProfiler,
    utilities: Arc<ServerUtilities<Self>>,
    /// The failures the server's tainted allocations still point at.
    ///
    /// Errors live on the memory here as they do on a real server: a failed
    /// launch leaves the buffers it was going to write carrying the failure,
    /// and a read, check or sync of one of them reports it. Nothing is queued
    /// and no call "drains" anything.
    failures: ErrorGraph,
}

#[derive(Debug, Clone)]
pub struct KernelTask {
    kernel: Arc<dyn DummyKernel>,
}

impl KernelMetadata for KernelTask {
    fn name(&self) -> &'static str {
        self.kernel.name()
    }

    fn id(&self) -> KernelId {
        self.kernel.id()
    }

    fn address_type(&self) -> cubecl_ir::ElemType {
        ElemType::UInt(UIntKind::U32)
    }
}

impl core::fmt::Display for KernelTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dummy kernel")
    }
}

impl CubeTask<DummyCompiler> for KernelTask {
    fn define(&self) -> cubecl_runtime::kernel::KernelDefinition {
        // The dummy server compiles directly and never keys a cache, so nothing here is observed.
        let settings =
            KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32);
        cubecl_runtime::kernel::KernelDefinition {
            body: cubecl_ir::Scope::root(settings.clone()),
            settings,
            info: Info::default(),
        }
    }

    fn compile(
        &self,
        _definition: cubecl_runtime::kernel::KernelDefinition,
        _compiler: &mut DummyCompiler,
        _compilation_options: &<DummyCompiler as cubecl_runtime::compiler::Compiler>::CompilationOptions,
    ) -> Result<cubecl_runtime::kernel::CompiledKernel<DummyCompiler>, CompilationError> {
        if let Some(err) = self.kernel.compilation_error() {
            return Err(err);
        }

        Ok(CompiledKernel {
            entrypoint_name: self.kernel.name().to_string(),
            debug_name: None,
            source: String::new(),
            repr: Some(self.clone()),
            io: None,
            cube_dim: CubeDim::new_single(),
            debug_info: None,
        })
    }
}

impl KernelTask {
    pub fn new(kernel: impl DummyKernel) -> Self {
        Self {
            kernel: Arc::new(kernel),
        }
    }

    pub fn compute(&self, resources: &mut [&mut BytesResource]) {
        self.kernel.compute(resources);
    }
}

impl ServerCommunication for DummyServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl ComputeServer for DummyServer {
    type Kernel = Box<dyn CubeTask<DummyCompiler>>;
    type Storage = BytesStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<cubecl_runtime::server::ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, _stream_id: StreamId) {
        let reserved = self
            .memory_management
            .reserve(size, &mut self.failures)
            .unwrap();
        self.memory_management
            .bind(reserved, memory.clone(), 0, &mut self.failures)
            .unwrap();
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // The claim check every real server's read makes: a buffer a failed
        // launch never filled reports the failure rather than handing back
        // whatever bytes were there before.
        if let Err(err) = self.ensure_written(descriptors.iter().map(|d| &d.handle)) {
            return Box::pin(async move { Err(err) });
        }
        let bytes: Vec<_> = descriptors
            .into_iter()
            .map(|b| {
                let size = b.handle.size_in_used();
                let resource = self
                    .memory_management
                    .get_resource(
                        b.handle.memory.clone(),
                        b.handle.offset_start,
                        b.handle.offset_end,
                    )
                    .unwrap();
                // Keep the binding alive in the future so the memory pool
                // doesn't reuse this storage while we still hold a pointer.
                (resource, size, b.handle.memory)
            })
            .collect();

        Box::pin(async move {
            Ok(bytes
                .into_iter()
                .map(|(b, size, _binding)| {
                    Bytes::from_bytes_vec(b.read()[0..size as usize].to_vec())
                })
                .collect())
        })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, _stream_id: StreamId) {
        for (descriptor, data) in descriptors {
            let storage_h = self
                .memory_management
                .get_storage(descriptor.handle.memory)
                .unwrap();
            let mut bytes = self.memory_management.storage().get(&storage_h).unwrap();
            bytes.write()[..data.len()].copy_from_slice(&data);
        }
    }

    fn sync(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        // The claim check a read would have made, without the read. There is
        // no device to fault here, so the barrier itself never fails.
        let result = self.ensure_written(handles.iter());
        Box::pin(async move { result })
    }

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        _stream_id: StreamId,
    ) -> Result<ManagedResource<BytesResource>, ServerError> {
        let resource = self.memory_management.get_resource(
            binding.memory.clone(),
            binding.offset_start,
            binding.offset_end,
        )?;

        Ok(ManagedResource::new(binding.memory, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        launch_mode: cubecl_runtime::dry_run::LaunchMode,
    ) {
        let kernel = match kernel.compile(kernel.define(), &mut DummyCompiler, &()) {
            Ok(kernel) => kernel,
            Err(err) => {
                // No IR, so no compiled answer about what the kernel writes:
                // the caller's declared IO decides, exactly as on a real
                // server — only the declared outputs carry the failure, and
                // the buffers the kernel was only going to read stay
                // readable for whatever launches next on them.
                let error = ServerError::from(cubecl_runtime::server::LaunchError::from(err));
                self.timestamps.failure(&error);
                if !launch_mode.is_skipped() {
                    let written: Vec<_> = bindings.buffers_written(None).cloned().collect();
                    self.taint(error, written.iter());
                }
                return;
            }
        };

        // Compiled above, exactly as a real server does — and, exactly as a
        // real server does, a skipped launch stops before anything touches a
        // buffer, so a dry run's lazily-carved allocations stay unmapped.
        if launch_mode.is_skipped() {
            return;
        }

        // The check a real server's write scope makes on the way in: an input
        // that carries a failure skips the launch, and the outputs take that
        // failure so a read downstream names the root cause.
        if let Err(error) = self.ensure_written(bindings.buffers_read(None)) {
            self.timestamps.failure(&error);
            let written: Vec<_> = bindings.buffers_written(None).cloned().collect();
            self.taint(error, written.iter());
            return;
        }

        let written: Vec<_> = bindings.buffers_written(None).cloned().collect();

        let mut resources: Vec<_> = bindings
            .resources
            .into_iter()
            .map(|res| match res {
                KernelResource::Buffer(binding) => binding,
                KernelResource::TensorMap(tensor_map) => tensor_map.binding,
            })
            .map(|b| {
                self.memory_management
                    .get_resource(b.memory, b.offset_start, b.offset_end)
                    .unwrap()
            })
            .collect();
        let data = bytemuck::cast_slice(&bindings.info.data);
        let metadata = Handle::new(stream_id, data.len() as u64);
        self.bind_with_data(data, metadata.clone(), stream_id);

        resources.push({
            self.memory_management
                .get_resource(
                    metadata.memory.binding(),
                    metadata.offset_start,
                    metadata.offset_end,
                )
                .unwrap()
        });

        let mut resources: Vec<_> = resources.iter_mut().collect();

        kernel.repr.unwrap().compute(resources.as_mut_slice());

        // The work ran: its write set has a writer again, which is what
        // releases an earlier failure's claim on those buffers — a relaunch
        // into a tainted buffer is exactly how the buffer gets repaired. A
        // panic in `compute` never reaches this, so an earlier failure's
        // claim survives a launch that blew up instead of writing.
        for handle in &written {
            self.memory_management
                .written(&handle.memory, handle.range(), &mut self.failures);
        }
    }

    fn flush(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        // A launch failure is not the flush's to report: it lives on the
        // buffers the launch left unwritten. There is no device fault here,
        // so nothing is left for a flush to say.
        Ok(())
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.ensure_written(handles.iter())
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> MemoryUsage {
        self.memory_management.memory_usage()
    }

    fn memory_report(
        &mut self,
        _stream_id: StreamId,
    ) -> cubecl_runtime::memory_management::MemoryReport {
        self.memory_management.memory_report()
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {
        self.memory_management.cleanup(true, &mut self.failures);
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        if REFUSE_PROFILES.load(core::sync::atomic::Ordering::Relaxed) {
            return Err(ServerError::Generic {
                reason: "this test server was told to refuse profiles".into(),
                backtrace: BackTrace::capture(),
            });
        }
        Ok(self.timestamps.start())
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        // A failure inside the window already invalidated the measurement at
        // the moment it happened, exactly as on a real server: the launch path
        // tags the profiler where it fails rather than at a drain.
        self.timestamps.stop(token)
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, _stream_id: StreamId) {
        self.memory_management.mode(mode)
    }
}

impl DummyServer {
    pub fn new(
        memory_management: MemoryManagement<BytesStorage>,
        mem_props: MemoryDeviceProperties,
    ) -> Self {
        let hardware = HardwareProperties {
            load_width: 128,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 32,
            max_shared_memory_size: 48000,
            max_cube_count: (u16::MAX as u32, u16::MAX as u32, u16::MAX as u32),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 64),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            last_level_cache_size: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };
        let features = Features::default();
        let timing_method = cubecl_common::profile::TimingMethod::System;
        let props = DeviceProperties::new(
            features,
            mem_props,
            hardware,
            timing_method,
            DeviceIdentity {
                name: "dummy".to_string(),
                fingerprint: "dummy".to_string(),
            },
        );
        let logger = Arc::new(ServerLogger::default());

        let utilities = Arc::new(ServerUtilities::new(
            props,
            logger,
            (),
            ContiguousMemoryLayoutPolicy::new(4),
        ));

        Self {
            memory_management,
            utilities,
            timestamps: TimestampProfiler::default(),
            failures: ErrorGraph::default(),
        }
    }

    /// Fails when the buffers `handles` name carry a failure — the claim check
    /// a read makes, through the same [`ErrorGraph::reports`] a real server's
    /// [`StreamPool::ensure_written`] goes through.
    ///
    /// [`StreamPool::ensure_written`]: cubecl_runtime::stream::StreamPool::ensure_written
    fn ensure_written<'a>(
        &self,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<(), ServerError> {
        self.failures.reports(handles.filter_map(|handle| {
            let failure = self
                .memory_management
                .failure(&handle.memory, handle.range())?;
            Some((failure, handle.memory.id()))
        }))
    }

    /// Taint every allocation in `written` with `error`: the work that was
    /// going to write those buffers did not run, so a read of any of them
    /// fails on this failure until something writes them again.
    fn taint<'a>(&mut self, error: ServerError, written: impl Iterator<Item = &'a BufferBinding>) {
        let failure = self.failures.insert(error);
        for handle in written {
            self.memory_management.taint(
                &handle.memory,
                handle.range(),
                failure,
                &mut self.failures,
            );
        }
        // A failure that named no buffer anything still holds has nothing to
        // wait for.
        self.failures.prune(failure);
    }

    /// Utility to create a new buffer and immediately copy contiguous data into it
    fn bind_with_data(&mut self, data: &[u8], handle: Handle, stream_id: StreamId) {
        let strides: Strides = [1].into();
        let shape: Shape = [data.len()].into();

        self.initialize_memory(handle.memory.clone(), handle.size(), stream_id);
        self.write(
            vec![(
                CopyDescriptor::new(handle.binding(), shape, strides, 1),
                Bytes::from_bytes_vec(data.to_vec()),
            )],
            stream_id,
        );
    }
}
