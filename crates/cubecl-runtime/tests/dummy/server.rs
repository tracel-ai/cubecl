use super::DummyKernel;
use crate::dummy::DummyCompiler;
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_ir::{
    DeviceIdentity, DeviceProperties, ElemType, HardwareProperties, MemoryDeviceProperties,
    StorageType, UIntKind, VectorSize, features::Features,
};
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement, MemoryUsage},
    server::{
        Binding, ComputeServer, CopyDescriptor, CubeCount, CubeDim, ExecutionMode, Handle,
        KernelArguments, ProfileError, ProfilingToken, ServerCommunication, ServerError,
        ServerUtilities,
    },
    storage::{BytesResource, BytesStorage, ComputeStorage, ManagedResource},
    timestamp_profiler::TimestampProfiler,
};
use cubecl_zspace::{Shape, Strides};
use std::sync::Arc;

/// The dummy server is used to test the cubecl-runtime infrastructure.
/// It uses simple memory management with a bytes storage on CPU, without asynchronous tasks.
#[derive(Debug)]
pub struct DummyServer {
    memory_management: MemoryManagement<BytesStorage>,
    timestamps: TimestampProfiler,
    utilities: Arc<ServerUtilities<Self>>,
    /// Errors are handled lazily, as on a real server: a failed launch records its error
    /// here, and it surfaces at the next `flush`/`sync`/`end_profile`, each of which
    /// drains the whole queue. Recording happens once, here; tagging the in-flight
    /// profile is the drain's job.
    errors: Vec<ServerError>,
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

    fn address_type(&self) -> cubecl_ir::StorageType {
        ElemType::UInt(UIntKind::U32).into()
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
        cubecl_runtime::kernel::KernelDefinition {
            buffers: Vec::new(),
            tensor_maps: Vec::new(),
            scalars: Vec::new(),
            cube_dim: CubeDim::new_single(),
            body: cubecl_ir::Scope::root(false),
            options: Default::default(),
        }
    }

    fn compile(
        &self,
        _definition: cubecl_runtime::kernel::KernelDefinition,
        _compiler: &mut DummyCompiler,
        _compilation_options: &<DummyCompiler as cubecl_runtime::compiler::Compiler>::CompilationOptions,
        _mode: ExecutionMode,
        _addr_type: StorageType,
    ) -> Result<cubecl_runtime::kernel::CompiledKernel<DummyCompiler>, CompilationError> {
        if let Some(err) = self.kernel.compilation_error() {
            return Err(err);
        }

        Ok(CompiledKernel {
            entrypoint_name: self.kernel.name().to_string(),
            debug_name: None,
            source: String::new(),
            repr: Some(self.clone()),
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
        let reserved = self.memory_management.reserve(size).unwrap();
        self.memory_management
            .bind(reserved, memory.clone(), 0)
            .unwrap();
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
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
            let mut bytes = self.memory_management.storage().get(&storage_h);
            bytes.write()[..data.len()].copy_from_slice(&data);
        }
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let result = self.take_pending_error();
        Box::pin(async move { result })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
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
        mode: ExecutionMode,
        stream_id: StreamId,
        launch_mode: cubecl_runtime::dry_run::LaunchMode,
    ) {
        let kernel = match kernel.compile(
            kernel.define(),
            &mut DummyCompiler,
            &(),
            mode,
            kernel.address_type(),
        ) {
            Ok(kernel) => kernel,
            Err(err) => {
                // Recorded once, in the error queue. Tagging the profiler is the drain's
                // job, exactly as on a real server: a launch failure the queue never gets
                // drained for would otherwise be reported twice.
                let err = cubecl_runtime::server::LaunchError::from(err);
                self.errors.push(err.into());
                return;
            }
        };

        // Compiled above, exactly as a real server does — and, exactly as a
        // real server does, a skipped launch stops before anything touches a
        // buffer, so a dry run's lazily-carved allocations stay unmapped.
        if launch_mode.is_skipped() {
            return;
        }

        let mut resources: Vec<_> = bindings
            .buffers
            .into_iter()
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
    }

    fn flush(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        self.take_pending_error()
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        Ok(self.memory_management.memory_usage())
    }

    fn memory_report(
        &mut self,
        _stream_id: StreamId,
    ) -> Result<cubecl_runtime::memory_management::MemoryReport, ServerError> {
        Ok(self.memory_management.memory_report())
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {
        self.memory_management.cleanup(true);
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        Ok(self.timestamps.start())
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        // The profile boundary drains the queue, like the real servers, which sync here.
        // Errors recorded during the profile belong to it, and leaving them pending would
        // hand them to whatever call comes next.
        if let Err(err) = self.take_pending_error() {
            self.timestamps.error(ProfileError::Server(Box::new(err)));
        }

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
            errors: Vec::new(),
        }
    }

    /// Drains the error queue, tagging any in-flight profile as wrong. Mirrors
    /// `flush_errors` on the cuda and wgpu servers.
    fn flush_errors(&mut self) -> Vec<ServerError> {
        let errors = core::mem::take(&mut self.errors);

        // It is very important to tag current profiles as being wrong.
        if !errors.is_empty() {
            self.timestamps.error(ProfileError::Unknown {
                reason: format!("{errors:?}"),
                backtrace: BackTrace::capture(),
            });
        }

        errors
    }

    /// Drains the error queue and aggregates it, as `flush`/`sync` do on a real server.
    /// Every error surfaces: none is silently dropped, and none is left behind to be
    /// reported by a later, unrelated call.
    fn take_pending_error(&mut self) -> Result<(), ServerError> {
        let errors = self.flush_errors();

        if errors.is_empty() {
            return Ok(());
        }

        Err(ServerError::ServerUnhealthy {
            errors,
            backtrace: BackTrace::capture(),
        })
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
