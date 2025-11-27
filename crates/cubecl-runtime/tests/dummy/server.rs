use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_common::{CubeDim, ExecutionMode};
use cubecl_runtime::compiler::CompilationError;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{DeviceProperties, Features};
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use cubecl_runtime::{id::KernelId, server::IoError};
use cubecl_runtime::{
    kernel::CompiledKernel,
    server::{
        Bindings, CopyDescriptor, ProfileError, ProfilingToken, ServerCommunication,
        ServerUtilities,
    },
};
use cubecl_runtime::{
    kernel::KernelMetadata,
    server::{Allocation, AllocationDescriptor},
};
use std::sync::Arc;

use crate::dummy::DummyCompiler;

use super::DummyKernel;
use cubecl_runtime::memory_management::{
    HardwareProperties, MemoryAllocationMode, MemoryDeviceProperties, MemoryUsage,
};
use cubecl_runtime::server::{CubeCount, LaunchError, RuntimeError};
use cubecl_runtime::storage::{BindingResource, BytesResource, ComputeStorage};
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{Binding, ComputeServer, Handle},
    storage::BytesStorage,
};

/// The dummy server is used to test the cubecl-runtime infrastructure.
/// It uses simple memory management with a bytes storage on CPU, without asynchronous tasks.
#[derive(Debug)]
pub struct DummyServer {
    memory_management: MemoryManagement<BytesStorage>,
    timestamps: TimestampProfiler,
    utilities: Arc<ServerUtilities<Self>>,
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
}

impl core::fmt::Display for KernelTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dummy kernel")
    }
}

impl CubeTask<DummyCompiler> for KernelTask {
    fn compile(
        &self,
        _compiler: &mut DummyCompiler,
        _compilation_options: &<DummyCompiler as cubecl_runtime::compiler::Compiler>::CompilationOptions,
        _mode: ExecutionMode,
    ) -> Result<cubecl_runtime::kernel::CompiledKernel<DummyCompiler>, CompilationError> {
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
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<cubecl_runtime::server::ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        descriptors
            .into_iter()
            .map(|descriptor| {
                let rank = descriptor.shape.len();
                let mut strides = vec![1; rank];
                for i in (0..rank - 1).rev() {
                    strides[i] = strides[i + 1] * descriptor.shape[i + 1];
                }
                let size: usize = descriptor.shape.iter().product();
                let handle = Handle::new(
                    self.memory_management.reserve(size as u64)?,
                    None,
                    None,
                    stream_id,
                    0,
                    size as u64,
                );
                Ok(Allocation::new(handle, strides))
            })
            .collect()
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let bytes: Vec<_> = descriptors
            .into_iter()
            .map(|b| {
                let bytes_handle = self.memory_management.get(b.binding.memory).unwrap();
                self.memory_management.storage().get(&bytes_handle)
            })
            .collect();

        Box::pin(async move {
            Ok(bytes
                .into_iter()
                .map(|b| {
                    let bytes = b.read();
                    Bytes::from_bytes_vec(bytes.to_vec())
                })
                .collect())
        })
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        _stream_id: StreamId,
    ) -> Result<(), IoError> {
        for (descriptor, data) in descriptors {
            let handle = self
                .memory_management
                .get(descriptor.binding.clone().memory)
                .unwrap();

            let mut bytes = self.memory_management.storage().get(&handle);
            bytes.write()[..data.len()].copy_from_slice(&data);
        }
        Ok(())
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<Result<(), RuntimeError>> {
        Box::pin(async move { Ok(()) })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        _stream_id: StreamId,
    ) -> BindingResource<BytesResource> {
        let handle = self.memory_management.get(binding.clone().memory).unwrap();
        BindingResource::new(binding, self.memory_management.storage().get(&handle))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let mut resources: Vec<_> = bindings
            .buffers
            .into_iter()
            .map(|b| self.memory_management.get(b.memory).unwrap())
            .collect();
        let metadata = self
            .create_with_data(bytemuck::cast_slice(&bindings.metadata.data), stream_id)
            .unwrap();
        resources.push(
            self.memory_management
                .get(metadata.binding().memory)
                .unwrap(),
        );

        let scalars = bindings
            .scalars
            .into_values()
            .map(|s| self.create_with_data(s.data(), stream_id).unwrap())
            .collect::<Vec<_>>();
        resources.extend(
            scalars
                .into_iter()
                .map(|h| self.memory_management.get(h.binding().memory).unwrap()),
        );
        let mut resources: Vec<_> = resources
            .iter_mut()
            .map(|x| self.memory_management.storage().get(x))
            .collect();
        let mut resources: Vec<_> = resources.iter_mut().collect();
        let kernel = kernel.compile(&mut DummyCompiler, &(), mode)?;
        kernel.repr.unwrap().compute(resources.as_mut_slice());

        Ok(())
    }

    fn flush(&mut self, _stream_id: StreamId) {
        // Nothing to do with dummy backend.
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> MemoryUsage {
        self.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {
        self.memory_management.cleanup(true);
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> ProfilingToken {
        self.timestamps.start()
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
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
            max_cube_count: CubeCount::new_3d(u16::MAX as u32, u16::MAX as u32, u16::MAX as u32),
            max_units_per_cube: 1024,
            max_cube_dim: CubeDim::new_3d(1024, 1024, 64),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
        };
        let features = Features::default();
        let timing_method = cubecl_common::profile::TimingMethod::System;
        let props = DeviceProperties::new(features, mem_props, hardware, timing_method);
        let logger = Arc::new(ServerLogger::default());

        let utilities = Arc::new(ServerUtilities::new(props, logger, ()));
        Self {
            memory_management,
            utilities,
            timestamps: TimestampProfiler::default(),
        }
    }
}
