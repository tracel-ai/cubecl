use super::DummyKernel;
use crate::dummy::DummyCompiler;
use cubecl_common::{bytes::Bytes, future::DynFut, profile::ProfileDuration, stream_id::StreamId};
use cubecl_ir::{
    DeviceProperties, ElemType, HardwareProperties, LineSize, MemoryDeviceProperties, StorageType,
    UIntKind, features::Features,
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryHandle, MemoryManagement, MemoryUsage, create_buffers,
    },
    server::{
        AllocationDescriptor, AllocationKind, Bindings, Buffer, ComputeServer, CopyDescriptor,
        CubeCount, CubeDim, ExecutionMode, Handle, IoError, LaunchError, NaiveAllocator,
        ProfileError, ProfilingToken, ServerAllocator, ServerCommunication, ServerError,
        ServerUtilities,
    },
    storage::{BindingResource, BytesResource, BytesStorage, ComputeStorage},
    timestamp_profiler::TimestampProfiler,
};
use std::sync::Arc;

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
    fn compile(
        &self,
        _compiler: &mut DummyCompiler,
        _compilation_options: &<DummyCompiler as cubecl_runtime::compiler::Compiler>::CompilationOptions,
        _mode: ExecutionMode,
        _addr_type: StorageType,
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
    type Allocator = NaiveAllocator;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<cubecl_runtime::server::ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn create(&mut self, handles: Vec<Handle>, stream_id: StreamId) {
        handles
            .into_iter()
            .map(|handle| {
                let size = handle.size_in_used();
                let reserved = self.memory_management.reserve(size).unwrap();
                let buffer = Buffer {
                    memory: reserved,
                    offset_start: None,
                    offset_end: None,
                    cursor: 0,
                    stream: stream_id,
                    size: size,
                };
                self.memory_management.map(handle.id, buffer);
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
                let slice_handle = self.memory_management.resolve(b.handle).unwrap();
                let bytes_handle = self
                    .memory_management
                    .get(slice_handle.memory.binding())
                    .unwrap();
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

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, _stream_id: StreamId) {
        for (descriptor, data) in descriptors {
            let slice_handle = self.memory_management.resolve(descriptor.handle).unwrap();
            let handle = self
                .memory_management
                .get(slice_handle.memory.binding())
                .unwrap();

            let mut bytes = self.memory_management.storage().get(&handle);
            bytes.write()[..data.len()].copy_from_slice(&data);
        }
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        Box::pin(async move { Ok(()) })
    }

    fn get_resource(
        &mut self,
        handle: Handle,
        _stream_id: StreamId,
    ) -> BindingResource<BytesResource> {
        let slice_handle = self.memory_management.resolve(handle.clone()).unwrap();
        let resource_handle = self
            .memory_management
            .get(slice_handle.memory.binding())
            .unwrap();
        BindingResource::new(
            handle,
            self.memory_management.storage().get(&resource_handle),
        )
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let mut resources: Vec<_> = bindings
            .buffers
            .into_iter()
            .map(|b| {
                let memory = self.memory_management.resolve(b).unwrap();
                self.memory_management.get(memory.memory.binding()).unwrap()
            })
            .collect();
        let data = bytemuck::cast_slice(&bindings.metadata.data);
        let alloc = self.utilities.allocator.alloc(
            stream_id,
            &AllocationDescriptor {
                kind: AllocationKind::Contiguous,
                shape: [data.len()].into(),
                elem_size: 1,
            },
        );
        let metadata = alloc.handle;
        self.create_with_data(data, metadata.clone(), stream_id);

        resources.push({
            let handle = self.memory_management.resolve(metadata).unwrap();
            self.memory_management.get(handle.memory.binding()).unwrap()
        });

        let scalars = bindings
            .scalars
            .into_values()
            .map(|s| {
                let data = s.data();
                let alloc = self.utilities.allocator.alloc(
                    stream_id,
                    &AllocationDescriptor {
                        kind: AllocationKind::Contiguous,
                        shape: [data.len()].into(),
                        elem_size: 1,
                    },
                );
                self.create_with_data(data, alloc.handle.clone(), stream_id);
                alloc.handle
            })
            .collect::<Vec<_>>();

        resources.extend(scalars.into_iter().map(|h| {
            let buffer = self.memory_management.resolve(h).unwrap();
            self.memory_management.get(buffer.memory.binding()).unwrap()
        }));

        let mut resources: Vec<_> = resources
            .iter_mut()
            .map(|x| self.memory_management.storage().get(x))
            .collect();
        let mut resources: Vec<_> = resources.iter_mut().collect();
        let kernel = kernel
            .compile(&mut DummyCompiler, &(), mode, kernel.address_type())
            .unwrap();
        kernel.repr.unwrap().compute(resources.as_mut_slice());
    }

    fn flush(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        // Nothing to do with dummy backend.
        Ok(())
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> MemoryUsage {
        self.memory_management.memory_usage()
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
        self.timestamps.stop(token)
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, _stream_id: StreamId) {
        self.memory_management.mode(mode)
    }

    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError> {
        Vec::new()
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
            max_line_size: LineSize::MAX,
        };
        let features = Features::default();
        let timing_method = cubecl_common::profile::TimingMethod::System;
        let props = DeviceProperties::new(features, mem_props, hardware, timing_method);
        let logger = Arc::new(ServerLogger::default());

        let utilities = Arc::new(ServerUtilities::new(props, logger, (), NaiveAllocator));
        Self {
            memory_management,
            utilities,
            timestamps: TimestampProfiler::default(),
        }
    }
}
