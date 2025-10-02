use cubecl_common::ExecutionMode;
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::server::{
    Bindings, CopyDescriptor, ProfileError, ProfilingToken, ServerCommunication,
};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{id::KernelId, server::IoError};
use cubecl_runtime::{
    kernel::KernelMetadata,
    server::{Allocation, AllocationDescriptor},
};
use std::sync::Arc;

use super::DummyKernel;
use cubecl_runtime::memory_management::{MemoryAllocationMode, MemoryUsage};
use cubecl_runtime::server::CubeCount;
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
    logger: Arc<ServerLogger>,
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
    type Kernel = KernelTask;
    type Storage = BytesStorage;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.logger.clone()
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
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        _stream_id: StreamId,
    ) -> Result<(), IoError> {
        for (descriptor, data) in descriptors {
            let handle = self
                .memory_management
                .get(descriptor.binding.clone().memory)
                .unwrap();

            let mut bytes = self.memory_management.storage().get(&handle);
            bytes.write()[..data.len()].copy_from_slice(data);
        }
        Ok(())
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<()> {
        Box::pin(async move {})
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        _stream_id: StreamId,
    ) -> BindingResource<BytesResource> {
        let handle = self.memory_management.get(binding.clone().memory).unwrap();
        BindingResource::new(binding, self.memory_management.storage().get(&handle))
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: Bindings,
        _mode: ExecutionMode,
        stream_id: StreamId,
    ) {
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
        kernel.compute(resources.as_mut_slice());
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
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self {
            logger: Arc::new(ServerLogger::default()),
            memory_management,
            timestamps: TimestampProfiler::default(),
        }
    }
}
