use cubecl_common::ExecutionMode;
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::server::{
    Bindings, CopyDescriptor, DataTransferService, ProfileError, ProfilingToken,
};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{id::KernelId, server::IoError};
use cubecl_runtime::{
    kernel::KernelMetadata,
    server::{Allocation, AllocationDescriptor},
};
use std::sync::Arc;

use super::DummyKernel;
use cubecl_runtime::memory_management::MemoryUsage;
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

    pub fn compute(&self, resources: &mut [&BytesResource]) {
        self.kernel.compute(resources);
    }
}

impl DataTransferService for DummyServer {}

impl ComputeServer for DummyServer {
    type Kernel = KernelTask;
    type Storage = BytesStorage;
    type Info = ();

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
                    size as u64,
                    0,
                );
                Ok(Allocation::new(handle, strides))
            })
            .collect()
    }

    fn read(&mut self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, IoError>> {
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

    fn write(&mut self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError> {
        for (descriptor, data) in descriptors {
            let resource = self.get_resource(descriptor.binding);
            let bytes = resource.resource().write();
            bytes[..data.len()].copy_from_slice(data);
        }
        Ok(())
    }

    fn sync(&mut self) -> DynFut<()> {
        Box::pin(async move {})
    }

    fn get_resource(&mut self, binding: Binding) -> BindingResource<BytesResource> {
        let handle = self.memory_management.get(binding.clone().memory).unwrap();
        BindingResource::new(binding, self.memory_management.storage().get(&handle))
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: Bindings,
        _mode: ExecutionMode,
        _logger: Arc<ServerLogger>,
        stream_id: StreamId,
    ) {
        let mut resources: Vec<_> = bindings
            .buffers
            .into_iter()
            .map(|b| self.get_resource(b))
            .collect();
        let metadata = self
            .create_with_data(bytemuck::cast_slice(&bindings.metadata.data), stream_id)
            .unwrap();
        resources.push(self.get_resource(metadata.binding()));

        let scalars = bindings
            .scalars
            .into_values()
            .map(|s| self.create_with_data(s.data(), stream_id).unwrap())
            .collect::<Vec<_>>();
        resources.extend(scalars.into_iter().map(|h| self.get_resource(h.binding())));

        let mut resources: Vec<_> = resources.iter().map(|x| x.resource()).collect();

        kernel.compute(&mut resources);
    }

    fn flush(&mut self) {
        // Nothing to do with dummy backend.
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.memory_management.cleanup(true);
    }

    fn start_profile(&mut self) -> ProfilingToken {
        self.timestamps.start()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        self.timestamps.stop(token)
    }

    fn allocation_mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
        self.memory_management.mode(mode)
    }
}

impl DummyServer {
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self {
            memory_management,
            timestamps: TimestampProfiler::default(),
        }
    }
}
