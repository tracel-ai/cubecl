use cubecl_common::future::DynFut;
use cubecl_common::{ExecutionMode, benchmark::ProfileDuration};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::KernelMetadata;
use cubecl_runtime::kernel_timestamps::KernelTimestamps;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::server::{BindingWithMeta, Bindings, ProfilingToken};
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
    timestamps: KernelTimestamps,
}

#[derive(Debug, Clone)]
pub struct KernelTask {
    kernel: Arc<dyn DummyKernel>,
}

impl KernelMetadata for KernelTask {
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

impl ComputeServer for DummyServer {
    type Kernel = KernelTask;
    type Storage = BytesStorage;
    type Info = ();
    type Feature = ();

    fn read(&mut self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        let bytes: Vec<_> = bindings
            .into_iter()
            .map(|b| {
                let bytes_handle = self.memory_management.get(b.memory).unwrap();
                self.memory_management.storage().get(&bytes_handle)
            })
            .collect();

        Box::pin(async move { bytes.into_iter().map(|b| b.read().to_vec()).collect() })
    }

    fn read_tensor(&mut self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let bindings = bindings.into_iter().map(|it| it.binding).collect();
        self.read(bindings)
    }

    fn sync(&mut self) -> DynFut<()> {
        Box::pin(async move {})
    }

    fn get_resource(&mut self, binding: Binding) -> BindingResource<BytesResource> {
        let handle = self.memory_management.get(binding.clone().memory).unwrap();
        BindingResource::new(binding, self.memory_management.storage().get(&handle))
    }

    fn create(&mut self, data: &[u8]) -> Handle {
        let handle = self.empty(data.len());
        let resource = self.get_resource(handle.clone().binding());
        let bytes = resource.resource().write();
        for (i, val) in data.iter().enumerate() {
            bytes[i] = *val;
        }

        handle
    }

    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shape: Vec<&[usize]>,
        _elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        data.into_iter()
            .zip(shape)
            .map(|(data, shape)| {
                let rank = shape.len();
                let mut strides = vec![1; rank];
                for i in (0..rank - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                let handle = self.create(data);

                (handle, strides)
            })
            .collect()
    }

    fn empty(&mut self, size: usize) -> Handle {
        Handle::new(
            self.memory_management.reserve(size as u64, None),
            None,
            None,
            size as u64,
        )
    }

    fn empty_tensors(
        &mut self,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        shape
            .into_iter()
            .zip(elem_size)
            .map(|(shape, elem_size)| {
                let rank = shape.len();
                let mut strides = vec![1; rank];
                for i in (0..rank - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                let size = (shape.iter().product::<usize>() * elem_size) as u64;
                let handle =
                    Handle::new(self.memory_management.reserve(size, None), None, None, size);
                (handle, strides)
            })
            .collect()
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: Bindings,
        _mode: ExecutionMode,
        _logger: Arc<ServerLogger>,
    ) {
        let mut resources: Vec<_> = bindings
            .buffers
            .into_iter()
            .map(|b| self.get_resource(b))
            .collect();
        let metadata = self.create(bytemuck::cast_slice(&bindings.metadata.data));
        resources.push(self.get_resource(metadata.binding()));

        let scalars = bindings
            .scalars
            .into_values()
            .map(|s| self.create(s.data()))
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

    fn end_profile(&mut self, token: ProfilingToken) -> ProfileDuration {
        self.timestamps.stop(token)
    }
}

impl DummyServer {
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self {
            memory_management,
            timestamps: KernelTimestamps::default(),
        }
    }
}
