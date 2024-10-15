use cubecl_common::stub::Duration;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::server::CubeCount;
use cubecl_runtime::storage::{BindingResource, ComputeStorage};
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{Binding, ComputeServer, Handle},
    storage::BytesStorage,
    ExecutionMode,
};
use derive_new::new;

use super::DummyKernel;

/// The dummy server is used to test the cubecl-runtime infrastructure.
/// It uses simple memory management with a bytes storage on CPU, without asynchronous tasks.
#[derive(new, Debug)]
pub struct DummyServer {
    memory_management: MemoryManagement<BytesStorage>,
    computation_start: Option<Instant>,
}

impl ComputeServer for DummyServer {
    type Kernel = Arc<dyn DummyKernel>;
    type Storage = BytesStorage;
    type Feature = ();

    fn read(&mut self, binding: Binding) -> impl Future<Output = Vec<u8>> + 'static {
        let bytes_handle = self.memory_management.get(binding.memory);
        let bytes = self.memory_management.storage().get(&bytes_handle);
        async move { bytes.read().to_vec() }
    }

    fn get_resource(&mut self, binding: Binding) -> BindingResource<Self> {
        let handle = self.memory_management.get(binding.clone().memory);
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

    fn empty(&mut self, size: usize) -> Handle {
        Handle::new(self.memory_management.reserve(size, None), None, None)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: Vec<Binding>,
        _mode: ExecutionMode,
    ) {
        if self.computation_start.is_none() {
            self.computation_start = Some(Instant::now());
        }

        let bind_resources = bindings
            .into_iter()
            .map(|binding| self.get_resource(binding))
            .collect::<Vec<_>>();

        let mut resources: Vec<_> = bind_resources.iter().map(|x| x.resource()).collect();

        kernel.compute(&mut resources);
    }

    fn flush(&mut self) {
        // Nothing to do with dummy backend.
    }

    #[allow(clippy::manual_async_fn)]
    fn sync(&mut self) -> impl Future<Output = Duration> + 'static {
        // Nothing to do with dummy backend.
        let duration = self
            .computation_start
            .map(|f| f.elapsed())
            .unwrap_or(Duration::from_secs_f32(0.0));
        self.computation_start = None;
        async move { duration }
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }
}
