use cubecl_runtime::{TimestampsError, TimestampsResult};
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use super::DummyKernel;
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::server::CubeCount;
use cubecl_runtime::storage::{BindingResource, ComputeStorage};
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{Binding, ComputeServer, Handle},
    storage::BytesStorage,
    ExecutionMode,
};

/// The dummy server is used to test the cubecl-runtime infrastructure.
/// It uses simple memory management with a bytes storage on CPU, without asynchronous tasks.
#[derive(Debug)]
pub struct DummyServer {
    memory_management: MemoryManagement<BytesStorage>,
    timestamps: KernelTimestamps,
}

#[derive(Debug)]
enum KernelTimestamps {
    Inferred { start_time: Instant },
    Disabled,
}

impl KernelTimestamps {
    fn enable(&mut self) {
        if !matches!(self, Self::Disabled) {
            return;
        }

        *self = Self::Inferred {
            start_time: Instant::now(),
        };
    }

    fn disable(&mut self) {
        *self = Self::Disabled;
    }
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
        Handle::new(
            self.memory_management.reserve(size as u64, None),
            None,
            None,
        )
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        bindings: Vec<Binding>,
        _mode: ExecutionMode,
    ) {
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
    fn sync(&mut self) -> impl Future<Output = ()> + 'static {
        async move {}
    }

    #[allow(clippy::manual_async_fn)]
    fn sync_elapsed(&mut self) -> impl Future<Output = TimestampsResult> + 'static {
        let duration = match &mut self.timestamps {
            KernelTimestamps::Inferred { start_time } => {
                let duration = start_time.elapsed();
                *start_time = Instant::now();
                Ok(duration)
            }
            KernelTimestamps::Disabled => Err(TimestampsError::Disabled),
        };

        async move { duration }
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }

    fn enable_timestamps(&mut self) {
        self.timestamps.enable();
    }

    fn disable_timestamps(&mut self) {
        self.timestamps.disable();
    }
}

impl DummyServer {
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self {
            memory_management,
            timestamps: KernelTimestamps::Disabled,
        }
    }
}
