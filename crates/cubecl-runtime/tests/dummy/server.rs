use cubecl_common::ExecutionMode;
use cubecl_runtime::{
    TimestampsError, TimestampsResult,
    server::{BindingWithMeta, ConstBinding},
};
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

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
    type Info = ();
    type Feature = ();

    fn read(&mut self, bindings: Vec<Binding>) -> impl Future<Output = Vec<Vec<u8>>> + 'static {
        let bytes: Vec<_> = bindings
            .into_iter()
            .map(|b| {
                let bytes_handle = self.memory_management.get(b.memory).unwrap();
                self.memory_management.storage().get(&bytes_handle)
            })
            .collect();

        async move { bytes.into_iter().map(|b| b.read().to_vec()).collect() }
    }

    fn read_tensor(
        &mut self,
        bindings: Vec<BindingWithMeta>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + 'static {
        let bindings = bindings.into_iter().map(|it| it.binding).collect();
        self.read(bindings)
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

    fn create_tensor(
        &mut self,
        data: &[u8],
        shape: &[usize],
        _elem_size: usize,
    ) -> (Handle, Vec<usize>) {
        let rank = shape.len();
        let mut strides = vec![1; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let handle = self.create(data);

        (handle, strides)
    }

    fn empty(&mut self, size: usize) -> Handle {
        Handle::new(
            self.memory_management.reserve(size as u64, None),
            None,
            None,
            size as u64,
        )
    }

    fn empty_tensor(&mut self, shape: &[usize], elem_size: usize) -> (Handle, Vec<usize>) {
        let rank = shape.len();
        let mut strides = vec![1; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let size = (shape.iter().product::<usize>() * elem_size) as u64;
        let handle = Handle::new(self.memory_management.reserve(size, None), None, None, size);
        (handle, strides)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        _count: CubeCount,
        constants: Vec<ConstBinding>,
        bindings: Vec<Binding>,
        _mode: ExecutionMode,
    ) {
        let mut resources = constants
            .into_iter()
            .map(|it| match it {
                ConstBinding::TensorMap { binding, .. } => self.get_resource(binding),
            })
            .collect::<Vec<_>>();
        resources.extend(
            bindings
                .into_iter()
                .map(|binding| self.get_resource(binding)),
        );

        let mut resources: Vec<_> = resources.iter().map(|x| x.resource()).collect();

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

    fn memory_cleanup(&mut self) {
        self.memory_management.cleanup(true);
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
