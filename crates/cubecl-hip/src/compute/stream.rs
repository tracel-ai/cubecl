use cubecl_core::{
    MemoryConfiguration,
    bytes::Bytes,
    ir::MemoryDeviceProperties,
    server::{Binding, ServerError},
};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryManagement, MemoryManagementOptions},
    stream::EventStreamBackend,
};
use std::sync::Arc;

use crate::compute::{
    cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryStorage},
    fence::Fence,
    gpu::GpuStorage,
};

#[derive(Debug)]
pub struct Stream {
    pub(crate) sys: cubecl_hip_sys::hipStream_t,
    pub memory_management_gpu: MemoryManagement<GpuStorage>,
    pub memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
    pub errors: Vec<ServerError>,
    pub cleaner: BytesCleaner,
}

#[derive(Default)]
pub struct BytesCleaner {
    fence: Option<Fence>,
    io_tasks_old: Vec<Bytes>,
    io_tasks_next: Vec<Bytes>,
}

impl BytesCleaner {
    pub fn push(&mut self, bytes: Bytes) {
        self.io_tasks_next.push(bytes);
    }

    pub fn should_clean(&self) -> bool {
        if self.io_tasks_next.len() >= 32 {
            true
        } else {
            false
        }
    }

    pub fn clean<F: Fn() -> Fence>(&mut self, fence_new: F) {
        if let Some(fence) = self.fence.take() {
            match fence.wait_sync() {
                Ok(_) => {
                    self.io_tasks_old.clear();
                }
                Err(_) => return,
            }
        }

        if !self.io_tasks_old.is_empty() {
            match fence_new().wait_sync() {
                Ok(_) => {
                    self.io_tasks_old.clear();
                }
                Err(_) => return,
            }
        }

        core::mem::swap(&mut self.io_tasks_old, &mut self.io_tasks_next);
        self.fence = Some(fence_new());
    }
}

impl core::fmt::Debug for BytesCleaner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BytesCleaner")
            .field("io_tasks_current", &self.io_tasks_old)
            .field("io_tasks_next", &self.io_tasks_next)
            .finish()
    }
}

#[derive(new, Debug)]
pub struct HipStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
    logger: Arc<ServerLogger>,
}

impl EventStreamBackend for HipStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(&self) -> Self::Stream {
        let stream = unsafe {
            let mut stream: cubecl_hip_sys::hipStream_t = std::ptr::null_mut();
            let stream_status = cubecl_hip_sys::hipStreamCreate(&mut stream);
            assert_eq!(stream_status, HIP_SUCCESS, "Should create a stream");
            stream
        };
        let storage = GpuStorage::new(self.mem_alignment, stream);
        let memory_management_gpu = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );
        // We use the same page size and memory pools configuration for CPU pinned memory, since we
        // expect the CPU to have at least the same amount of RAM as GPU memory.
        let memory_management_cpu = MemoryManagement::from_configuration(
            PinnedMemoryStorage::new(),
            &MemoryDeviceProperties {
                max_page_size: self.mem_props.max_page_size,
                alignment: PINNED_MEMORY_ALIGNMENT as u64,
            },
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Pinned CPU Memory").mode(MemoryAllocationMode::Auto),
        );

        Stream {
            sys: stream,
            memory_management_gpu,
            memory_management_cpu,
            errors: Vec::new(),
            cleaner: Default::default(),
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        Fence::new(stream.sys)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(stream.sys);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait_sync()
    }

    fn handle_cursor(stream: &Self::Stream, binding: &Binding) -> u64 {
        stream
            .memory_management_gpu
            .get_cursor(binding.memory.clone())
            .unwrap()
    }

    fn is_healthy(stream: &Self::Stream) -> bool {
        stream.errors.is_empty()
    }
}
