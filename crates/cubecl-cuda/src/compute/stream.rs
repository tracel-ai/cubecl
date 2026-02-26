use crate::compute::{
    storage::{
        cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryStorage},
        gpu::GpuStorage,
    },
    sync::Fence,
};
use cubecl_core::{MemoryConfiguration, ir::MemoryDeviceProperties, server::ServerError};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryManagement, MemoryManagementOptions},
    stream::EventStreamBackend,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct Stream {
    pub sys: cudarc::driver::sys::CUstream,
    pub memory_management_gpu: MemoryManagement<GpuStorage>,
    pub memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
    pub errors: Vec<ServerError>,
}

#[derive(new, Debug)]
pub struct CudaStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
    logger: Arc<ServerLogger>,
}

impl EventStreamBackend for CudaStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(&self) -> Self::Stream {
        let stream = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .expect("Can create a new stream.");

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

    fn handle_cursor(stream: &Self::Stream, handle: &cubecl_core::server::HandleBinding) -> u64 {
        let slot = stream.memory_management_gpu.get_slot_ref(handle).unwrap();
        slot.cursor
    }

    fn is_healty(stream: &Self::Stream) -> bool {
        stream.errors.is_empty()
    }
}
