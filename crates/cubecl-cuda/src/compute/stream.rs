use crate::compute::{
    storage::{
        cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryStorage},
        gpu::GpuStorage,
    },
    sync::Fence,
};
use cubecl_common::stream_id::StreamId;
use cubecl_core::MemoryConfiguration;
use cubecl_runtime::{
    memory_management::{MemoryDeviceProperties, MemoryManagement},
    stream::StreamBackend,
};

pub struct Stream {
    pub sys: cudarc::driver::sys::CUstream,
    id: StreamId,
    pub memory_management_gpu: MemoryManagement<GpuStorage>,
    pub memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
}

impl core::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("id", &self.id)
            .field("memory_management_gpu", &self.memory_management_gpu)
            .field("memory_management_cpu", &self.memory_management_cpu)
            .finish()
    }
}

#[derive(new, Debug)]
pub struct CudaStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
}

impl StreamBackend for CudaStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(&self, id: StreamId) -> Self::Stream {
        let stream = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .expect("Can create a new stream.");

        let storage = GpuStorage::new(self.mem_alignment, stream);
        let memory_management_gpu =
            MemoryManagement::from_configuration(storage, &self.mem_props, self.mem_config.clone());
        // We use the same page size and memory pools configuration for CPU pinned memory, since we
        // expect the CPU to have at least the same amount of RAM as GPU memory.
        let memory_management_cpu = MemoryManagement::from_configuration(
            PinnedMemoryStorage::new(),
            &MemoryDeviceProperties {
                max_page_size: self.mem_props.max_page_size,
                alignment: PINNED_MEMORY_ALIGNMENT as u64,
                data_transfer_async: false,
            },
            self.mem_config.clone(),
        );

        Stream {
            sys: stream,
            id,
            memory_management_gpu,
            memory_management_cpu,
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        Fence::new(stream.sys)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        // event.wait_async(stream.sys);
        event.wait_sync();
    }

    fn wait_event_sync(event: Self::Event) {
        event.wait_sync();
    }
}
