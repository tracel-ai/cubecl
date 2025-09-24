use cubecl_core::MemoryConfiguration;
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    memory_management::{MemoryDeviceProperties, MemoryManagement},
    stream::EventStreamBackend,
};

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
}

#[derive(new, Debug)]
pub struct HipStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
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
        let storage = GpuStorage::new(self.mem_alignment);
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
            memory_management_gpu,
            memory_management_cpu,
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        Fence::new(stream.sys)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(stream.sys);
    }

    fn wait_event_sync(event: Self::Event) {
        event.wait_sync();
    }
}
