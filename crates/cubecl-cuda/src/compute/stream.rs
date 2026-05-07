use crate::compute::{
    storage::{
        cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryStorage},
        gpu::GpuStorage,
    },
    sync::Fence,
};
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{Binding, ServerError},
};
use cubecl_runtime::{
    config::streaming::StreamPriority,
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryManagement, MemoryManagementOptions, drop_queue,
    },
    stream::EventStreamBackend,
};
use std::{mem::MaybeUninit, sync::Arc};

#[derive(Debug)]
pub struct Stream {
    pub sys: cudarc::driver::sys::CUstream,
    pub memory_management_gpu: MemoryManagement<GpuStorage>,
    pub memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
    pub errors: Vec<ServerError>,
    pub drop_queue: drop_queue::PendingDropQueue<Fence>,
}

impl drop_queue::Fence for Fence {
    fn sync(self) {
        let _ = self.wait_sync().ok();
    }
}

#[derive(new, Debug)]
pub struct CudaStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
    logger: Arc<ServerLogger>,
    priority: StreamPriority,
}

/// Create a non-blocking CUDA stream, applying the requested priority hint.
///
/// `StreamPriority::Default` preserves the historical `cuStreamCreate` path so
/// existing users see no change. `Low`/`High` go through
/// `cuStreamCreateWithPriority` using the device's range as queried via
/// `cuCtxGetStreamPriorityRange`. CUDA convention: lower number = higher
/// priority, so the queried `greatest` is numerically smallest (most
/// aggressive) and `least` is numerically largest (least aggressive). On
/// devices without priority support both values are 0 and CUDA silently
/// ignores the priority argument — equivalent to the default path.
///
/// Both calls require a current CUDA context; callers in this crate always
/// set the context before invoking stream creation.
pub(crate) fn create_cuda_stream(priority: StreamPriority) -> cudarc::driver::sys::CUstream {
    use cudarc::driver::sys::{self, CUstream_flags};

    let use_greatest = match priority {
        StreamPriority::Default => {
            return cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .expect("Can create a new stream.");
        }
        StreamPriority::High => true,
        StreamPriority::Low => false,
    };

    // SAFETY: `cuCtxGetStreamPriorityRange` writes through both pointers on
    // success; we only read the locals after the `.expect()` confirms success.
    let value = unsafe {
        let mut least: i32 = 0;
        let mut greatest: i32 = 0;
        sys::cuCtxGetStreamPriorityRange(&mut least, &mut greatest)
            .result()
            .expect("Can query CUDA stream priority range.");
        if use_greatest { greatest } else { least }
    };

    // SAFETY: `cuStreamCreateWithPriority` writes the new stream handle through
    // the out pointer on success; `.expect()` ensures we only `assume_init` on
    // success.
    unsafe {
        let mut stream = MaybeUninit::uninit();
        sys::cuStreamCreateWithPriority(
            stream.as_mut_ptr(),
            CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            value,
        )
        .result()
        .expect("Can create a new CUDA stream with priority.");
        stream.assume_init()
    }
}

impl EventStreamBackend for CudaStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(&self) -> Self::Stream {
        let stream = create_cuda_stream(self.priority);

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
            drop_queue: Default::default(),
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
