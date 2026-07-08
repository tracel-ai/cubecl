use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{Binding, Handle, ServerError},
};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    id::KernelId,
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
        drop_queue::{self, FlushingPolicy, PendingDropQueue},
    },
    stream::EventStreamBackend,
};
use std::collections::HashMap;
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
    pub drop_queue: drop_queue::PendingDropQueue<Fence>,
    /// Set while a graph capture is recording on this stream. A fenced drop-queue
    /// flush (or any host sync) issued between `hipStreamBeginCapture` and
    /// `hipStreamEndCapture` aborts the capture (`hipErrorStreamCaptureUnsupported`),
    /// so the execution path defers those flushes while this is `true`; the
    /// deferred buffers are reclaimed when the capture ends.
    pub capturing: bool,
    /// Reusable per-launch **info buffers** (kernel shapes/strides/scalars),
    /// keyed by kernel and the exact info bytes. A kernel's info depends only on
    /// its shapes and scalar args — not the tensor data pointers, which are
    /// separate kernel arguments — so two launches with identical info can share
    /// one read-only device buffer. Caching them avoids a fresh allocation and a
    /// host→device copy on every launch, and makes captured graphs clean: a
    /// stable-shape decode's launches all hit warm buffers, so nothing is
    /// allocated or copied inside the capture window. Bounded by [`INFO_CACHE_MAX`].
    pub info_cache: HashMap<(KernelId, Vec<u64>), Handle>,
}

/// Cap on [`Stream::info_cache`] entries; past it, launches fall back to a fresh
/// per-launch info buffer (correct, just uncached) so the cache can't grow
/// without bound on workloads with many distinct shapes.
pub const INFO_CACHE_MAX: usize = 4096;

impl drop_queue::Fence for Fence {
    fn sync(self) {
        let _ = self.wait_sync().ok();
    }
}

#[derive(new, Debug)]
pub struct HipStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
    is_integrated: bool,
    logger: Arc<ServerLogger>,
}

impl EventStreamBackend for HipStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(&self) -> Self::Stream {
        // SAFETY: Calling HIP FFI to create a non-blocking stream. The stream handle is
        // initialized by HIP on success (asserted below) and stored for the lifetime of
        // this `Stream`.
        let stream = unsafe {
            let mut stream: cubecl_hip_sys::hipStream_t = std::ptr::null_mut();
            let stream_status = cubecl_hip_sys::hipStreamCreateWithFlags(
                &mut stream,
                cubecl_hip_sys::hipStreamNonBlocking,
            );
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
            PinnedMemoryStorage::new(stream),
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
            capturing: false,
            info_cache: HashMap::new(),
            drop_queue: PendingDropQueue::new(FlushingPolicy {
                max_bytes_count: match self.is_integrated {
                    // Integrated GPUs (APUs) share memory and IOMMU with the CPU.
                    // Flushing more frequently prevents the GPU from reaching 100%
                    // utilization, which avoids transient voltage droops and IOMMU
                    // TLB invalidation races that cause GPU hangs on 0→100% transitions.
                    //
                    // 16 was found empirically to be a good balance between stability
                    // and performance, 32 still exhibited intermittent hangs.
                    //
                    // In practice the performance difference is negligible since integrated
                    // GPUs are typically thermally constrained anyway.
                    true => 16,
                    false => 64,
                },
                ..Default::default()
            }),
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
