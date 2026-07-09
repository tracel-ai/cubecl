use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{Binding, Handle, ServerError},
};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
        drop_queue::{self, FlushingPolicy, PendingDropQueue},
    },
    metadata_cache::{CacheMode, MetadataCachePolicy, MetadataInfoCache},
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
    pub drop_queue: drop_queue::PendingDropQueue<Fence>,
    /// This stream's position in the graph-capture lifecycle (see
    /// [`StreamCaptureState`]). Enforces the ordered `graph_prepare` →
    /// `begin_capture` → `end_capture` transitions and gates the deferral of
    /// fenced drop-queue flushes while a capture is actively recording.
    pub capturing: StreamCaptureState,
    /// Reusable per-launch info buffers (kernel shapes/strides/scalars), keyed
    /// by kernel and the exact info bytes. Admission and least-recently-used
    /// eviction are decided by the cache's [`MetadataCachePolicy`]; the launch
    /// path sets its [`CacheMode`] from the capture lifecycle, so during graph
    /// capture every buffer is cached and none is evicted mid-capture. See
    /// [`StreamCaptureState::cache_mode`].
    pub info_cache: MetadataInfoCache<Handle>,
}

/// Where a stream sits in the graph-capture lifecycle. Capture is a strict
/// `NoCapture → Prepare → Capture → NoCapture` progression: `graph_prepare`
/// arms the pools (`NoCapture → Prepare`), `begin_capture` opens the recording
/// window (`Prepare → Capture`), and `end_capture` closes it (`Capture →
/// NoCapture`). Every transition rejects an out-of-order call, so a capture can
/// never start unprepared and two captures can never overlap on one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCaptureState {
    /// No capture is prepared or recording.
    NoCapture,
    /// `graph_prepare` has armed the persistent pools and snapshotted them for
    /// the warmup run; `begin_capture` may now open the window.
    Prepare,
    /// `hipStreamBeginCapture` is recording launches. A fenced drop-queue flush
    /// (or any host sync) issued now aborts the capture
    /// (`hipErrorStreamCaptureUnsupported`), so the execution path defers those
    /// flushes until `end_capture`, which reclaims the deferred buffers.
    Capture,
}

impl StreamCaptureState {
    /// Whether launches on the stream are being recorded into a graph right
    /// now — the window during which a host sync would abort the capture.
    pub fn is_recording(&self) -> bool {
        matches!(self, StreamCaptureState::Capture)
    }

    /// The [`CacheMode`] the metadata info cache should run in at this lifecycle
    /// position. Both while a graph is being *prepared* (warmup, which primes
    /// the cache) and while it is being *recorded* the cache runs in
    /// [`CacheMode::Capture`] — caching every buffer and invalidating none — so
    /// the capture window finds every info buffer warm and drops none out from
    /// under a recorded launch. Normal operation uses [`CacheMode::Normal`].
    pub fn cache_mode(&self) -> CacheMode {
        match self {
            StreamCaptureState::NoCapture => CacheMode::Normal,
            StreamCaptureState::Prepare | StreamCaptureState::Capture => CacheMode::Capture,
        }
    }
}

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

        // Resolve the configured dynamic-pool strategy for the main GPU
        // activation pool only (the server does it, so `from_configuration`
        // purely honors the config it's handed). The pinned pool below is left
        // alone: the dynamic-pool override targets GPU activations, and the other
        // pools have deliberate configurations that must not be overridden.
        let memory_management_gpu = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone().resolve(&self.mem_props),
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
            capturing: StreamCaptureState::NoCapture,
            info_cache: MetadataInfoCache::new(MetadataCachePolicy::default()),
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
