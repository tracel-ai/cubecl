use crate::compute::capture::RecordedDispatch;
use crate::compute::driver::MetalBindings;
use crate::compute::pipelines::MetalCompiledKernel;
use crate::memory::MetalStorage;
use cubecl_core::{
    MemoryConfiguration,
    server::{DeviceCaptures, Handle, IoError, LaunchError, ServerError},
};
use cubecl_environment::sync::Mutex;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, FailureId, MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
        drop_queue::PendingDropQueue,
    },
    metadata_cache::{MetadataCachePolicy, MetadataInfoCache},
    server::BufferBinding,
    storage::BytesStorage,
    stream::{EventStreamBackend, StreamCapture, StreamMemory},
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLEvent, MTLResourceOptions,
    MTLSharedEvent, MTLSize,
};
use std::cell::{RefCell, RefMut};
use std::ptr::NonNull;
use std::sync::Arc;

/// The most metadata a dispatch hands the encoder inline; Metal's limit for
/// `setBytes`. Anything larger rides a buffer of its own.
const MAX_INLINE_INFO_BYTES: usize = 4096;

/// The pass a batch's command buffer has open.
enum Pass {
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

impl Pass {
    fn end(self) {
        match self {
            Pass::Compute(encoder) => encoder.endEncoding(),
            Pass::Blit(encoder) => encoder.endEncoding(),
        }
    }
}

/// The command buffer a stream is filling, and the pass open on it.
pub struct Batch {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// `None` until something is encoded.
    pass: Option<Pass>,
    /// Temporary buffers that must stay alive until this batch's work completes.
    temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Batch {
    fn open(queue: &ProtocolObject<dyn MTLCommandQueue>) -> Self {
        Self {
            command_buffer: queue
                .commandBuffer()
                .expect("Failed to create command buffer"),
            pass: None,
            temporaries: Vec::new(),
        }
    }

    /// The batch's compute pass, opened in place of whatever pass is open.
    pub fn compute(&mut self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        if !matches!(self.pass, Some(Pass::Compute(_))) {
            self.end_pass();
            let encoder = self
                .command_buffer
                .computeCommandEncoder()
                .expect("Failed to create compute command encoder");
            self.pass = Some(Pass::Compute(encoder));
        }
        match &self.pass {
            Some(Pass::Compute(encoder)) => encoder,
            _ => unreachable!("the compute pass was opened above"),
        }
    }

    /// The batch's blit pass, opened in place of whatever pass is open.
    pub fn blit(&mut self) -> &ProtocolObject<dyn MTLBlitCommandEncoder> {
        if !matches!(self.pass, Some(Pass::Blit(_))) {
            self.end_pass();
            let encoder = self
                .command_buffer
                .blitCommandEncoder()
                .expect("Failed to create blit command encoder");
            self.pass = Some(Pass::Blit(encoder));
        }
        match &self.pass {
            Some(Pass::Blit(encoder)) => encoder,
            _ => unreachable!("the blit pass was opened above"),
        }
    }

    fn end_pass(&mut self) {
        if let Some(pass) = self.pass.take() {
            pass.end();
        }
    }

    /// End the open pass and commit the command buffer, which signals
    /// `signal` once it has run.
    fn commit(
        mut self,
        signal: Option<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
        fault: Arc<Mutex<Option<String>>>,
    ) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.end_pass();
        if let Some((event, value)) = &signal {
            let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&**event);
            self.command_buffer.encodeSignalEvent_value(event, *value);
        }
        install_completion_handler(&self.command_buffer, self.temporaries, signal, fault);
        self.command_buffer.commit();
        self.command_buffer
    }
}

/// Installs a completion handler that drops `temporaries` and, on a failed
/// command buffer, records the fault on the stream's sticky slot. `signal_event`
/// is `Some` when the buffer signals an event; it is forced on failure so
/// dependent waiters return promptly — and fail, because every wait checks the
/// fault slot after its event lands.
fn install_completion_handler(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    signal_event: Option<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
    fault: Arc<Mutex<Option<String>>>,
) {
    let temporaries = Mutex::new(Some(temporaries));
    let block = block2::RcBlock::new(
        move |cmd_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let _ = temporaries.lock().take();

            let cmd_buf = unsafe { cmd_buf.as_ref() };
            if cmd_buf.status() == MTLCommandBufferStatus::Error {
                let reason = match cmd_buf.error() {
                    Some(err) => format!(
                        "Metal command buffer failed: {}",
                        err.localizedDescription()
                    ),
                    None => "Metal command buffer failed with an unknown error".to_string(),
                };
                log::warn!("{reason}");

                // A fault at execution time can name no buffer: the work's
                // claims were released at enqueue — a claim covers enqueue,
                // not execution — and this handler holds only the event and
                // the staging temporaries. So the fault is recorded on the
                // stream instead, sticky, and every later wait on it fails:
                // reads report the fault instead of returning garbage, and
                // writes taint their destinations with it. First fault wins,
                // and none is ever cleared — clearing is exactly how stale
                // bytes would start reading clean again.
                let mut slot = fault.lock();
                if slot.is_none() {
                    *slot = Some(reason);
                }

                // Metal leaves encoded events unsignaled on fault; signal
                // manually so a dependent wait returns promptly — with the
                // fault recorded above.
                if let Some((event, value)) = &signal_event {
                    event.setSignaledValue(*value);
                }
            }
        },
    );

    // SAFETY: `addCompletedHandler` copies the block, so the pointer need not outlive
    // this call. The raw-pointer form bypasses block2's `Send` bound, but everything the
    // block touches on the Metal completion thread is thread-safe: `Retained` drops via
    // atomic Obj-C `release`, `setSignaledValue` is an atomic write, the temporaries are
    // behind a `Mutex`, and `log` is `Sync`.
    unsafe {
        command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&block) as *mut _);
    }
}

/// A stream's command queue: the batch it is filling, and what its committed
/// work signals.
pub struct MetalQueue {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The batch being filled, `None` between batches.
    batch: Option<Batch>,
    batch_ops: usize,
    batch_bytes: usize,
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Next event signal value.
    event_counter: u64,
    /// Device-specific batch flush thresholds.
    max_ops_per_batch: usize,
    max_mb_per_batch: usize,
    /// Ops submitted without a GPU wait, used for back-pressure.
    submitted_ops: usize,
    /// Max submitted ops before we wait on the GPU to drain.
    max_submitted_ops: usize,
    /// Last committed command buffer, kept alive for back-pressure waits.
    last_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// When `Some`, device profiling is active on this stream: each work-bearing command
    /// buffer committed during the window is collected here so its GPU timestamps
    /// (`GPUStartTime`/`GPUEndTime`) can be read after completion.
    pub profiling: Option<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// The first GPU-time fault a completed command buffer reported, sticky
    /// for the stream's life. Shared with every completion handler and every
    /// [`MetalEvent`], whose waits fail on it — see
    /// [`install_completion_handler`] for why the fault lives here and not on
    /// a buffer.
    fault: Arc<Mutex<Option<String>>>,
}

impl MetalQueue {
    /// The device the queue submits to.
    pub fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        self.queue.device()
    }

    /// The batch being filled, opened when none is.
    pub fn batch(&mut self) -> &mut Batch {
        self.batch.get_or_insert_with(|| Batch::open(&self.queue))
    }

    /// Encode one dispatch of `kernel` over `count` threadgroups, given
    /// `bindings`.
    ///
    /// # Errors
    ///
    /// The device's refusal of a buffer for metadata too large to hand the
    /// encoder inline.
    pub fn dispatch(
        &mut self,
        kernel: &MetalCompiledKernel,
        count: (u32, u32, u32),
        bindings: &MetalBindings,
    ) -> Result<(), LaunchError> {
        let info: &[u8] = bytemuck::cast_slice(&bindings.info);
        let info_buffer = match info.len() > MAX_INLINE_INFO_BYTES {
            true => Some(self.info_buffer(info)?),
            false => None,
        };
        let info_index = bindings.buffers.len();

        let batch = self.batch();
        let encoder = batch.compute();
        encoder.setComputePipelineState(&kernel.pipeline);
        for (index, resource) in bindings.buffers.iter().enumerate() {
            // SAFETY: the resource is a live slice of its buffer, which the
            // batch's command buffer retains until it completes.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(resource.buffer()), resource.offset(), index)
            };
        }
        match &info_buffer {
            Some(buffer) => unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, info_index) },
            None if !info.is_empty() => unsafe {
                // SAFETY: `setBytes` copies the bytes into the command buffer
                // before it returns.
                encoder.setBytes_length_atIndex(
                    NonNull::new(info.as_ptr() as *mut _).expect("a slice is never null"),
                    info.len(),
                    info_index,
                )
            },
            None => {}
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: count.0 as usize,
                height: count.1 as usize,
                depth: count.2 as usize,
            },
            kernel.threads_per_threadgroup(),
        );
        batch.temporaries.extend(info_buffer);

        let bytes = bindings.buffers.iter().map(|resource| resource.size()).sum();
        self.encoded(1, bytes);
        Ok(())
    }

    /// A buffer holding `info`, for metadata too large to hand the encoder
    /// inline.
    fn info_buffer(
        &self,
        info: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, LaunchError> {
        // SAFETY: the pointer and length are a live slice, which the device
        // copies into the new buffer.
        unsafe {
            self.device().newBufferWithBytes_length_options(
                NonNull::new(info.as_ptr() as *mut _).expect("a slice is never null"),
                info.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| LaunchError::Unknown {
            reason: format!(
                "failed to allocate a {} B Metal buffer for the kernel's metadata",
                info.len()
            ),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        })
    }

    /// Count `ops` dispatches touching `bytes` of memory toward the open
    /// batch, and submit it once it is past the device's thresholds.
    pub fn encoded(&mut self, ops: usize, bytes: usize) {
        self.batch_ops += ops;
        self.batch_bytes += bytes;

        if self.batch_ops > self.max_ops_per_batch
            || (self.batch_bytes >> 20) > self.max_mb_per_batch
        {
            self.flush();
        }
    }

    /// Commit the open batch — or an empty command buffer when none is open —
    /// signalling an event once everything submitted so far has run.
    pub fn flush(&mut self) -> MetalEvent {
        self.event_counter += 1;
        let signal_value = self.event_counter;
        let signal = Some((self.shared_event.clone(), signal_value));

        let batch = match self.batch.take() {
            Some(batch) => batch,
            None => Batch::open(&self.queue),
        };
        let command_buffer = batch.commit(signal, self.fault.clone());

        let ops_in_batch = self.batch_ops;

        // While profiling, collect command buffers that actually carried dispatches; skip
        // empty signal-only buffers (ops_in_batch == 0) so they don't widen the measured span.
        if ops_in_batch > 0
            && let Some(buffers) = self.profiling.as_mut()
        {
            buffers.push(command_buffer.clone());
        }

        self.last_command_buffer = Some(command_buffer);

        self.batch_ops = 0;
        self.batch_bytes = 0;

        self.regulate(ops_in_batch);

        MetalEvent::new(
            self.shared_event.clone(),
            signal_value,
            self.fault.clone(),
        )
    }

    /// Commit the open batch and wait until everything submitted has run.
    ///
    /// # Errors
    ///
    /// The fault the wait reveals, when the stream failed at execution time.
    pub fn drain(&mut self) -> Result<(), ServerError> {
        self.flush().wait_sync()
    }

    /// Waits on a previously submitted command buffer if total queued ops
    /// exceed `max_submitted_ops`, then resets the counter. The memory is
    /// cleaned up by its reservations, not here.
    fn regulate(&mut self, ops_in_batch: usize) {
        self.submitted_ops += ops_in_batch;

        if self.submitted_ops >= self.max_submitted_ops {
            if let Some(cmd_buf) = self.last_command_buffer.take() {
                (*cmd_buf).waitUntilCompleted();
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            }
            self.submitted_ops = 0;
        }
    }
}

/// A stream's queue as the shared command names it: copyable, so a fence can
/// be recorded while the stream's own fields are borrowed.
#[derive(Clone, Copy)]
pub struct QueueRef(NonNull<RefCell<MetalQueue>>);

impl QueueRef {
    /// The queue, borrowed for one call.
    pub fn get(&self) -> RefMut<'_, MetalQueue> {
        // SAFETY: a `QueueRef` is only made by `MetalStream::signal` and used
        // within the command that took it, while the stream — whose box holds
        // the queue — is alive and stays where it is.
        unsafe { self.0.as_ref() }.borrow_mut()
    }
}

/// Metal stream with its own command queue and memory management.
pub struct MetalStream {
    /// Boxed so its address is stable for the [`QueueRef`]s handed out, and
    /// behind a cell because the shared command reaches it through them.
    queue: Box<RefCell<MetalQueue>>,
    pub memory_management: MemoryManagement<MetalStorage>,
    /// The host memory reads are copied into.
    pub host_memory: MemoryManagement<BytesStorage>,
    pub drop_queue: PendingDropQueue<MetalEvent>,
    /// This stream's graph capture (see [`StreamCapture`]): its position in
    /// the lifecycle, and the memory its recorded launches were given.
    pub capturing: StreamCapture,
    /// Always empty: a dispatch hands its metadata to the encoder, and a
    /// recorded one to the graph it seals into, so no launch builds a
    /// metadata buffer to cache. Held because the capture window seals and
    /// releases what a backend caches through it.
    pub info_cache: MetadataInfoCache<Handle>,
    /// The dispatches the open capture window has recorded, in order.
    pub recording: Vec<RecordedDispatch>,
}

impl std::fmt::Debug for MetalStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let queue = self.queue();
        f.debug_struct("MetalStream")
            .field("has_batch", &queue.batch.is_some())
            .field("batch_ops", &queue.batch_ops)
            .field("batch_bytes", &queue.batch_bytes)
            .field("event_counter", &queue.event_counter)
            .field("recording", &self.recording.len())
            .finish()
    }
}

impl StreamMemory for MetalStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory_management
            .failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.memory_management
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory_management
            .written(&binding.memory, binding.range(), failures)
    }
}

impl MetalStream {
    /// The stream's queue, borrowed for one call.
    pub fn queue(&self) -> RefMut<'_, MetalQueue> {
        self.queue.borrow_mut()
    }

    /// The stream's queue, as the shared command names it.
    pub fn queue_ref(&self) -> QueueRef {
        QueueRef(NonNull::from(&*self.queue))
    }

    /// Refuse `operation` while the stream records a graph.
    ///
    /// A recorded graph holds dispatches and nothing else, so a host copy, a
    /// profile or a replay inside the window has no recorded form and must
    /// not silently run once instead of on every replay.
    ///
    /// # Errors
    ///
    /// Fails while [`StreamCapture::is_recording`] holds, naming `operation`.
    pub fn reject_while_recording(&self, operation: &str) -> Result<(), ServerError> {
        if !self.capturing.is_recording() {
            return Ok(());
        }
        Err(ServerError::graph_state(format!(
            "{operation}: a Metal capture window records dispatches only, so this operation \
             cannot be part of a graph"
        )))
    }

    /// Ready the stream's memory for the host to copy into or out of: refused
    /// inside a capture window, and otherwise once everything submitted to
    /// the stream has run.
    ///
    /// # Errors
    ///
    /// The refusal while recording, and the fault the wait reveals.
    pub fn host_access(&self, operation: &str) -> Result<(), IoError> {
        self.reject_while_recording(operation)
            .and_then(|_| self.queue().drain())
            .map_err(|err| IoError::Unknown {
                description: err.to_string(),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            })
    }
}

/// Metal event for synchronization using `MTLSharedEvent`.
#[derive(Clone)]
pub struct MetalEvent {
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub value: u64,
    /// The stream's sticky fault slot, checked after every wait: a forced
    /// event completes the wait, and this is what fails it.
    fault: Arc<Mutex<Option<String>>>,
}

// SAFETY: MTLSharedEvent's signaledValue is atomically updated by the GPU.
unsafe impl Send for MetalEvent {}

impl MetalEvent {
    pub fn new(
        shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        value: u64,
        fault: Arc<Mutex<Option<String>>>,
    ) -> Self {
        Self {
            shared_event,
            value,
            fault,
        }
    }

    /// Check if the event has been signaled (non-blocking).
    pub fn is_complete(&self) -> bool {
        (*self.shared_event).signaledValue() >= self.value
    }

    /// Block until the event is signaled.
    ///
    /// # Errors
    ///
    /// A timeout, and the stream's recorded GPU-time fault: a faulted command
    /// buffer force-signals its event so the wait itself returns, and this
    /// check is what turns that into the failure every dependent caller has
    /// to hear — a read reports it instead of returning garbage, a write
    /// taints its destinations with it.
    pub fn wait_sync(self) -> Result<(), ServerError> {
        let timeout_ms = 60_000;
        let result = (*self.shared_event).waitUntilSignaledValue_timeoutMS(self.value, timeout_ms);
        if !result {
            return Err(ServerError::Generic {
                reason: "Metal event wait timed out".to_string(),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            });
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        if let Some(reason) = self.fault.lock().clone() {
            return Err(ServerError::Generic {
                reason: format!("the Metal stream faulted at execution time: {reason}"),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            });
        }
        Ok(())
    }

    /// Make the work `queue` submits from here wait for this event.
    pub fn wait_async(self, queue: &mut MetalQueue) {
        if std::ptr::eq(
            &*self.shared_event as *const _,
            &*queue.shared_event as *const _,
        ) {
            return;
        }

        if let Some(batch) = queue.batch.take() {
            queue.last_command_buffer = Some(batch.commit(None, queue.fault.clone()));
        }

        let command_buffer = queue
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        let event_ref: &ProtocolObject<dyn MTLEvent> =
            ProtocolObject::from_ref(&*self.shared_event);
        command_buffer.encodeWaitForEvent_value(event_ref, self.value);
        command_buffer.commit();
    }
}

impl cubecl_server::memory_management::drop_queue::Fence for MetalEvent {
    fn wait(self) -> Result<(), ServerError> {
        self.wait_sync()
    }
}

/// Backend for creating Metal streams
#[derive(Debug)]
pub struct MetalStreamBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
    /// The device's captures, which every stream this creates takes its capture state from.
    captures: DeviceCaptures,
}

impl MetalStreamBackend {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        captures: DeviceCaptures,
    ) -> Self {
        Self {
            device,
            mem_props,
            mem_config,
            logger,
            captures,
        }
    }
}

impl EventStreamBackend for MetalStreamBackend {
    type Stream = MetalStream;
    type Event = MetalEvent;

    fn create_stream(&self) -> Result<Self::Stream, ServerError> {
        let queue = (*self.device)
            .newCommandQueue()
            .expect("Failed to create command queue");

        let shared_event = (*self.device)
            .newSharedEvent()
            .expect("Failed to create shared event");

        let storage = MetalStorage::new(self.device.clone());

        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Metal GPU Memory"),
        );
        // Reads land in host memory pooled the way the device's is.
        let host_memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties::new(
                self.mem_props.max_page_size,
                BytesStorage::ALIGNMENT as u64,
            ),
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Metal Host Memory").mode(MemoryAllocationMode::Auto),
        );

        // Tier batch limits by GPU architecture: the architecture name's last
        // character encodes the tier ('p' phone, 'g' base/pro, 's' max, 'd' ultra).
        let arch = (*self.device).architecture().name().to_string();
        let (max_ops_per_batch, max_mb_per_batch, max_submitted_ops) = match arch.chars().last() {
            Some('s' | 'd') => (50, 50, 512), // max, ultra
            Some('p') => (20, 20, 256),       // phone
            _ => (40, 40, 512),               // base, pro, and unrecognized
        };

        Ok(MetalStream {
            queue: Box::new(RefCell::new(MetalQueue {
                queue,
                batch: None,
                batch_ops: 0,
                batch_bytes: 0,
                shared_event,
                event_counter: 0,
                max_ops_per_batch,
                max_mb_per_batch,
                submitted_ops: 0,
                max_submitted_ops,
                last_command_buffer: None,
                profiling: None,
                fault: Arc::new(Mutex::new(None)),
            })),
            memory_management,
            host_memory,
            drop_queue: Default::default(),
            capturing: StreamCapture::new(&self.captures),
            info_cache: MetadataInfoCache::new(MetadataCachePolicy::default()),
            recording: Vec::new(),
        })
    }

    fn flush(stream: &mut Self::Stream, _failures: &mut ErrorGraph) -> Self::Event {
        stream.queue().flush()
    }

    fn handle_cursor(stream: &Self::Stream, handle: &BufferBinding) -> u64 {
        // The slice cursor the sync logic compares against the origin stream's `last_synced`
        // to decide whether to wait. A freed/reallocated slice falls back to `u64::MAX`,
        // which conservatively forces a wait.
        stream
            .memory_management
            .get_cursor(handle.memory.clone())
            .unwrap_or(u64::MAX)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) -> Result<(), ServerError> {
        event.wait_async(&mut stream.queue());
        Ok(())
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait_sync()
    }
}
