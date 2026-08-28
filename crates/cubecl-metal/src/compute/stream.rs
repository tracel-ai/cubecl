use crate::memory::MetalStorage;
use cubecl_core::{MemoryConfiguration, server::ServerError};
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::Mutex;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{ErrorGraph, FailureId, MemoryManagement, MemoryManagementOptions},
    server::BufferBinding,
    stream::{EventStreamBackend, StreamMemory},
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLSharedEvent,
};
use std::ptr::NonNull;
use std::sync::Arc;

/// Active encoder state for batching multiple kernel dispatches.
pub struct ActiveEncoder {
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Temporary buffers that must stay alive until this encoder's work completes.
    pub temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
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

/// Metal stream with its own command queue and memory management.
pub struct MetalStream {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub memory_management: MemoryManagement<MetalStorage>,
    /// Encoder for the current dispatch batch, `None` between batches.
    pub active_encoder: Option<ActiveEncoder>,
    pub batch_ops: usize,
    pub batch_bytes: usize,
    pub shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Next event signal value.
    pub event_counter: u64,
    /// Device-specific batch flush thresholds.
    pub max_ops_per_batch: usize,
    pub max_mb_per_batch: usize,
    /// Ops submitted without a GPU wait, used for back-pressure.
    pub submitted_ops: usize,
    /// Max submitted ops before we wait on the GPU to drain.
    pub max_submitted_ops: usize,
    /// Last committed command buffer, kept alive for back-pressure waits.
    pub last_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// When `Some`, device profiling is active on this stream: each work-bearing command
    /// buffer committed during the window is collected here so its GPU timestamps
    /// (`GPUStartTime`/`GPUEndTime`) can be read after completion.
    pub profiling: Option<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// The first GPU-time fault a completed command buffer reported, sticky
    /// for the stream's life. Shared with every completion handler and every
    /// [`MetalEvent`], whose waits fail on it — see
    /// [`install_completion_handler`] for why the fault lives here and not on
    /// a buffer.
    pub fault: Arc<Mutex<Option<String>>>,
}

impl std::fmt::Debug for MetalStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalStream")
            .field("has_active_encoder", &self.active_encoder.is_some())
            .field("batch_ops", &self.batch_ops)
            .field("batch_bytes", &self.batch_bytes)
            .field("event_counter", &self.event_counter)
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
    /// Returns the active batch encoder, creating one if none is open.
    pub fn get_or_create_encoder(&mut self) -> &mut ActiveEncoder {
        if self.active_encoder.is_none() {
            let command_buffer = (*self.queue)
                .commandBuffer()
                .expect("Failed to create command buffer");

            let encoder = (*command_buffer)
                .computeCommandEncoder()
                .expect("Failed to create compute command encoder");

            self.active_encoder = Some(ActiveEncoder {
                command_buffer,
                encoder,
                temporaries: Vec::new(),
            });
        }

        self.active_encoder.as_mut().unwrap()
    }

    /// Waits on a previously submitted command buffer if total queued ops
    /// exceed `max_submitted_ops`, then resets the counter and runs memory cleanup.
    pub fn regulate(&mut self, ops_in_batch: usize, failures: &mut ErrorGraph) {
        self.submitted_ops += ops_in_batch;

        if self.submitted_ops >= self.max_submitted_ops {
            if let Some(cmd_buf) = self.last_command_buffer.take() {
                (*cmd_buf).waitUntilCompleted();
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            }
            self.submitted_ops = 0;
            self.memory_management.cleanup(false, failures);
        }
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

    pub fn wait_async(self, stream: &mut MetalStream) {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLEvent};

        if std::ptr::eq(
            &*self.shared_event as *const _,
            &*stream.shared_event as *const _,
        ) {
            return;
        }

        if let Some(active) = stream.active_encoder.take() {
            (*active.encoder).endEncoding();
            install_completion_handler(
                &active.command_buffer,
                active.temporaries,
                None,
                stream.fault.clone(),
            );
            (*active.command_buffer).commit();
        }

        let command_buffer = (*stream.queue)
            .commandBuffer()
            .expect("Failed to create command buffer");

        let event_ref: &ProtocolObject<dyn MTLEvent> =
            ProtocolObject::from_ref(&*self.shared_event);
        (*command_buffer).encodeWaitForEvent_value(event_ref, self.value);
        (*command_buffer).commit();
    }
}

/// Backend for creating Metal streams
#[derive(Debug)]
pub struct MetalStreamBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
    /// Programmatic main-GPU pool layout (see
    /// [`ComputeServer::install_memory_pools`](cubecl_runtime::server::ComputeServer::install_memory_pools)):
    /// streams created after it is set build their GPU pools from it instead
    /// of the runtime default.
    gpu_pools_override: Option<MemoryConfiguration>,
}

impl MetalStreamBackend {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        Self {
            device,
            mem_props,
            mem_config,
            logger,
            gpu_pools_override: None,
        }
    }

    /// The layout streams build their main-GPU pools with, and the memory
    /// properties they are resolved against.
    pub(crate) fn gpu_pools(&self) -> (MemoryConfiguration, MemoryDeviceProperties) {
        let config = self
            .gpu_pools_override
            .clone()
            .unwrap_or_else(|| self.mem_config.clone());
        (config, self.mem_props.clone())
    }

    /// Set the main-GPU pool layout for streams created from now on.
    pub(crate) fn set_gpu_pools(&mut self, config: MemoryConfiguration) {
        self.gpu_pools_override = Some(config);
    }
}

impl EventStreamBackend for MetalStreamBackend {
    type Stream = MetalStream;
    type Event = MetalEvent;

    fn create_stream(&self) -> Self::Stream {
        let queue = (*self.device)
            .newCommandQueue()
            .expect("Failed to create command queue");

        let shared_event = (*self.device)
            .newSharedEvent()
            .expect("Failed to create shared event");

        let storage = MetalStorage::new(self.device.clone());

        // The main GPU pool honors the programmatic pool override when one was
        // installed (`install_memory_pools`).
        let (gpu_config, _) = self.gpu_pools();
        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            gpu_config,
            self.logger.clone(),
            MemoryManagementOptions::new("Metal GPU Memory"),
        );

        // Tier batch limits by GPU architecture: the architecture name's last
        // character encodes the tier ('p' phone, 'g' base/pro, 's' max, 'd' ultra).
        let arch = (*self.device).architecture().name().to_string();
        let (max_ops_per_batch, max_mb_per_batch, max_submitted_ops) = match arch.chars().last() {
            Some('s' | 'd') => (50, 50, 512), // max, ultra
            Some('p') => (20, 20, 256),       // phone
            _ => (40, 40, 512),               // base, pro, and unrecognized
        };

        MetalStream {
            device: self.device.clone(),
            queue,
            memory_management,
            active_encoder: None,
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
        }
    }

    fn flush(stream: &mut Self::Stream, failures: &mut ErrorGraph) -> Self::Event {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLEvent};

        stream.event_counter += 1;
        let signal_value = stream.event_counter;

        let signal = Some((stream.shared_event.clone(), signal_value));

        let command_buffer = if let Some(active) = stream.active_encoder.take() {
            (*active.encoder).endEncoding();

            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*stream.shared_event);
            (*active.command_buffer).encodeSignalEvent_value(event_ref, signal_value);

            install_completion_handler(
                &active.command_buffer,
                active.temporaries,
                signal,
                stream.fault.clone(),
            );
            (*active.command_buffer).commit();
            active.command_buffer
        } else {
            let signal_buffer = (*stream.queue)
                .commandBuffer()
                .expect("Failed to create command buffer");

            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*stream.shared_event);
            (*signal_buffer).encodeSignalEvent_value(event_ref, signal_value);
            install_completion_handler(&signal_buffer, Vec::new(), signal, stream.fault.clone());
            (*signal_buffer).commit();
            signal_buffer
        };

        let ops_in_batch = stream.batch_ops;

        // While profiling, collect command buffers that actually carried dispatches; skip
        // empty signal-only buffers (ops_in_batch == 0) so they don't widen the measured span.
        if ops_in_batch > 0
            && let Some(buffers) = stream.profiling.as_mut()
        {
            buffers.push(command_buffer.clone());
        }

        stream.last_command_buffer = Some(command_buffer);

        stream.batch_ops = 0;
        stream.batch_bytes = 0;

        stream.regulate(ops_in_batch, failures);

        MetalEvent::new(
            stream.shared_event.clone(),
            signal_value,
            stream.fault.clone(),
        )
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

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(stream);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait_sync()
    }
}
