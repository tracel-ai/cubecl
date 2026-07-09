use crate::memory::MetalStorage;
use cubecl_core::{MemoryConfiguration, server::ServerError};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryManagement, MemoryManagementOptions},
    server::Binding,
    stream::EventStreamBackend,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLSharedEvent,
};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Active encoder state for batching multiple kernel dispatches.
pub struct ActiveEncoder {
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Temporary buffers that must stay alive until this encoder's work completes.
    pub temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

/// Installs a completion handler that drops `temporaries` and, on fault, records the
/// Metal error into `errors` (poisoning the stream). `signal_event` is `Some` when the
/// buffer signals an event; it is forced on fault so dependent waiters fail fast.
fn install_completion_handler(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    errors: Arc<Mutex<Vec<ServerError>>>,
    signal_event: Option<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
) {
    if temporaries.is_empty() && signal_event.is_none() {
        return;
    }

    let temporaries = Mutex::new(Some(temporaries));
    let block = block2::RcBlock::new(
        move |cmd_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let _ = temporaries.lock().unwrap().take();

            let cmd_buf = unsafe { cmd_buf.as_ref() };
            if cmd_buf.status() == MTLCommandBufferStatus::Error {
                let reason = match cmd_buf.error() {
                    Some(err) => format!(
                        "Metal command buffer failed: {}",
                        err.localizedDescription()
                    ),
                    None => "Metal command buffer failed with an unknown error".to_string(),
                };
                errors.lock().unwrap().push(ServerError::Generic {
                    reason,
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                });

                // Metal leaves encoded events unsignaled on fault; signal manually
                // so a dependent `wait_sync` fails fast.
                if let Some((event, value)) = &signal_event {
                    event.setSignaledValue(*value);
                }
            }
        },
    );

    // SAFETY: `addCompletedHandler` copies the block, so the pointer need not outlive
    // this call. The raw-pointer form bypasses block2's `Send` bound, but everything the
    // block touches on the Metal completion thread is thread-safe: `Retained` drops via
    // atomic Obj-C `release`, `setSignaledValue` is an atomic write, and the error sink
    // is an `Arc<Mutex<_>>`.
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
    /// GPU command-buffer faults recorded asynchronously by completion handlers; a
    /// non-empty sink poisons the stream (see [`MetalStreamBackend::is_healthy`]).
    pub errors: Arc<Mutex<Vec<ServerError>>>,
    /// When `Some`, device profiling is active on this stream: each work-bearing command
    /// buffer committed during the window is collected here so its GPU timestamps
    /// (`GPUStartTime`/`GPUEndTime`) can be read after completion.
    pub profiling: Option<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
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

    /// Drains GPU command-buffer faults recorded asynchronously by completion handlers.
    pub fn take_errors(&self) -> Vec<ServerError> {
        core::mem::take(&mut self.errors.lock().unwrap())
    }

    /// Waits on a previously submitted command buffer if total queued ops
    /// exceed `max_submitted_ops`, then resets the counter and runs memory cleanup.
    pub fn regulate(&mut self, ops_in_batch: usize) {
        self.submitted_ops += ops_in_batch;

        if self.submitted_ops >= self.max_submitted_ops {
            if let Some(cmd_buf) = self.last_command_buffer.take() {
                (*cmd_buf).waitUntilCompleted();
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            }
            self.submitted_ops = 0;
            self.memory_management.cleanup(false);
        }
    }
}

/// Metal event for synchronization using `MTLSharedEvent`.
#[derive(Clone)]
pub struct MetalEvent {
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub value: u64,
}

// SAFETY: MTLSharedEvent's signaledValue is atomically updated by the GPU.
unsafe impl Send for MetalEvent {}

impl MetalEvent {
    pub fn new(shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>, value: u64) -> Self {
        Self {
            shared_event,
            value,
        }
    }

    /// Check if the event has been signaled (non-blocking).
    pub fn is_complete(&self) -> bool {
        (*self.shared_event).signaledValue() >= self.value
    }

    /// Block until the event is signaled.
    pub fn wait_sync(self) -> Result<(), ServerError> {
        let timeout_ms = 60_000;
        let result = (*self.shared_event).waitUntilSignaledValue_timeoutMS(self.value, timeout_ms);
        if !result {
            return Err(ServerError::Generic {
                reason: "Metal event wait timed out".to_string(),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            });
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
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
                stream.errors.clone(),
                None,
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
        }
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
        // Resolve the configured dynamic-pool strategy here (the server), so
        // `from_configuration` purely honors the config it's handed.
        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone().resolve(&self.mem_props),
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
            errors: Arc::new(Mutex::new(Vec::new())),
            profiling: None,
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
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
                stream.errors.clone(),
                signal,
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
            install_completion_handler(&signal_buffer, Vec::new(), stream.errors.clone(), signal);
            (*signal_buffer).commit();
            signal_buffer
        };

        let ops_in_batch = stream.batch_ops;

        // While profiling, collect command buffers that actually carried dispatches; skip
        // empty signal-only buffers (ops_in_batch == 0) so they don't widen the measured span.
        if ops_in_batch > 0 {
            if let Some(buffers) = stream.profiling.as_mut() {
                buffers.push(command_buffer.clone());
            }
        }

        stream.last_command_buffer = Some(command_buffer);

        stream.batch_ops = 0;
        stream.batch_bytes = 0;

        stream.regulate(ops_in_batch);

        MetalEvent::new(stream.shared_event.clone(), signal_value)
    }

    fn handle_cursor(stream: &Self::Stream, handle: &Binding) -> u64 {
        // The slice cursor the sync logic compares against the origin stream's `last_synced`
        // to decide whether to wait. A freed/reallocated slice falls back to `u64::MAX`,
        // which conservatively forces a wait.
        stream
            .memory_management
            .get_cursor(handle.memory.clone())
            .unwrap_or(u64::MAX)
    }

    fn is_healthy(stream: &Self::Stream) -> bool {
        stream.errors.lock().unwrap().is_empty()
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(stream);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait_sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_stream() -> MetalStream {
        let device = crate::device::default_device().expect("No Metal device found");
        let mem_props = MemoryDeviceProperties {
            max_page_size: (*device).maxBufferLength() as u64,
            alignment: 256,
        };
        let backend = MetalStreamBackend::new(
            device,
            mem_props,
            MemoryConfiguration::default(),
            Arc::new(ServerLogger::default()),
        );
        backend.create_stream()
    }

    /// A populated error sink makes `is_healthy` false, and draining it clears the poison.
    #[test]
    fn error_sink_poisons_is_healthy() {
        let stream = test_stream();
        assert!(MetalStreamBackend::is_healthy(&stream));

        stream.errors.lock().unwrap().push(ServerError::Generic {
            reason: "injected fault".to_string(),
            backtrace: cubecl_common::backtrace::BackTrace::capture(),
        });
        assert!(!MetalStreamBackend::is_healthy(&stream));

        let drained = stream.take_errors();
        assert_eq!(drained.len(), 1);
        assert!(MetalStreamBackend::is_healthy(&stream));
    }
}
