use crate::memory::MetalStorage;
use cubecl_core::{server::ExecutionError, MemoryConfiguration};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryManagement, MemoryManagementOptions},
    stream::EventStreamBackend,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice,
    MTLSharedEvent,
};
use std::sync::Arc;

/// Active encoder state for batching multiple kernel dispatches.
pub struct ActiveEncoder {
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Temporary buffers that must stay alive until this encoder's work completes.
    pub temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

/// Metal stream with its own command queue and memory management.
pub struct MetalStream {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub memory_management: MemoryManagement<MetalStorage>,
    /// Active encoder for batching kernel dispatches (None if no active batch).
    pub active_encoder: Option<ActiveEncoder>,
    /// Number of kernel dispatches in current batch.
    pub batch_ops: usize,
    /// Total buffer memory in current batch (bytes).
    pub batch_bytes: usize,
    /// Shared event for synchronization.
    pub shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Counter for the next event signal value.
    pub event_counter: u64,
    /// Maximum operations per batch (device-specific).
    pub max_ops_per_batch: usize,
    /// Maximum MB per batch (device-specific).
    pub max_mb_per_batch: usize,
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
    /// Get or create the active encoder for batching kernel dispatches.
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
}

/// Metal event for synchronization using MTLSharedEvent.
pub struct MetalEvent {
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: u64,
}

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
    pub fn wait_sync(self) -> Result<(), ExecutionError> {
        let timeout_ms = 60_000;
        if !(*self.shared_event).waitUntilSignaledValue_timeoutMS(self.value, timeout_ms) {
            return Err(ExecutionError::Generic {
                reason: "Metal event wait timed out".to_string(),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            });
        }
        Ok(())
    }

    pub fn wait_async(&self, _queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) {
        // No-op: command buffer ordering on the same queue provides synchronization
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
        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Metal GPU Memory"),
        );

        let device_name = (*self.device).name().to_string();
        let (max_ops_per_batch, max_mb_per_batch) =
            if device_name.contains("Max") || device_name.contains("Ultra") {
                (50, 50)
            } else if device_name.contains("Pro") {
                (40, 40)
            } else if device_name.contains("iPhone") || device_name.contains("iPad") {
                (20, 20)
            } else {
                (40, 40)
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
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        use objc2_metal::{MTLCommandEncoder, MTLEvent};

        stream.event_counter += 1;
        let signal_value = stream.event_counter;

        if let Some(active) = stream.active_encoder.take() {
            (*active.encoder).endEncoding();

            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*stream.shared_event);
            (*active.command_buffer).encodeSignalEvent_value(event_ref, signal_value);

            if !active.temporaries.is_empty() {
                let cell = std::cell::Cell::new(Some(active.temporaries));
                let handler = block2::RcBlock::new(move |_cmd_buf: std::ptr::NonNull<_>| {
                    if let Some(temps) = cell.take() {
                        drop(temps);
                    }
                });
                unsafe {
                    (*active.command_buffer).addCompletedHandler(block2::RcBlock::as_ptr(&handler));
                }
                std::mem::forget(handler);
            }

            (*active.command_buffer).commit();
        } else {
            let signal_buffer = (*stream.queue)
                .commandBuffer()
                .expect("Failed to create command buffer");

            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*stream.shared_event);
            (*signal_buffer).encodeSignalEvent_value(event_ref, signal_value);
            (*signal_buffer).commit();
        }

        stream.batch_ops = 0;
        stream.batch_bytes = 0;

        MetalEvent::new(stream.shared_event.clone(), signal_value)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(&stream.queue);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ExecutionError> {
        event.wait_sync()
    }
}
