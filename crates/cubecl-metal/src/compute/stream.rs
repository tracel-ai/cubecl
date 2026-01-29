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
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLSharedEvent};
use std::sync::Arc;

/// A pending command buffer with its associated temporary buffers.
pub struct PendingCommandBuffer {
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Temporary buffers (metadata, scalars) bound to this command buffer.
    /// Must stay alive until GPU execution completes.
    pub temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl std::fmt::Debug for PendingCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingCommandBuffer")
            .field("temporaries_count", &self.temporaries.len())
            .finish()
    }
}

/// Metal stream with its own command queue and memory management.
pub struct MetalStream {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub memory_management: MemoryManagement<MetalStorage>,
    pub pending_buffers: Vec<PendingCommandBuffer>,
    /// Number of operations (kernel launches) in current batch.
    pub buffer_ops: usize,
    /// Total size of buffers bound in current batch (in bytes).
    pub buffer_bytes: usize,
    /// Shared event for synchronization.
    pub shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Counter for the next event signal value.
    pub event_counter: u64,
    /// Maximum operations per command buffer (device-specific).
    pub max_ops_per_buffer: usize,
    /// Maximum MB per command buffer (device-specific).
    pub max_mb_per_buffer: usize,
}

impl std::fmt::Debug for MetalStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalStream")
            .field("pending_buffers", &self.pending_buffers.len())
            .field("buffer_ops", &self.buffer_ops)
            .field("buffer_bytes", &self.buffer_bytes)
            .field("event_counter", &self.event_counter)
            .finish()
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
        let (max_ops_per_buffer, max_mb_per_buffer) =
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
            pending_buffers: Vec::new(),
            buffer_ops: 0,
            buffer_bytes: 0,
            shared_event,
            event_counter: 0,
            max_ops_per_buffer,
            max_mb_per_buffer,
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        use objc2_metal::MTLEvent;

        let pending_buffers: Vec<_> = stream.pending_buffers.drain(..).collect();

        for pending in pending_buffers.iter() {
            (*pending.command_buffer).commit();
        }

        stream.event_counter += 1;
        let signal_value = stream.event_counter;

        let signal_buffer = (*stream.queue)
            .commandBuffer()
            .expect("Failed to create command buffer");

        let event_ref: &ProtocolObject<dyn MTLEvent> =
            ProtocolObject::from_ref(&*stream.shared_event);
        (*signal_buffer).encodeSignalEvent_value(event_ref, signal_value);

        if !pending_buffers.is_empty() {
            let cell = std::cell::Cell::new(Some(pending_buffers));
            let handler = block2::RcBlock::new(move |_cmd_buf: std::ptr::NonNull<_>| {
                if let Some(buffers) = cell.take() {
                    drop(buffers);
                }
            });
            unsafe {
                (*signal_buffer).addCompletedHandler(block2::RcBlock::as_ptr(&handler));
            }
            std::mem::forget(handler);
        }

        (*signal_buffer).commit();

        stream.buffer_ops = 0;
        stream.buffer_bytes = 0;

        MetalEvent::new(stream.shared_event.clone(), signal_value)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(&stream.queue);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ExecutionError> {
        event.wait_sync()
    }
}
