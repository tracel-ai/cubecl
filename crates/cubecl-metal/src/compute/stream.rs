use crate::memory::MetalStorage;
use cubecl_core::{MemoryConfiguration, server::ExecutionError};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryManagement, MemoryManagementOptions, SliceBinding},
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
    /// Bindings in use by last submitted work; flushed and GC'd before next launch.
    pub pending_bindings: Vec<SliceBinding>,
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

/// Metal event for synchronization using `MTLSharedEvent`.
pub struct MetalEvent {
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub value: u64,
    command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    #[allow(dead_code)]
    temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Clone for MetalEvent {
    fn clone(&self) -> Self {
        Self {
            shared_event: self.shared_event.clone(),
            value: self.value,
            command_buffer: None,
            temporaries: self.temporaries.clone(),
        }
    }
}

unsafe impl Send for MetalEvent {}

impl MetalEvent {
    pub fn new(
        shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        value: u64,
        command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
        temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> Self {
        Self {
            shared_event,
            value,
            command_buffer,
            temporaries,
        }
    }

    /// Check if the event has been signaled (non-blocking).
    pub fn is_complete(&self) -> bool {
        (*self.shared_event).signaledValue() >= self.value
    }

    /// Block until the event is signaled.
    pub fn wait_sync(self) -> Result<(), ExecutionError> {
        if let Some(ref cmd_buf) = self.command_buffer {
            (*cmd_buf).waitUntilCompleted();
        } else {
            let timeout_ms = 60_000;
            let result =
                (*self.shared_event).waitUntilSignaledValue_timeoutMS(self.value, timeout_ms);
            if !result {
                return Err(ExecutionError::Generic {
                    reason: "Metal event wait timed out".to_string(),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                });
            }
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
            pending_bindings: Vec::new(),
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        use objc2_metal::{MTLCommandEncoder, MTLEvent};

        stream.event_counter += 1;
        let signal_value = stream.event_counter;

        let (command_buffer, temporaries) = if let Some(active) = stream.active_encoder.take() {
            (*active.encoder).endEncoding();

            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*stream.shared_event);
            (*active.command_buffer).encodeSignalEvent_value(event_ref, signal_value);

            (*active.command_buffer).commit();
            (Some(active.command_buffer), active.temporaries)
        } else {
            let signal_buffer = (*stream.queue)
                .commandBuffer()
                .expect("Failed to create command buffer");

            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*stream.shared_event);
            (*signal_buffer).encodeSignalEvent_value(event_ref, signal_value);
            (*signal_buffer).commit();
            (Some(signal_buffer), Vec::new())
        };

        stream.batch_ops = 0;
        stream.batch_bytes = 0;

        MetalEvent::new(
            stream.shared_event.clone(),
            signal_value,
            command_buffer,
            temporaries,
        )
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(stream);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ExecutionError> {
        event.wait_sync()
    }
}
