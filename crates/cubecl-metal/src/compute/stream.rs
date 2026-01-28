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
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice};
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
#[derive(Debug)]
pub struct MetalStream {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub memory_management: MemoryManagement<MetalStorage>,
    pub pending_buffers: Vec<PendingCommandBuffer>,
}

/// Metal event for synchronization, wrapping a command buffer.
pub struct MetalEvent {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

unsafe impl Send for MetalEvent {}

impl MetalEvent {
    pub fn new(command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self { command_buffer }
    }

    pub fn wait_sync(self) -> Result<(), ExecutionError> {
        (*self.command_buffer).waitUntilCompleted();
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

        let storage = MetalStorage::new(self.device.clone());
        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Metal GPU Memory"),
        );

        MetalStream {
            device: self.device.clone(),
            queue,
            memory_management,
            pending_buffers: Vec::new(),
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        let pending_buffers: Vec<_> = stream.pending_buffers.drain(..).collect();

        for pending in pending_buffers.iter() {
            (*pending.command_buffer).commit();
        }

        // Create fence and wait for completion before dropping temporaries
        let fence_buffer = (*stream.queue)
            .commandBuffer()
            .expect("Failed to create command buffer");
        (*fence_buffer).commit();
        (*fence_buffer).waitUntilCompleted();

        drop(pending_buffers);

        // Return a new completed event for the caller
        let command_buffer = (*stream.queue)
            .commandBuffer()
            .expect("Failed to create command buffer");
        (*command_buffer).commit();

        MetalEvent::new(command_buffer)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(&stream.queue);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ExecutionError> {
        event.wait_sync()
    }
}
