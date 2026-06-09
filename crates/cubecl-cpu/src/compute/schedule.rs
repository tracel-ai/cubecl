use crate::{compiler::mlir_engine::MlirEngine, compute::stream::CpuStream};
use cubecl_common::{bytes::Bytes, stream_id::StreamId};
use cubecl_core::{
    CubeDim, ExecutionMode, MemoryConfiguration, ir::MemoryDeviceProperties,
    server::MetadataBindingInfo,
};
use cubecl_runtime::{
    logging::ServerLogger,
    storage::{BytesResource, ManagedResource},
    stream::{StreamFactory, scheduler::SchedulerStreamBackend},
};
use std::sync::Arc;

/// Defines tasks that can be scheduled on a cpu stream.
pub enum ScheduleTask {
    /// Represents a task to write data to a buffer.
    Write {
        stream_id: StreamId,
        data: Bytes,
        buffer: ManagedResource<BytesResource>,
    },
    /// Represents a task to execute a kernel.
    Execute {
        stream_id: StreamId,
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        kind: ExecutionMode,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
    },
}

impl core::fmt::Debug for ScheduleTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write {
                stream_id,
                data,
                buffer,
            } => f
                .debug_struct("Write")
                .field("stream_id", stream_id)
                .field("data", data)
                .field("buffer", buffer)
                .finish(),
            Self::Execute {
                stream_id,
                mlir_engine: _,
                bindings: _,
                kind,
                cube_dim,
                cube_count,
            } => f
                .debug_struct("Execute")
                .field("stream_id", stream_id)
                .field("kind", kind)
                .field("cube_dim", cube_dim)
                .field("cube_count", cube_count)
                .finish(),
        }
    }
}

/// Represents a collection of resources and bindings for a compute task.
#[derive(Debug)]
pub struct BindingsResource {
    /// List of cpu resources used in the task.
    pub resources: Vec<ManagedResource<BytesResource>>,
    /// Metadata for uniform bindings.
    pub info: MetadataBindingInfo,
}

/// Represents a cpu backend for scheduling tasks on streams.
#[derive(Debug)]
pub struct ScheduledCpuBackend {
    /// Factory for creating cpu streams.
    factory: CpuStreamFactory,
}

/// Factory for creating cpu streams with specific configurations.
#[derive(Debug)]
pub struct CpuStreamFactory {
    max_units_per_cube: u32,
    memory_properties: MemoryDeviceProperties,
    memory_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
}

impl StreamFactory for CpuStreamFactory {
    type Stream = CpuStream;

    fn create(&mut self, index: usize) -> Self::Stream {
        CpuStream::new(
            self.max_units_per_cube,
            self.memory_properties.clone(),
            self.memory_config.clone(),
            self.logger.clone(),
            index,
        )
    }
}

impl ScheduledCpuBackend {
    /// Creates a new [`ScheduledCpuBackend`] with the given configurations.
    pub fn new(
        max_units_per_cube: u32,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        Self {
            factory: CpuStreamFactory {
                max_units_per_cube,
                memory_properties,
                memory_config,
                logger,
            },
        }
    }
}

impl SchedulerStreamBackend for ScheduledCpuBackend {
    type Task = ScheduleTask;
    type Stream = CpuStream;
    type Factory = CpuStreamFactory;

    fn enqueue(task: Self::Task, stream: &mut Self::Stream) {
        stream.enqueue_task(task);
    }

    fn flush(stream: &mut Self::Stream) {
        let _ = stream
            .flush(cubecl_core::server::StreamErrorMode {
                ignore: true,
                flush: false,
            })
            .ok();
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}
