#[cfg(feature = "nothreading")]
use crate::compute::shared_memory::reserve_shared_memories;
use crate::compute::stream::CpuStream;
use cubecl_common::bytes::Bytes;
use cubecl_core::{
    CubeDim, MemoryConfiguration, ir::MemoryDeviceProperties, server::MetadataBindingInfo,
};
use cubecl_environment::stream::StreamId;
#[cfg(feature = "nothreading")]
use cubecl_llvm::PlironData;
use cubecl_llvm::PlironEngine;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::ErrorGraph,
    storage::{BytesResource, ManagedResource},
    stream::{StreamFactory, scheduler::SchedulerStreamBackend},
};
#[cfg(feature = "nothreading")]
use cubecl_server::{memory_management::MemoryManagement, storage::BytesStorage};

use std::sync::Arc;

/// Defines tasks that can be scheduled on a cpu stream.
pub enum ScheduleTask {
    /// Represents a task to write data to a buffer.
    Write {
        data: Bytes,
        buffer: ManagedResource<BytesResource>,
    },
    /// Represents a task to execute a kernel.
    Execute {
        stream_id: StreamId,
        pliron_engine: PlironEngine,
        bindings: BindingsResource,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
    },
}

impl core::fmt::Debug for ScheduleTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write { data, buffer } => f
                .debug_struct("Write")
                .field("data", data)
                .field("buffer", buffer)
                .finish(),
            Self::Execute {
                stream_id,
                pliron_engine: _,
                bindings: _,
                cube_dim,
                cube_count,
            } => f
                .debug_struct("Execute")
                .field("stream_id", stream_id)
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
    memory_properties: MemoryDeviceProperties,
    memory_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
}

impl StreamFactory for CpuStreamFactory {
    type Stream = CpuStream;

    fn create(&mut self) -> Self::Stream {
        CpuStream::new(
            self.memory_properties.clone(),
            self.memory_config.clone(),
            self.logger.clone(),
        )
    }
}

impl ScheduledCpuBackend {
    /// Creates a new [`ScheduledCpuBackend`] with the given configurations.
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        Self {
            factory: CpuStreamFactory {
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

    fn enqueue(task: Self::Task, stream: &mut Self::Stream, failures: &mut ErrorGraph) {
        stream.enqueue_task(task, failures);
    }

    fn flush(stream: &mut Self::Stream, _failures: &mut ErrorGraph) {
        stream.submit();
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}

#[cfg(feature = "nothreading")]
pub(crate) fn execute_data_inline(
    pliron_engine: PlironEngine,
    bindings: BindingsResource,
    cube_dim: CubeDim,
    cube_count: [u32; 3],
    memory: &mut MemoryManagement<BytesStorage>,
    failures: &mut ErrorGraph,
) {
    let requirements = pliron_engine.requirements().clone();

    if requirements.needs_parallelism {
        panic!("nothreading feature does not support kernels requiring cube barriers");
    }

    let BindingsResource { resources, info } = bindings;
    let mut buffer_ptrs: Vec<*mut std::ffi::c_void> = resources
        .iter()
        .map(|r| r.resource().get_write_ptr_and_length().0 as *mut std::ffi::c_void)
        .collect();

    reserve_shared_memories(
        memory,
        failures,
        &requirements.shared_memories,
        &mut buffer_ptrs,
    );

    let keepalive: Vec<Box<dyn std::any::Any + Send>> = resources
        .into_iter()
        .map(|r| Box::new(r) as Box<dyn std::any::Any + Send>)
        .collect();

    let base_data = PlironData::new(buffer_ptrs, info.data, cube_count, keepalive);

    for x in 0..cube_dim.x {
        for y in 0..cube_dim.y {
            for z in 0..cube_dim.z {
                let engine = pliron_engine.clone();
                let mut data = base_data.clone();
                data.set_unit_pos([x, y, z]);
                engine.run_kernel(&mut data);
            }
        }
    }
}
