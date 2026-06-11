use cubecl_core::{CubeDim, config::RuntimeConfig};
use cubecl_runtime::{
    config::CubeClRuntimeConfig, memory_management::MemoryManagement, storage::BytesStorage,
};
use std::sync::{Arc, OnceLock, atomic::AtomicU64};
use sysinfo::System;

use crate::{
    compiler::{mlir_data::MlirData, mlir_engine::MlirEngine},
    compute::{
        affinity::{CoreId, get_active_cores},
        schedule::BindingsResource,
        threadpool::{compute_task::ComputeTask, thread_buffer::ThreadBuffer, worker::Worker},
        utils::cache_padded::CachePadded,
    },
};

pub mod circular_buffer;
pub mod compute_task;
pub mod thread_buffer;
pub mod worker;

static INSTANCE: OnceLock<spin::Mutex<Threadpool>> = OnceLock::new();

/// The kernel runner is responsible to manage shared memory as well as threads to execute kernels.
///
/// A single kernel runner is currently used for all kernels.
/// To register work, you have to use the execution queue.
pub struct Threadpool {
    threads_buffer: Arc<[spin::Mutex<ThreadBuffer<ComputeTask>>]>,
}

impl Threadpool {
    fn init() -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        let mut system = System::new();
        system.refresh_memory();

        let active_cores: Vec<CoreId> = get_active_cores().collect();

        let buffers = (0..active_cores.len())
            .into_iter()
            .map(|i| spin::Mutex::new(ThreadBuffer::new(i, max_streams as usize)))
            .collect::<Vec<_>>();
        let threads_buffer: Arc<[_]> = buffers.into();
        for buffer in threads_buffer.iter() {
            buffer.lock().set_threads_buffer(threads_buffer.clone());
        }

        for (thread_id, core_id) in active_cores.into_iter().enumerate() {
            let threads_buffer = Arc::clone(&threads_buffer);
            Worker::spawn_thread(core_id, thread_id, threads_buffer);
        }

        Self { threads_buffer }
    }

    /// Resolves the global execution queue instance.
    pub fn get() -> &'static spin::Mutex<Self> {
        INSTANCE.get_or_init(|| spin::Mutex::new(Self::init()))
    }

    pub fn execute_data(
        &mut self,
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
        memory: &mut MemoryManagement<BytesStorage>,
        stream_id: usize,
        next_counter_step: u64,
        atomic_counter: &Arc<CachePadded<AtomicU64>>,
    ) {
        let mlir_data = MlirData::new(
            bindings,
            &mlir_engine.0.shared_memories,
            memory,
            cube_dim,
            cube_count,
        );

        let mut workers = self.threads_buffer.iter();
        for unit_pos_x in 0..cube_dim.x {
            for unit_pos_y in 0..cube_dim.y {
                for unit_pos_z in 0..cube_dim.z {
                    let unit_pos = [unit_pos_x, unit_pos_y, unit_pos_z];
                    let worker = workers.next().expect("The CubeDim are too large");
                    let mlir_engine = mlir_engine.clone();
                    let mut mlir_data = mlir_data.clone();
                    mlir_data.builtin.set_unit_pos(unit_pos);

                    let atomic_counter = Arc::clone(atomic_counter);
                    let compute_task = ComputeTask {
                        mlir_engine,
                        mlir_data,
                        stream_id,
                        next_counter_step,
                        atomic_counter,
                    };
                    worker.lock().push(compute_task);
                }
            }
        }
    }
}
