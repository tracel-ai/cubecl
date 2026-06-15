use super::{
    affinity::get_active_cores, compute_task::ComputeTask, notification::Notifications,
    schedule::BindingsResource, threadpool::worker::Worker,
};
use crate::compiler::{mlir_data::MlirData, mlir_engine::MlirEngine};
use cubecl_core::CubeDim;
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::{fmt::Debug, sync::OnceLock};
use sysinfo::System;

pub mod circular_buffer;
pub mod thread_buffer;
pub mod worker;

static INSTANCE: OnceLock<spin::Mutex<Threadpool>> = OnceLock::new();

/// The kernel runner is responsible to manage shared memory as well as threads to execute kernels.
///
/// A single kernel runner is currently used for all kernels.
/// To register work, you have to use the execution queue.
pub struct Threadpool {
    workers: Vec<Worker>,
}

impl Debug for Threadpool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.workers)
    }
}

impl Threadpool {
    fn init() -> Self {
        let mut system = System::new();
        system.refresh_memory();

        let workers = get_active_cores().map(Worker::new_with_affinity).collect();

        Self { workers }
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
        memory_management_shared_memory: &mut MemoryManagement<BytesStorage>,
    ) -> Notifications {
        let cube_dim_size = cube_dim.num_elems();

        let mlir_data = MlirData::new(
            bindings,
            &mlir_engine.0.shared_memories,
            memory_management_shared_memory,
            cube_dim,
            cube_count,
        );

        let notifications = Notifications::new(cube_dim_size);
        let mut workers = self.workers.iter_mut();
        for unit_pos_x in 0..cube_dim.x {
            for unit_pos_y in 0..cube_dim.y {
                for unit_pos_z in 0..cube_dim.z {
                    let unit_pos = [unit_pos_x, unit_pos_y, unit_pos_z];
                    let worker = workers.next().expect("The CubeDim are too large");
                    let mlir_engine = mlir_engine.clone();
                    let mut mlir_data = mlir_data.clone();
                    mlir_data.builtin.set_unit_pos(unit_pos);

                    let notifications = notifications.clone();
                    let compute_task = ComputeTask {
                        mlir_engine,
                        mlir_data,
                        notifications,
                    };
                    worker.send_task(compute_task);
                }
            }
        }

        notifications
    }
}
