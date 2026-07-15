use crossbeam_utils::CachePadded;
use cubecl_core::CubeDim;
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::sync::{Arc, OnceLock, atomic::AtomicU64};

use crate::{
    compiler::{mlir_data::MlirData, mlir_engine::MlirEngine},
    compute::{
        schedule::BindingsResource,
        threadpool::{
            compute_task::ComputeTask,
            scheduler::{Scheduler, SchedulerVariant},
        },
    },
};

pub mod compute_task;
pub mod scheduler;

trait ThreadTask {
    fn is_ready(&self) -> bool;
}

static INSTANCE: OnceLock<spin::Mutex<Threadpool>> = OnceLock::new();

/// The kernel runner is responsible to manage shared memory as well as threads to execute kernels.
///
/// A single kernel runner is currently used for all kernels.
/// To register work, you have to use the execution queue.
pub struct Threadpool {
    scheduler: Scheduler,
}

impl Threadpool {
    fn init() -> Self {
        let scheduler = Scheduler::new(SchedulerVariant::Dispatcher);

        Self { scheduler }
    }

    /// Resolves the global execution queue instance.
    pub fn get() -> &'static spin::Mutex<Self> {
        INSTANCE.get_or_init(|| spin::Mutex::new(Self::init()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_data(
        &mut self,
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
        memory: &mut MemoryManagement<BytesStorage>,
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

        // A `sync_cube` barrier only resolves when every unit of the cube runs
        // on its own thread, so grow the pool to one worker per unit. Kernels
        // without a barrier load-balance and need no extra workers.
        if mlir_engine.0.needs_parallelism {
            self.scheduler.ensure_workers(cube_dim.num_elems() as usize);
        }

        let mut i = 0;
        for unit_pos_x in 0..cube_dim.x {
            for unit_pos_y in 0..cube_dim.y {
                for unit_pos_z in 0..cube_dim.z {
                    let unit_pos = [unit_pos_x, unit_pos_y, unit_pos_z];
                    let mlir_engine = mlir_engine.clone();
                    let mut mlir_data = mlir_data.clone();
                    mlir_data.builtin.set_unit_pos(unit_pos);

                    let atomic_counter = Arc::clone(atomic_counter);
                    let compute_task = ComputeTask {
                        mlir_engine,
                        mlir_data,
                        next_counter_step,
                        atomic_counter,
                    };
                    self.scheduler.send(i, compute_task);
                    i += 1;
                }
            }
        }
    }
}
