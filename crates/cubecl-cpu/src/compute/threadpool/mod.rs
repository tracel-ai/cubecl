use crossbeam_utils::CachePadded;
use cubecl_core::CubeDim;
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::sync::{Arc, OnceLock, atomic::AtomicU64};

use crate::{
    compiler::jit::{data::PlironData, engine::PlironEngine},
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
        _pliron_engine: PlironEngine,
        _bindings: BindingsResource,
        cube_dim: CubeDim,
        _cube_count: [u32; 3],
        _memory: &mut MemoryManagement<BytesStorage>,
        next_counter_step: u64,
        atomic_counter: &Arc<CachePadded<AtomicU64>>,
    ) {
        // let mlir_data = PlironData::new(bindings, &[], memory, cube_dim, cube_count);

        // // A `sync_cube` barrier only resolves when every unit of the cube runs
        // // on its own thread, so grow the pool to one worker per unit. Kernels
        // // without a barrier load-balance and need no extra workers.
        // if pliron_engine.0.needs_parallelism {
        //     self.scheduler.ensure_workers(cube_dim.num_elems() as usize);
        // }

        let mut i = 0;
        for _ in 0..cube_dim.x {
            for _ in 0..cube_dim.y {
                for _ in 0..cube_dim.z {
                    // let unit_pos = [unit_pos_x, unit_pos_y, unit_pos_z];
                    let pliron_engine = PlironEngine;
                    let pliron_data = PlironData;
                    // mlir_data.builtin.set_unit_pos(unit_pos);

                    let atomic_counter = Arc::clone(atomic_counter);
                    let compute_task = ComputeTask {
                        pliron_engine,
                        pliron_data,
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
