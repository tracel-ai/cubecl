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
        pliron_engine: PlironEngine,
        bindings: BindingsResource,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
        _memory: &mut MemoryManagement<BytesStorage>,
        next_counter_step: u64,
        atomic_counter: &Arc<CachePadded<AtomicU64>>,
    ) {
        let BindingsResource { resources, info } = bindings;
        let buffer_ptrs = resources
            .iter()
            .map(|resource| {
                resource.resource().get_write_ptr_and_length().0 as *mut std::ffi::c_void
            })
            .collect();
        let base_data = PlironData::new(buffer_ptrs, info.data, cube_count);

        let mut i = 0;
        for unit_pos_x in 0..cube_dim.x {
            for unit_pos_y in 0..cube_dim.y {
                for unit_pos_z in 0..cube_dim.z {
                    let pliron_engine = pliron_engine.clone();
                    let mut pliron_data = base_data.clone();
                    pliron_data.set_unit_pos([unit_pos_x, unit_pos_y, unit_pos_z]);

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
