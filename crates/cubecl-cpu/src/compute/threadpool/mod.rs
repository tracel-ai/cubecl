use crossbeam_utils::CachePadded;
use cubecl_core::CubeDim;
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::sync::{Arc, OnceLock, atomic::AtomicU64};

use crate::{
    compiler::jit::{data::PlironData, engine::PlironEngine},
    compiler::shared_memory::SharedMemoryLayout,
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
        memory: &mut MemoryManagement<BytesStorage>,
        next_counter_step: u64,
        atomic_counter: &Arc<CachePadded<AtomicU64>>,
    ) {
        let requirements = pliron_engine.requirements();

        let BindingsResource { resources, info } = bindings;
        let buffer_ptrs = resources
            .iter()
            .map(|resource| {
                resource.resource().get_write_ptr_and_length().0 as *mut std::ffi::c_void
            })
            .collect();
        let shared_memory = reserve_shared_memory(memory, requirements.shared_memory);
        let base_data = PlironData::new(buffer_ptrs, info.data, shared_memory, cube_count);

        // A cube barrier only completes if every unit of the cube is running, so such a kernel
        // needs as many workers as the cube has units.
        if requirements.needs_parallelism {
            self.scheduler.ensure_workers(cube_dim.num_elems() as usize);
        }

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

/// Reserves the shared memory block of a launch out of the stream's shared memory pool, aligned
/// the way the kernel laid it out. The pool guarantees no alignment of its own, so the block is
/// over-reserved and its base rounded up.
///
/// Launches of a stream never overlap (a stream flushes before enqueuing), so the reservation is
/// released right away: the block is the pool's to hand out again once this launch is done with
/// it.
fn reserve_shared_memory(
    memory: &mut MemoryManagement<BytesStorage>,
    layout: SharedMemoryLayout,
) -> *mut u8 {
    if layout.size == 0 {
        return core::ptr::null_mut();
    }

    let handle = memory
        .reserve((layout.size + layout.align - 1) as u64)
        .expect("Failed to reserve the shared memory of the launch");
    let block = memory
        .get_resource(handle.binding(), None, None)
        .expect("Failed to resolve the shared memory of the launch");

    let (ptr, _) = block.get_write_ptr_and_length();
    ptr.wrapping_add(ptr.align_offset(layout.align))
}
