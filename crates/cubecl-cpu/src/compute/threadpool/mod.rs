use crossbeam_utils::CachePadded;
use cubecl_core::CubeDim;
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::sync::{Arc, OnceLock, atomic::AtomicU64};

use cubecl_llvm::{PlironData, PlironEngine, shared::shared_memory::SharedMemories};

use crate::compute::{
    schedule::BindingsResource,
    threadpool::{
        compute_task::ComputeTask,
        scheduler::{Scheduler, SchedulerVariant},
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
        let requirements = pliron_engine.requirements().clone();

        let BindingsResource { resources, info } = bindings;
        let mut buffer_ptrs: Vec<*mut std::ffi::c_void> = resources
            .iter()
            .map(|resource| {
                resource.resource().get_write_ptr_and_length().0 as *mut std::ffi::c_void
            })
            .collect();
        reserve_shared_memories(memory, &requirements.shared_memories, &mut buffer_ptrs);
        // Pin the resources for the launch's lifetime (see
        // `SharedData::keepalive`).
        let keepalive: Vec<Box<dyn std::any::Any + Send>> = resources
            .into_iter()
            .map(|resource| Box::new(resource) as Box<dyn std::any::Any + Send>)
            .collect();
        let base_data = PlironData::new(buffer_ptrs, info.data, cube_count, keepalive);

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

/// Reserves the shared memory of a launch out of the stream's dedicated pool, and writes each
/// block into the slot of `table` the kernel reads it from. Those slots follow the buffers, so
/// the table is padded first when the kernel takes fewer buffers than the launch provides.
///
/// The pool guarantees an alignment of its own too small for a vector, so a block is
/// over-reserved and its base rounded up.
///
/// The reservations are released right away: shared-memory launches never overlap — the stream
/// drains before enqueuing one (see `CpuStream::enqueue_task`).
fn reserve_shared_memories(
    memory: &mut MemoryManagement<BytesStorage>,
    shared_memories: &SharedMemories,
    table: &mut Vec<*mut std::ffi::c_void>,
) {
    let end = shared_memories.base + shared_memories.blocks.len();
    if table.len() < end {
        table.resize(end, core::ptr::null_mut());
    }

    // The handles are held until every block is reserved: releasing one right away would let
    // the pool hand the same memory out to the next shared memory of the same launch.
    let mut handles = Vec::with_capacity(shared_memories.blocks.len());

    for (slot, block) in shared_memories.blocks.iter().enumerate() {
        let handle = memory
            .reserve((block.size + block.align - 1) as u64)
            .expect("Failed to reserve the shared memory of the launch");
        let reserved = memory
            .get_resource(handle.clone().binding(), None, None)
            .expect("Failed to resolve the shared memory of the launch");
        handles.push(handle);

        let (ptr, _) = reserved.get_write_ptr_and_length();
        let ptr = ptr.wrapping_add(ptr.align_offset(block.align));
        table[shared_memories.base + slot] = ptr as *mut std::ffi::c_void;
    }
}
