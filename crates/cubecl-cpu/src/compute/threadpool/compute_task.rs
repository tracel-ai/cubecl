use std::sync::{Arc, atomic::AtomicU64};

use crossbeam_utils::CachePadded;

use crate::{
    compiler::jit::{data::PlironData, engine::PlironEngine},
    compute::threadpool::ThreadTask,
};

pub struct ComputeTask {
    pub pliron_engine: PlironEngine,
    pub pliron_data: PlironData,
    pub next_counter_step: u64,
    pub atomic_counter: Arc<CachePadded<AtomicU64>>,
}

impl ThreadTask for ComputeTask {
    fn is_ready(&self) -> bool {
        self.atomic_counter
            .load(std::sync::atomic::Ordering::Acquire)
            >= self.next_counter_step
    }
}

impl ComputeTask {
    pub fn compute(&mut self) {
        // self.pliron_data.push_builtin();
        self.pliron_engine.run_kernel(&mut self.pliron_data);
        self.pliron_data.complete_unit();
        self.atomic_counter
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}
