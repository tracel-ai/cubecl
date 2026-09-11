use std::sync::Arc;

use cubecl_llvm::{PlironData, PlironEngine};

use crate::compute::threadpool::{ThreadTask, completion_counter::CompletionCounter};

pub struct ComputeTask {
    pub pliron_engine: PlironEngine,
    pub pliron_data: PlironData,
    pub next_counter_step: u64,
    pub atomic_counter: Arc<CompletionCounter>,
}

impl ThreadTask for ComputeTask {
    fn is_ready(&self) -> bool {
        self.atomic_counter.load() >= self.next_counter_step
    }
}

impl ComputeTask {
    pub fn compute(&mut self) {
        self.pliron_engine.run_kernel(&mut self.pliron_data);
        self.atomic_counter.add_done();
    }
}
