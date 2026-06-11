use std::sync::{Arc, atomic::AtomicU64};

use crate::{
    compiler::{mlir_data::MlirData, mlir_engine::MlirEngine},
    compute::threadpool::thread_buffer::ThreadTask,
};

pub struct ComputeTask {
    pub mlir_engine: MlirEngine,
    pub mlir_data: MlirData,
    pub stream_id: usize,
    pub next_counter_step: u64,
    pub atomic_counter: Arc<AtomicU64>,
}

impl ThreadTask for ComputeTask {
    fn get_stream_id(&self) -> usize {
        self.stream_id
    }

    fn is_ready(&self) -> bool {
        self.atomic_counter
            .load(std::sync::atomic::Ordering::Acquire)
            >= self.next_counter_step
    }
}

impl ComputeTask {
    pub fn compute(mut self) {
        self.mlir_data.push_builtin();
        unsafe {
            self.mlir_engine.run_kernel(&mut self.mlir_data);
        }
        self.mlir_data.complete_unit();
        self.atomic_counter
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}
