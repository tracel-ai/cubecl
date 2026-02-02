use cubecl_core::server::ExecutionMode;

use crate::compiler::{mlir_data::MlirData, mlir_engine::MlirEngine};
use std::sync::{
    atomic::{AtomicI32, Ordering},
    mpsc,
};

pub static BARRIER_COUNTER: AtomicI32 = AtomicI32::new(0);
pub static STOPPED_COUNTER: AtomicI32 = AtomicI32::new(0);
pub static BARRIER_TARGET: AtomicI32 = AtomicI32::new(0);
pub static CURRENT_CUBE_DIM: AtomicI32 = AtomicI32::new(-1);

pub fn sync_cube() {
    let barrier_target = BARRIER_TARGET.load(Ordering::Acquire);
    if barrier_target <= 1 {
        return;
    }

    while STOPPED_COUNTER.load(Ordering::Acquire) != 0 {
        std::hint::spin_loop();
    }

    std::sync::atomic::fence(Ordering::Release);
    let arrived = BARRIER_COUNTER.fetch_add(1, Ordering::AcqRel) + 1;

    if arrived < barrier_target {
        while BARRIER_COUNTER.load(Ordering::Acquire) < barrier_target {
            std::hint::spin_loop();
        }
    }

    std::sync::atomic::fence(Ordering::Acquire);

    let stopped = STOPPED_COUNTER.fetch_add(1, Ordering::AcqRel) + 1;
    if stopped == barrier_target {
        BARRIER_COUNTER.store(0, Ordering::Release);
        STOPPED_COUNTER.store(0, Ordering::Release);
    }
}

pub enum Message {
    ComputeTask(ComputeTask),
    EndTask(mpsc::Sender<()>),
}

pub struct ComputeTask {
    pub mlir_engine: MlirEngine,
    pub mlir_data: MlirData,
    pub unit_pos: [u32; 3],
    pub kind: ExecutionMode,
}

impl ComputeTask {
    pub fn compute(mut self) {
        self.mlir_data.push_builtin();
        self.mlir_data.builtin.set_unit_pos(self.unit_pos);
        unsafe {
            self.mlir_engine.run_kernel(&mut self.mlir_data);
        }
        CURRENT_CUBE_DIM.fetch_sub(1, Ordering::AcqRel);
    }
}
