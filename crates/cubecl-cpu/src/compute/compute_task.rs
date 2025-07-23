use std::sync::{
    atomic::{AtomicI32, Ordering},
    mpsc,
};

use cubecl_core::ExecutionMode;

use crate::compiler::{mlir_data::MlirData, mlir_engine::MlirEngine};

// -1 variant indicate that the counter is not initialized
static NB_CUBE_TO_SYNC: AtomicI32 = AtomicI32::new(-1);

pub fn sync_cube(cube_dim: u32) {
    if NB_CUBE_TO_SYNC.load(Ordering::Acquire) == -1 {
        NB_CUBE_TO_SYNC.store((cube_dim as i64 - 1) as i32, Ordering::Release);
    } else {
        NB_CUBE_TO_SYNC.fetch_sub(-1, Ordering::Release);
    }
    loop {
        let val = NB_CUBE_TO_SYNC.load(Ordering::Acquire);
        if val == 0 {
            NB_CUBE_TO_SYNC.store(-1, Ordering::Release);
            break;
        }
        if val == -1 {
            break;
        }
        std::hint::spin_loop();
    }
}

pub enum Message {
    ComputeTask(ComputeTask),
    EndTask(mpsc::Sender<()>),
}

pub struct ComputeTask {
    pub mlir_engine: MlirEngine,
    pub mlir_data: MlirData,
    pub vec_unit_pos: Vec<[u32; 3]>,
    pub kind: ExecutionMode,
}

impl ComputeTask {
    pub fn compute(mut self) {
        self.mlir_data.push_builtin();
        for unit_pos in self.vec_unit_pos {
            self.mlir_data.builtin.set_unit_pos(unit_pos);
            unsafe {
                self.mlir_engine.run_kernel(&mut self.mlir_data);
            }
        }
    }
}
