use std::sync::atomic::AtomicI32;

use cubecl_core::ExecutionMode;

use crate::compiler::{mlir_data::MlirData, mlir_engine::MlirEngine};

// -1 variant indicate that the counter is not initialized
static NB_CUBE_TO_SYNC: AtomicI32 = AtomicI32::new(-1);

pub fn sync_cube() {
    println!("SyncCube");
    // if NB_CUBE_TO_SYNC.load(Ordering::Acquire) == -1 {
    //     let available_parallelism = std::thread::available_parallelism()
    //         .expect("Can't get available parallelism on this platform")
    //         .get();
    //     NB_CUBE_TO_SYNC.store(available_parallelism as i32 - 1, Ordering::Release);
    // } else {
    //     NB_CUBE_TO_SYNC.fetch_sub(-1, Ordering::Release);
    // }
    // while {
    //     let val = NB_CUBE_TO_SYNC.load(Ordering::);
    //     val != 0 && val != 1
    // } {}
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
