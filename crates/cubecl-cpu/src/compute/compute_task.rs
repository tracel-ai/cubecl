use cubecl_core::ExecutionMode;

use crate::compiler::{mlir_data::MlirData, mlir_engine::MlirEngine};

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
