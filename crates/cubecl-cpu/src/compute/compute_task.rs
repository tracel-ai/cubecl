use cubecl_core::{ExecutionMode, server::ScalarBinding};
use cubecl_runtime::storage::BytesResource;

use crate::compiler::mlir_engine::MlirEngine;

pub struct ComputeTask {
    pub mlir_engine: MlirEngine,
    pub handles: Vec<BytesResource>,
    pub scalars: Vec<ScalarBinding>,
    pub vec_unit_pos: Vec<[u32; 3]>,
    pub cube_count: [u32; 3],
    pub kind: ExecutionMode,
}

impl ComputeTask {
    pub fn compute(mut self) {
        for handle in self.handles {
            let ptr = handle.write();
            unsafe {
                self.mlir_engine.push_buffer(ptr);
            }
        }

        for scalar in self.scalars.into_iter() {
            self.mlir_engine.push_scalar(scalar);
        }

        self.mlir_engine.push_builtin();
        self.mlir_engine.builtin.set_cube_count(self.cube_count);

        for unit_pos in self.vec_unit_pos {
            self.mlir_engine.builtin.set_unit_pos(unit_pos);
            unsafe {
                self.mlir_engine.run_kernel();
            }
        }
    }
}
