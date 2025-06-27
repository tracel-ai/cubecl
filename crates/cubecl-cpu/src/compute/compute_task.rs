use cubecl_core::{ExecutionMode, compute::CubeTask, server::ScalarBinding};
use cubecl_runtime::storage::BytesResource;

use crate::{CpuCompiler, compiler::MlirCompilerOptions};

pub struct ComputeTask {
    kernel: Box<dyn CubeTask<CpuCompiler>>,
    handles: Vec<BytesResource>,
    scalars: Vec<ScalarBinding>,
    vec_unit_pos: Vec<[u32; 3]>,
    cube_count: [u32; 3],
    execution_mode: ExecutionMode,
}

impl ComputeTask {
    pub fn compute(self) {
        let kernel = self.kernel.compile(
            &mut Default::default(),
            &MlirCompilerOptions::default(),
            self.execution_mode,
        );
        let mut mlir_engine = kernel.repr.unwrap();

        for handle in self.handles {
            let ptr = handle.write();
            unsafe {
                mlir_engine.push_buffer(ptr);
            }
        }

        for scalar in self.scalars.into_iter() {
            mlir_engine.push_scalar(scalar);
        }

        mlir_engine.builtin.set_cube_count(self.cube_count);
        mlir_engine.push_builtin();

        for unit_pos in self.vec_unit_pos {
            mlir_engine.builtin.set_unit_pos(unit_pos);
            unsafe {
                mlir_engine.run_kernel();
            }
        }
    }
}
