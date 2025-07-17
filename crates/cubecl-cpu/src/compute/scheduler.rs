use cubecl_core::{ExecutionMode, compute::CubeTask, server::Bindings};
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};

use crate::{
    CpuCompiler,
    compiler::{MlirCompilerOptions, mlir_data::MlirData},
};

use super::{compute_task::ComputeTask, worker::Worker};

#[derive(Debug)]
pub struct Scheduler {
    workers: Vec<Worker>,
}

impl Default for Scheduler {
    fn default() -> Self {
        let available_parallelism = std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get();
        let workers = (0..available_parallelism)
            .map(|_| Worker::default())
            .collect();

        Scheduler { workers }
    }
}

impl Scheduler {
    pub fn sync(&mut self) {
        for worker in self.workers.iter_mut() {
            worker.sync();
        }
    }

    pub fn dispatch_execute(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        cube_count: [u32; 3],
        bindings: Bindings,
        kind: ExecutionMode,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) {
        let cube_dim = kernel
            .compile(
                &mut Default::default(),
                &MlirCompilerOptions::default(),
                kind,
            )
            .cube_dim;

        let mut unit_pos_vec = Vec::with_capacity((cube_dim.x * cube_dim.y * cube_dim.z) as usize);

        for unit_pos_x in 0..cube_dim.x {
            for unit_pos_y in 0..cube_dim.y {
                for unit_pos_z in 0..cube_dim.z {
                    unit_pos_vec.push([unit_pos_x, unit_pos_y, unit_pos_z]);
                }
            }
        }

        let Bindings {
            buffers,
            scalars,
            metadata,
            ..
        } = bindings;

        let handles: Vec<_> = buffers
            .into_iter()
            .map(|b| {
                memory_management
                    .get_resource(b.memory, b.offset_start, b.offset_end)
                    .expect("Failed to find resource")
            })
            .collect();

        let scalars: Vec<_> = scalars.into_values().collect();

        let mut mlir_data = MlirData::new(handles, scalars, metadata);
        mlir_data.builtin.set_cube_dim(cube_dim);
        mlir_data.builtin.set_cube_count(cube_count);

        let kernel = kernel.compile(
            &mut Default::default(),
            &MlirCompilerOptions::default(),
            kind,
        );
        let mlir_engine = kernel.repr.unwrap();

        for (slice, worker) in unit_pos_vec
            .chunks(unit_pos_vec.len().div_ceil(self.workers.len()))
            .zip(self.workers.iter_mut())
        {
            let mlir_engine = mlir_engine.clone();
            let mlir_data = mlir_data.clone();
            let vec_unit_pos = slice.to_vec();

            let compute_task = ComputeTask {
                mlir_engine,
                mlir_data,
                vec_unit_pos,
                kind,
            };
            worker.send_task(compute_task);
        }
        self.sync();
    }
}
