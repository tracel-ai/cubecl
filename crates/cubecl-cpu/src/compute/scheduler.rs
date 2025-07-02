use cubecl_core::{CubeDim, ExecutionMode, compute::CubeTask, server::Bindings};
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};

use crate::{CpuCompiler, compiler::MlirCompilerOptions};

use super::{compute_task::ComputeTask, worker::Worker};

#[derive(Debug)]
pub struct Scheduler {
    workers: Vec<Worker>,
}

impl Scheduler {
    pub fn new() -> Scheduler {
        let available_parallelism = std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get();
        let threads = (0..available_parallelism)
            .into_iter()
            .map(|_| Worker::new())
            .collect();

        Scheduler { workers: threads }
    }

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
        let CubeDim { x, y, z } = kernel
            .compile(
                &mut Default::default(),
                &MlirCompilerOptions::default(),
                kind,
            )
            .cube_dim;

        let mut cube_dims = Vec::with_capacity((x * y * z) as usize);

        for cube_dim_x in 0..x {
            for cube_dim_y in 0..y {
                for cube_dim_z in 0..z {
                    cube_dims.push([cube_dim_x, cube_dim_y, cube_dim_z]);
                }
            }
        }

        let Bindings {
            buffers, scalars, ..
        } = bindings;

        let handles: Vec<_> = buffers
            .into_iter()
            .map(|b| {
                memory_management
                    .get_resource(b.memory, b.offset_start, b.offset_end)
                    .expect("Failed to find resource")
            })
            .collect();

        let scalars: Vec<_> = scalars.into_iter().map(|(_, b)| b).collect();

        for (slice, worker) in cube_dims
            .chunks(cube_dims.len().div_ceil(self.workers.len()))
            .zip(self.workers.iter_mut())
        {
            let kernel = kernel.compile(
                &mut Default::default(),
                &MlirCompilerOptions::default(),
                kind,
            );
            let mlir_engine = kernel.repr.unwrap();
            let vec_unit_pos = slice.to_vec();

            let handles = handles.clone();
            let scalars = scalars.clone();

            let compute_task = ComputeTask {
                mlir_engine,
                handles,
                scalars,
                vec_unit_pos,
                cube_count,
                kind,
            };
            worker.send_task(compute_task);
        }
        self.sync();
    }
}
