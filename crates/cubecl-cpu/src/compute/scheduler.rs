use std::fmt::Debug;
use std::{collections::HashMap, sync::mpsc};

use cubecl_core::{ExecutionMode, compute::CubeTask, prelude::CompiledKernel, server::Bindings};
use cubecl_runtime::{id::KernelId, memory_management::MemoryManagement, storage::BytesStorage};

use crate::{
    CpuCompiler,
    compiler::{MlirCompiler, MlirCompilerOptions, mlir_data::MlirData},
};

use super::{compute_task::ComputeTask, worker::Worker};

pub struct Scheduler {
    workers: Vec<Worker>,
    compilation_cache: HashMap<KernelId, CompiledKernel<MlirCompiler>>,
}

impl Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.workers)
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        let available_parallelism = std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get();
        let workers = (0..available_parallelism)
            .map(|_| Worker::default())
            .collect();

        let compilation_cache = HashMap::new();
        Scheduler {
            workers,
            compilation_cache,
        }
    }
}

impl Scheduler {
    pub fn dispatch_execute(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        cube_count: [u32; 3],
        bindings: Bindings,
        kind: ExecutionMode,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) {
        let kernel = self
            .compilation_cache
            .entry(kernel.id())
            .or_insert_with(|| {
                kernel.compile(
                    &mut Default::default(),
                    &MlirCompilerOptions::default(),
                    kind,
                )
            });

        let cube_dim = kernel.cube_dim;
        let mut unit_pos_vec = Vec::with_capacity((cube_dim.x * cube_dim.y * cube_dim.z) as usize);

        for unit_pos_x in 0..cube_dim.x {
            for unit_pos_y in 0..cube_dim.y {
                for unit_pos_z in 0..cube_dim.z {
                    unit_pos_vec.push([unit_pos_x, unit_pos_y, unit_pos_z]);
                }
            }
        }

        let mlir_engine = kernel.repr.clone().unwrap();
        let mut mlir_data =
            MlirData::new(bindings, &mlir_engine.0.shared_memories, memory_management);
        mlir_data.builtin.set_cube_dim(cube_dim);
        mlir_data.builtin.set_cube_count(cube_count);

        let (send, receive) = mpsc::channel();
        let mut msg_count = 0;
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
            msg_count += 1;
            worker.send_task(compute_task);
            worker.send_stop(send.clone());
        }
        for _ in receive.into_iter() {
            msg_count -= 1;
            if msg_count == 0 {
                break;
            }
        }
    }
}
