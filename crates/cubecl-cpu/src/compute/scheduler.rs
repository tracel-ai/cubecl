use std::fmt::Debug;
use std::sync::atomic::Ordering;
use std::{collections::HashMap, sync::mpsc};

use cubecl_core::{ExecutionMode, compute::CubeTask, prelude::CompiledKernel, server::Bindings};
use cubecl_runtime::{id::KernelId, memory_management::MemoryManagement, storage::BytesStorage};

use crate::{
    CpuCompiler,
    compiler::{MlirCompiler, MlirCompilerOptions, mlir_data::MlirData},
};

use super::compute_task::{BARRIER_COUNTER, CURRENT_CUBE_DIM, STOPPED_COUNTER};
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
        let cube_dim_size = cube_dim.num_elems();

        let mlir_engine = kernel.repr.clone().unwrap();
        let mut mlir_data =
            MlirData::new(bindings, &mlir_engine.0.shared_memories, memory_management);
        mlir_data.builtin.set_cube_dim(cube_dim);
        mlir_data.builtin.set_cube_count(cube_count);

        let (send, receive) = mpsc::channel();
        let mut msg_count = 0;

        CURRENT_CUBE_DIM.store(cube_dim_size as i32, Ordering::Release);
        BARRIER_COUNTER.store(0, Ordering::Release);
        STOPPED_COUNTER.store(0, Ordering::Release);

        if cube_dim_size > self.workers.len() as u32 {
            self.workers
                .extend((0..cube_dim_size - self.workers.len() as u32).map(|_| Worker::default()));
        }

        let mut workers = self.workers.iter_mut();
        for unit_pos_x in 0..cube_dim.x {
            for unit_pos_y in 0..cube_dim.y {
                for unit_pos_z in 0..cube_dim.z {
                    let unit_pos = [unit_pos_x, unit_pos_y, unit_pos_z];
                    let worker = workers.next().expect("The CubeDim are too large");
                    let mlir_engine = mlir_engine.clone();
                    let mlir_data = mlir_data.clone();

                    let compute_task = ComputeTask {
                        mlir_engine,
                        mlir_data,
                        unit_pos,
                        kind,
                    };
                    msg_count += 1;
                    worker.send_task(compute_task);
                    worker.send_stop(send.clone());
                }
            }
        }

        for _ in receive.into_iter() {
            msg_count -= 1;
            if msg_count == 0 {
                break;
            }
        }
    }
}
