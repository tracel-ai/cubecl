use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::{collections::HashMap, sync::mpsc};

use cubecl_core::CubeDim;
use cubecl_core::{ExecutionMode, prelude::CompiledKernel, server::Bindings};
use cubecl_runtime::compiler::CompilationError;
use cubecl_runtime::{
    compiler::CubeTask, id::KernelId, memory_management::MemoryManagement, storage::BytesStorage,
};

use crate::compiler::mlir_engine::MlirEngine;
use crate::{
    CpuCompiler,
    compiler::{MlirCompiler, MlirCompilerOptions, mlir_data::MlirData},
};

use super::compute_task::{BARRIER_COUNTER, CURRENT_CUBE_DIM, STOPPED_COUNTER};
use super::{compute_task::ComputeTask, worker::Worker};

pub struct KernelRunner {
    workers: Vec<Worker>,
    compilation_cache: HashMap<KernelId, CpuKernel>,
}

/// A compiled cpu kernel.
pub struct CpuKernel {
    mlir: Arc<CompiledKernel<MlirCompiler>>,
}

impl CpuKernel {
    pub fn new(kernel: CompiledKernel<MlirCompiler>) -> Self {
        Self {
            mlir: Arc::new(kernel),
        }
    }
}

impl Debug for CpuKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuKernel")
            .field("entrypoint_name", &self.mlir.entrypoint_name)
            .field("debug_name", &self.mlir.debug_name)
            .finish()
    }
}

impl Debug for KernelRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.workers)
    }
}

impl Default for KernelRunner {
    fn default() -> Self {
        let available_parallelism = std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get();
        let workers = (0..available_parallelism)
            .map(|_| Worker::default())
            .collect();

        let compilation_cache = HashMap::new();
        KernelRunner {
            workers,
            compilation_cache,
        }
    }
}

impl KernelRunner {
    pub fn execute(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        cube_count: [u32; 3],
        bindings: Bindings,
        kind: ExecutionMode,
        memory_management: &mut MemoryManagement<BytesStorage>,
        memory_management_shared_memory: &mut MemoryManagement<BytesStorage>,
    ) -> Result<(), CompilationError> {
        let kernel_id = kernel.id();
        let kernel = if let Some(kernel) = self.compilation_cache.get(&kernel_id) {
            kernel
        } else {
            let kernel = kernel.compile(
                &mut Default::default(),
                &MlirCompilerOptions::default(),
                kind,
            )?;
            self.compilation_cache
                .insert(kernel_id.clone(), CpuKernel::new(kernel));
            self.compilation_cache
                .get_mut(&kernel_id)
                .expect("Just inserted")
        };

        let cube_dim = kernel.mlir.cube_dim;
        let cube_dim_size = cube_dim.num_elems();

        let mlir_engine = kernel.mlir.repr.clone().unwrap();
        let mut mlir_data = MlirData::new(
            bindings,
            &mlir_engine.0.shared_memories,
            memory_management,
            memory_management_shared_memory,
        );
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

        Ok(())
    }

    pub fn execute_data(
        &mut self,
        mlir_engine: MlirEngine,
        mlir_data: MlirData,
        kind: ExecutionMode,
        cube_dim: CubeDim,
    ) {
        let (send, receive) = mpsc::channel();
        let mut msg_count = 0;
        let cube_dim_size = cube_dim.num_elems();

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
