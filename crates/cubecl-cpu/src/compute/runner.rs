use super::{
    compute_task::{BARRIER_COUNTER, CURRENT_CUBE_DIM, ComputeTask, STOPPED_COUNTER},
    schedule::BindingsResource,
    worker::Worker,
};
use crate::{
    CpuCompiler,
    compiler::{MlirCompiler, MlirCompilerOptions, mlir_data::MlirData, mlir_engine::MlirEngine},
    compute::schedule::ScheduleTask,
};
use cubecl_core::{
    CubeDim, ExecutionMode, MemoryConfiguration, ir::MemoryDeviceProperties,
    prelude::CompiledKernel,
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    logging::ServerLogger,
    memory_management::{MemoryManagement, MemoryManagementOptions},
    storage::BytesStorage,
};
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, atomic::Ordering, mpsc},
};
use sysinfo::System;

/// The kernel runner is responsible to manage shared memory as well as threads to execute kernels.
///
/// A single kernel runner is currently used for all kernels.
/// To register work, you have to use the execution queue.
pub struct KernelRunner {
    workers: Vec<Worker>,
    compilation_cache: HashMap<KernelId, CpuKernel>,
    memory_management_shared_memory: MemoryManagement<BytesStorage>,
}

/// A compiled cpu kernel.
pub struct CpuKernel {
    pub(crate) mlir: Arc<CompiledKernel<MlirCompiler>>,
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

impl KernelRunner {
    pub fn new(logger: Arc<ServerLogger>) -> Self {
        let system = System::new_all();
        let max_shared_memory_size = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory()) as usize;

        const ALIGNMENT: u64 = 4;
        let memory_properties = MemoryDeviceProperties {
            max_page_size: max_shared_memory_size as u64,
            alignment: ALIGNMENT,
        };

        let memory_management_shared_memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
            logger,
            MemoryManagementOptions::new("Shared Memory"),
        );

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
            memory_management_shared_memory,
        }
    }
    pub fn prepare(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        cube_count: [u32; 3],
        bindings: BindingsResource,
        kind: ExecutionMode,
    ) -> Result<ScheduleTask, CompilationError> {
        let kernel_id = kernel.id();
        let kernel = if let Some(kernel) = self.compilation_cache.get(&kernel_id) {
            kernel
        } else {
            let kernel = kernel.compile(
                &mut Default::default(),
                &MlirCompilerOptions::default(),
                kind,
                kernel.address_type(),
            )?;
            self.compilation_cache
                .insert(kernel_id.clone(), CpuKernel::new(kernel));
            self.compilation_cache
                .get_mut(&kernel_id)
                .expect("Just inserted")
        };

        let cube_dim = kernel.mlir.cube_dim;

        let mlir_engine = kernel.mlir.repr.clone().unwrap();

        let task = ScheduleTask::Execute {
            mlir_engine,
            bindings,
            kind,
            cube_dim,
            cube_count,
        };

        Ok(task)
    }

    pub fn execute_data(
        &mut self,
        mlir_engine: MlirEngine,
        resources: BindingsResource,
        kind: ExecutionMode,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
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

        let mut mlir_data = MlirData::new(
            resources,
            &mlir_engine.0.shared_memories,
            &mut self.memory_management_shared_memory,
        );
        mlir_data.builtin.set_cube_dim(cube_dim);
        mlir_data.builtin.set_cube_count(cube_count);

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
