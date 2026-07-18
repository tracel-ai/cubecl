use super::{compute_task::ComputeTask, schedule::BindingsResource, worker::Worker};
use crate::{
    compiler::{
        PlironCompiler,
        jit::{data::PlironData, engine::PlironEngine},
    },
    compute::{affinity::get_active_cores, notification::Notifications},
};
use cubecl_core::{
    CubeDim, MemoryConfiguration, ir::MemoryDeviceProperties, prelude::CompiledKernel,
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryManagement, MemoryManagementOptions},
    storage::BytesStorage,
};
use std::{fmt::Debug, sync::Arc};
use sysinfo::System;

/// The kernel runner is responsible to manage shared memory as well as threads to execute kernels.
///
/// A single kernel runner is currently used for all kernels.
/// To register work, you have to use the execution queue.
pub struct Threadpool {
    workers: Vec<Worker>,
    _memory_management_shared_memory: MemoryManagement<BytesStorage>,
}

/// A compiled cpu kernel.
pub struct CpuKernel {
    pub(crate) mlir: Arc<CompiledKernel<PlironCompiler>>,
}

impl CpuKernel {
    pub fn new(kernel: CompiledKernel<PlironCompiler>) -> Self {
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

impl Debug for Threadpool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.workers)
    }
}

impl Threadpool {
    pub fn new(logger: Arc<ServerLogger>) -> Self {
        let mut system = System::new();
        system.refresh_memory();
        let max_page_size = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory());

        const ALIGNMENT: u64 = 4;
        let memory_properties = MemoryDeviceProperties {
            max_page_size,
            alignment: ALIGNMENT,
        };

        let _memory_management_shared_memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
            logger,
            MemoryManagementOptions::new("Shared Memory"),
        );

        let workers = get_active_cores().map(Worker::new_with_affinity).collect();

        Self {
            workers,
            _memory_management_shared_memory,
        }
    }
    pub fn execute_data(
        &mut self,
        pliron_engine: PlironEngine,
        _resources: BindingsResource,
        cube_dim: CubeDim,
        _cube_count: [u32; 3],
    ) -> Notifications {
        let cube_dim_size = cube_dim.num_elems();

        if cube_dim_size > self.workers.len() as u32 {
            self.workers
                .extend((0..cube_dim_size - self.workers.len() as u32).map(|_| Worker::new()));
        }

        let mlir_data = PlironData;

        let notifications = Notifications::new(cube_dim_size);
        let mut workers = self.workers.iter_mut();
        for _unit_pos_x in 0..cube_dim.x {
            for _unit_pos_y in 0..cube_dim.y {
                for _unit_pos_z in 0..cube_dim.z {
                    // let unit_pos = [unit_pos_x, unit_pos_y, unit_pos_z];
                    let worker = workers.next().expect("The CubeDim are too large");
                    let pliron_engine = pliron_engine.clone();
                    let pliron_data = mlir_data.clone();
                    // pliron_data.builtin.set_unit_pos(unit_pos);

                    let notifications = notifications.clone();
                    let compute_task = ComputeTask {
                        pliron_engine,
                        pliron_data,
                        notifications,
                    };
                    worker.send_task(compute_task);
                }
            }
        }

        notifications
    }
}
