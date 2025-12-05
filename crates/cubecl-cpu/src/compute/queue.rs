use cubecl_common::bytes::Bytes;
use cubecl_core::{CubeDim, ExecutionMode};
use cubecl_runtime::storage::BytesResource;

use crate::{
    compiler::{mlir_data::MlirData, mlir_engine::MlirEngine},
    compute::{schedule::ScheduleTask, scheduler::KernelRunner},
};

pub struct CpuExecutionQueue {}

struct CpuExecutionQueueServer {
    runner: KernelRunner,
}

impl CpuExecutionQueueServer {
    pub fn execute_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write { data, buffer } => self.write(data, buffer),
            ScheduleTask::Execute {
                mlir_engine,
                mlir_data,
                kind,
                cube_dim,
            } => self.kernel(mlir_engine, mlir_data, kind, cube_dim),
        }
    }

    pub fn write(&mut self, data: Bytes, mut buffer: BytesResource) {
        buffer.write().copy_from_slice(&data);
    }

    pub fn kernel(
        &mut self,
        mlir_engine: MlirEngine,
        mlir_data: MlirData,
        kind: ExecutionMode,
        cube_dim: CubeDim,
    ) {
        self.runner
            .execute_data(mlir_engine, mlir_data, kind, cube_dim)
    }
}
