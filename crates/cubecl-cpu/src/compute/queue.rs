use crate::{
    compiler::{mlir_data::MlirData, mlir_engine::MlirEngine},
    compute::{schedule::ScheduleTask, scheduler::KernelRunner},
};
use cubecl_common::bytes::Bytes;
use cubecl_core::{CubeDim, ExecutionMode};
use cubecl_runtime::storage::BytesResource;
use std::sync::{OnceLock, mpsc::SyncSender};

static INSTANCE: OnceLock<CpuExecutionQueue> = OnceLock::new();

#[derive(Clone)]
pub struct CpuExecutionQueue {
    sender: SyncSender<QueueItem>,
}

enum QueueItem {
    Task(ScheduleTask),
    Flush(std::sync::mpsc::SyncSender<()>),
}

impl CpuExecutionQueue {
    pub fn push(&self, task: ScheduleTask) {
        self.sender.send(QueueItem::Task(task)).unwrap();
    }

    pub fn flush(&self) {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.sender.send(QueueItem::Flush(sender)).unwrap();
        receiver.recv().unwrap()
    }
    pub fn get() -> Self {
        INSTANCE.get_or_init(Self::create).clone()
    }

    fn create() -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(32);

        std::thread::spawn(move || {
            let mut server = CpuExecutionQueueServer::default();
            loop {
                match receiver.recv() {
                    Ok(item) => match item {
                        QueueItem::Task(task) => server.execute_task(task),
                        QueueItem::Flush(sender) => sender.send(()).unwrap(),
                    },
                    Err(err) => panic!("{err:?}"),
                }
            }
        });

        Self { sender }
    }
}

#[derive(Default)]
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
