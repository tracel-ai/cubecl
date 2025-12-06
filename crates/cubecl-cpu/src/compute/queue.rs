use crate::{
    compiler::mlir_engine::MlirEngine,
    compute::{
        schedule::{BindingsResource, ScheduleTask},
        scheduler::KernelRunner,
    },
};
use cubecl_common::bytes::Bytes;
use cubecl_core::{CubeDim, ExecutionMode};
use cubecl_runtime::{logging::ServerLogger, storage::BytesResource};
use std::sync::{Arc, OnceLock, mpsc::SyncSender};

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
    pub fn get(logger: Arc<ServerLogger>) -> Self {
        INSTANCE.get_or_init(|| Self::create(logger)).clone()
    }

    fn create(logger: Arc<ServerLogger>) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(32);

        std::thread::spawn(move || {
            let mut server = CpuExecutionQueueServer {
                runner: KernelRunner::new(logger),
            };

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

struct CpuExecutionQueueServer {
    runner: KernelRunner,
}

impl CpuExecutionQueueServer {
    pub fn execute_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write { data, buffer } => self.write(data, buffer),
            ScheduleTask::Execute {
                mlir_engine,
                bindings,
                kind,
                cube_dim,
                cube_count,
            } => self.kernel(mlir_engine, bindings, kind, cube_dim, cube_count),
        }
    }

    pub fn write(&mut self, data: Bytes, mut buffer: BytesResource) {
        buffer.write().copy_from_slice(&data);
    }

    pub fn kernel(
        &mut self,
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        kind: ExecutionMode,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
    ) {
        self.runner
            .execute_data(mlir_engine, bindings, kind, cube_dim, cube_count)
    }
}
