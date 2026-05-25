use crate::{
    compiler::mlir_engine::MlirEngine,
    compute::{
        notification::{Notification, Notifications},
        schedule::{BindingsResource, ScheduleTask},
        threadpool::Threadpool,
    },
};
use cubecl_common::bytes::Bytes;
use cubecl_core::CubeDim;
use cubecl_runtime::{logging::ServerLogger, storage::BytesResource};
use std::sync::{Arc, OnceLock, mpsc::SyncSender};

static INSTANCE: OnceLock<CpuExecutionQueue> = OnceLock::new();

#[derive(Clone)]
/// There is a single execution queue instance for the whole CPU runtime.
///
/// This type allows users to send tasks to the global execution queue.
pub struct CpuExecutionQueue {
    sender: SyncSender<QueueItem>,
}

enum QueueItem {
    Task(ScheduleTask),
    Flush(Notification),
}

impl CpuExecutionQueue {
    /// Adds a new task to the queue.
    pub fn add(&self, task: ScheduleTask) {
        self.sender.send(QueueItem::Task(task)).unwrap();
    }

    /// Flushes the queue, making sure all enqueued tasks before this point are executed.
    pub fn flush(&self) {
        let notification = Notification::new();
        self.sender
            .send(QueueItem::Flush(notification.clone()))
            .unwrap();
        notification.wait();
    }

    /// Resolves the global execution queue instance.
    pub fn get(logger: Arc<ServerLogger>) -> Self {
        INSTANCE.get_or_init(|| Self::init(logger)).clone()
    }

    fn init(logger: Arc<ServerLogger>) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(32);

        std::thread::spawn(move || {
            let mut server = CpuExecutionQueueServer {
                runner: Threadpool::new(logger),
                pending_notifications: Vec::new(),
            };

            loop {
                match receiver.recv() {
                    Ok(item) => match item {
                        QueueItem::Task(task) => server.execute_task(task),
                        QueueItem::Flush(notification) => {
                            server.flush();
                            notification.send();
                        }
                    },
                    Err(err) => panic!("{err}"),
                }
            }
        });

        Self { sender }
    }
}

struct CpuExecutionQueueServer {
    runner: Threadpool,
    pending_notifications: Vec<Notifications>,
}

impl CpuExecutionQueueServer {
    fn execute_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write { data, buffer } => self.write(data, buffer),
            ScheduleTask::Execute {
                mlir_engine,
                bindings,
                cube_dim,
                cube_count,
                ..
            } => self.kernel(mlir_engine, bindings, cube_dim, cube_count),
        }
    }

    fn flush(&mut self) {
        for notifications in self.pending_notifications.drain(..) {
            notifications.wait();
        }
    }

    fn write(&mut self, data: Bytes, mut buffer: BytesResource) {
        self.flush();
        buffer.write().copy_from_slice(&data);
    }

    fn kernel(
        &mut self,
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
    ) {
        let notifications = self
            .runner
            .execute_data(mlir_engine, bindings, cube_dim, cube_count);
        self.pending_notifications.push(notifications);
    }
}
