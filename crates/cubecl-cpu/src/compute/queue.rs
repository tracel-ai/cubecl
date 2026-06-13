use crate::{
    compiler::mlir_engine::MlirEngine,
    compute::{
        notification::{Notification, Notifications},
        schedule::{BindingsResource, ScheduleTask},
        threadpool::Threadpool,
    },
};
use cubecl_common::bytes::Bytes;
use cubecl_core::{CubeDim, stream_id::StreamId};
use cubecl_runtime::{
    logging::ServerLogger,
    storage::{BytesResource, ManagedResource},
};
use std::collections::HashMap;
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
                pending_notifications: HashMap::new(),
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
    pending_notifications: HashMap<StreamId, Vec<Notifications>>,
}

impl CpuExecutionQueueServer {
    fn execute_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write {
                stream_id,
                data,
                buffer,
            } => {
                self.flush_stream(stream_id);
                self.write(data, buffer)
            }
            ScheduleTask::Execute {
                stream_id,
                mlir_engine,
                bindings,
                cube_dim,
                cube_count,
                ..
            } => {
                self.flush_stream(stream_id);
                self.kernel(stream_id, mlir_engine, bindings, cube_dim, cube_count)
            }
        }
    }

    fn flush(&mut self) {
        for notifications in self.pending_notifications.drain().flat_map(|(_, v)| v) {
            notifications.wait();
        }
    }

    fn flush_stream(&mut self, stream_id: StreamId) {
        if let Some(notifications) = self.pending_notifications.remove(&stream_id) {
            for notifications in notifications {
                notifications.wait();
            }
        }
    }

    fn write(&mut self, data: Bytes, mut buffer: ManagedResource<BytesResource>) {
        buffer.resource_mut().write().copy_from_slice(&data);
    }

    fn kernel(
        &mut self,
        stream_id: StreamId,
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
    ) {
        let notifications = self
            .runner
            .execute_data(mlir_engine, bindings, cube_dim, cube_count);
        self.pending_notifications
            .entry(stream_id)
            .or_default()
            .push(notifications);
    }
}
