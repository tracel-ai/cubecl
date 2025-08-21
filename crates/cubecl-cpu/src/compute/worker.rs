use std::sync::mpsc;
use std::thread;

use super::compute_task::{ComputeTask, Message};

pub const MAX_STACK_SIZE: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct Worker {
    // TODO: A circular sync buffer with cache alignment would be a better fit, but for the moment a mpsc channel will do the job.
    tx: mpsc::Sender<Message>,
}

impl Default for Worker {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let inner_worker = InnerWorker { rx };
        thread::Builder::new()
            .stack_size(MAX_STACK_SIZE)
            .spawn(move || inner_worker.work())
            .unwrap();
        Self { tx }
    }
}

impl Worker {
    pub fn send_task(&mut self, compute_task: ComputeTask) {
        self.tx.send(Message::ComputeTask(compute_task)).unwrap();
    }

    pub fn send_stop(&mut self, callback: mpsc::Sender<()>) {
        self.tx.send(Message::EndTask(callback)).unwrap();
    }
}

struct InnerWorker {
    rx: mpsc::Receiver<Message>,
}

impl InnerWorker {
    fn work(self) {
        for msg in self.rx.into_iter() {
            match msg {
                Message::ComputeTask(compute_task) => compute_task.compute(),
                Message::EndTask(end_task) => end_task.send(()).unwrap(),
            }
        }
    }
}
