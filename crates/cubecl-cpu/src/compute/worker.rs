use std::sync::mpsc;
use std::thread;

use super::compute_task::{ComputeTask, Message};

pub const MAX_STACK_SIZE: usize = 16 * 1024 * 1024;
pub const DEFAULT_STACK_SIZE: usize = 64 * 1024 * 1024;

fn resolve_stack_size() -> usize {
    if let Ok(value) = std::env::var("CUBECL_CPU_STACK_SIZE")
        && let Ok(bytes) = value.parse::<usize>()
    {
        return bytes.max(MAX_STACK_SIZE);
    }
    if let Ok(value) = std::env::var("CUBECL_CPU_STACK_MB")
        && let Ok(mb) = value.parse::<usize>()
    {
        return (mb.saturating_mul(1024 * 1024)).max(MAX_STACK_SIZE);
    }
    DEFAULT_STACK_SIZE
}

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
            .stack_size(resolve_stack_size())
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
