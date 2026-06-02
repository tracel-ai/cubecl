use std::sync::mpsc;
use std::thread;

use crate::compute::{
    affinity::{CoreId, set_for_current},
    compute_task::ComputeTask,
};

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
    tx: mpsc::Sender<ComputeTask>,
}

impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}

impl Worker {
    pub fn new_with_affinity(core_id: CoreId) -> Self {
        let (tx, rx) = mpsc::channel();
        let inner_worker = InnerWorker { rx };
        thread::Builder::new()
            .stack_size(resolve_stack_size())
            .spawn(move || {
                set_for_current(core_id);
                inner_worker.work()
            })
            .unwrap();
        Self { tx }
    }
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let inner_worker = InnerWorker { rx };
        thread::Builder::new()
            .stack_size(resolve_stack_size())
            .spawn(move || inner_worker.work())
            .unwrap();
        Self { tx }
    }
    pub fn send_task(&mut self, compute_task: ComputeTask) {
        self.tx.send(compute_task).unwrap();
    }
}

struct InnerWorker {
    rx: mpsc::Receiver<ComputeTask>,
}

impl InnerWorker {
    fn work(self) {
        for compute_task in self.rx.into_iter() {
            compute_task.compute();
        }
    }
}
