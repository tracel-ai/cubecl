use std::sync::mpsc;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use super::compute_task::ComputeTask;

#[derive(Debug)]
pub struct Worker {
    // TODO: A circular sync buffer with cache alignment would be a better fit, but for the moment a mpsc channel will do the job.
    tx: mpsc::Sender<ComputeTask>,
}

impl Worker {
    pub fn new(thread_id: usize, stop: Arc<AtomicBool>) -> Self {
        let (tx, rx) = mpsc::channel();
        let inner_worker = InnerWorker {
            thread_id,
            stop,
            rx,
        };
        thread::spawn(move || inner_worker.work());
        Self { tx }
    }

    pub fn send_task(&mut self, compute_task: ComputeTask) {
        self.tx.send(compute_task).unwrap();
    }
}

struct InnerWorker {
    thread_id: usize,
    stop: Arc<AtomicBool>,
    rx: mpsc::Receiver<ComputeTask>,
}

impl InnerWorker {
    fn work(self) {
        log::trace!("Thread numero {} started", self.thread_id);
        for compute_task in self.rx.iter() {
            compute_task.compute();
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
        }
        log::trace!("Thread numero {} stopped", self.thread_id);
    }
}
