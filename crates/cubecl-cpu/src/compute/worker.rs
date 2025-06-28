use std::sync::mpsc;
use std::time::Duration;
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
        loop {
            let rx_fifo = self.rx.recv_timeout(Duration::from_millis(100));
            if let Ok(compute_task) = rx_fifo {
                compute_task.compute();
            }
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
        }
        log::trace!("Thread numero {} stopped", self.thread_id);
    }
}
