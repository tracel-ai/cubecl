use dtor::dtor;
use std::sync::mpsc;
use std::time::Duration;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use super::compute_task::ComputeTask;

pub static STOP_SIGNAL: AtomicBool = AtomicBool::new(false);

#[dtor]
fn stop_program() {
    STOP_SIGNAL.store(true, Ordering::Relaxed);
}

#[derive(Debug)]
pub struct Worker {
    // TODO: A circular sync buffer with cache alignment would be a better fit, but for the moment a mpsc channel will do the job.
    pub thread_id: usize,
    tx: mpsc::Sender<ComputeTask>,
}

impl Worker {
    pub fn new(thread_id: usize) -> Self {
        let (tx, rx) = mpsc::channel();
        let inner_worker = InnerWorker { thread_id, rx };
        thread::spawn(move || inner_worker.work());
        Self { thread_id, tx }
    }

    pub fn send_task(&mut self, compute_task: ComputeTask) {
        self.tx.send(compute_task).unwrap();
    }
}

struct InnerWorker {
    thread_id: usize,
    rx: mpsc::Receiver<ComputeTask>,
}

impl InnerWorker {
    fn work(self) {
        log::trace!("Thread numero {} started", self.thread_id);
        loop {
            let rx_fifo = self.rx.recv_timeout(Duration::from_millis(100));
            if let Ok(compute_task) = rx_fifo {
                compute_task.compute();
                println!("Computing!");
            }
            if STOP_SIGNAL.load(Ordering::Relaxed) {
                break;
            }
        }
        log::trace!("Thread numero {} stopped", self.thread_id);
    }
}
