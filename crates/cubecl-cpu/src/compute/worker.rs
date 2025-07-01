use dtor::dtor;
use std::sync::{Arc, mpsc};
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
    waiting: Arc<AtomicBool>,
    tx: mpsc::Sender<ComputeTask>,
}

impl Worker {
    pub fn new(thread_id: usize) -> Self {
        let (tx, rx) = mpsc::channel();
        let waiting = Arc::new(AtomicBool::new(true));
        let inner_worker = InnerWorker {
            thread_id,
            rx,
            waiting: Arc::clone(&waiting),
        };
        thread::spawn(move || inner_worker.work());
        Self {
            thread_id,
            tx,
            waiting,
        }
    }

    pub fn send_task(&mut self, compute_task: ComputeTask) {
        self.tx.send(compute_task).unwrap();
    }

    // Spin lock inspired by https://marabos.nl/atomics/building-spinlock.html
    pub fn sync(&self) {
        while !self.waiting.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }
}

struct InnerWorker {
    thread_id: usize,
    waiting: Arc<AtomicBool>,
    rx: mpsc::Receiver<ComputeTask>,
}

impl InnerWorker {
    fn work(self) {
        log::trace!("Thread numero {} started", self.thread_id);
        loop {
            let rx_fifo = self.rx.recv_timeout(Duration::from_millis(1));
            match rx_fifo {
                Ok(compute_task) => {
                    self.waiting.store(false, Ordering::Release);
                    compute_task.compute()
                }
                Err(mpsc::RecvTimeoutError::Timeout) => self.waiting.store(true, Ordering::Release),
                _ => break,
            }
            if STOP_SIGNAL.load(Ordering::Relaxed) {
                break;
            }
        }
        log::trace!("Thread numero {} stopped", self.thread_id);
    }
}
