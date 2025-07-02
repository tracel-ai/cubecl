use std::sync::{Arc, mpsc};
use std::time::Duration;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use dtor::dtor;

use super::compute_task::ComputeTask;

pub static STOP_SIGNAL: AtomicBool = AtomicBool::new(false);

#[dtor]
fn stop_program() {
    STOP_SIGNAL.store(true, Ordering::Release);
}

#[derive(Debug)]
pub struct Worker {
    // TODO: A circular sync buffer with cache alignment would be a better fit, but for the moment a mpsc channel will do the job.
    waiting: Arc<AtomicBool>,
    tx: mpsc::Sender<ComputeTask>,
}

impl Worker {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let waiting = Arc::new(AtomicBool::new(true));
        let inner_worker = InnerWorker {
            rx,
            waiting: Arc::clone(&waiting),
        };
        thread::spawn(move || inner_worker.work());
        Self { tx, waiting }
    }

    pub fn send_task(&mut self, compute_task: ComputeTask) {
        self.waiting.store(false, Ordering::Release);
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
    waiting: Arc<AtomicBool>,
    rx: mpsc::Receiver<ComputeTask>,
}

impl InnerWorker {
    fn work(self) {
        loop {
            let rx_fifo = self.rx.recv_timeout(Duration::from_millis(1));
            match rx_fifo {
                Ok(compute_task) => compute_task.compute(),
                Err(mpsc::RecvTimeoutError::Timeout) => self.waiting.store(true, Ordering::Release),
                _ => (),
            }
            if STOP_SIGNAL.load(Ordering::Acquire) {
                break;
            }
        }
    }
}
