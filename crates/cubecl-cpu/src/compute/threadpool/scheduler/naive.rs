use std::sync::mpsc;

use crate::compute::{
    affinity::get_active_cores,
    threadpool::{ThreadTask, compute_task::ComputeTask, scheduler::Worker},
};

pub struct NaiveSender {
    tx: Vec<mpsc::Sender<ComputeTask>>,
}

impl NaiveSender {
    pub fn new() -> Self {
        let tx = get_active_cores()
            .map(|core_id| {
                let (worker, tx) = NaiveWorker::new();
                worker.spawn_thread(core_id);
                tx
            })
            .collect();

        Self { tx }
    }

    pub fn send(&mut self, index: usize, elem: ComputeTask) {
        let _ = self.tx[index].send(elem);
    }
}

pub struct NaiveWorker {
    rx: mpsc::Receiver<ComputeTask>,
}

impl NaiveWorker {
    fn new() -> (Self, mpsc::Sender<ComputeTask>) {
        let (tx, rx) = mpsc::channel();
        let rx = Self { rx };
        (rx, tx)
    }
}

impl Worker for NaiveWorker {
    fn work(self) {
        loop {
            let task = self.rx.recv().unwrap();
            while !task.is_ready() {
                std::hint::spin_loop();
            }
            task.compute();
        }
    }
}
