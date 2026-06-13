use std::sync::mpsc;

use crate::compute::threadpool::compute_task::ComputeTask;

pub struct NaiveSender {
    tx: Vec<mpsc::Sender<ComputeTask>>,
}

impl NaiveSender {
    pub fn send(&mut self, index: usize, elem: ComputeTask) {
        let _ = self.tx[index].send(elem);
    }
}

pub struct NaiveScheduler {
    rx: mpsc::Receiver<ComputeTask>,
}

impl NaiveScheduler {
    pub fn pop(&mut self) -> ComputeTask {
        self.rx.recv().unwrap()
    }
}
