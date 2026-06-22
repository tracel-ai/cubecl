use std::sync::mpsc;

use crate::compute::{
    affinity::get_active_cores,
    threadpool::{ThreadTask, compute_task::ComputeTask, scheduler::Worker},
};

pub struct AsideScheduler {
    tx: Vec<mpsc::Sender<ComputeTask>>,
}

impl AsideScheduler {
    pub fn new() -> Self {
        let tx = get_active_cores()
            .map(|core_id| {
                let (worker, tx) = AsideWorker::new();
                worker.spawn_thread(core_id);
                tx
            })
            .collect();

        Self { tx }
    }

    pub fn send(&mut self, index: usize, task: ComputeTask) {
        let _ = self.tx[index].send(task);
    }
}

pub struct AsideWorker {
    rx: mpsc::Receiver<ComputeTask>,
    aside: Option<ComputeTask>,
}

impl AsideWorker {
    fn new() -> (Self, mpsc::Sender<ComputeTask>) {
        let (tx, rx) = mpsc::channel();
        let aside = None;
        let rx = Self { rx, aside };
        (rx, tx)
    }
}

impl Worker for AsideWorker {
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
