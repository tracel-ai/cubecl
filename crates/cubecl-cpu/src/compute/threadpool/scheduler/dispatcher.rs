use std::{collections::VecDeque, sync::mpsc};

use crate::compute::{
    affinity::get_active_cores,
    threadpool::{ThreadTask, compute_task::ComputeTask, scheduler::Worker},
};

pub struct DispatcherScheduler {
    tx: Vec<mpsc::Sender<ComputeTask>>,
}

impl DispatcherScheduler {
    pub fn new() -> Self {
        let tx = get_active_cores()
            .map(|core_id| {
                let (worker, tx) = DispatcherWorker::new();
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

pub struct DispatcherWorker {
    rx: mpsc::Receiver<ComputeTask>,
    aside: VecDeque<ComputeTask>,
}

impl DispatcherWorker {
    fn new() -> (Self, mpsc::Sender<ComputeTask>) {
        let (tx, rx) = mpsc::channel();
        let aside = VecDeque::with_capacity(4);
        let rx = Self { rx, aside };
        (rx, tx)
    }
}

impl Worker for DispatcherWorker {
    fn work(mut self) {
        loop {
            if self.aside.is_empty() {
                let task = self.rx.recv();
                if let Ok(mut task) = task {
                    if task.is_ready() {
                        task.compute();
                    } else {
                        self.aside.push_back(task);
                    }
                }
            } else if self.aside.len() < 4 {
                let task = self.rx.try_recv();
                if let Ok(task) = task {
                    self.aside.push_back(task);
                }
            }
            self.aside.retain_mut(|elem| {
                if elem.is_ready() {
                    elem.compute();
                    false
                } else {
                    std::hint::spin_loop();
                    true
                }
            });
        }
    }
}
