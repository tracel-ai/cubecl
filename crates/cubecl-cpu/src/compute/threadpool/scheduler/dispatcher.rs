use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{self, AtomicUsize},
        mpsc,
    },
};

use crossbeam_utils::CachePadded;

use crate::compute::{
    affinity::get_active_cores,
    threadpool::{ThreadTask, compute_task::ComputeTask, scheduler::Worker},
};

pub struct DispatcherScheduler {
    tx: Vec<mpsc::Sender<ComputeTask>>,
    lens: Arc<[CachePadded<AtomicUsize>]>,
}

impl DispatcherScheduler {
    pub fn new() -> Self {
        let cores: Vec<_> = get_active_cores().collect();
        let lens: Vec<_> = cores
            .iter()
            .map(|_| CachePadded::new(AtomicUsize::new(0)))
            .collect();
        let lens: Arc<[_]> = lens.into();
        let tx = cores
            .iter()
            .enumerate()
            .map(|(thread_id, &core_id)| {
                let (worker, tx) = DispatcherWorker::new(thread_id, lens.clone());
                worker.spawn_thread(core_id);
                tx
            })
            .collect();

        Self { tx, lens }
    }

    pub fn send(&mut self, index: usize, task: ComputeTask) {
        let _ = self.tx[index].send(task);
        self.lens[index].fetch_add(1, atomic::Ordering::Relaxed);
    }
}

pub struct DispatcherWorker {
    rx: mpsc::Receiver<ComputeTask>,
    aside: VecDeque<ComputeTask>,
    thread_id: usize,
    lens: Arc<[CachePadded<AtomicUsize>]>,
}

impl DispatcherWorker {
    fn new(
        thread_id: usize,
        lens: Arc<[CachePadded<AtomicUsize>]>,
    ) -> (Self, mpsc::Sender<ComputeTask>) {
        let (tx, rx) = mpsc::channel();
        let aside = VecDeque::with_capacity(4);
        let rx = Self {
            rx,
            aside,
            thread_id,
            lens,
        };
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
                        self.lens[self.thread_id].fetch_sub(1, atomic::Ordering::Relaxed);
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
                    self.lens[self.thread_id].fetch_sub(1, atomic::Ordering::Relaxed);
                    false
                } else {
                    std::hint::spin_loop();
                    true
                }
            });
        }
    }
}
