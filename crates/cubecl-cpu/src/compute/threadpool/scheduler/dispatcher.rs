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
    affinity::{CoreId, get_active_cores},
    threadpool::{ThreadTask, compute_task::ComputeTask, scheduler::Worker},
};

pub struct DispatcherScheduler {
    cores: Vec<CoreId>,
    tx: Vec<mpsc::Sender<ComputeTask>>,
    lens: Vec<Arc<CachePadded<AtomicUsize>>>,
}

impl Default for DispatcherScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatcherScheduler {
    pub fn new() -> Self {
        let cores: Vec<_> = get_active_cores().collect();
        let mut scheduler = Self {
            cores,
            tx: Vec::new(),
            lens: Vec::new(),
        };
        // Start with one worker per active core; the pool grows on demand when a
        // parallel cube needs more units than we currently have workers.
        let cores = scheduler.cores.len();
        scheduler.ensure_workers(cores);
        scheduler
    }

    /// Spawns workers until at least `n` exist. Overflow workers past the core
    /// count round-robin over the active cores, so a cube with more units than
    /// cores still gets one thread per unit — required so the `sync_cube` spin
    /// barrier never queues two units of the same cube behind each other.
    pub fn ensure_workers(&mut self, n: usize) {
        while self.tx.len() < n {
            let core_id = self.cores[self.tx.len() % self.cores.len()];
            let (worker, tx, len) = DispatcherWorker::new();
            worker.spawn_thread(core_id);
            self.tx.push(tx);
            self.lens.push(len);
        }
    }

    pub fn send(&mut self, index: usize, task: ComputeTask) {
        let target = if
        /*task.pliron_engine.0.needs_parallelism*/
        false {
            // Barrier kernels need one dedicated worker per unit; the caller
            // grew the pool via `ensure_workers` so `index` is in range.
            index
        } else {
            // Independent units load-balance onto the least-loaded worker. The
            // incoming `index` is a unit position that can exceed the worker
            // count, so never use it directly here.
            let mut best = 0;
            let mut min_value = self.lens[0].load(atomic::Ordering::Relaxed);
            for i in 1..self.lens.len() {
                let len = self.lens[i].load(atomic::Ordering::Relaxed);
                if len < min_value {
                    best = i;
                    min_value = len;
                }
            }
            best
        };
        let _ = self.tx[target].send(task);
        self.lens[target].fetch_add(1, atomic::Ordering::Relaxed);
    }
}

pub struct DispatcherWorker {
    rx: mpsc::Receiver<ComputeTask>,
    aside: VecDeque<ComputeTask>,
    len: Arc<CachePadded<AtomicUsize>>,
}

impl DispatcherWorker {
    fn new() -> (
        Self,
        mpsc::Sender<ComputeTask>,
        Arc<CachePadded<AtomicUsize>>,
    ) {
        let (tx, rx) = mpsc::channel();
        let aside = VecDeque::with_capacity(4);
        let len = Arc::new(CachePadded::new(AtomicUsize::new(0)));
        let worker = Self {
            rx,
            aside,
            len: len.clone(),
        };
        (worker, tx, len)
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
                        self.len.fetch_sub(1, atomic::Ordering::Relaxed);
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
                    self.len.fetch_sub(1, atomic::Ordering::Relaxed);
                    false
                } else {
                    std::hint::spin_loop();
                    true
                }
            });
        }
    }
}
