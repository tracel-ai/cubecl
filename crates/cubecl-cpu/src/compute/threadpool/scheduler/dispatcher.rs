use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{self, AtomicUsize},
        mpsc,
    },
};

use crossbeam_utils::CachePadded;

use cubecl_llvm::KernelRequirements;

use crate::compute::{
    affinity::{CoreId, CoreOrder},
    threadpool::{ThreadTask, compute_task::ComputeTask, scheduler::Worker},
};

pub struct DispatcherScheduler {
    cpus: Vec<CoreId>,
    tx: Vec<mpsc::Sender<ComputeTask>>,
    lens: Vec<Arc<CachePadded<AtomicUsize>>>,
    spread: LaunchSpread,
}

impl Default for DispatcherScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatcherScheduler {
    pub fn new() -> Self {
        let order = CoreOrder::active();
        let mut scheduler = Self {
            spread: LaunchSpread::new(&order),
            cpus: order.cpus,
            tx: Vec::new(),
            lens: Vec::new(),
        };
        // One worker per active CPU; a barrier kernel grows the pool past it.
        scheduler.grow(scheduler.cpus.len());
        scheduler
    }

    /// Spawns workers until at least `workers` exist. Overflow workers past the CPU count
    /// round-robin over the active CPUs, so a cube with more units than CPUs still gets one
    /// thread per unit, which the `sync_cube` spin barrier needs to never queue two units of a
    /// cube behind each other.
    fn grow(&mut self, workers: usize) {
        while self.tx.len() < workers {
            let core_id = self.cpus[self.tx.len() % self.cpus.len()];
            let (worker, tx, len) = DispatcherWorker::new();
            worker.spawn_thread(core_id);
            self.tx.push(tx);
            self.lens.push(len);
        }
    }

    pub fn send_launch(
        &mut self,
        tasks: impl ExactSizeIterator<Item = ComputeTask>,
        requirements: &KernelRequirements,
    ) {
        if requirements.needs_parallelism {
            self.send_worker_per_unit(tasks);
        } else {
            self.send_spread(tasks);
        }
    }

    /// A cube barrier only completes if every unit of the cube is running, so each unit gets a
    /// worker of its own.
    fn send_worker_per_unit(&mut self, tasks: impl ExactSizeIterator<Item = ComputeTask>) {
        self.grow(tasks.len());
        for (worker, task) in tasks.enumerate() {
            self.dispatch(worker, task);
        }
    }

    fn send_spread(&mut self, tasks: impl Iterator<Item = ComputeTask>) {
        self.spread.start_launch();
        for task in tasks {
            let loads = self
                .lens
                .iter()
                .map(|len| len.load(atomic::Ordering::Relaxed));
            let worker = self.spread.pick(loads);
            self.dispatch(worker, task);
        }
    }

    fn dispatch(&self, worker: usize, task: ComputeTask) {
        // Counted before the send, so a worker finishing at once never takes the count below zero.
        self.lens[worker].fetch_add(1, atomic::Ordering::Relaxed);
        let _ = self.tx[worker].send(task);
    }
}

/// The units of the launch being sent so far, which choose the worker of its next unit.
///
/// Each unit runs over every cube, so a launch ends with its busiest physical core. A unit goes
/// to the core with the fewest units of the launch, then the lowest load, then to the worker on
/// it with the fewest units of the launch, lowest index first. So a launch with no more units
/// than physical cores gets a core each even while an earlier launch keeps the primaries busy,
/// a larger one fills the SMT siblings next, and past one unit per CPU it wraps evenly.
struct LaunchSpread {
    /// The physical core of each worker pinned to a distinct CPU, the only ones chosen.
    core_of: Vec<usize>,
    units_on_core: Vec<usize>,
    units_on_worker: Vec<usize>,
    core_load: Vec<usize>,
}

impl LaunchSpread {
    fn new(order: &CoreOrder) -> Self {
        Self {
            core_of: order.core_of.clone(),
            units_on_core: vec![0; order.physical_cores],
            units_on_worker: vec![0; order.core_of.len()],
            core_load: vec![0; order.physical_cores],
        }
    }

    fn start_launch(&mut self) {
        self.units_on_core.fill(0);
        self.units_on_worker.fill(0);
    }

    /// The worker of the next unit, given the live load of every worker in the pool.
    fn pick(&mut self, loads: impl Iterator<Item = usize>) -> usize {
        self.core_load.fill(0);
        for (worker, load) in loads.enumerate() {
            self.core_load[self.core_of[worker % self.core_of.len()]] += load;
        }
        let worker = (0..self.core_of.len())
            .min_by_key(|&worker| {
                let core = self.core_of[worker];
                (
                    self.units_on_core[core],
                    self.core_load[core],
                    self.units_on_worker[worker],
                )
            })
            .expect("a pool has at least one worker");
        self.units_on_core[self.core_of[worker]] += 1;
        self.units_on_worker[worker] += 1;
        worker
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

/// How long an idle worker polls `try_recv`, yielding the CPU between
/// misses, before parking in the blocking `recv`. A parked worker costs a
/// futex wake per launch, and a barrier kernel runs at the latency of its
/// last-woken unit, while the budget caps what an actually-idle pool burns.
const IDLE_POLL: std::time::Duration = std::time::Duration::from_micros(200);

impl Worker for DispatcherWorker {
    fn work(mut self) {
        loop {
            if self.aside.is_empty() {
                let mut received = None;
                let poll_start = std::time::Instant::now();
                while poll_start.elapsed() < IDLE_POLL {
                    match self.rx.try_recv() {
                        Ok(task) => {
                            received = Some(task);
                            break;
                        }
                        // The workers cover every logical CPU, so polling in earnest
                        // starves the client and any still-running units.
                        Err(_) => std::thread::yield_now(),
                    }
                }
                let task = match received {
                    Some(task) => Ok(task),
                    None => self.rx.recv(),
                };
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

#[cfg(test)]
mod tests {
    use super::LaunchSpread;
    use crate::compute::affinity::CoreOrder;

    /// Four physical cores with two SMT threads each, siblings listed after the primaries.
    fn smt_spread() -> LaunchSpread {
        LaunchSpread::new(&CoreOrder {
            cpus: Vec::new(),
            core_of: vec![0, 1, 2, 3, 0, 1, 2, 3],
            physical_cores: 4,
        })
    }

    /// Picks the workers of a launch of `units` units, with every load fixed at `loads`.
    fn launch(spread: &mut LaunchSpread, loads: &[usize], units: usize) -> Vec<usize> {
        spread.start_launch();
        (0..units)
            .map(|_| spread.pick(loads.iter().copied()))
            .collect()
    }

    #[test]
    fn two_units_of_the_launch_never_share_a_physical_core() {
        // Every load reads zero, as if each unit finished before the next was sent.
        assert_eq!(launch(&mut smt_spread(), &[0; 8], 4), [0, 1, 2, 3]);
    }

    #[test]
    fn a_launch_behind_a_running_one_stays_on_the_primaries() {
        // The earlier launch holds the primaries while the siblings idle.
        let loads = [1, 1, 1, 1, 0, 0, 0, 0];
        assert_eq!(launch(&mut smt_spread(), &loads, 4), [0, 1, 2, 3]);
    }

    #[test]
    fn a_busy_physical_core_is_chosen_last_in_the_launch() {
        let loads = [0, 2, 0, 0, 0, 0, 0, 0];
        assert_eq!(launch(&mut smt_spread(), &loads, 4), [0, 2, 3, 1]);
    }

    #[test]
    fn more_units_than_physical_cores_fill_the_siblings_then_spread_evenly() {
        let mut spread = smt_spread();
        assert_eq!(launch(&mut spread, &[0; 8], 8), [0, 1, 2, 3, 4, 5, 6, 7]);
        let mut units = [0; 8];
        for worker in launch(&mut spread, &[0; 8], 24) {
            units[worker] += 1;
        }
        assert_eq!(units, [3; 8]);
    }

    #[test]
    fn overflow_workers_load_the_core_they_share() {
        // Worker 8 is a barrier overflow worker pinned beside worker 0.
        let loads = [0, 0, 0, 0, 0, 0, 0, 0, 1];
        assert_eq!(launch(&mut smt_spread(), &loads, 4), [1, 2, 3, 0]);
    }
}
