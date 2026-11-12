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

/// Starts the worker of one CPU, returning the channel that feeds it and its count of queued
/// units.
type SpawnWorker<T> =
    Box<dyn FnMut(CoreId) -> (mpsc::Sender<T>, Arc<CachePadded<AtomicUsize>>) + Send>;

/// Places the units of a launch on pinned workers. Generic over the unit sent so the placement
/// can be driven through the real send path with workers a test supplies.
pub struct DispatcherScheduler<T = ComputeTask> {
    cpus: Vec<CoreId>,
    tx: Vec<mpsc::Sender<T>>,
    lens: Vec<Arc<CachePadded<AtomicUsize>>>,
    spread: LaunchSpread,
    spawn: SpawnWorker<T>,
}

impl Default for DispatcherScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatcherScheduler {
    pub fn new() -> Self {
        Self::with_spawn(
            CoreOrder::active(),
            Box::new(|core_id| {
                let (worker, tx, len) = DispatcherWorker::new();
                worker.spawn_thread(core_id);
                (tx, len)
            }),
        )
    }

    pub fn send_launch(
        &mut self,
        tasks: impl ExactSizeIterator<Item = ComputeTask>,
        requirements: &KernelRequirements,
    ) {
        self.send(tasks, requirements.needs_parallelism);
    }
}

impl<T> DispatcherScheduler<T> {
    fn with_spawn(order: &CoreOrder, spawn: SpawnWorker<T>) -> Self {
        let mut scheduler = Self {
            spread: LaunchSpread::new(order),
            cpus: order.cpus.clone(),
            tx: Vec::new(),
            lens: Vec::new(),
            spawn,
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
            let (tx, len) = (self.spawn)(core_id);
            self.tx.push(tx);
            self.lens.push(len);
        }
    }

    fn send(&mut self, tasks: impl ExactSizeIterator<Item = T>, needs_parallelism: bool) {
        if needs_parallelism {
            self.send_worker_per_unit(tasks);
        } else {
            self.send_spread(tasks);
        }
    }

    /// A cube barrier only completes if every unit of the cube is running, so each unit gets a
    /// worker of its own.
    fn send_worker_per_unit(&mut self, tasks: impl ExactSizeIterator<Item = T>) {
        self.grow(tasks.len());
        for (worker, task) in tasks.enumerate() {
            self.dispatch(worker, task);
        }
    }

    fn send_spread(&mut self, tasks: impl Iterator<Item = T>) {
        self.spread.start_launch(
            self.lens
                .iter()
                .map(|len| len.load(atomic::Ordering::Relaxed)),
        );
        for task in tasks {
            let worker = self.spread.pick();
            self.dispatch(worker, task);
        }
    }

    fn dispatch(&self, worker: usize, task: T) {
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
    /// The queued units of each physical core, read once at the start of the launch and
    /// advanced as the launch places its own units.
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

    /// Starts a launch, given the queued units of every worker in the pool.
    fn start_launch(&mut self, loads: impl Iterator<Item = usize>) {
        self.units_on_core.fill(0);
        self.units_on_worker.fill(0);
        self.core_load.fill(0);
        for (worker, load) in loads.enumerate() {
            self.core_load[self.core_of[worker % self.core_of.len()]] += load;
        }
    }

    /// The worker of the next unit of the launch.
    fn pick(&mut self) -> usize {
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
        let core = self.core_of[worker];
        self.units_on_core[core] += 1;
        self.units_on_worker[worker] += 1;
        self.core_load[core] += 1;
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
    use std::sync::{Arc, Mutex};

    use super::{DispatcherScheduler, LaunchSpread};
    use crate::compute::affinity::{CoreId, CoreOrder};

    /// Four physical cores with two SMT threads each, siblings listed after the primaries.
    fn smt_order() -> CoreOrder {
        CoreOrder {
            cpus: (0..8).map(CoreId::new_for_test).collect(),
            core_of: vec![0, 1, 2, 3, 0, 1, 2, 3],
            physical_cores: 4,
            topology_known: true,
        }
    }

    fn smt_spread() -> LaunchSpread {
        LaunchSpread::new(&smt_order())
    }

    /// Picks the workers of a launch of `units` units, with every load fixed at `loads`.
    fn launch(spread: &mut LaunchSpread, loads: &[usize], units: usize) -> Vec<usize> {
        spread.start_launch(loads.iter().copied());
        (0..units).map(|_| spread.pick()).collect()
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

    /// A scheduler whose workers are channels the test reads, in place of pinned threads. Each
    /// worker keeps the unit numbers it received.
    struct Harness {
        scheduler: DispatcherScheduler<usize>,
        inboxes: Arc<Mutex<Vec<std::sync::mpsc::Receiver<usize>>>>,
    }

    impl Harness {
        fn new() -> Self {
            let inboxes = Arc::new(Mutex::new(Vec::new()));
            let spawned = inboxes.clone();
            let scheduler = DispatcherScheduler::with_spawn(
                &smt_order(),
                Box::new(move |_| {
                    let (tx, rx) = std::sync::mpsc::channel();
                    spawned.lock().unwrap().push(rx);
                    (tx, Arc::new(Default::default()))
                }),
            );
            Self { scheduler, inboxes }
        }

        /// The sorted physical cores of the units received since the last call.
        fn cores_received(&self) -> Vec<usize> {
            let order = smt_order();
            let mut cores: Vec<usize> = self
                .received()
                .iter()
                .enumerate()
                .flat_map(|(worker, units)| vec![order.core_of[worker]; units.len()])
                .collect();
            cores.sort_unstable();
            cores
        }

        /// For each worker, the units it received since the last call.
        fn received(&self) -> Vec<Vec<usize>> {
            self.inboxes
                .lock()
                .unwrap()
                .iter()
                .map(|rx| rx.try_iter().collect())
                .collect()
        }
    }

    #[test]
    fn a_launch_of_up_to_the_physical_cores_lands_one_unit_per_core() {
        let mut harness = Harness::new();
        harness.scheduler.send(0..4, false);
        let cores = harness.cores_received();
        assert_eq!(cores, [0, 1, 2, 3]);
    }

    #[test]
    fn a_second_launch_behind_the_first_still_takes_a_core_per_unit() {
        // The first launch's units are never finished, so its workers read as busy.
        let mut harness = Harness::new();
        harness.scheduler.send(0..4, false);
        harness.received();
        harness.scheduler.send(0..4, false);
        let cores = harness.cores_received();
        assert_eq!(cores, [0, 1, 2, 3]);
    }

    #[test]
    fn a_barrier_launch_gives_worker_i_to_unit_i_and_grows_past_the_cpus() {
        let mut harness = Harness::new();
        harness.scheduler.send(0..10, true);
        let received = harness.received();
        assert_eq!(received.len(), 10);
        for (worker, units) in received.iter().enumerate() {
            assert_eq!(units, &[worker]);
        }
    }
}
