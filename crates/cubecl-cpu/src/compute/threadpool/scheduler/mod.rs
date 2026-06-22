use std::thread;

use crate::compute::{
    affinity::{CoreId, set_for_current},
    threadpool::{
        compute_task::ComputeTask,
        scheduler::{aside::AsideScheduler, naive::NaiveScheduler},
    },
};

pub mod aside;
pub mod naive;

pub enum SchedulerVariant {
    Naive,
    Aside,
}

pub const MAX_STACK_SIZE: usize = 16 * 1024 * 1024;
pub const DEFAULT_STACK_SIZE: usize = 64 * 1024 * 1024;

fn resolve_stack_size() -> usize {
    if let Ok(value) = std::env::var("CUBECL_CPU_STACK_SIZE")
        && let Ok(bytes) = value.parse::<usize>()
    {
        return bytes.max(MAX_STACK_SIZE);
    }
    if let Ok(value) = std::env::var("CUBECL_CPU_STACK_MB")
        && let Ok(mb) = value.parse::<usize>()
    {
        return (mb.saturating_mul(1024 * 1024)).max(MAX_STACK_SIZE);
    }
    DEFAULT_STACK_SIZE
}

pub enum Scheduler {
    Naive(NaiveScheduler),
    Crossbeam(AsideScheduler),
}

impl Scheduler {
    pub fn new(option: SchedulerVariant) -> Self {
        match option {
            SchedulerVariant::Naive => Scheduler::Naive(NaiveScheduler::new()),
            SchedulerVariant::Aside => Scheduler::Crossbeam(AsideScheduler::new()),
        }
    }

    pub fn send(&mut self, index: usize, task: ComputeTask) {
        match self {
            Scheduler::Naive(naive) => naive.send(index, task),
            Scheduler::Crossbeam(crossbeam) => crossbeam.send(index, task),
        }
    }
}

trait Worker: Sized + Send + 'static {
    fn work(self);
    fn spawn_thread(self, core_id: CoreId) {
        thread::Builder::new()
            .stack_size(resolve_stack_size())
            .spawn(move || {
                set_for_current(core_id);
                self.work()
            })
            .unwrap();
    }
}
