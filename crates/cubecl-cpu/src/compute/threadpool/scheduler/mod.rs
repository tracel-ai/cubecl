use std::thread;

use crate::compute::{
    affinity::{CoreId, set_for_current},
    threadpool::{
        compute_task::ComputeTask,
        scheduler::{
            aside::AsideScheduler, dispatcher::DispatcherScheduler, naive::NaiveScheduler,
        },
    },
};

pub mod aside;
pub mod dispatcher;
pub mod naive;

pub enum SchedulerVariant {
    Naive,
    Aside,
    Dispatcher,
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
    Aside(AsideScheduler),
    Dispatcher(DispatcherScheduler),
}

impl Scheduler {
    pub fn new(option: SchedulerVariant) -> Self {
        match option {
            SchedulerVariant::Naive => Scheduler::Naive(NaiveScheduler::new()),
            SchedulerVariant::Aside => Scheduler::Aside(AsideScheduler::new()),
            SchedulerVariant::Dispatcher => Scheduler::Dispatcher(DispatcherScheduler::new()),
        }
    }

    pub fn send(&mut self, index: usize, task: ComputeTask) {
        match self {
            Scheduler::Naive(naive) => naive.send(index, task),
            Scheduler::Aside(aside) => aside.send(index, task),
            Scheduler::Dispatcher(dispatcher) => dispatcher.send(index, task),
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
