use std::thread;

use cubecl_llvm::KernelRequirements;

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

    /// Sends the tasks of one launch, one per unit of the cube, in unit order.
    pub fn send_launch(
        &mut self,
        tasks: impl ExactSizeIterator<Item = ComputeTask>,
        requirements: &KernelRequirements,
    ) {
        match self {
            Scheduler::Naive(naive) => naive.send_launch(tasks),
            Scheduler::Aside(aside) => aside.send_launch(tasks),
            Scheduler::Dispatcher(dispatcher) => dispatcher.send_launch(tasks, requirements),
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
