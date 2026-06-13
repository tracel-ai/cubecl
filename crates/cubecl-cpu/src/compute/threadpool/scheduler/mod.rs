use crate::compute::threadpool::{
    compute_task::ComputeTask,
    scheduler::simple::{SimpleScheduler, SimpleSender},
};

pub mod naive;
pub mod simple;

pub enum SchedulerVariant {
    Simple,
}

pub enum Sender {
    Simple(SimpleSender<ComputeTask>),
}

impl Sender {
    pub fn new(option: SchedulerVariant) -> Self {
        match option {
            SchedulerVariant::Simple => Sender::Simple(SimpleSender::new()),
        }
    }
    pub fn send(&mut self, index: usize, task: ComputeTask) {
        match self {
            Sender::Simple(simple) => simple.send(index, task),
        }
    }
}

pub enum Scheduler {
    Simple(SimpleScheduler<ComputeTask>),
}

pub trait ThreadTask {
    fn get_stream_id(&self) -> usize;
    fn is_ready(&self) -> bool;
}
