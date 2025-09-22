pub trait SchedulerStreamBackend: 'static {
    type Task: core::fmt::Debug;

    fn execute(&mut self, tasks: impl Iterator<Item = Self::Task>);
}

#[derive(Debug)]
pub struct SchedulerMultiStream {}
