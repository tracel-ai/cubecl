pub trait SchedulerStreamBackend: 'static {
    type Task: core::fmt::Debug;

    fn execute(&mut self, tasks: impl Iterator<Item = Self::Task>);
}

/// Manages multiple streams with synchronization logic based on shared bindings.
///
/// This struct handles the creation and alignment of streams to ensure proper synchronization
/// when bindings (e.g., buffers) are shared across different streams.
#[derive(Debug)]
pub struct SchedulerMultiStream<B: SchedulerStreamBackend> {}
