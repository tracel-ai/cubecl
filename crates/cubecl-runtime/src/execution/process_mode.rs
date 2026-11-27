use super::{OPEN_MODE, PROCESS_MODES, StatisticsCollector, StreamMode};

/// What the process does with the work it is asked to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessMode {
    /// Launches run. What the process does when no override is open.
    Execute,
    /// Launches and tune candidates queue their kernels and are discarded. The
    /// queue compiles in one batch, at the next launch that loads a kernel. A
    /// tune measures and decides nothing, and queues only the top of its plan.
    ///
    /// # Guarantees
    ///
    /// - A tune queues every candidate tied at its plan's highest priority, then
    ///   the next ones in plan order until as many as compile at once
    ///   (`compilation.parallelism`) have served the problem.
    /// - A candidate that declines the problem queues nothing and counts for
    ///   nothing toward that number.
    /// - A batch of the plan in which no candidate serves is queued whole, and
    ///   the top is taken from the next one.
    /// - A tunable set with no group states no priority, so all of it is queued.
    CompileOnly,
    /// Launches compile their kernels and are discarded. A tune measures its
    /// candidates for real, compiling at their turn those a `CompileOnly` pass
    /// left out.
    CompileAndAutotune,
}

impl ProcessMode {
    /// The process mode now.
    pub fn current() -> Self {
        OPEN_MODE.load().unwrap_or(Self::Execute)
    }

    /// The mode every stream has under it, unless a
    /// [`StreamModeOverride`](super::StreamModeOverride) sets its own.
    pub fn stream_mode(self) -> StreamMode {
        match self {
            Self::Execute => StreamMode::Execute,
            Self::CompileOnly | Self::CompileAndAutotune => StreamMode::Discard,
        }
    }

    /// Whether launches are discarded under this mode, unless a stream's override runs them.
    pub fn discards_launches(self) -> bool {
        self.stream_mode() == StreamMode::Discard
    }
}

/// Applies a process mode to the whole process while it lives, counting what it
/// triggers into a collector, on every thread and every device.
///
/// A build that compiles everything and then tunes with it compiled opens
/// two, one after the other, with one collector:
///
/// ```no_run
/// # fn warm_up() {}
/// use cubecl_runtime::execution::{ProcessModeOverride, ProcessMode, StatisticsCollector};
///
/// let collector = StatisticsCollector::new();
///
/// // Gather every kernel the warm-up reaches, and compile them together...
/// let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
/// warm_up();
/// core::mem::drop(compile);
///
/// // ...then tune with them compiled.
/// let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
/// warm_up();
/// core::mem::drop(tune);
///
/// let statistics = collector.statistics();
/// assert_eq!(statistics.compilation.settled_count(), statistics.compilation.registered_count);
/// ```
///
/// Overlapping overrides of one process mode and one collector compose, so one
/// opened while another is still open leaves the process mode until the last of
/// them drops. One process mode and one collector at a time: the process mode is
/// process-wide, and the work under it counts to one collector.
///
/// The process mode is read where a launch is issued, with relaxed ordering, so a
/// launch another thread had already begun issuing may still execute. What
/// is guaranteed is the launches issued by the thread that opened it, and
/// every launch issued after other threads observe it.
///
/// There is deliberately no configuration file or environment variable for
/// it. An override left open by accident turns the rest of the process into
/// launches that quietly do nothing and read back uninitialized memory, so
/// its lifetime belongs to a scope in the code that wants it.
#[derive(Debug)]
pub struct ProcessModeOverride {
    mode: ProcessMode,
}

impl ProcessModeOverride {
    /// Applies `mode` to the whole process until the override drops, counting
    /// what it triggers into `collector`. Opened while an override of the
    /// same mode and collector is, it joins that one.
    ///
    /// # Panics
    ///
    /// If an override of another process mode, or of another collector, is open.
    pub fn new(mode: ProcessMode, collector: &StatisticsCollector) -> Self {
        PROCESS_MODES.set(mode, collector);
        Self { mode }
    }

    /// The process mode it applies.
    pub fn mode(&self) -> ProcessMode {
        self.mode
    }
}

impl Drop for ProcessModeOverride {
    fn drop(&mut self) {
        PROCESS_MODES.unset();
    }
}

#[cfg(test)]
mod small_tests {
    use super::*;

    #[test]
    fn a_process_mode_that_discards_discards_on_every_stream() {
        assert_eq!(ProcessMode::CompileOnly.stream_mode(), StreamMode::Discard);
    }

    #[test]
    fn the_execute_process_mode_runs_every_stream() {
        assert_eq!(ProcessMode::Execute.stream_mode(), StreamMode::Execute);
    }
}

#[cfg(test)]
mod tests;
