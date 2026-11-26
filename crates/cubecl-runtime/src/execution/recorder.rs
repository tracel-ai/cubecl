use cubecl_environment::sync::Arc;

use super::{Countable, IssuedLaunch, PROCESS_MODES, StatisticsCollector, StatisticsCounter};
use crate::id::KernelId;
use crate::server::LaunchError;

/// Where the work under an override is counted: what a
/// [`Registration`](super::Registration) holds, and the open override hands
/// out. A clone counts into the same collector, which is how a kernel queued
/// under one override is counted to it when its batch compiles.
///
/// Public for the kernel loader, which lives in `cubecl-server`.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct StatisticsRecorder {
    counter: Arc<StatisticsCounter>,
}

impl StatisticsRecorder {
    /// Where the work triggered now counts: the collector of the override
    /// open where the launch running on this thread was issued, or, outside a
    /// launch, of the one open now, if one is.
    pub fn current() -> Option<Self> {
        IssuedLaunch::running_recorder().unwrap_or_else(Self::active)
    }

    /// List `kernel` as one the work needed and could not obtain, once: the
    /// work may meet the same failure again, at its batch and at a later
    /// launch of it. Only for a kernel a launch of the work asked for; an
    /// autotune candidate's failure is its race's alone.
    pub fn fail_work(&self, kernel: &KernelId, error: &LaunchError) {
        self.counter.count_work_failure(kernel, error);
    }
}

impl StatisticsRecorder {
    /// A recorder that counts into `counter`.
    pub fn new(counter: Arc<StatisticsCounter>) -> Self {
        Self { counter }
    }

    /// Where the work the override open now triggers counts, or `None` when
    /// none is open.
    pub fn active() -> Option<Self> {
        PROCESS_MODES.recorder()
    }

    /// Whether it counts into `collector`.
    pub fn counts_into(&self, collector: &StatisticsCollector) -> bool {
        Arc::ptr_eq(&self.counter, &collector.to_recorder().counter)
    }

    /// Count one item of `O` registered.
    pub fn count_registered<O: Countable>(&self) {
        self.counter.count_registered::<O>();
    }

    /// Count one item of `O` settled in `outcome`.
    pub fn count_outcome<O: Countable>(&self, outcome: O) {
        self.counter.count_outcome(outcome);
    }

    /// Count `later` happening to one settled item of `O`.
    pub fn count_later<O: Countable>(&self, later: O::Later) {
        self.counter.count_later::<O>(later);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recorder_counts_into_the_collector_it_came_from() {
        let collector = StatisticsCollector::new();
        assert!(collector.to_recorder().counts_into(&collector));
    }

    #[test]
    fn a_recorder_counts_into_no_other_collector() {
        let recorder = StatisticsCollector::new().to_recorder();
        assert!(!recorder.counts_into(&StatisticsCollector::new()));
    }
}
