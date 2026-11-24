use alloc::vec::Vec;
use cubecl_environment::sync::Arc;

use super::{
    ExecutionStatistics, StatisticsCounter, StatisticsReader, StatisticsRecorder,
    WorkCompilationFailure,
};

/// Collects what the [overrides](super::ProcessModeOverride) opened with it
/// triggered.
///
/// Not `Clone`: whoever holds it opens overrides that count into it. A
/// reader on another thread holds a [`StatisticsReader`].
#[derive(Debug, Default)]
pub struct StatisticsCollector {
    counter: Arc<StatisticsCounter>,
}

impl StatisticsCollector {
    /// A collector nothing has counted into yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// What reads it from another thread.
    pub fn to_reader(&self) -> StatisticsReader {
        StatisticsReader::new(self.counter.clone())
    }

    /// What its overrides triggered so far.
    pub fn statistics(&self) -> ExecutionStatistics {
        self.counter.counted().statistics
    }

    /// The kernels the work under its overrides needed and could not
    /// obtain so far, each once, in the order they first failed. A pass that
    /// discards its launches reports a kernel that can never run only here.
    pub fn work_failures(&self) -> Arc<Vec<WorkCompilationFailure>> {
        self.counter.counted().work_failures
    }
}

impl StatisticsCollector {
    /// Where an override opened with it counts.
    pub(crate) fn to_recorder(&self) -> StatisticsRecorder {
        StatisticsRecorder::new(self.counter.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::KernelOutcome;

    #[test]
    fn a_collector_reads_what_its_recorder_counted() {
        let collector = StatisticsCollector::new();
        collector.to_recorder().count_outcome(KernelOutcome::Failed);
        assert_eq!(collector.statistics().compilation.failed_count, 1);
    }

    #[test]
    fn a_reader_reads_what_its_collector_counted() {
        let collector = StatisticsCollector::new();
        collector.to_recorder().count_outcome(KernelOutcome::Failed);
        assert_eq!(collector.to_reader().statistics(), collector.statistics());
    }
}
