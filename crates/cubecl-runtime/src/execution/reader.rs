use alloc::vec::Vec;
use cubecl_environment::sync::Arc;

use super::{ExecutionStatistics, StatisticsCounter, WorkCompilationFailure};

/// Reads a [`StatisticsCollector`](super::StatisticsCollector)'s statistics
/// from any thread, and opens nothing.
#[derive(Debug, Clone)]
pub struct StatisticsReader {
    counter: Arc<StatisticsCounter>,
}

impl StatisticsReader {
    /// What the collector's overrides triggered so far.
    pub fn statistics(&self) -> ExecutionStatistics {
        self.counter.counted().statistics
    }

    /// The kernels the work under the collector's overrides needed and could
    /// not obtain so far:
    /// [`StatisticsCollector::work_failures`](super::StatisticsCollector::work_failures).
    pub fn work_failures(&self) -> Arc<Vec<WorkCompilationFailure>> {
        self.counter.counted().work_failures
    }
}

impl StatisticsReader {
    /// A reader of what `counter` counts.
    pub(crate) fn new(counter: Arc<StatisticsCounter>) -> Self {
        Self { counter }
    }
}
