use core::marker::PhantomData;

use super::{Countable, KernelOutcome, StatisticsRecorder};
use crate::id::KernelId;
use crate::server::LaunchError;

/// A registered kernel once its outcome is counted.
#[doc(hidden)]
pub type SettledKernel = Settled<KernelOutcome>;

/// A [`Registration`](super::Registration) whose outcome is counted: what
/// alone can count what happens to the work afterwards.
///
/// # Assumptions
///
/// Each afterwards is recorded at most once for one settled item.
#[doc(hidden)]
#[derive(Debug)]
pub struct Settled<O: Countable> {
    recorder: StatisticsRecorder,
    outcome: PhantomData<O>,
}

impl<O: Countable> Settled<O> {
    /// Count `later` happening to the item.
    pub fn record(&self, later: O::Later) {
        self.recorder.count_later::<O>(later);
    }
}

impl<O: Countable> Settled<O> {
    /// An item `recorder` counted as ended in `outcome`.
    pub(crate) fn new(recorder: StatisticsRecorder, outcome: O) -> Self {
        recorder.count_outcome(outcome);
        Self {
            recorder,
            outcome: PhantomData,
        }
    }
}

impl SettledKernel {
    /// List the kernel as one the work needed and could not obtain, to the
    /// collector it was registered with: it failed to compile, or the device
    /// refused it. Only for a kernel a launch of the work asked for; an
    /// autotune candidate's failure is its race's alone.
    pub fn fail_work(&self, kernel: &KernelId, error: &LaunchError) {
        self.recorder.fail_work(kernel, error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{KernelHandling, StatisticsCollector};

    #[test]
    fn a_settled_item_counts_its_outcome() {
        let collector = StatisticsCollector::new();
        let _settled = Settled::new(collector.to_recorder(), KernelOutcome::Compiled);
        assert_eq!(collector.statistics().compilation.compiled_count, 1);
    }

    #[test]
    fn what_happens_to_a_settled_item_counts() {
        let collector = StatisticsCollector::new();
        let settled = Settled::new(collector.to_recorder(), KernelOutcome::Compiled);
        settled.record(KernelHandling::Refused);
        assert_eq!(collector.statistics().compilation.refused_count, 1);
    }
}
