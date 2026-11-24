use core::marker::PhantomData;

use super::{Countable, KernelOutcome, StatisticsRecorder};
use crate::id::KernelId;
use crate::server::LaunchError;

/// A registered kernel once its outcome is counted.
#[doc(hidden)]
pub type SettledKernel = Settled<KernelOutcome>;

/// A [`Registration`](super::Registration) whose outcome is counted: what
/// alone can count what happens to the work afterwards, each thing once.
#[doc(hidden)]
#[derive(Debug)]
pub struct Settled<O: Countable> {
    recorder: StatisticsRecorder,
    /// The afterwards already counted.
    recorded: RecordedLater,
    outcome: PhantomData<O>,
}

impl<O: Countable> Settled<O> {
    /// Count `later`, once however often it is told.
    pub fn record(&mut self, later: O::Later) {
        let Some(recorded) = self.recorded.adding_index(O::later_index(later)) else {
            return;
        };
        self.recorded = recorded;
        self.recorder.count_later::<O>(later);
    }
}

impl<O: Countable> Settled<O> {
    /// An item `recorder` counted as ended in `outcome`.
    pub(crate) fn new(recorder: StatisticsRecorder, outcome: O) -> Self {
        recorder.count_outcome(outcome);
        Self {
            recorder,
            recorded: RecordedLater::default(),
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

/// The afterwards of a [`Settled`] already counted, one bit each by
/// [`Countable::later_index`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RecordedLater {
    pub bits: u8,
}

impl RecordedLater {
    /// The afterwards counted once the one at `index` is too, or `None` when
    /// it already was.
    fn adding_index(self, index: usize) -> Option<Self> {
        let bit = 1 << index;
        (self.bits & bit == 0).then_some(Self {
            bits: self.bits | bit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{KernelHandling, StatisticsCollector};

    #[test]
    fn what_happens_to_settled_work_counts_once_however_often_it_is_told() {
        let collector = StatisticsCollector::new();
        let mut settled = Settled::new(collector.to_recorder(), KernelOutcome::Compiled);
        settled.record(KernelHandling::Refused);
        settled.record(KernelHandling::Refused);
        assert_eq!(collector.statistics().compilation.refused_count, 1);
    }

    #[test]
    fn an_afterwards_not_counted_yet_is_added() {
        let recorded = RecordedLater::default().adding_index(1);
        assert_eq!(recorded, Some(RecordedLater { bits: 0b10 }));
    }

    #[test]
    fn an_afterwards_counted_already_is_not_added_again() {
        let recorded = RecordedLater { bits: 0b10 }.adding_index(1);
        assert_eq!(recorded, None);
    }
}
