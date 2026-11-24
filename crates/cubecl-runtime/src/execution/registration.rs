use core::marker::PhantomData;

use super::{Countable, KernelOutcome, Settled, StatisticsRecorder, TuneOutcome};

/// One kernel registered with the collector of the override open when it
/// was.
#[doc(hidden)]
pub type KernelRegistration = Registration<KernelOutcome>;
/// One autotune key registered with the collector of the override open when
/// it was.
pub type TuneRegistration = Registration<TuneOutcome>;

/// One item of work registered with the collector of the override open when
/// it was, which it is counted to once it [settles](Self::settle) —
/// wherever and whenever that is, whether or not that override is still
/// open. Dropped before it settled — its batch or its tune panicked, or the
/// key it holds was dropped — it settles as abandoned.
///
/// What holds it decides when that is: a kernel queued and never loaded, or
/// a key gathered and never measured again, stays registered until a later
/// pass reaches it, its environment switches or its server goes away.
///
/// Public for the kernel loader, which lives in `cubecl-server`; counting is
/// not for callers, which read their own
/// [`StatisticsCollector`](super::StatisticsCollector).
#[doc(hidden)]
#[must_use]
#[derive(Debug)]
pub struct Registration<O: Countable> {
    recorder: StatisticsRecorder,
    /// Set as it settles, so a drop knows whether it did.
    settled: bool,
    outcome: PhantomData<O>,
}

impl<O: Countable> Registration<O> {
    /// Register one item with the collector the work triggered now counts
    /// to ([`StatisticsRecorder::current`]), if there is one.
    pub fn register() -> Option<Self> {
        StatisticsRecorder::current().map(Self::with_recorder)
    }

    /// Count how it ended.
    pub fn settle(mut self, outcome: O) -> Settled<O> {
        self.settled = true;
        Settled::new(self.recorder.clone(), outcome)
    }
}

impl<O: Countable> Registration<O> {
    /// One item registered with `recorder`.
    fn with_recorder(recorder: StatisticsRecorder) -> Self {
        recorder.count_registered::<O>();
        Self {
            recorder,
            settled: false,
            outcome: PhantomData,
        }
    }
}

impl<O: Countable> Drop for Registration<O> {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        self.recorder.count_outcome(O::ABANDONED);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::StatisticsCollector;

    #[test]
    fn a_registration_counts_as_registered() {
        let collector = StatisticsCollector::new();
        let registered = KernelRegistration::with_recorder(collector.to_recorder());
        assert_eq!(collector.statistics().compilation.registered_count, 1);
        core::mem::drop(registered);
    }

    #[test]
    fn a_settled_registration_counts_its_outcome() {
        let collector = StatisticsCollector::new();
        let registered = KernelRegistration::with_recorder(collector.to_recorder());
        let _settled = registered.settle(KernelOutcome::Loaded);
        assert_eq!(collector.statistics().compilation.loaded_count, 1);
    }

    #[test]
    fn a_settled_registration_is_not_abandoned_as_it_drops() {
        let collector = StatisticsCollector::new();
        let registered = KernelRegistration::with_recorder(collector.to_recorder());
        let _settled = registered.settle(KernelOutcome::Loaded);
        assert_eq!(collector.statistics().compilation.abandoned_count, 0);
    }

    /// Its batch or its tune panicked: it settles as abandoned.
    #[test]
    fn a_registration_dropped_before_it_settled_is_abandoned() {
        let collector = StatisticsCollector::new();
        drop(TuneRegistration::with_recorder(collector.to_recorder()));
        assert_eq!(collector.statistics().autotune.abandoned_count, 1);
    }
}
