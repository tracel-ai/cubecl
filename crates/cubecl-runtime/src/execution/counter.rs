use alloc::vec::Vec;
use cubecl_environment::sync::{Arc, AtomicUsize, Mutex, Ordering};

use super::{
    AutotuneStatistics, CompilationStatistics, Countable, CountedWork, ExecutionStatistics,
    KernelHandling, KernelOutcome, TuneHandling, TuneOutcome, WorkCompilationFailure,
};
use crate::id::KernelId;
use crate::server::LaunchError;

/// Counts what the work under a collector's overrides triggered: shared by the
/// collector, its readers and its recorders.
#[derive(Debug, Default)]
pub struct StatisticsCounter {
    kernels: Tally,
    tunes: Tally,
    /// The kernels the work needed and could not obtain, in the order they
    /// first failed: read whole, and searched for a kernel before one is
    /// added. Copied on write, so a read shares it.
    work_failures: Mutex<Arc<Vec<WorkCompilationFailure>>>,
}

/// What a [`StatisticsCounter`] counted, at one moment.
#[derive(Debug, Clone)]
pub struct CounterReading {
    /// The counts.
    pub statistics: ExecutionStatistics,
    /// The kernels the work needed and could not obtain.
    pub work_failures: Arc<Vec<WorkCompilationFailure>>,
}

impl StatisticsCounter {
    /// What it counted so far.
    pub fn counted(&self) -> CounterReading {
        CounterReading {
            statistics: ExecutionStatistics {
                compilation: compilation_statistics(&TallyRead::load(&self.kernels)),
                autotune: autotune_statistics(&TallyRead::load(&self.tunes)),
            },
            work_failures: self.work_failures.lock().clone(),
        }
    }

    /// Count one item of `O` registered.
    pub fn count_registered<O: Countable>(&self) {
        self.tally(O::WORK).register();
    }

    /// Count one item of `O` settled in `outcome`.
    pub fn count_outcome<O: Countable>(&self, outcome: O) {
        self.tally(O::WORK).settle_outcome(outcome.index());
    }

    /// Count `later` happening to one settled item of `O`.
    pub fn count_later<O: Countable>(&self, later: O::Later) {
        self.tally(O::WORK).record_later(O::later_index(later));
    }

    /// List `kernel` as one the work could not obtain, unless it is listed.
    pub fn count_work_failure(&self, kernel: &KernelId, error: &LaunchError) {
        let mut failures = self.work_failures.lock();
        if failures.iter().any(|failure| &failure.kernel == kernel) {
            return;
        }
        Arc::make_mut(&mut failures).push(WorkCompilationFailure {
            kernel: kernel.clone(),
            error: error.clone(),
        });
    }
}

impl StatisticsCounter {
    /// The tally that counts `kind`.
    fn tally(&self, kind: CountedWork) -> &Tally {
        match kind {
            CountedWork::Kernel => &self.kernels,
            CountedWork::Tune => &self.tunes,
        }
    }
}

/// The kernel counts `read` holds.
fn compilation_statistics(read: &TallyRead) -> CompilationStatistics {
    CompilationStatistics {
        registered_count: read.registered_count,
        compiled_count: read.settled_counts[KernelOutcome::Compiled.index()],
        loaded_count: read.settled_counts[KernelOutcome::Loaded.index()],
        failed_count: read.settled_counts[KernelOutcome::Failed.index()],
        abandoned_count: read.settled_counts[KernelOutcome::Abandoned.index()],
        refused_count: read.later_counts[KernelOutcome::later_index(KernelHandling::Refused)],
        stored_count: read.later_counts[KernelOutcome::later_index(KernelHandling::Stored)],
    }
}

/// The key counts `read` holds.
fn autotune_statistics(read: &TallyRead) -> AutotuneStatistics {
    AutotuneStatistics {
        registered_count: read.registered_count,
        measured_count: read.settled_counts[TuneOutcome::Measured.index()],
        loaded_count: read.settled_counts[TuneOutcome::Loaded.index()],
        failed_count: read.settled_counts[TuneOutcome::Failed.index()],
        abandoned_count: read.settled_counts[TuneOutcome::Abandoned.index()],
        persisted_count: read.later_counts[TuneOutcome::later_index(TuneHandling::Persisted)],
    }
}

/// Room for every outcome a kind of work has.
const MAX_OUTCOME_COUNT: usize = 4;
/// Room for everything that can happen to a kind of work once it settled.
const MAX_LATER_COUNT: usize = 2;

// Every kind of work fits its tally: a variant added past the room fails
// here rather than out of bounds on a count.
const _: () = assert!(
    KernelOutcome::OUTCOME_COUNT <= MAX_OUTCOME_COUNT
        && KernelOutcome::LATER_COUNT <= MAX_LATER_COUNT
);
const _: () = assert!(
    TuneOutcome::OUTCOME_COUNT <= MAX_OUTCOME_COUNT && TuneOutcome::LATER_COUNT <= MAX_LATER_COUNT
);

/// Items registered, how many ended in each outcome, and how many of those
/// something happened to afterwards: the one counting mechanism every kind
/// of work shares.
///
/// An item is registered before its outcome, and its outcome before
/// anything after it; a [read](TallyRead::load) loads them the other way
/// round, so it never sees more settled than registered, nor more
/// afterwards than settled.
#[derive(Debug, Default)]
struct Tally {
    registered_count: AtomicUsize,
    /// How many items ended in each outcome, by [`Countable::index`].
    settled_counts: [AtomicUsize; MAX_OUTCOME_COUNT],
    /// How many settled items each afterwards happened to, by
    /// [`Countable::later_index`].
    later_counts: [AtomicUsize; MAX_LATER_COUNT],
}

impl Tally {
    fn register(&self) {
        self.registered_count.fetch_add(1, Ordering::Release);
    }

    fn settle_outcome(&self, index: usize) {
        self.settled_counts[index].fetch_add(1, Ordering::Release);
    }

    fn record_later(&self, index: usize) {
        self.later_counts[index].fetch_add(1, Ordering::Release);
    }
}

/// A [`Tally`] as read at one moment.
struct TallyRead {
    pub registered_count: usize,
    /// How many items ended in each outcome, by [`Countable::index`].
    pub settled_counts: [usize; MAX_OUTCOME_COUNT],
    /// How many settled items each afterwards happened to, by
    /// [`Countable::later_index`].
    pub later_counts: [usize; MAX_LATER_COUNT],
}

impl TallyRead {
    /// What `tally` counts now.
    fn load(tally: &Tally) -> Self {
        let later_counts = loaded(&tally.later_counts);
        let settled_counts = loaded(&tally.settled_counts);
        Self {
            registered_count: tally.registered_count.load(Ordering::Acquire),
            settled_counts,
            later_counts,
        }
    }
}

/// Each of `counts` as it stands now.
fn loaded<const N: usize>(counts: &[AtomicUsize; N]) -> [usize; N] {
    core::array::from_fn(|index| counts[index].load(Ordering::Acquire))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> KernelId {
        KernelId::new::<()>()
    }

    fn out_of_memory() -> LaunchError {
        LaunchError::OutOfMemory {
            reason: alloc::string::String::new(),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        }
    }

    #[test]
    fn a_registered_kernel_counts_as_registered() {
        let counter = StatisticsCounter::default();
        counter.count_registered::<KernelOutcome>();
        assert_eq!(counter.counted().statistics.compilation.registered_count, 1);
    }

    #[test]
    fn a_settled_kernel_counts_in_its_outcome() {
        let counter = StatisticsCounter::default();
        counter.count_outcome(KernelOutcome::Loaded);
        assert_eq!(counter.counted().statistics.compilation.loaded_count, 1);
    }

    #[test]
    fn a_tune_counts_apart_from_the_kernels() {
        let counter = StatisticsCounter::default();
        counter.count_outcome(TuneOutcome::Loaded);
        assert_eq!(counter.counted().statistics.compilation.loaded_count, 0);
    }

    #[test]
    fn a_kept_tune_pick_counts_as_persisted() {
        let counter = StatisticsCounter::default();
        counter.count_later::<TuneOutcome>(TuneHandling::Persisted);
        assert_eq!(counter.counted().statistics.autotune.persisted_count, 1);
    }

    #[test]
    fn a_work_failure_met_twice_is_listed_once() {
        let counter = StatisticsCounter::default();
        counter.count_work_failure(&kernel(), &out_of_memory());
        counter.count_work_failure(&kernel(), &out_of_memory());
        assert_eq!(counter.counted().work_failures.len(), 1);
    }
}
