//! What a dry run observes of the work it provokes: the kernels it compiles
//! and the tunes it measures, counted to it by the code doing the work.

use super::DryRunId;
use cubecl_environment::sync::{Arc, AtomicUsize, Ordering};

/// What one [`DryRun`](super::DryRun) has provoked so far, as
/// [`DryRun::observe`](super::DryRun::observe) reads it.
///
/// A pass under a dry run spends its time where the launch that started it
/// does not return — a batch of kernels compiling on every core, a tune
/// measuring its candidates — and either can run for minutes. These are what
/// a caller reads, from any thread, to show that work going. Both counts only
/// grow over the dry run's passes, on every device it reaches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DryRunObservation {
    /// The kernels set out to be obtained under the dry run — queued, or
    /// asked for by a launch that found them neither loaded nor queued — and
    /// those a compilation settled, from the compilation store or the
    /// compiler. A kernel counts to the dry run that queued it, whenever and
    /// wherever its batch compiles, and a batch settles each kernel as it
    /// finishes, so the count moves while the batch runs.
    pub kernels: Progress,
    /// The autotune keys set out to be tuned — gathered by a
    /// [`Compile`](super::DryRunScope::Compile) pass, or reached ungathered
    /// by a tune that measures — and those whose measured tune committed a
    /// pick. A key with one candidate is answered, not tuned, and counts
    /// nowhere.
    pub tunes: Progress,
}

/// A count of work: what was set out to be done, and how much of it is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Progress {
    /// The items set out to be done.
    pub requested: usize,
    /// Those done, failed ones included: never more than
    /// [`requested`](Self::requested).
    pub settled: usize,
}

impl Progress {
    /// The items requested and not settled yet.
    pub fn pending(self) -> usize {
        self.requested.saturating_sub(self.settled)
    }
}

/// Where the code doing a dry run's work counts it: the kernel loader and
/// autotune, which reach it through [`counted`](super::counted). Cloning it
/// keeps counting to the same dry run, which is how a kernel queued under
/// one is settled to it when its batch compiles.
#[derive(Debug, Clone)]
pub struct DryRunCounter {
    observed: Arc<Observed>,
}

impl DryRunCounter {
    /// The dry run it counts to.
    pub fn id(&self) -> DryRunId {
        self.observed.id
    }

    /// Where kernels are counted.
    pub fn kernels(&self) -> &Counter {
        &self.observed.kernels
    }

    /// Where tunes are counted.
    pub fn tunes(&self) -> &Counter {
        &self.observed.tunes
    }
}

/// Reads what one dry run has provoked, from any thread, and nothing else:
/// what a caller showing a dry run's progress holds, where only the dry
/// run's owner opens its passes.
#[derive(Debug, Clone)]
pub struct DryRunObserver {
    observed: Arc<Observed>,
}

impl DryRunObserver {
    /// The dry run it reads.
    pub fn id(&self) -> DryRunId {
        self.observed.id
    }

    /// What the dry run has provoked so far, over every pass and device.
    pub fn observe(&self) -> DryRunObservation {
        self.observed.observe()
    }
}

/// One dry run's counts, shared by the dry run, its observers and every
/// counter handed out for it.
#[derive(Debug)]
pub(super) struct Observed {
    id: DryRunId,
    kernels: Counter,
    tunes: Counter,
}

impl Observed {
    pub(super) fn new(id: DryRunId) -> Arc<Self> {
        Arc::new(Self {
            id,
            kernels: Counter::new(),
            tunes: Counter::new(),
        })
    }

    pub(super) fn id(&self) -> DryRunId {
        self.id
    }

    pub(super) fn counter(self: &Arc<Self>) -> DryRunCounter {
        DryRunCounter {
            observed: self.clone(),
        }
    }

    pub(super) fn observer(self: &Arc<Self>) -> DryRunObserver {
        DryRunObserver {
            observed: self.clone(),
        }
    }

    pub(super) fn observe(&self) -> DryRunObservation {
        DryRunObservation {
            kernels: self.kernels.read(),
            tunes: self.tunes.read(),
        }
    }
}

/// One count of a dry run's work.
///
/// `requested` is always raised before `settled` for the same items, and a
/// read loads `settled` first, so a read never sees more settled than
/// requested.
#[derive(Debug)]
pub struct Counter {
    requested: AtomicUsize,
    settled: AtomicUsize,
}

impl Counter {
    const fn new() -> Self {
        Self {
            requested: AtomicUsize::new(0),
            settled: AtomicUsize::new(0),
        }
    }

    /// Set out to do `items` more.
    pub fn request(&self, items: usize) {
        self.requested.fetch_add(items, Ordering::Release);
    }

    /// `items` more are done.
    pub fn settle(&self, items: usize) {
        self.settled.fetch_add(items, Ordering::Release);
    }

    fn read(&self) -> Progress {
        let settled = self.settled.load(Ordering::Acquire);
        let requested = self.requested.load(Ordering::Acquire);
        Progress { requested, settled }
    }
}
