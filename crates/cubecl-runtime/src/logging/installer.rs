use alloc::boxed::Box;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};
use cubecl_environment::sync::RwLock;

use super::{
    Execution, ExecutionId, ExecutionKind, ExecutionObserver, ExecutionScope, Measurement,
    ProfileDuration, ReportedExecution, Timed, TimingRequest,
};

/// Installs the process's one [`ExecutionObserver`], reports executions to it
/// and delivers their measurements.
///
/// # Assumptions
///
/// Observations are restored in the reverse order they were installed.
///
/// # Guarantees
///
/// An observer is called with the slot's read lock held, so an observation
/// that drops waits for every call already under way, and is not called once
/// it has dropped. A measurement reaches the observer its execution was
/// reported to, and only while it is still installed.
pub struct ObserverInstaller {
    /// Whether an observer is installed, read without the lock so the
    /// unobserved path, every ordinary run, is one relaxed load.
    observing: AtomicBool,
    observer: RwLock<Option<Arc<dyn ExecutionObserver>>>,
}

/// The process's observer.
pub static OBSERVERS: ObserverInstaller = ObserverInstaller {
    observing: AtomicBool::new(false),
    observer: RwLock::new(None),
};

/// The request under which an observer times communications, and so is told
/// of their waits.
const TIMES_COMMUNICATIONS: TimingRequest =
    TimingRequest::Deferred(Timed::KernelsAndCommunications);

impl ObserverInstaller {
    /// Installs `observer`, handing back the one it replaced.
    pub fn install(
        &self,
        observer: Arc<dyn ExecutionObserver>,
    ) -> Option<Arc<dyn ExecutionObserver>> {
        let previous = self.observer.write().replace(observer);
        // Last, so the flag is never set over an empty slot.
        self.observing.store(true, Ordering::Relaxed);
        previous
    }

    /// Puts back `previous`, the observer an [`install`](Self::install)
    /// replaced.
    pub fn restore(&self, previous: Option<Arc<dyn ExecutionObserver>>) {
        let still_observed = previous.is_some();
        *self.observer.write() = previous;
        // Last, so the flag is never cleared while an observer is still
        // installed: an issue that reads it in between finds the slot and
        // reports to it.
        self.observing.store(still_observed, Ordering::Relaxed);
    }

    /// Reports the execution `kind` builds to the installed observer, parented
    /// by the [`ExecutionScope`] open on this thread, or builds nothing when
    /// none is installed.
    pub fn issue(&self, kind: impl FnOnce() -> ExecutionKind) -> Option<ReportedExecution> {
        if !self.observing.load(Ordering::Relaxed) {
            return None;
        }
        // Built before taking the lock, so the caller's closure never runs under it.
        let execution = Execution {
            id: ExecutionId::mint(),
            parent: ExecutionScope::current(),
            kind: kind(),
        };
        let slot = self.observer.read();
        let observer = slot.as_ref()?;
        observer.issued(&execution);
        Some(ReportedExecution::new(
            execution,
            Arc::clone(observer),
            observer.timing(),
        ))
    }

    /// What the installed observer asks be done with each launch's
    /// measurement, [`None`](TimingRequest::None) when nothing observes.
    pub fn timing(&self) -> TimingRequest {
        if !self.observing.load(Ordering::Relaxed) {
            return TimingRequest::None;
        }
        self.observer
            .read()
            .as_ref()
            .map_or(TimingRequest::None, |observer| observer.timing())
    }

    /// Gives `reported`'s observer `measurement`, while it is still installed.
    pub fn deliver(&self, reported: &ReportedExecution, measurement: Measurement) {
        let slot = self.observer.read();
        if !slot
            .as_ref()
            .is_some_and(|installed| reported.is_reported_to(installed))
        {
            return;
        }
        reported
            .observer()
            .measured(reported.execution(), measurement);
    }

    /// Delivers a launch's measurement the way `reported`'s observer asked
    /// for it.
    ///
    /// [`Deferred`](TimingRequest::Deferred) hands it over unread.
    /// [`Resolved`](TimingRequest::Resolved) reads it first, with the slot's
    /// lock released: holding it across a read would stall every other
    /// launching thread behind this one's kernel.
    ///
    /// # Assumptions
    ///
    /// `reported`'s observer asked for launches to be timed.
    ///
    /// # Panics
    ///
    /// When it asked for none, breaking the Assumption.
    pub fn deliver_profile(&self, reported: &ReportedExecution, profile: ProfileDuration) {
        match reported.timing() {
            TimingRequest::Deferred(_) => self.deliver(reported, Measurement::Unread(profile)),
            TimingRequest::Resolved => self.read_and_deliver(reported, profile),
            TimingRequest::None => unreachable!("an untimed launch should not be profiled"),
        }
    }

    /// Reads `profile` for `reported`'s observer, and hands the reading back
    /// for the profiling logger.
    ///
    /// Both want the same measurement and a measurement is read once, so the
    /// launch path reads it here and passes on what it got. An observer that
    /// asked to keep its measurements unread is told a duration instead, and
    /// its kernels run one at a time: that is said once per process, since it
    /// is a configuration mistake and not a per-launch event.
    pub fn read_for_logger(
        &self,
        reported: &ReportedExecution,
        profile: ProfileDuration,
    ) -> ProfileDuration {
        if let TimingRequest::Deferred(_) = reported.timing() {
            warn_logger_takes_deferred_measurements();
        }
        let method = profile.timing_method();
        let ticks = cubecl_environment::future::block_on(profile.resolve());
        match &ticks {
            Some(ticks) => self.deliver(
                reported,
                Measurement::Read {
                    duration: ticks.duration(),
                    method,
                },
            ),
            None => warn_unmeasured(reported.execution()),
        }
        ProfileDuration::new(Box::pin(async move { ticks }), method)
    }
}

impl ObserverInstaller {
    /// Reads `profile` with the slot's lock released, and delivers the reading.
    fn read_and_deliver(&self, reported: &ReportedExecution, profile: ProfileDuration) {
        let method = profile.timing_method();
        let Some(ticks) = cubecl_environment::future::block_on(profile.resolve()) else {
            warn_unmeasured(reported.execution());
            return;
        };
        self.deliver(
            reported,
            Measurement::Read {
                duration: ticks.duration(),
                method,
            },
        );
    }
}

/// Reports the execution `kind` builds to the installed observer:
/// [`ObserverInstaller::issue`].
pub fn issue(kind: impl FnOnce() -> ExecutionKind) -> Option<ReportedExecution> {
    OBSERVERS.issue(kind)
}

/// What the installed observer asks be done with each launch's measurement.
pub fn timing_requested() -> TimingRequest {
    OBSERVERS.timing()
}

/// Whether the installed observer times communications, and so is told of
/// their waits.
pub fn times_communications() -> bool {
    timing_requested() == TIMES_COMMUNICATIONS
}

/// Says once per process that the profiling logger is taking the measurements
/// an observer asked to keep: it never receives
/// [`Measurement::Unread`] and its kernels stop
/// overlapping, which is a pass other than the one it asked to measure.
fn warn_logger_takes_deferred_measurements() {
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    log::warn!(
        "The profiling logger is reading every launch's measurement, so this run's \
         execution observer is told durations instead of keeping them, and its kernels \
         run one at a time. Turn the profile logging off to measure the pass as it runs."
    );
}

/// Says a measurement was skipped: the window carried none, and a zero would
/// put a launch that was never timed in the timings.
fn warn_unmeasured(execution: &Execution) {
    log::warn!(
        "Skipped timing {:?} for its observer: the profiled window carried no measurement",
        execution.kind
    );
}

#[cfg(test)]
mod tests;
