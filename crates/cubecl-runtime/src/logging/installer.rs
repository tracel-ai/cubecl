use alloc::boxed::Box;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};
use cubecl_common::profile::{ProfileTicks, TimingMethod};
use cubecl_environment::sync::RwLock;

use super::{
    Execution, ExecutionId, ExecutionKind, ExecutionObserver, ExecutionScope, Measurement,
    ProfileDuration, ReportedExecution, TimingRequest,
};

/// Installs the process's one [`ExecutionObserver`], reports executions to it
/// and delivers their measurements.
///
/// # Assumptions
///
/// One observer is installed at a time: it is
/// [uninstalled](Self::uninstall) before another is
/// [installed](Self::install). An observer called here neither launches,
/// issues an execution, nor installs or uninstalls one: it is called with the
/// slot's read lock held.
///
/// # Guarantees
///
/// An observer is called with the slot's read lock held, so uninstalling it
/// waits for every call already under way, and it is not called once it is
/// uninstalled. A measurement reaches the observer its execution was
/// reported to, and only while it is still installed.
pub struct ObserverInstaller {
    /// Whether an observer is installed, read without the lock so the
    /// unobserved path, every ordinary run, is one relaxed load.
    observing: AtomicBool,
    observer: RwLock<Option<Arc<dyn ExecutionObserver>>>,
}

/// The process's observer.
pub static OBSERVERS: ObserverInstaller = ObserverInstaller::new();

impl ObserverInstaller {
    /// An installer with no observer installed.
    pub const fn new() -> Self {
        Self {
            observing: AtomicBool::new(false),
            observer: RwLock::new(None),
        }
    }

    /// Installs `observer`.
    pub fn install(&self, observer: Arc<dyn ExecutionObserver>) {
        *self.observer.write() = Some(observer);
        // Last, so the flag is never set over an empty slot.
        self.observing.store(true, Ordering::Relaxed);
    }

    /// Uninstalls the installed observer.
    pub fn uninstall(&self) {
        *self.observer.write() = None;
        self.observing.store(false, Ordering::Relaxed);
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
        observer.observe(&execution);
        Some(ReportedExecution::new(
            execution,
            Arc::clone(observer),
            observer.timing(),
        ))
    }

    /// What the installed observer asks be done with each launch's
    /// measurement now, [`None`](TimingRequest::None) when nothing observes.
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
            .observe_measurement(reported.execution(), measurement);
    }

    /// Delivers a launch's measurement the way `reported`'s observer asked
    /// for it: [`Deferred`](TimingRequest::Deferred) unread,
    /// [`Resolved`](TimingRequest::Resolved) read first, with the slot's lock
    /// released, since holding it across a read would stall every other
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
            TimingRequest::Resolved => drop(self.read_and_deliver(reported, profile)),
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
    ///
    /// # Guarantees
    ///
    /// The measurement it hands back carries the reading the observer was
    /// told, and how it was taken.
    pub fn read_for_logger(
        &self,
        reported: &ReportedExecution,
        profile: ProfileDuration,
    ) -> ProfileDuration {
        if let TimingRequest::Deferred(_) = reported.timing() {
            warn_logger_takes_deferred_measurements();
        }
        let method = profile.timing_method();
        let ticks = self.read_and_deliver(reported, profile);
        ProfileDuration::new(Box::pin(async move { ticks }), method)
    }
}

impl ObserverInstaller {
    /// Reads `profile` with the slot's lock released, delivers the reading,
    /// and hands it back.
    fn read_and_deliver(
        &self,
        reported: &ReportedExecution,
        profile: ProfileDuration,
    ) -> Option<ProfileTicks> {
        let method = profile.timing_method();
        let ticks = cubecl_environment::future::block_on(profile.resolve());
        match &ticks {
            Some(ticks) => self.deliver(reported, read(ticks, method)),
            None => warn_unmeasured(reported.execution()),
        }
        ticks
    }
}

/// The reading `ticks` hold, taken by `method`.
fn read(ticks: &ProfileTicks, method: TimingMethod) -> Measurement {
    Measurement::Read {
        duration: ticks.duration(),
        method,
    }
}

/// [`ObserverInstaller::issue`] on the process's observer.
pub fn issue(kind: impl FnOnce() -> ExecutionKind) -> Option<ReportedExecution> {
    OBSERVERS.issue(kind)
}

/// [`ObserverInstaller::timing`] on the process's observer.
pub fn timing_requested() -> TimingRequest {
    OBSERVERS.timing()
}

/// Says once per process that the profiling logger is taking the measurements
/// an observer asked to keep: it never receives [`Measurement::Unread`] and
/// its kernels stop overlapping, which is a pass other than the one it asked
/// to measure.
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
mod small_tests {
    use super::*;
    use crate::logging::{ExecutionPlace, StreamRole, TimedExecutions};
    use alloc::vec::Vec;
    use cubecl_common::device::{DeviceId, ServiceId};
    use cubecl_environment::stream::StreamId;
    use cubecl_environment::sync::Mutex;

    /// Answers `timing`, and keeps how many measurements it was told.
    struct Counting(TimingRequest, Mutex<usize>);

    impl Counting {
        fn answering(timing: TimingRequest) -> Arc<Self> {
            Arc::new(Self(timing, Mutex::new(0)))
        }
    }

    impl ExecutionObserver for Counting {
        fn observe(&self, _execution: &Execution) {}
        fn timing(&self) -> TimingRequest {
            self.0
        }
        fn observe_measurement(&self, _execution: &Execution, _measurement: Measurement) {
            *self.1.lock() += 1;
        }
    }

    fn kernel() -> ExecutionKind {
        ExecutionKind::Kernel {
            place: ExecutionPlace {
                service: ServiceId::of::<()>(DeviceId::new(0, 0)),
                stream: StreamRole::Compute(StreamId { value: 0 }),
            },
            name: "a_kernel",
        }
    }

    /// A measurement whose window carried none, as one a graph capture
    /// refused.
    fn unmeasured() -> ProfileDuration {
        ProfileDuration::new(Box::pin(async { None }), TimingMethod::Device)
    }

    fn installed(observer: Arc<dyn ExecutionObserver>) -> ObserverInstaller {
        let installer = ObserverInstaller::new();
        installer.install(observer);
        installer
    }

    #[test]
    fn nothing_is_reported_while_nothing_is_installed() {
        assert!(ObserverInstaller::new().issue(kernel).is_none());
    }

    #[test]
    fn an_installed_observer_is_reported_to() {
        let installer = installed(Counting::answering(TimingRequest::None));
        assert!(installer.issue(kernel).is_some());
    }

    #[test]
    fn nothing_is_reported_once_uninstalled() {
        let installer = installed(Counting::answering(TimingRequest::None));
        installer.uninstall();
        assert!(installer.issue(kernel).is_none());
    }

    #[test]
    fn the_answer_is_the_installed_observers() {
        let installer = installed(Counting::answering(TimingRequest::Resolved));
        assert_eq!(installer.timing(), TimingRequest::Resolved);
    }

    #[test]
    fn a_reported_execution_keeps_its_observers_answer() {
        let installer = installed(Counting::answering(TimingRequest::Deferred(
            TimedExecutions::Kernels,
        )));
        let reported = installer
            .issue(kernel)
            .expect("an observer should be installed");
        assert_eq!(
            reported.timing(),
            TimingRequest::Deferred(TimedExecutions::Kernels)
        );
    }

    #[test]
    fn a_measurement_reaches_no_observer_but_its_own() {
        let installer = installed(Counting::answering(TimingRequest::Resolved));
        let reported = installer
            .issue(kernel)
            .expect("an observer should be installed");
        let other = Counting::answering(TimingRequest::Resolved);
        installer.uninstall();
        installer.install(other.clone());

        installer.deliver(&reported, Measurement::Unread(unmeasured()));

        assert_eq!(*other.1.lock(), 0);
    }

    #[test]
    fn an_unmeasured_launch_is_not_delivered() {
        let observer = Counting::answering(TimingRequest::Resolved);
        let installer = installed(observer.clone());
        let reported = installer
            .issue(kernel)
            .expect("an observer should be installed");

        installer.deliver_profile(&reported, unmeasured());

        assert_eq!(*observer.1.lock(), 0);
    }

    #[test]
    fn an_unmeasured_launch_hands_the_logger_nothing() {
        let installer = installed(Counting::answering(TimingRequest::Resolved));
        let reported = installer
            .issue(kernel)
            .expect("an observer should be installed");

        let for_the_logger = installer.read_for_logger(&reported, unmeasured());

        assert!(cubecl_environment::future::block_on(for_the_logger.resolve()).is_none());
    }

    #[test]
    fn executions_are_reported_in_issue_order() {
        let installer = installed(Counting::answering(TimingRequest::None));
        let ids: Vec<_> = (0..2)
            .filter_map(|_| installer.issue(kernel))
            .map(|reported| reported.id())
            .collect();
        assert!(ids[0] < ids[1]);
    }
}

#[cfg(test)]
mod tests;
