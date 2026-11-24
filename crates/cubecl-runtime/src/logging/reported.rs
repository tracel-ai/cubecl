use alloc::sync::Arc;

use super::{
    Execution, ExecutionId, ExecutionObserver, OBSERVERS, ProfileDuration, Timed, TimingRequest,
};

/// An execution as it was reported: what it was, the observer it was
/// reported to, and what that observer asked be done with its measurement.
///
/// # Guarantees
///
/// The observer is asked how to time the execution once, when it is issued,
/// and that answer decides how its measurement is delivered, whatever the
/// observer answers afterwards.
#[derive(Clone)]
pub struct ReportedExecution {
    execution: Execution,
    observer: Arc<dyn ExecutionObserver>,
    timing: TimingRequest,
}

impl ReportedExecution {
    /// An execution reported to `observer`, which answered `timing`.
    pub fn new(
        execution: Execution,
        observer: Arc<dyn ExecutionObserver>,
        timing: TimingRequest,
    ) -> Self {
        Self {
            execution,
            observer,
            timing,
        }
    }

    /// What was reported.
    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    /// The id it was reported under.
    pub fn id(&self) -> ExecutionId {
        self.execution.id
    }

    /// What its observer asked be done with its measurement.
    pub fn timing(&self) -> TimingRequest {
        self.timing
    }

    /// Whether its observer asked for launches to be timed.
    pub fn wants_timing(&self) -> bool {
        self.timing != TimingRequest::None
    }

    /// Whether its observer times communications, and so is told of their
    /// waits.
    pub fn times_communications(&self) -> bool {
        self.timing == TimingRequest::Deferred(Timed::KernelsAndCommunications)
    }

    /// Whether it was reported to `observer`.
    pub fn is_reported_to(&self, observer: &Arc<dyn ExecutionObserver>) -> bool {
        Arc::ptr_eq(&self.observer, observer)
    }

    /// The observer it was reported to.
    pub fn observer(&self) -> &dyn ExecutionObserver {
        &*self.observer
    }

    /// Hands a communication part's or wait's measurement over unread. It is
    /// never read here: a part ends only once its peers' run, so a thread
    /// that queues several would wait forever on the first.
    ///
    /// # Assumptions
    ///
    /// Its observer [times communications](Self::times_communications).
    pub fn deliver_unread(&self, measurement: ProfileDuration) {
        OBSERVERS.deliver(self, super::Measurement::Unread(measurement));
    }

    /// Delivers a launch's measurement the way its observer asked for it:
    /// [`ObserverInstaller::deliver_profile`](super::installer::ObserverInstaller::deliver_profile).
    ///
    /// # Assumptions
    ///
    /// Its observer [wants timing](Self::wants_timing).
    pub fn deliver_profile(&self, profile: ProfileDuration) {
        OBSERVERS.deliver_profile(self, profile);
    }

    /// Reads a launch's measurement for its observer and hands the reading
    /// back for the profiling logger:
    /// [`ObserverInstaller::read_for_logger`](super::installer::ObserverInstaller::read_for_logger).
    pub fn read_for_logger(&self, profile: ProfileDuration) -> ProfileDuration {
        OBSERVERS.read_for_logger(self, profile)
    }
}
