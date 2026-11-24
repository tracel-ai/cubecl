use alloc::sync::Arc;

use super::{
    Execution, ExecutionId, ExecutionObserver, Measurement, OBSERVERS, ProfileDuration,
    TimingRequest,
};

/// An execution as it was reported: what it was, the observer it was
/// reported to, and what that observer asked be done with its measurement.
///
/// # Guarantees
///
/// The answer its observer gave when it was issued decides how its
/// measurement is delivered, whatever the observer answers afterwards.
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
        self.timing.times_launches()
    }

    /// Whether its observer times communications, and so is told of their
    /// waits.
    pub fn times_communications(&self) -> bool {
        self.timing.times_communications()
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
        OBSERVERS.deliver(self, Measurement::Unread(measurement));
    }

    /// [`ObserverInstaller::deliver_profile`](super::installer::ObserverInstaller::deliver_profile)
    /// on the process's observer.
    ///
    /// # Assumptions
    ///
    /// Its observer [wants timing](Self::wants_timing).
    pub fn deliver_profile(&self, profile: ProfileDuration) {
        OBSERVERS.deliver_profile(self, profile);
    }

    /// [`ObserverInstaller::read_for_logger`](super::installer::ObserverInstaller::read_for_logger)
    /// on the process's observer.
    pub fn read_for_logger(&self, profile: ProfileDuration) -> ProfileDuration {
        OBSERVERS.read_for_logger(self, profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::{ExecutionKind, ExecutionPlace, StreamRole, TimedExecutions};
    use cubecl_common::device::{DeviceId, ServiceId};
    use cubecl_environment::stream::StreamId;

    struct Silent;

    impl ExecutionObserver for Silent {
        fn observe(&self, _execution: &Execution) {}
    }

    fn reported(timing: TimingRequest) -> ReportedExecution {
        let execution = Execution {
            id: ExecutionId::mint(),
            parent: None,
            kind: ExecutionKind::Kernel {
                place: ExecutionPlace {
                    service: ServiceId::of::<()>(DeviceId::new(0, 0)),
                    stream: StreamRole::Compute(StreamId { value: 0 }),
                },
                name: "a_kernel",
            },
        };
        ReportedExecution::new(execution, Arc::new(Silent), timing)
    }

    #[test]
    fn an_untimed_execution_wants_no_timing() {
        assert!(!reported(TimingRequest::None).wants_timing());
    }

    #[test]
    fn a_resolved_execution_wants_timing() {
        assert!(reported(TimingRequest::Resolved).wants_timing());
    }

    #[test]
    fn an_execution_timing_kernels_only_times_no_communication() {
        let reported = reported(TimingRequest::Deferred(TimedExecutions::Kernels));
        assert!(!reported.times_communications());
    }

    #[test]
    fn an_execution_timing_communications_times_them() {
        let reported = reported(TimingRequest::Deferred(
            TimedExecutions::KernelsAndCommunications,
        ));
        assert!(reported.times_communications());
    }

    #[test]
    fn an_execution_is_reported_to_its_own_observer() {
        let observer: Arc<dyn ExecutionObserver> = Arc::new(Silent);
        let reported = ReportedExecution::new(
            reported(TimingRequest::None).execution().clone(),
            observer.clone(),
            TimingRequest::None,
        );
        assert!(reported.is_reported_to(&observer));
    }

    #[test]
    fn an_execution_is_not_reported_to_another_observer() {
        let other: Arc<dyn ExecutionObserver> = Arc::new(Silent);
        assert!(!reported(TimingRequest::None).is_reported_to(&other));
    }
}
