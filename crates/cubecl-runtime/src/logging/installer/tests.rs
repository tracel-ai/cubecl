use super::*;
use crate::logging::{
    Communication, Duration, ExecutionObservation, ExecutionPlace, ProfileTicks, StreamRole,
    TimedExecutions, TimingMethod, TransferRoute,
};
use alloc::vec::Vec;
// `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
// bring in itself.
use alloc::vec;
use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::Mutex;

fn service(index: u16) -> ServiceId {
    ServiceId::of::<()>(DeviceId::new(0, index))
}

fn launch(name: &'static str) -> Option<ReportedExecution> {
    issue(|| ExecutionKind::Kernel {
        place: ExecutionPlace {
            service: service(0),
            stream: StreamRole::Compute(StreamId { value: 0 }),
        },
        name,
    })
}

fn transfer() -> Communication {
    Communication::Transfer {
        from: service(0),
        to: service(1),
        bytes: 4,
        route: TransferRoute::Host,
    }
}

/// An observer is called before the launch is submitted, so whatever the
/// caller was doing when it issued the kernel is still true.
#[test]
#[serial_test::serial]
fn executions_arrive_in_issue_order() {
    let recorder = Arc::new(Recorder::default());
    let watching = ExecutionObservation::new(recorder.clone());
    launch("first");
    launch("second");
    drop(watching);
    assert_eq!(recorder.kernels(), ["first", "second"]);
}

#[test]
#[serial_test::serial]
fn an_observation_that_ended_is_told_nothing_more() {
    let recorder = Arc::new(Recorder::default());
    drop(ExecutionObservation::new(recorder.clone()));
    launch("after");
    assert!(recorder.kernels().is_empty());
}

#[test]
#[serial_test::serial]
fn nothing_is_built_while_nothing_observes() {
    assert!(issue(|| unreachable!("built with nothing observing")).is_none());
}

#[test]
#[serial_test::serial]
fn a_scope_builds_nothing_while_nothing_observes() {
    let scope = ExecutionScope::communication(|| unreachable!("built with nothing observing"));
    assert_eq!(ExecutionScope::current(), None);
    drop(scope);
}

#[test]
#[serial_test::serial]
#[cfg(feature = "std")]
fn a_scope_parents_what_is_issued_inside_it() {
    let watching = ExecutionObservation::new(Arc::new(Recorder::default()));
    let scope = ExecutionScope::communication(transfer);
    let inside = launch("inside").expect("an observer should be installed");
    drop(scope);
    drop(watching);
    assert!(inside.execution().parent.is_some());
}

#[test]
#[serial_test::serial]
fn nothing_installed_asks_for_no_timing() {
    assert_eq!(timing_requested(), TimingRequest::None);
}

/// Timing is opt-in, and the launch path asks before paying for it: an
/// observer that only wants the executions must not make every launch blocking.
#[test]
#[serial_test::serial]
fn an_observer_asks_for_no_timing_by_default() {
    let watching = ExecutionObservation::new(Arc::new(Recorder::default()));
    let reported = launch("a_kernel").expect("an observer should be installed");
    drop(watching);
    assert!(!reported.wants_timing());
}

/// A measurement of `micros`, as a backend without device timestamps
/// hands one over: already known, so reading it back waits on nothing.
fn measured(micros: u64) -> ProfileDuration {
    let start = cubecl_common::profile::Instant::now();
    ProfileDuration::new_system_time(start, start + Duration::from_micros(micros))
}

/// The same, claiming the device timer, to check what an observer is told
/// about how a measurement was taken.
fn measured_on_device(micros: u64) -> ProfileDuration {
    let start = cubecl_common::profile::Instant::now();
    let ticks = ProfileTicks::from_start_end(start, start + Duration::from_micros(micros));
    ProfileDuration::new(
        alloc::boxed::Box::pin(async move { Some(ticks) }),
        TimingMethod::Device,
    )
}

/// A measurement whose read ends the observation that asked for it, the
/// way a pass that finishes while its last kernel is still running does.
fn measured_while(micros: u64, during_the_read: impl FnOnce() + Send + 'static) -> ProfileDuration {
    let start = cubecl_common::profile::Instant::now();
    ProfileDuration::new(
        alloc::boxed::Box::pin(async move {
            during_the_read();
            Some(ProfileTicks::from_start_end(
                start,
                start + Duration::from_micros(micros),
            ))
        }),
        TimingMethod::System,
    )
}

#[test]
#[serial_test::serial]
fn a_measurement_is_read_back_for_an_observer_that_asked_for_it_resolved() {
    let timed = Arc::new(ReadBack::default());
    let watching = ExecutionObservation::new(timed.clone());
    let reported = launch("a_kernel").expect("an observer should be installed");
    reported.deliver_profile(measured(7));
    drop(watching);
    assert_eq!(
        *timed.0.lock(),
        [(
            reported.id(),
            Duration::from_micros(7),
            TimingMethod::System
        )]
    );
}

/// The method travels with the duration: a backend that fell back to the
/// system timer measured host wall around a blocking submit, and an observer
/// that could not tell would report it as device time.
#[test]
#[serial_test::serial]
fn a_reading_tells_the_observer_how_it_was_measured() {
    let timed = Arc::new(ReadBack::default());
    let watching = ExecutionObservation::new(timed.clone());
    let reported = launch("a_kernel").expect("an observer should be installed");
    reported.deliver_profile(measured_on_device(7));
    drop(watching);
    assert_eq!(
        *timed.0.lock(),
        [(
            reported.id(),
            Duration::from_micros(7),
            TimingMethod::Device
        )]
    );
}

/// The read-back runs outside the lock, so the guard can drop mid-read,
/// and the owner has already collected what it wanted.
#[test]
#[serial_test::serial]
fn an_observation_that_ended_mid_read_is_not_told_the_duration() {
    let timed = Arc::new(ReadBack::default());
    let watching = Arc::new(Mutex::new(Some(ExecutionObservation::new(timed.clone()))));
    let reported = launch("a_kernel").expect("an observer should be installed");
    let ends_the_observation = watching.clone();

    reported.deliver_profile(measured_while(7, move || {
        drop(ends_the_observation.lock().take())
    }));

    assert!(timed.0.lock().is_empty());
}

/// Kept unread, the kernels around a timed launch keep running back to back.
#[test]
#[serial_test::serial]
fn an_observer_can_keep_a_measurement_unread() {
    let kept = Kept::asking(TimingRequest::Deferred(TimedExecutions::Kernels));
    let watching = ExecutionObservation::new(kept.clone());
    let reported = launch("a_kernel").expect("an observer should be installed");
    reported.deliver_profile(measured(3));
    drop(watching);
    let kept: Vec<_> = kept.0.lock().iter().map(|(id, _)| *id).collect();
    assert_eq!(kept, [reported.id()]);
}

/// A measurement belongs to the observer the execution was reported to: one
/// installed after it never saw the execution.
#[test]
#[serial_test::serial]
fn a_measurement_does_not_reach_an_observer_that_never_saw_its_execution() {
    let watching = ExecutionObservation::new(Kept::asking(TimingRequest::Deferred(
        TimedExecutions::Kernels,
    )));
    let reported = launch("a_kernel").expect("an observer should be installed");
    drop(watching);
    let replacement = Kept::asking(TimingRequest::Deferred(TimedExecutions::Kernels));
    let replacing = ExecutionObservation::new(replacement.clone());

    reported.deliver_profile(measured(3));

    drop(replacing);
    assert!(replacement.0.lock().is_empty());
}

/// How a measurement is delivered is the answer the observer gave when the
/// execution was issued, so an answer that changes afterwards loses nothing.
#[test]
#[serial_test::serial]
fn a_measurement_is_delivered_as_its_observer_asked_when_it_was_issued() {
    let kept = Kept::asking(TimingRequest::Deferred(TimedExecutions::Kernels));
    let watching = ExecutionObservation::new(kept.clone());
    let reported = launch("a_kernel").expect("an observer should be installed");
    *kept.1.lock() = TimingRequest::None;

    reported.deliver_profile(measured(3));

    drop(watching);
    assert_eq!(kept.0.lock().len(), 1);
}

#[test]
#[serial_test::serial]
fn a_communication_part_is_handed_over_unread() {
    let every = Kept::asking(TimingRequest::Deferred(
        TimedExecutions::KernelsAndCommunications,
    ));
    let watching = ExecutionObservation::new(every.clone());
    let part = launch("standing_in_for_a_part").expect("an observer should be installed");
    part.deliver_unread(measured(3));
    drop(watching);
    let kept: Vec<_> = every.0.lock().iter().map(|(id, _)| *id).collect();
    assert_eq!(kept, [part.id()]);
}

/// Keeps its measurements unread, under the request it currently answers.
struct Kept(
    Mutex<Vec<(ExecutionId, ProfileDuration)>>,
    Mutex<TimingRequest>,
);

impl Kept {
    fn asking(timing: TimingRequest) -> Arc<Self> {
        Arc::new(Self(Mutex::new(Vec::new()), Mutex::new(timing)))
    }
}

impl ExecutionObserver for Kept {
    fn observe(&self, _execution: &Execution) {}
    fn timing(&self) -> TimingRequest {
        *self.1.lock()
    }
    fn observe_measurement(&self, execution: &Execution, measurement: Measurement) {
        let Measurement::Unread(profile) = measurement else {
            panic!("asked to keep its measurements unread, with no logger reading them");
        };
        self.0.lock().push((execution.id, profile));
    }
}

#[derive(Default)]
struct Recorder(Mutex<Vec<Execution>>);

impl Recorder {
    fn kernels(&self) -> Vec<&'static str> {
        self.0
            .lock()
            .iter()
            .filter_map(|execution| match execution.kind {
                ExecutionKind::Kernel { name, .. } => Some(name),
                _ => None,
            })
            .collect()
    }
}

impl ExecutionObserver for Recorder {
    fn observe(&self, execution: &Execution) {
        self.0.lock().push(execution.clone());
    }
}

#[derive(Default)]
struct ReadBack(Mutex<Vec<(ExecutionId, Duration, TimingMethod)>>);

impl ExecutionObserver for ReadBack {
    fn observe(&self, _execution: &Execution) {}
    fn timing(&self) -> TimingRequest {
        TimingRequest::Resolved
    }
    fn observe_measurement(&self, execution: &Execution, measurement: Measurement) {
        let Measurement::Read { duration, method } = measurement else {
            panic!("asked for its measurements read back");
        };
        self.0.lock().push((execution.id, duration, method));
    }
}
