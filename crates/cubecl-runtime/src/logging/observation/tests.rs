use super::*;
use crate::logging::ProfileTicks;
use crate::logging::{Communication, ExecutionPlace, ExecutionScope, StreamRole, TransferRoute};
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

fn launch(name: &'static str) -> Option<Execution> {
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
    assert_eq!(recorder.kernels(), ["first", "second"]);

    drop(watching);
    launch("after");
    assert_eq!(
        recorder.0.lock().len(),
        2,
        "an observation that ended must not keep receiving"
    );
}

#[test]
#[serial_test::serial]
fn nothing_is_built_or_minted_while_nothing_observes() {
    let recorder = Arc::new(Recorder::default());
    let watching = ExecutionObservation::new(recorder.clone());
    let before = launch("before").unwrap().id;
    drop(watching);

    assert!(issue(|| unreachable!("built with nothing observing")).is_none());
    let scope = ExecutionScope::communication(|| unreachable!("built with nothing observing"));
    assert_eq!(ExecutionScope::current(), None);
    drop(scope);

    let watching = ExecutionObservation::new(recorder.clone());
    let after = launch("after").unwrap().id;
    drop(watching);
    assert_eq!(after.0, before.0 + 1);
}

#[test]
#[serial_test::serial]
#[cfg(feature = "std")]
fn a_scope_parents_what_is_issued_inside_it_and_restores_its_parent() {
    let recorder = Arc::new(Recorder::default());
    let watching = ExecutionObservation::new(recorder.clone());

    let outer = ExecutionScope::communication(transfer);
    launch("in_the_outer");
    {
        let _inner = ExecutionScope::communication(transfer);
        launch("in_the_inner");
    }
    launch("back_in_the_outer");
    drop(outer);
    launch("outside");
    drop(watching);

    let issued = recorder.0.lock().clone();
    let (outer, inner) = (issued[0].id, issued[2].id);
    let parents: Vec<_> = issued.iter().map(|execution| execution.parent).collect();
    assert_eq!(
        parents,
        [
            None,
            Some(outer),
            Some(outer),
            Some(inner),
            Some(outer),
            None
        ]
    );
    assert!(outer < inner, "ids increase in issue order");
}

/// Timing is opt-in, and the launch path asks before paying for it: an
/// observer that only wants the executions must not make every launch blocking.
#[test]
#[serial_test::serial]
fn timing_is_off_unless_an_observer_asks() {
    assert!(!timing_wanted(), "nothing installed, nothing to time");

    let names_only = ExecutionObservation::new(Arc::new(Recorder::default()));
    assert!(!timing_wanted(), "executions only, by default");
    drop(names_only);

    let timed = Arc::new(ReadBack::default());
    let watching = ExecutionObservation::new(timed.clone());
    assert!(timing_wanted());
    let execution = launch("a_kernel").unwrap();
    read_and_notify_timed(&execution, measured_on_device(7));
    // The method travels with the duration: a backend that fell back to
    // the system timer measured host wall around a blocking submit, and an
    // observer that could not tell would report it as device time.
    assert_eq!(
        *timed.0.lock(),
        [(execution.id, Duration::from_micros(7), TimingMethod::Device)]
    );

    drop(watching);
    assert!(!timing_wanted());
}

/// A nested observation puts back the one it replaced, so a caller that
/// watches a sub-pass does not silently take the process's only slot from
/// whoever was already watching.
#[test]
#[serial_test::serial]
fn an_observation_restores_the_one_it_replaced() {
    let outer = Arc::new(Recorder::default());
    let inner = Arc::new(Recorder::default());

    let watching_outer = ExecutionObservation::new(outer.clone());
    {
        let _watching_inner = ExecutionObservation::new(inner.clone());
        launch("during_the_inner_pass");
    }
    launch("after_the_inner_pass");
    drop(watching_outer);
    launch("unobserved");

    assert_eq!(inner.kernels(), ["during_the_inner_pass"]);
    assert_eq!(outer.kernels(), ["after_the_inner_pass"]);
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
    let execution = launch("a_kernel").unwrap();
    notify_profiled(&execution, measured(7));
    drop(watching);

    assert_eq!(
        *timed.0.lock(),
        [(execution.id, Duration::from_micros(7), TimingMethod::System)]
    );
}

/// The read-back runs outside the lock, so the guard can drop mid-read,
/// and the owner has already collected what it wanted.
#[test]
#[serial_test::serial]
fn an_observation_that_ended_mid_read_is_not_told_the_duration() {
    let timed = Arc::new(ReadBack::default());
    let watching = Arc::new(Mutex::new(Some(ExecutionObservation::new(timed.clone()))));
    let execution = launch("a_kernel").unwrap();

    let ends_the_observation = watching.clone();
    notify_profiled(
        &execution,
        measured_while(7, move || drop(ends_the_observation.lock().take())),
    );

    assert!(watching.lock().is_none(), "the read-back ended it");
    assert!(
        timed.0.lock().is_empty(),
        "an observation that ended must not keep receiving"
    );
}

/// The same guarantee on the path the profiling logger takes, where the
/// launch path reads the measurement for both of them: the reading still
/// reaches the logger, and no observer is told a launch it never saw.
#[test]
#[serial_test::serial]
fn an_observation_that_ended_mid_read_is_not_told_the_loggers_reading() {
    let timed = Arc::new(ReadBack::default());
    let watching = Arc::new(Mutex::new(Some(ExecutionObservation::new(timed.clone()))));
    let execution = launch("a_kernel").unwrap();

    let ends_the_observation = watching.clone();
    let for_the_logger = read_and_notify_timed(
        &execution,
        measured_while(7, move || drop(ends_the_observation.lock().take())),
    );

    assert!(watching.lock().is_none(), "the read-back ended it");
    assert!(
        timed.0.lock().is_empty(),
        "an observation that ended must not keep receiving"
    );
    let ticks = cubecl_environment::future::block_on(for_the_logger.resolve())
        .expect("the logger still gets the reading");
    assert_eq!(
        ticks.duration(),
        Duration::from_micros(7),
        "read once, and handed on"
    );
}

/// Kept unread, the kernels around a timed launch keep running back to back.
#[test]
#[serial_test::serial]
fn an_observer_can_keep_a_measurement_unread() {
    let kept = Kept::asking(TimingRequest::Deferred(Timed::Kernels));
    let watching = ExecutionObservation::new(kept.clone());
    let first = launch("first").unwrap();
    let second = launch("second").unwrap();
    notify_profiled(&first, measured(3));
    notify_profiled(&second, measured(5));
    drop(watching);

    let read: Vec<(ExecutionId, Duration)> = core::mem::take(&mut *kept.0.lock())
        .into_iter()
        .map(|(id, profile)| {
            let ticks = cubecl_environment::future::block_on(profile.resolve())
                .expect("a system measurement always carries its ticks");
            (id, ticks.duration())
        })
        .collect();
    assert_eq!(
        read,
        [
            (first.id, Duration::from_micros(3)),
            (second.id, Duration::from_micros(5))
        ],
        "in issue order, each still carrying its own window"
    );
}

#[test]
#[serial_test::serial]
fn a_communication_is_measured_only_for_an_observer_timing_communications() {
    let deferred = Kept::asking(TimingRequest::Deferred(Timed::Kernels));
    let watching = ExecutionObservation::new(deferred.clone());
    let part = launch("standing_in_for_a_part").unwrap();
    notify_unread(&part, measured(3));
    drop(watching);
    assert!(deferred.0.lock().is_empty());

    let every = Kept::asking(TimingRequest::Deferred(Timed::KernelsAndCommunications));
    let watching = ExecutionObservation::new(every.clone());
    assert!(times_communications());
    let part = launch("standing_in_for_a_part").unwrap();
    let kernel = launch("a_kernel").unwrap();
    notify_unread(&part, measured(3));
    notify_profiled(&kernel, measured(5));
    drop(watching);

    let kept: Vec<_> = every.0.lock().iter().map(|(id, _)| *id).collect();
    assert_eq!(kept, [part.id, kernel.id], "both kept unread");
}

struct Kept(Mutex<Vec<(ExecutionId, ProfileDuration)>>, TimingRequest);

impl Kept {
    fn asking(timing: TimingRequest) -> Arc<Self> {
        Arc::new(Self(Mutex::new(Vec::new()), timing))
    }
}

impl ExecutionObserver for Kept {
    fn issued(&self, _execution: &Execution) {}
    fn timing(&self) -> TimingRequest {
        self.1
    }
    fn measured(&self, execution: &Execution, measurement: Measurement) {
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
    fn issued(&self, execution: &Execution) {
        self.0.lock().push(execution.clone());
    }
}

#[derive(Default)]
struct ReadBack(Mutex<Vec<(ExecutionId, Duration, TimingMethod)>>);

impl ExecutionObserver for ReadBack {
    fn issued(&self, _execution: &Execution) {}
    fn timing(&self) -> TimingRequest {
        TimingRequest::Resolved
    }
    fn measured(&self, execution: &Execution, measurement: Measurement) {
        let Measurement::Read { duration, method } = measurement else {
            panic!("asked for its measurements read back");
        };
        self.0.lock().push((execution.id, duration, method));
    }
}
