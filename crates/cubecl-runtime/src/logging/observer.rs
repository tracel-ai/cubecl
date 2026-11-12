//! Watching what the process asks its devices to run, from the threads that ask.
//!
//! The profiling logger ([`ServerLogger`](super::ServerLogger)) already knows
//! every kernel that runs, but it formats them into a sink: durations are
//! aggregated by name into a private table and written out through a detached
//! task. That is the right shape for reading a log and the wrong one for a
//! caller that wants to *attribute* the launches: by the time a line is
//! written, the context that issued it is gone, and nothing about the ordering
//! is guaranteed against the caller's own state.
//!
//! An observer is the other half: a hook called **synchronously, on the thread
//! that issued an [`Execution`], before it is submitted**. That is the only
//! point where host-side context still exists, so a caller that keeps a stack
//! of what it is currently doing can pair a kernel, a collective or a transfer
//! with it.
//!
//! ```
//! use std::collections::HashMap;
//! use std::sync::{Arc, Mutex};
//!
//! use cubecl_runtime::logging::{Execution, ExecutionKind, ExecutionObservation, ExecutionObserver};
//!
//! #[derive(Default)]
//! struct CountThem(Mutex<HashMap<&'static str, usize>>);
//!
//! impl ExecutionObserver for CountThem {
//!     fn issued(&self, execution: &Execution) {
//!         if let ExecutionKind::Kernel { name, .. } = execution.kind {
//!             *self.0.lock().unwrap().entry(name).or_default() += 1;
//!         }
//!     }
//! }
//!
//! let counts = Arc::new(CountThem::default());
//! let watching = ExecutionObservation::new(counts.clone());
//! the_pass_to_attribute();
//! drop(watching);
//!
//! for (kernel, count) in counts.0.lock().unwrap().iter() {
//!     println!("{count} × {kernel}");
//! }
//! # fn the_pass_to_attribute() {}
//! ```
//!
//! # Cost
//!
//! One relaxed atomic load per execution when nothing is installed, which is
//! every ordinary run: no [`ExecutionId`] is minted and nothing is allocated. An observer that asks for timing is the
//! expensive case, and pays per launch → [`timing`](ExecutionObserver::timing).
//!
//! # What it reports
//!
//! An execution that was **issued**, not one that finished: [`issued`] arrives
//! before it reaches the server, and a measurement, when one was asked for,
//! arrives afterwards through [`measured`]. Only kernels are measured.
//!
//! Issued is not the same as executed. A launch whose
//! [mode](crate::execution::LaunchMode) discards it is still compiled and
//! still reported here, and is then discarded instead of reaching the device; a
//! duration measured over one is the compile and the submit, with no kernel
//! under it. Which launches are discarded is decided per stream, where they are
//! issued, and is not visible here.
//!
//! A replayed [`Graph`](crate::client::Graph) is the other direction: its
//! kernels were observed once, when the capture window recorded them, and a
//! replay re-executes the whole graph without issuing them again, so an
//! observed benchmark of a graph-replayed pass reports the capture run and
//! nothing per replay.
//!
//! [`issued`]: ExecutionObserver::issued
//! [`measured`]: ExecutionObserver::measured

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

use super::{Execution, ExecutionId, ExecutionKind, ExecutionScope};
/// Re-exported because [`Measurement`] names them: an implementor that reached
/// this trait through `cubecl` cannot otherwise spell its own arguments, and
/// `Duration` is not `core::time::Duration` on every target.
pub use cubecl_common::profile::{Duration, ProfileDuration, ProfileTicks, TimingMethod};
use cubecl_environment::sync::RwLock;

/// What an observer asks be done with each launch's measurement.
///
/// Declared once, by [`timing`](ExecutionObserver::timing): the launch path has
/// to know where a measurement will go *before* it takes one.
///
/// **It must not change while an observation is installed.** A launch bracketed
/// under one answer and delivered under another loses a measurement that has
/// already been paid for. An observer that measures only part of a run keeps
/// that region in its own state, rather than changing its answer here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimingRequest {
    /// Don't time launches. The default, because timing is not free.
    #[default]
    None,
    /// Time each launch and read its measurement back, delivering it as
    /// [`Measurement::Read`].
    ///
    /// Reading blocks the issuing thread until the kernel has run, so kernels
    /// run one at a time and their sum is not the pass's device time. Right for
    /// each kernel's own cost, wrong for measuring a pipeline.
    Resolved,
    /// Time each launch and hand its measurement over **unread**, as
    /// [`Measurement::Unread`].
    ///
    /// Nothing waits, so the kernels around it keep running back to back and
    /// the observer reads them once the pass is over. Right for measuring where
    /// a pass spends its time.
    ///
    /// **Only while the profiling logger is off.** A measurement is read once,
    /// and past [`ExecutionOnly`](super::ProfileLevel::ExecutionOnly) the logger
    /// needs it too, so the launch path reads it there and delivers
    /// [`Measurement::Read`] instead, which runs the kernels one at a time. An
    /// observer whose figures only mean something on a pipelined pass should
    /// say so when timings arrive that way.
    Deferred,
}

/// A launch's measurement, as an observer asked for it.
pub enum Measurement {
    /// Not yet read back, under [`TimingRequest::Deferred`]. Keep it, and read
    /// it once the work being measured is over.
    ///
    /// Where the backend times on the device without waiting (CUDA, HIP, and
    /// wgpu's timestamp queries) the measurement is two events in the stream
    /// that nothing has waited for, and the resolved [`ProfileTicks`] carry the
    /// window's start and end on one clock, so an observer can also say where
    /// the device sat idle between kernels. Metal waits for the window and
    /// places it at the moment it was read: its lengths are device time, its
    /// starts and ends do not line up, and keeping the measurement saves
    /// nothing there.
    ///
    /// Reading it, [`ProfileDuration::resolve`], blocks until the device has
    /// reached both events: doing that inside
    /// [`measured`](ExecutionObserver::measured) would hold the observer's lock
    /// for the length of the kernel, which is the thing
    /// [`Deferred`](TimingRequest::Deferred) exists to avoid.
    Unread(ProfileDuration),
    /// Read back by the launch path: under [`TimingRequest::Resolved`], and
    /// under [`TimingRequest::Deferred`] whenever the profiling logger reads
    /// measurements as well.
    Read {
        /// How long the kernel took.
        duration: Duration,
        /// How it was measured, which is not a detail.
        ///
        /// A backend falls back to [`System`](TimingMethod::System) where it
        /// cannot get a device timestamp (wgpu does exactly that once the
        /// timestamp-query budget is spent) and a system timing is host wall
        /// around a blocking submit, which includes submission, sync, and the
        /// kernel's compilation on its first launch, rather than the kernel.
        /// The two are not the same measurement, and an observer reporting them
        /// as one shows a number that moves several-fold between runs.
        method: TimingMethod,
    },
}

/// Notified of every execution the process issues, on the thread that issued it.
///
/// One trait for every kind of [`Execution`]: an implementer matches the kinds
/// it cares about.
///
/// Implementations must be cheap, must not launch, and must not install or drop
/// an [`ExecutionObservation`] or open an [`ExecutionScope`]: this runs inside
/// the launch path, before the work reaches the server, and holds the lock that
/// guards the installed observer.
pub trait ExecutionObserver: Send + Sync {
    /// An execution was issued.
    fn issued(&self, execution: &Execution);

    /// Whether each launch should be timed, and how its measurement reaches
    /// [`measured`](Self::measured).
    ///
    /// **[`None`](TimingRequest::None) by default, because timing is not free.**
    /// Bracketing a launch with profile markers costs the issuing thread a round
    /// trip to the server per kernel, and [`Resolved`](TimingRequest::Resolved)
    /// also blocks until the kernel has run, removing the overlap between
    /// kernels. An observer that only wants to know *which* kernels ran should
    /// leave this alone; one measuring where a pass spends its time wants
    /// [`Deferred`](TimingRequest::Deferred).
    ///
    /// Communications are issued at every level, and never timed.
    ///
    /// Two situations refuse the measurement without refusing the launch:
    ///
    /// * A profile the server cannot take: a graph capture window refuses
    ///   them on the spot. The kernel is still launched, still reported to
    ///   [`issued`](Self::issued), and the measurement is skipped for it, with
    ///   a warning in the log.
    /// * Don't ask for [`Resolved`](TimingRequest::Resolved) around **collective**
    ///   kernels. Reading blocks until the kernel completes, and a collective
    ///   completes only when its peers launch: a thread that issues more than
    ///   one side of a collective deadlocks waiting for the first.
    fn timing(&self) -> TimingRequest {
        TimingRequest::None
    }

    /// An execution reported to [`issued`](Self::issued) was measured.
    ///
    /// Called on the thread that issued it, after the launch, under the same
    /// lock as every other method here, so keeping the measurement must be
    /// cheap. **An observer that asked to keep measurements unread still
    /// receives [`Measurement::Read`]** whenever the profiling logger is on.
    ///
    /// It arrives *after* the launch rather than before it, so an observer
    /// pairing executions with its own state should do that in
    /// [`issued`](Self::issued) and match the [`ExecutionId`] here. A
    /// measurement goes to the observer the execution was reported to, and only
    /// while it is still installed: an observation that ends mid-read is not
    /// told.
    ///
    /// The default drops it: an observer asking for timing owes an
    /// implementation.
    #[allow(unused_variables)]
    fn measured(&self, execution: &Execution, measurement: Measurement) {}
}

/// Watches every execution the process issues for as long as it lives, then
/// puts back whatever it replaced.
///
/// Process-wide rather than per client: a caller attributing executions wants
/// every one its work causes, and work reaches several clients on several
/// streams. Filtering is the observer's to do, since only it knows what it is
/// attributing to.
///
/// A guard rather than an install/stop pair so the scope being attributed is
/// the scope the observer is installed for, with no restore step a caller can
/// skip on an early return. There is one slot, so a second observation replaces
/// the first for its lifetime; guards dropped in the order they were taken
/// leave the process as they found it.
#[must_use = "an observation stops as soon as it is dropped"]
pub struct ExecutionObservation {
    previous: Option<Arc<dyn ExecutionObserver>>,
}

impl ExecutionObservation {
    /// Installs `observer` until the guard drops.
    pub fn new(observer: Arc<dyn ExecutionObserver>) -> Self {
        let previous = OBSERVER.write().replace(observer);
        // Last, so the flag is never set over an empty slot.
        OBSERVING.store(true, Ordering::Relaxed);
        Self { previous }
    }
}

impl Drop for ExecutionObservation {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let still_observed = previous.is_some();
        *OBSERVER.write() = previous;
        // Last, so the flag is never cleared while an observer is still
        // installed: the executions this guard covers are the ones it must not
        // miss, and an issue that reads the flag in between finds an empty
        // slot and does nothing.
        OBSERVING.store(still_observed, Ordering::Relaxed);
    }
}

impl core::fmt::Debug for ExecutionObservation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutionObservation")
            .field("replaced_an_observer", &self.previous.is_some())
            .finish()
    }
}

/// Tell the installed observer, if there is one, of the execution `kind` builds,
/// parented by the [`ExecutionScope`] open on this thread. Returns what it was
/// told, or `None` without building anything when nothing observes.
pub(crate) fn issue(kind: impl FnOnce() -> ExecutionKind) -> Option<Execution> {
    if !OBSERVING.load(Ordering::Relaxed) {
        return None;
    }
    // Built before taking the lock, so the caller's closure never runs under it.
    let execution = Execution {
        id: ExecutionId::mint(),
        parent: ExecutionScope::current(),
        kind: kind(),
    };
    OBSERVER.read().as_ref()?.issued(&execution);
    Some(execution)
}

/// Whether the installed observer asked for each launch to be timed.
pub(crate) fn timing_wanted() -> bool {
    timing_requested() != TimingRequest::None
}

/// Say once that the profiling logger is taking the measurements an observer
/// asked to keep.
///
/// A measurement is read once, so with the logger set past `ExecutionOnly` the
/// launch path reads it and the observer is told a duration instead: it never
/// receives [`Measurement::Unread`] and its kernels stop overlapping, which is
/// a pass other than the one it asked to measure, with nothing in the numbers
/// to say so.
///
/// Once per process, because it is a configuration mistake and not a per-launch
/// event: at one line per kernel it would be the log.
pub(crate) fn warn_logger_takes_deferred_measurements() {
    static SAID: AtomicBool = AtomicBool::new(false);
    if timing_requested() == TimingRequest::Deferred && !SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            "The profiling logger is reading every launch's measurement, so this run's \
             execution observer is told durations instead of keeping them, and its kernels \
             run one at a time. Turn the profile logging off to measure the pass as it runs."
        );
    }
}

/// What the installed observer asked be done with each measurement.
fn timing_requested() -> TimingRequest {
    if !OBSERVING.load(Ordering::Relaxed) {
        return TimingRequest::None;
    }
    OBSERVER
        .read()
        .as_ref()
        .map_or(TimingRequest::None, |observer| observer.timing())
}

/// Read `profile` for the observer, and hand the reading back for the logger.
///
/// Both want the same measurement and a measurement is read once, so the
/// launch path reads it here and passes on what it got. Reading blocks for the
/// length of the kernel, so it happens with the slot's lock released, and the
/// reading reaches only the observer that was installed when the read started,
/// the same guarantee [`notify_profiled`] gives.
pub(crate) fn read_and_notify_timed(
    execution: &Execution,
    profile: ProfileDuration,
) -> ProfileDuration {
    let method = profile.timing_method();
    let observer = installed_observer();
    let ticks = cubecl_environment::future::block_on(profile.resolve());

    match (&ticks, &observer) {
        (Some(ticks), Some(observer)) => {
            deliver_read(observer, execution, ticks.duration(), method)
        }
        // Nothing to report: the window carried no measurement, and a zero
        // would put a launch that was never timed in the timings.
        (None, _) => warn_unmeasured(execution),
        (Some(_), None) => {}
    }

    ProfileDuration::new(alloc::boxed::Box::pin(async move { ticks }), method)
}

/// The observer installed right now, taken out from under the lock so a read
/// can outlive holding it.
fn installed_observer() -> Option<Arc<dyn ExecutionObserver>> {
    if !OBSERVING.load(Ordering::Relaxed) {
        return None;
    }
    OBSERVER.read().as_ref().map(Arc::clone)
}

/// Report a reading to `observer`, if it is still the installed one.
///
/// A reading belongs to the observer the execution was reported to, and reading
/// a measurement takes as long as the kernel: an observation that ended
/// underneath the read is done receiving, and the one that replaced it never
/// saw the execution.
fn deliver_read(
    observer: &Arc<dyn ExecutionObserver>,
    execution: &Execution,
    duration: Duration,
    method: TimingMethod,
) {
    let slot = OBSERVER.read();
    if slot
        .as_ref()
        .is_some_and(|installed| Arc::ptr_eq(installed, observer))
    {
        observer.measured(execution, Measurement::Read { duration, method });
    }
}

fn warn_unmeasured(execution: &Execution) {
    log::warn!(
        "Skipped timing {:?} for its observer: the profiled window carried no measurement",
        execution.kind
    );
}

/// Deliver a launch's measurement the way the observer asked for it.
///
/// [`Deferred`](TimingRequest::Deferred) hands it over unread and is done.
/// [`Resolved`](TimingRequest::Resolved) reads it here with the slot's lock
/// released, since holding the lock across a read would stall every other
/// launching thread behind this one's kernel, and delivers through
/// [`deliver_read`] like every other reading.
pub(crate) fn notify_profiled(execution: &Execution, profile: ProfileDuration) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    let (observer, profile) = {
        let slot = OBSERVER.read();
        let Some(observer) = slot.as_ref() else {
            return;
        };
        match observer.timing() {
            TimingRequest::Deferred => {
                observer.measured(execution, Measurement::Unread(profile));
                return;
            }
            // Asked for no timing between the launch being bracketed and this
            // call: an observation that ended underneath it, or one answering
            // differently at two moments, which `TimingRequest` forbids. Said
            // out loud because the measurement has already been paid for, and a
            // gap in a breakdown with nothing in the log is unattributable.
            TimingRequest::None => {
                // Once per process, like the logger's: a `timing()` that
                // answers differently at two moments does so on every launch,
                // and at one line each the warning would be the log.
                static SAID: AtomicBool = AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "Dropped a timing of {:?}: its observer asked for none by the time the \
                         measurement arrived",
                        execution.kind
                    );
                }
                return;
            }
            TimingRequest::Resolved => (Arc::clone(observer), profile),
        }
    };

    let method = profile.timing_method();
    let Some(ticks) = cubecl_environment::future::block_on(profile.resolve()) else {
        warn_unmeasured(execution);
        return;
    };

    deliver_read(&observer, execution, ticks.duration(), method);
}

/// Whether anything is watching. Separate from the observer itself so the
/// unobserved path, every ordinary run, is one relaxed load rather than a
/// lock acquisition.
static OBSERVING: AtomicBool = AtomicBool::new(false);

static OBSERVER: RwLock<Option<Arc<dyn ExecutionObserver>>> = RwLock::new(None);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::{
        Communication, ExecutionPlace, ExecutionScope, StreamRole, TransferRoute,
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

        let timed = Arc::new(Timed::default());
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
    fn measured_while(
        micros: u64,
        during_the_read: impl FnOnce() + Send + 'static,
    ) -> ProfileDuration {
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
        let timed = Arc::new(Timed::default());
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
        let timed = Arc::new(Timed::default());
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
        let timed = Arc::new(Timed::default());
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
        let kept = Arc::new(Kept::default());
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

    #[derive(Default)]
    struct Kept(Mutex<Vec<(ExecutionId, ProfileDuration)>>);

    impl ExecutionObserver for Kept {
        fn issued(&self, _execution: &Execution) {}
        fn timing(&self) -> TimingRequest {
            TimingRequest::Deferred
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
    struct Timed(Mutex<Vec<(ExecutionId, Duration, TimingMethod)>>);

    impl ExecutionObserver for Timed {
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
}
