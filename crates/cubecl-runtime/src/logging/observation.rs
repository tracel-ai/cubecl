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
//! arrives afterwards through [`measured`]. Kernels are measured, and so are
//! communication parts and the waits on them when the observer
//! [times them too](Timed::KernelsAndCommunications).
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

use super::{
    Duration, Execution, ExecutionId, ExecutionKind, ExecutionObserver, ExecutionScope,
    Measurement, ProfileDuration, Timed, TimingMethod, TimingRequest,
};
use cubecl_environment::sync::RwLock;

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

/// Whether the installed observer times communications, and so is told of their waits.
pub(crate) fn times_communications() -> bool {
    timing_requested() == TIMES_COMMUNICATIONS
}

const TIMES_COMMUNICATIONS: TimingRequest =
    TimingRequest::Deferred(Timed::KernelsAndCommunications);

/// Hand a communication part's or wait's measurement over unread, to an observer that times
/// communications.
///
/// Never read here, whatever the observer answers by now: a part ends only once its peers' run,
/// so a thread that queues several would wait forever on the first.
pub(crate) fn notify_unread(execution: &Execution, measurement: ProfileDuration) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    if let Some(observer) = OBSERVER.read().as_ref()
        && observer.timing() == TIMES_COMMUNICATIONS
    {
        observer.measured(execution, Measurement::Unread(measurement));
    }
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
    if matches!(timing_requested(), TimingRequest::Deferred(_))
        && !SAID.swap(true, Ordering::Relaxed)
    {
        log::warn!(
            "The profiling logger is reading every launch's measurement, so this run's \
             execution observer is told durations instead of keeping them, and its kernels \
             run one at a time. Turn the profile logging off to measure the pass as it runs."
        );
    }
}

/// What the installed observer asked be done with each measurement.
pub(crate) fn timing_requested() -> TimingRequest {
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
            TimingRequest::Deferred(_) => {
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
mod tests;
