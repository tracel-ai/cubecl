//! Watching kernel launches from the process that issues them.
//!
//! The profiling logger ([`ServerLogger`](super::ServerLogger)) already knows
//! every kernel that runs, but it formats them into a sink: durations are
//! aggregated by name into a private table and written out through a detached
//! task. That is the right shape for reading a log and the wrong one for a
//! caller that wants to *attribute* the launches — by the time a line is
//! written, the context that issued it is gone, and nothing about the ordering
//! is guaranteed against the caller's own state.
//!
//! An observer is the other half: a hook called **synchronously, on the thread
//! that issued the launch, before it is submitted**. That is the only point
//! where host-side context still exists, so a caller that keeps a stack of what
//! it is currently doing can pair a kernel with it.
//!
//! ```
//! use std::collections::HashMap;
//! use std::sync::{Arc, Mutex};
//!
//! use cubecl_runtime::logging::{LaunchObservation, LaunchObserver};
//!
//! #[derive(Default)]
//! struct CountThem(Mutex<HashMap<&'static str, usize>>);
//!
//! impl LaunchObserver for CountThem {
//!     fn launched(&self, kernel: &'static str) {
//!         *self.0.lock().unwrap().entry(kernel).or_default() += 1;
//!     }
//! }
//!
//! let counts = Arc::new(CountThem::default());
//! let watching = LaunchObservation::new(counts.clone());
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
//! One relaxed atomic load per launch when nothing is installed, which is every
//! ordinary run. The kernel's name is a `&'static str` the kernel already
//! carries, so an idle hook allocates and formats nothing. An observer that
//! asks for timing is the expensive case, and pays per launch →
//! [`wants_timing`](LaunchObserver::wants_timing).
//!
//! # What it reports
//!
//! A launch that was **issued**, not one that finished: [`launched`] arrives
//! before the kernel reaches the server, and a duration — when one was asked
//! for — arrives separately, afterwards, through [`timed`].
//!
//! Issued is not the same as executed. Under a
//! [`DryRun`](crate::dry_run::DryRun) every launch is still compiled and still
//! reported here, and is then dropped instead of reaching the device; a
//! duration measured over one is the compile and the submit, with no kernel
//! under it. An observer that cares about the difference checks
//! [`dry_run`](crate::dry_run::dry_run).
//!
//! A replayed [`Graph`](crate::client::Graph) is the other direction: its
//! kernels were observed once, when the capture window recorded them, and a
//! replay re-executes the whole graph without issuing them again — so an
//! observed benchmark of a graph-replayed pass reports the capture run and
//! nothing per replay.
//!
//! [`launched`]: LaunchObserver::launched
//! [`timed`]: LaunchObserver::timed

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

/// Re-exported because [`LaunchObserver::timed`]'s signature names them: an
/// implementor that reached this trait through `cubecl` cannot otherwise spell
/// its own arguments, and `Duration` is not `core::time::Duration` on every
/// target.
pub use cubecl_common::profile::{Duration, TimingMethod};
use cubecl_environment::sync::RwLock;

/// Notified of every kernel launch, on the thread that issued it.
///
/// Implementations must be cheap, must not launch, and must not install or drop
/// a [`LaunchObservation`]: this runs inside the launch path, before the kernel
/// reaches the server, and holds the lock that guards the installed observer.
pub trait LaunchObserver: Send + Sync {
    /// A kernel was issued, named as the kernel names itself. Pass it through
    /// [`type_name_format`](crate::config::type_name_format) to shorten it the
    /// way the profiling logger does.
    fn launched(&self, kernel: &'static str);

    /// Whether each launch should be timed, and [`timed`](Self::timed) called
    /// with what it took.
    ///
    /// **Off by default, because it is not free.** Timing a launch means
    /// bracketing it with profile markers and resolving them, which costs a
    /// blocking submit per kernel and removes the overlap between them. An
    /// observer that only wants to know *which* kernels ran should leave this
    /// alone; one measuring where a pass spends its time is paying for the
    /// answer either way.
    ///
    /// Two situations refuse the measurement without refusing the launch:
    ///
    /// * A profile the server cannot take — a graph capture window refuses
    ///   them on the spot. The kernel is still launched, still reported to
    ///   [`launched`](Self::launched), and [`timed`](Self::timed) is skipped
    ///   for it, with a warning in the log.
    /// * Don't ask for timing around **collective** kernels. Resolving a
    ///   launch blocks the issuing thread until the kernel completes, and a
    ///   collective kernel completes only when its peers launch — a thread
    ///   that issues more than one side of a collective deadlocks waiting for
    ///   the first.
    fn wants_timing(&self) -> bool {
        false
    }

    /// A kernel finished, and took this long.
    ///
    /// Only called when [`wants_timing`](Self::wants_timing) is true. It
    /// arrives *after* the launch rather than before it, so an observer
    /// pairing kernels with its own state should do that in
    /// [`launched`](Self::launched) and use this only for the duration.
    ///
    /// **`method` is not a detail.** A backend falls back to
    /// [`System`](TimingMethod::System) where it cannot get a device
    /// timestamp — wgpu does exactly that once the timestamp-query budget is
    /// spent — and a system timing is host wall around a blocking submit,
    /// which includes submission, sync, and the kernel's compilation on its
    /// first launch, rather than the kernel. The two are not the same
    /// measurement and an observer reporting them as one will show a number
    /// that moves several-fold between runs.
    fn timed(&self, _kernel: &'static str, _duration: Duration, _method: TimingMethod) {}
}

/// Watches every launch the process issues for as long as it lives, then puts
/// back whatever it replaced.
///
/// Process-wide rather than per client: a caller attributing launches wants
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
pub struct LaunchObservation {
    previous: Option<Arc<dyn LaunchObserver>>,
}

impl LaunchObservation {
    /// Installs `observer` until the guard drops.
    pub fn new(observer: Arc<dyn LaunchObserver>) -> Self {
        let previous = OBSERVER.write().replace(observer);
        // Last, so the flag is never set over an empty slot.
        OBSERVING.store(true, Ordering::Relaxed);
        Self { previous }
    }
}

impl Drop for LaunchObservation {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let still_observed = previous.is_some();
        *OBSERVER.write() = previous;
        // Last, so the flag is never cleared while an observer is still
        // installed: the launches this guard covers are the ones it must not
        // miss, and a notify that reads the flag in between finds an empty
        // slot and does nothing.
        OBSERVING.store(still_observed, Ordering::Relaxed);
    }
}

impl core::fmt::Debug for LaunchObservation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LaunchObservation")
            .field("replaced_an_observer", &self.previous.is_some())
            .finish()
    }
}

/// Tell the installed observer, if there is one, that `kernel` was issued.
pub(crate) fn notify_launch(kernel: &'static str) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    if let Some(observer) = OBSERVER.read().as_ref() {
        observer.launched(kernel);
    }
}

/// Whether the installed observer asked for each launch to be timed.
pub(crate) fn timing_wanted() -> bool {
    if !OBSERVING.load(Ordering::Relaxed) {
        return false;
    }
    OBSERVER
        .read()
        .as_ref()
        .is_some_and(|observer| observer.wants_timing())
}

/// Report what a launch took, and how it was measured.
pub(crate) fn notify_timed(kernel: &'static str, duration: Duration, method: TimingMethod) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    if let Some(observer) = OBSERVER.read().as_ref() {
        observer.timed(kernel, duration, method);
    }
}

/// Whether anything is watching. Separate from the observer itself so the
/// unobserved path — every ordinary run — is one relaxed load rather than a
/// lock acquisition on the launch path.
static OBSERVING: AtomicBool = AtomicBool::new(false);

static OBSERVER: RwLock<Option<Arc<dyn LaunchObserver>>> = RwLock::new(None);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;
    use cubecl_environment::sync::Mutex;

    /// The order launches arrive in, which is what makes attribution possible:
    /// an observer is called before the launch is submitted, so whatever the
    /// caller was doing when it issued the kernel is still true.
    #[test]
    #[serial_test::serial]
    fn launches_arrive_in_issue_order() {
        let recorder = Arc::new(Recorder::default());
        let watching = LaunchObservation::new(recorder.clone());

        notify_launch("first");
        notify_launch("second");
        assert_eq!(*recorder.0.lock(), ["first", "second"]);

        drop(watching);
        notify_launch("after");
        assert_eq!(
            recorder.0.lock().len(),
            2,
            "an observation that ended must not keep receiving"
        );
    }

    /// Timing is opt-in, and the launch path asks before paying for it: an
    /// observer that only wants the names must not make every launch blocking.
    #[test]
    #[serial_test::serial]
    fn timing_is_off_unless_an_observer_asks() {
        assert!(!timing_wanted(), "nothing installed, nothing to time");

        let names_only = LaunchObservation::new(Arc::new(Recorder::default()));
        assert!(!timing_wanted(), "names only, by default");
        drop(names_only);

        let timed = Arc::new(Timed::default());
        let watching = LaunchObservation::new(timed.clone());
        assert!(timing_wanted());
        notify_timed("a_kernel", Duration::from_micros(7), TimingMethod::Device);
        // The method travels with the duration: a backend that fell back to
        // the system timer measured host wall around a blocking submit, and an
        // observer that could not tell would report it as device time.
        assert_eq!(
            *timed.0.lock(),
            [("a_kernel", Duration::from_micros(7), TimingMethod::Device)]
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

        let watching_outer = LaunchObservation::new(outer.clone());
        {
            let _watching_inner = LaunchObservation::new(inner.clone());
            notify_launch("during_the_inner_pass");
        }
        notify_launch("after_the_inner_pass");
        drop(watching_outer);
        notify_launch("unobserved");

        assert_eq!(*inner.0.lock(), ["during_the_inner_pass"]);
        assert_eq!(*outer.0.lock(), ["after_the_inner_pass"]);
    }

    #[derive(Default)]
    struct Recorder(Mutex<Vec<&'static str>>);

    impl LaunchObserver for Recorder {
        fn launched(&self, kernel: &'static str) {
            self.0.lock().push(kernel);
        }
    }

    #[derive(Default)]
    struct Timed(Mutex<Vec<(&'static str, Duration, TimingMethod)>>);

    impl LaunchObserver for Timed {
        fn launched(&self, _kernel: &'static str) {}
        fn wants_timing(&self) -> bool {
            true
        }
        fn timed(&self, kernel: &'static str, duration: Duration, method: TimingMethod) {
            self.0.lock().push((kernel, duration, method));
        }
    }
}
