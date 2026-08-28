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
//! ```ignore
//! struct CountThem(Mutex<HashMap<&'static str, usize>>);
//!
//! impl LaunchObserver for CountThem {
//!     fn launched(&self, kernel: &'static str) {
//!         *self.0.lock().unwrap().entry(kernel).or_default() += 1;
//!     }
//! }
//!
//! let _watching = observe_launches(Arc::new(CountThem::default()));
//! ```
//!
//! # Cost
//!
//! One relaxed atomic load per launch when nothing is installed, which is every
//! ordinary run. The kernel's name is a `&'static str` the kernel already
//! carries, so an idle hook allocates and formats nothing.
//!
//! # What it is not
//!
//! It reports that a launch was *issued*, not that it finished, and carries no
//! duration — the work is asynchronous and the timing belongs to the profiling
//! path. A caller wanting both pairs this with a profile region around the same
//! span.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

use cubecl_environment::sync::RwLock;

/// Notified of every kernel launch, on the thread that issued it.
///
/// Implementations must be cheap and must not launch: this runs inside the
/// launch path, before the kernel reaches the server.
pub trait LaunchObserver: Send + Sync {
    /// A kernel was issued, named as the kernel names itself. Pass it through
    /// [`type_name_format`](crate::config::type_name_format) to shorten it the
    /// way the profiling logger does.
    fn launched(&self, kernel: &'static str);
}

/// Whether anything is watching. Separate from the observer itself so the
/// unobserved path — every ordinary run — is one relaxed load rather than a
/// lock acquisition on the launch path.
static OBSERVING: AtomicBool = AtomicBool::new(false);

static OBSERVER: RwLock<Option<Arc<dyn LaunchObserver>>> = RwLock::new(None);

/// Install `observer` for the process, returning whatever it replaced.
///
/// Process-wide rather than per client: a caller attributing launches wants
/// every one its work causes, and work reaches several clients on several
/// streams. Filtering is the observer's to do, since only it knows what it is
/// attributing to.
pub fn observe_launches(observer: Arc<dyn LaunchObserver>) -> Option<Arc<dyn LaunchObserver>> {
    let previous = OBSERVER.write().replace(observer);
    OBSERVING.store(true, Ordering::Relaxed);
    previous
}

/// Stop watching, returning the observer that was installed.
pub fn stop_observing_launches() -> Option<Arc<dyn LaunchObserver>> {
    // Cleared first: a launch that reads the flag after this point takes the
    // unobserved path rather than the lock.
    OBSERVING.store(false, Ordering::Relaxed);
    OBSERVER.write().take()
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use cubecl_environment::sync::Mutex;

    #[derive(Default)]
    struct Recorder(Mutex<Vec<&'static str>>);

    impl LaunchObserver for Recorder {
        fn launched(&self, kernel: &'static str) {
            self.0.lock().push(kernel);
        }
    }

    /// One test, because the registry is one process-wide resource: split
    /// across `#[test]`s they run in parallel and each one's `stop` clears the
    /// observer another just installed. The interference is real rather than a
    /// test artifact — an observer is global by design, and two callers wanting
    /// one at once is a thing the API does not support.
    #[test]
    fn an_observer_is_installed_then_stopped() {
        assert!(
            stop_observing_launches().is_none(),
            "nothing watches until something is installed — every ordinary run"
        );
        notify_launch("a_kernel_nobody_asked_about");

        let recorder = Arc::new(Recorder::default());
        assert!(observe_launches(recorder.clone()).is_none());

        notify_launch("first");
        notify_launch("second");
        // In issue order, which is what makes attribution possible: the
        // observer is called before the launch is submitted, so whatever the
        // caller was doing is still true.
        assert_eq!(*recorder.0.lock(), ["first", "second"]);

        assert!(
            stop_observing_launches().is_some(),
            "the observer is handed back"
        );
        notify_launch("after");
        assert_eq!(
            recorder.0.lock().len(),
            2,
            "a stopped observer must not keep receiving"
        );
    }

    /// Installing over an observer hands the old one back, so a caller can put
    /// it back rather than silently taking the process's only slot.
    #[test]
    fn installing_over_one_returns_it() {
        let first = Arc::new(Recorder::default());
        let second = Arc::new(Recorder::default());
        stop_observing_launches();

        observe_launches(first);
        let replaced = observe_launches(second);
        assert!(replaced.is_some());
        stop_observing_launches();
    }
}
