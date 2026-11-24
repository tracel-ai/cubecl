//! What a launch carries from the thread that issues it to the thread that
//! runs it.
//!
//! A launch is decided on the thread that issues it and runs on the device's,
//! later: an override closing in between must not take the launch's kernels
//! out of its count, nor hand them to the next override's.

use super::{LaunchPurpose, StatisticsRecorder, with_value};

/// One launch as it was issued: what it was issued for, and where the work it
/// triggers counts, the collector of the override open where it was issued.
///
/// Public for the tests of the kernel loader in `cubecl-server`, which stand
/// in for a launch's submit with it.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct IssuedLaunch {
    purpose: LaunchPurpose,
    recorder: Option<StatisticsRecorder>,
}

impl IssuedLaunch {
    /// A launch issued now for `purpose`: the work it triggers counts to the
    /// override open now, wherever and whenever it runs.
    pub fn new(purpose: LaunchPurpose) -> Self {
        Self {
            purpose,
            recorder: StatisticsRecorder::active(),
        }
    }

    /// Runs `launch` on this thread as this launch: the kernels it loads are
    /// loaded for its purpose, and the work it registers counts where it was
    /// issued.
    pub fn run<R>(&self, launch: impl FnOnce() -> R) -> R {
        with_value(running::replace, self.clone(), launch)
    }
}

impl IssuedLaunch {
    /// The purpose of the launch running on this thread, when one is.
    pub(crate) fn running_purpose() -> Option<LaunchPurpose> {
        running::purpose()
    }

    /// Where the work of the launch running on this thread counts, when a
    /// launch is running: `Some(None)` for one issued with no override open.
    pub(crate) fn running_recorder() -> Option<Option<StatisticsRecorder>> {
        running::recorder()
    }
}

/// The launch running on this thread.
#[cfg(feature = "std")]
mod running {
    use super::{IssuedLaunch, LaunchPurpose, StatisticsRecorder};
    use core::cell::RefCell;

    std::thread_local! {
        static RUNNING: RefCell<Option<IssuedLaunch>> = const { RefCell::new(None) };
    }

    pub fn replace(launch: Option<IssuedLaunch>) -> Option<IssuedLaunch> {
        RUNNING.with(|running| running.replace(launch))
    }

    pub fn purpose() -> Option<LaunchPurpose> {
        RUNNING.with(|running| running.borrow().as_ref().map(|launch| launch.purpose))
    }

    pub fn recorder() -> Option<Option<StatisticsRecorder>> {
        RUNNING.with(|running| {
            running
                .borrow()
                .as_ref()
                .map(|launch| launch.recorder.clone())
        })
    }
}

/// The launch running: with no threads to be local to, one for the process.
#[cfg(not(feature = "std"))]
mod running {
    use super::{IssuedLaunch, LaunchPurpose, StatisticsRecorder};
    use cubecl_environment::sync::Mutex;

    static RUNNING: Mutex<Option<IssuedLaunch>> = Mutex::new(None);

    pub fn replace(launch: Option<IssuedLaunch>) -> Option<IssuedLaunch> {
        core::mem::replace(&mut *RUNNING.lock(), launch)
    }

    pub fn purpose() -> Option<LaunchPurpose> {
        RUNNING.lock().as_ref().map(|launch| launch.purpose)
    }

    pub fn recorder() -> Option<Option<StatisticsRecorder>> {
        RUNNING
            .lock()
            .as_ref()
            .map(|launch| launch.recorder.clone())
    }
}

#[cfg(all(test, feature = "std"))]
mod small_tests {
    use super::*;

    /// The purpose travels with the launch: a race's launch loads as the
    /// race's on a thread no race runs on.
    #[test]
    fn a_launch_loads_under_the_purpose_it_was_issued_for() {
        let issued = IssuedLaunch::new(LaunchPurpose::Race);
        assert_eq!(issued.run(LaunchPurpose::running), LaunchPurpose::Race);
    }

    #[test]
    fn a_launch_issued_under_no_override_counts_nowhere() {
        let issued = IssuedLaunch {
            purpose: LaunchPurpose::Work,
            recorder: None,
        };
        assert!(issued.run(StatisticsRecorder::current).is_none());
    }
}

#[cfg(test)]
mod tests;
