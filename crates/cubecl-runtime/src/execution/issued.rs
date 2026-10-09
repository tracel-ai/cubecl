//! Where the work a launch triggers counts: the collector of the override
//! open where the launch was issued, carried to the thread that runs it.
//!
//! A launch is decided on the thread that issues it and runs on the device's,
//! later: an override closing in between must not take the launch's kernels
//! out of its count, nor hand them to the next override's.

use super::{ProcessModeOverride, StatisticsRecorder};

/// The recorder in force where a launch was issued.
#[derive(Debug, Clone)]
pub(crate) struct IssuedRecorder {
    recorder: Option<StatisticsRecorder>,
}

impl IssuedRecorder {
    /// The recorder of the override open now, for a launch issued now.
    pub(crate) fn new() -> Self {
        Self {
            recorder: ProcessModeOverride::active_recorder(),
        }
    }

    /// Run `launch`, the work it registers counted where it was issued.
    pub(crate) fn apply<R>(&self, launch: impl FnOnce() -> R) -> R {
        let previous = running::replace(Some(self.recorder.clone()));
        let restore = Restore { previous };
        let launched = launch();
        core::mem::drop(restore);
        launched
    }

    /// Where the work of the launch running on this thread counts, when a
    /// launch is running: `Some(None)` for one issued with no override open.
    pub(crate) fn running() -> Option<Option<StatisticsRecorder>> {
        running::get()
    }
}

/// Puts back the launch that was running, however the one applied ends.
struct Restore {
    previous: Option<Option<StatisticsRecorder>>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        running::replace(self.previous.take());
    }
}

/// The recorder of the launch running on this thread.
#[cfg(feature = "std")]
mod running {
    use super::StatisticsRecorder;
    use core::cell::RefCell;

    std::thread_local! {
        static RUNNING: RefCell<Option<Option<StatisticsRecorder>>> = const { RefCell::new(None) };
    }

    pub(super) fn replace(
        recorder: Option<Option<StatisticsRecorder>>,
    ) -> Option<Option<StatisticsRecorder>> {
        RUNNING.with(|running| running.replace(recorder))
    }

    pub(super) fn get() -> Option<Option<StatisticsRecorder>> {
        RUNNING.with(|running| running.borrow().clone())
    }
}

/// The recorder of the launch running: with no threads to be local to, one
/// for the process.
#[cfg(not(feature = "std"))]
mod running {
    use super::StatisticsRecorder;
    use cubecl_environment::sync::Mutex;

    static RUNNING: Mutex<Option<Option<StatisticsRecorder>>> = Mutex::new(None);

    pub(super) fn replace(
        recorder: Option<Option<StatisticsRecorder>>,
    ) -> Option<Option<StatisticsRecorder>> {
        core::mem::replace(&mut *RUNNING.lock(), recorder)
    }

    pub(super) fn get() -> Option<Option<StatisticsRecorder>> {
        RUNNING.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        KernelOutcome, KernelRegistration, ProcessMode, ProcessModeOverride, StatisticsCollector,
    };
    use super::*;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;

    /// A launch issued under one override counts there when it runs after
    /// that override closed — even while another collector's is open — and
    /// one issued under none counts nowhere.
    #[test]
    #[serial_test::serial]
    fn a_launch_counts_where_it_was_issued() {
        let issuing = StatisticsCollector::new();
        let gathering = ProcessModeOverride::new(ProcessMode::CompileOnly, &issuing);
        let issued = IssuedRecorder::new();
        core::mem::drop(gathering);

        let later = StatisticsCollector::new();
        let tuning = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &later);
        let registered = issued.apply(KernelRegistration::register);
        let outside = IssuedRecorder { recorder: None }.apply(KernelRegistration::register);
        core::mem::drop(tuning);

        let settled = registered
            .expect("issued under an override")
            .settle(KernelOutcome::Compiled);
        assert!(outside.is_none(), "issued under none, it counts nowhere");
        assert_eq!(issuing.statistics().compilation.compiled, 1);
        assert_eq!(later.statistics().compilation.registered, 0);
        core::mem::drop(settled);
    }
}
