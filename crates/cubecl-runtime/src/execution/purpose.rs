//! What a launch is issued for: the work itself, or an autotune race measuring
//! or gathering candidates. Decided where the launch is issued and carried to
//! the thread that loads its kernel, as [`IssuedRecorder`](super::IssuedRecorder)
//! carries where its work counts.
//!
//! A kernel the work launched is one a later run of that work launches again,
//! whatever shape it was launched for; a kernel only a race launched is a
//! candidate the work may never pick. That is what keeps an environment's
//! prune from removing anything the work needs: the server records each
//! kernel the work launches, and nothing a race launches.

/// What a launch is issued for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchPurpose {
    /// The work: a call of an operation, its tuned pick included.
    Work,
    /// An autotune race: a candidate run to be measured, or to gather its
    /// kernels before it is.
    Race,
}

impl LaunchPurpose {
    /// The purpose of a launch issued on this thread now: a race inside
    /// [`racing`](Self::racing), the work otherwise.
    pub(crate) fn issued() -> Self {
        match racing::depth() {
            0 => Self::Work,
            _ => Self::Race,
        }
    }

    /// The purpose of the launch whose kernel this thread is loading. A kernel
    /// loaded outside a launch, or by one that carried no purpose, counts as
    /// the work's: keeping a kernel a race launched costs space, losing one
    /// the work launches costs a compile.
    pub fn running() -> Self {
        running::get().unwrap_or(Self::Work)
    }

    /// Run `candidate`, a race's run of one candidate: the launches it issues
    /// on this thread are issued for the race. The thread is the one that
    /// issues them, the device's own when the race runs there.
    pub fn racing<R>(candidate: impl FnOnce() -> R) -> R {
        racing::enter();
        let leave = Leave;
        let output = candidate();
        core::mem::drop(leave);
        output
    }

    /// Run `launch`, a launch issued for this purpose, on the thread that
    /// loads its kernel: what a launch's submit does on the server's side,
    /// and what a test of a server stands in for it with.
    pub fn apply<R>(self, launch: impl FnOnce() -> R) -> R {
        let previous = running::replace(Some(self));
        let restore = Restore { previous };
        let launched = launch();
        core::mem::drop(restore);
        launched
    }
}

/// Leaves the race [`LaunchPurpose::racing`] entered, however the candidate
/// ends.
struct Leave;

impl Drop for Leave {
    fn drop(&mut self) {
        racing::leave();
    }
}

/// Puts back the purpose of the launch that was running, however the one
/// applied ends.
struct Restore {
    previous: Option<LaunchPurpose>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        running::replace(self.previous.take());
    }
}

/// How many races are running a candidate on this thread.
#[cfg(feature = "std")]
mod racing {
    use core::cell::Cell;

    std::thread_local! {
        static DEPTH: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn depth() -> usize {
        DEPTH.with(Cell::get)
    }

    pub(super) fn enter() {
        DEPTH.with(|depth| depth.set(depth.get() + 1));
    }

    pub(super) fn leave() {
        DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// How many races are running a candidate: with no threads to be local to,
/// one count for the process.
#[cfg(not(feature = "std"))]
mod racing {
    use cubecl_environment::sync::{AtomicUsize, Ordering};

    static DEPTH: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn depth() -> usize {
        DEPTH.load(Ordering::Relaxed)
    }

    pub(super) fn enter() {
        DEPTH.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn leave() {
        DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The purpose of the launch running on this thread.
#[cfg(feature = "std")]
mod running {
    use super::LaunchPurpose;
    use core::cell::Cell;

    std::thread_local! {
        static RUNNING: Cell<Option<LaunchPurpose>> = const { Cell::new(None) };
    }

    pub(super) fn replace(purpose: Option<LaunchPurpose>) -> Option<LaunchPurpose> {
        RUNNING.with(|running| running.replace(purpose))
    }

    pub(super) fn get() -> Option<LaunchPurpose> {
        RUNNING.with(Cell::get)
    }
}

/// The purpose of the launch running: with no threads to be local to, one for
/// the process.
#[cfg(not(feature = "std"))]
mod running {
    use super::LaunchPurpose;
    use cubecl_environment::sync::Mutex;

    static RUNNING: Mutex<Option<LaunchPurpose>> = Mutex::new(None);

    pub(super) fn replace(purpose: Option<LaunchPurpose>) -> Option<LaunchPurpose> {
        core::mem::replace(&mut *RUNNING.lock(), purpose)
    }

    pub(super) fn get() -> Option<LaunchPurpose> {
        *RUNNING.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_launch_is_the_works_unless_a_race_issues_it() {
        assert_eq!(LaunchPurpose::issued(), LaunchPurpose::Work);
        let inside = LaunchPurpose::racing(LaunchPurpose::issued);
        assert_eq!(inside, LaunchPurpose::Race);
        assert_eq!(LaunchPurpose::issued(), LaunchPurpose::Work);
    }

    #[test]
    fn a_kernel_loaded_outside_a_launch_is_the_works() {
        assert_eq!(LaunchPurpose::running(), LaunchPurpose::Work);
    }

    /// The purpose travels with the launch: a race's launch loads as the
    /// race's on a thread no race runs on.
    #[test]
    fn a_launch_loads_under_the_purpose_it_was_issued_for() {
        let issued = LaunchPurpose::racing(LaunchPurpose::issued);
        let loaded = issued.apply(LaunchPurpose::running);
        assert_eq!(loaded, LaunchPurpose::Race);
        assert_eq!(LaunchPurpose::running(), LaunchPurpose::Work);
    }
}
