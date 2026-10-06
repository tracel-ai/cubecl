//! Running a workload for the compilation and tuning it provokes, without
//! running the workload itself.
//!
//! Under a [`DryRun`] every launch is still expanded, compiled, validated and
//! cached, and is then dropped instead of reaching the device. A warm-up pass
//! then pays for compilation and tuning without also paying for the work that
//! provoked them, which is what makes producing a shippable environment
//! affordable.
//!
//! The launches autotune issues are the exception: they *are* the measurement,
//! so [`RealRun`] opts them back into executing.
//!
//! [`Precompile`] is the other way round, and works with or without a dry run:
//! the launches it covers only queue their kernels, and the server compiles
//! the whole queue at once, on its compiling threads, when it next loads a
//! kernel for a launch. Autotune opens one over the batch of candidates it is
//! about to measure.
//!
//! **Buffers are left as they were**, so anything read back during a dry run is
//! meaningless. It only suits a pass driven by the *shapes* it produces, which
//! is what keys the caches, and never one that branches on a computed value.
//!
//! The decision is made here, once, on the thread that issues the launch.
//! Servers receive the verdict as a [`LaunchMode`] argument rather than
//! deriving it: by the time a launch reaches a server thread, the context that
//! produced it is gone.

use cubecl_environment::sync::{AtomicUsize, Ordering};

/// What a server should do with a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchMode {
    /// Compile if needed, then run it. The normal case.
    Execute,
    /// Compile if needed, cache the artifact, and drop the launch.
    ///
    /// A server honoring this must still do everything a first launch does
    /// short of dispatching — expand, compile, validate, populate its caches —
    /// or the pass buys nothing.
    Skip,
    /// Queue the kernel to be compiled with others, and drop the launch.
    ///
    /// A server honoring this compiles the queue when it next loads a kernel
    /// for a launch, and only then: flushing or syncing compiles nothing, so
    /// a pass that only queues gathers everything it reaches into one batch.
    /// A kernel that fails to compile there reports it when it is launched.
    Precompile,
}

impl LaunchMode {
    /// Whether the launch should be dropped rather than run.
    pub fn is_skipped(self) -> bool {
        matches!(self, LaunchMode::Skip | LaunchMode::Precompile)
    }
}

/// What to do with a launch issued on this thread, right now: what the
/// innermost [`RealRun`] or [`Precompile`] open on it says, and otherwise
/// whether a [`DryRun`] is open.
pub fn launch_mode() -> LaunchMode {
    if let Some(mode) = scope::mode() {
        return mode;
    }

    match dry_run() {
        true => LaunchMode::Skip,
        false => LaunchMode::Execute,
    }
}

/// How many dry runs are open in this process.
///
/// A depth rather than a flag so overlapping guards compose: a swap-and-restore
/// would let one thread's guard end a dry run another thread is still inside,
/// and leave the process dry-running forever once that one dropped in turn.
static DRY_RUN: AtomicUsize = AtomicUsize::new(0);

/// Whether launches are currently compiled and dropped rather than run.
pub fn dry_run() -> bool {
    DRY_RUN.load(Ordering::Relaxed) > 0
}

/// Makes every launch a dry run for as long as it lives, on every thread and
/// every device.
///
/// Overlapping guards compose, so a pass that opens one while another is still
/// open leaves the mode on until the last of them drops.
///
/// The flag is read on the thread issuing a launch, with relaxed ordering, so a
/// launch another thread had already begun issuing may still execute. What is
/// guaranteed is the launches issued by the thread that opened the guard, and
/// every launch issued after other threads observe it.
///
/// This is the only way in: there is deliberately no configuration file or
/// environment variable for it. A dry run left on by accident turns the rest of
/// the process into launches that quietly do nothing and read back
/// uninitialized memory, so its lifetime belongs to a scope in the code that
/// wants it, not to an ambient default nothing in the process can see.
///
/// ```no_run
/// # fn warm_up() {}
/// let _dry_run = cubecl_runtime::dry_run::DryRun::new();
/// warm_up();
/// ```
#[derive(Debug)]
pub struct DryRun {
    _private: (),
}

impl DryRun {
    /// Opens a dry run until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        DRY_RUN.fetch_add(1, Ordering::Relaxed);
        Self { _private: () }
    }
}

impl Drop for DryRun {
    fn drop(&mut self) {
        DRY_RUN.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Makes the launches issued on this thread execute for real even inside a
/// [`DryRun`], for as long as it lives.
///
/// Autotune holds one: its launches are the measurement a dry run exists to
/// provoke, not the workload it exists to skip. Held across warm-up and samples
/// alike, since a candidate that was never warmed is a candidate measured on
/// its first, slowest run.
///
/// Thread-local, and the thread that matters is the one issuing the launches,
/// which is not always the one that asked for them: a task handed to
/// [`Client::exclusive`](crate::client::Client::exclusive) runs on
/// the device thread. The guard has to live inside that task, alongside the
/// launches it covers, not around the call that submits it.
#[derive(Debug)]
pub struct RealRun {
    outer: Option<LaunchMode>,
}

impl RealRun {
    /// Opts this thread back into executing until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        Self {
            outer: scope::enter(LaunchMode::Execute),
        }
    }
}

impl Drop for RealRun {
    fn drop(&mut self) {
        scope::exit(self.outer);
    }
}

/// Makes the launches issued on this thread only queue their kernels for
/// compilation, for as long as it lives — see [`LaunchMode::Precompile`].
///
/// Autotune holds one while it runs the batch of candidates it will measure,
/// so that the first of them to execute compiles all of them at once. A nested
/// [`RealRun`] still executes, which is what lets a candidate that dispatches
/// through another tuner have that one measure for real.
///
/// Thread-local, like [`RealRun`], and for the same reason it has to live on
/// the thread issuing the launches.
#[derive(Debug)]
pub struct Precompile {
    outer: Option<LaunchMode>,
}

impl Precompile {
    /// Makes this thread's launches queue their kernels until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        Self {
            outer: scope::enter(LaunchMode::Precompile),
        }
    }
}

impl Drop for Precompile {
    fn drop(&mut self) {
        scope::exit(self.outer);
    }
}

/// The mode the innermost guard open on this thread sets.
///
/// Each guard keeps the mode it replaced and restores it when it drops, so
/// guards nest in either order and the innermost decides — a swap that is
/// safe here, unlike for [`DryRun`], because nothing outside the thread can
/// see it and guards on one thread drop in reverse order.
#[cfg(feature = "std")]
mod scope {
    use super::LaunchMode;
    use core::cell::Cell;

    std::thread_local! {
        static MODE: Cell<Option<LaunchMode>> = const { Cell::new(None) };
    }

    pub(super) fn mode() -> Option<LaunchMode> {
        MODE.with(|mode| mode.get())
    }

    pub(super) fn enter(mode: LaunchMode) -> Option<LaunchMode> {
        MODE.with(|current| current.replace(Some(mode)))
    }

    pub(super) fn exit(outer: Option<LaunchMode>) {
        MODE.with(|current| current.set(outer));
    }
}

#[cfg(not(feature = "std"))]
mod scope {
    // No threads to be local to: no guard changes anything, so every launch
    // follows the dry run. This keeps the call sites uniform.
    use super::LaunchMode;

    pub(super) fn mode() -> Option<LaunchMode> {
        None
    }
    pub(super) fn enter(_mode: LaunchMode) -> Option<LaunchMode> {
        None
    }
    pub(super) fn exit(_outer: Option<LaunchMode>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;

    /// The guard nests: an inner measurement ending must not cancel the outer
    /// one, or a tunable that dispatches through another tuner would have the
    /// rest of its own measurement dropped.
    #[test]
    fn real_run_nests() {
        assert_eq!(scope::mode(), None);
        let outer = RealRun::new();
        {
            let _inner = RealRun::new();
            assert_eq!(scope::mode(), Some(LaunchMode::Execute));
        }
        assert_eq!(
            scope::mode(),
            Some(LaunchMode::Execute),
            "the outer guard is still open"
        );
        drop(outer);
        assert_eq!(scope::mode(), None);
    }

    /// The innermost guard decides: autotune precompiles inside its own
    /// measurement, and a candidate's nested tuner measures inside that.
    #[test]
    #[serial_test::serial]
    fn the_innermost_guard_decides() {
        let _real_run = RealRun::new();
        {
            let _precompile = Precompile::new();
            assert_eq!(launch_mode(), LaunchMode::Precompile);
            {
                let _nested = RealRun::new();
                assert_eq!(
                    launch_mode(),
                    LaunchMode::Execute,
                    "a nested tuner measures"
                );
            }
            assert_eq!(launch_mode(), LaunchMode::Precompile);
        }
        assert_eq!(launch_mode(), LaunchMode::Execute);
    }

    /// Precompiling needs no dry run: autotune at run time compiles its
    /// candidates as much as a build's does.
    #[test]
    #[serial_test::serial]
    fn precompile_works_outside_a_dry_run() {
        assert!(!dry_run());
        let _precompile = Precompile::new();
        assert_eq!(launch_mode(), LaunchMode::Precompile);
        assert!(launch_mode().is_skipped());
    }

    /// Nothing is skipped outside a dry run, whatever the depth.
    #[test]
    #[serial_test::serial]
    fn launches_execute_by_default() {
        assert_eq!(launch_mode(), LaunchMode::Execute);
        let _real_run = RealRun::new();
        assert_eq!(launch_mode(), LaunchMode::Execute);
    }

    /// The whole contract in one place: in a dry run every launch is dropped
    /// *except* the ones a measurement issues, which are the tuning the mode
    /// exists to keep.
    #[test]
    #[serial_test::serial]
    fn a_dry_run_spares_the_measurements() {
        let _dry_run = DryRun::new();

        assert_eq!(launch_mode(), LaunchMode::Skip);
        {
            let _real_run = RealRun::new();
            assert_eq!(launch_mode(), LaunchMode::Execute, "a measurement runs");
        }
        assert_eq!(launch_mode(), LaunchMode::Skip);
    }

    /// Overlapping guards compose, so neither an inner guard ending nor an
    /// outer one can leave the process in the wrong mode. This is what a
    /// swap-and-restore got wrong across threads.
    #[test]
    #[serial_test::serial]
    fn dry_runs_nest() {
        assert!(!dry_run());
        {
            let _outer = DryRun::new();
            {
                let _inner = DryRun::new();
                assert!(dry_run());
            }
            assert!(dry_run(), "the outer guard is still in force");
        }
        assert!(!dry_run(), "and the process is back to executing");
    }
}
