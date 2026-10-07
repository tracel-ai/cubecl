//! Running a workload for the compilation and tuning it provokes, without
//! running the workload itself.
//!
//! Under a [`DryRun`] every launch is still expanded, compiled, validated and
//! cached, and is then dropped instead of reaching the device. A warm-up pass
//! then pays for compilation and tuning without also paying for the work that
//! provoked them, which is what makes producing a shippable environment
//! affordable.
//!
//! Under a [`DryRunScope::Profile`] dry run, the launches autotune issues are
//! the exception: they *are* the measurement, so [`RealRun`] opts them back
//! into executing. A [`DryRunScope::Compile`] dry run measures nothing: every
//! launch, autotune's included, only queues its kernel, so a pass under it
//! gathers every kernel the workload and its tuning reach. It compiles none of
//! them: the first launch of a later `Profile` pass, or the first tune before
//! it measures anything, compiles the whole queue in one batch, and that
//! pass's tunes only measure.
//!
//! [`Precompile`] does on one thread what a `Compile` dry run does on all of
//! them, with or without a dry run: the launches it covers only queue their
//! kernels, and the server compiles the whole queue at once, on its compiling
//! threads, when it next loads a kernel for a launch.
//!
//! **Buffers are left as they were**, so anything read back during a dry run is
//! meaningless. It only suits a pass driven by the *shapes* it produces, which
//! is what keys the caches, and never one that branches on a computed value.
//!
//! The decision is made here, once, on the thread that issues the launch.
//! Servers receive the verdict as a [`LaunchMode`] argument rather than
//! deriving it: by the time a launch reaches a server thread, the context that
//! produced it is gone.

use core::marker::PhantomData;
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
/// what the open [`DryRun`], if any, does with a launch.
pub fn launch_mode() -> LaunchMode {
    if let Some(mode) = scope::mode() {
        return mode;
    }

    match dry_run_scope() {
        Some(DryRunScope::Compile) => LaunchMode::Precompile,
        Some(DryRunScope::Profile) => LaunchMode::Skip,
        None => LaunchMode::Execute,
    }
}

/// What a [`DryRun`] does with the work it drops.
///
/// Each scope is a level, the number the process's open dry run holds in its
/// low bits while a dry run of it is open; zero is none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum DryRunScope {
    /// Gather kernels to compile: every launch queues its kernel, and a tune
    /// queues its candidates' kernels without measuring them or deciding
    /// anything. Nothing compiles under it; the queue compiles, in one batch,
    /// when a later launch loads a kernel — the first of a
    /// [`Profile`](Self::Profile) pass, which then tunes every key it reaches
    /// with its kernels compiled.
    Compile = 1,
    /// Compile every kernel the workload launches, and tune what it tunes by
    /// measuring the candidates for real.
    Profile = 2,
}

impl DryRunScope {
    /// The scope at `level`, or `None` at level zero, outside any dry run.
    fn at(level: usize) -> Option<Self> {
        match level {
            1 => Some(Self::Compile),
            2 => Some(Self::Profile),
            _ => None,
        }
    }
}

/// The dry run open in this process: its scope's level in the low
/// [`LEVEL_BITS`], and above them how many guards hold it open.
///
/// A count, so overlapping guards compose: a swap-and-restore would let one
/// thread's guard end a dry run another thread is still inside, and leave the
/// process dry-running forever once that one dropped in turn. One word, so the
/// scope and its count change together, and a second scope cannot open beside
/// the first.
static OPEN: AtomicUsize = AtomicUsize::new(0);
/// The bits of [`OPEN`] that hold the scope's level.
const LEVEL_BITS: u32 = 2;
/// Masks [`OPEN`] down to the scope's level.
const LEVEL_MASK: usize = (1 << LEVEL_BITS) - 1;
/// One guard, in [`OPEN`]'s count.
const GUARD: usize = 1 << LEVEL_BITS;

/// Whether a dry run of either scope is open.
pub fn dry_run() -> bool {
    dry_run_scope().is_some()
}

/// The scope of the dry run open in this process, if one is.
pub fn dry_run_scope() -> Option<DryRunScope> {
    DryRunScope::at(OPEN.load(Ordering::Relaxed) & LEVEL_MASK)
}

/// Makes every launch a dry run for as long as it lives, on every thread and
/// every device.
///
/// Overlapping guards of one scope compose, so a pass that opens one while
/// another is still open leaves the mode on until the last of them drops. One
/// scope is open at a time: a dry run is process-wide, and inside a `Compile`
/// one there is nothing measured for a `Profile` one to read.
///
/// The open scope is read on the thread issuing a launch, with relaxed ordering, so a
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
/// use cubecl_runtime::dry_run::{DryRun, DryRunScope};
///
/// // Gather every kernel the warm-up reaches, and compile them together...
/// let compile = DryRun::new(DryRunScope::Compile);
/// warm_up();
/// drop(compile);
///
/// // ...then tune with them compiled: this pass only measures.
/// let _profile = DryRun::new(DryRunScope::Profile);
/// warm_up();
/// ```
#[derive(Debug)]
pub struct DryRun {
    scope: DryRunScope,
}

impl DryRun {
    /// Opens a dry run of `scope` until the guard drops.
    ///
    /// # Panics
    ///
    /// If a dry run of the other scope is open.
    pub fn new(scope: DryRunScope) -> Self {
        let level = scope as usize;
        #[allow(
            deprecated,
            reason = "portable_atomic lacks try_update on targets without native atomics"
        )]
        let opened = OPEN.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |open| {
            match open & LEVEL_MASK {
                0 => Some(GUARD | level),
                open_level if open_level == level => Some(open + GUARD),
                _ => None,
            }
        });
        if let Err(open) = opened {
            panic!(
                "a {scope:?} dry run cannot open while a {:?} one is",
                DryRunScope::at(open & LEVEL_MASK).expect("open, since the update refused")
            );
        }
        Self { scope }
    }

    /// The scope this guard opened.
    pub fn scope(&self) -> DryRunScope {
        self.scope
    }
}

impl Drop for DryRun {
    fn drop(&mut self) {
        // The last guard out closes the dry run, its level with it.
        #[allow(
            deprecated,
            reason = "portable_atomic lacks try_update on targets without native atomics"
        )]
        let _ = OPEN.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |open| {
            let open = open - GUARD;
            Some(if open < GUARD { 0 } else { open })
        });
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
    /// Keeps the guard on the thread whose mode it set: dropped on another,
    /// it would restore that thread's mode and leave its own set forever.
    on_thread: PhantomData<*const ()>,
}

impl RealRun {
    /// Opts this thread back into executing until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        Self {
            outer: scope::enter(LaunchMode::Execute),
            on_thread: PhantomData,
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
/// A nested [`RealRun`] still executes, which is what lets a candidate that
/// dispatches through another tuner have that one measure for real.
///
/// Thread-local, like [`RealRun`], and for the same reason it has to live on
/// the thread issuing the launches.
#[derive(Debug)]
pub struct Precompile {
    outer: Option<LaunchMode>,
    /// Keeps the guard on the thread whose mode it set: dropped on another,
    /// it would restore that thread's mode and leave its own set forever.
    on_thread: PhantomData<*const ()>,
}

impl Precompile {
    /// Makes this thread's launches queue their kernels until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        Self {
            outer: scope::enter(LaunchMode::Precompile),
            on_thread: PhantomData,
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

    /// The innermost guard decides: a precompile inside a measurement queues,
    /// and a nested tuner measures inside that.
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

    /// A compile-only dry run queues every launch, and a measurement's
    /// `RealRun` inside it still executes.
    #[test]
    #[serial_test::serial]
    fn a_compile_dry_run_queues_launches() {
        let _compile = DryRun::new(DryRunScope::Compile);
        assert_eq!(dry_run_scope(), Some(DryRunScope::Compile));
        assert_eq!(launch_mode(), LaunchMode::Precompile);
        let _real_run = RealRun::new();
        assert_eq!(launch_mode(), LaunchMode::Execute);
    }

    /// One scope is open at a time: a dry run of the other scope refuses to
    /// open beside it.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "a Compile dry run cannot open while a Profile one is")]
    fn scopes_do_not_overlap() {
        let _profile = DryRun::new(DryRunScope::Profile);
        let _compile = DryRun::new(DryRunScope::Compile);
    }

    /// A refused open leaves the open scope as it was: still open, and closed
    /// by its own guard's drop.
    #[test]
    #[serial_test::serial]
    fn a_refused_open_leaves_the_open_scope() {
        let profile = DryRun::new(DryRunScope::Profile);
        let refused = std::panic::catch_unwind(|| DryRun::new(DryRunScope::Compile));
        assert!(refused.is_err());
        assert_eq!(dry_run_scope(), Some(DryRunScope::Profile));
        drop(profile);
        assert_eq!(dry_run_scope(), None);
    }

    /// The last guard of a scope closes it, so the other scope opens next.
    #[test]
    #[serial_test::serial]
    fn a_closed_scope_makes_way_for_the_other() {
        drop(DryRun::new(DryRunScope::Compile));
        assert_eq!(dry_run_scope(), None);
        let _profile = DryRun::new(DryRunScope::Profile);
        assert_eq!(launch_mode(), LaunchMode::Skip);
    }

    /// Precompiling needs no dry run: the guard queues on its own.
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
        let _dry_run = DryRun::new(DryRunScope::Profile);

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
            let _outer = DryRun::new(DryRunScope::Profile);
            {
                let _inner = DryRun::new(DryRunScope::Profile);
                assert!(dry_run());
            }
            assert!(dry_run(), "the outer guard is still in force");
        }
        assert!(!dry_run(), "and the process is back to executing");
    }
}
