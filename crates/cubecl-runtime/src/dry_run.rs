//! Running a workload for the compilation and tuning it provokes, without
//! running the workload itself.
//!
//! Under a [`DryRun`]'s [pass](DryRun::pass) every launch is dropped instead
//! of reaching the device, and its kernel is still compiled — at once under a
//! `Profile` pass, queued to compile with the others under a `Compile` one. A
//! warm-up then pays for compilation and tuning without also paying for the
//! work that provoked them, which is what makes producing a shippable
//! environment affordable.
//!
//! A [`DryRun`] is the one place a caller reaches all of this through: it
//! opens its passes, and [observes](DryRun::observe) the kernels and tunes
//! they provoke, counted to it under its [`DryRunId`] by the code doing the
//! work, on every device it reaches.
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
//! [`CompileOnly`] does on one thread what a `Compile` dry run does on all of
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
use cubecl_environment::sync::{Arc, AtomicUsize, Mutex, Ordering};

mod observation;

use observation::Observed;
pub use observation::{Counter, DryRunCounter, DryRunObservation, Progress};

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
    CompileOnly,
}

impl LaunchMode {
    /// Whether the launch should be dropped rather than run.
    pub fn is_skipped(self) -> bool {
        matches!(self, LaunchMode::Skip | LaunchMode::CompileOnly)
    }
}

/// What to do with a launch issued on this thread, right now: what the
/// innermost [`RealRun`] or [`CompileOnly`] open on it says, and otherwise
/// what the open [`DryRun`], if any, does with a launch.
pub fn launch_mode() -> LaunchMode {
    if let Some(mode) = scope::mode() {
        return mode;
    }

    match dry_run_scope() {
        Some(DryRunScope::Compile) => LaunchMode::CompileOnly,
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

/// The pass open in this process: its scope's level in the low
/// [`LEVEL_BITS`], and above them how many [`DryRunPass`] guards hold it
/// open. Written only under [`ACTIVE`]'s lock, and read without it on every
/// launch.
static OPEN: AtomicUsize = AtomicUsize::new(0);
/// The bits of [`OPEN`] that hold the scope's level.
const LEVEL_BITS: u32 = 2;
/// Masks [`OPEN`] down to the scope's level.
const LEVEL_MASK: usize = (1 << LEVEL_BITS) - 1;
/// One guard, in [`OPEN`]'s count.
const GUARD: usize = 1 << LEVEL_BITS;

/// The dry run whose pass is open, while one is: what the work it provokes is
/// counted to.
static ACTIVE: Mutex<Option<Arc<Observed>>> = Mutex::new(None);

/// The id the next [`DryRun`] takes.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Whether a dry run's pass is open, in either scope.
pub fn dry_run() -> bool {
    dry_run_scope().is_some()
}

/// The scope of the pass open in this process, if one is.
pub fn dry_run_scope() -> Option<DryRunScope> {
    DryRunScope::at(OPEN.load(Ordering::Relaxed) & LEVEL_MASK)
}

/// Where the work the open pass provokes is counted, or `None` outside one.
///
/// For the code doing that work — the kernel loader queuing and compiling,
/// autotune gathering and measuring — not for a caller, which reads its own
/// [`DryRun::observe`].
pub fn counted() -> Option<DryRunCounter> {
    if !dry_run() {
        return None;
    }
    ACTIVE.lock().as_ref().map(|observed| DryRunCounter {
        observed: observed.clone(),
    })
}

/// Identifies one [`DryRun`] in the process: what its work is counted under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DryRunId(pub usize);

/// One dry run: a workload run for the compilation and tuning it provokes,
/// over as many passes, and on as many devices, as it takes.
///
/// It drops nothing on its own. Each [`pass`](Self::pass) opens a scope —
/// gather and compile, or profile — for as long as its guard lives, and the
/// work the launches under it provoke is counted to this dry run, which
/// [`observe`](Self::observe) reads. A build that compiles everything and
/// then tunes with it compiled is one dry run with two passes:
///
/// ```no_run
/// # fn warm_up() {}
/// use cubecl_runtime::dry_run::{DryRun, DryRunScope};
///
/// let dry_run = DryRun::new();
///
/// // Gather every kernel the warm-up reaches, and compile them together...
/// let compile = dry_run.pass(DryRunScope::Compile);
/// warm_up();
/// drop(compile);
///
/// // ...then tune with them compiled: this pass only measures.
/// let _profile = dry_run.pass(DryRunScope::Profile);
/// warm_up();
///
/// let observed = dry_run.observe();
/// assert_eq!(observed.kernels.pending(), 0);
/// ```
///
/// There is deliberately no configuration file or environment variable for
/// it. A pass left open by accident turns the rest of the process into
/// launches that quietly do nothing and read back uninitialized memory, so
/// its lifetime belongs to a scope in the code that wants it, not to an
/// ambient default nothing in the process can see.
///
/// A clone is the same dry run — its passes, its id, its counts — which is
/// how a reader on another thread observes one while it runs.
#[derive(Debug, Clone)]
pub struct DryRun {
    observed: Arc<Observed>,
}

impl DryRun {
    /// A dry run with no pass open yet, under an id of its own.
    #[allow(clippy::new_without_default, reason = "each one takes a new id")]
    pub fn new() -> Self {
        let id = DryRunId(NEXT_ID.fetch_add(1, Ordering::Relaxed));
        Self {
            observed: Arc::new(Observed::new(id)),
        }
    }

    /// Its id.
    pub fn id(&self) -> DryRunId {
        self.observed.id
    }

    /// Make every launch a dry run of `scope` for as long as the guard
    /// lives, on every thread and every device, counting what they provoke
    /// to this dry run.
    ///
    /// Overlapping passes of one scope compose, so a pass opened while
    /// another is still open leaves the scope open until the last of them
    /// drops. One scope is open at a time — a pass is process-wide, and
    /// inside a `Compile` one there is nothing measured for a `Profile` one
    /// to read — and one dry run: the work under an open pass counts to one
    /// dry run.
    ///
    /// The open scope is read on the thread issuing a launch, with relaxed
    /// ordering, so a launch another thread had already begun issuing may
    /// still execute. What is guaranteed is the launches issued by the
    /// thread that opened the pass, and every launch issued after other
    /// threads observe it.
    ///
    /// # Panics
    ///
    /// If a pass of the other scope, or of another dry run, is open.
    pub fn pass(&self, scope: DryRunScope) -> DryRunPass {
        let level = scope as usize;
        let mut active = ACTIVE.lock();
        let open = OPEN.load(Ordering::Relaxed);
        match (open & LEVEL_MASK, active.as_ref()) {
            (0, _) => {
                *active = Some(self.observed.clone());
                OPEN.store(GUARD | level, Ordering::Relaxed);
            }
            (open_level, Some(owner))
                if open_level == level && Arc::ptr_eq(owner, &self.observed) =>
            {
                OPEN.store(open + GUARD, Ordering::Relaxed);
            }
            (open_level, owner) => {
                let open_scope = DryRunScope::at(open_level).expect("open, matched above");
                let another = owner.is_some_and(|owner| !Arc::ptr_eq(owner, &self.observed));
                drop(active);
                match another {
                    true => panic!(
                        "a pass of dry run {:?} cannot open while one of another is",
                        self.id()
                    ),
                    false => {
                        panic!("a {scope:?} pass cannot open while a {open_scope:?} one is")
                    }
                }
            }
        }
        DryRunPass { scope }
    }

    /// What it has provoked so far, over every pass and device.
    pub fn observe(&self) -> DryRunObservation {
        self.observed.observe()
    }
}

/// A [`DryRun`]'s pass while it is open: every launch is a dry run of its
/// scope until it drops.
#[derive(Debug)]
pub struct DryRunPass {
    scope: DryRunScope,
}

impl DryRunPass {
    /// The scope it opened.
    pub fn scope(&self) -> DryRunScope {
        self.scope
    }
}

impl Drop for DryRunPass {
    fn drop(&mut self) {
        // The last guard out closes the pass, its level and its dry run with
        // it.
        let mut active = ACTIVE.lock();
        let open = OPEN.load(Ordering::Relaxed) - GUARD;
        if open < GUARD {
            OPEN.store(0, Ordering::Relaxed);
            *active = None;
        } else {
            OPEN.store(open, Ordering::Relaxed);
        }
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
    /// Held for its drop, which restores the mode it replaced.
    _guard: ModeGuard,
}

impl RealRun {
    /// Opts this thread back into executing until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        Self {
            _guard: ModeGuard::new(LaunchMode::Execute),
        }
    }
}

/// Makes the launches issued on this thread only queue their kernels for
/// compilation, for as long as it lives — see [`LaunchMode::CompileOnly`].
///
/// A nested [`RealRun`] still executes, which is what lets a candidate that
/// dispatches through another tuner have that one measure for real.
///
/// Thread-local, like [`RealRun`], and for the same reason it has to live on
/// the thread issuing the launches.
#[derive(Debug)]
pub struct CompileOnly {
    /// Held for its drop, which restores the mode it replaced.
    _guard: ModeGuard,
}

impl CompileOnly {
    /// Makes this thread's launches queue their kernels until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        Self {
            _guard: ModeGuard::new(LaunchMode::CompileOnly),
        }
    }
}

/// Sets this thread's launch mode for as long as it lives, and restores the
/// one it replaced when it drops: what [`RealRun`] and [`CompileOnly`] are,
/// each with its mode.
#[derive(Debug)]
struct ModeGuard {
    outer: Option<LaunchMode>,
    /// Keeps the guard on the thread whose mode it set: dropped on another,
    /// it would restore that thread's mode and leave its own set forever.
    on_thread: PhantomData<*const ()>,
}

impl ModeGuard {
    fn new(mode: LaunchMode) -> Self {
        Self {
            outer: scope::enter(mode),
            on_thread: PhantomData,
        }
    }
}

impl Drop for ModeGuard {
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

    /// The innermost guard decides: a compile-only guard inside a measurement queues,
    /// and a nested tuner measures inside that.
    #[test]
    #[serial_test::serial]
    fn the_innermost_guard_decides() {
        let _real_run = RealRun::new();
        {
            let _compile_only = CompileOnly::new();
            assert_eq!(launch_mode(), LaunchMode::CompileOnly);
            {
                let _nested = RealRun::new();
                assert_eq!(
                    launch_mode(),
                    LaunchMode::Execute,
                    "a nested tuner measures"
                );
            }
            assert_eq!(launch_mode(), LaunchMode::CompileOnly);
        }
        assert_eq!(launch_mode(), LaunchMode::Execute);
    }

    /// A compile-only dry run queues every launch, and a measurement's
    /// `RealRun` inside it still executes.
    #[test]
    #[serial_test::serial]
    fn a_compile_dry_run_queues_launches() {
        let _compile = DryRun::new().pass(DryRunScope::Compile);
        assert_eq!(dry_run_scope(), Some(DryRunScope::Compile));
        assert_eq!(launch_mode(), LaunchMode::CompileOnly);
        let _real_run = RealRun::new();
        assert_eq!(launch_mode(), LaunchMode::Execute);
    }

    /// One scope is open at a time: a dry run of the other scope refuses to
    /// open beside it.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "a Compile pass cannot open while a Profile one is")]
    fn scopes_do_not_overlap() {
        let dry_run = DryRun::new();
        let _profile = dry_run.pass(DryRunScope::Profile);
        let _compile = dry_run.pass(DryRunScope::Compile);
    }

    /// A refused open leaves the open scope as it was: still open, and closed
    /// by its own guard's drop.
    #[test]
    #[serial_test::serial]
    fn a_refused_open_leaves_the_open_scope() {
        let dry_run = DryRun::new();
        let profile = dry_run.pass(DryRunScope::Profile);
        let refused = std::panic::catch_unwind(|| dry_run.pass(DryRunScope::Compile));
        assert!(refused.is_err());
        assert_eq!(dry_run_scope(), Some(DryRunScope::Profile));
        drop(profile);
        assert_eq!(dry_run_scope(), None);
    }

    /// The last guard of a scope closes it, so the other scope opens next.
    #[test]
    #[serial_test::serial]
    fn a_closed_scope_makes_way_for_the_other() {
        let dry_run = DryRun::new();
        drop(dry_run.pass(DryRunScope::Compile));
        assert_eq!(dry_run_scope(), None);
        let _profile = dry_run.pass(DryRunScope::Profile);
        assert_eq!(launch_mode(), LaunchMode::Skip);
    }

    /// Compiling only needs no dry run: the guard queues on its own.
    #[test]
    #[serial_test::serial]
    fn compile_only_works_outside_a_dry_run() {
        assert!(!dry_run());
        let _compile_only = CompileOnly::new();
        assert_eq!(launch_mode(), LaunchMode::CompileOnly);
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
        let _dry_run = DryRun::new().pass(DryRunScope::Profile);

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
        let build = DryRun::new();
        {
            let _outer = build.pass(DryRunScope::Profile);
            {
                let _inner = build.pass(DryRunScope::Profile);
                assert!(dry_run());
            }
            assert!(dry_run(), "the outer guard is still in force");
        }
        assert!(!dry_run(), "and the process is back to executing");
    }

    /// A pass counts to its own dry run: one dry run's pass refuses to open
    /// beside another's, whatever their scopes.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "cannot open while one of another is")]
    fn two_dry_runs_do_not_overlap() {
        let (first, second) = (DryRun::new(), DryRun::new());
        let _first = first.pass(DryRunScope::Profile);
        let _second = second.pass(DryRunScope::Profile);
    }

    /// What the open pass provokes counts to its dry run, over every pass of
    /// it, and nothing counts outside one; a counter handed out under a pass
    /// keeps counting to its dry run after the pass closes.
    #[test]
    #[serial_test::serial]
    fn work_counts_to_the_dry_run_whose_pass_is_open() {
        assert!(counted().is_none(), "nothing counts outside a pass");
        let build = DryRun::new();

        let compile = build.pass(DryRunScope::Compile);
        let counter = counted().expect("a pass is open");
        assert_eq!(counter.id(), build.id());
        counter.kernels().request(3);
        counter.tunes().request(2);
        drop(compile);
        assert!(counted().is_none());

        // A kernel queued in the compile pass settles in the profile one.
        counter.kernels().settle(3);
        let _profile = build.pass(DryRunScope::Profile);
        counted().expect("a pass is open").tunes().settle(1);

        let observed = build.observe();
        assert_eq!(
            observed.kernels,
            Progress {
                requested: 3,
                settled: 3
            }
        );
        assert_eq!(observed.tunes.pending(), 1);
        assert_eq!(DryRun::new().observe(), DryRunObservation::default());
    }
}
