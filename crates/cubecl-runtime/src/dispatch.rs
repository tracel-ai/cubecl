//! Whether a launch reaches the device, or is only compiled.
//!
//! Normally every launch does both. Under
//! [`compile_only`](crate::config::CompilationConfig::compile_only) only the
//! launches that *are* an autotune measurement still run; everything else is
//! compiled, cached, and dropped. A warm-up pass then pays for compilation and
//! tuning without also running the workload that provoked them.
//!
//! The decision is made here, once, on the thread that issues the launch —
//! which is the thread [`tune_benchmark`](crate::tune) measures on, so
//! [`Measuring`] is a plain thread-local depth. Servers receive the verdict as
//! a [`Dispatch`] argument rather than deriving it: by the time a launch
//! reaches a server thread, the context that produced it is gone.

use core::cell::Cell;
use cubecl_environment::sync::{AtomicBool, Lazy, Ordering};

/// What a server should do with a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// Compile if needed, then run it. The normal case.
    Execute,
    /// Compile if needed, cache the artifact, and drop the launch.
    ///
    /// A server honoring this must still do everything a first launch does
    /// short of dispatching — expand, compile, validate, populate its caches —
    /// or the pass buys nothing. Buffers are left as they were, which is why
    /// this is only ever correct for a pass driving shapes rather than values.
    CompileOnly,
}

impl Dispatch {
    /// Whether the launch should be dropped rather than run.
    pub fn is_compile_only(self) -> bool {
        matches!(self, Dispatch::CompileOnly)
    }
}

#[cfg(feature = "std")]
mod measuring {
    use super::Cell;

    std::thread_local! {
        /// How many autotune measurements are open on this thread. A depth
        /// rather than a flag: a tunable may itself dispatch through another
        /// tuner, and the inner one finishing must not un-mark the outer.
        static MEASURING: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn measuring() -> bool {
        MEASURING.with(|depth| depth.get() > 0)
    }

    pub(super) fn enter() {
        MEASURING.with(|depth| depth.set(depth.get() + 1));
    }

    pub(super) fn exit() {
        MEASURING.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(not(feature = "std"))]
mod measuring {
    // No threads to be local to, and no configuration file to turn
    // `compile_only` on with either — so this is dead weight that keeps the
    // call sites uniform.
    pub(super) fn measuring() -> bool {
        false
    }
    pub(super) fn enter() {}
    pub(super) fn exit() {}
}

/// Marks the current thread as running an autotune measurement for as long as
/// it lives, so the launches it issues still reach the device under
/// [`compile_only`](crate::config::CompilationConfig::compile_only).
///
/// Held across warm-up and samples alike: a candidate that was never warmed is
/// a candidate measured on its first, slowest run.
#[derive(Debug)]
pub struct Measuring {
    _private: (),
}

impl Measuring {
    /// Enters a measurement on this thread.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        measuring::enter();
        Self { _private: () }
    }
}

impl Drop for Measuring {
    fn drop(&mut self) {
        measuring::exit();
    }
}

/// What to do with a launch issued on this thread, right now.
pub fn dispatch() -> Dispatch {
    if !compile_only() || measuring::measuring() {
        return Dispatch::Execute;
    }

    Dispatch::CompileOnly
}

/// The mode, seeded from
/// [`compilation.compile_only`](crate::config::CompilationConfig::compile_only)
/// the first time it is read and switchable afterwards.
///
/// Seeded lazily rather than at configuration load because it must also be
/// switchable *at runtime*: an application warms one model's environment with
/// the mode on and then goes back to real work with it off, in one process. A
/// file-only knob could not express that.
static COMPILE_ONLY: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(configured()));

/// Whether launches are compiled and dropped rather than run.
pub fn compile_only() -> bool {
    COMPILE_ONLY.load(Ordering::Relaxed)
}

/// Turns compile-only mode on or off for the whole process, returning what it
/// was.
///
/// Process-wide and immediate, like an environment switch — every launch
/// issued after it lands is affected, on every thread and every device. Prefer
/// [`CompileOnly`] to setting it directly: a pass that panics with the mode
/// left on turns every later launch in the process into a no-op, and the
/// symptom is silently wrong results rather than an error.
pub fn set_compile_only(enabled: bool) -> bool {
    COMPILE_ONLY.swap(enabled, Ordering::Relaxed)
}

/// Turns [`compile_only`] on for as long as it lives, then restores what it
/// found.
///
/// The safe way to run a warm-up pass: the mode comes back off however the
/// pass ends.
#[derive(Debug)]
pub struct CompileOnly {
    previous: bool,
}

impl CompileOnly {
    /// Turns the mode on until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        Self {
            previous: set_compile_only(true),
        }
    }
}

impl Drop for CompileOnly {
    fn drop(&mut self) {
        set_compile_only(self.previous);
    }
}

/// What the configuration asks for, read once to seed [`COMPILE_ONLY`].
fn configured() -> bool {
    #[cfg(feature = "std")]
    {
        use crate::config::RuntimeConfig;
        crate::config::CubeClRuntimeConfig::get()
            .compilation
            .compile_only
    }

    #[cfg(not(feature = "std"))]
    false
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
    fn measuring_nests() {
        assert!(!measuring::measuring());
        let outer = Measuring::new();
        {
            let _inner = Measuring::new();
            assert!(measuring::measuring());
        }
        assert!(
            measuring::measuring(),
            "the outer measurement is still open"
        );
        drop(outer);
        assert!(!measuring::measuring());
    }

    /// Nothing is skipped unless compile-only is on, whatever the depth.
    #[test]
    #[serial_test::serial]
    fn dispatch_defaults_to_executing() {
        assert_eq!(dispatch(), Dispatch::Execute);
        let _measuring = Measuring::new();
        assert_eq!(dispatch(), Dispatch::Execute);
    }

    /// The whole contract in one place: under compile-only every launch is
    /// dropped *except* the ones a measurement issues, which are the tuning the
    /// mode exists to keep.
    #[test]
    #[serial_test::serial]
    fn compile_only_spares_the_measurements() {
        let _compile_only = CompileOnly::new();

        assert_eq!(dispatch(), Dispatch::CompileOnly);
        {
            let _measuring = Measuring::new();
            assert_eq!(dispatch(), Dispatch::Execute, "a measurement still runs");
        }
        assert_eq!(dispatch(), Dispatch::CompileOnly);
    }

    /// The guard restores what it found, so a nested pass cannot leave the
    /// process compiling-only forever.
    #[test]
    #[serial_test::serial]
    fn the_guard_restores_the_previous_mode() {
        assert!(!compile_only());
        {
            let _outer = CompileOnly::new();
            {
                let _inner = CompileOnly::new();
                assert!(compile_only());
            }
            assert!(compile_only(), "the outer guard is still in force");
        }
        assert!(!compile_only(), "and the process is back to executing");
    }
}
