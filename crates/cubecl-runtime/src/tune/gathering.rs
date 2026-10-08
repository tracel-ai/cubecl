//! Whether this thread runs a key's candidates to gather their kernels: a key reached there is
//! gathered by another key's candidates.

/// Held while a key's candidates run under a `CompileOnly` override.
pub(super) struct InsideCandidates {
    _private: (),
}

#[cfg(feature = "std")]
std::thread_local! {
    static DEPTH: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

// No threads to be local to: one depth for the process.
#[cfg(not(feature = "std"))]
static DEPTH: cubecl_environment::sync::AtomicUsize = cubecl_environment::sync::AtomicUsize::new(0);

impl InsideCandidates {
    pub(super) fn enter() -> Self {
        enter_candidates();
        Self { _private: () }
    }
}

impl Drop for InsideCandidates {
    fn drop(&mut self) {
        exit_candidates();
    }
}

/// Whether another key's candidates are running on this thread.
pub(super) fn inside_candidates() -> bool {
    depth() > 0
}

#[cfg(feature = "std")]
fn depth() -> usize {
    DEPTH.with(core::cell::Cell::get)
}

#[cfg(feature = "std")]
fn enter_candidates() {
    DEPTH.with(|depth| depth.set(depth.get() + 1));
}

#[cfg(feature = "std")]
fn exit_candidates() {
    DEPTH.with(|depth| depth.set(depth.get() - 1));
}

#[cfg(not(feature = "std"))]
fn depth() -> usize {
    DEPTH.load(cubecl_environment::sync::Ordering::Relaxed)
}

#[cfg(not(feature = "std"))]
fn enter_candidates() {
    DEPTH.fetch_add(1, cubecl_environment::sync::Ordering::Relaxed);
}

#[cfg(not(feature = "std"))]
fn exit_candidates() {
    DEPTH.fetch_sub(1, cubecl_environment::sync::Ordering::Relaxed);
}
