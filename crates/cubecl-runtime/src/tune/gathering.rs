//! Whether this thread runs a key's candidates to gather their kernels: a key
//! reached there is gathered by another key's candidates.

/// Held while a key's candidates run under a `CompileOnly` override, so a
/// key their launches reach knows it was reached inside them.
pub(super) struct CandidateGathering {
    _private: (),
}

impl CandidateGathering {
    /// Gather a key's candidates until the guard drops.
    pub(super) fn enter() -> Self {
        depth::add(1);
        Self { _private: () }
    }

    /// Whether another key's candidates are running on this thread.
    pub(super) fn active() -> bool {
        depth::get() > 0
    }
}

impl Drop for CandidateGathering {
    fn drop(&mut self) {
        depth::sub(1);
    }
}

/// How many gatherings are open on this thread.
#[cfg(feature = "std")]
mod depth {
    std::thread_local! {
        static DEPTH: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    }

    pub(super) fn get() -> usize {
        DEPTH.with(core::cell::Cell::get)
    }

    pub(super) fn add(by: usize) {
        DEPTH.with(|depth| depth.set(depth.get() + by));
    }

    pub(super) fn sub(by: usize) {
        DEPTH.with(|depth| depth.set(depth.get() - by));
    }
}

/// How many gatherings are open: with no threads to be local to, one depth
/// for the process.
#[cfg(not(feature = "std"))]
mod depth {
    use cubecl_environment::sync::{AtomicUsize, Ordering};

    static DEPTH: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn get() -> usize {
        DEPTH.load(Ordering::Relaxed)
    }

    pub(super) fn add(by: usize) {
        DEPTH.fetch_add(by, Ordering::Relaxed);
    }

    pub(super) fn sub(by: usize) {
        DEPTH.fetch_sub(by, Ordering::Relaxed);
    }
}
