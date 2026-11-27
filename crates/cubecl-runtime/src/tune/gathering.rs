//! Whether this thread runs a key's candidates to gather their kernels: a key
//! reached there is gathered by another key's candidates.

/// Held while a key's candidates run with their launches discarded, under a
/// `CompileOnly` override or to compile a batch together, so a key their
/// launches reach knows it was reached inside them.
///
/// # Assumptions
///
/// - Gatherers drop in the order opposite to the one they were opened in, on the thread that
///   opened them.
pub struct CandidateGatherer {
    _private: (),
}

impl CandidateGatherer {
    /// Gather a key's candidates on this thread until the gatherer drops.
    pub fn gather() -> Self {
        depth::enter();
        Self { _private: () }
    }

    /// Whether another key's candidates are running on this thread.
    pub fn is_gathering() -> bool {
        depth::get() > 0
    }
}

impl Drop for CandidateGatherer {
    fn drop(&mut self) {
        depth::leave();
    }
}

/// How many gatherings are open on this thread.
#[cfg(feature = "std")]
mod depth {
    std::thread_local! {
        static DEPTH: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    }

    pub fn get() -> usize {
        DEPTH.with(core::cell::Cell::get)
    }

    pub fn enter() {
        DEPTH.with(|depth| depth.set(depth.get() + 1));
    }

    pub fn leave() {
        DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// How many gatherings are open: with no threads to be local to, one depth
/// for the process.
#[cfg(not(feature = "std"))]
mod depth {
    use cubecl_environment::sync::{AtomicUsize, Ordering};

    static DEPTH: AtomicUsize = AtomicUsize::new(0);

    pub fn get() -> usize {
        DEPTH.load(Ordering::Relaxed)
    }

    pub fn enter() {
        DEPTH.fetch_add(1, Ordering::Relaxed);
    }

    pub fn leave() {
        DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_gathers_until_its_last_gatherer_drops() {
        let outer = CandidateGatherer::gather();
        let inner = CandidateGatherer::gather();
        drop(inner);
        let still = CandidateGatherer::is_gathering();
        drop(outer);

        assert!(still);
        assert!(!CandidateGatherer::is_gathering());
    }
}
