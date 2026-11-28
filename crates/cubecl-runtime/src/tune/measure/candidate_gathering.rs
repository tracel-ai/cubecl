/// One key's candidates running on this thread with their launches discarded, under a
/// `CompileOnly` override or to compile a batch together, so a key their launches reach knows it
/// was reached inside them.
///
/// # Assumptions
///
/// - A gathering drops on the thread that opened it.
pub struct CandidateGathering {
    _private: (),
}

impl CandidateGathering {
    /// A gathering open on this thread until it drops.
    pub fn open() -> Self {
        depth::enter();
        Self { _private: () }
    }

    /// Whether another key's candidates are running on this thread.
    pub fn is_open() -> bool {
        depth::get() > 0
    }
}

impl Drop for CandidateGathering {
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
    fn a_nested_gathering_leaves_the_outer_one_open() {
        let outer = CandidateGathering::open();
        drop(CandidateGathering::open());
        let still = CandidateGathering::is_open();
        drop(outer);

        assert!(still);
    }

    #[test]
    fn a_thread_gathers_until_its_last_gathering_drops() {
        drop(CandidateGathering::open());

        assert!(!CandidateGathering::is_open());
    }
}
