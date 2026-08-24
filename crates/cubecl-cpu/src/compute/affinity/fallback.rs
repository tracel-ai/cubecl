use super::{CoreId, ThreadAffinity};

/// Platforms without an affinity API: std parallelism, no topology, no pinning.
pub(super) struct Platform;

impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        (0..cores).map(CoreId).collect()
    }

    fn physical_core(_cpu: CoreId) -> Option<CoreId> {
        None
    }

    fn l1d_cache_size() -> Option<usize> {
        None
    }

    fn llc_cache_size() -> Option<usize> {
        None
    }

    fn pin_current(_cpu: CoreId) {}
}

/// What the tests in [`super`] need of this platform beyond the trait.
#[cfg(test)]
impl Platform {
    /// No topology to read, so the tests require nothing of it.
    pub(super) const READS_TOPOLOGY: bool = false;

    /// Nothing is pinned, so there is nothing to observe.
    pub(super) fn current_cpu() -> Option<CoreId> {
        None
    }
}
