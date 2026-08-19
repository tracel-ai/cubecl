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

    fn pin_current(_cpu: CoreId) {}
}
