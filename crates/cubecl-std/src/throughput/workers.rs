use cubecl_common::device::ServiceId;
use cubecl_environment::{collections::HashMap, sync::Mutex};
use cubecl_runtime::{client::Client, throughput::MemoryAccess};

use crate::throughput::LaunchConfig;

/// Worker counts a memory probe is swept over before the fastest is kept, once
/// per device and access. Five halvings reach a sixteenth of the cores, and
/// each one costs a ranking pass.
const MEMORY_WORKER_SHAPES: usize = 5;

/// The worker count each device streams each access fastest at, once one probe
/// of that access has swept for it.
static SATURATING_WORKERS: Mutex<Option<HashMap<(ServiceId, MemoryAccess), u32>>> =
    Mutex::new(None);

/// How many workers a memory probe is launched across, swept once per device
/// and access and remembered for every working set after.
pub(super) struct WorkerSweep;

impl WorkerSweep {
    /// The launch shapes a memory probe is measured in. A GPU keeps the cube it
    /// was pinned to, since a wider one reports several times the bus rate. A
    /// CPU sweeps instead, because what saturates a memory system is a property
    /// of the controller and not of the window a probe moves through it.
    pub(super) fn shapes(
        client: &Client,
        config: LaunchConfig,
        access: MemoryAccess,
    ) -> alloc::vec::Vec<LaunchConfig> {
        if client.properties().hardware.num_cpu_cores.is_none() {
            return alloc::vec![config];
        }

        if let Some(units) = Self::remembered(client, access) {
            return alloc::vec![config.with_units(client, units)];
        }

        Self::counts(config.cube_dim.num_elems() as usize)
            .into_iter()
            .map(|units| config.with_units(client, units as u32))
            .collect()
    }

    /// The full launch first and halvings after. Little's law would derive the
    /// count from the bandwidth, which is the thing being measured, so it is
    /// swept instead.
    fn counts(units: usize) -> alloc::vec::Vec<usize> {
        core::iter::successors(Some(units.max(1)), |units| (*units > 1).then(|| units / 2))
            .take(MEMORY_WORKER_SHAPES)
            .collect()
    }

    fn remembered(client: &Client, access: MemoryAccess) -> Option<u32> {
        let workers = SATURATING_WORKERS.lock();
        let workers = workers.as_ref()?;

        workers.get(&(client.service_id(), access)).copied()
    }

    pub(super) fn remember(client: &Client, access: MemoryAccess, units: u32) {
        let mut workers = SATURATING_WORKERS.lock();

        workers
            .get_or_insert_with(HashMap::new)
            .insert((client.service_id(), access), units);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full launch is measured first, so a device whose peak is there
    /// keeps the shape it had before the sweep existed.
    #[test]
    fn the_sweep_starts_at_the_full_launch_and_halves_to_the_budget() {
        assert_eq!(WorkerSweep::counts(16), alloc::vec![16, 8, 4, 2, 1]);
        assert_eq!(WorkerSweep::counts(128), alloc::vec![128, 64, 32, 16, 8]);
    }

    /// Every count is a launch, so one core must not be handed an empty sweep
    /// and reported as untimeable.
    #[test]
    fn one_core_is_still_a_shape() {
        assert_eq!(WorkerSweep::counts(1), alloc::vec![1]);
    }
}
