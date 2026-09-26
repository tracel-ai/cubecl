use cubecl_common::device::ServiceId;
use cubecl_environment::{collections::HashMap, sync::Mutex};
use cubecl_runtime::client::Client;

/// What each device has open, and what it has yet to give back.
static POOLED_PROBES: Mutex<Option<HashMap<ServiceId, Pooled>>> = Mutex::new(None);

/// One device's pooled probes.
#[derive(Default)]
struct Pooled {
    /// Sweeps measuring against its pools right now.
    holders: usize,
    /// A release a probe could not make while one of them held the pools,
    /// left for whoever finds them free next.
    owed: bool,
}

/// A device whose probes leave their pools with the allocator, for as long as
/// one of these is alive, since faulting the next one back in costs more than
/// the measurement. Counted per device, so a sweep that ends cannot release the
/// pool another is still measuring against.
pub(super) struct PooledProbes {
    service: ServiceId,
}

impl PooledProbes {
    pub(super) fn enter(client: &Client) -> Self {
        Self::enter_service(client.service_id())
    }

    fn enter_service(service: ServiceId) -> Self {
        let mut pooled = POOLED_PROBES.lock();

        pooled
            .get_or_insert_with(HashMap::new)
            .entry(service)
            .or_default()
            .holders += 1;

        Self { service }
    }

    /// Give back the pools a probe took, unless a sweep is still measuring
    /// against them, where the release is left owed instead.
    ///
    /// `probed` is whether this caller took any. One that took none still makes
    /// a release another owes: a sweep answered from the cache is otherwise the
    /// last one out, and the unconditional cleanup this gate replaced meant
    /// whoever finished last released.
    pub(super) fn release(client: &Client, probed: bool) {
        // A cleanup refused because a stream records a graph leaves the memory
        // for the next one; the sweep needs nothing from it.
        Self::release_service(client.service_id(), probed, || {
            let _ = client.memory_cleanup();
        });
    }

    /// `release` runs with the lock dropped: it blocks on the device thread,
    /// which takes this lock itself when it autotunes.
    fn release_service(service: ServiceId, probed: bool, release: impl FnOnce()) {
        let owed = {
            let mut pooled = POOLED_PROBES.lock();
            let devices = pooled.get_or_insert_with(HashMap::new);
            let device = devices.entry(service).or_default();

            let owed = probed || device.owed;
            let held = device.holders > 0;
            device.owed = owed && held;

            if !held && !device.owed {
                devices.remove(&service);
            }

            owed && !held
        };

        if owed {
            release();
        }
    }
}

impl Drop for PooledProbes {
    fn drop(&mut self) {
        let mut pooled = POOLED_PROBES.lock();
        let Some(sweeps) = pooled.as_mut() else {
            return;
        };

        if let Some(device) = sweeps.get_mut(&self.service) {
            device.holders -= 1;

            if device.holders == 0 && !device.owed {
                sweeps.remove(&self.service);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::device::DeviceId;

    /// A service to key pooled state on, distinct per `index`.
    fn service(index: u16) -> ServiceId {
        ServiceId::of::<u8>(DeviceId::new(0, index))
    }

    fn pooled(service: ServiceId) -> bool {
        POOLED_PROBES
            .lock()
            .as_ref()
            .and_then(|devices| devices.get(&service))
            .is_some_and(|device| device.holders > 0)
    }

    /// A sweep that ends while another is running must not release the pool the
    /// other is still measuring against, which is what a single flag would do.
    #[test]
    fn a_sweep_that_ends_leaves_an_overlapping_one_pooled() {
        let device = service(1);
        let held = PooledProbes::enter_service(device);

        {
            let _ends_first = PooledProbes::enter_service(device);
            assert!(pooled(device));
        }

        assert!(pooled(device), "one sweep is still running");
        drop(held);
        assert!(!pooled(device));
    }

    /// The same, across threads, where the two sweeps genuinely overlap rather
    /// than nest.
    #[test]
    fn a_sweep_on_another_thread_keeps_its_own_device_pooled() {
        use std::sync::{Arc, Barrier};

        let device = service(2);
        let (entered, ended) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));

        let holder = std::thread::spawn({
            let (entered, ended) = (entered.clone(), ended.clone());

            move || {
                let _pooled = PooledProbes::enter_service(device);
                entered.wait();
                ended.wait();
            }
        });

        {
            let _pooled = PooledProbes::enter_service(device);
            entered.wait();
        }

        assert!(pooled(device), "the other thread's sweep is still running");
        ended.wait();
        holder.join().expect("the holding thread finishes");
        assert!(!pooled(device));
    }

    /// Pools are a device's own, and two cards of the same model are two
    /// devices: a sweep on one says nothing about the other.
    #[test]
    fn a_sweep_pools_only_the_device_it_measures() {
        let _pooled = PooledProbes::enter_service(service(3));

        assert!(pooled(service(3)));
        assert!(!pooled(service(4)));
    }

    #[test]
    fn a_running_sweep_is_not_released() {
        let device = service(5);
        let _pooled = PooledProbes::enter_service(device);
        let mut released = false;

        PooledProbes::release_service(device, true, || released = true);

        assert!(!released);
    }

    /// The cache answering is not a probe, and releasing after one costs a
    /// blocking device call for pools nothing took.
    #[test]
    fn a_caller_that_probed_nothing_releases_nothing() {
        let mut released = false;

        PooledProbes::release_service(service(7), false, || released = true);

        assert!(!released);
    }

    /// A probe that runs beside a sweep cannot release while the sweep holds
    /// the pools, and the sweep, answered from the cache, has nothing of its
    /// own to release. Without the debt, what the probe took stays with the
    /// allocator until another probe on that device happens to release it.
    #[test]
    fn a_release_a_probe_could_not_make_is_made_by_the_sweep() {
        let device = service(8);
        let sweep = PooledProbes::enter_service(device);
        let mut released = false;

        PooledProbes::release_service(device, true, || released = true);
        assert!(!released, "the sweep is still measuring against the pools");

        drop(sweep);
        PooledProbes::release_service(device, false, || released = true);

        assert!(released);
    }

    /// And once made, it is not owed twice.
    #[test]
    fn a_release_that_was_made_is_not_owed_again() {
        let device = service(9);
        let sweep = PooledProbes::enter_service(device);

        PooledProbes::release_service(device, true, || ());
        drop(sweep);
        PooledProbes::release_service(device, false, || ());

        let mut released = false;
        PooledProbes::release_service(device, false, || released = true);

        assert!(!released);
    }

    #[test]
    fn a_release_is_issued_with_the_lock_dropped() {
        use std::{sync::mpsc, time::Duration};

        let (took_lock, lock_taken) = mpsc::channel();

        PooledProbes::release_service(service(6), true, || {
            std::thread::spawn(move || {
                let _pooled = POOLED_PROBES.lock();
                let _ = took_lock.send(());
            });

            lock_taken
                .recv_timeout(Duration::from_secs(5))
                .expect("another thread takes the lock while a release runs");
        });
    }
}
