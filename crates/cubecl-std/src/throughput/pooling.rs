use cubecl_common::device::ServiceId;
use cubecl_environment::{collections::HashMap, sync::Mutex};
use cubecl_runtime::client::Client;

/// How many sweeps are holding each device's pools open.
static POOLED_PROBES: Mutex<Option<HashMap<ServiceId, usize>>> = Mutex::new(None);

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

        *pooled
            .get_or_insert_with(HashMap::new)
            .entry(service)
            .or_insert(0) += 1;

        Self { service }
    }

    pub(super) fn cleanup_unless_held(client: &Client) {
        Self::cleanup_unless_held_by(client.service_id(), || client.memory_cleanup());
    }

    /// `release` runs with the lock dropped: it blocks on the device thread,
    /// which takes this lock itself when it autotunes.
    fn cleanup_unless_held_by(service: ServiceId, release: impl FnOnce()) {
        let held = {
            let pooled = POOLED_PROBES.lock();
            Self::held_by(&pooled, service)
        };

        if !held {
            release();
        }
    }

    fn held_by(pooled: &Option<HashMap<ServiceId, usize>>, service: ServiceId) -> bool {
        pooled
            .as_ref()
            .is_some_and(|sweeps| sweeps.contains_key(&service))
    }
}

impl Drop for PooledProbes {
    fn drop(&mut self) {
        let mut pooled = POOLED_PROBES.lock();
        let Some(sweeps) = pooled.as_mut() else {
            return;
        };

        if let Some(holders) = sweeps.get_mut(&self.service) {
            *holders -= 1;

            if *holders == 0 {
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
        PooledProbes::held_by(&POOLED_PROBES.lock(), service)
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

        PooledProbes::cleanup_unless_held_by(device, || released = true);

        assert!(!released);
    }

    #[test]
    fn a_release_is_issued_with_the_lock_dropped() {
        use std::{sync::mpsc, time::Duration};

        let (took_lock, lock_taken) = mpsc::channel();

        PooledProbes::cleanup_unless_held_by(service(6), || {
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
