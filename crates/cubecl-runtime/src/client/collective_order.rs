//! The order collective operations reach every device of the process in.

use alloc::vec::Vec;
use cubecl_common::device::DeviceId;
use cubecl_environment::sync::Mutex;

/// NCCL pairs the operations of a communicator in the order each device queues them, so every
/// device has to queue them in one order.
pub static COLLECTIVE_ORDER: CollectiveOrder = CollectiveOrder::new();

/// How long a transfer waits on a split `all_reduce` before it says so.
#[cfg(feature = "std")]
const WAIT_WARNING: core::time::Duration = core::time::Duration::from_secs(10);

/// Queues collective operations one at a time, holding back a transfer while an `all_reduce` is
/// queued on some of its devices and not yet on the others.
pub struct CollectiveOrder {
    groups: Mutex<ReduceGroups>,
    #[cfg(feature = "std")]
    rejoined: std::sync::Condvar,
}

impl CollectiveOrder {
    const fn new() -> Self {
        Self {
            groups: Mutex::new(ReduceGroups {
                groups: Vec::new(),
                waiting: 0,
            }),
            #[cfg(feature = "std")]
            rejoined: std::sync::Condvar::new(),
        }
    }

    /// Runs `queue`, which queues both halves of a transfer between `source` and `destination`,
    /// once no `all_reduce` over either device is split: the transfer would land between its
    /// halves, and the two devices would pair them in opposite orders.
    pub fn transfer<R>(
        &self,
        source: DeviceId,
        destination: DeviceId,
        queue: impl FnOnce() -> R,
    ) -> R {
        let mut groups = self.groups.lock();
        while let Some(split) = groups.split_over(source, destination) {
            groups.waiting += 1;
            #[cfg(feature = "std")]
            {
                let (guard, waited) = self
                    .rejoined
                    .wait_timeout(groups, WAIT_WARNING)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                groups = guard;
                if waited.timed_out() {
                    log::warn!(
                        "A transfer from {source:?} to {destination:?} is waiting on an all_reduce over {split:?} that only some of those devices have queued"
                    );
                }
            }
            #[cfg(not(feature = "std"))]
            {
                let _ = split;
                drop(groups);
                core::hint::spin_loop();
                groups = self.groups.lock();
            }
            groups.waiting -= 1;
        }
        queue()
    }

    /// Runs `queue` with `devices`, to queue `rank`'s part of an `all_reduce` over them.
    pub fn all_reduce<R>(
        &self,
        rank: DeviceId,
        devices: Vec<DeviceId>,
        queue: impl FnOnce(Vec<DeviceId>) -> R,
    ) -> R {
        let mut groups = self.groups.lock();
        let rejoined = groups.queued(rank, &devices);
        let output = queue(devices);
        if rejoined && groups.waiting > 0 {
            #[cfg(feature = "std")]
            self.rejoined.notify_all();
        }
        output
    }
}

/// Every set of devices an `all_reduce` has run over.
struct ReduceGroups {
    groups: Vec<ReduceGroup>,
    /// Transfers waiting for a group to rejoin: waking them is a system call even with none.
    waiting: usize,
}

impl ReduceGroups {
    /// The devices of an `all_reduce` split across `source` or `destination`, if there is one.
    fn split_over(&self, source: DeviceId, destination: DeviceId) -> Option<Vec<DeviceId>> {
        self.groups
            .iter()
            .find(|group| group.is_split() && (group.has(source) || group.has(destination)))
            .map(ReduceGroup::devices)
    }

    /// Counts an `all_reduce` `rank` queued over `devices`, and returns whether every one of them
    /// has now queued as many.
    fn queued(&mut self, rank: DeviceId, devices: &[DeviceId]) -> bool {
        let index = match self.groups.iter().position(|known| known.is_over(devices)) {
            Some(index) => index,
            None => {
                self.groups.push(ReduceGroup::over(devices));
                self.groups.len() - 1
            }
        };
        let group = &mut self.groups[index];
        group.count(rank);
        !group.is_split()
    }
}

/// The devices of an `all_reduce`, each with how many `all_reduce` calls over them it has queued.
struct ReduceGroup {
    queued: Vec<(DeviceId, u64)>,
}

impl ReduceGroup {
    fn over(devices: &[DeviceId]) -> Self {
        let mut devices = devices.to_vec();
        devices.sort();
        devices.dedup();
        Self {
            queued: devices.into_iter().map(|device| (device, 0)).collect(),
        }
    }

    fn devices(&self) -> Vec<DeviceId> {
        self.queued.iter().map(|(device, _)| *device).collect()
    }

    /// Whether `devices` names exactly this group's devices, in any order.
    fn is_over(&self, devices: &[DeviceId]) -> bool {
        devices.iter().all(|device| self.has(*device))
            && self
                .queued
                .iter()
                .all(|(member, _)| devices.contains(member))
    }

    fn has(&self, device: DeviceId) -> bool {
        self.queued.iter().any(|(member, _)| *member == device)
    }

    fn count(&mut self, rank: DeviceId) {
        if let Some((_, queued)) = self.queued.iter_mut().find(|(member, _)| *member == rank) {
            *queued += 1;
        }
    }

    /// Whether some device has queued an `all_reduce` that another has not yet.
    fn is_split(&self) -> bool {
        let mut counts = self.queued.iter().map(|(_, queued)| *queued);
        let first = counts.next();
        counts.any(|count| Some(count) != first)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    fn device(index_id: u16) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id,
        }
    }

    #[test]
    fn a_transfer_queues_after_an_all_reduce_split_across_its_devices() {
        let order = Arc::new(CollectiveOrder::new());
        let queued = Arc::new(StdMutex::new(Vec::new()));
        let group = [device(0), device(1)];
        let record = |name: &'static str| {
            let queued = queued.clone();
            move || queued.lock().unwrap().push(name)
        };
        let reduce = |name| {
            let record = record(name);
            move |_| record()
        };

        order.all_reduce(device(0), group.to_vec(), reduce("reduce on 0"));
        let transfer = {
            let order = order.clone();
            let queue = record("transfer");
            std::thread::spawn(move || order.transfer(device(1), device(0), queue))
        };
        order.all_reduce(device(1), group.to_vec(), reduce("reduce on 1"));
        transfer.join().unwrap();

        assert_eq!(
            *queued.lock().unwrap(),
            ["reduce on 0", "reduce on 1", "transfer"]
        );
    }

    #[test]
    fn a_transfer_between_other_devices_does_not_wait() {
        let order = CollectiveOrder::new();
        order.all_reduce(device(0), alloc::vec![device(0), device(1)], |_| ());

        assert!(order.transfer(device(2), device(3), || true));
    }
}
