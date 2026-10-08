//! The order collective operations reach every device of the process in.

use alloc::vec::Vec;
use cubecl_common::device::DeviceId;
use cubecl_environment::sync::{AtomicUsize, Ordering};
#[cfg(not(feature = "std"))]
use cubecl_environment::sync::{Mutex, RwLock};
#[cfg(feature = "std")]
use parking_lot::{Condvar, Mutex, RwLock};

/// NCCL pairs the operations of a communicator in the order each device queues them, so every
/// device has to queue them in one order.
pub static COLLECTIVE_ORDER: CollectiveOrder = CollectiveOrder::new();

/// How long a transfer waits on a split `all_reduce` before it says so.
#[cfg(feature = "std")]
const WAIT_WARNING: core::time::Duration = core::time::Duration::from_secs(10);

/// Queues collective operations so every device sees them in one order: the parts of
/// `all_reduce` calls queue side by side, and a transfer queues alone, held back while an
/// `all_reduce` is queued on some of its devices and not yet on the others.
pub struct CollectiveOrder {
    /// Shared by `all_reduce` parts, and held alone by a transfer.
    queueing: RwLock<()>,
    groups: RwLock<Vec<ReduceGroup>>,
    /// Taken to wake waiting transfers, so the wake cannot fall between a transfer's last look
    /// and its wait.
    rejoin: Mutex<()>,
    #[cfg(feature = "std")]
    rejoined: Condvar,
}

impl CollectiveOrder {
    const fn new() -> Self {
        Self {
            queueing: RwLock::new(()),
            groups: RwLock::new(Vec::new()),
            rejoin: Mutex::new(()),
            #[cfg(feature = "std")]
            rejoined: Condvar::new(),
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
        let mut queueing = self.queueing.write();
        while let Some(split) = self.split_over(source, destination) {
            #[cfg_attr(not(feature = "std"), allow(unused_mut))]
            let mut rejoin = self.rejoin.lock();
            // The parts still missing queue on the shared side.
            drop(queueing);
            #[cfg(feature = "std")]
            if self
                .rejoined
                .wait_for(&mut rejoin, WAIT_WARNING)
                .timed_out()
            {
                log::warn!(
                    "A transfer from {source:?} to {destination:?} is waiting on an all_reduce over {split:?} that only some of those devices have queued"
                );
            }
            #[cfg(not(feature = "std"))]
            let _ = split;
            drop(rejoin);
            #[cfg(not(feature = "std"))]
            core::hint::spin_loop();
            queueing = self.queueing.write();
        }
        let output = queue();
        drop(queueing);
        output
    }

    /// Runs `queue` with `devices`, to queue `rank`'s part of an `all_reduce` over them.
    pub fn all_reduce<R>(
        &self,
        rank: DeviceId,
        devices: Vec<DeviceId>,
        queue: impl FnOnce(Vec<DeviceId>) -> R,
    ) -> R {
        let _queueing = self.queueing.read();
        if self.count(rank, &devices) {
            let _rejoin = self.rejoin.lock();
            #[cfg(feature = "std")]
            self.rejoined.notify_all();
        }
        queue(devices)
    }

    /// The devices of an `all_reduce` split across `source` or `destination`, if there is one.
    fn split_over(&self, source: DeviceId, destination: DeviceId) -> Option<Vec<DeviceId>> {
        self.groups
            .read()
            .iter()
            .find(|group| group.is_split() && (group.has(source) || group.has(destination)))
            .map(ReduceGroup::devices)
    }

    /// Counts an `all_reduce` `rank` queued over `devices`, and returns whether every one of them
    /// has now queued as many.
    fn count(&self, rank: DeviceId, devices: &[DeviceId]) -> bool {
        if let Some(group) = self
            .groups
            .read()
            .iter()
            .find(|group| group.is_over(devices))
        {
            return group.count(rank);
        }
        let mut groups = self.groups.write();
        let index = match groups.iter().position(|group| group.is_over(devices)) {
            Some(index) => index,
            None => {
                groups.push(ReduceGroup::over(devices));
                groups.len() - 1
            }
        };
        groups[index].count(rank)
    }
}

/// The devices of an `all_reduce`, each with how many `all_reduce` calls over them it has queued.
struct ReduceGroup {
    queued: Vec<(DeviceId, AtomicUsize)>,
}

impl ReduceGroup {
    fn over(devices: &[DeviceId]) -> Self {
        let mut devices = devices.to_vec();
        devices.sort();
        devices.dedup();
        Self {
            queued: devices
                .into_iter()
                .map(|device| (device, AtomicUsize::new(0)))
                .collect(),
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

    /// Counts one more `all_reduce` on `rank`, and returns whether the group has rejoined. Of
    /// ranks counting at once, the last to read sees every count, so a rejoin is never missed.
    fn count(&self, rank: DeviceId) -> bool {
        if let Some((_, queued)) = self.queued.iter().find(|(member, _)| *member == rank) {
            queued.fetch_add(1, Ordering::SeqCst);
        }
        !self.is_split()
    }

    /// Whether some device has queued an `all_reduce` that another has not yet.
    fn is_split(&self) -> bool {
        let mut counts = self
            .queued
            .iter()
            .map(|(_, queued)| queued.load(Ordering::SeqCst));
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
