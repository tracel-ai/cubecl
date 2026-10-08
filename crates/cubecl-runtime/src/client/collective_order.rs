//! The order collective operations reach every device of the process in.

use alloc::{boxed::Box, vec::Vec};
use core::sync::atomic::AtomicPtr;
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

/// Gives collective operations one order on every device: each takes its places in the lines of
/// its devices at once, then queues on each device in turn. A transfer takes its places only once
/// no `all_reduce` over its devices is queued on some of them and not yet on the others.
///
/// Nothing queues while holding the order: a device's queue can be full while its thread waits on
/// another device, whose part may be next in line behind the order.
pub struct CollectiveOrder {
    /// Shared by `all_reduce` parts taking a place, and held alone by a transfer taking its two.
    placing: RwLock<()>,
    groups: AddOnlyList<ReduceGroup>,
    lines: AddOnlyList<DeviceLine>,
    /// Taken to wake waiting transfers, so the wake cannot fall between a transfer's last look
    /// and its wait.
    rejoin: Mutex<()>,
    #[cfg(feature = "std")]
    rejoined: Condvar,
}

impl CollectiveOrder {
    const fn new() -> Self {
        Self {
            placing: RwLock::new(()),
            groups: AddOnlyList::new(),
            lines: AddOnlyList::new(),
            rejoin: Mutex::new(()),
            #[cfg(feature = "std")]
            rejoined: Condvar::new(),
        }
    }

    /// Runs `send` in its turn on `source`, then `receive` in its turn on `destination`, once no
    /// `all_reduce` over either device is split: the transfer would land between its parts, and
    /// the two devices would pair them in opposite orders.
    pub fn transfer(
        &self,
        source: DeviceId,
        destination: DeviceId,
        send: impl FnOnce(),
        receive: impl FnOnce(),
    ) {
        let mut placing = self.placing.write();
        while let Some(split) = self.split_over(source, destination) {
            #[cfg_attr(not(feature = "std"), allow(unused_mut))]
            let mut rejoin = self.rejoin.lock();
            // The parts still missing take their places on the shared side.
            drop(placing);
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
            placing = self.placing.write();
        }
        let sending = self.line(source).take();
        let receiving = self.line(destination).take();
        drop(placing);
        sending.queue(send);
        receiving.queue(receive);
    }

    /// Runs `queue` with `devices` in `rank`'s turn, to queue its part of an `all_reduce` over
    /// them.
    pub fn all_reduce<R>(
        &self,
        rank: DeviceId,
        devices: Vec<DeviceId>,
        queue: impl FnOnce(Vec<DeviceId>) -> R,
    ) -> R {
        let place = {
            let _placing = self.placing.read();
            if self.count(rank, &devices) {
                let _rejoin = self.rejoin.lock();
                #[cfg(feature = "std")]
                self.rejoined.notify_all();
            }
            self.line(rank).take()
        };
        place.queue(|| queue(devices))
    }

    /// The devices of an `all_reduce` split across `source` or `destination`, if there is one.
    fn split_over(&self, source: DeviceId, destination: DeviceId) -> Option<Vec<DeviceId>> {
        self.groups
            .iter()
            .find(|group| group.is_split() && (group.has(source) || group.has(destination)))
            .map(ReduceGroup::devices)
    }

    /// Counts an `all_reduce` `rank` queued over `devices`, and returns whether every one of them
    /// has now queued as many.
    fn count(&self, rank: DeviceId, devices: &[DeviceId]) -> bool {
        self.groups
            .find_or_add(
                |group| group.is_over(devices),
                || ReduceGroup::over(devices),
            )
            .count(rank)
    }

    fn line(&self, device: DeviceId) -> &DeviceLine {
        self.lines
            .find_or_add(|line| line.device == device, || DeviceLine::new(device))
    }
}

/// A list that only grows, so readers walk it without a lock, and that frees its items with
/// itself. Its atomic pointer makes it `Sync` whatever it holds, hence the bound.
struct AddOnlyList<T: Send + Sync> {
    newest: AtomicPtr<Node<T>>,
    adding: Mutex<()>,
}

struct Node<T> {
    item: T,
    older: *mut Node<T>,
}

impl<T: Send + Sync> AddOnlyList<T> {
    const fn new() -> Self {
        Self {
            newest: AtomicPtr::new(core::ptr::null_mut()),
            adding: Mutex::new(()),
        }
    }

    fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        let mut node = self.newest.load(Ordering::Acquire);
        core::iter::from_fn(move || {
            // SAFETY: a node is built before it is published, and only `drop`, which has no
            // reader left, frees it.
            let current = unsafe { node.as_ref() }?;
            node = current.older;
            Some(&current.item)
        })
    }

    /// The item that `matches`, added from `make` if there is none yet.
    fn find_or_add(&self, matches: impl Fn(&T) -> bool, make: impl FnOnce() -> T) -> &T {
        if let Some(item) = self.iter().find(|item| matches(item)) {
            return item;
        }
        let _adding = self.adding.lock();
        if let Some(item) = self.iter().find(|item| matches(item)) {
            return item;
        }
        let older = self.newest.load(Ordering::Relaxed);
        let node = Box::into_raw(Box::new(Node {
            item: make(),
            older,
        }));
        self.newest.store(node, Ordering::Release);
        // SAFETY: the node was just published, and lives until `drop`.
        unsafe { &(*node).item }
    }
}

impl<T: Send + Sync> Drop for AddOnlyList<T> {
    fn drop(&mut self) {
        let mut node = *self.newest.get_mut();
        while !node.is_null() {
            // SAFETY: every node came from `Box::into_raw` in `find_or_add`, and `&mut self`
            // leaves no reader behind.
            let current = unsafe { Box::from_raw(node) };
            node = current.older;
        }
    }
}

/// The places taken on one device, served in the order they were taken.
struct DeviceLine {
    device: DeviceId,
    taken: AtomicUsize,
    served: AtomicUsize,
    #[cfg(feature = "std")]
    waiting: AtomicUsize,
    #[cfg(feature = "std")]
    turn: Mutex<()>,
    #[cfg(feature = "std")]
    turned: Condvar,
}

impl DeviceLine {
    fn new(device: DeviceId) -> Self {
        Self {
            device,
            taken: AtomicUsize::new(0),
            served: AtomicUsize::new(0),
            #[cfg(feature = "std")]
            waiting: AtomicUsize::new(0),
            #[cfg(feature = "std")]
            turn: Mutex::new(()),
            #[cfg(feature = "std")]
            turned: Condvar::new(),
        }
    }

    fn take(&self) -> Place<'_> {
        Place {
            line: self,
            number: self.taken.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn wait_for(&self, number: usize) {
        #[cfg(feature = "std")]
        if self.served.load(Ordering::Acquire) != number {
            let mut turn = self.turn.lock();
            self.waiting.fetch_add(1, Ordering::SeqCst);
            while self.served.load(Ordering::SeqCst) != number {
                self.turned.wait(&mut turn);
            }
            self.waiting.fetch_sub(1, Ordering::SeqCst);
        }
        #[cfg(not(feature = "std"))]
        while self.served.load(Ordering::Acquire) != number {
            core::hint::spin_loop();
        }
    }

    fn pass(&self) {
        // Of this and a waiter's count, both `SeqCst`, the later one's read sees the earlier.
        self.served.fetch_add(1, Ordering::SeqCst);
        #[cfg(feature = "std")]
        if self.waiting.load(Ordering::SeqCst) > 0 {
            let _turn = self.turn.lock();
            self.turned.notify_all();
        }
    }
}

/// A place in a device's line, passed on when dropped, so a panic cannot stall the line.
struct Place<'a> {
    line: &'a DeviceLine,
    number: usize,
}

impl Place<'_> {
    /// Runs `operation` once every earlier place on the device has queued.
    fn queue<R>(self, operation: impl FnOnce() -> R) -> R {
        self.line.wait_for(self.number);
        operation()
    }
}

impl Drop for Place<'_> {
    fn drop(&mut self) {
        self.line.wait_for(self.number);
        self.line.pass();
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
    use std::sync::{Arc, Mutex as StdMutex, mpsc};

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
            let (send, receive) = (record("send"), record("receive"));
            std::thread::spawn(move || order.transfer(device(1), device(0), send, receive))
        };
        order.all_reduce(device(1), group.to_vec(), reduce("reduce on 1"));
        transfer.join().unwrap();

        assert_eq!(
            *queued.lock().unwrap(),
            ["reduce on 0", "reduce on 1", "send", "receive"]
        );
    }

    #[test]
    fn a_transfer_between_other_devices_does_not_wait() {
        let order = CollectiveOrder::new();
        let mut sent = false;
        order.all_reduce(device(0), alloc::vec![device(0), device(1)], |_| ());

        order.transfer(device(2), device(3), || sent = true, || ());

        assert!(sent);
    }

    #[test]
    fn a_transfer_does_not_wait_for_a_part_still_queueing() {
        let order = Arc::new(CollectiveOrder::new());
        let (transferred, done) = mpsc::channel();

        let transfer = order.all_reduce(device(0), alloc::vec![device(0)], |_| {
            let order = order.clone();
            let transfer = std::thread::spawn(move || {
                order.transfer(
                    device(1),
                    device(2),
                    || (),
                    move || transferred.send(()).unwrap(),
                )
            });
            done.recv().unwrap();
            transfer
        });

        transfer.join().unwrap();
    }
}
