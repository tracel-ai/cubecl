//! The order collective operations reach every device of the process in.

use alloc::{boxed::Box, vec::Vec};
use core::sync::atomic::AtomicPtr;
use cubecl_common::device::DeviceId;
use cubecl_environment::sync::{AtomicUsize, Mutex, Ordering, RwLock};
#[cfg(feature = "std")]
use cubecl_environment::{
    sync::Condvar,
    time::{Duration, Instant},
};
#[cfg(feature = "std")]
use std::sync::PoisonError;

/// NCCL pairs the operations of a communicator in the order each device queues them, so every
/// device has to queue them in one order.
pub static COLLECTIVE_ORDER: CollectiveOrder = CollectiveOrder::new();

/// Gives collective operations one order on every device. Nothing queues while holding the order:
/// a device's queue can be full while its thread waits on another device, whose part may be next
/// in line behind the order.
#[derive(Debug)]
pub struct CollectiveOrder {
    /// Shared by `all_reduce` parts taking a ticket, and held alone by a transfer taking its two.
    ticketing: RwLock<()>,
    groups: AddOnlyList<ReduceGroup>,
    sequencers: AddOnlyList<DeviceSequencer>,
    /// Taken to wake waiting transfers, so the wake cannot fall between a transfer's last look
    /// and its wait.
    #[cfg(feature = "std")]
    rejoin: Mutex<()>,
    #[cfg(feature = "std")]
    rejoined: Condvar,
    #[cfg(feature = "std")]
    transfers_waiting: AtomicUsize,
}

impl CollectiveOrder {
    const fn new() -> Self {
        Self {
            ticketing: RwLock::new(()),
            groups: AddOnlyList::new(),
            sequencers: AddOnlyList::new(),
            #[cfg(feature = "std")]
            rejoin: Mutex::new(()),
            #[cfg(feature = "std")]
            rejoined: Condvar::new(),
            #[cfg(feature = "std")]
            transfers_waiting: AtomicUsize::new(0),
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
        let mut ticketing = self.ticketing.write();
        #[cfg(feature = "std")]
        let mut warn_at = None;
        while let Some(split) = self.split_over(source, destination) {
            #[cfg(feature = "std")]
            {
                let rejoin = self.rejoin.lock();
                // Counted before the parts still missing can take their tickets on the shared
                // side, so the one that rejoins the group sees this transfer waiting.
                self.transfers_waiting.fetch_add(1, Ordering::SeqCst);
                drop(ticketing);
                let deadline = *warn_at.get_or_insert_with(|| Instant::now() + WAIT_WARNING);
                let (rejoin, waited) = self
                    .rejoined
                    .wait_timeout(rejoin, deadline.saturating_duration_since(Instant::now()))
                    .unwrap_or_else(PoisonError::into_inner);
                self.transfers_waiting.fetch_sub(1, Ordering::SeqCst);
                drop(rejoin);
                if waited.timed_out() {
                    log::warn!(
                        "A transfer from {source:?} to {destination:?} is waiting on an all_reduce over {split:?} that only some of those devices have queued"
                    );
                    warn_at = Some(deadline + WAIT_WARNING);
                }
            }
            #[cfg(not(feature = "std"))]
            {
                let _ = split;
                drop(ticketing);
                core::hint::spin_loop();
            }
            ticketing = self.ticketing.write();
        }
        let sending = self.sequencer(source).ticket();
        let receiving = self.sequencer(destination).ticket();
        drop(ticketing);
        sending.queue(send);
        receiving.queue(receive);
    }

    /// Runs `queue` with `devices` in `device`'s turn, to queue its part of an `all_reduce` over
    /// them.
    pub fn all_reduce<R>(
        &self,
        device: DeviceId,
        devices: Vec<DeviceId>,
        queue: impl FnOnce(Vec<DeviceId>) -> R,
    ) -> R {
        let ticket = {
            let _ticketing = self.ticketing.read();
            let group = self.group(&devices);
            group.count(device);
            // A transfer starts waiting only under the exclusive side, so a part counted before
            // it was seen there, and one counted after sees it waiting.
            #[cfg(feature = "std")]
            if self.transfers_waiting.load(Ordering::SeqCst) > 0 && !group.is_split() {
                let _rejoin = self.rejoin.lock();
                self.rejoined.notify_all();
            }
            self.sequencer(device).ticket()
        };
        ticket.queue(|| queue(devices))
    }

    /// The devices of an `all_reduce` split across `source` or `destination`, if there is one.
    fn split_over(&self, source: DeviceId, destination: DeviceId) -> Option<Vec<DeviceId>> {
        self.groups
            .iter()
            .find(|group| group.is_split() && (group.has(source) || group.has(destination)))
            .map(ReduceGroup::devices)
    }

    fn group(&self, devices: &[DeviceId]) -> &ReduceGroup {
        self.groups
            .find_or_add(|group| group.is_over(devices), || ReduceGroup::new(devices))
    }

    fn sequencer(&self, device: DeviceId) -> &DeviceSequencer {
        self.sequencers.find_or_add(
            |sequencer| sequencer.device == device,
            || DeviceSequencer::new(device),
        )
    }
}

/// How long a transfer waits on a split `all_reduce` before it says so, and again after that.
#[cfg(feature = "std")]
const WAIT_WARNING: Duration = Duration::from_secs(10);

/// A list that only grows, so readers walk it without a lock, and that frees its items with
/// itself. Its atomic pointer makes it `Sync` whatever it holds, hence the bound.
#[derive(Debug)]
struct AddOnlyList<T: Send + Sync> {
    newest: AtomicPtr<Node<T>>,
    adding: Mutex<()>,
}

#[derive(Debug)]
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

/// Queues the operations on one device in the order of the tickets they took.
#[derive(Debug)]
struct DeviceSequencer {
    device: DeviceId,
    next_ticket: AtomicUsize,
    now_serving: AtomicUsize,
    #[cfg(feature = "std")]
    waiting: AtomicUsize,
    #[cfg(feature = "std")]
    serving: Mutex<()>,
    #[cfg(feature = "std")]
    served: Condvar,
}

impl DeviceSequencer {
    fn new(device: DeviceId) -> Self {
        Self {
            device,
            next_ticket: AtomicUsize::new(0),
            now_serving: AtomicUsize::new(0),
            #[cfg(feature = "std")]
            waiting: AtomicUsize::new(0),
            #[cfg(feature = "std")]
            serving: Mutex::new(()),
            #[cfg(feature = "std")]
            served: Condvar::new(),
        }
    }

    fn ticket(&self) -> Ticket<'_> {
        Ticket {
            sequencer: self,
            number: self.next_ticket.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn wait_turn(&self, number: usize) {
        #[cfg(feature = "std")]
        if self.now_serving.load(Ordering::Acquire) != number {
            let serving = self.serving.lock();
            self.waiting.fetch_add(1, Ordering::SeqCst);
            let serving = self
                .served
                .wait_while(serving, |_| {
                    self.now_serving.load(Ordering::SeqCst) != number
                })
                .unwrap_or_else(PoisonError::into_inner);
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            drop(serving);
        }
        #[cfg(not(feature = "std"))]
        while self.now_serving.load(Ordering::Acquire) != number {
            core::hint::spin_loop();
        }
    }

    fn advance(&self) {
        // Of this and a waiter's count, both `SeqCst`, the later one's read sees the earlier.
        self.now_serving.fetch_add(1, Ordering::SeqCst);
        #[cfg(feature = "std")]
        if self.waiting.load(Ordering::SeqCst) > 0 {
            let _serving = self.serving.lock();
            self.served.notify_all();
        }
    }
}

/// A turn to queue on a device, given up in turn when dropped, so a panic cannot stall the device.
#[derive(Debug)]
struct Ticket<'a> {
    sequencer: &'a DeviceSequencer,
    number: usize,
}

impl Ticket<'_> {
    /// Runs `operation` once every earlier ticket on the device has queued.
    fn queue<R>(self, operation: impl FnOnce() -> R) -> R {
        self.sequencer.wait_turn(self.number);
        operation()
    }
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        self.sequencer.wait_turn(self.number);
        self.sequencer.advance();
    }
}

/// The devices of an `all_reduce`, each with how many `all_reduce` calls over them it has queued.
#[derive(Debug)]
struct ReduceGroup {
    members: Vec<Member>,
}

#[derive(Debug)]
struct Member {
    device: DeviceId,
    /// Written by its own device on every `all_reduce`, so it keeps cache lines of its own.
    queued: CacheLine<AtomicUsize>,
}

/// A value on cache lines of its own.
#[derive(Debug)]
#[repr(align(128))]
struct CacheLine<T>(T);

impl<T> core::ops::Deref for CacheLine<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl ReduceGroup {
    fn new(devices: &[DeviceId]) -> Self {
        let mut devices = devices.to_vec();
        devices.sort();
        devices.dedup();
        Self {
            members: devices
                .into_iter()
                .map(|device| Member {
                    device,
                    queued: CacheLine(AtomicUsize::new(0)),
                })
                .collect(),
        }
    }

    fn devices(&self) -> Vec<DeviceId> {
        self.members.iter().map(|member| member.device).collect()
    }

    /// Whether `devices` names exactly this group's devices, in any order.
    fn is_over(&self, devices: &[DeviceId]) -> bool {
        devices.iter().all(|device| self.has(*device))
            && self
                .members
                .iter()
                .all(|member| devices.contains(&member.device))
    }

    fn has(&self, device: DeviceId) -> bool {
        self.members.iter().any(|member| member.device == device)
    }

    /// Counts one more `all_reduce` queued on `device`.
    fn count(&self, device: DeviceId) {
        if let Some(member) = self.members.iter().find(|member| member.device == device) {
            member.queued.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Whether some device has queued an `all_reduce` that another has not yet.
    fn is_split(&self) -> bool {
        let mut counts = self
            .members
            .iter()
            .map(|member| member.queued.load(Ordering::SeqCst));
        let first = counts.next();
        counts.any(|count| Some(count) != first)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex, mpsc};

    /// Long enough for a transfer that does not wait to have queued.
    const SETTLE: Duration = Duration::from_millis(50);
    /// Long enough for anything that is not stuck.
    const STUCK: Duration = Duration::from_secs(10);

    #[test]
    fn a_transfer_queues_after_an_all_reduce_split_across_its_devices() {
        let order = Arc::new(CollectiveOrder::new());
        let queued = Arc::new(StdMutex::new(Vec::new()));
        let (transferred, done) = mpsc::channel();
        let group = [device(0), device(1)];

        order.all_reduce(device(0), group.to_vec(), |_| {
            queued.lock().unwrap().push("reduce on 0")
        });
        {
            let (order, queued) = (order.clone(), queued.clone());
            std::thread::spawn(move || {
                order.transfer(
                    device(1),
                    device(0),
                    || queued.lock().unwrap().push("send"),
                    || queued.lock().unwrap().push("receive"),
                );
                transferred.send(()).unwrap();
            });
        }
        assert!(
            done.recv_timeout(SETTLE).is_err(),
            "the transfer queued between the parts of an all_reduce"
        );
        order.all_reduce(device(1), group.to_vec(), |_| {
            queued.lock().unwrap().push("reduce on 1")
        });
        done.recv_timeout(STUCK).unwrap();

        assert_eq!(
            *queued.lock().unwrap(),
            ["reduce on 0", "reduce on 1", "send", "receive"]
        );
    }

    #[test]
    fn a_transfer_between_other_devices_does_not_wait() {
        let order = CollectiveOrder::new();
        let (transferred, done) = mpsc::channel();
        order.all_reduce(device(0), alloc::vec![device(0), device(1)], |_| ());

        std::thread::spawn(move || {
            order.transfer(device(2), device(3), || (), || ());
            transferred.send(()).unwrap();
        });

        done.recv_timeout(STUCK).unwrap();
    }

    #[test]
    fn a_transfer_does_not_wait_for_a_part_still_queueing() {
        let order = Arc::new(CollectiveOrder::new());
        let (transferred, done) = mpsc::channel();

        order.all_reduce(device(0), alloc::vec![device(0)], |_| {
            let order = order.clone();
            std::thread::spawn(move || {
                order.transfer(device(1), device(2), || (), || ());
                transferred.send(()).unwrap();
            });
            done.recv_timeout(STUCK).unwrap();
        });
    }

    fn device(index_id: u16) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id,
        }
    }
}
