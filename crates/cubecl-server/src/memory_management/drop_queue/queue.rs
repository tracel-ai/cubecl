use crate::memory_management::drop_queue::{DeviceWaiter, FlushingPolicy, policy::FlushTracker};
use crate::server::BufferBinding;
use alloc::vec::Vec;
use cubecl_environment::bytes::Bytes;

/// Defers the drop of host [`Bytes`] and of memory of other devices until the
/// device has finished reading them.
///
/// # How it works
///
/// The device uploads are asynchronous: after you copy bytes into a staging buffer
/// and enqueue an upload command, the host memory must remain valid until the
/// device is done. The queue manages this lifetime in two phases:
///
/// 1. **Stage** – what is handed over in flight joins the current batch.
/// 2. **Flush** – the batch before is freed once its fence signals, and the
///    current one becomes pending behind a new fence marking the end of it.
///
/// This double-buffer scheme means host memory is held for at most two flush
/// cycles, while avoiding any unnecessary stalls on the hot path.
///
/// # Guarantees
///
/// Nothing is freed before the fence of the batch it was staged in signals,
/// or while a fence's wait failed on a device that is not poisoned: it is
/// leaked then, since the device may still be reading it.
pub struct PendingDropQueue<F: DeviceWaiter> {
    flusher: DropFlusher<F>,
}

impl<F: DeviceWaiter> PendingDropQueue<F> {
    /// An empty queue that asks to be flushed as `policy` says.
    pub fn new(policy: FlushingPolicy) -> Self {
        Self {
            flusher: DropFlusher::new(policy),
        }
    }
}

impl<F: DeviceWaiter> PendingDropQueue<F> {
    /// Keep `bytes` until the device has finished reading them: they join the
    /// current batch, freed on the flush cycle *after* the next flush.
    ///
    /// Crate-visible, like every mutation of the queue: the only paths allowed
    /// to move it are the shared [`Command`](crate::command::Command) and
    /// [`Window`](crate::command::Window), which carry the capture-deferral
    /// rule a backend touching the queue directly would have to remember.
    pub(crate) fn push(&mut self, bytes: Bytes) {
        self.flusher.push(bytes);
    }

    /// Keep `binding`, memory of another device, allocated and in place
    /// until the device finished the work queued so far, the kernel that
    /// binds it included.
    ///
    /// It counts toward the policy as an entry and not by its size: holding it
    /// costs this device nothing, and a size-driven flush after each large
    /// peer buffer would make every such launch wait on the one before. A
    /// sync drains whatever peer memory is still held, so it does not stay
    /// pinned on its owner for as long as this stream idles.
    pub(crate) fn hold_peer(&mut self, binding: BufferBinding) {
        self.flusher.hold_peer(binding);
    }

    /// Whether the queue holds memory of another device.
    pub(crate) fn holds_peers(&self) -> bool {
        self.flusher.holds_peers()
    }

    /// Whether the staged batch has reached the policy's thresholds.
    pub(crate) fn should_flush(&self) -> bool {
        self.flusher.should_flush()
    }

    /// Flush until nothing is held back.
    ///
    /// One [`flush`](Self::flush) frees the batch staged two cycles ago and
    /// rotates the current one into pending, so what was just dropped is still
    /// held when it returns. A caller that needs the memory *now* — an
    /// explicit cleanup, a capture window about to open onto pools that may
    /// not allocate — wants both rotations.
    pub(crate) fn drain<Factory: Fn() -> F>(&mut self, factory: Factory) {
        self.flusher.flush(&factory);
        self.flusher.flush(&factory);
    }

    /// Free the batch the device is done with, and make the current one
    /// pending behind a fence `factory` produces. `factory` submits (or
    /// records) a device signal, so that syncing the fence guarantees all
    /// preceding device work is complete.
    ///
    /// An idle queue mints no fence: nothing is held, so there is nothing a
    /// fence would protect — and the fence is not free: it is recorded on the
    /// stream, which an open capture window cannot tolerate.
    pub(crate) fn flush<Factory: Fn() -> F>(&mut self, factory: Factory) {
        self.flusher.flush(factory);
    }
}

impl<F: DeviceWaiter> core::fmt::Debug for PendingDropQueue<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PendingDropQueue")
            .field(
                "pending",
                &self.flusher.pending.as_ref().map(|(_, held)| held),
            )
            .field("staged", &self.flusher.staged)
            .field("policy", &self.flusher.policy)
            .field("tracker", &self.flusher.tracker)
            .finish()
    }
}

impl<F: DeviceWaiter> Default for PendingDropQueue<F> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

/// Flushes a [`PendingDropQueue`]: holds each batch until the fence that
/// follows it signals.
struct DropFlusher<F: DeviceWaiter> {
    /// What the previous flush cycle held, in the order it was staged
    /// (Arrival), with the fence that signals the device consumed it.
    pending: Option<(F, Vec<HeldMemory>)>,
    /// What the current cycle holds, not yet behind a fence, in the order it
    /// was staged (Arrival).
    staged: Vec<HeldMemory>,
    policy: FlushingPolicy,
    tracker: FlushTracker,
}

/// What the queue keeps alive until the device is done with it.
#[derive(Debug)]
enum HeldMemory {
    /// Host memory an upload reads.
    Bytes(#[allow(unused)] Bytes),
    /// Memory of another device a kernel binds: held, it stays allocated and
    /// in place there.
    Peer(#[allow(unused)] BufferBinding),
}

impl<F: DeviceWaiter> DropFlusher<F> {
    fn new(policy: FlushingPolicy) -> Self {
        Self {
            pending: None,
            staged: Vec::new(),
            policy,
            tracker: FlushTracker::default(),
        }
    }

    fn push(&mut self, bytes: Bytes) {
        self.tracker.track_staged_bytes(bytes.len());
        self.staged.push(HeldMemory::Bytes(bytes));
    }

    fn hold_peer(&mut self, binding: BufferBinding) {
        self.tracker.track_staged_bytes(0);
        self.staged.push(HeldMemory::Peer(binding));
    }

    fn holds_peers(&self) -> bool {
        self.staged
            .iter()
            .chain(self.pending.iter().flat_map(|(_, held)| held))
            .any(|held| matches!(held, HeldMemory::Peer(_)))
    }

    fn should_flush(&self) -> bool {
        self.tracker.should_flush(&self.policy)
    }

    fn flush<Factory: Fn() -> F>(&mut self, factory: Factory) {
        if self.pending.is_none() && self.staged.is_empty() {
            return;
        }
        let mut batch = self
            .pending
            .take()
            .map_or_else(Vec::new, |(fence, held)| retire(held, fence.sync()));
        batch.append(&mut self.staged);
        self.pending = Some((factory(), batch));
        self.tracker.reset();
    }
}

/// Free `held` when the device is done with it, or leak it when the device
/// may still be reading it, and answer an empty batch to stage into next.
fn retire(mut held: Vec<HeldMemory>, device_done: bool) -> Vec<HeldMemory> {
    if !device_done {
        core::mem::forget(held);
        return Vec::new();
    }
    held.clear();
    held
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerError;
    use alloc::vec;
    use core::cell::Cell;

    #[derive(Clone)]
    struct MockFence<'a> {
        sync_count: &'a Cell<u32>,
    }

    impl DeviceWaiter for MockFence<'_> {
        fn wait(self) -> Result<(), ServerError> {
            self.sync_count.set(self.sync_count.get() + 1);
            Ok(())
        }
    }

    fn make_queue<'a>(
        sync_count: &'a Cell<u32>,
    ) -> (
        PendingDropQueue<MockFence<'a>>,
        impl Fn() -> MockFence<'a> + 'a,
    ) {
        let queue = PendingDropQueue::new(test_policy());
        let factory = move || MockFence { sync_count };
        (queue, factory)
    }

    fn sample_bytes() -> Bytes {
        Bytes::from_elems(vec![1u8, 2, 3])
    }

    fn test_policy() -> FlushingPolicy {
        FlushingPolicy {
            max_staged_count: 2048,
            max_staged_bytes: 8,
        }
    }

    fn pending_len(queue: &PendingDropQueue<MockFence<'_>>) -> Option<usize> {
        queue.flusher.pending.as_ref().map(|(_, held)| held.len())
    }

    #[test]
    fn push_at_count_threshold_triggers_flush_hint() {
        let sync_count = Cell::new(0u32);
        let (mut queue, _factory) = make_queue(&sync_count);

        for _ in 0..test_policy().max_staged_count {
            queue.push(sample_bytes());
        }

        assert!(queue.should_flush());
    }

    #[test]
    fn first_flush_creates_fence_without_syncing() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        queue.push(sample_bytes());
        queue.flush(&factory);

        // The fence is created but must not be synced yet — that happens on
        // the next flush.
        assert_eq!(
            sync_count.get(),
            0,
            "fence should not be synced on first flush"
        );
    }

    #[test]
    fn each_subsequent_flush_syncs_the_previous_fence() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..10 {
            queue.push(sample_bytes());
            queue.flush(&factory);
        }

        // Each flush except the first syncs the fence from the previous one.
        assert_eq!(sync_count.get(), 9);
    }

    #[test]
    fn pending_holds_previously_staged_bytes_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..5 {
            queue.push(sample_bytes());
        }
        queue.flush(&factory);

        assert!(queue.flusher.staged.is_empty());
        assert_eq!(pending_len(&queue), Some(5));
    }

    #[test]
    fn pending_is_replaced_on_second_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..5 {
            queue.push(sample_bytes());
        }
        queue.flush(&factory); // pending = 5 items

        queue.push(sample_bytes());
        queue.flush(&factory); // syncs fence → pending cleared, rotated

        // Only the one item staged between the two flushes should be pending.
        assert_eq!(pending_len(&queue), Some(1));
    }

    #[test]
    fn should_flush_resets_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..test_policy().max_staged_count {
            queue.push(sample_bytes());
        }

        queue.flush(&factory);

        assert!(
            !queue.should_flush(),
            "policy state should be reset after flush"
        );
    }

    #[test]
    fn flush_on_empty_queue_mints_no_fence() {
        let sync_count = Cell::new(0u32);
        let (mut queue, _factory) = make_queue(&sync_count);

        queue.flush(|| panic!("an idle queue should mint no fence"));

        assert_eq!(pending_len(&queue), None);
    }

    /// Two flushes free everything held, and a third finds nothing to do.
    #[test]
    fn a_drain_frees_everything_held() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);
        queue.push(sample_bytes());

        queue.drain(&factory);

        assert_eq!(pending_len(&queue), Some(0));
        assert_eq!(sync_count.get(), 1);
    }

    /// Memory of another device is held, and counts toward the policy as an
    /// entry and not by its size.
    #[test]
    fn peer_memory_is_held_as_an_entry() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);
        assert!(!queue.holds_peers());

        let service =
            cubecl_common::device::ServiceId::of::<()>(cubecl_common::device::DeviceId::new(0, 0));
        let stream = cubecl_environment::stream::StreamId { value: 1 };
        queue.hold_peer(crate::server::Handle::new(service, stream, 8).binding());

        assert!(queue.holds_peers());
        assert!(!queue.should_flush(), "a peer buffer weighs nothing");
        queue.flush(&factory);
        assert!(queue.holds_peers(), "still held behind its fence");
    }
}
