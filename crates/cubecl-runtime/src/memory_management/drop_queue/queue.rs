use alloc::vec::Vec;
use cubecl_common::bytes::Bytes;

use crate::memory_management::{
    drop_queue::FlushingPolicy, drop_queue::policy::FlushingPolicyState,
};

/// A synchronization primitive that blocks until the device has finished
/// processing all commands submitted before the fence was created.
pub trait Fence {
    /// Block the current thread until the signals this fence.
    fn sync(self);
}

/// Defers the drop of CPU-side [`Bytes`] allocations until the device has
/// finished reading them.
///
/// # How it works
///
/// The device uploads are asynchronous: after you copy bytes into a staging buffer
/// and enqueue an upload command, the CPU memory must remain valid until the
/// device is done. `PendingDropQueue` manages this lifetime with a two-phase
/// approach:
///
/// 1. **Stage** – call [`push`](Self::push) to hand over bytes that are
///    in-flight. They land in the `staged` list.
/// 2. **Flush** – call [`flush`](Self::flush) to rotate the lists. The
///    previously staged bytes move to `pending`, a new [`Fence`] is created
///    to mark the end of the current upload batch, and any bytes that were
///    *already* pending (i.e. the batch before that) are freed after syncing
///    the previous fence.
///
/// This double-buffer scheme means CPU memory is held for at most two flush
/// cycles, while avoiding any unnecessary stalls on the hot path.
///
/// # Flushing policy
///
/// Call [`should_flush`](Self::should_flush) to check whether enough bytes
/// have accumulated to warrant a flush. You may also flush unconditionally
/// (e.g. at the end of a frame).
pub struct PendingDropQueue<E: Fence> {
    /// Fence signalling that the device has consumed everything in `pending`.
    fence: Option<E>,
    /// Bytes from the *previous* flush cycle, kept alive until `event` fires.
    pending: Vec<Bytes>,
    /// Bytes queued in the *current* cycle, not yet associated with a fence.
    staged: Vec<Bytes>,
    /// The configuration of the queue.
    policy: FlushingPolicy,
    /// The current state of the policy.
    policy_state: FlushingPolicyState,
}

impl<E: Fence> core::fmt::Debug for PendingDropQueue<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PendingDropQueue")
            .field("pending", &self.pending)
            .field("staged", &self.staged)
            .field("policy", &self.policy)
            .field("policy_state", &self.policy_state)
            .finish()
    }
}

impl<E: Fence> Default for PendingDropQueue<E> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<F: Fence> PendingDropQueue<F> {
    /// Creates a new `PendingDropQueue`.
    pub fn new(policy: FlushingPolicy) -> Self {
        Self {
            fence: None,
            pending: Vec::new(),
            staged: Vec::new(),
            policy,
            policy_state: Default::default(),
        }
    }
    /// Enqueue `bytes` to be dropped once the device has finished reading them.
    ///
    /// The bytes are added to the current staged batch and will be freed on
    /// the flush cycle *after* the next call to [`flush`](Self::flush).
    pub fn push(&mut self, bytes: Bytes) {
        self.policy_state.register(&bytes);
        self.staged.push(bytes);
    }

    /// Returns `true` when the staged batch is large enough to justify a
    /// flush.
    pub fn should_flush(&self) -> bool {
        self.policy_state.should_flush(&self.policy)
    }

    /// Rotate the double-buffer and free any memory the device is done with.
    ///
    /// `factory` is called to produce a [`Fence`]. It should submit (or
    /// record) a device signal command so that syncing the fence guarantees all
    /// preceding device work is complete.
    pub fn flush<Factory: Fn() -> F>(&mut self, factory: Factory) {
        // Sync the fence from the previous flush and free the bytes it was
        // protecting.
        if let Some(event) = self.fence.take() {
            event.sync();
            self.pending.clear();
        }

        // Safety net: if pending is somehow still populated (no prior fence),
        // stall immediately rather than freeing memory the GPU might still
        // be reading.
        if !self.pending.is_empty() {
            let event = factory();
            event.sync();
            self.pending.clear();
        }

        // The current staged batch becomes the new pending batch.
        core::mem::swap(&mut self.pending, &mut self.staged);

        // Record a fence so the *next* flush knows when this batch is safe to
        // free.
        self.fence = Some(factory());
        self.policy_state.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::cell::Cell;

    // ---------------------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    struct MockFence<'a> {
        sync_count: &'a Cell<u32>,
    }

    impl Fence for MockFence<'_> {
        fn sync(self) {
            self.sync_count.set(self.sync_count.get() + 1);
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
            max_bytes_count: 2048,
            max_bytes_size: 8,
        }
    }

    // ---------------------------------------------------------------------------
    // push / should_flush
    // ---------------------------------------------------------------------------

    #[test]
    fn push_at_count_threshold_triggers_flush_hint() {
        let sync_count = Cell::new(0u32);
        let (mut queue, _factory) = make_queue(&sync_count);

        for _ in 0..test_policy().max_bytes_count {
            queue.push(sample_bytes());
        }

        assert!(queue.should_flush());
    }

    #[test]
    fn push_large_allocation_triggers_flush_via_size_threshold() {
        let sync_count = Cell::new(0u32);
        let (mut queue, _factory) = make_queue(&sync_count);
        let big = Bytes::from_elems(vec![0u8; test_policy().max_bytes_size as usize + 1]);

        queue.push(big);

        assert!(queue.should_flush());
    }

    // ---------------------------------------------------------------------------
    // flush – fence / sync behaviour
    // ---------------------------------------------------------------------------

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
    fn second_flush_syncs_fence_from_first_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        queue.push(sample_bytes());
        queue.flush(&factory); // flush 1 – creates fence A

        queue.push(sample_bytes());
        queue.flush(&factory); // flush 2 – syncs fence A, creates fence B

        assert_eq!(sync_count.get(), 1, "exactly one sync after two flushes");
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

    // ---------------------------------------------------------------------------
    // flush – buffer rotation
    // ---------------------------------------------------------------------------

    #[test]
    fn staged_is_empty_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..5 {
            queue.push(sample_bytes());
        }
        queue.flush(&factory);

        assert!(queue.staged.is_empty());
    }

    #[test]
    fn pending_holds_previously_staged_bytes_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..5 {
            queue.push(sample_bytes());
        }
        queue.flush(&factory);

        assert_eq!(queue.pending.len(), 5);
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
        assert_eq!(queue.pending.len(), 1);
    }

    // ---------------------------------------------------------------------------
    // flush – policy state reset
    // ---------------------------------------------------------------------------

    #[test]
    fn should_flush_resets_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..test_policy().max_bytes_count {
            queue.push(sample_bytes());
        }
        assert!(queue.should_flush());

        queue.flush(&factory);

        assert!(
            !queue.should_flush(),
            "policy state should be reset after flush"
        );
    }

    // ---------------------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------------------

    #[test]
    fn flush_on_empty_queue_is_safe() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        // Should not panic regardless of how many times it is called.
        queue.flush(&factory);
        queue.flush(&factory);
        queue.flush(&factory);
    }
}
