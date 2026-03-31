use cubecl_common::bytes::Bytes;

/// Defines the thresholds that determine when a [`PendingDropQueue`] should be
/// flushed.
///
/// A flush is triggered when **any** limit is exceeded — whichever comes first.
/// Set a field to its type's `MAX` to effectively disable that particular threshold.
///
/// # Relationship with `DevicePtrStaging`
///
/// `max_check_count` controls how many kernel launches can occur between flushes.
/// The [`DevicePtrStaging`] ring buffer is sized to
/// `max_bindings_per_kernel × max_check_count × 2` — the `× 2` accounts for the
/// double-buffer in [`PendingDropQueue`]: after a flush, kernels from **two**
/// consecutive cycles (the still-pending batch and the newly-staged batch) may
/// reference slots simultaneously. **Changing `max_check_count` without updating
/// the `DevicePtrStaging` capacity (or vice versa) will break this invariant and
/// may cause use-after-free on the device.**
///
/// [`PendingDropQueue`]: super::PendingDropQueue
///
/// [`DevicePtrStaging`]: crate::memory_management::DevicePtrStaging
#[derive(Debug)]
pub struct FlushingPolicy {
    /// Flush when this many allocations have been staged.
    pub max_bytes_count: u16,
    /// Flush when this many calls to [`FlushingPolicyState::should_flush`] have been made.
    ///
    /// Each call corresponds to one kernel launch, so this value caps the number of
    /// kernels that can be in-flight before a fence is inserted. It **must** match the
    /// `max_queue_depth` parameter passed to [`DevicePtrStaging::new`] on the same
    /// stream.
    ///
    /// [`DevicePtrStaging::new`]: crate::memory_management::DevicePtrStaging::new
    pub max_check_count: u16,
    /// Flush when the total staged size reaches this many bytes.
    pub max_bytes_size: u32,
}

impl Default for FlushingPolicy {
    fn default() -> Self {
        Self {
            max_bytes_count: 64,
            max_bytes_size: 64 * 1024 * 1024, // 64 MiB
            max_check_count: 128,
        }
    }
}

/// Tracks staged allocations and evaluates them against a [`FlushingPolicy`].
#[derive(Default, Debug)]
pub(crate) struct FlushingPolicyState {
    bytes_count: u16,
    check_count: u16,
    bytes_size: u32,
}

impl FlushingPolicyState {
    /// Record a newly staged [`Bytes`] allocation.
    pub(crate) fn register(&mut self, bytes: &Bytes) {
        self.bytes_count += 1;
        self.bytes_size += bytes.len() as u32;
    }

    /// Reset all counters, typically called after a flush.
    pub(crate) fn reset(&mut self) {
        self.bytes_count = 0;
        self.bytes_size = 0;
        self.check_count = 0;
    }

    /// Returns `true` if either threshold in `policy` has been reached.
    pub(crate) fn should_flush(&mut self, policy: &FlushingPolicy) -> bool {
        self.check_count += 1;

        self.check_count >= policy.max_check_count
            || self.bytes_count >= policy.max_bytes_count
            || self.bytes_size >= policy.max_bytes_size
    }
}

#[cfg(test)]
mod policy_tests {
    use std::vec;

    use super::*;

    fn policy() -> FlushingPolicy {
        FlushingPolicy {
            max_bytes_count: 4,
            max_bytes_size: 100,
            max_check_count: 16,
        }
    }

    fn state() -> FlushingPolicyState {
        FlushingPolicyState {
            bytes_count: 0,
            check_count: 0,
            bytes_size: 0,
        }
    }

    #[test]
    fn no_flush_when_below_all_thresholds() {
        let mut s = state();
        assert!(!s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_bytes_count_threshold_reached() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_bytes_size_threshold_reached() {
        let mut s = state();
        s.register(&Bytes::from_elems(vec![0u8; 101]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_check_count_threshold_reached() {
        let mut s = state();
        let p = policy();
        // Call should_flush enough times to hit max_check_count.
        for _ in 0..(p.max_check_count - 1) {
            assert!(!s.should_flush(&p));
        }
        assert!(s.should_flush(&p));
    }

    #[test]
    fn flush_triggered_by_whichever_limit_comes_first() {
        let mut s = state();
        // Only 2 allocations but already over the size limit.
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn reset_clears_all_state() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
        s.reset();
        assert!(!s.should_flush(&policy()));
    }
}
