/// The thresholds past which a [`PendingDropQueue`](super::PendingDropQueue)
/// asks to be flushed.
///
/// A flush is wanted when **either** limit is reached — whichever comes
/// first. A limit of `u32::MAX` is never reached.
#[derive(Debug)]
pub struct FlushingPolicy {
    /// Flush when this many allocations have been staged.
    pub max_staged_count: u32,
    /// Flush when the staged allocations total this many bytes.
    pub max_staged_bytes: u32,
}

impl Default for FlushingPolicy {
    fn default() -> Self {
        Self {
            max_staged_count: 64,
            max_staged_bytes: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

/// Tracks what is staged and evaluates it against a [`FlushingPolicy`].
#[derive(Default, Debug)]
pub(crate) struct FlushTracker {
    staged_count: u32,
    staged_bytes: u32,
}

impl FlushTracker {
    /// Count a newly staged allocation of `staged_bytes`, saturating at the
    /// largest count.
    pub(crate) fn track_staged_bytes(&mut self, staged_bytes: usize) {
        self.staged_count = self.staged_count.saturating_add(1);
        self.staged_bytes = self
            .staged_bytes
            .saturating_add(u32::try_from(staged_bytes).unwrap_or(u32::MAX));
    }

    /// Forget everything staged, once a flush took it.
    pub(crate) fn reset(&mut self) {
        self.staged_count = 0;
        self.staged_bytes = 0;
    }

    /// Whether either threshold of `policy` has been reached.
    pub(crate) fn should_flush(&self, policy: &FlushingPolicy) -> bool {
        self.staged_count >= policy.max_staged_count || self.staged_bytes >= policy.max_staged_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> FlushingPolicy {
        FlushingPolicy {
            max_staged_count: 4,
            max_staged_bytes: 100,
        }
    }

    #[test]
    fn no_flush_when_below_both_thresholds() {
        let tracker = FlushTracker::default();
        assert!(!tracker.should_flush(&policy()));
    }

    #[test]
    fn flush_when_count_threshold_reached() {
        let mut tracker = FlushTracker::default();
        for _ in 0..4 {
            tracker.track_staged_bytes(1);
        }
        assert!(tracker.should_flush(&policy()));
    }

    #[test]
    fn flush_triggered_by_whichever_limit_comes_first() {
        let mut tracker = FlushTracker::default();
        // Only 2 allocations but already over the size limit.
        tracker.track_staged_bytes(60);
        tracker.track_staged_bytes(60);
        assert!(tracker.should_flush(&policy()));
    }

    #[test]
    fn register_saturates_instead_of_overflowing() {
        // Regression test for #1359: staging allocations without an intervening
        // reset must not panic with "attempt to add with overflow".
        let mut tracker = FlushTracker {
            staged_count: u32::MAX,
            staged_bytes: u32::MAX - 1,
        };
        tracker.track_staged_bytes(8);
        assert_eq!(tracker.staged_count, u32::MAX);
        assert_eq!(tracker.staged_bytes, u32::MAX);
    }

    #[test]
    fn a_size_past_a_u32_counts_as_the_largest() {
        let mut tracker = FlushTracker::default();
        tracker.track_staged_bytes(usize::MAX);
        assert_eq!(tracker.staged_bytes, u32::MAX);
    }

    #[test]
    fn reset_clears_state() {
        let mut tracker = FlushTracker::default();
        for _ in 0..4 {
            tracker.track_staged_bytes(1);
        }
        tracker.reset();
        assert!(!tracker.should_flush(&policy()));
    }
}
