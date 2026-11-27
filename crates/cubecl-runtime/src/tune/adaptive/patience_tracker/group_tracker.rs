use core::time::Duration;

use crate::tune::Patience;
use crate::tune::adaptive::sample_keeper::TimeChange;

/// Tracks how one group fares against its [`Patience`]: how many of its members were measured,
/// and how many in a row failed to improve on its leader.
#[derive(Debug)]
pub struct GroupTracker {
    patience: Patience,
    measured_count: usize,
    miss_count: usize,
    /// The last time that was an improvement.
    leader: Option<Duration>,
}

impl GroupTracker {
    pub fn new(patience: Patience) -> Self {
        Self {
            patience,
            measured_count: 0,
            miss_count: 0,
            leader: None,
        }
    }

    /// Count one measured member by its best time.
    pub fn track(&mut self, time: Duration) {
        self.measured_count += 1;

        match self.leader.map(|leader| TimeChange::of(time, leader)) {
            None | Some(TimeChange::Improvement) => self.lead(time),
            Some(TimeChange::Neutral | TimeChange::Regression) => self.miss_count += 1,
        }
    }

    /// The members its patience measures before the group may stop.
    pub fn min_measured_count(&self) -> usize {
        self.patience.min_measured_count
    }

    /// Whether the group stops: one member is always measured, whatever the patience says.
    pub fn exhausted(&self) -> bool {
        self.measured_count >= self.patience.min_measured_count.max(1)
            && self.miss_count >= self.patience.max_miss_count
    }
}

impl GroupTracker {
    /// `time` leads the group from now on, and its misses count afresh.
    fn lead(&mut self, time: Duration) {
        self.leader = Some(time);
        self.miss_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds each member's best time in microseconds to a fresh [`GroupTracker`], returning how
    /// many were measured before patience ran out, if it did.
    fn stops_after(patience: Patience, times: &[u64]) -> Option<usize> {
        let mut tracker = GroupTracker::new(patience);
        times
            .iter()
            .position(|time| {
                tracker.track(Duration::from_micros(*time));
                tracker.exhausted()
            })
            .map(|slot| slot + 1)
    }

    fn patience(min_measured_count: usize, max_miss_count: usize) -> Patience {
        Patience {
            min_measured_count,
            max_miss_count,
        }
    }

    #[test]
    fn patience_runs_out_after_the_misses_in_a_row() {
        assert_eq!(
            stops_after(patience(2, 2), &[100, 120, 90, 95, 100, 80]),
            Some(5)
        );
    }

    #[test]
    fn patience_waits_for_the_minimum_measured() {
        assert_eq!(
            stops_after(patience(5, 2), &[100, 120, 130, 140, 150, 160]),
            Some(5)
        );
    }

    #[test]
    fn a_neutral_time_is_a_miss() {
        // 99 is within the threshold of 100 and misses, and 97 improves on 100 and resets the
        // misses.
        assert_eq!(stops_after(patience(1, 2), &[100, 99, 97, 96]), None);
    }

    #[test]
    fn a_neutral_time_keeps_the_leader() {
        // 990 and 985 both miss against 1000, which a leader moved to 990 would not have counted.
        assert_eq!(stops_after(patience(1, 2), &[1000, 990, 985]), Some(3));
    }

    #[test]
    fn patience_holds_while_members_keep_improving() {
        assert_eq!(stops_after(patience(1, 1), &[100, 90, 80, 70]), None);
    }

    #[test]
    fn a_zero_patience_holds_before_any_member_is_measured() {
        assert!(!GroupTracker::new(patience(0, 0)).exhausted());
    }

    #[test]
    fn a_zero_patience_stops_after_one_member() {
        assert_eq!(stops_after(patience(0, 0), &[100]), Some(1));
    }
}
