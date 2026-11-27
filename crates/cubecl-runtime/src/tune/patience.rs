use alloc::vec::Vec;
use core::time::Duration;

use cubecl_environment::collections::HashMap;

use crate::tune::Patience;
use crate::tune::group::GroupId;
use crate::tune::plan::Batch;
use crate::tune::sampler::TimeChange;

/// Tracks how every group with a [`Patience`] in a [`Batch`] fares, through the groups that
/// brought each slot of the batch in.
#[derive(Debug)]
pub struct PatienceTracker {
    /// The groups that brought in each slot, by slot (Position Usage).
    groups: Vec<Vec<GroupId>>,
    /// The tracker of each group with a patience, by group (Keyed Usage).
    trackers: HashMap<GroupId, GroupTracker>,
}

impl PatienceTracker {
    pub fn new(batch: &Batch) -> Self {
        Self {
            groups: batch
                .entries
                .iter()
                .map(|entry| entry.groups.clone())
                .collect(),
            trackers: batch
                .patience
                .iter()
                .map(|(group, patience)| (*group, GroupTracker::new(*patience)))
                .collect(),
        }
    }

    /// Count the best time of the candidate at `slot` against each of the groups that brought it
    /// in.
    pub fn track(&mut self, slot: usize, time: Duration) {
        for group in &self.groups[slot] {
            if let Some(tracker) = self.trackers.get_mut(group) {
                tracker.track(time);
            }
        }
    }

    /// The most members any group's patience measures before it may stop, and at least one: the
    /// candidates the batch measures before the bounds' time limit may end it.
    pub fn min_measured_count(&self) -> usize {
        self.trackers
            .values()
            .map(|tracker| tracker.patience.min_measured_count)
            .max()
            .unwrap_or(0)
            .max(1)
    }

    /// Whether the candidate at `slot` is skipped: every group that brought it in ran out of
    /// patience. One that a group without patience brought in, or that no group did, never is.
    pub fn skips_slot(&self, slot: usize) -> bool {
        let groups = &self.groups[slot];
        !groups.is_empty()
            && groups.iter().all(|group| {
                self.trackers
                    .get(group)
                    .is_some_and(GroupTracker::exhausted)
            })
    }
}

/// Tracks how one group fares against its [`Patience`]: how many of its members were measured,
/// and how many in a row failed to improve on its leader.
#[derive(Debug)]
struct GroupTracker {
    patience: Patience,
    measured_count: usize,
    miss_count: usize,
    /// The last time that was an improvement.
    leader: Option<Duration>,
}

impl GroupTracker {
    fn new(patience: Patience) -> Self {
        Self {
            patience,
            measured_count: 0,
            miss_count: 0,
            leader: None,
        }
    }

    /// Count one measured member by its best time.
    fn track(&mut self, time: Duration) {
        self.measured_count += 1;

        match self.leader.map(|leader| TimeChange::of(time, leader)) {
            None | Some(TimeChange::Improvement) => self.lead(time),
            Some(TimeChange::Neutral | TimeChange::Regression) => self.miss_count += 1,
        }
    }

    /// Make `time` the group's leader.
    fn lead(&mut self, time: Duration) {
        self.leader = Some(time);
        self.miss_count = 0;
    }

    /// Whether the group stops: one member is always measured, whatever the patience says.
    fn exhausted(&self) -> bool {
        self.measured_count >= self.patience.min_measured_count.max(1)
            && self.miss_count >= self.patience.max_miss_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::plan::BatchEntry;

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
    fn a_neutral_time_is_a_miss_and_keeps_the_leader() {
        // 99 is within the threshold of 100 and misses, and 97 improves on 100 and resets the
        // misses. 990 and 985 both miss against 1000, which a leader moved to 990 would not
        // have counted.
        assert_eq!(stops_after(patience(1, 2), &[100, 99, 97, 96]), None);
        assert_eq!(stops_after(patience(1, 2), &[1000, 990, 985]), Some(3));
    }

    #[test]
    fn patience_holds_while_members_keep_improving() {
        assert_eq!(stops_after(patience(1, 1), &[100, 90, 80, 70]), None);
    }

    #[test]
    fn a_zero_patience_still_measures_one_member() {
        let tracker = GroupTracker::new(patience(0, 0));
        assert!(!tracker.exhausted());

        assert_eq!(stops_after(patience(0, 0), &[100]), Some(1));
    }

    fn batch(groups: &[&[GroupId]], patience: &[(GroupId, Patience)]) -> Batch {
        Batch {
            entries: groups
                .iter()
                .enumerate()
                .map(|(index, groups)| BatchEntry {
                    index,
                    groups: groups.to_vec(),
                    priority: None,
                })
                .collect(),
            patience: patience.iter().copied().collect(),
        }
    }

    #[test]
    fn the_batch_minimum_is_its_most_patient_group_s() {
        let tracker = PatienceTracker::new(&batch(
            &[&[GroupId(0)], &[GroupId(1)]],
            &[(GroupId(0), patience(2, 1)), (GroupId(1), patience(5, 1))],
        ));

        assert_eq!(tracker.min_measured_count(), 5);
    }

    #[test]
    fn a_batch_without_patience_measures_one_before_it_may_stop() {
        let tracker = PatienceTracker::new(&batch(&[&[]], &[]));

        assert_eq!(tracker.min_measured_count(), 1);
    }

    #[test]
    fn a_group_out_of_patience_skips_only_what_it_alone_brought_in() {
        let (spent, patient, no_patience) = (GroupId(0), GroupId(1), GroupId(2));
        let mut tracker = PatienceTracker::new(&batch(
            &[
                &[spent],
                &[patient],
                &[spent],
                &[patient],
                &[spent],
                &[patient],
                &[spent, patient],
                &[spent, no_patience],
                &[],
            ],
            &[(spent, patience(1, 1)), (patient, patience(1, 1))],
        ));

        // Interleaved: `spent` misses on its second member, `patient` keeps improving.
        tracker.track(0, Duration::from_micros(100));
        tracker.track(1, Duration::from_micros(100));
        tracker.track(2, Duration::from_micros(120));
        tracker.track(3, Duration::from_micros(80));

        assert!(tracker.skips_slot(4));
        assert!(!tracker.skips_slot(5));
        assert!(!tracker.skips_slot(6));
        assert!(!tracker.skips_slot(7));
        assert!(!tracker.skips_slot(8));
    }
}
