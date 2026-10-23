use alloc::vec::Vec;
use core::time::Duration;

use cubecl_environment::collections::HashMap;

use crate::tune::sampler::TimeChange;
use crate::tune::{Batch, GroupId, Patience};

/// The [`PatienceTracker`] of every group with a [`Patience`] in a [`Batch`], and the groups
/// that brought each slot of the batch in.
#[derive(Debug)]
pub(crate) struct PatienceTable {
    groups: Vec<Vec<GroupId>>,
    trackers: HashMap<GroupId, PatienceTracker>,
}

impl PatienceTable {
    pub(crate) fn new(batch: &Batch) -> Self {
        Self {
            groups: batch
                .entries
                .iter()
                .map(|entry| entry.groups.clone())
                .collect(),
            trackers: batch
                .patience
                .iter()
                .map(|(group, patience)| (*group, PatienceTracker::new(*patience)))
                .collect(),
        }
    }

    /// Count the best time of the candidate at `slot` against each of the groups that brought
    /// it in.
    pub(crate) fn record(&mut self, slot: usize, time: Duration) {
        for group in &self.groups[slot] {
            if let Some(tracker) = self.trackers.get_mut(group) {
                tracker.record(time);
            }
        }
    }

    /// Whether the candidate at `slot` is skipped: every group that brought it in ran out of
    /// patience. One that a group without patience brought in, or that no group did, never is.
    pub(crate) fn skips(&self, slot: usize) -> bool {
        let groups = &self.groups[slot];
        !groups.is_empty()
            && groups.iter().all(|group| {
                self.trackers
                    .get(group)
                    .is_some_and(PatienceTracker::exhausted)
            })
    }
}

/// How one group fares against its [`Patience`]: how many of its members were measured, and
/// how many in a row failed to improve on its leader.
#[derive(Debug)]
struct PatienceTracker {
    patience: Patience,
    measured: usize,
    misses: usize,
    /// The last time that was an improvement.
    leader: Option<Duration>,
}

impl PatienceTracker {
    fn new(patience: Patience) -> Self {
        Self {
            patience,
            measured: 0,
            misses: 0,
            leader: None,
        }
    }

    /// Count one measured member by its best time.
    fn record(&mut self, time: Duration) {
        self.measured += 1;

        match self.leader.map(|leader| TimeChange::of(time, leader)) {
            None | Some(TimeChange::Improvement) => {
                self.leader = Some(time);
                self.misses = 0;
            }
            Some(TimeChange::Neutral | TimeChange::Regression) => self.misses += 1,
        }
    }

    /// Whether the group stops: one member is always measured, whatever the patience says.
    fn exhausted(&self) -> bool {
        self.measured >= self.patience.min_measured.max(1)
            && self.misses >= self.patience.max_misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::BatchEntry;

    /// Feeds each member's best time in microseconds to a fresh [`PatienceTracker`],
    /// returning how many were measured before patience ran out, if it did.
    fn stops_after(patience: Patience, times: &[u64]) -> Option<usize> {
        let mut tracker = PatienceTracker::new(patience);
        times
            .iter()
            .position(|time| {
                tracker.record(Duration::from_micros(*time));
                tracker.exhausted()
            })
            .map(|slot| slot + 1)
    }

    fn patience(min_measured: usize, max_misses: usize) -> Patience {
        Patience {
            min_measured,
            max_misses,
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
        let tracker = PatienceTracker::new(patience(0, 0));
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
                })
                .collect(),
            patience: patience.iter().copied().collect(),
        }
    }

    #[test]
    fn a_group_out_of_patience_skips_only_what_it_alone_brought_in() {
        let (spent, patient, no_patience) = (GroupId::new(0), GroupId::new(1), GroupId::new(2));
        let mut table = PatienceTable::new(&batch(
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
        table.record(0, Duration::from_micros(100));
        table.record(1, Duration::from_micros(100));
        table.record(2, Duration::from_micros(120));
        table.record(3, Duration::from_micros(80));

        assert!(table.skips(4));
        assert!(!table.skips(5));
        assert!(!table.skips(6));
        assert!(!table.skips(7));
        assert!(!table.skips(8));
    }
}
