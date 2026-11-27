use alloc::vec::Vec;
use core::time::Duration;

use cubecl_environment::collections::HashMap;

use crate::tune::plan::batch::Batch;
use crate::tune::set::group::GroupId;

mod group_tracker;

use group_tracker::GroupTracker;

/// Tracks how every group with a [`Patience`](crate::tune::Patience) in a [`Batch`] fares,
/// through the groups that brought each slot of the batch in.
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
        for group in self
            .groups
            .get(slot)
            .expect("a slot should be one of the batch's candidates")
        {
            if let Some(tracker) = self.trackers.get_mut(group) {
                tracker.track(time);
            }
        }
    }

    /// Whether the candidate at `slot` is skipped: every group that brought it in ran out of
    /// patience. One that a group without patience brought in, or that no group did, never is.
    pub fn skips_slot(&self, slot: usize) -> bool {
        let groups = self
            .groups
            .get(slot)
            .expect("a slot should be one of the batch's candidates");
        !groups.is_empty()
            && groups.iter().all(|group| {
                self.trackers
                    .get(group)
                    .is_some_and(GroupTracker::exhausted)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::Patience;
    use crate::tune::plan::batch::BatchEntry;

    fn patience(min_measured_count: usize, max_miss_count: usize) -> Patience {
        Patience {
            min_measured_count,
            max_miss_count,
        }
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

    const SPENT: GroupId = GroupId(0);
    const PATIENT: GroupId = GroupId(1);
    const NO_PATIENCE: GroupId = GroupId(2);

    /// A batch where `SPENT` ran out of patience on its second member while `PATIENT` kept
    /// improving, its slots from 4 on not tracked yet.
    fn interleaved() -> PatienceTracker {
        let mut tracker = PatienceTracker::new(&batch(
            &[
                &[SPENT],
                &[PATIENT],
                &[SPENT],
                &[PATIENT],
                &[SPENT],
                &[PATIENT],
                &[SPENT, PATIENT],
                &[SPENT, NO_PATIENCE],
                &[],
            ],
            &[(SPENT, patience(1, 1)), (PATIENT, patience(1, 1))],
        ));
        tracker.track(0, Duration::from_micros(100));
        tracker.track(1, Duration::from_micros(100));
        tracker.track(2, Duration::from_micros(120));
        tracker.track(3, Duration::from_micros(80));
        tracker
    }

    #[test]
    fn a_group_out_of_patience_skips_what_it_alone_brought_in() {
        assert!(interleaved().skips_slot(4));
    }

    #[test]
    fn a_patient_group_skips_nothing() {
        assert!(!interleaved().skips_slot(5));
    }

    #[test]
    fn a_member_a_patient_group_brought_in_too_is_not_skipped() {
        assert!(!interleaved().skips_slot(6));
    }

    #[test]
    fn a_member_a_group_without_patience_brought_in_too_is_not_skipped() {
        assert!(!interleaved().skips_slot(7));
    }

    #[test]
    fn a_member_no_group_brought_in_is_never_skipped() {
        assert!(!interleaved().skips_slot(8));
    }

    #[test]
    fn a_slot_only_a_group_without_patience_brought_in_is_never_skipped() {
        let mut tracker = PatienceTracker::new(&batch(&[&[GroupId(0)]], &[]));
        tracker.track(0, Duration::from_micros(100));

        assert!(!tracker.skips_slot(0));
    }
}
