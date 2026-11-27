use alloc::vec::Vec;
use cubecl_environment::collections::HashMap;

use crate::tune::plan::group::{GroupId, Patience};
use crate::tune::plan::planner::PlannedTunable;

/// One batch of a [`TunePlanner`](crate::tune::plan::planner::TunePlanner)'s plan: the tunables
/// to benchmark, and the [`Patience`] of each group that brought one in.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Batch {
    /// The tunables in the order they are benchmarked (Arrival Usage), each found by its slot
    /// (Position Usage).
    pub entries: Vec<BatchEntry>,
    /// The patience of each group an entry names that has one, once a group (Membership Usage),
    /// read whole.
    pub patience: HashMap<GroupId, Patience>,
}

/// One tunable of a [`Batch`], with every group that brought it into the batch. An ungrouped
/// tunable has none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEntry {
    pub index: usize,
    /// The groups that brought it in, in the order they did (Arrival Usage), each once
    /// (Membership Usage).
    pub groups: Vec<GroupId>,
    /// The priority it was placed at, by the first group that brought it in. `None` for an
    /// ungrouped tunable, which states none.
    pub priority: Option<i8>,
}

impl Batch {
    /// A batch of `planned`, then of `ungrouped`, carrying the patience `patience` holds for the
    /// groups that brought an entry in. A tunable several groups bring in is benchmarked once, at
    /// its first place, and answers to each of them.
    pub fn new(
        planned: impl IntoIterator<Item = PlannedTunable>,
        ungrouped: impl IntoIterator<Item = usize>,
        patience: &HashMap<GroupId, Patience>,
    ) -> Self {
        /// `group` among `groups`, named once.
        fn join(groups: &mut Vec<GroupId>, group: GroupId) {
            if !groups.contains(&group) {
                groups.push(group);
            }
        }

        let mut entries = Vec::<BatchEntry>::new();
        // The slot of each tunable's entry, by tunable index (Keyed Usage).
        let mut slots = HashMap::<usize, usize>::new();
        for planned in planned {
            let slot = *slots.entry(planned.index).or_insert_with(|| {
                entries.push(BatchEntry::placed(&planned));
                entries.len() - 1
            });
            let entry = entries
                .get_mut(slot)
                .expect("a kept slot should be one of the entries");
            join(&mut entry.groups, planned.group);
        }
        entries.extend(ungrouped.into_iter().map(BatchEntry::ungrouped_at));
        let patience = entries
            .iter()
            .flat_map(|entry| &entry.groups)
            .filter_map(|group| Some((*group, *patience.get(group)?)))
            .collect();
        Self { entries, patience }
    }

    /// A batch that measures every one of `indices`, no group's patience applying.
    #[cfg(not(target_family = "wasm"))]
    pub fn unconditional(indices: impl IntoIterator<Item = usize>) -> Self {
        Self {
            entries: indices.into_iter().map(BatchEntry::ungrouped_at).collect(),
            patience: HashMap::new(),
        }
    }

    /// The candidates the batch measures before the bounds' time limit may end it: the most
    /// members any of its groups' [`Patience`] measures, and at least one.
    pub fn min_measured_count(&self) -> usize {
        self.patience
            .values()
            .map(|patience| patience.min_measured_count)
            .max()
            .unwrap_or(0)
            .max(1)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().map(|entry| entry.index)
    }

    /// How many entries lead the batch at its first entry's priority: the candidates the plan
    /// ranks first, among which it states no preference.
    pub fn leading_len(&self) -> usize {
        let first = self
            .entries
            .first()
            .expect("the batch should hold an entry: only a plan's non-empty batch is ranked");
        self.entries
            .iter()
            .take_while(|entry| entry.priority == first.priority)
            .count()
    }
}

impl BatchEntry {
    /// The entry of `planned`, at its priority, before any group joins it.
    fn placed(planned: &PlannedTunable) -> Self {
        Self {
            index: planned.index,
            groups: Vec::new(),
            priority: Some(planned.priority),
        }
    }

    fn ungrouped_at(index: usize) -> Self {
        Self {
            index,
            groups: Vec::new(),
            priority: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planned(index: usize, group: u32) -> PlannedTunable {
        PlannedTunable {
            index,
            group: GroupId(group),
            priority: 1,
        }
    }

    #[test]
    fn a_tunable_two_groups_bring_in_is_one_entry_naming_both() {
        let batch = Batch::new([planned(0, 0), planned(0, 1)], [], &HashMap::new());

        assert_eq!(batch.entries[0].groups, [GroupId(0), GroupId(1)]);
    }

    #[test]
    fn the_ungrouped_tunables_trail_the_batch() {
        let batch = Batch::new([planned(1, 0)], [0], &HashMap::new());

        assert_eq!(batch.indices().collect::<Vec<_>>(), [1, 0]);
    }

    #[test]
    fn a_batch_carries_the_patience_of_its_groups_only() {
        let patience = |count| Patience {
            min_measured_count: count,
            max_miss_count: count,
        };
        let held = HashMap::from_iter([(GroupId(0), patience(1)), (GroupId(9), patience(2))]);

        let batch = Batch::new([planned(0, 0)], [], &held);

        assert_eq!(
            batch.patience,
            HashMap::from_iter([(GroupId(0), patience(1))])
        );
    }

    #[test]
    fn a_group_without_patience_brings_none() {
        let batch = Batch::new([planned(0, 0)], [], &HashMap::new());

        assert!(batch.patience.is_empty());
    }

    #[test]
    fn the_batch_minimum_is_its_most_patient_group_s() {
        let patience = |count| Patience {
            min_measured_count: count,
            max_miss_count: 1,
        };
        let held = HashMap::from_iter([(GroupId(0), patience(2)), (GroupId(1), patience(5))]);

        let batch = Batch::new([planned(0, 0), planned(1, 1)], [], &held);

        assert_eq!(batch.min_measured_count(), 5);
    }

    #[test]
    fn a_batch_without_patience_measures_one_before_it_may_stop() {
        let batch = Batch::new([], [0], &HashMap::new());

        assert_eq!(batch.min_measured_count(), 1);
    }

    #[test]
    fn the_leading_entries_share_the_first_priority() {
        let batch = Batch::new([planned(0, 0), planned(1, 0)], [2], &HashMap::new());

        assert_eq!(batch.leading_len(), 2);
    }
}
