use alloc::collections::BTreeMap;
#[cfg(persistence)]
use alloc::string::ToString;
use alloc::{string::String, vec, vec::Vec};
use cubecl_environment::collections::HashMap;

use super::group::{GroupId, Patience, TuneGroup};
use super::{Tunable, TuneInputs};

/// One batch of a [`TunePlanner`]'s plan: the [tunables](Tunable) to benchmark, in order, and the
/// [`Patience`] of each group that brought one in.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Batch {
    /// The tunables in the order they are benchmarked (Arrival Usage), each found by its index
    /// (Membership Usage).
    pub entries: Vec<BatchEntry>,
    /// The patience of each group an entry names, by group (Keyed Usage).
    pub patience: HashMap<GroupId, Patience>,
}

/// One [tunable](Tunable) of a [`Batch`], with every group that brought it into the batch.
/// An ungrouped tunable has none.
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
    /// A batch of `planned`, then of `ungrouped`, carrying the `patience` of the groups that
    /// brought an entry in. A tunable several groups bring in is benchmarked once, at its first
    /// place, and answers to each of them.
    pub fn new(
        planned: impl IntoIterator<Item = Planned>,
        ungrouped: impl IntoIterator<Item = usize>,
        patience: &HashMap<GroupId, Patience>,
    ) -> Self {
        let mut entries = Vec::<BatchEntry>::new();
        for planned in planned {
            let Some(entry) = entries
                .iter_mut()
                .find(|entry| entry.index == planned.index)
            else {
                entries.push(BatchEntry::grouped(&planned));
                continue;
            };
            if !entry.groups.contains(&planned.group) {
                entry.groups.push(planned.group);
            }
        }
        entries.extend(ungrouped.into_iter().map(BatchEntry::ungrouped));
        let patience = patience.clone();
        Self { entries, patience }
    }

    #[cfg(not(target_family = "wasm"))]
    /// A batch that measures every one of `indices`, no group's patience applying.
    pub fn unconditional(indices: impl IntoIterator<Item = usize>) -> Self {
        Self {
            entries: indices.into_iter().map(BatchEntry::ungrouped).collect(),
            patience: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The tunable indices, in the order they are benchmarked.
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
    fn grouped(planned: &Planned) -> Self {
        Self {
            index: planned.index,
            groups: vec![planned.group],
            priority: Some(planned.priority),
        }
    }

    fn ungrouped(index: usize) -> Self {
        Self {
            index,
            groups: Vec::new(),
            priority: None,
        }
    }
}

/// Plans which [tunables](Tunable) a tune benchmarks, and in what order: a batch at a time, each
/// the round of the highest group priority and member level left.
#[derive(Debug, Clone)]
pub struct TunePlanner {
    /// How many tunables the plan is of.
    count: usize,
    /// The plan of each group priority, highest last (Keyed and Ordered Usages).
    groups: BTreeMap<i8, GroupPlan>,
    /// The tunables in no group, in registration order (Arrival Usage), drained into the first
    /// batch.
    ungrouped: Vec<usize>,
    /// The tunables a batch already held (Membership Usage).
    returned: Vec<usize>,
    /// The [`Patience`] of every group that has one, by group (Keyed Usage).
    patience: HashMap<GroupId, Patience>,
    /// Each tunable's place in each of its groups, as the key priced it, by tunable index
    /// (Position Usage).
    #[cfg(persistence)]
    placements: Vec<Vec<Placement>>,
}

/// A tunable's place in one of its [groups](TuneGroup), as one key priced it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Placement {
    /// The group's name.
    pub group: String,
    /// Whether the group is [ordered](TuneGroup::ordered): its members' priorities order one round
    /// rather than cutting it into rounds.
    pub ordered: bool,
    /// The group's own priority: higher groups run first.
    pub group_priority: i8,
    /// The tunable's priority within the group; negative skips it.
    pub priority: i8,
}

/// The groups that share one group priority.
#[derive(Default, Debug, Clone)]
pub struct GroupPlan {
    /// The tunables of each member level, highest last (Keyed and Ordered Usages), each level in
    /// the order its tunables were priced (Arrival Usage).
    pub levels: BTreeMap<i8, Vec<Planned>>,
}

/// One tunable's place in a [`GroupPlan`]: which tunable, through which group, and the priority
/// that orders it within its batch.
///
/// The group is its id, not its name: a name is the caller's label and two groups may share one,
/// which the cross-level dedup in [`TunePlanner::strike`] would then read as a single group and
/// strike a live candidate out of the plan.
#[derive(Debug, Clone)]
pub struct Planned {
    pub index: usize,
    pub group: GroupId,
    pub priority: i8,
}

impl Planned {
    /// Whether `other` is the same tunable through the same group.
    fn joins_as(&self, other: &Planned) -> bool {
        self.index == other.index && self.group == other.group
    }
}

/// One tunable's membership of one group, priced for a key.
struct Priced<'g, K> {
    pub index: usize,
    pub group: &'g TuneGroup<K>,
    pub group_priority: i8,
    pub priority: i8,
}

impl<K> Priced<'_, K> {
    #[cfg(persistence)]
    fn placement(&self) -> Placement {
        Placement {
            group: self.group.name().to_string(),
            ordered: self.group.is_ordered(),
            group_priority: self.group_priority,
            priority: self.priority,
        }
    }
}

impl TunePlanner {
    /// The plan of `tunables` for `key`.
    ///
    /// Each of the caller's priority functions is asked once and its answer carried: an ordered
    /// group's level is its best member's priority, which is known only once every member is
    /// priced.
    pub fn new<K, F: TuneInputs, Out: 'static, Id>(
        key: &K,
        tunables: &[Tunable<K, F, Out, Id>],
    ) -> Self {
        let priced: Vec<Priced<'_, K>> = tunables
            .iter()
            .enumerate()
            .flat_map(|(index, tunable)| {
                tunable.groups().iter().map(move |membership| Priced {
                    index,
                    group: membership.group(),
                    group_priority: membership.group().priority(key),
                    priority: membership.priority(key),
                })
            })
            .collect();

        Self {
            count: tunables.len(),
            groups: group_plans(&priced),
            ungrouped: tunables
                .iter()
                .enumerate()
                .filter(|(_, tunable)| tunable.groups().is_empty())
                .map(|(index, _)| index)
                .collect(),
            returned: Vec::new(),
            patience: priced
                .iter()
                .filter_map(|priced| Some((priced.group.id(), priced.group.patience()?)))
                .collect(),
            #[cfg(persistence)]
            placements: placements(&priced, tunables.len()),
        }
    }

    /// Every tunable in the order the plan releases them, with the batch each comes in. A tunable
    /// no batch holds, its every priority negative, comes last in declaration order, in no batch.
    pub fn order(&self) -> Vec<(usize, Option<usize>)> {
        let mut plan = self.clone();
        let batches = core::iter::from_fn(|| {
            Some(plan.plan_next())
                .filter(|batch| !batch.is_empty())
                .map(|batch| batch.indices().collect::<Vec<_>>())
        });
        let mut order: Vec<_> = batches
            .enumerate()
            .flat_map(|(batch, indices)| indices.into_iter().map(move |index| (index, Some(batch))))
            .collect();
        let excluded: Vec<_> = (0..self.count)
            .filter(|index| !order.iter().any(|(seen, _)| seen == index))
            .map(|index| (index, None))
            .collect();
        order.extend(excluded);
        order
    }

    /// The whole plan as a recorded tune reports it: every tunable named by `names`, in the
    /// [order](Self::order) the plan releases them, with the batch each comes in and its place in
    /// each of its groups.
    #[cfg(persistence)]
    pub fn account(&self, names: &[&str]) -> Vec<super::record::PlannedCandidate> {
        self.order()
            .into_iter()
            .map(|(index, batch)| super::record::PlannedCandidate {
                name: names[index].to_string(),
                index,
                batch,
                groups: self.placements[index].clone(),
            })
            .collect()
    }

    /// Hand `queue` the top of the plan, in its order, as a pass that compiles ahead of the tune
    /// does: the candidates leading the first batch at its highest priority, then the candidates
    /// after them until `queue` has accepted `parallelism` of them. `queue` runs a candidate and
    /// answers whether it served the problem; one that declines queues nothing and counts for
    /// nothing.
    ///
    /// A batch in which no candidate served hands its every candidate over, then the next batch's,
    /// as the tune falls back to it. Once one has served, no later batch is reached, since the tune
    /// stops at that batch.
    pub fn plan_top(&mut self, parallelism: usize, mut queue: impl FnMut(usize) -> bool) {
        assert!(
            parallelism > 0,
            "the parallelism should be at least one: the top is extended until that many served"
        );
        loop {
            let batch = self.plan_next();
            if batch.is_empty() {
                return;
            }
            let leading = batch.leading_len();
            let mut served = 0;
            for (position, index) in batch.indices().enumerate() {
                if position >= leading && served >= parallelism {
                    return;
                }
                served += usize::from(queue(index));
            }
            if served > 0 {
                return;
            }
        }
    }

    /// The next batch to benchmark. An empty one means the plan is exhausted.
    ///
    /// A tunable in no group states no priority, so it trails the first batch: the grouped
    /// candidates decide what is compiled and benchmarked first.
    pub fn plan_next(&mut self) -> Batch {
        let grouped = self.take_top_level();
        let batch = Batch::new(
            grouped
                .iter()
                .flatten()
                .filter(|planned| !self.returned.contains(&planned.index))
                .cloned(),
            self.ungrouped.drain(..),
            &self.patience,
        );
        // A level whose every tunable was skipped or already returned says nothing of the levels
        // under it.
        if batch.is_empty() && grouped.is_some() {
            return self.plan_next();
        }
        self.returned.extend(batch.indices());
        batch
    }
}

impl TunePlanner {
    /// The tunables of the highest level of the highest group priority left, highest priority
    /// first and registration order among equals, struck out of every other level they joined
    /// through the same group. Empty where the group priority or the level is negative, which skips
    /// them; `None` once no group is left.
    fn take_top_level(&mut self) -> Option<Vec<Planned>> {
        let mut top = self.groups.last_entry()?;
        let group_priority = *top.key();
        let (level, mut planned) = top
            .get_mut()
            .levels
            .pop_last()
            .expect("a group plan in the plan should hold a level");
        planned.sort_by_key(|planned| (core::cmp::Reverse(planned.priority), planned.index));
        self.strike(&planned);
        planned.retain(|_| group_priority >= 0 && level >= 0);
        Some(planned)
    }

    /// Take `struck` out of every level, dropping the levels and the group plans it empties.
    fn strike(&mut self, struck: &[Planned]) {
        for plan in self.groups.values_mut() {
            for level in plan.levels.values_mut() {
                level.retain(|planned| !struck.iter().any(|other| other.joins_as(planned)));
            }
            plan.levels.retain(|_, level| !level.is_empty());
        }
        self.groups.retain(|_, plan| !plan.levels.is_empty());
    }
}

/// The plan of each group priority of `priced`.
///
/// An ordered group is one batch at its best member's level, so a cutoff group at the same group
/// priority interleaves with it by priority.
fn group_plans<K>(priced: &[Priced<'_, K>]) -> BTreeMap<i8, GroupPlan> {
    let mut ordered_levels = HashMap::<GroupId, i8>::new();
    for member in priced
        .iter()
        .filter(|member| member.group.orders_priority(member.priority))
    {
        ordered_levels
            .entry(member.group.id())
            .and_modify(|level| *level = (*level).max(member.priority))
            .or_insert(member.priority);
    }

    let mut groups = BTreeMap::<i8, GroupPlan>::new();
    for member in priced {
        let level = ordered_levels
            .get(&member.group.id())
            .copied()
            .filter(|_| member.group.orders_priority(member.priority))
            .unwrap_or(member.priority);
        groups
            .entry(member.group_priority)
            .or_default()
            .levels
            .entry(level)
            .or_default()
            .push(Planned {
                index: member.index,
                group: member.group.id(),
                priority: member.priority,
            });
    }
    groups
}

/// The placements of each of `len` tunables, in the order it joined its groups.
#[cfg(persistence)]
fn placements<K>(priced: &[Priced<'_, K>], len: usize) -> Vec<Vec<Placement>> {
    let mut placements = vec![Vec::new(); len];
    for member in priced {
        placements[member.index].push(member.placement());
    }
    placements
}

#[cfg(test)]
mod tests {
    use core::fmt::Display;

    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::tune::{AutotuneKeyable, OrderedSettings};
    use alloc::string::String;

    fn indices(batch: &Batch) -> Vec<usize> {
        batch.indices().collect()
    }

    #[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Debug)]
    struct FakeAutotuneKey;

    impl Display for FakeAutotuneKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("FakeAutotuneKey")
        }
    }

    impl AutotuneKeyable for FakeAutotuneKey {}

    #[test_log::test]
    fn test_plan_order() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        // tunable0 is in no group, so it trails the batch rather than leading it.
        assert_eq!(indices(&plan.plan_next()), vec![2, 0]);
        assert_eq!(indices(&plan.plan_next()), vec![1]);
        assert_eq!(indices(&plan.plan_next()), vec![3]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_order_multi_groups_same_priority() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 1);
        let group2 = TuneGroup::<FakeAutotuneKey>::new("group2", |_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 2);
        let tunable4 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group2, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3, tunable4]);

        assert_eq!(indices(&plan.plan_next()), vec![2, 0]);
        assert_eq!(indices(&plan.plan_next()), vec![1]);
        assert_eq!(indices(&plan.plan_next()), vec![3, 4]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_order_tunable_multiple_groups() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 1);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 2);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
            .group(&group0, |_| 1)
            .group(&group1, |_| 2);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 3);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(indices(&plan.plan_next()), vec![3, 0]);
        assert_eq!(indices(&plan.plan_next()), vec![1]);
        assert_eq!(indices(&plan.plan_next()), vec![2]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_negative_priority() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| -1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(indices(&plan.plan_next()), vec![2, 0]);
        assert_eq!(indices(&plan.plan_next()), vec![3]);
        assert!(plan.plan_next().is_empty());
    }

    #[cfg(persistence)]
    #[test_log::test]
    fn test_plan_account_lists_every_tunable_in_release_order() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| -1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 2);

        let key = FakeAutotuneKey;
        let plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3]);
        let account = plan.account(&["t0", "t1", "t2", "t3"]);

        let order: Vec<_> = account
            .iter()
            .map(|planned| (planned.name.as_str(), planned.batch))
            .collect();
        assert_eq!(
            order,
            vec![
                ("t2", Some(0)),
                ("t0", Some(0)),
                ("t3", Some(1)),
                ("t1", None)
            ]
        );
        assert_eq!(
            account[0].groups,
            vec![Placement {
                group: "group0".into(),
                ordered: false,
                group_priority: 2,
                priority: 2,
            }]
        );
        assert!(account[1].groups.is_empty());
        // Accounting for the plan leaves it to run from its first batch.
        let mut plan = plan;
        assert_eq!(indices(&plan.plan_next()), vec![2, 0]);
    }

    #[test_log::test]
    fn test_plan_no_group() {
        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1]);

        assert_eq!(indices(&plan.plan_next()), vec![0, 1]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_falls_through_when_all_group_tunables_fail() {
        // Every tunable lives in exactly one group; the caller treats every batch as a failure
        // by continuing to call next(). The plan must still surface every tunable, in priority
        // order, before going empty.
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 1);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 1);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        let mut all_returned: Vec<usize> = Vec::new();
        loop {
            let batch = plan.plan_next();
            if batch.is_empty() {
                break;
            }
            all_returned.extend(batch.indices());
        }

        // Highest group (prio 2) drains first from highest intra-priority down, then next group.
        assert_eq!(all_returned, vec![1, 0, 3, 2]);
    }

    #[test_log::test]
    fn test_plan_single_group_exhausts_all_intra_priorities() {
        // A single group with multiple intra-priorities should yield each batch separately,
        // allowing the caller to continue on failures until the group is exhausted.
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 0);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 1);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 3);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(indices(&plan.plan_next()), vec![2]);
        assert_eq!(indices(&plan.plan_next()), vec![1]);
        assert_eq!(indices(&plan.plan_next()), vec![0]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_all_negative_group_advances_to_next_group() {
        // A group whose every tunable has a negative intra-priority should be skipped entirely
        // without stopping autotuning — the next group must still be reached.
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| -1);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| -2);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(indices(&plan.plan_next()), vec![2]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_no_group_tunables_only_emitted_once_even_on_failures() {
        // The ungrouped tunables are emitted together with the first group batch. If the caller
        // keeps calling next() (treating the first batch as failing), they must not be
        // re-emitted, and the plan must still advance to later groups.
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group1, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(indices(&plan.plan_next()), vec![1, 0]);
        assert_eq!(indices(&plan.plan_next()), vec![2]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_multi_group_tunable_not_duplicated_across_failed_groups() {
        // tunable1 belongs to both group0 and group1. It must be returned exactly once (via its
        // higher-priority group), even if the caller continues iterating after failures.
        let group0 = TuneGroup::<FakeAutotuneKey>::new("group0", |_| 1);
        let group1 = TuneGroup::<FakeAutotuneKey>::new("group1", |_| 2);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
            .group(&group0, |_| 1)
            .group(&group1, |_| 1);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group0, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1]);

        let mut all_returned: Vec<usize> = Vec::new();
        loop {
            let batch = plan.plan_next();
            if batch.is_empty() {
                break;
            }
            all_returned.extend(batch.indices());
        }

        // tunable0 comes from group1 (higher priority). tunable1 is the sole member of group0
        // after cross-group dedup. No duplicates.
        assert_eq!(all_returned, vec![0, 1]);
    }

    #[test_log::test]
    fn test_plan_recurses_when_batch_is_fully_already_returned() {
        // Regression test: a tunable that lives in multiple groups was already emitted via its
        // higher-priority group, so when its lower-priority group's batch fires the only index
        // is one already present in `returned`. The plan must NOT return an empty batch here
        // (that signals "no more work" to the caller and aborts with NoValidKernelFound); it
        // must recurse to the next intra-priority and surface the remaining tunable.
        //
        // Cross-group dedup in group_plan_next compares (index, Arc<String> group_name), so a
        // tunable appearing in both group_hi and group_lo isn't auto-removed from group_lo
        // when popped from group_hi — the `returned` + `all_skip` path is the only guard.
        let group_hi = TuneGroup::<FakeAutotuneKey>::new("hi", |_| 2);
        let group_lo = TuneGroup::<FakeAutotuneKey>::new("lo", |_| 1);

        // tunable0 is in both groups. tunable1 is only in group_lo at a lower intra-priority.
        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
            .group(&group_hi, |_| 1)
            .group(&group_lo, |_| 2);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group_lo, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1]);

        // First call: group_hi yields tunable0.
        assert_eq!(indices(&plan.plan_next()), vec![0]);
        // Second call: group_lo's higher intra-priority batch is just tunable0 (already
        // returned). Without the fix this returns [] and the autotuner aborts. With the fix
        // the plan recurses and yields tunable1.
        assert_eq!(indices(&plan.plan_next()), vec![1]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_ordered_group_is_one_batch_best_first() {
        // Every member is in the round, best first, and a negative priority still skips.
        let group =
            TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1, OrderedSettings::default());

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 1);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 3);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| -1);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 3);
        let tunable4 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3, tunable4]);

        assert_eq!(indices(&plan.plan_next()), vec![1, 3, 4, 0]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_batch_carries_every_group_of_an_entry() {
        // The batch carries patience only for the group that has one, and a tunable two
        // groups bring in is one entry naming both.
        let patience = Patience {
            min_measured_count: 10,
            max_miss_count: 5,
        };
        let ordered = TuneGroup::<FakeAutotuneKey>::ordered(
            "ordered",
            |_| 1,
            OrderedSettings::default().with_patience(patience),
        );
        let cutoff = TuneGroup::<FakeAutotuneKey>::new("cutoff", |_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 2);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&cutoff, |_| 2);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 1);
        let tunable4 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
            .group(&cutoff, |_| 2)
            .group(&ordered, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3, tunable4]);
        let batch = plan.plan_next();
        let entry = |index, groups: &[GroupId], priority| BatchEntry {
            index,
            groups: groups.to_vec(),
            priority,
        };

        assert_eq!(
            batch.entries,
            vec![
                entry(1, &[ordered.id()], Some(2)),
                entry(2, &[cutoff.id()], Some(2)),
                entry(4, &[cutoff.id(), ordered.id()], Some(2)),
                entry(3, &[ordered.id()], Some(1)),
                entry(0, &[], None),
            ]
        );
        assert_eq!(
            batch.patience,
            HashMap::from_iter([(ordered.id(), patience)])
        );
    }

    #[test_log::test]
    fn test_plan_entry_names_a_group_listed_twice_once() {
        // A group listed twice on one tunable is still one group: naming it twice would
        // count each of the tunable's times twice against its patience.
        let ordered =
            TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1, OrderedSettings::default());

        let tunable = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
            .group(&ordered, |_| 2)
            .group(&ordered, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable]);

        assert_eq!(
            plan.plan_next().entries,
            vec![BatchEntry {
                index: 0,
                groups: vec![ordered.id()],
                priority: Some(2),
            }]
        );
    }

    #[test_log::test]
    fn test_plan_ordered_group_keeps_the_group_cutoff() {
        // The order is within the group. A lower-priority group is still a fallback,
        // reached only once the ordered batch fails.
        let first =
            TuneGroup::<FakeAutotuneKey>::ordered("first", |_| 2, OrderedSettings::default());
        let fallback = TuneGroup::<FakeAutotuneKey>::new("fallback", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&first, |_| 1);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&first, |_| 2);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&fallback, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(indices(&plan.plan_next()), vec![1, 0]);
        assert_eq!(indices(&plan.plan_next()), vec![2]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_ordered_batch_leads_the_ungrouped_tunables() {
        // A tunable in no group must not preempt the ordered group's best candidate. The
        // batch stops at the first candidate under the bound, so whatever leads it is
        // what gets compiled.
        let group =
            TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1, OrderedSettings::default());

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 5);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(indices(&plan.plan_next()), vec![2, 1, 0]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_ordered_batch_is_not_jumped_by_a_cutoff_group_beside_it() {
        // Both groups sit at group priority 1. The ordered batch is planned at its best
        // member's priority, so the cutoff member under it is a fallback behind the batch
        // and cannot take a round of its own in front of it, where a short circuit would
        // skip the ordered group whole.
        let ordered =
            TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1, OrderedSettings::default());
        let cutoff = TuneGroup::<FakeAutotuneKey>::new("cutoff", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 3);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&cutoff, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(indices(&plan.plan_next()), vec![0, 1]);
        assert_eq!(indices(&plan.plan_next()), vec![2]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_cutoff_member_above_the_ordered_batch_still_leads() {
        // The interleaving cuts both ways. A cutoff member priced above the ordered
        // group's best keeps its round in front, and one priced level with it joins the
        // batch, ordered among the members by priority.
        let ordered =
            TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1, OrderedSettings::default());
        let cutoff = TuneGroup::<FakeAutotuneKey>::new("cutoff", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 2);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&cutoff, |_| 3);
        let tunable3 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&cutoff, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(indices(&plan.plan_next()), vec![2]);
        assert_eq!(indices(&plan.plan_next()), vec![0, 3, 1]);
        assert!(plan.plan_next().is_empty());
    }

    #[test_log::test]
    fn test_plan_same_named_groups_do_not_strike_each_other_out() {
        // Two groups may share a name, and the cross-level dedup must still tell them
        // apart. Keyed on the name, popping `hi`'s discarded negative level takes
        // tunable1 out of `lo`'s plan and the only viable candidate is never benchmarked.
        let hi = TuneGroup::<FakeAutotuneKey>::new("shared", |_| 2);
        let lo = TuneGroup::<FakeAutotuneKey>::new("shared", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&hi, |_| 1);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
            .group(&hi, |_| -1)
            .group(&lo, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlanner::new(&key, &[tunable0, tunable1]);

        assert_eq!(indices(&plan.plan_next()), vec![0]);
        assert_eq!(indices(&plan.plan_next()), vec![1]);
        assert!(plan.plan_next().is_empty());
    }

    #[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Debug)]
    struct SizeKey(pub i8);

    impl Display for SizeKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "SizeKey({})", self.0)
        }
    }

    impl AutotuneKeyable for SizeKey {}

    #[test_log::test]
    fn test_plan_orders_each_key_by_its_own_priorities() {
        // The plan asks every priority function of the key it is built for: two keys that
        // rank the same members in opposite orders get opposite batches, and members the
        // key ties stay in registration order.
        let group = TuneGroup::<SizeKey>::ordered("ordered", |_| 2, OrderedSettings::default());
        let tunables = || {
            [
                Tunable::<SizeKey, (), ()>::new("rises", fake_kernel)
                    .group(&group, |key: &SizeKey| key.0),
                Tunable::<SizeKey, (), ()>::new("falls", fake_kernel)
                    .group(&group, |key: &SizeKey| 10 - key.0),
                Tunable::<SizeKey, (), ()>::new("flat", fake_kernel).group(&group, |_| 5),
            ]
        };

        assert_eq!(
            indices(&TunePlanner::new(&SizeKey(9), &tunables()).plan_next()),
            vec![0, 2, 1]
        );
        assert_eq!(
            indices(&TunePlanner::new(&SizeKey(1), &tunables()).plan_next()),
            vec![1, 2, 0]
        );
        assert_eq!(
            indices(&TunePlanner::new(&SizeKey(5), &tunables()).plan_next()),
            vec![0, 1, 2]
        );
    }

    /// An ordered group of `priorities.len()` members, each at its priority, and an ungrouped
    /// tunable after them.
    fn ordered_plan(priorities: &[i8]) -> TunePlanner {
        let group =
            TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1, OrderedSettings::default());
        let mut tunables: Vec<_> = priorities
            .iter()
            .map(|&priority| {
                Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
                    .group(&group, move |_| priority)
            })
            .collect();
        tunables.push(Tunable::new("ungrouped", fake_kernel));
        TunePlanner::new(&FakeAutotuneKey, &tunables)
    }

    /// What [`TunePlanner::plan_top`] hands over, each candidate serving unless `declines` names it.
    fn queued(mut plan: TunePlanner, parallelism: usize, declines: &[usize]) -> Vec<usize> {
        let mut queued = Vec::new();
        plan.plan_top(parallelism, |index| {
            queued.push(index);
            !declines.contains(&index)
        });
        queued
    }

    #[test_log::test]
    fn test_plan_top_takes_every_candidate_tied_at_the_top() {
        let plan = ordered_plan(&[9, 9, 9, 9, 3, 3]);
        assert_eq!(queued(plan, 2, &[]), vec![0, 1, 2, 3]);
    }

    #[test_log::test]
    fn test_plan_top_extends_a_small_top_to_the_parallelism() {
        let plan = ordered_plan(&[3, 9, 5, 7, 1]);
        assert_eq!(queued(plan, 3, &[]), vec![1, 3, 2]);
    }

    #[test_log::test]
    fn test_plan_top_does_not_count_a_declining_candidate() {
        let plan = ordered_plan(&[3, 9, 5, 7, 1]);
        assert_eq!(queued(plan, 2, &[3]), vec![1, 3, 2]);
    }

    #[test_log::test]
    fn test_plan_top_falls_back_only_past_a_batch_that_served_nothing() {
        let first = TuneGroup::<FakeAutotuneKey>::new("first", |_| 2);
        let fallback = TuneGroup::<FakeAutotuneKey>::new("fallback", |_| 1);
        let tunables = || {
            [
                Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&first, |_| 1),
                Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
                    .group(&fallback, |_| 1),
                Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
                    .group(&fallback, |_| 1),
            ]
        };
        let plan = || TunePlanner::new(&FakeAutotuneKey, &tunables());

        assert_eq!(queued(plan(), 4, &[]), vec![0]);
        assert_eq!(queued(plan(), 4, &[0]), vec![0, 1, 2]);
    }

    #[test_log::test]
    fn test_plan_top_hands_over_the_whole_plan_when_nothing_serves() {
        let plan = ordered_plan(&[2, 1]);
        assert_eq!(queued(plan, 1, &[0, 1, 2]), vec![0, 1, 2]);
    }

    fn fake_kernel(_: ()) -> Result<(), String> {
        Ok(())
    }

    #[test_log::test]
    #[should_panic(expected = "the parallelism should be at least one")]
    fn test_plan_top_needs_a_parallelism() {
        ordered_plan(&[1]).plan_top(0, |_| true);
    }
}
