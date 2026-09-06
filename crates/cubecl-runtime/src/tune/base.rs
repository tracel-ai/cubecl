use super::{AutotuneError, AutotuneKey, TuneFn, TuneInputs};

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::sync::atomic::{AtomicU32, Ordering};
use cubecl_environment::collections::HashMap;

/// A single candidate for autotune: a named [`TuneFn`] plus the [groups](TuneGroup) it
/// belongs to. A tunable is autotuned whenever any of its groups is prioritized.
pub struct Tunable<K, F: TuneInputs, Output> {
    pub(crate) function: TuneFn<F, Output>,
    groups: Vec<(TuneGroup<K>, PriorityFunc<K>)>,
}

impl<K, F: TuneInputs, Output: 'static> Tunable<K, F, Output> {
    /// Create a tunable from a closure.
    ///
    /// The `for<'a> Fn(F::At<'a>) -> _` bound is spelled out directly in the
    /// `where`-clause (rather than hidden behind a helper trait) so that Rust closure
    /// inference sees it: otherwise `move |input| …` picks a single concrete lifetime
    /// and fails with `implementation of FnOnce is not general enough` whenever
    /// `F::At<'a>` actually depends on `'a`.
    ///
    /// For multi-input kernels, destructure a tuple:
    /// `Tunable::new("name", |(lhs, rhs, out)| body)`.
    ///
    /// A tunable left in no [group](Tunable::group) is a candidate in the first round for
    /// every key, but it states no priority, so it trails that round's grouped candidates
    /// and a short circuit among them leaves it unmeasured. Give it a group to say where
    /// in the round it belongs.
    pub fn new<Func, Err>(name: &str, func: Func) -> Self
    where
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<F as TuneInputs>::At<'a>) -> Result<Output, Err> + Send + Sync + 'static,
    {
        let name: String = name.into();
        let name_for_err = name.clone();
        Self {
            function: TuneFn::new(
                name,
                Box::new(move |inputs| {
                    func(inputs).map_err(|err| AutotuneError::Unknown {
                        name: name_for_err.to_string(),
                        err: err.into(),
                    })
                }),
            ),
            groups: Vec::new(),
        }
    }

    /// Add this tunable to a [`TuneGroup`] with the given intra-group priority.
    ///
    /// Groups are autotuned in order of their priority. Within a group the priority is a
    /// cutoff: the highest `priority(key)` whose tunables run is the round, and a lower one
    /// is a fallback, reached only once every tunable above it has failed. Within an
    /// [ordered](TuneGroup::ordered) group it is an order instead: every tunable is in the
    /// one round, tried from the highest `priority(key)` down. A negative priority skips
    /// the tunable for this key in either kind of group.
    pub fn group(
        mut self,
        group: &TuneGroup<K>,
        priority: impl Fn(&K) -> i8 + Send + Sync + 'static,
    ) -> Self {
        self.groups.push((group.clone(), Arc::new(priority)));
        self
    }
}

/// A priority bucket for tunables, computed from the [autotune key](AutotuneKey).
///
/// Higher-priority groups are autotuned first; once any tunable in a group returns a
/// valid result, no later groups are tried.
pub struct TuneGroup<K> {
    id: u32,
    name: Arc<String>,
    pub(crate) priority: PriorityFunc<K>,
    /// Whether a member's priority orders the round rather than cutting it off.
    ordered: bool,
}

impl<K> core::fmt::Debug for TuneGroup<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TuneGroup")
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

impl<K> Clone for TuneGroup<K> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            name: self.name.clone(),
            priority: self.priority.clone(),
            ordered: self.ordered,
        }
    }
}

impl<K> TuneGroup<K> {
    /// Create a new group based on a priority function.
    ///
    /// A member's own priority ([`Tunable::group`]) is a cutoff within it: only the
    /// highest one is the round, and the ones below it are fallbacks reached in turn as
    /// each round fails.
    pub fn new(name: &str, f: impl Fn(&K) -> i8 + Send + Sync + 'static) -> Self {
        Self::build(name, f, false)
    }

    /// Create a group whose members' priorities **order** the round rather than cut it
    /// off: every member with a non-negative priority is planned in one batch, highest
    /// first, and a lower priority means later, never left out.
    ///
    /// What that buys is a round that stops early on the strength of a bound. Candidates
    /// are benchmarked in batch order and a candidate confirmed under the
    /// [bounds](super::Bounds)' time limit ends the batch before the ones after it are
    /// compiled — so a priority that says which candidate is *likeliest* to be close
    /// enough decides how much of the group is ever compiled, while a priority that is
    /// wrong costs a compile rather than the winner.
    ///
    /// That early exit is the short circuit, which needs a bounds generator to give the
    /// round a time limit and is off on wasm, where a benchmark cannot be resolved
    /// inline, and wherever the config disables it. Without it every member of the batch
    /// is compiled and benchmarked on a cache miss, so an ordered group costs what its
    /// whole membership costs where a cutoff group would have stopped at one level.
    ///
    /// The batch is planned at the priority of its best member, so a cutoff group at the
    /// same group priority interleaves with it the way two cutoff groups do: a cutoff
    /// member above that priority is tried first, one at it joins the batch, and lower
    /// ones remain fallbacks behind the whole batch.
    pub fn ordered(name: &str, f: impl Fn(&K) -> i8 + Send + Sync + 'static) -> Self {
        Self::build(name, f, true)
    }

    fn build(name: &str, f: impl Fn(&K) -> i8 + Send + Sync + 'static, ordered: bool) -> Self {
        let id = GROUP_COUNTER.fetch_add(1, Ordering::Relaxed);

        Self {
            id,
            name: Arc::new(name.into()),
            priority: Arc::new(f),
            ordered,
        }
    }
}

#[derive(Debug)]
/// A group plan dictates which [tunables](Tunable) should be executed, and in what order.
pub(crate) struct TunePlan {
    priorities: Vec<i8>,
    no_groups: Vec<usize>,
    groups: HashMap<i8, GroupPlan>,
    returned: Vec<usize>,
}

#[derive(Default, Debug)]
struct GroupPlan {
    priorities: Vec<i8>,
    indices: HashMap<i8, Vec<Planned>>,
}

/// One tunable's place in a [`GroupPlan`]: which tunable, through which group, and the
/// priority that orders it within its batch.
///
/// The group is its [id](TuneGroup::id) rather than its name, because a name is the
/// caller's label and two groups may well share one: keying the cross-level dedup in
/// [`TunePlan::group_plan_next`] on the name would let one group's batch strike a
/// same-named group's candidate out of the plan entirely.
#[derive(Debug)]
struct Planned {
    index: usize,
    group: u32,
    priority: i8,
}

#[derive(Debug)]
struct Cleanup {
    groups: Vec<i8>,
    tunables: Vec<(i8, i8)>,
    /// Within group priority is too low to even try.
    skipped: bool,
}

impl TunePlan {
    pub fn new<K: AutotuneKey, F: TuneInputs, Out>(
        key: &K,
        tunables: &[Tunable<K, F, Out>],
    ) -> Self {
        let mut priorities = Vec::<i8>::new();
        let mut no_groups = Vec::new();
        let mut groups = HashMap::<i8, GroupPlan>::new();

        // The priority functions belong to the caller, so each is asked once and its
        // answer carried: an [ordered](TuneGroup::ordered) group is planned as one batch
        // at its best member's priority, which is not known until every member is priced.
        let mut priced = Vec::new();
        let mut ordered_levels = HashMap::<u32, i8>::new();

        for (index, tunable) in tunables.iter().enumerate() {
            if tunable.groups.is_empty() {
                no_groups.push(index);
                continue;
            }

            for (group, within_group_priority_fn) in tunable.groups.iter() {
                let group_priority = (group.priority)(key);
                let priority = within_group_priority_fn(key);

                if group.ordered && priority >= 0 {
                    let level = ordered_levels.entry(group.id).or_insert(priority);
                    *level = (*level).max(priority);
                }

                priced.push((index, group, group_priority, priority));
            }
        }

        for (index, group, group_priority, priority) in priced {
            if !priorities.contains(&group_priority) {
                priorities.push(group_priority);
            }

            let group_plan = match groups.get_mut(&group_priority) {
                Some(val) => val,
                None => {
                    groups.insert(group_priority, GroupPlan::default());
                    groups.get_mut(&group_priority).unwrap()
                }
            };

            // An ordered group plans every member that is in the round as one batch; the
            // priority then orders the batch (`group_plan_next`). The batch sits at the
            // level of its best member so that a cutoff group planned at the same group
            // priority interleaves with it: a cutoff member above that level is still
            // tried first, one level with it joins the batch, and lower ones stay the
            // fallback behind it. Pinning the batch to `0` instead would let any cutoff
            // member jump ahead of the whole ordered group.
            let level = match group.ordered && priority >= 0 {
                true => ordered_levels[&group.id],
                false => priority,
            };
            let planned = Planned {
                index,
                group: group.id,
                priority,
            };

            if group_plan.priorities.contains(&level) {
                group_plan.indices.get_mut(&level).unwrap().push(planned);
            } else {
                group_plan.priorities.push(level);
                group_plan.indices.insert(level, vec![planned]);
            }
        }

        priorities.sort();

        for group in groups.iter_mut() {
            group.1.priorities.sort();
        }

        Self {
            priorities,
            no_groups,
            groups,
            returned: Vec::new(),
        }
    }

    /// Get the next batch of [tunable](Tunable) index to be autotuned.
    ///
    /// Note that if the list is empty, it means no more autotuned entry can be executed.
    pub(crate) fn next(&mut self) -> Vec<usize> {
        // A tunable in no group states no priority, so it trails the batch rather than
        // leading it: the group's own order decides what is compiled and benchmarked
        // first, which is the whole point of an [ordered](TuneGroup::ordered) group.
        let ungrouped = core::mem::take(&mut self.no_groups);
        let mut indices = Vec::new();
        let priority = self.priorities.last();

        let priority = match priority {
            Some(val) => *val,
            None => return ungrouped,
        };

        let (group_indices, cleanup) = self.group_plan_next(priority);
        // Some entries are skipped for this round of prioritizing.
        let skipped = cleanup.skipped || priority < 0;
        let mut all_skip = true;

        self.cleanup(cleanup);

        if priority >= 0 {
            for index in group_indices {
                if !self.returned.contains(&index) && !indices.contains(&index) {
                    all_skip = false;
                    indices.push(index);
                }
            }
        }

        indices.extend(ungrouped);

        // The indices list is empty, but it doesn't mean we should stop
        // autotuning, since some entries were skipped.

        if indices.is_empty() && (skipped || all_skip) {
            self.next()
        } else {
            for i in indices.iter() {
                self.returned.push(*i);
            }
            indices
        }
    }

    fn cleanup(&mut self, cleanup: Cleanup) {
        for group_p in cleanup.groups {
            let index = self
                .priorities
                .iter()
                .enumerate()
                .find(|p| *p.1 == group_p)
                .unwrap();

            self.priorities.remove(index.0);
            self.groups.remove(&group_p);
        }

        for (group_p, tunable_p) in cleanup.tunables {
            if let Some(group) = self.groups.get_mut(&group_p) {
                let index = group
                    .priorities
                    .iter()
                    .enumerate()
                    .find(|p| *p.1 == tunable_p)
                    .unwrap();
                group.priorities.remove(index.0);
                group.indices.remove(&tunable_p);
            }
        }
    }

    fn group_plan_next(&mut self, priority: i8) -> (Vec<usize>, Cleanup) {
        let group_plan = self.groups.get_mut(&priority).expect("To be filled");
        let within_group_prio = group_plan.priorities.pop().unwrap();
        let mut next_indices = group_plan.indices.remove(&within_group_prio).unwrap();
        // Highest priority first, registration order among equals. A level of purely
        // cutoff members holds one priority and is already in registration order, so this
        // only moves anything once an ordered group's batch is in the level.
        next_indices.sort_by_key(|planned| (core::cmp::Reverse(planned.priority), planned.index));

        let mut cleanup_groups = Vec::new();
        let mut cleanup_tunables = Vec::new();

        for (pg, group) in self.groups.iter_mut() {
            let mut num_empty_tunables = 0;
            let num_tunables = group.priorities.len();

            for (pt, indices) in group.indices.iter_mut() {
                for n in &next_indices {
                    let entry = indices
                        .iter()
                        .position(|p| p.index == n.index && p.group == n.group);
                    if let Some(entry) = entry {
                        indices.remove(entry);
                    }
                }

                if indices.is_empty() {
                    num_empty_tunables += 1;
                    cleanup_tunables.push((*pg, *pt));
                }
            }

            if num_empty_tunables == num_tunables {
                cleanup_groups.push(*pg);
            }
        }

        if within_group_prio < 0 {
            // Discard algorithms with negative priority
            next_indices.clear();
        }

        (
            next_indices
                .into_iter()
                .map(|planned| planned.index)
                .collect(),
            Cleanup {
                groups: cleanup_groups,
                tunables: cleanup_tunables,
                skipped: within_group_prio < 0,
            },
        )
    }
}

type PriorityFunc<K> = Arc<dyn Fn(&K) -> i8 + Send + Sync>;

static GROUP_COUNTER: AtomicU32 = AtomicU32::new(0);

#[cfg(test)]
mod tests {
    use core::fmt::Display;

    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Debug)]
    struct FakeAutotuneKey;

    impl Display for FakeAutotuneKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("FakeAutotuneKey")
        }
    }

    impl AutotuneKey for FakeAutotuneKey {}

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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        // tunable0 is in no group, so it trails the batch rather than leading it.
        assert_eq!(plan.next(), vec![2, 0]);
        assert_eq!(plan.next(), vec![1]);
        assert_eq!(plan.next(), vec![3]);
        assert!(plan.next().is_empty());
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3, tunable4]);

        assert_eq!(plan.next(), vec![2, 0]);
        assert_eq!(plan.next(), vec![1]);
        assert_eq!(plan.next(), vec![3, 4]);
        assert!(plan.next().is_empty());
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(plan.next(), vec![3, 0]);
        assert_eq!(plan.next(), vec![1]);
        assert_eq!(plan.next(), vec![2]);
        assert!(plan.next().is_empty());
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(plan.next(), vec![2, 0]);
        assert_eq!(plan.next(), vec![3]);
        assert!(plan.next().is_empty());
    }

    #[test_log::test]
    fn test_plan_no_group() {
        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1]);

        assert_eq!(plan.next(), vec![0, 1]);
        assert!(plan.next().is_empty());
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        let mut all_returned: Vec<usize> = Vec::new();
        loop {
            let batch = plan.next();
            if batch.is_empty() {
                break;
            }
            all_returned.extend(batch);
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(plan.next(), vec![2]);
        assert_eq!(plan.next(), vec![1]);
        assert_eq!(plan.next(), vec![0]);
        assert!(plan.next().is_empty());
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(plan.next(), vec![2]);
        assert!(plan.next().is_empty());
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(plan.next(), vec![1, 0]);
        assert_eq!(plan.next(), vec![2]);
        assert!(plan.next().is_empty());
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1]);

        let mut all_returned: Vec<usize> = Vec::new();
        loop {
            let batch = plan.next();
            if batch.is_empty() {
                break;
            }
            all_returned.extend(batch);
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1]);

        // First call: group_hi yields tunable0.
        assert_eq!(plan.next(), vec![0]);
        // Second call: group_lo's higher intra-priority batch is just tunable0 (already
        // returned). Without the fix this returns [] and the autotuner aborts. With the fix
        // the plan recurses and yields tunable1.
        assert_eq!(plan.next(), vec![1]);
        assert!(plan.next().is_empty());
    }

    #[test_log::test]
    fn test_plan_ordered_group_is_one_batch_best_first() {
        // Every member is in the round, highest priority first and registration order
        // among equals; a negative priority still skips.
        let group = TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1);

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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3, tunable4]);

        assert_eq!(plan.next(), vec![1, 3, 4, 0]);
        assert!(plan.next().is_empty());
    }

    #[test_log::test]
    fn test_plan_ordered_group_keeps_the_group_cutoff() {
        // The order is within the group; a lower-priority group is still a fallback
        // reached only when the ordered batch fails.
        let first = TuneGroup::<FakeAutotuneKey>::ordered("first", |_| 2);
        let fallback = TuneGroup::<FakeAutotuneKey>::new("fallback", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&first, |_| 1);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&first, |_| 2);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&fallback, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(plan.next(), vec![1, 0]);
        assert_eq!(plan.next(), vec![2]);
        assert!(plan.next().is_empty());
    }

    #[test_log::test]
    fn test_plan_ordered_batch_leads_the_ungrouped_tunables() {
        // A tunable in no group has no priority to state, so it must not preempt the
        // ordered group's best candidate: the batch is benchmarked in order and stops on
        // the first candidate under the bound, so whatever leads it is what gets compiled.
        let group = TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&group, |_| 5);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(plan.next(), vec![2, 1, 0]);
        assert!(plan.next().is_empty());
    }

    #[test_log::test]
    fn test_plan_ordered_batch_is_not_jumped_by_a_cutoff_group_beside_it() {
        // Both groups are at group priority 1. The ordered batch is planned at its best
        // member's priority (3), so the cutoff member at 2 is a fallback behind it rather
        // than a round of its own in front of it — which would both delay the ordered
        // group's best candidate and let a short circuit skip the group entirely.
        let ordered = TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1);
        let cutoff = TuneGroup::<FakeAutotuneKey>::new("cutoff", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 3);
        let tunable1 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&ordered, |_| 1);
        let tunable2 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&cutoff, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2]);

        assert_eq!(plan.next(), vec![0, 1]);
        assert_eq!(plan.next(), vec![2]);
        assert!(plan.next().is_empty());
    }

    #[test_log::test]
    fn test_plan_cutoff_member_above_the_ordered_batch_still_leads() {
        // The interleaving cuts both ways: a cutoff member priced above the ordered
        // group's best member keeps its round in front, and one priced level with it
        // joins the batch, ordered among the members by priority.
        let ordered = TuneGroup::<FakeAutotuneKey>::ordered("ordered", |_| 1);
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
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(plan.next(), vec![2]);
        assert_eq!(plan.next(), vec![0, 3, 1]);
        assert!(plan.next().is_empty());
    }

    #[test_log::test]
    fn test_plan_same_named_groups_do_not_strike_each_other_out() {
        // A name is the caller's label, not an identity: two groups may share one. The
        // cross-level dedup must key on the group itself, or popping `hi`'s discarded
        // negative level takes tunable1 out of `lo`'s plan with it and the only viable
        // candidate is never benchmarked.
        let hi = TuneGroup::<FakeAutotuneKey>::new("shared", |_| 2);
        let lo = TuneGroup::<FakeAutotuneKey>::new("shared", |_| 1);

        let tunable0 =
            Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel).group(&hi, |_| 1);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel)
            .group(&hi, |_| -1)
            .group(&lo, |_| 1);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1]);

        assert_eq!(plan.next(), vec![0]);
        assert_eq!(plan.next(), vec![1]);
        assert!(plan.next().is_empty());
    }

    fn fake_kernel(_: ()) -> Result<(), String> {
        Ok(())
    }
}
