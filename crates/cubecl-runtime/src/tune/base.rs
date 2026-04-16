use super::{AutotuneError, AutotuneKey, TuneFn, TuneInputs};
use alloc::string::ToString;
use alloc::{format, string::String, sync::Arc, vec, vec::Vec};
use core::sync::atomic::{AtomicU32, Ordering};
use hashbrown::HashMap;

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
    pub fn new<Func, Err>(name: &str, func: Func) -> Self
    where
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<F as TuneInputs>::At<'a>) -> Result<Output, Err> + Send + Sync + 'static,
    {
        let name = name.to_string();
        let name_for_err = name.clone();
        Self {
            function: TuneFn {
                name,
                func: Arc::new(move |inputs| {
                    func(inputs).map_err(|err| AutotuneError::Unknown {
                        name: name_for_err.clone(),
                        err: err.into(),
                    })
                }),
            },
            groups: Vec::new(),
        }
    }

    /// Add this tunable to a [`TuneGroup`] with the given intra-group priority.
    ///
    /// Groups are autotuned in order of their priority; within each group, tunables are
    /// tried in order of `priority(key)`. A negative priority skips the tunable for this
    /// key.
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
}

impl<K> core::fmt::Debug for TuneGroup<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TuneGroup").field("id", &self.id).finish()
    }
}

impl<K> Clone for TuneGroup<K> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            name: self.name.clone(),
            priority: self.priority.clone(),
        }
    }
}

impl<K> TuneGroup<K> {
    /// Create a new group based on a priority function.
    pub fn new(name: &str, f: impl Fn(&K) -> i8 + Send + Sync + 'static) -> Self {
        let id = GROUP_COUNTER.fetch_add(1, Ordering::Relaxed);

        Self {
            id,
            name: Arc::new(name.to_string()),
            priority: Arc::new(f),
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
    indices: HashMap<i8, Vec<(usize, Arc<String>)>>,
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

        for (index, tunable) in tunables.iter().enumerate() {
            if tunable.groups.is_empty() {
                no_groups.push(index);
            } else {
                for (group, within_group_priority_fn) in tunable.groups.iter() {
                    let priority_fn = &group.priority;
                    let priority = priority_fn(key);
                    if !priorities.contains(&priority) {
                        priorities.push(priority);
                    }

                    let group_priorities = match groups.get_mut(&priority) {
                        Some(val) => val,
                        None => {
                            groups.insert(priority, GroupPlan::default());
                            groups.get_mut(&priority).unwrap()
                        }
                    };
                    let priority = within_group_priority_fn(key);

                    if group_priorities.priorities.contains(&priority) {
                        group_priorities
                            .indices
                            .get_mut(&priority)
                            .unwrap()
                            .push((index, group.name.clone()));
                    } else {
                        group_priorities.priorities.push(priority);
                        group_priorities
                            .indices
                            .insert(priority, vec![(index, group.name.clone())]);
                    }
                }
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
    pub(crate) fn next(&mut self, mut context_logs: Option<&mut String>) -> Vec<usize> {
        let mut indices = core::mem::take(&mut self.no_groups);
        let priority = self.priorities.last();

        let priority = match priority {
            Some(val) => *val,
            None => return indices,
        };

        let (group_indices, cleanup) = self.group_plan_next(priority);
        // Some entries are skipped for this round of prioritizing.
        let skipped = cleanup.skipped || priority < 0;
        let mut all_skip = true;

        self.cleanup(cleanup);

        if priority >= 0 {
            if let Some(ctx) = context_logs.take() {
                *ctx += format!("\n - Tuning: {group_indices:?}").as_str();
                context_logs = Some(ctx);
            }
            for (index, _name) in group_indices {
                if !self.returned.contains(&index) {
                    all_skip = false;
                    indices.push(index);
                }
            }
        }

        // The indices list is empty, but it doesn't mean we should stop
        // autotuning, since some entries were skipped.

        if indices.is_empty() && (skipped || all_skip) {
            self.next(context_logs)
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

    fn group_plan_next(&mut self, priority: i8) -> (Vec<(usize, Arc<String>)>, Cleanup) {
        let group_plan = self.groups.get_mut(&priority).expect("To be filled");
        let within_group_prio = group_plan.priorities.pop().unwrap();
        let mut next_indices = group_plan.indices.remove(&within_group_prio).unwrap();

        let mut cleanup_groups = Vec::new();
        let mut cleanup_tunables = Vec::new();

        for (pg, group) in self.groups.iter_mut() {
            let mut num_empty_tunables = 0;
            let num_tunables = group.priorities.len();

            for (pt, indices) in group.indices.iter_mut() {
                for n in &next_indices {
                    let entry = indices.iter().enumerate().find(|p| *p.1 == *n);
                    if let Some(entry) = entry {
                        indices.remove(entry.0);
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
            next_indices,
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

        assert_eq!(plan.next(None), vec![0, 2]);
        assert_eq!(plan.next(None), vec![1]);
        assert_eq!(plan.next(None), vec![3]);
        assert!(plan.next(None).is_empty());
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

        assert_eq!(plan.next(None), vec![0, 2]);
        assert_eq!(plan.next(None), vec![1]);
        assert_eq!(plan.next(None), vec![3, 4]);
        assert!(plan.next(None).is_empty());
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

        assert_eq!(plan.next(None), vec![0, 3]);
        assert_eq!(plan.next(None), vec![1]);
        assert_eq!(plan.next(None), vec![2]);
        assert!(plan.next(None).is_empty());
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

        assert_eq!(plan.next(None), vec![0, 2]);
        assert_eq!(plan.next(None), vec![3]);
        assert!(plan.next(None).is_empty());
    }

    #[test_log::test]
    fn test_plan_no_group() {
        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new("fake", fake_kernel);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1]);

        assert_eq!(plan.next(None), vec![0, 1]);
        assert!(plan.next(None).is_empty());
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
            let batch = plan.next(None);
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

        assert_eq!(plan.next(None), vec![2]);
        assert_eq!(plan.next(None), vec![1]);
        assert_eq!(plan.next(None), vec![0]);
        assert!(plan.next(None).is_empty());
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

        assert_eq!(plan.next(None), vec![2]);
        assert!(plan.next(None).is_empty());
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

        assert_eq!(plan.next(None), vec![0, 1]);
        assert_eq!(plan.next(None), vec![2]);
        assert!(plan.next(None).is_empty());
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
            let batch = plan.next(None);
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
        assert_eq!(plan.next(None), vec![0]);
        // Second call: group_lo's higher intra-priority batch is just tunable0 (already
        // returned). Without the fix this returns [] and the autotuner aborts. With the fix
        // the plan recurses and yields tunable1.
        assert_eq!(plan.next(None), vec![1]);
        assert!(plan.next(None).is_empty());
    }

    fn fake_kernel(_: ()) -> Result<(), String> {
        Ok(())
    }
}
