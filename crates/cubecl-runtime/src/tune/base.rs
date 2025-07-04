use super::{AutotuneKey, IntoTuneFn, TuneFn};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use hashbrown::HashMap;

/// A tunable wraps a [function](TuneFn) that can be included in multiple [groups](TuneGroup).
///
/// When a tunable is part of multiple groups, it will be autotuned when one of those groups is
/// prioritized.
pub struct Tunable<K, Inputs, Output> {
    pub(crate) function: Arc<dyn TuneFn<Inputs = Inputs, Output = Output>>,
    groups: Vec<(TuneGroup<K>, PriorityFunc<K>)>,
}

impl<K, Inputs, Output> Tunable<K, Inputs, Output> {
    /// Create a new tunable based on a function.
    pub fn new<Marker>(function: impl IntoTuneFn<Inputs, Output, Marker>) -> Self {
        Self {
            function: Arc::new(function.into_tunable()),
            groups: Vec::new(),
        }
    }

    /// Tag the current tunable as part of the given [group](TuneGroup).
    pub fn group<F: Fn(&K) -> u8 + 'static>(mut self, group: &TuneGroup<K>, priority: F) -> Self {
        self.groups.push((group.clone(), Arc::new(priority)));
        self
    }
}

/// A tune group encapsulates a priority that can be calculated based on an
/// [autotune key](AutotuneKey).
///
/// During autotuning, the higher prioritized groups will be autotuned first, and if a tunable
/// returns a valid result, no more groups will be autotuned afterward.
///
/// Note that tunables themselves have a priority dictating the order in which they are autotuned in
/// each group.
pub struct TuneGroup<K> {
    id: u32,
    pub(crate) priority: PriorityFunc<K>,
}

impl<K> Clone for TuneGroup<K> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            priority: self.priority.clone(),
        }
    }
}

impl<K> TuneGroup<K> {
    /// Create a new group based on a priority function.
    pub fn new<F: Fn(&K) -> u8 + 'static>(f: F) -> Self {
        let id = GROUP_COUNTER.fetch_add(1, Ordering::Relaxed);

        Self {
            id,
            priority: Arc::new(f),
        }
    }
}

#[derive(Debug)]
/// A group plan dictates which [tunables](Tunable) should be executed, and in what order.
pub(crate) struct TunePlan {
    priorities: Vec<u8>,
    no_groups: Vec<usize>,
    groups: HashMap<u8, GroupPlan>,
}

#[derive(Default, Debug)]
struct GroupPlan {
    priorities: Vec<u8>,
    indices: HashMap<u8, Vec<usize>>,
}

struct Cleanup {
    groups: Vec<u8>,
    tunables: Vec<(u8, u8)>,
}

impl TunePlan {
    pub fn new<K: AutotuneKey, In, Out>(key: &K, tunables: &[Tunable<K, In, Out>]) -> Self {
        let mut priorities = Vec::<u8>::new();
        let mut no_groups = Vec::new();
        let mut groups = HashMap::<u8, GroupPlan>::new();

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
                            .push(index);
                    } else {
                        group_priorities.priorities.push(priority);
                        group_priorities.indices.insert(priority, vec![index]);
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
        }
    }

    /// Get the next batch of [tunable](Tunable) index to be autotuned.
    ///
    /// Note that if the list is empty, it means no more autotuned entry can be executed.
    pub(crate) fn next(&mut self) -> Vec<usize> {
        let mut indices = core::mem::take(&mut self.no_groups);
        let priority = self.priorities.last();

        let priority = match priority {
            Some(val) => val,
            None => return indices,
        };

        let (mut group_indices, cleanup) = self.group_plan_next(*priority);
        self.cleanup(cleanup);
        indices.append(&mut group_indices);

        indices
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

    fn group_plan_next(&mut self, priority: u8) -> (Vec<usize>, Cleanup) {
        let plan = self.groups.get_mut(&priority).expect("To be filled");
        let within_group_prio = plan.priorities.pop().unwrap();
        let next_indices = plan.indices.remove(&within_group_prio).unwrap();

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

        (
            next_indices,
            Cleanup {
                groups: cleanup_groups,
                tunables: cleanup_tunables,
            },
        )
    }
}

type PriorityFunc<K> = Arc<dyn Fn(&K) -> u8>;

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

    #[test]
    fn test_plan_order() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new(|_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new(|_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group0, |_| 1);
        let tunable2 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group0, |_| 2);
        let tunable3 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group1, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(plan.next(), vec![0, 2]);
        assert_eq!(plan.next(), vec![1]);
        assert_eq!(plan.next(), vec![3]);
        assert!(plan.next().is_empty());
    }

    #[test]
    fn test_plan_order_multi_groups_same_priority() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new(|_| 2);
        let group1 = TuneGroup::<FakeAutotuneKey>::new(|_| 1);
        let group2 = TuneGroup::<FakeAutotuneKey>::new(|_| 1);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group0, |_| 1);
        let tunable2 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group0, |_| 2);
        let tunable3 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group1, |_| 2);
        let tunable4 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group2, |_| 2);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3, tunable4]);

        assert_eq!(plan.next(), vec![0, 2]);
        assert_eq!(plan.next(), vec![1]);
        assert_eq!(plan.next(), vec![3, 4]);
        assert!(plan.next().is_empty());
    }

    #[test]
    fn test_plan_order_tunable_multiple_groups() {
        let group0 = TuneGroup::<FakeAutotuneKey>::new(|_| 1);
        let group1 = TuneGroup::<FakeAutotuneKey>::new(|_| 2);

        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel)
            .group(&group0, |_| 1)
            .group(&group1, |_| 2);
        let tunable2 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group0, |_| 2);
        let tunable3 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel).group(&group1, |_| 3);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1, tunable2, tunable3]);

        assert_eq!(plan.next(), vec![0, 3]);
        assert_eq!(plan.next(), vec![1]);
        assert_eq!(plan.next(), vec![2]);
        assert!(plan.next().is_empty());
    }

    #[test]
    fn test_plan_no_group() {
        let tunable0 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel);
        let tunable1 = Tunable::<FakeAutotuneKey, (), ()>::new(fake_kernel);

        let key = FakeAutotuneKey;
        let mut plan = TunePlan::new(&key, &[tunable0, tunable1]);

        assert_eq!(plan.next(), vec![0, 1]);
        assert!(plan.next().is_empty());
    }

    fn fake_kernel() -> Result<(), String> {
        Ok(())
    }
}
