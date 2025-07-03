use core::sync::atomic::{AtomicU32, Ordering};

use alloc::sync::Arc;
use hashbrown::HashMap;

use super::{AutotuneKey, IntoTuneFn, TuneFn};

type PriorityFunc<K> = Arc<dyn Fn(&K) -> u8>;

static GROUP_COUNTER: AtomicU32 = AtomicU32::new(0);

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
    pub fn new<F: Fn(&K) -> u8 + 'static>(f: F) -> Self {
        let id = GROUP_COUNTER.fetch_add(1, Ordering::Relaxed);

        Self {
            id,
            priority: Arc::new(f),
        }
    }
}

pub struct Tunable<K, Inputs, Output> {
    pub(crate) function: Arc<dyn TuneFn<Inputs = Inputs, Output = Output>>,
    groups: Vec<(TuneGroup<K>, PriorityFunc<K>)>,
}

impl<K, Inputs, Output> Tunable<K, Inputs, Output> {
    pub fn new<Marker>(function: impl IntoTuneFn<Inputs, Output, Marker>) -> Self {
        Self {
            function: Arc::new(function.into_tunable()),
            groups: Vec::new(),
        }
    }
    pub fn group<F: Fn(&K) -> u8 + 'static>(mut self, group: &TuneGroup<K>, priority: F) -> Self {
        self.groups.push((group.clone(), Arc::new(priority)));
        self
    }
}

#[derive(Debug)]
pub struct TunePlan {
    priorities: Vec<(u32, u8)>,
    no_groups: Vec<usize>,
    groups: HashMap<u32, GroupPlan>,
}

#[derive(Default, Debug)]
struct GroupPlan {
    priorities: Vec<u8>,
    indices: HashMap<u8, Vec<usize>>,
}

impl TunePlan {
    pub fn new<K: AutotuneKey, In, Out>(key: &K, tunables: &[Tunable<K, In, Out>]) -> Self {
        let mut priorities = Vec::<(u32, u8)>::new();
        let mut no_groups = Vec::new();
        let mut groups = HashMap::<u32, GroupPlan>::new();

        for (index, tunable) in tunables.iter().enumerate() {
            if tunable.groups.is_empty() {
                no_groups.push(index);
            } else {
                for (group, within_group_priority_fn) in tunable.groups.iter() {
                    let group_priorities = match groups.get_mut(&group.id) {
                        Some(val) => val,
                        None => {
                            let priority_fn = &group.priority;
                            let priority = priority_fn(key);
                            priorities.push((group.id, priority));
                            groups.insert(group.id, GroupPlan::default());
                            groups.get_mut(&group.id).unwrap()
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

        priorities.sort_by(|(_, priority_a), (_, priority_b)| priority_b.cmp(priority_a));

        for group in groups.iter_mut() {
            group.1.priorities.sort();
        }

        Self {
            priorities,
            no_groups,
            groups,
        }
    }

    pub fn next(&mut self) -> Vec<usize> {
        let mut indices = core::mem::take(&mut self.no_groups);
        let group = self.priorities.last();

        let mut cleanup = false;

        if let Some(group) = group {
            let plan = self.groups.get_mut(&group.0).expect("To be filled");
            let within_group_prio = plan.priorities.pop().unwrap();
            let mut next_indices = plan.indices.remove(&within_group_prio).unwrap();

            indices.append(&mut next_indices);

            if plan.priorities.is_empty() {
                cleanup = true;
            }
        };

        if cleanup {
            let group_prio = self.priorities.pop().unwrap();
            self.groups.remove(&group_prio.0);
        }

        indices
    }
}
