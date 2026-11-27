use alloc::sync::Arc;
use alloc::vec::Vec;
use core::hash::Hash;
use cubecl_common::hash::{StableHash, StableHasher};
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::Mutex;

use crate::tune::plan::planner::TunePlanner;
use crate::tune::set::tune_fn::TuneFn;
use crate::tune::{AutotuneKeyable, Tunable, TuneInputs};

/// Plans the candidates of a [`TunableSet`](crate::tune::TunableSet): holds them in registration order,
/// and the order each key runs them in, planned once a key.
pub struct CandidatePlanner<K, F: TuneInputs, Out, Id> {
    /// The candidates in registration order (Arrival Usage), each found by its index (Position
    /// Usage).
    tunables: Vec<Tunable<K, F, Out, Id>>,
    /// The run order of each key planned, by key (Keyed Usage). Planned once a key: ranking
    /// candidates can build every one's blueprint, which costs more than the launch, and an
    /// untuned key is reached at every layer of a compile pass.
    run_orders: Mutex<HashMap<K, Arc<[usize]>>>,
}

impl<K: AutotuneKeyable, F: TuneInputs, Out: 'static, Id> CandidatePlanner<K, F, Out, Id> {
    pub fn new() -> Self {
        Self {
            tunables: Vec::new(),
            run_orders: Mutex::new(HashMap::new()),
        }
    }

    /// Add `tunable` after the others, forgetting every run order planned without it, and answer
    /// the [hash](names_hash) of the names of the candidates the planner now holds.
    pub fn add(&mut self, tunable: Tunable<K, F, Out, Id>) -> StableHash {
        self.tunables.push(tunable);
        self.run_orders.lock().clear();
        names_hash(&self.tunables)
    }

    /// Every candidate's function, in registration order.
    pub fn functions(&self) -> impl ExactSizeIterator<Item = &TuneFn<F, Out>> {
        self.tunables.iter().map(Tunable::function)
    }

    /// The candidate at `index`, an index into the registration order.
    pub fn tunable_at(&self, index: usize) -> &Tunable<K, F, Out, Id> {
        &self.tunables[index]
    }

    /// The plan of the candidates for `key`.
    pub fn plan(&self, key: &K) -> TunePlanner {
        TunePlanner::new(key, &self.tunables)
    }

    /// The candidates of `key` in the order an untuned key runs them: its plan's
    /// [order](TunePlanner::order).
    pub fn plan_run_order(&self, key: &K) -> Arc<[usize]> {
        if let Some(order) = self.run_orders.lock().get(key) {
            return order.clone();
        }
        // Planned outside the lock: a priority may run for long.
        let order: Arc<[usize]> = self
            .plan(key)
            .order()
            .into_iter()
            .map(|(index, _)| index)
            .collect();
        self.run_orders.lock().insert(key.clone(), order.clone());
        order
    }
}

/// The hash of the names of `tunables`, in order. Each name is hashed with its own terminator, so
/// `["a", "bc"]` and `["abc"]` differ.
pub fn names_hash<K, F: TuneInputs, Out: 'static, Id>(
    tunables: &[Tunable<K, F, Out, Id>],
) -> StableHash {
    let mut hasher = StableHasher::new();
    for tunable in tunables {
        tunable.function().name().hash(&mut hasher);
    }
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::TuneGroup;
    use alloc::string::String;
    use core::sync::atomic::{AtomicUsize, Ordering};

    fn tunable(name: &str, index: usize) -> Tunable<String, (), usize> {
        Tunable::new(name, move |_: ()| Ok::<_, String>(index))
    }

    #[test]
    fn an_untuned_key_is_planned_once() {
        let asked = Arc::new(AtomicUsize::new(0));
        let counted = asked.clone();
        let group = TuneGroup::<String>::new("group", move |_| {
            counted.fetch_add(1, Ordering::Relaxed);
            1
        });
        let mut planner = CandidatePlanner::<String, (), usize, ()>::new();
        planner.add(tunable("first", 0).group(&group, |_| 1));
        planner.add(tunable("second", 1).group(&group, |_| 2));
        let key = String::from("key");

        let first = planner.plan_run_order(&key);
        let planned = asked.load(Ordering::Relaxed);
        let again = planner.plan_run_order(&key);

        assert_eq!((&*first, &*again), (&[1, 0][..], &[1, 0][..]));
        assert_eq!(asked.load(Ordering::Relaxed), planned);
    }

    #[test]
    fn a_candidate_added_after_planning_is_planned_in() {
        let group = TuneGroup::<String>::new("group", |_| 1);
        let mut planner = CandidatePlanner::<String, (), usize, ()>::new();
        planner.add(tunable("first", 0));
        planner.add(tunable("second", 1));
        let key = String::from("key");
        assert_eq!(&*planner.plan_run_order(&key), &[0, 1]);

        planner.add(tunable("added", 2).group(&group, |_| 1));

        assert_eq!(planner.plan_run_order(&key)[0], 2);
    }

    #[test]
    fn the_names_hash_tells_a_split_name_from_a_joined_one() {
        let mut split = CandidatePlanner::<String, (), usize, ()>::new();
        split.add(tunable("a", 0));
        let split = split.add(tunable("bc", 1));
        let joined = CandidatePlanner::<String, (), usize, ()>::new().add(tunable("abc", 0));

        assert_ne!(split, joined);
    }
}
