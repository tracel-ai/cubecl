use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneInputs, Tuner};
#[cfg(feature = "autotune-checks")]
use crate::tune::AutotuneLoggerExt;
use crate::{client::Client, tune::TuneCacheResult};
use alloc::string::ToString;
use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    fmt::Display,
    hash::Hash,
};
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::{Mutex, RwLock};

/// The tunable sets a [`LocalTuner`] has built, keyed by device as well as by
/// initializer: a set is built from the device it will run on — its client, its
/// hardware properties — so one device's set cannot answer for another's. See
/// [`LocalTuner::init`].
type Sets<ID> = RwLock<Option<HashMap<(TypeId, ID), Arc<dyn Any + Send + Sync>>>>;

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    state: Mutex<Option<HashMap<ID, Arc<Tuner<AK>>>>>,
    name: &'static str,
    sets: Sets<ID>,
}

/// Create a local tuner with the provided name.
#[macro_export]
macro_rules! local_tuner {
    ($name:expr) => {
        LocalTuner::new(concat!(module_path!(), "-", $name));
    };
    () => {
        LocalTuner::new(module_path!());
    };
}

pub use local_tuner;

impl<AK, ID> LocalTuner<AK, ID>
where
    AK: AutotuneKey + 'static,
    ID: Hash + PartialEq + Eq + Clone + Display,
{
    /// Create a new local tuner.
    pub const fn new(name: &'static str) -> Self {
        Self {
            state: Mutex::new(None),
            name,
            sets: RwLock::new(None),
        }
    }

    /// Get or initialize the [`TunableSet`] for `id`.
    ///
    /// Returns a cached `Arc<TunableSet>` keyed by the `TypeId` of `init_set`
    /// *and* by `id`, so the initializer runs once per device rather than once
    /// per process.
    ///
    /// The device is part of the key because a set is routinely built from the
    /// device it will run on: a closure captures that device's
    /// [`Client`] to ask what it supports, or reads its hardware
    /// properties to decide which tunables are worth offering at all. Keyed by
    /// the initializer alone, whichever device tuned first would answer those
    /// questions for every device that followed — promoting kernels onto
    /// hardware that cannot run them, or withholding kernels from hardware
    /// that can.
    pub fn init<I, Out, F>(&self, id: &ID, init_set: F) -> Arc<TunableSet<AK, I, Out>>
    where
        F: Fn() -> TunableSet<AK, I, Out> + 'static + Send + Sync,
        I: TuneInputs,
        Out: AutotuneOutput,
    {
        let key = (TypeId::of::<F>(), id.clone());
        let sets = self.sets.read();

        static DOWNCAST_ERROR: &str = "Local tuner only support one set of tunable that must work on the same input and output declared with the init function.";

        if let Some(sets) = sets.as_ref()
            && let Some(set) = sets.get(&key)
        {
            return set.clone().downcast().expect(DOWNCAST_ERROR);
        };

        core::mem::drop(sets);

        let mut sets = self.sets.write();

        if let Some(sets) = sets.as_ref()
            && let Some(set) = sets.get(&key)
        {
            return set.clone().downcast().expect(DOWNCAST_ERROR);
        };

        let content = Arc::new(init_set());

        if let Some(sets) = sets.as_mut() {
            sets.insert(key, content.clone());
        } else {
            let mut map = HashMap::<(TypeId, ID), Arc<dyn Any + Send + Sync>>::new();
            map.insert(key, content.clone());
            *sets = Some(map);
        };

        content
    }

    /// Clear the autotune state.
    pub fn clear(&self) {
        if let Some(s) = self.state.lock().as_mut() {
            s.clear()
        }
    }

    #[cfg(feature = "autotune-checks")]
    fn checks<'a, I: TuneInputs, Out: AutotuneOutput>(
        &self,
        operations: &TunableSet<AK, I, Out>,
        inputs: &<I as TuneInputs>::At<'a>,
    ) -> alloc::vec::Vec<crate::tune::log::CheckResult>
    where
        <I as TuneInputs>::At<'a>: Clone + Send,
    {
        use alloc::vec::Vec;

        let mut checks_outputs = Vec::new();
        for i in 0..operations.len() {
            let op = operations.fastest(i);
            let result = op.execute(inputs.clone());
            checks_outputs.push((op.name.to_string(), result));
        }
        super::check_autotune_outputs(checks_outputs)
    }

    /// Execute the fastest operation in a [`TunableSet`], triggering a tuning pass on
    /// the first call for a given key.
    pub fn execute<'a, I: TuneInputs, Out>(
        &self,
        id: &ID,
        client: &Client,
        operations: Arc<TunableSet<AK, I, Out>>,
        inputs: <I as TuneInputs>::At<'a>,
    ) -> Out
    where
        <I as TuneInputs>::At<'a>: Clone + Send,
        Out: AutotuneOutput,
    {
        let key = operations.generate_key(&inputs);

        let tuner = {
            let mut state_lock = self.state.lock();
            let state_map = state_lock.get_or_insert_with(|| HashMap::new());
            state_map
                .entry(id.clone())
                .or_insert_with(move || {
                    let name = self.name.replace("::", "-");
                    Arc::new(Tuner::new(&name, &id.to_string()))
                })
                .clone()
        };

        #[allow(unused_mut)]
        let mut log_context = crate::tune::AutotuneLogContext::new(&mut tuner.logger().lock());

        #[cfg(feature = "autotune-checks")]
        log_context.set_checks(|| self.checks::<I, Out>(&operations, &inputs));

        // Fast path: a cached hit skips straight to the fastest operation.
        // `fastest` also resets the tuner cache if the environment switched, so
        // a miss here falls through to `check_tune`, which re-hydrates.
        if let TuneCacheResult::Hit { fastest_index } = tuner.fastest(&key) {
            return operations
                .fastest(fastest_index)
                .execute(inputs)
                .expect("Should run when selected by autotune.");
        }

        let fastest = tuner.check_tune::<I, Out>(
            &key,
            &inputs,
            &operations,
            || operations.compute_checksum(),
            client,
            log_context,
        );

        // Run the execution depending on the cache state.
        match fastest {
            TuneCacheResult::Hit { fastest_index } => operations
                .fastest(fastest_index)
                .execute(inputs)
                .expect("Should run when selected by autotune."),
            TuneCacheResult::Unchecked | TuneCacheResult::Miss => {
                panic!(
                    "Somehow we STILL didn't check a tuning checksum or start tuning, something has gone wrong."
                )
            }
            TuneCacheResult::Pending => {
                // Still waiting (e.g. on wasm): the candidates run in the plan's order, the one
                // the measurement follows, so the pending period runs the best ranked candidate
                // that launches rather than the first declared.
                execute_in_plan_order(&operations, &key, inputs)
                    .expect("All autotune operations failed, no viable operation found.")
            }
        }
    }
}

/// Runs the candidates in the order the plan measures them, then whatever the plan left out
/// in declaration order, and hands back the first output. A candidate that fails to launch is
/// skipped, as the measurement skips it.
fn execute_in_plan_order<'a, AK: AutotuneKey, I: TuneInputs, Out: 'static>(
    operations: &TunableSet<AK, I, Out>,
    key: &AK,
    inputs: <I as TuneInputs>::At<'a>,
) -> Option<Out>
where
    <I as TuneInputs>::At<'a>: Clone,
{
    let mut plan = operations.plan(key);
    let mut tried = alloc::vec::Vec::with_capacity(operations.len());
    loop {
        let batch = plan.next();
        if batch.is_empty() {
            break;
        }
        for index in batch {
            tried.push(index);
            if let Ok(output) = operations.fastest(index).execute(inputs.clone()) {
                return Some(output);
            }
        }
    }
    (0..operations.len())
        .filter(|index| !tried.contains(index))
        .find_map(|index| operations.fastest(index).execute(inputs.clone()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{Tunable, TuneGroup};
    use alloc::string::String;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Debug)]
    struct FakeAutotuneKey;

    impl Display for FakeAutotuneKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("FakeAutotuneKey")
        }
    }

    impl AutotuneKey for FakeAutotuneKey {}

    fn set() -> TunableSet<FakeAutotuneKey, (), usize> {
        TunableSet::new(|_: &()| FakeAutotuneKey, |_: &FakeAutotuneKey, _: &()| ())
    }

    fn answers(index: usize) -> impl Fn(()) -> Result<usize, String> {
        move |_| Ok(index)
    }

    fn fails(_: ()) -> Result<usize, String> {
        Err("does not launch here".into())
    }

    #[test]
    fn the_pending_fallback_runs_the_plan_s_first_candidate_not_the_first_declared() {
        let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
        let operations = set()
            .with(Tunable::<FakeAutotuneKey, (), usize>::new(
                "declared_first",
                answers(0),
            ))
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_first", answers(1))
                    .group(&group, |_| 1),
            );

        assert_eq!(
            execute_in_plan_order(&operations, &FakeAutotuneKey, ()),
            Some(1)
        );
    }

    #[test]
    fn the_pending_fallback_moves_down_the_plan_when_a_candidate_does_not_launch() {
        let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
        let operations = set()
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_last", answers(0))
                    .group(&group, |_| 0),
            )
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_first", fails)
                    .group(&group, |_| 2),
            )
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_second", answers(2))
                    .group(&group, |_| 1),
            );

        assert_eq!(
            execute_in_plan_order(&operations, &FakeAutotuneKey, ()),
            Some(2)
        );
    }

    #[test]
    fn the_pending_fallback_tries_what_the_plan_left_out_last() {
        let never = TuneGroup::<FakeAutotuneKey>::new("never", |_| -1);
        let operations = set()
            .with(Tunable::<FakeAutotuneKey, (), usize>::new(
                "declared_first",
                fails,
            ))
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("left_out", answers(1))
                    .group(&never, |_| 1),
            );

        assert_eq!(
            execute_in_plan_order(&operations, &FakeAutotuneKey, ()),
            Some(1)
        );
        let nothing = set().with(Tunable::<FakeAutotuneKey, (), usize>::new("fails", fails));
        assert_eq!(execute_in_plan_order(&nothing, &FakeAutotuneKey, ()), None);
    }
}
