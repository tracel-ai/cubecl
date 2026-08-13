use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneInputs, Tuner};
#[cfg(feature = "autotune-checks")]
use crate::tune::AutotuneLoggerExt;
use crate::{client::ComputeClient, runtime::Runtime, tune::TuneCacheResult};
use alloc::string::ToString;
use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    fmt::Display,
    hash::Hash,
};
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::{Mutex, RwLock};

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    state: Mutex<Option<HashMap<ID, Arc<Tuner<AK>>>>>,
    name: &'static str,
    sets: RwLock<Option<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
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

    /// Get or initialize the [`TunableSet`] for this tuner.
    ///
    /// Returns a cached `Arc<TunableSet>` keyed by the `TypeId` of `init_set`. The
    /// initializer runs at most once per process.
    pub fn init<I, Out, F>(&self, init_set: F) -> Arc<TunableSet<AK, I, Out>>
    where
        F: Fn() -> TunableSet<AK, I, Out> + 'static + Send + Sync,
        I: TuneInputs,
        Out: AutotuneOutput,
    {
        let sets = self.sets.read();
        let type_id = TypeId::of::<F>();

        static DOWNCAST_ERROR: &str = "Local tuner only support one set of tunable that must work on the same input and output declared with the init function.";

        if let Some(sets) = sets.as_ref()
            && let Some(set) = sets.get(&type_id)
        {
            return set.clone().downcast().expect(DOWNCAST_ERROR);
        };

        core::mem::drop(sets);

        let mut sets = self.sets.write();

        if let Some(sets) = sets.as_ref()
            && let Some(set) = sets.get(&type_id)
        {
            return set.clone().downcast().expect(DOWNCAST_ERROR);
        };

        let content = Arc::new(init_set());

        if let Some(sets) = sets.as_mut() {
            sets.insert(type_id, content.clone());
        } else {
            let mut map = HashMap::<TypeId, Arc<dyn Any + Send + Sync>>::new();
            map.insert(type_id, content.clone());
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

    /// Run the cache-hit winner; if its launch fails, fall back to the
    /// remaining candidates instead of panicking.
    ///
    /// The autotune key does not capture everything launch-time validation
    /// checks (and benchmark inputs are not always identical to the real
    /// ones), so a winner tuned on one representative can be an invalid
    /// config for a later cache hit. Panicking here is not a recoverable
    /// position for the caller: on an async device runner the panic is
    /// caught and warn-logged by the task loop while the op's registered
    /// outputs stay unwritten, which surfaces as silent NaN corruption in
    /// training. Falling back trades one slow launch for a correct result.
    fn execute_hit<'a, I: TuneInputs, Out: AutotuneOutput>(
        operations: &TunableSet<AK, I, Out>,
        inputs: <I as TuneInputs>::At<'a>,
        fastest_index: usize,
    ) -> Out
    where
        <I as TuneInputs>::At<'a>: Clone + Send,
    {
        match operations.fastest(fastest_index).execute(inputs.clone()) {
            Ok(out) => out,
            Err(err) => {
                log::warn!(
                    "Autotune winner {fastest_index} failed at launch ({err:?}); falling back to remaining candidates."
                );
                for i in 0..operations.len() {
                    if i == fastest_index {
                        continue;
                    }
                    if let Ok(out) = operations.fastest(i).execute(inputs.clone()) {
                        return out;
                    }
                }
                panic!("All autotune operations failed after the selected winner failed: {err:?}");
            }
        }
    }

    /// Execute the fastest operation in a [`TunableSet`], triggering a tuning pass on
    /// the first call for a given key.
    pub fn execute<'a, R: Runtime, I: TuneInputs, Out>(
        &self,
        id: &ID,
        client: &ComputeClient<R>,
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
            return Self::execute_hit(&operations, inputs, fastest_index);
        }

        let fastest = tuner.check_tune::<R, I, Out>(
            &key,
            &inputs,
            &operations,
            || operations.compute_checksum(),
            client,
            log_context,
        );

        // Run the execution depending on the cache state.
        match fastest {
            TuneCacheResult::Hit { fastest_index } => {
                Self::execute_hit(&operations, inputs, fastest_index)
            }
            TuneCacheResult::Unchecked | TuneCacheResult::Miss => {
                panic!(
                    "Somehow we STILL didn't check a tuning checksum or start tuning, something has gone wrong."
                )
            }
            TuneCacheResult::Pending => {
                // Still waiting (e.g. on wasm). Try all operations as a fallback.
                for i in 0..operations.len() {
                    if let Ok(output) = operations.fastest(i).execute(inputs.clone()) {
                        return output;
                    }
                }
                panic!("All autotune operations failed, no viable operation found.");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{Tunable, TunableSet};
    use alloc::string::{String, ToString};
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::fmt::Display;
    use cubecl_environment::sync::Mutex;

    #[derive(Hash, Eq, PartialEq, Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct FakeKey;

    impl Display for FakeKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("FakeKey")
        }
    }

    impl AutotuneKey for FakeKey {}

    /// The autotune key does not capture everything launch-time validation
    /// checks, so a cached winner can fail on a later cache hit even though
    /// it benchmarked fine. That failure must fall back to the remaining
    /// candidates (in order, skipping the winner) instead of panicking.
    #[test_log::test]
    fn cache_hit_winner_failure_falls_back() {
        let ran: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log = |name: &'static str, ok: bool| {
            let ran = ran.clone();
            move |_: u32| {
                ran.lock().push(name.to_string());
                if ok { Ok(()) } else { Err("invalid config") }
            }
        };

        let set: TunableSet<FakeKey, u32, ()> = TunableSet::new_cloning_inputs(|_: &u32| FakeKey)
            .with(Tunable::new(
                "winner_now_invalid",
                log("winner_now_invalid", false),
            ))
            .with(Tunable::new("also_broken", log("also_broken", false)))
            .with(Tunable::new("works", log("works", true)));

        LocalTuner::<FakeKey, String>::execute_hit(&set, 7u32, 0);

        assert_eq!(
            ran.lock().as_slice(),
            ["winner_now_invalid", "also_broken", "works"]
        );
    }

    /// A winner that keeps working takes the fast path untouched.
    #[test_log::test]
    fn cache_hit_winner_success_runs_alone() {
        let ran: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let ran2 = ran.clone();
        let ran3 = ran.clone();

        let set: TunableSet<FakeKey, u32, ()> = TunableSet::new_cloning_inputs(|_: &u32| FakeKey)
            .with(Tunable::new("winner", move |_: u32| {
                ran2.lock().push("winner".to_string());
                Ok::<(), String>(())
            }))
            .with(Tunable::new("unused", move |_: u32| {
                ran3.lock().push("unused".to_string());
                Ok::<(), String>(())
            }));

        LocalTuner::<FakeKey, String>::execute_hit(&set, 7u32, 0);

        assert_eq!(ran.lock().as_slice(), ["winner"]);
    }
}
