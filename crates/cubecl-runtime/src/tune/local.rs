use super::{AutotuneKey, AutotuneOutput, TunableSet, Tuner};
use crate::{client::ComputeClient, runtime::Runtime, tune::TuneCacheResult};
use alloc::string::ToString;
use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    fmt::Display,
    hash::Hash,
};
use cubecl_common::map::SharedStateMap;
use hashbrown::HashMap;

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    state: SharedStateMap<ID, Tuner<AK>>,
    name: &'static str,
    sets: spin::RwLock<Option<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
}

unsafe impl<AK: AutotuneKey, ID> Sync for LocalTuner<AK, ID> {}

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
            state: SharedStateMap::new(),
            name,
            sets: spin::RwLock::new(None),
        }
    }

    /// Init the [tunable set](TunableSet)
    pub fn init<In, Out, F>(&self, init_set: F) -> Arc<TunableSet<AK, In, Out>>
    where
        F: Fn() -> TunableSet<AK, In, Out> + 'static + Send + Sync,
        In: Clone + Send + 'static,
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
        self.state.clear()
    }

    #[cfg(feature = "autotune-checks")]
    fn checks<In: Send + Clone + 'static, Out: AutotuneOutput>(
        &self,
        operations: &TunableSet<AK, In, Out>,
        inputs: &In,
    ) {
        use alloc::vec::Vec;

        let mut checks_outputs = Vec::new();
        for i in 0..operations.len() {
            let op = operations.fastest(i);
            let result = op.execute(inputs.clone());
            checks_outputs.push(result);
        }
        super::check_autotune_outputs(checks_outputs);
    }

    /// Try every operation in order and return the first successful result.
    ///
    /// Used as a fallback when autotuning results aren't available yet
    /// (e.g. on wasm where tuning is async).
    fn try_all_operations<In, Out>(operations: &TunableSet<AK, In, Out>, inputs: In) -> Out
    where
        In: Clone + Send + 'static,
        Out: AutotuneOutput,
    {
        for i in 0..operations.len() {
            if let Ok(output) = operations.fastest(i).execute(inputs.clone()) {
                return output;
            }
        }
        panic!("All autotune operations failed, no viable operation found.");
    }

    /// Execute the best operation in the provided [tunable set](TunableSet)
    pub fn execute<R: Runtime, In, Out>(
        &self,
        id: &ID,
        client: &ComputeClient<R>,
        operations: Arc<TunableSet<AK, In, Out>>,
        inputs: In,
    ) -> Out
    where
        In: Clone + Send + 'static,
        Out: AutotuneOutput,
    {
        let key = operations.generate_key(&inputs);

        // If this is cached and ready, use the operation.
        let tuner_state = self.state.get_or_init(id, move |id| {
            let name = self.name.replace("::", "-");
            Tuner::new(&name, &id.to_string())
        });
        let tuner = tuner_state.read();

        let mut tuner = match tuner.fastest(&key) {
            TuneCacheResult::Hit { fastest_index } => {
                core::mem::drop(tuner);
                core::mem::drop(tuner_state);

                #[cfg(feature = "autotune-checks")]
                self.checks(&operations, &inputs);

                let op = operations.fastest(fastest_index);
                let result = op
                    .execute(inputs)
                    .expect("Should run when selected by autotune.");
                return result;
            }
            TuneCacheResult::Pending => {
                core::mem::drop(tuner);
                core::mem::drop(tuner_state);

                #[cfg(feature = "autotune-checks")]
                self.checks(&operations, &inputs);

                return Self::try_all_operations(&operations, inputs);
            }
            #[cfg(std_io)]
            TuneCacheResult::Unchecked => {
                core::mem::drop(tuner);
                let mut tuner = tuner_state.write();

                // If the cache checksum hasn't been checked, do so now, and retry.
                let checksum = operations.compute_checksum();
                tuner.validate_checksum(&key, &checksum);

                // Check if with validation we can use its result
                if let TuneCacheResult::Hit { fastest_index } = tuner.fastest(&key) {
                    core::mem::drop(tuner);
                    core::mem::drop(tuner_state);

                    let op = operations.fastest(fastest_index);
                    let result = op
                        .execute(inputs)
                        .expect("Should run when selected by autotune.");
                    return result;
                }

                tuner
            }

            #[cfg(not(std_io))]
            TuneCacheResult::Unchecked => {
                core::mem::drop(tuner);
                tuner_state.write()
            }
            TuneCacheResult::Miss => {
                core::mem::drop(tuner);
                tuner_state.write()
            }
        };

        let job = if !tuner.autotuning.contains(&key) {
            tuner.autotuning.insert(key.clone());
            Some(tuner.prepare_autotune(key.clone(), &inputs, &operations, client))
        } else {
            None
        };

        // Drop the write lock before running the (potentially blocking) job
        // and before re-acquiring the lock below.
        core::mem::drop(tuner);
        core::mem::drop(tuner_state);

        if let Some(job) = job {
            job();
        }

        let index_to_run = {
            let tuner_state = self.state.get(id).unwrap();
            let mut tuner = tuner_state.write();

            tuner.handle_results();

            match tuner.fastest(&key) {
                TuneCacheResult::Hit { fastest_index } => {
                    // There's a known good value - just run it.
                    fastest_index
                }
                TuneCacheResult::Pending | TuneCacheResult::Miss => {
                    // We're still waiting for the results of the autotune task.
                    // This should only happen on wasm since we can't block waiting
                    // on the results there. Try all options.
                    return Self::try_all_operations(&operations, inputs);
                }
                TuneCacheResult::Unchecked => {
                    panic!("Should have checked the cache.")
                }
            }
        };

        operations
            .fastest(index_to_run)
            .execute(inputs)
            .expect("Should run when selected by autotune.")
    }
}
