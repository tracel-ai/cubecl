use super::{AutotuneKey, AutotuneOutput, TunableSet, Tuner};
use crate::{client::ComputeClient, runtime::Runtime, tune::TuneCacheResult};
use alloc::string::ToString;
use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    fmt::Display,
    hash::Hash,
};
use hashbrown::HashMap;
use spin::Mutex;

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    state: Mutex<Option<HashMap<ID, Arc<Tuner<AK>>>>>,
    name: &'static str,
    sets: spin::RwLock<Option<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
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
        if let Some(s) = self.state.lock().as_mut() {
            s.clear()
        }
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

        // First, check for a cache hit under a read lock.
        if let TuneCacheResult::Hit { fastest_index } = tuner.fastest(&key) {
            #[cfg(feature = "autotune-checks")]
            self.checks(&operations, &inputs);
            return operations
                .fastest(fastest_index)
                .execute(inputs)
                .expect("Should run when selected by autotune.");
        }

        // Checksum validation may retroactively turn an Unchecked entry into a Hit.
        #[cfg(std_io)]
        if matches!(tuner.fastest(&key), TuneCacheResult::Unchecked) {
            let checksum = operations.compute_checksum();
            tuner.validate_checksum(&key, &checksum);
        }

        // Resolve the cache state into a `done_rx` we can wait on. Hit → run immediately;
        // Pending → attach to the in-flight tune; Miss → kick one off.
        let done_rx = match tuner.fastest(&key) {
            TuneCacheResult::Hit { fastest_index } => {
                #[cfg(feature = "autotune-checks")]
                self.checks(&operations, &inputs);

                return operations
                    .fastest(fastest_index)
                    .execute(inputs)
                    .expect("Should run when selected by autotune.");
            }
            TuneCacheResult::Unchecked => {
                panic!("Somehow we STILL didn't check a tuning checksum, something has gone wrong.")
            }
            TuneCacheResult::Pending(done_rx) => done_rx,
            TuneCacheResult::Miss => {
                // `tune` atomically claims the key under the cache mutex — if we lost the race
                // to another thread, it returns a receiver for the existing in-flight job
                // (or an already-closed receiver for the Hit-race case).
                tuner.tune(key.clone(), inputs.clone(), &operations, client)
            }
        };

        // If we're still waiting for the result, eg. on wasm, just fallback to trying all operations.
        if done_rx.try_recv().is_err() {
            let operations: &TunableSet<AK, In, Out> = &operations;
            for i in 0..operations.len() {
                if let Ok(output) = operations.fastest(i).execute(inputs.clone()) {
                    return output;
                }
            }
            panic!("All autotune operations failed, no viable operation found.");
        }
        let fastest = tuner.fastest(&key);
        let TuneCacheResult::Hit { fastest_index } = fastest else {
            panic!("Something went wrong: expected a hit, got {fastest:?}")
        };
        operations
            .fastest(fastest_index)
            .execute(inputs)
            .expect("Should run when selected by autotune.")
    }
}
