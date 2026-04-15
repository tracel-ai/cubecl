use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneInputs, Tuner};
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
    ) where
        <I as TuneInputs>::At<'a>: Clone + Send,
    {
        use alloc::vec::Vec;

        let mut checks_outputs = Vec::new();
        for i in 0..operations.len() {
            let op = operations.fastest(i);
            let result = op.execute(inputs.clone());
            checks_outputs.push(result);
        }
        super::check_autotune_outputs(checks_outputs);
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

        // First, check for a cache hit under a read lock.
        if let TuneCacheResult::Hit { fastest_index } = tuner.fastest(&key) {
            #[cfg(feature = "autotune-checks")]
            self.checks::<I, Out>(&operations, &inputs);
            return operations
                .fastest(fastest_index)
                .execute(inputs)
                .expect("Should run when selected by autotune.");
        }

        let fastest = tuner.check_tune::<R, I, Out>(
            &key,
            &inputs,
            &operations,
            || operations.compute_checksum(),
            client,
        );

        match fastest {
            TuneCacheResult::Hit { fastest_index } => {
                #[cfg(feature = "autotune-checks")]
                self.checks::<I, Out>(&operations, &inputs);
                return operations
                    .fastest(fastest_index)
                    .execute(inputs)
                    .expect("Should run when selected by autotune.");
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
