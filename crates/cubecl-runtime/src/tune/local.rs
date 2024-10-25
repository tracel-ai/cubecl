use super::{AutotuneKey, AutotuneOperationSet, Tuner};
use crate::{
    channel::ComputeChannel, client::ComputeClient, server::ComputeServer, tune::TuneCacheResult,
};
use core::{fmt::Display, hash::Hash};
use hashbrown::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::ToString};

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    state: spin::RwLock<Option<HashMap<ID, Tuner<AK>>>>,
    name: &'static str,
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

impl<AK: AutotuneKey + 'static, ID: Hash + PartialEq + Eq + Clone + Display> LocalTuner<AK, ID> {
    /// Create a new local tuner.
    pub const fn new(name: &'static str) -> Self {
        Self {
            state: spin::RwLock::new(None),
            name,
        }
    }

    /// Clear the autotune state.
    pub fn clear(&self) {
        let mut state = self.state.write();
        *state = None;
    }

    /// Execute the best operation in the provided [autotune operation set](AutotuneOperationSet)
    pub fn execute<S, C, Out: Send + 'static>(
        &self,
        id: &ID,
        client: &ComputeClient<S, C>,
        autotune_operation_set: Box<dyn AutotuneOperationSet<AK, Out>>,
    ) -> Out
    where
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    {
        // We avoid locking in write mode when possible.
        //
        // This makes us potentially check the cache twice, but allows to avoid
        // locking the state if the cache is hit.
        let mut should_init = true;
        #[cfg(autotune_persistent_cache)]
        let mut should_perform_checksum = false;

        if let Some(state) = self.state.read().as_ref() {
            if let Some(tuner) = state.get(id) {
                // Tuner exists for the given ID.
                should_init = false;

                let key = autotune_operation_set.key();
                match tuner.fastest(&key) {
                    TuneCacheResult::Hit { fastest_index } => {
                        let op = autotune_operation_set.fastest(fastest_index);
                        return op.execute();
                    }
                    TuneCacheResult::Miss => {}
                    #[cfg(autotune_persistent_cache)]
                    TuneCacheResult::Unchecked => {
                        should_perform_checksum = true;
                    }
                }
            }
        }

        // If we are not able to get a tuner for the given ID, we have to create one.
        if should_init {
            let mut state = self.state.write();
            let map = state.get_or_insert_with(Default::default);

            if !map.contains_key(id) {
                let name = self.name.replace("::", "-");
                let tuner = Tuner::new(&name, &id.to_string());
                map.insert(id.clone(), tuner);
            };
        }

        // When loading the first time the result of an autotune set, we need to verify the checksum.
        #[cfg(autotune_persistent_cache)]
        if should_perform_checksum {
            let mut state = self.state.write();
            let map = state.get_or_insert_with(Default::default);

            if let Some(tuner) = map.get_mut(id) {
                if let TuneCacheResult::Hit { fastest_index } =
                    tuner.fastest_with_checksum(autotune_operation_set.as_ref())
                {
                    let op = autotune_operation_set.fastest(fastest_index);
                    return op.execute();
                }
            }
        }

        // Running benchmarks shound't lock the tuner, since an autotune operation can recursively use the
        // same tuner.
        //
        // # Example
        //
        // ```
        // - tune_1 start
        //   - tune_2 start
        //   - tune_2 save
        // - tune_1 save
        // ```
        let mut result = None;
        if let Some(state) = self.state.read().as_ref() {
            if let Some(tuner) = state.get(id) {
                result = Some(tuner.execute_autotune(autotune_operation_set, client));
            }
        }

        // We store the autotune result if present.
        if let Some(result) = result {
            let mut state = self.state.write();
            let map = state.get_or_insert_with(Default::default);

            if let Some(tuner) = map.get_mut(id) {
                return tuner.register_autotune(result);
            }
        }

        // The Result and Tuner for this ID must exist at this point since we validated them earlier.
        unreachable!();
    }

    /// Return the autotune result given a key.
    pub fn autotune_result(&self, id: &ID, key: &AK) -> TuneCacheResult {
        if let Some(state) = self.state.read().as_ref() {
            if let Some(tuner) = state.get(id) {
                return tuner.fastest(key);
            }
        }

        TuneCacheResult::Miss
    }
}
