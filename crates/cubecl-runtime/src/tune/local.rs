use super::{AutotuneKey, AutotuneOperationSet, Tuner};
use crate::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::ComputeServer,
    tune::{AutotuneError, TuneCacheResult},
};
use core::{fmt::Display, hash::Hash};
use hashbrown::HashMap;

#[cfg(target_family = "wasm")]
use crate::tune::AutotuneOperation;

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
        enum Action {
            PerformInit,
            #[cfg(target_family = "wasm")]
            RegisterDeferred,
            #[cfg(autotune_persistent_cache)]
            ValidateChecksum,
            Autotune,
        }

        let action = match self.state.read().as_ref() {
            Some(state) => match state.get(id) {
                Some(tuner) => {
                    let key = autotune_operation_set.key();

                    match tuner.fastest(&key) {
                        TuneCacheResult::Hit { fastest_index } => {
                            let op = autotune_operation_set.fastest(fastest_index);
                            return op.execute();
                        }
                        TuneCacheResult::Miss => Action::Autotune,
                        #[cfg(autotune_persistent_cache)]
                        TuneCacheResult::Unchecked => Action::ValidateChecksum,
                        #[cfg(target_family = "wasm")]
                        TuneCacheResult::Deferred => Action::RegisterDeferred,
                    }
                }
                None => Action::PerformInit,
            },
            None => Action::PerformInit,
        };

        match action {
            Action::PerformInit => {
                let mut state = self.state.write();
                let map = state.get_or_insert_with(Default::default);

                if !map.contains_key(id) {
                    let name = self.name.replace("::", "-");
                    let tuner = Tuner::new(&name, &id.to_string());
                    map.insert(id.clone(), tuner);
                };
            }
            #[cfg(target_family = "wasm")]
            Action::RegisterDeferred => {
                let mut state = self.state.write();
                let map = state.get_or_insert_with(Default::default);

                if let Some(tuner) = map.get_mut(id) {
                    let key = autotune_operation_set.key();
                    let index = tuner.resolve(&key);
                    let op = autotune_operation_set.fastest(index);

                    return AutotuneOperation::execute(op);
                }
            }
            #[cfg(autotune_persistent_cache)]
            Action::ValidateChecksum => {
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
            Action::Autotune => {}
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

        let result = result.expect("Should have a result at this point");

        match result {
            Ok(response) => {
                let mut state = self.state.write();
                let map = state.get_or_insert_with(Default::default);

                if let Some(tuner) = map.get_mut(id) {
                    tuner.register_autotune(response)
                } else {
                    panic!("Should be init");
                }
            }
            Err(err) => match err {
                #[cfg(target_family = "wasm")]
                AutotuneError::Deferred(set) => {
                    let op = set.fastest(0);
                    AutotuneOperation::execute(op)
                }
                AutotuneError::Unknown(msg, _) => panic!("{msg}"),
            },
        }
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
