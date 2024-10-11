use core::{fmt::Display, hash::Hash};
use hashbrown::HashMap;
use std::sync::Arc;

use crate::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

use super::{AutotuneKey, AutotuneOperationSet, Tuner};

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
        autotune_operation_set: Arc<dyn AutotuneOperationSet<AK, Out>>,
    ) -> Out
    where
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    {
        // We avoid locking in write mode when possible.
        // (this makes us potentially check the cache twice, but allows to avoid
        // locking the state if the cache is hit)
        if let Some(state) = self.state.read().as_ref() {
            if let Some(tuner) = state.get(id) {
                let key = autotune_operation_set.key();
                if let Some(index) = tuner.autotune_fastest(&key) {
                    let op = autotune_operation_set.fastest(index);
                    return op.execute();
                }
            }
        }

        // We have to run the autotune.
        let mut state = self.state.write();
        let map = state.get_or_insert_with(Default::default);

        let tuner = if let Some(tuner) = map.get_mut(id) {
            tuner
        } else {
            let name = self.name.replace("::", "-");
            let tuner = Tuner::new(&name, &id.to_string());
            map.insert(id.clone(), tuner);
            map.get_mut(id).unwrap()
        };

        tuner.execute_autotune(autotune_operation_set, client)
    }

    /// Return the autotune result given a key.
    pub fn autotune_result(&self, id: &ID, key: &AK) -> Option<usize> {
        if let Some(state) = self.state.read().as_ref() {
            if let Some(tuner) = state.get(id) {
                return tuner.autotune_fastest(key);
            }
        }

        None
    }
}
