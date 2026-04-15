use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;

#[cfg(std_io)]
use alloc::format;

use super::{
    AutotuneError, input_generator::InputGenerator, key_generator::KeyGenerator,
    tune_inputs::TuneInputs,
};
use super::{Tunable, TunePlan};

/// A type-erased tunable function: a name plus a `for<'a> Fn` closure that accepts the
/// input type at any lifetime.
///
/// This is what [`Tunable`] wraps and what [`TunableSet`] stores. Callers don't construct
/// this directly — use [`Tunable::new`](super::Tunable::new).
pub struct TuneFn<I: TuneInputs, Out> {
    pub(crate) name: String,
    #[allow(clippy::type_complexity)]
    pub(crate) func: Arc<
        dyn for<'a> Fn(<I as TuneInputs>::At<'a>) -> Result<Out, AutotuneError> + Send + Sync,
    >,
}

impl<I: TuneInputs, Out> Clone for TuneFn<I, Out> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            func: self.func.clone(),
        }
    }
}

impl<I: TuneInputs, Out: 'static> TuneFn<I, Out> {
    /// The configured name of the tunable function.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the wrapped function on the given inputs.
    pub fn execute<'a>(&self, inputs: <I as TuneInputs>::At<'a>) -> Result<Out, AutotuneError> {
        (self.func)(inputs)
    }
}

/// Groups operations of the same type for autotune.
///
/// `I: TuneInputs` describes the tuning inputs — a `'static` marker type whose GAT `I::At<'a>`
/// gives the concrete input type at lifetime `'a`. This indirection lets `TunableSet` be
/// `'static` (and hence cacheable in [`LocalTuner::init`]) while still accepting borrowed
/// inputs at `execute` time via HRTB over `'a`.
pub struct TunableSet<K: AutotuneKey, F: TuneInputs, Output: 'static> {
    tunables: Vec<Tunable<K, F, Output>>,
    key_gen: Arc<dyn KeyGenerator<K, F> + Send + Sync>,
    input_gen: Arc<dyn InputGenerator<K, F> + Send + Sync>,
}

impl<K: AutotuneKey, F: TuneInputs, Output: 'static> TunableSet<K, F, Output> {
    /// The number of tunables in the set.
    pub fn len(&self) -> usize {
        self.tunables.len()
    }

    /// Create a tunable set from a key generator and an input generator.
    pub fn new(key_gen: impl KeyGenerator<K, F>, input_gen: impl InputGenerator<K, F>) -> Self {
        Self {
            tunables: Default::default(),
            input_gen: Arc::new(input_gen),
            key_gen: Arc::new(key_gen),
        }
    }

    /// Shorthand for [`new`](Self::new) with a [`CloneInputGenerator`] — the tuning
    /// inputs are just clones of the reference inputs. This is the common case for
    /// autotune setups that rerun the same fused op on the real inputs
    /// (burn-cubecl-fusion, and anywhere the tuner doesn't need to synthesize fresh test
    /// data).
    pub fn new_cloning_inputs(key_gen: impl KeyGenerator<K, F>) -> Self {
        Self::new(key_gen, super::CloneInputGenerator)
    }

    /// Register a tunable with this tunable set.
    pub fn with(mut self, tunable: Tunable<K, F, Output>) -> Self {
        self.tunables.push(tunable);
        self
    }

    /// All candidate operations in this set, in registration order.
    pub fn autotunables(&self) -> Vec<TuneFn<F, Output>> {
        self.tunables
            .iter()
            .map(|tunable| tunable.function.clone())
            .collect()
    }

    /// Returns the [autotune plan](TunePlan) for the given set.
    pub(crate) fn plan(&self, key: &K) -> TunePlan {
        TunePlan::new(key, &self.tunables)
    }

    /// Returns the operation for the given index, matching the order returned by
    /// `autotunables`. Tunables are tried in order, so index 0 should be a good default.
    pub fn fastest(&self, fastest_index: usize) -> TuneFn<F, Output> {
        self.tunables[fastest_index].function.clone()
    }

    /// Compute a checksum that invalidates outdated cached auto-tune results when the
    /// set of tunable names changes.
    #[cfg(std_io)]
    pub fn compute_checksum(&self) -> String {
        let mut checksum = String::new();
        for tune in &self.tunables {
            checksum += tune.function.name();
        }
        format!("{:x}", md5::compute(checksum))
    }

    /// Generate a key from a set of inputs
    pub fn generate_key<'a>(&self, inputs: &F::At<'a>) -> K {
        self.key_gen.generate(inputs)
    }

    /// Generate a set of test inputs from a key and reference inputs.
    pub fn generate_inputs<'a>(&self, key: &K, inputs: &F::At<'a>) -> F::At<'a> {
        self.input_gen.generate(key, inputs)
    }
}

#[cfg(std_io)]
/// Trait alias with support for persistent caching
pub trait AutotuneKey:
    Clone
    + Debug
    + PartialEq
    + Eq
    + Hash
    + Display
    + serde::Serialize
    + serde::de::DeserializeOwned
    + Send
    + Sync
    + 'static
{
}
#[cfg(not(std_io))]
/// Trait alias
pub trait AutotuneKey:
    Clone + Debug + PartialEq + Eq + Hash + Display + Send + Sync + 'static
{
}

impl AutotuneKey for String {}
