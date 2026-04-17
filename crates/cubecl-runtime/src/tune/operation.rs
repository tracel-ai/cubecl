use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;

use alloc::format;

use super::{
    AutotuneError, input_generator::InputGenerator, key_generator::KeyGenerator,
    tune_inputs::TuneInputs,
};
use super::{Tunable, TunePlan};

/// A type-erased delegate for a tunable function.
///
/// The lifetime `'inp` is the lifetime of the input data, the function must be defined such that
/// it can be called for any lifetime `inp` and produce a `Result<Out, AutotuneError>`.
type TuneDelegate<I, Out> =
    dyn for<'inp> Fn(<I as TuneInputs>::At<'inp>) -> Result<Out, AutotuneError> + Send + Sync;

/// A named, type-erased tunable function stored in a [`TunableSet`]. Constructed via
/// [`Tunable::new`](super::Tunable::new); callers don't name this type directly.
pub struct TuneFn<I: TuneInputs, Out> {
    pub(crate) name: Arc<str>,

    #[allow(clippy::type_complexity)]
    pub(crate) func: Arc<TuneDelegate<I, Out>>,
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
    /// Run the wrapped function on the given inputs.
    pub fn execute<'a>(&self, inputs: <I as TuneInputs>::At<'a>) -> Result<Out, AutotuneError> {
        (self.func)(inputs)
    }
}

/// A set of candidate tunable functions for autotune, sharing a key generator and an
/// input generator. See [`TuneInputs`] for the `F` parameter.
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

    /// Whether this set contains no tunables.
    pub fn is_empty(&self) -> bool {
        self.tunables.is_empty()
    }

    /// Create a tunable set from a key generator and an input generator.
    pub fn new(key_gen: impl KeyGenerator<K, F>, input_gen: impl InputGenerator<K, F>) -> Self {
        Self {
            tunables: Default::default(),
            input_gen: Arc::new(input_gen),
            key_gen: Arc::new(key_gen),
        }
    }

    /// Shorthand for [`new`](Self::new) with a [`CloneInputGenerator`]: benchmarks run
    /// on clones of the real call inputs.
    pub fn new_cloning_inputs(key_gen: impl KeyGenerator<K, F>) -> Self {
        Self::new(key_gen, super::CloneInputGenerator)
    }

    /// Register a tunable with this tunable set.
    pub fn with(mut self, tunable: Tunable<K, F, Output>) -> Self {
        self.tunables.push(tunable);
        self
    }

    /// All candidate operations in this set, in registration order.
    pub fn autotunables(&self) -> impl Iterator<Item = &TuneFn<F, Output>> {
        self.tunables.iter().map(|tunable| &tunable.function)
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
    pub fn compute_checksum(&self) -> String {
        let mut checksum = String::new();
        for tune in &self.tunables {
            checksum += &tune.function.name;
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
