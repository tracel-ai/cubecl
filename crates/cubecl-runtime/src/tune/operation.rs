use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use core::hash::Hash;
use core::{
    fmt::{Debug, Display},
    marker::PhantomData,
};
use variadics_please::all_tuples;

use super::{
    input_generator::{InputGenerator, IntoInputGenerator},
    key_generator::{IntoKeyGenerator, KeyGenerator},
    AutotuneError,
};

/// Default checksum for an operation set
#[cfg(autotune_persistent_cache)]
pub fn compute_checksum<In: Clone + Send + 'static, Out: Send + 'static>(
    autotunables: &[Arc<dyn Tunable<Inputs = In, Output = Out>>],
) -> String {
    let mut checksum = String::new();
    autotunables.iter().for_each(|op| {
        checksum += op.name();
    });
    format!("{:x}", md5::compute(checksum))
}

/// Groups operations of the same type for autotune
pub struct TunableSet<K: AutotuneKey, Inputs: Send + 'static, Output: Send + 'static> {
    tunables: Vec<Arc<dyn Tunable<Inputs = Inputs, Output = Output>>>,
    key_gen: Box<dyn KeyGenerator<K, Inputs>>,
    input_gen: Box<dyn InputGenerator<K, Inputs>>,
    #[allow(clippy::type_complexity)]
    checksum_override: Option<Box<dyn Fn(&Self) -> String + Send + Sync>>,
}

impl<K: AutotuneKey, Inputs: Clone + Send + 'static, Output: Send + 'static>
    TunableSet<K, Inputs, Output>
{
    /// Create a tunable set from a key generator and an input generator
    pub fn new<KMarker, IMarker>(
        key_gen: impl IntoKeyGenerator<K, Inputs, KMarker>,
        input_gen: impl IntoInputGenerator<K, Inputs, IMarker>,
    ) -> Self {
        Self {
            tunables: Default::default(),
            input_gen: Box::new(input_gen.into_input_gen()),
            key_gen: Box::new(key_gen.into_key_gen()),
            checksum_override: None,
        }
    }

    /// Register a tunable with this tunable set
    pub fn with_tunable<Marker>(
        mut self,
        tunable: impl IntoTunable<Inputs, Output, Marker>,
    ) -> Self {
        self.tunables.push(Arc::new(tunable.into_tunable()));
        self
    }

    /// Override the checksum algorithm
    pub fn with_custom_checksum(
        mut self,
        checksum: impl Fn(&Self) -> String + Send + Sync + 'static,
    ) -> Self {
        self.checksum_override = Some(Box::new(checksum));
        self
    }

    /// All candidate operations for autotuning this operation type
    /// Operations can run on toy tensors of relevant size
    pub fn autotunables(&self) -> Vec<Arc<dyn Tunable<Inputs = Inputs, Output = Output>>> {
        self.tunables.clone()
    }

    /// Returns the operation for the given index, matching the order
    /// returned by autotunables. Operation obtained here runs on original tensors
    /// Nb: The 0 index is used a "good default".
    pub fn fastest(
        &self,
        fastest_index: usize,
    ) -> Arc<dyn Tunable<Inputs = Inputs, Output = Output>> {
        self.tunables[fastest_index].clone()
    }

    /// Compute a checksum that can invalidate outdated cached auto-tune results.
    #[cfg(autotune_persistent_cache)]
    pub fn compute_checksum(&self) -> String {
        if let Some(checksum_override) = &self.checksum_override {
            checksum_override(self)
        } else {
            compute_checksum(&self.tunables)
        }
    }

    /// Generate a key from a set of inputs
    pub fn generate_key(&self, inputs: &Inputs) -> K {
        self.key_gen.generate(inputs)
    }

    /// Generate a set of test inputs from a key and reference inputs
    pub fn generate_inputs(&self, key: &K, inputs: &Inputs) -> Inputs {
        self.input_gen.generate(key, inputs)
    }
}

/// A tunable entry in a tunable set
pub trait Tunable: Send + Sync + 'static {
    /// Inputs to the tunable function
    type Inputs: Clone;
    /// Output from the tunable function
    type Output;

    /// Run a tuneable function
    fn execute(&self, inputs: Self::Inputs) -> Result<Self::Output, AutotuneError>;

    /// The name of the tuneable function
    fn name(&self) -> &str {
        core::any::type_name::<Self>()
    }
}

/// Something that can be turned into a [Tunable]
///
/// # Marker
/// The marker generic is used to work around limitations in the trait resolver that causes
/// conflicting implementation errors.
pub trait IntoTunable<In, Out, Marker> {
    /// The output tunable type
    type Tunable: Tunable<Inputs = In, Output = Out>;

    /// Convert to a tunable
    fn into_tunable(self) -> Self::Tunable;
}

/// Dummy marker for [`IntoTunable`] on [`Tunable`]s
pub struct IsIdentity;

impl<T: Tunable> IntoTunable<T::Inputs, T::Output, IsIdentity> for T {
    type Tunable = T;

    fn into_tunable(self) -> Self::Tunable {
        self
    }
}

/// Tunable implemented as a function or closure
///
/// # Marker
/// The marker generic is used to work around limitations in the trait resolver that causes
/// conflicting implementation errors.
pub struct FunctionTunable<F: AsFunctionTunable<Marker>, Marker: Send + Sync + 'static> {
    func: F,
    _marker: PhantomData<Marker>,
}

impl<F: AsFunctionTunable<Marker>, Marker: Send + Sync + 'static> Tunable
    for FunctionTunable<F, Marker>
{
    type Inputs = F::Inputs;
    type Output = F::Output;

    fn execute(&self, inputs: Self::Inputs) -> Result<Self::Output, AutotuneError> {
        self.func.execute(inputs)
    }
}

struct IsFunction;

impl<F: AsFunctionTunable<Marker>, Marker: Send + Sync + 'static>
    IntoTunable<F::Inputs, F::Output, (Marker, IsFunction)> for F
{
    type Tunable = FunctionTunable<F, Marker>;

    fn into_tunable(self) -> Self::Tunable {
        FunctionTunable {
            func: self,
            _marker: PhantomData,
        }
    }
}

/// A function that can be turned into a tunable.
///
/// # Marker
/// The marker generic is used to work around limitations in the trait resolver that causes
/// conflicting implementation errors.
pub trait AsFunctionTunable<Marker>: Send + Sync + 'static {
    /// Function inputs
    type Inputs: Clone;
    /// Function output
    type Output;

    /// Run a tuneable function
    fn execute(&self, inputs: Self::Inputs) -> Result<Self::Output, AutotuneError>;

    /// The name of the tuneable function
    fn name(&self) -> &str {
        core::any::type_name::<Self>()
    }
}

macro_rules! impl_tunable {
    ($($params:ident),*) => {
        #[allow(unused_parens)]
        impl<Out: 'static, Func, $($params: Clone + Send + 'static,)*> AsFunctionTunable<fn($($params),*) -> Out> for Func where Func: Send + Sync + 'static, for<'a> &'a Func: Fn($($params),*) -> Out {
            type Inputs = ($($params),*);
            type Output = Out;

            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[inline]
            fn execute(&self, ($($params),*): ($($params),*)) -> Result<Out, AutotuneError> {
                fn call_inner<Out, $($params,)*>(
                    f: impl Fn($($params,)*) -> Out,
                    $($params: $params,)*
                ) -> Result<Out, AutotuneError> {
                    Ok(f($($params,)*))
                }
                call_inner(self, $($params),*)
            }
        }
    };
}

struct IsResult;

macro_rules! impl_tunable_result {
    ($($params:ident),*) => {
        #[allow(unused_parens)]
        impl<Out: 'static, Func, $($params: Clone + Send + 'static,)*> AsFunctionTunable<(IsResult, fn($($params),*) -> Result<Out, AutotuneError>)> for Func
            where Func: Send + Sync + 'static,
            for<'a> &'a Func: Fn($($params),*) -> Result<Out, AutotuneError>
            {
            type Inputs = ($($params),*);
            type Output = Out;

            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[inline]
            fn execute(&self, ($($params),*): ($($params),*)) -> Result<Out, AutotuneError> {
                fn call_inner<Out, $($params,)*>(
                    f: impl Fn($($params,)*) -> Result<Out, AutotuneError>,
                    $($params: $params,)*
                ) -> Result<Out, AutotuneError> {
                    f($($params,)*)
                }
                call_inner(self, $($params),*)
            }
        }
    };
}

all_tuples!(impl_tunable, 0, 16, I);
all_tuples!(impl_tunable_result, 0, 16, I);

#[cfg(autotune_persistent_cache)]
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
#[cfg(not(autotune_persistent_cache))]
/// Trait alias
pub trait AutotuneKey:
    Clone + Debug + PartialEq + Eq + Hash + Display + Send + Sync + 'static
{
}

impl AutotuneKey for String {}
