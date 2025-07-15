use core::marker::PhantomData;

use variadics_please::all_tuples;

use super::{AutotuneError, IntoTuneFn, TuneFn};

/// Tunable implemented as a function or closure
///
/// # Marker
/// The marker generic is used to work around limitations in the trait resolver that causes
/// conflicting implementation errors.
pub struct FunctionTunable<F: AsFunctionTunableResult<Marker>, Marker> {
    func: F,
    _marker: PhantomData<Marker>,
}

unsafe impl<F: AsFunctionTunableResult<Marker> + Send, Marker> Send for FunctionTunable<F, Marker> {}
unsafe impl<F: AsFunctionTunableResult<Marker> + Sync, Marker> Sync for FunctionTunable<F, Marker> {}

impl<F: AsFunctionTunableResult<Marker>, Marker: 'static> TuneFn for FunctionTunable<F, Marker> {
    type Inputs = F::Inputs;
    type Output = F::Output;

    fn execute(&self, inputs: Self::Inputs) -> Result<Self::Output, AutotuneError> {
        self.func.execute(inputs)
    }
}

/// Dummy marker for function tunables
#[doc(hidden)]
pub struct IsFunction;

impl<F: AsFunctionTunableResult<Marker>, Marker: 'static>
    IntoTuneFn<F::Inputs, F::Output, (Marker, IsFunction)> for F
{
    type Tunable = FunctionTunable<F, Marker>;

    fn into_tunable(self) -> Self::Tunable {
        FunctionTunable {
            func: self,
            _marker: PhantomData,
        }
    }
}

/// Tunable implemented as a function or closure that returns a plain value, wrapped in `Ok`.
///
/// # Marker
/// The marker generic is used to work around limitations in the trait resolver that causes
/// conflicting implementation errors.
pub struct FunctionTunableResultMap<F: AsFunctionTunable<Marker>, Marker> {
    func: F,
    _marker: PhantomData<Marker>,
}

unsafe impl<F: AsFunctionTunable<Marker> + Send, Marker> Send
    for FunctionTunableResultMap<F, Marker>
{
}
unsafe impl<F: AsFunctionTunable<Marker> + Sync, Marker> Sync
    for FunctionTunableResultMap<F, Marker>
{
}

impl<F: AsFunctionTunable<Marker>, Marker: 'static> TuneFn for FunctionTunableResultMap<F, Marker> {
    type Inputs = F::Inputs;
    type Output = F::Output;

    fn execute(&self, inputs: Self::Inputs) -> Result<Self::Output, AutotuneError> {
        Ok(self.func.execute(inputs))
    }
}

/// A function that can be turned into a tunable.
///
/// # Marker
/// The marker generic is used to work around limitations in the trait resolver that causes
/// conflicting implementation errors.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid tunable.",
    label = "invalid tunable"
)]
pub trait AsFunctionTunable<Marker>: Sized + Send + Sync + 'static {
    /// Function inputs
    type Inputs: Clone;
    /// Function output
    type Output;

    /// Run a tuneable function
    fn execute(&self, inputs: Self::Inputs) -> Self::Output;

    /// Wrap infallible tunable with `Ok`
    fn ok(self) -> FunctionTunableResultMap<Self, Marker> {
        FunctionTunableResultMap {
            func: self,
            _marker: PhantomData,
        }
    }

    /// The name of the tuneable function
    fn name(&self) -> &str {
        core::any::type_name::<Self>()
    }
}

/// An infallible function that can be turned into a tunable.
///
/// # Marker
/// The marker generic is used to work around limitations in the trait resolver that causes
/// conflicting implementation errors.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid tunable. For infallible kernels, use `AsFunctionTunable::ok`",
    label = "invalid tunable"
)]
pub trait AsFunctionTunableResult<Marker>: Send + Sync + 'static {
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
    ($(#[$meta:meta])* $($params:ident),*) => {
        #[allow(unused_parens)]
        $(#[$meta])*
        impl<Out: 'static, Func, $($params: Clone + Send + 'static,)*> AsFunctionTunable<fn($($params),*) -> Out> for Func
            where Func: Send + Sync + 'static,
            for<'a> &'a Func: Fn($($params),*) -> Out
        {
            type Inputs = ($($params),*);
            type Output = Out;

            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[inline]
            fn execute(&self, ($($params),*): ($($params),*)) -> Out {
                fn call_inner<Out, $($params,)*>(
                    f: impl Fn($($params,)*) -> Out,
                    $($params: $params,)*
                ) -> Out {
                    f($($params,)*)
                }
                call_inner(self, $($params),*)
            }
        }
    };
}

macro_rules! impl_tunable_result {
    ($(#[$meta:meta])* $($params:ident),*) => {
        #[allow(unused_parens)]
        $(#[$meta])*
        impl<Out: 'static, Err, Func, $($params: Clone + Send + 'static,)*> AsFunctionTunableResult<fn($($params),*) -> Result<Out, Err>> for Func
            where Func: Send + Sync + 'static,
            for<'a> &'a Func: Fn($($params),*) -> Result<Out, Err>,
            Err: Into<AutotuneError>
        {
            type Inputs = ($($params),*);
            type Output = Out;

            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[inline]
            fn execute(&self, ($($params),*): ($($params),*)) -> Result<Out, AutotuneError> {
                fn call_inner<Out, Err, $($params,)*>(
                    f: impl Fn($($params,)*) -> Result<Out, Err>,
                    $($params: $params,)*
                ) -> Result<Out, Err> {
                    f($($params,)*)
                }
                call_inner(self, $($params),*).map_err(Into::into)
            }
        }
    };
}

all_tuples!(impl_tunable, 0, 12, I);
all_tuples!(impl_tunable_result, 0, 12, I);
