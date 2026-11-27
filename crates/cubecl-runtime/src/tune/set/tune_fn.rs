use alloc::boxed::Box;
use alloc::string::String;

use crate::tune::{AutotuneError, TuneInputs};

/// A type-erased delegate for a tunable function, callable on inputs of any lifetime.
type TuneDelegate<I, Out> =
    dyn for<'inp> Fn(<I as TuneInputs>::At<'inp>) -> Result<Out, AutotuneError> + Send + Sync;

/// A named, type-erased tunable function, as a [`Tunable`](crate::tune::Tunable) holds it.
#[derive(new)]
pub struct TuneFn<I: TuneInputs, Out> {
    name: String,
    func: Box<TuneDelegate<I, Out>>,
}

impl<I: TuneInputs, Out: 'static> TuneFn<I, Out> {
    /// The name the tunable was declared with.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the function as the call's work.
    pub fn execute<'a>(&self, inputs: <I as TuneInputs>::At<'a>) -> Result<Out, AutotuneError> {
        (self.func)(inputs)
    }

    /// Run the function as an autotune race runs a candidate, to measure it or gather its
    /// kernels: what it launches is [issued for the race](crate::execution::LaunchPurpose::Race),
    /// never the work's.
    pub fn race<'a>(&self, inputs: <I as TuneInputs>::At<'a>) -> Result<Out, AutotuneError> {
        crate::execution::race(|| (self.func)(inputs))
    }
}
