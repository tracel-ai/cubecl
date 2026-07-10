use alloc::vec::Vec;

use crate::tune::TuneInputs;

/// Produces a set of [`AutotuneBound`]s for a given key and reference inputs.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid input generator",
    label = "invalid bounds generator"
)]
pub trait BoundsGenerator<K, I: TuneInputs, B: TimeBound>: Send + Sync + 'static {
    /// Generate a set of inputs for a given key and reference inputs.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> Vec<B>;
}

/// `Fn(&K, &A) -> A` acts as an [`InputGenerator`] when `A` is an owned type. For
/// multi-input kernels, `A` is a tuple that the closure destructures internally.
impl<K, Func, A, B: TimeBound> BoundsGenerator<K, A, B> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> Vec<B>,
{
    #[inline]
    fn generate<'a>(&self, key: &K, inputs: &<A as TuneInputs>::At<'a>) -> Vec<B> {
        (self)(key, inputs)
    }
}

/// A calculator that determines the time limit for autotune bounds.
pub trait TimeBound {
    /// Returns the time limit for autotune bounds.
    fn time_limit(&self) -> f64;
}

/// A bound for autotuning a throughput kernel, specifying the key, threshold, and number of operations.
pub struct AutotuneBound {
    /// The key for this bound, specifying the mode and data type of the throughput kernel.
    pub throughput: f64,
    /// The threshold for this bound, over which the kernel will be considered accurate.
    pub threshold: f32,
    /// The number of operations the kernel will run.
    pub ops_count: usize,
}

impl TimeBound for AutotuneBound {
    fn time_limit(&self) -> f64 {
        (self.ops_count as f64 / self.throughput) / self.threshold as f64
    }
}
