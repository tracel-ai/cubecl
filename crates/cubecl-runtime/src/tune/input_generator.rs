use super::TuneInputs;

/// Produces the benchmark inputs for a given key and reference inputs.
///
/// A tuner runs each candidate on the value `generate` returns, not on the real call
/// inputs directly, so callers can synthesize test inputs (for example, allocate fresh
/// output buffers) without mutating the real ones. The common case is just cloning the
/// reference inputs; use [`CloneInputGenerator`] for that.
///
/// There's no `Fn`-based blanket impl for arbitrary [`TuneInputs`] families because
/// `for<'a> Fn(&K, &I::At<'a>) -> I::At<'a>` runs into E0582 (`Fn`'s `Output` cannot
/// depend on the higher-ranked lifetime). For the owned-input case the HRTB collapses
/// and the `Fn` blanket below works; borrowed-input families implement this trait
/// directly.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid input generator",
    label = "invalid input generator"
)]
pub trait InputGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// Generate a set of inputs for a given key and reference inputs.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> I::At<'a>;
}

/// [`InputGenerator`] that clones the reference inputs verbatim.
#[derive(Copy, Clone, Debug, Default)]
pub struct CloneInputGenerator;

impl<K, I: TuneInputs> InputGenerator<K, I> for CloneInputGenerator {
    fn generate<'a>(&self, _key: &K, inputs: &I::At<'a>) -> I::At<'a> {
        inputs.clone()
    }
}

/// `Fn(&K, &A) -> A` acts as an [`InputGenerator`] when `A` is an owned type. For
/// multi-input kernels, `A` is a tuple that the closure destructures internally.
impl<K, Func, A> InputGenerator<K, A> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> A,
{
    #[inline]
    fn generate<'a>(
        &self,
        key: &K,
        inputs: &<A as TuneInputs>::At<'a>,
    ) -> <A as TuneInputs>::At<'a> {
        (self)(key, inputs)
    }
}
