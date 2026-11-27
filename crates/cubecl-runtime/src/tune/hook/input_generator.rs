use crate::tune::TuneInputs;

/// Produces the benchmark inputs for a given key and reference inputs.
///
/// A tuner runs each candidate on the value `generate` returns, not on the real call
/// inputs directly, so callers can synthesize test inputs (for example, allocate fresh
/// output buffers) without mutating the real ones. The common case is just cloning the
/// reference inputs, which [`TunableSet::new_cloning_inputs`](crate::tune::TunableSet::new_cloning_inputs)
/// does.
///
/// There's no `Fn`-based blanket impl for arbitrary [`TuneInputs`] families because
/// `for<'a> Fn(&K, &I::At<'a>) -> I::At<'a>` runs into E0582 (`Fn`'s `Output` cannot
/// depend on the higher-ranked lifetime). For the owned-input case the HRTB collapses
/// and the `Fn` blanket below works; borrowed-input families implement this trait
/// directly.
///
/// # Guarantees
///
/// - Every implementation answers inputs posing the problem the reference inputs pose: the tune
///   measures the one and keeps its answer for the other.
/// - Inputs that share buffers with the reference ones, as a clone of device handles does, let a
///   candidate that writes them write the reference ones too: only a generator that allocates
///   them keeps the call's state out of the tune.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid input generator",
    label = "invalid input generator"
)]
pub trait InputGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// The inputs the candidates of `key` are measured on, made from the reference `inputs`.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> I::At<'a>;
}

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
