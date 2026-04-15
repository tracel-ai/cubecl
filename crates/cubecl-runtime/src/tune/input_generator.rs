use super::{OwnedInputs, TuneInputs};

/// Produces the inputs an autotune pass benchmarks against, given the real call inputs and
/// the computed key.
///
/// Like [`KeyGenerator`](super::KeyGenerator), `generate` is HRTB over `'a` so a
/// `dyn InputGenerator<K, I>` can live `'static` inside a cached
/// [`TunableSet`](super::TunableSet) while still accepting `I::At<'a>` at call time.
///
/// ## Why there's no HRTB-generic `Fn` blanket impl
///
/// Rust's HRTB machinery can't express `for<'a> Fn(&K, &I::At<'a>) -> I::At<'a>` when the
/// return type references the higher-ranked lifetime — `Fn`'s `Output` associated type
/// can't be bound to a type that depends on `'a` (E0582). Two ways around it are
/// provided: [`CloneInputGenerator`] for the "just clone the reference inputs" case, and
/// a blanket impl for `Fn(&K, &A) -> A` specialized to [`OwnedInputs<A>`] where
/// `OwnedInputs::At<'a> = A` is lifetime-independent so the HRTB collapses. Multi-input
/// kernels use a tuple `A = (X, Y, Z)` and destructure inside.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid input generator",
    label = "invalid input generator"
)]
pub trait InputGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// Generate a set of inputs for a given key and reference inputs.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> I::At<'a>;
}

/// The trivial [`InputGenerator`] — clones the reference inputs verbatim. Used by burn
/// and the dummy tests: autotune just re-runs the op on the same inputs, so the "test
/// inputs" are literally a clone of the real inputs.
#[derive(Copy, Clone, Debug, Default)]
pub struct CloneInputGenerator;

impl<K, I: TuneInputs> InputGenerator<K, I> for CloneInputGenerator {
    fn generate<'a>(&self, _key: &K, inputs: &I::At<'a>) -> I::At<'a> {
        inputs.clone()
    }
}

/// Blanket impl for any `Fn(&K, &A) -> A` — specialized to [`OwnedInputs<A>`] so the
/// HRTB collapses (see trait docs). `A` can be a tuple like `(X, Y, Z)` to handle
/// multi-input kernels; destructure inside the function body.
impl<K, Func, A> InputGenerator<K, OwnedInputs<A>> for Func
where
    A: Clone + Send + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> A,
{
    #[inline]
    fn generate<'a>(
        &self,
        key: &K,
        inputs: &<OwnedInputs<A> as TuneInputs>::At<'a>,
    ) -> <OwnedInputs<A> as TuneInputs>::At<'a> {
        (self)(key, inputs)
    }
}
