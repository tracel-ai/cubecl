use super::TuneInputs;

/// A generator that produces an autotune key from a borrowed view of the tuning inputs.
///
/// The `generate` method is HRTB over `'a` so a `dyn KeyGenerator<K, I>` can be stored
/// `'static` inside a cached [`TunableSet`](super::TunableSet) while still accepting
/// `I::At<'a>` at call time for any `'a`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid key generator",
    label = "invalid key generator"
)]
pub trait KeyGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// Generate a key from a set of inputs.
    fn generate<'a>(&self, inputs: &I::At<'a>) -> K;
}

/// Blanket impl for any `for<'a> Fn(&I::At<'a>) -> K`. HRTB over `'a` means one concrete
/// function satisfies the bound for every lifetime at once, which is what lets the outer
/// [`KeyGenerator`] trait object be `'static`.
impl<K, I, Func> KeyGenerator<K, I> for Func
where
    I: TuneInputs,
    Func: Send + Sync + 'static,
    for<'a> Func: Fn(&I::At<'a>) -> K,
{
    #[inline]
    fn generate<'a>(&self, inputs: &I::At<'a>) -> K {
        (self)(inputs)
    }
}
