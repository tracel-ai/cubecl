use super::TuneInputs;

/// Produces an autotune key from a borrowed view of the tuning inputs.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid key generator",
    label = "invalid key generator"
)]
pub trait KeyGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// Generate a key from a set of inputs.
    fn generate<'a>(&self, inputs: &I::At<'a>) -> K;
}

/// Any `for<'a> Fn(&I::At<'a>) -> K` is a [`KeyGenerator`].
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
