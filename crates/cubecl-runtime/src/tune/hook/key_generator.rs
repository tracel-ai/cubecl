use crate::tune::TuneInputs;

/// Produces an autotune key from a borrowed view of the tuning inputs.
///
/// # Guarantees
///
/// - Every implementation answers from the inputs alone, and equal inputs give equal keys: a key
///   that changes for one problem re-tunes it, and one shared by two problems serves one's winner
///   to the other.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid key generator",
    label = "invalid key generator"
)]
pub trait KeyGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// The key of the problem `inputs` pose.
    fn generate(&self, inputs: &I::At<'_>) -> K;
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
