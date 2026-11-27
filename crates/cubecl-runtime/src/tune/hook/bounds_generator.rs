use crate::tune::{Bounds, TuneInputs};

/// Produces the [`Bounds`] a tune of a key is measured against, from the key and the reference
/// inputs.
///
/// # Guarantees
///
/// - Every implementation answers from the problem the key and the inputs pose, so every tune of
///   one key on one device is held to the same time limit.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid bounds generator",
    label = "invalid bounds generator"
)]
pub trait BoundsGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// The bounds of the problem `key` and `inputs` pose.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> Bounds;
}

/// `Fn(&K, &A) -> Bounds` acts as a [`BoundsGenerator`] when `A` is an owned type. For
/// multi-input kernels, `A` is a tuple that the closure destructures internally.
impl<K, Func, A> BoundsGenerator<K, A> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> Bounds,
{
    #[inline]
    fn generate<'a>(&self, key: &K, inputs: &<A as TuneInputs>::At<'a>) -> Bounds {
        (self)(key, inputs)
    }
}
