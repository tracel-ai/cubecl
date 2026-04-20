/// Describes the set of inputs a [`TunableSet`](super::TunableSet) accepts.
///
/// The associated type [`At`](Self::At) gives the concrete input type at a specific
/// lifetime. The indirection exists because [`TunableSet`](super::TunableSet) must be
/// `'static` (to live in [`LocalTuner`](super::LocalTuner)'s `Arc<dyn Any>` cache), but
/// some callers want to pass *borrowed* inputs. A `'static` marker type with a
/// lifetime-parameterized works here: the set is `'static`, but every
/// tunable function accepts `Self::At<'a>` for any `'a` via HRTB.
///
/// Implementing it manually is needed when the inputs genuinely borrowed. Example:
/// burn's fusion autotune passes `TuneInput<'a, R, O>` (which wraps `&'a mut Context`)
/// by defining a `FusionTuneInputs<R, O>` marker with
/// `At<'a> = TuneInput<'a, R, O>`. Such a marker must not implement `Clone`, otherwise
/// it would overlap with the blanket impl below.
pub trait TuneInputs: Send + Sync + 'static {
    /// The concrete input type at lifetime `'a`.
    type At<'a>: Clone + Send;
}

impl<T: Clone + Send + Sync + 'static> TuneInputs for T {
    type At<'a> = T;
}
