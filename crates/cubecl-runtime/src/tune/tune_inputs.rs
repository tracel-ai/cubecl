/// Describes the set of inputs a [`TunableSet`](super::TunableSet) accepts.
///
/// The associated type [`At`](Self::At) gives the concrete input type at a specific
/// lifetime. The indirection exists because [`TunableSet`](super::TunableSet) must be
/// `'static` (to live in [`LocalTuner`](super::LocalTuner)'s `Arc<dyn Any>` cache), but
/// some callers want to pass *borrowed* inputs. A `'static` marker type with a
/// lifetime-parameterized GAT threads the needle: the set is `'static`, but every
/// tunable function accepts `Self::At<'a>` for any `'a` via HRTB.
///
/// Most callers never implement this trait. The blanket impl below makes every plain
/// `'static + Clone + Send + Sync` type its own family with `At<'a> = Self`, so you can
/// just write `TunableSet<K, (A, B, C), Out>` for owned inputs.
///
/// Implementing it manually is only needed when the inputs genuinely borrow. Example:
/// burn's fusion autotune passes `TuneInput<'a, R, O>` (which wraps `&'a mut Context`)
/// by defining a `FusionTuneInputs<R, O>` marker with
/// `At<'a> = TuneInput<'a, R, O>`. Such a marker must not implement `Clone`, otherwise
/// it would overlap with the blanket impl below. A `PhantomData`-only struct without
/// `#[derive(Clone)]` satisfies that trivially.
pub trait TuneInputs: Send + Sync + 'static {
    /// The concrete input type at lifetime `'a`.
    type At<'a>: Clone + Send;
}

impl<T: Clone + Send + Sync + 'static> TuneInputs for T {
    type At<'a> = T;
}
