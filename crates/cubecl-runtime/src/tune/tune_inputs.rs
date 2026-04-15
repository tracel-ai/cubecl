/// Describes the set of inputs a [`TunableSet`](super::TunableSet) accepts, in a way that
/// lets the set store borrowed data.
///
/// Rust doesn't have higher-kinded types, so we can't directly express "a `TunableSet`
/// that's generic over any lifetime of its inputs". Instead, `TunableSet` is parameterized
/// by a `'static` marker type implementing this trait, and the trait carries a generic
/// associated type [`At`](Self::At) giving the actual input type at a specific lifetime.
///
/// The motivating use case is burn's fusion `TuneInput<'a, R, O>`: it holds a real `&'a mut`
/// borrow of the fused-op context and cannot be `'static`, but the `TunableSet` that stores
/// the tunable functions *must* be `'static` so it can live in `LocalTuner::sets`'s
/// `Arc<dyn Any + Send + Sync>` cache. Making `TunableSet` generic over
/// `I: TuneInputs` with `I::At<'a> = TuneInput<'a, R, O>` threads that needle: the set
/// itself is `'static`, but every stored function's signature — via HRTB + GAT — accepts
/// `I::At<'a>` for any `'a`. Cache-hit paths pay exactly one `Arc<TunableSet>` clone per
/// call; no per-call rebuild of the tunable set.
///
/// The trivial impl [`OwnedInputs`] is provided for any `'static` type: `At<'a> = T`
/// ignores the lifetime. This is what callers whose inputs don't contain borrows use, and
/// it's the equivalent of the pre-refactor cached path.
pub trait TuneInputs: Send + Sync + 'static {
    /// The concrete input type at lifetime `'a`.
    type At<'a>: Clone + Send;
}

/// Trivial [`TuneInputs`] impl for any `'static` input type: `At<'a> = T`.
///
/// Use this when your tunable inputs don't need to carry a lifetime (i.e. they're plain
/// owned/`'static` types). The resulting `TunableSet<K, OwnedInputs<T>, Out>` is equivalent
/// to the pre-refactor `TunableSet<K, T, Out>` — cached `'static`, zero allocation per
/// call on cache hits.
pub struct OwnedInputs<T>(core::marker::PhantomData<fn() -> T>);

impl<T: Clone + Send + 'static> TuneInputs for OwnedInputs<T> {
    type At<'a> = T;
}
