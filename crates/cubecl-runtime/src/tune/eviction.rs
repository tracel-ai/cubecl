use crate::tune::TuneInputs;
use alloc::sync::Arc;

/// Runs before every measured sample of a set's candidates, so a candidate whose operands fit
/// the device's last-level cache is timed reading memory the way a real call does.
///
/// A tuner samples one candidate several times over the same inputs, and a warm-up runs
/// first: from the second launch on, an operand small enough for the cache is served from it.
/// A memory-bound kernel then reads as several times faster than the bus, the round's
/// throughput bound is met by the first candidate tried, and the round ends there — with a
/// winner chosen on a read production never gets. The set says how to evict, since only it
/// knows what the device holds and what its operands weigh; the tuner says when.
///
/// What the eviction runs is the set's business — typically a launch that writes past the
/// cache's capacity through the inputs' client. It is issued outside the profiled region, so
/// it costs the round time and the sample nothing.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid eviction",
    label = "invalid eviction"
)]
pub trait Eviction<K, I: TuneInputs>: Send + Sync + 'static {
    /// Evict what the last measured launch left in cache, for a given key and reference inputs.
    fn evict<'a>(&self, key: &K, inputs: &I::At<'a>);
}

/// `Fn(&K, &A)` acts as an [`Eviction`] when `A` is an owned type. For multi-input kernels,
/// `A` is a tuple that the closure destructures internally.
impl<K, Func, A> Eviction<K, A> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A),
{
    #[inline]
    fn evict<'a>(&self, key: &K, inputs: &<A as TuneInputs>::At<'a>) {
        (self)(key, inputs)
    }
}

/// An [`Eviction`] bound to one key: what a benchmark loop calls before each sample, with the
/// inputs it is about to measure on.
pub type Evictor<I> = Arc<dyn for<'a> Fn(&<I as TuneInputs>::At<'a>) + Send + Sync>;
