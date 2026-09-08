use crate::tune::{AutotuneError, TuneInputs};
use alloc::string::{String, ToString};

/// Runs before every measured sample of a set's candidates, so a candidate whose operands fit
/// the device's last-level cache is timed reading memory the way a real call does.
///
/// A tuner samples one candidate several times over the same inputs, and a warm-up runs
/// first: from the second launch on, an operand small enough for the cache is served from it.
/// A memory-bound kernel then reads as several times faster than the bus, the round's
/// throughput bound is met by the first candidate tried, and the round ends there — with a
/// winner chosen on a read production never gets. The set says how to evict, since it knows
/// what its operands weigh; the tuner says when.
///
/// What the eviction runs is the set's business — typically a launch that writes past the
/// cache's capacity through the inputs' client. It receives the reference inputs the tune was
/// called with rather than the generated ones the candidates are measured on, so it can scrub
/// through buffers the caller already holds instead of allocating its own: an output the
/// winner overwrites afterwards is scratch until then. Whether that reaches past the cache is
/// the set's call — an operand-sized write evicts an operand-sized cache, no more.
///
/// The eviction is issued outside the profiled region, so it costs the round time rather than
/// the sample, as long as it goes through the stream the sample is profiled on: the tuner
/// orders nothing between the two, and a launch on another stream can overlap the sample or
/// land after it.
///
/// A failed eviction is logged and the sample is measured anyway, warm: an eviction is a
/// measurement aid, and losing one is not worth failing the tune for.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid eviction",
    label = "invalid eviction"
)]
pub trait Eviction<K, I: TuneInputs>: Send + Sync + 'static {
    /// Evict what the last measured launch left in cache, for a given key and the reference
    /// inputs the tune was called with.
    fn evict<'a>(&self, key: &K, inputs: &I::At<'a>) -> Result<(), AutotuneError>;
}

/// `Fn(&K, &A) -> Result<(), E>` acts as an [`Eviction`] when `A` is an owned type. For
/// multi-input kernels, `A` is a tuple that the closure destructures internally.
impl<K, Func, A, Err> Eviction<K, A> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Err: Into<String> + 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> Result<(), Err>,
{
    #[inline]
    fn evict<'a>(&self, key: &K, inputs: &<A as TuneInputs>::At<'a>) -> Result<(), AutotuneError> {
        (self)(key, inputs).map_err(|err| AutotuneError::Unknown {
            name: "eviction".to_string(),
            err: err.into(),
        })
    }
}

/// An [`Eviction`] bound to one key and the reference inputs of one tune: what a benchmark
/// loop calls before each sample.
///
/// `FnMut` rather than `Fn` because the loop lends it out exclusively: the closure holds the
/// inputs, which are only `Send`, and a shared borrow would need them `Sync` to reach the
/// device thread the samples are launched from.
pub type Evictor<'i> = dyn FnMut() -> Result<(), AutotuneError> + Send + 'i;
