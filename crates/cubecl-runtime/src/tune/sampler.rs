use alloc::vec::Vec;
use core::time::Duration;
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};
use cubecl_common::profile::TimingMethod;

/// Total samples required before the warmup-biased first sample is discarded.
const DISCARD_THRESHOLD: usize = 3;
/// Relative improvement below which a new sample counts as no progress.
const CONVERGENCE_EPSILON: f64 = 0.02;
/// Consecutive non-improving samples before a candidate is considered converged.
const CONVERGENCE_ROUNDS: u8 = 2;

/// The timings collected for one candidate, plus the small amount of state needed to decide
/// whether it is still worth sampling.
///
/// `BenchmarkComputations::score` is not usable while tuning is in flight: at a single sample the
/// variance is zero and reads as perfect stability. This tracks the sample count explicitly so
/// elimination can require evidence instead of inferring it from a degenerate variance.
#[derive(Debug, Default)]
pub(crate) struct SampleSet {
    durations: Vec<Duration>,
    stalled: u8,
}

impl SampleSet {
    pub(crate) fn push(&mut self, duration: Duration) {
        let before = self.best();
        self.durations.push(duration);

        // This push ages the biased first sample out, so the two bests are taken over different
        // samples and their difference says nothing about progress.
        if self.durations.len() == DISCARD_THRESHOLD {
            self.stalled = 0;
            return;
        }

        let improved = match (before, self.best()) {
            (Some(before), Some(after)) => after < before.mul_f64(1.0 - CONVERGENCE_EPSILON),
            _ => true,
        };

        self.stalled = if improved {
            0
        } else {
            self.stalled.saturating_add(1)
        };
    }

    pub(crate) fn len(&self) -> usize {
        self.durations.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.durations.is_empty()
    }

    /// The first sample follows a single warmup, so it still carries allocation and clock ramp
    /// costs. It is dropped once enough samples remain without it.
    fn reliable(&self) -> &[Duration] {
        if self.durations.len() >= DISCARD_THRESHOLD {
            &self.durations[1..]
        } else {
            &self.durations
        }
    }

    pub(crate) fn best(&self) -> Option<Duration> {
        self.reliable().iter().min().copied()
    }

    pub(crate) fn converged(&self) -> bool {
        self.stalled >= CONVERGENCE_ROUNDS
    }

    /// Whether enough samples independently landed under `limit` to trust a short circuit.
    ///
    /// Unlike [`Self::best`] this counts the warmup-biased first sample, deliberately: the bias
    /// is toward being slower, so clearing the limit despite it is the conservative direction.
    /// Excluding it would also make a short circuit impossible during the first pass, where it
    /// is the only sample there is.
    pub(crate) fn confirmed_under(&self, limit: Duration, required: usize) -> bool {
        self.durations.iter().filter(|d| **d <= limit).count() >= required
    }

    pub(crate) fn computation(&self, method: TimingMethod) -> BenchmarkComputations {
        BenchmarkComputations::new(&BenchmarkDurations::from_durations(
            method,
            self.reliable().to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(durations: impl IntoIterator<Item = u64>) -> SampleSet {
        let mut set = SampleSet::default();
        for millis in durations {
            set.push(Duration::from_millis(millis));
        }
        set
    }

    #[test]
    fn keeps_every_sample_below_the_discard_threshold() {
        assert_eq!(
            set([10, 20]).reliable(),
            &[Duration::from_millis(10), Duration::from_millis(20)]
        );
    }

    #[test]
    fn drops_the_warmup_biased_first_sample_once_enough_remain() {
        // The first sample is the fastest here, so dropping it has to move `best` upward:
        // that is the bias being removed, not data being lost.
        let set = set([5, 20, 21]);
        assert_eq!(set.reliable().len(), 2);
        assert_eq!(set.best(), Some(Duration::from_millis(20)));
    }

    #[test]
    fn converges_after_consecutive_non_improving_samples() {
        // Gains under the 2% epsilon do not count as progress. The third push only shifts the
        // window, so two further flat samples are what trips convergence.
        let mut set = set([100, 20, 20]);
        assert!(!set.converged());
        set.push(Duration::from_millis(20));
        assert!(!set.converged());
        set.push(Duration::from_millis(20));
        assert!(set.converged());
    }

    #[test]
    fn aging_out_the_biased_sample_does_not_count_as_a_stall() {
        // The first sample is the fastest, so dropping it raises `best`. That is the window
        // moving, not the candidate failing to improve, and must not count toward convergence.
        let mut set = set([5, 20, 21]);
        assert!(!set.converged());
        set.push(Duration::from_millis(20));
        assert!(!set.converged());
        set.push(Duration::from_millis(20));
        assert!(set.converged());
    }

    #[test]
    fn a_real_improvement_resets_convergence() {
        let mut set = set([20, 20, 20, 20, 20]);
        assert!(set.converged());
        set.push(Duration::from_millis(5));
        assert!(!set.converged());
    }

    #[test]
    fn short_circuit_needs_independent_confirmations() {
        let limit = Duration::from_millis(10);
        // One fast sample among slow ones is exactly the lucky measurement that must not
        // commit a decision to the persistent cache on its own.
        assert!(!set([5, 50]).confirmed_under(limit, 2));
        assert!(set([5, 50, 6]).confirmed_under(limit, 2));
    }

    #[test]
    fn computation_is_built_from_the_reliable_samples_only() {
        let set = set([1, 20, 30]);
        let computation = set.computation(TimingMethod::System);
        assert_eq!(computation.min, Duration::from_millis(20));
        assert_eq!(computation.max, Duration::from_millis(30));
    }

    #[test]
    fn empty_set_has_no_best() {
        let set = SampleSet::default();
        assert!(set.is_empty());
        assert_eq!(set.best(), None);
        assert_eq!(set.len(), 0);
    }
}
