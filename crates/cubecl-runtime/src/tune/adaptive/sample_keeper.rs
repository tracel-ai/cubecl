use alloc::vec::Vec;
use core::time::Duration;
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};
use cubecl_common::profile::TimingMethod;

/// Keeps the timings of one candidate, and the little state needed to decide whether it is still
/// worth sampling.
///
/// `BenchmarkComputations::score` is not usable while tuning is in flight: at a single sample the
/// variance is zero and reads as perfect stability. This tracks the sample count explicitly so
/// elimination can require evidence instead of inferring it from a degenerate variance.
#[derive(Debug, Default)]
pub struct SampleKeeper {
    /// Every sample, in the order it was taken (Arrival Usage), read from the second on once
    /// enough were taken (Position Usage).
    durations: Vec<Duration>,
    /// Samples in a row that did not improve on the best.
    stalled_count: u8,
    summary: SampleSummary,
}

/// What a [`SampleKeeper`] knows of its candidate so far.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SampleSummary {
    pub count: usize,
    /// The fastest reliable sample: the first one is left out once enough remain without it.
    pub best: Option<Duration>,
    /// Whether the candidate stopped improving.
    pub converged: bool,
}

impl SampleKeeper {
    /// Keep `duration` as the candidate's latest sample.
    pub fn keep(&mut self, duration: Duration) {
        let before = self.summary.best;
        self.durations.push(duration);
        let best = self.reliable().iter().min().copied();
        self.stalled_count = self.stalled_after(before, best);
        self.summary = SampleSummary {
            count: self.durations.len(),
            best,
            converged: self.stalled_count >= CONVERGENCE_SAMPLE_COUNT,
        };
    }

    pub fn summary(&self) -> SampleSummary {
        self.summary
    }

    /// Whether enough samples independently landed under `limit` to trust a short circuit.
    ///
    /// Unlike [`SampleSummary::best`] this counts the warmup-biased first sample, deliberately:
    /// the bias is toward being slower, so clearing the limit despite it is the conservative
    /// direction. Excluding it would also make a short circuit impossible during the first pass,
    /// where it is the only sample there is.
    pub fn confirmed_under(&self, limit: Duration, required: usize) -> bool {
        self.durations.iter().filter(|d| **d <= limit).count() >= required
    }

    /// Whether any sample at all landed under `limit` — the cheap gate before a candidate is worth
    /// spending confirmation samples on.
    pub fn any_under(&self, limit: Duration) -> bool {
        self.durations.iter().any(|d| *d <= limit)
    }

    /// What the reliable samples measure, timed by `method`.
    pub fn computation(&self, method: TimingMethod) -> BenchmarkComputations {
        BenchmarkComputations::new(&BenchmarkDurations::from_durations(
            method,
            self.reliable().to_vec(),
        ))
    }
}

impl SampleKeeper {
    /// The first sample follows a single warmup, so it still carries allocation and clock ramp
    /// costs. It is dropped once enough samples remain without it.
    fn reliable(&self) -> &[Duration] {
        if self.durations.len() >= DISCARD_AFTER_SAMPLE_COUNT {
            &self.durations[1..]
        } else {
            &self.durations
        }
    }

    /// The stalled count once the best moved from `before` to `after`.
    ///
    /// The sample that ages the biased first one out resets it: the two bests are then taken over
    /// different samples, and their difference says nothing about progress.
    fn stalled_after(&self, before: Option<Duration>, after: Option<Duration>) -> u8 {
        if self.durations.len() == DISCARD_AFTER_SAMPLE_COUNT {
            return 0;
        }
        match (before, after) {
            (Some(before), Some(after))
                if TimeChange::of(after, before) != TimeChange::Improvement =>
            {
                self.stalled_count.saturating_add(1)
            }
            _ => 0,
        }
    }
}

/// How a time compares to a baseline, once noise is set aside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeChange {
    /// Faster than the baseline by more than the [`IMPROVEMENT_THRESHOLD_FRACTION`].
    Improvement,
    /// Within the threshold of the baseline either way: indistinguishable from noise.
    Neutral,
    /// Slower than the baseline by more than the threshold.
    Regression,
}

impl TimeChange {
    pub fn of(time: Duration, baseline: Duration) -> Self {
        let faster = time < baseline.mul_f64(1.0 - IMPROVEMENT_THRESHOLD_FRACTION);
        let slower = time > baseline.mul_f64(1.0 + IMPROVEMENT_THRESHOLD_FRACTION);
        match (faster, slower) {
            (true, _) => Self::Improvement,
            (_, true) => Self::Regression,
            _ => Self::Neutral,
        }
    }
}

/// Samples required before the warmup-biased first sample is discarded.
const DISCARD_AFTER_SAMPLE_COUNT: usize = 3;
/// Relative change a time must exceed to count as an improvement or a regression rather than
/// noise.
const IMPROVEMENT_THRESHOLD_FRACTION: f64 = 0.02;
/// Consecutive non-improving samples before a candidate is considered converged.
const CONVERGENCE_SAMPLE_COUNT: u8 = 2;

#[cfg(test)]
mod tests {
    use super::*;

    fn keeper(durations: impl IntoIterator<Item = u64>) -> SampleKeeper {
        let mut keeper = SampleKeeper::default();
        for millis in durations {
            keeper.keep(Duration::from_millis(millis));
        }
        keeper
    }

    #[test]
    fn keeps_every_sample_below_the_discard_count() {
        assert_eq!(
            keeper([10, 20]).reliable(),
            &[Duration::from_millis(10), Duration::from_millis(20)]
        );
    }

    #[test]
    fn drops_the_warmup_biased_first_sample_once_enough_remain() {
        assert_eq!(keeper([5, 20, 21]).reliable().len(), 2);
    }

    #[test]
    fn dropping_the_first_sample_raises_the_best() {
        // The first sample is the fastest here, so dropping it has to move `best` upward:
        // that is the bias being removed, not data being lost.
        assert_eq!(
            keeper([5, 20, 21]).summary().best,
            Some(Duration::from_millis(20))
        );
    }

    #[test]
    fn one_flat_sample_after_the_window_shift_does_not_converge() {
        // Gains under the 2% epsilon do not count as progress, but the third sample only shifts
        // the window.
        assert!(!keeper([100, 20, 20, 20]).summary().converged);
    }

    #[test]
    fn two_flat_samples_after_the_window_shift_converge() {
        assert!(keeper([100, 20, 20, 20, 20]).summary().converged);
    }

    #[test]
    fn aging_out_the_biased_sample_does_not_count_as_a_stall() {
        // The first sample is the fastest, so dropping it raises `best`. That is the window
        // moving, not the candidate failing to improve, and must not count toward convergence.
        assert!(!keeper([5, 20, 21, 20]).summary().converged);
    }

    #[test]
    fn flat_samples_after_the_biased_one_aged_out_converge() {
        assert!(keeper([5, 20, 21, 20, 20]).summary().converged);
    }

    #[test]
    fn a_real_improvement_resets_convergence() {
        assert!(!keeper([20, 20, 20, 20, 20, 5]).summary().converged);
    }

    #[test]
    fn one_fast_sample_among_slow_ones_is_no_confirmation() {
        // The lucky measurement that must not commit a decision to the persistent cache on its
        // own.
        assert!(!keeper([5, 50]).confirmed_under(Duration::from_millis(10), 2));
    }

    #[test]
    fn two_fast_samples_confirm_the_limit() {
        assert!(keeper([5, 50, 6]).confirmed_under(Duration::from_millis(10), 2));
    }

    #[test]
    fn any_sample_under_the_limit_is_worth_confirming() {
        assert!(keeper([50, 5]).any_under(Duration::from_millis(10)));
    }

    #[test]
    fn samples_over_the_limit_are_not_worth_confirming() {
        assert!(!keeper([50, 60]).any_under(Duration::from_millis(10)));
    }

    #[test]
    fn computation_is_built_from_the_reliable_samples_only() {
        let computation = keeper([1, 20, 30]).computation(TimingMethod::System);
        assert_eq!(computation.min, Duration::from_millis(20));
        assert_eq!(computation.max, Duration::from_millis(30));
    }

    #[test]
    fn an_empty_keeper_has_no_best() {
        assert_eq!(SampleKeeper::default().summary(), SampleSummary::default());
    }

    #[test]
    fn a_time_within_the_threshold_is_neutral() {
        let change = TimeChange::of(Duration::from_micros(99), Duration::from_micros(100));
        assert_eq!(change, TimeChange::Neutral);
    }

    #[test]
    fn a_time_faster_past_the_threshold_is_an_improvement() {
        let change = TimeChange::of(Duration::from_micros(90), Duration::from_micros(100));
        assert_eq!(change, TimeChange::Improvement);
    }

    #[test]
    fn a_time_slower_past_the_threshold_is_a_regression() {
        let change = TimeChange::of(Duration::from_micros(110), Duration::from_micros(100));
        assert_eq!(change, TimeChange::Regression);
    }
}
