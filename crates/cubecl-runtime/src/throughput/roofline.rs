use core::time::Duration;

use alloc::vec::Vec;

/// A resource's roofline record: how much of it a run must move, against the
/// peak rate that resource can sustain.
///
/// Neutral by design: nothing here says whether `amount` counts bytes or
/// operations, or where `peak_per_s` came from, measured or modeled. That is
/// what lets [`AutotuneBound`](crate::tune::AutotuneBound) compose this with
/// a threshold instead of reimplementing the arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceBound {
    /// How much of the resource the run must move.
    pub amount: usize,
    /// The peak rate this resource can sustain, in the same unit as `amount`
    /// per second.
    pub peak_per_s: f64,
}

impl ResourceBound {
    /// Time `amount` would take running at `peak_per_s`, with no allowance
    /// for anything else the kernel does concurrently.
    ///
    /// # Guarantees
    ///
    /// - `None` for a `peak_per_s` that is zero, subnormal, negative, `NaN`, or
    ///   infinite: none of those describe a real ceiling to divide by.
    /// - `None` too for a time no [`Duration`] holds.
    pub fn time_at_peak(&self) -> Option<Duration> {
        if !self.peak_per_s.is_normal() || self.peak_per_s < 0.0 {
            return None;
        }
        Duration::try_from_secs_f64(self.amount as f64 / self.peak_per_s).ok()
    }

    /// The roofline convention this module uses throughout: a run cannot finish
    /// faster than its slowest resource allows even under perfect overlap
    /// between them, so that resource is the one binding the achievable
    /// duration, whatever the others manage. [`AchievedThroughput::binding`] applies the
    /// same reduction to a run's actually measured rates.
    ///
    /// The resource requiring the most time even at its own peak: the entry with
    /// the largest [`ResourceBound::time_at_peak`].
    ///
    /// `None` if every entry's `time_at_peak` is `None`, or `bounds` is empty.
    pub fn binding(bounds: &[Self]) -> Option<&Self> {
        bounds
            .iter()
            .filter(|bound| bound.time_at_peak().is_some())
            .max_by_key(|bound| bound.time_at_peak())
    }
}

/// One resource's achieved rate during a measured run, against its modeled
/// peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AchievedThroughput {
    /// Achieved rate: `amount / duration`.
    pub achieved_per_s: f64,
    /// `achieved_per_s / peak_per_s`, as a fraction. Not clamped to `[0, 1]`:
    /// a run beating the modeled peak is a finding about the model, not an
    /// error to hide.
    pub fraction_of_peak: f64,
}

impl AchievedThroughput {
    /// Scores an actual run against a set of [`ResourceBound`]s, one resource at
    /// a time.
    ///
    /// `duration` is the run's actual, measured duration, shared by every bound:
    /// they all describe the same execution, so the same wall time buys each
    /// resource a different achieved rate against its own peak. Results come
    /// back in `bounds` order. A zero duration yields `NaN` achieved rates rather
    /// than dividing by zero.
    pub fn score(duration: Duration, bounds: &[ResourceBound]) -> Vec<Self> {
        bounds
            .iter()
            .map(|bound| {
                let achieved_per_s = if duration.is_zero() {
                    f64::NAN
                } else {
                    bound.amount as f64 / duration.as_secs_f64()
                };

                Self {
                    achieved_per_s,
                    fraction_of_peak: achieved_per_s / bound.peak_per_s,
                }
            })
            .collect()
    }

    /// The resource that actually governed the run's duration: [`ResourceBound::binding`]'s
    /// reduction applied to achieved scores instead of raw bounds. Every entry
    /// here shares one `duration`, so ranking by `fraction_of_peak` gives the
    /// same order ranking the underlying bounds by `time_at_peak` would.
    ///
    /// `NaN` entries (a zero duration, or a zero, negative, or non-finite peak)
    /// cannot be compared and are skipped; `None` if every entry is such, or
    /// `scores` is empty.
    pub fn binding(scores: &[Self]) -> Option<&Self> {
        scores
            .iter()
            .filter(|score| score.fraction_of_peak.is_finite())
            .max_by(|a, b| a.fraction_of_peak.total_cmp(&b.fraction_of_peak))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(amount: usize, peak_per_s: f64) -> ResourceBound {
        ResourceBound { amount, peak_per_s }
    }

    fn score(fraction_of_peak: f64) -> AchievedThroughput {
        AchievedThroughput {
            achieved_per_s: fraction_of_peak,
            fraction_of_peak,
        }
    }

    #[test]
    fn time_at_peak_is_amount_over_peak() {
        assert_eq!(bound(8, 4.0).time_at_peak(), Some(Duration::from_secs(2)));
    }

    #[test]
    fn time_at_peak_is_none_for_a_non_normal_peak() {
        for peak in [0.0, f64::NAN, f64::INFINITY] {
            assert_eq!(bound(8, peak).time_at_peak(), None, "at {peak}");
        }
    }

    #[test]
    fn time_at_peak_is_none_for_a_negative_peak() {
        assert_eq!(bound(8, -4.0).time_at_peak(), None);
    }

    #[test]
    fn time_at_peak_is_none_for_a_time_no_duration_holds() {
        assert_eq!(bound(usize::MAX, f64::MIN_POSITIVE).time_at_peak(), None);
    }

    /// 8 ops at 4 ops/s takes 2s at peak; 8 ops at 8 ops/s takes 1s: the first resource is the
    /// one that would still take longer even running flat out, so it is the one that binds.
    #[test]
    fn the_binding_resource_needs_the_most_time_at_peak() {
        let (slower, faster) = (bound(8, 4.0), bound(8, 8.0));

        assert_eq!(ResourceBound::binding(&[slower, faster]), Some(&slower));
    }

    #[test]
    fn a_non_normal_peak_never_binds() {
        let (unusable, usable) = (bound(8, 0.0), bound(8, 4.0));

        assert_eq!(ResourceBound::binding(&[unusable, usable]), Some(&usable));
    }

    #[test]
    fn nothing_binds_without_a_usable_peak() {
        assert_eq!(ResourceBound::binding(&[bound(8, 0.0)]), None);
    }

    #[test]
    fn a_score_is_the_achieved_rate_and_its_fraction_of_peak() {
        let scores = AchievedThroughput::score(Duration::from_secs(1), &[bound(100, 200.0)]);

        assert_eq!(
            scores,
            [AchievedThroughput {
                achieved_per_s: 100.0,
                fraction_of_peak: 0.5,
            }]
        );
    }

    #[test]
    fn a_zero_duration_scores_nan_instead_of_dividing_by_zero() {
        let scores = AchievedThroughput::score(Duration::ZERO, &[bound(100, 200.0)]);

        assert!(scores[0].achieved_per_s.is_nan());
    }

    /// A matmul-shaped run: large A/B reads against a read peak, a small C write against a write
    /// peak. Each resource scores against its own peak.
    #[test]
    fn resources_with_different_peaks_score_independently() {
        let read = bound(900_000, 1_000_000.0);
        let write = bound(100_000, 200_000.0);

        let scores = AchievedThroughput::score(Duration::from_secs(1), &[read, write]);

        assert_eq!(
            scores
                .iter()
                .map(|s| s.fraction_of_peak)
                .collect::<Vec<_>>(),
            [0.9, 0.5]
        );
    }

    #[test]
    fn the_binding_score_is_the_highest_fraction_of_peak() {
        assert_eq!(
            AchievedThroughput::binding(&[score(0.9), score(0.5)]),
            Some(&score(0.9))
        );
    }

    #[test]
    fn a_non_finite_score_never_binds() {
        assert_eq!(
            AchievedThroughput::binding(&[score(f64::NAN), score(0.4)]),
            Some(&score(0.4))
        );
    }

    #[test]
    fn nothing_binds_without_a_finite_score() {
        assert_eq!(AchievedThroughput::binding(&[score(f64::NAN)]), None);
    }
}
