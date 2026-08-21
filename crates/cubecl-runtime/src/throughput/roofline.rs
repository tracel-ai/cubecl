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
#[cfg_attr(autotune_persistence, derive(serde::Serialize, serde::Deserialize))]
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
    /// `None` for a `peak_per_s` that is zero, negative, `NaN`, or infinite:
    /// none of those describe a real ceiling to divide by.
    pub fn time_at_peak(&self) -> Option<Duration> {
        if self.peak_per_s.is_normal() {
            Some(Duration::from_secs_f64(
                self.amount as f64 / self.peak_per_s,
            ))
        } else {
            None
        }
    }
}

/// The roofline convention this module uses throughout: a run cannot finish
/// faster than its slowest resource allows even under perfect overlap
/// between them, so that resource is the one binding the achievable
/// duration, whatever the others manage. [`binding_achieved`] applies the
/// same reduction to a run's actually measured rates.
///
/// The resource requiring the most time even at its own peak: the entry with
/// the largest [`ResourceBound::time_at_peak`].
///
/// `None` if every entry's `time_at_peak` is `None`, or `bounds` is empty.
pub fn binding_resource(bounds: &[ResourceBound]) -> Option<&ResourceBound> {
    bounds
        .iter()
        .filter(|bound| bound.time_at_peak().is_some())
        .max_by_key(|bound| bound.time_at_peak())
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

/// Scores an actual run against a set of [`ResourceBound`]s, one resource at
/// a time.
///
/// `duration` is the run's actual, measured duration, shared by every bound:
/// they all describe the same execution, so the same wall time buys each
/// resource a different achieved rate against its own peak. Results come
/// back in `bounds` order. A zero duration yields `NaN` achieved rates rather
/// than dividing by zero.
pub fn score_resources(duration: Duration, bounds: &[ResourceBound]) -> Vec<AchievedThroughput> {
    bounds
        .iter()
        .map(|bound| {
            let achieved_per_s = if duration.is_zero() {
                f64::NAN
            } else {
                bound.amount as f64 / duration.as_secs_f64()
            };

            AchievedThroughput {
                achieved_per_s,
                fraction_of_peak: achieved_per_s / bound.peak_per_s,
            }
        })
        .collect()
}

/// The resource that actually governed the run's duration: [`binding_resource`]'s
/// reduction applied to achieved scores instead of raw bounds. Every entry
/// here shares one `duration`, so ranking by `fraction_of_peak` gives the
/// same order ranking the underlying bounds by `time_at_peak` would.
///
/// `NaN` entries (a zero duration, or a zero, negative, or non-finite peak)
/// cannot be compared and are skipped; `None` if every entry is such, or
/// `scores` is empty.
pub fn binding_achieved(scores: &[AchievedThroughput]) -> Option<&AchievedThroughput> {
    scores
        .iter()
        .filter(|score| score.fraction_of_peak.is_finite())
        .max_by(|a, b| a.fraction_of_peak.total_cmp(&b.fraction_of_peak))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(amount: usize, peak_per_s: f64) -> ResourceBound {
        ResourceBound { amount, peak_per_s }
    }

    #[test]
    fn time_at_peak_is_amount_over_peak() {
        assert_eq!(bound(8, 4.0).time_at_peak(), Some(Duration::from_secs(2)));
    }

    #[test]
    fn time_at_peak_is_none_for_a_non_normal_peak() {
        assert_eq!(bound(8, 0.0).time_at_peak(), None);
        assert_eq!(bound(8, f64::NAN).time_at_peak(), None);
        assert_eq!(bound(8, f64::INFINITY).time_at_peak(), None);
    }

    #[test]
    fn binding_resource_is_the_one_needing_the_most_time_at_peak() {
        // 8 ops at 4 ops/s takes 2s at peak; 8 ops at 8 ops/s takes 1s: the
        // first resource is the one that would still take longer even
        // running flat out, so it is the one that binds.
        let slower = bound(8, 4.0);
        let faster = bound(8, 8.0);

        assert_eq!(binding_resource(&[slower, faster]), Some(&slower));
    }

    #[test]
    fn binding_resource_skips_non_normal_peaks_and_is_none_if_all_are() {
        let unusable = bound(8, 0.0);
        let usable = bound(8, 4.0);

        assert_eq!(binding_resource(&[unusable, usable]), Some(&usable));
        assert_eq!(binding_resource(&[unusable]), None);
        assert_eq!(binding_resource(&[]), None);
    }

    #[test]
    fn score_resources_reports_achieved_rate_and_fraction_of_peak() {
        let bounds = [bound(100, 200.0), bound(400, 800.0)];

        let scores = score_resources(Duration::from_secs(1), &bounds);

        assert_eq!(scores[0].achieved_per_s, 100.0);
        assert_eq!(scores[0].fraction_of_peak, 0.5);
        assert_eq!(scores[1].achieved_per_s, 400.0);
        assert_eq!(scores[1].fraction_of_peak, 0.5);
    }

    #[test]
    fn a_zero_duration_reports_nan_instead_of_dividing_by_zero() {
        let scores = score_resources(Duration::ZERO, &[bound(100, 200.0)]);

        assert!(scores[0].achieved_per_s.is_nan());
        assert!(scores[0].fraction_of_peak.is_nan());
    }

    /// A matmul-shaped run: large A/B reads against a read peak, a small C
    /// write against a write peak. The two must score independently, and the
    /// read, which alone would still need more time even at its own peak
    /// (0.9s vs the write's 0.5s), must be the one that binds even though it
    /// moves far more bytes than the write, not fewer.
    #[test]
    fn resources_with_different_peaks_score_independently_and_pick_the_slower_one() {
        let duration = Duration::from_secs(1);
        let read = bound(900_000, 1_000_000.0); // 0.9s at peak
        let write = bound(100_000, 200_000.0); // 0.5s at peak

        assert_eq!(binding_resource(&[read, write]), Some(&read));

        let scores = score_resources(duration, &[read, write]);

        assert_eq!(scores[0].achieved_per_s, 900_000.0);
        assert_eq!(scores[0].fraction_of_peak, 0.9);
        assert_eq!(scores[1].achieved_per_s, 100_000.0);
        assert_eq!(scores[1].fraction_of_peak, 0.5);

        let binding = binding_achieved(&scores).unwrap();
        assert_eq!(binding.fraction_of_peak, 0.9);
    }

    #[test]
    fn binding_achieved_skips_non_finite_entries_and_is_none_if_all_are() {
        let finite = AchievedThroughput {
            achieved_per_s: 10.0,
            fraction_of_peak: 0.4,
        };
        let non_finite = AchievedThroughput {
            achieved_per_s: f64::NAN,
            fraction_of_peak: f64::NAN,
        };

        assert_eq!(
            binding_achieved(&[non_finite, finite])
                .unwrap()
                .fraction_of_peak,
            0.4
        );
        assert!(binding_achieved(&[non_finite]).is_none());
        assert!(binding_achieved(&[]).is_none());
    }
}
