use core::time::Duration;

use alloc::vec::Vec;

use crate::config::autotune::AutotuneLevel;
use crate::throughput::ResourceBound;

/// What a tune of one key is measured against: the throughput bounds of its problem, and the cost
/// of a launch on top of them.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct Bounds {
    /// Every resource the problem moves, each against its peak: read whole for the slowest
    /// (Arrival Usage).
    pub bounds: Vec<AutotuneBound>,
    /// What a launch costs beside the work, added to the time limit.
    pub launch_overhead: Duration,
}

// Sound because [`AutotuneBound`] compares its floats bitwise, so equality stays reflexive.
impl Eq for Bounds {}

/// A bound for autotuning a throughput kernel: a [`ResourceBound`] plus the
/// threshold over which the kernel is considered accurate.
#[derive(Debug, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct AutotuneBound {
    /// How much work, against what peak throughput.
    pub resource: ResourceBound,
    /// The fraction of the peak over which a candidate is close enough to it.
    pub threshold_fraction: f32,
}

/// Bitwise comparison of the measured throughputs, so that equality is reflexive even if a
/// degenerate measurement ever produces a `NaN`, which is what makes the [`Eq`] below sound.
impl PartialEq for AutotuneBound {
    fn eq(&self, other: &Self) -> bool {
        self.resource.peak_per_s.to_bits() == other.resource.peak_per_s.to_bits()
            && self.threshold_fraction.to_bits() == other.threshold_fraction.to_bits()
            && self.resource.amount == other.resource.amount
    }
}

impl Eq for AutotuneBound {}

/// Target fractions of modeled peak compute and memory roofline throughput.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct Thresholds {
    /// Fraction of peak compute throughput expected.
    pub compute_fraction: f32,
    /// Fraction of peak memory bandwidth expected.
    pub memory_fraction: f32,
}

impl Thresholds {
    /// `fraction` for both bounds.
    pub const fn uniform_fraction(fraction: f32) -> Self {
        Self {
            compute_fraction: fraction,
            memory_fraction: fraction,
        }
    }

    /// The threshold that sets no limit, so every candidate is measured.
    /// [`AutotuneBound::time_limit`] declines it.
    pub const UNBOUNDED: Self = Self::uniform_fraction(0.0);

    /// The fraction of peak an [`AutotuneLevel`] settles for.
    ///
    /// A higher threshold is a tighter time limit, since the limit is the roofline time
    /// divided by the threshold. The fractions are ad-hoc observations rather than a
    /// systematic sweep, so nothing should depend on the exact values.
    pub const fn for_level(level: &AutotuneLevel) -> Self {
        match level {
            AutotuneLevel::Minimal => Self::uniform_fraction(0.6),
            AutotuneLevel::Balanced => Self::uniform_fraction(0.8),
            AutotuneLevel::Extensive => Self::uniform_fraction(0.95),
            AutotuneLevel::Full => Self::UNBOUNDED,
        }
    }
}

impl Default for Thresholds {
    /// The roofline itself: a candidate is expected to reach 100% of the modeled peak, which
    /// is the only fraction that needs no justification.
    fn default() -> Self {
        Self::uniform_fraction(1.0)
    }
}

impl AutotuneBound {
    /// The slowest median a candidate may take and still count as close enough to the peak: the
    /// time at the peak divided by the threshold.
    ///
    /// # Guarantees
    ///
    /// - `None` for a threshold that is zero, negative, subnormal or `NaN`, and for a peak
    ///   [`ResourceBound::time_at_peak`] gives no time for.
    pub fn time_limit(&self) -> Option<Duration> {
        // The threshold divides the roofline time. A negative one panics `div_f64`, and
        // zero or a subnormal divides the limit away, so neither is a limit to compute.
        if self.threshold_fraction <= 0.0 || !self.threshold_fraction.is_normal() {
            return None;
        }
        self.resource
            .time_at_peak()
            .map(|limit| limit.div_f64(self.threshold_fraction as f64))
    }
}

impl Bounds {
    /// The time limit of the slowest bound, plus the launch overhead.
    ///
    /// # Guarantees
    ///
    /// - `None` where no bound gives a [time limit](AutotuneBound::time_limit): the overhead alone
    ///   is no limit.
    pub fn time_limit(&self) -> Option<Duration> {
        self.bounds
            .iter()
            .filter_map(AutotuneBound::time_limit)
            .max()
            .map(|limit| limit + self.launch_overhead)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn bound(ops_count: usize, throughput: f64, threshold_fraction: f32) -> AutotuneBound {
        AutotuneBound {
            resource: ResourceBound {
                amount: ops_count,
                peak_per_s: throughput,
            },
            threshold_fraction,
        }
    }

    fn without_overhead(bounds: Vec<AutotuneBound>) -> Bounds {
        Bounds {
            bounds,
            launch_overhead: Duration::ZERO,
        }
    }

    #[test]
    fn time_limit_is_ops_over_throughput_scaled_by_threshold() {
        // (8 ops / 4 ops/s) / 0.5 = 4s. Powers of two keep the f64 math exact.
        let limit = bound(8, 4.0, 0.5).time_limit();
        assert_eq!(limit, Some(Duration::from_secs(4)));
    }

    #[test]
    fn time_limit_is_none_when_the_peak_is_not_a_rate() {
        // A zero/NaN/inf throughput or a zero threshold would divide by zero or blow up,
        // so the bound disables the short-circuit instead of producing a garbage limit.
        assert_eq!(bound(8, 0.0, 0.5).time_limit(), None);
        assert_eq!(bound(8, f64::NAN, 0.5).time_limit(), None);
        assert_eq!(bound(8, f64::INFINITY, 0.5).time_limit(), None);
        assert_eq!(bound(8, -4.0, 0.5).time_limit(), None);
        assert_eq!(bound(8, 4.0, 0.0).time_limit(), None);
    }

    #[test]
    fn time_limit_declines_a_negative_threshold_rather_than_panicking() {
        // A negative threshold is normal, so `is_normal` alone lets it reach
        // `Duration::div_f64`, which panics on a negative divisor. The bound builders take
        // the threshold straight from the caller, where it can be computed.
        assert_eq!(bound(8, 4.0, -0.5).time_limit(), None);
        assert_eq!(bound(8, 4.0, f32::NEG_INFINITY).time_limit(), None);
    }

    #[test]
    fn vec_time_limit_takes_the_roofline_max_not_min() {
        // Two simultaneous resource bounds (e.g. compute vs memory): the achievable floor
        // is the *slower* one, so the reduction must be `max`. `min` would pick the
        // unreachable 1s and the short-circuit would never fire.
        let compute = bound(8, 4.0, 1.0); // 2s
        let memory = bound(8, 8.0, 1.0); // 1s
        let limit = without_overhead(vec![compute, memory]).time_limit();
        assert_eq!(limit, Some(Duration::from_secs(2)));
    }

    #[test]
    fn vec_time_limit_skips_non_normal_bounds_and_is_none_when_empty() {
        // A non-normal bound is filtered out rather than poisoning the reduction.
        let limit = without_overhead(vec![bound(8, 0.0, 1.0), bound(8, 4.0, 1.0)]).time_limit();
        assert_eq!(limit, Some(Duration::from_secs(2)));
    }

    #[test]
    fn bounds_time_limit_adds_launch_overhead() {
        let bounds = Bounds {
            bounds: vec![bound(8, 4.0, 1.0)], // 2s
            launch_overhead: Duration::from_millis(500),
        };
        assert_eq!(bounds.time_limit(), Some(Duration::from_millis(2500)));
    }

    #[test]
    fn a_level_tightens_the_limit_as_it_rises() {
        let threshold = |level| Thresholds::for_level(&level).compute_fraction;
        assert!(threshold(AutotuneLevel::Minimal) < threshold(AutotuneLevel::Balanced));
        assert!(threshold(AutotuneLevel::Balanced) < threshold(AutotuneLevel::Extensive));
        assert_eq!(
            Thresholds::for_level(&AutotuneLevel::Full),
            Thresholds::UNBOUNDED,
            "full measures everything"
        );
    }

    #[test]
    fn an_unbounded_threshold_gives_no_time_limit() {
        // What lets `Full` be handed to a bound like any other threshold.
        let unbounded = bound(8, 4.0, Thresholds::UNBOUNDED.compute_fraction);
        assert_eq!(unbounded.time_limit(), None);
    }

    #[test]
    fn the_default_threshold_is_the_roofline_itself() {
        assert_eq!(Thresholds::default(), Thresholds::uniform_fraction(1.0));
    }

    #[test]
    fn bounds_time_limit_is_none_without_usable_bounds() {
        // No usable bound means no limit at all — the launch overhead is not a limit on
        // its own, so the short-circuit stays disabled.
        let bounds = Bounds {
            bounds: vec![],
            launch_overhead: Duration::from_millis(500),
        };
        assert_eq!(bounds.time_limit(), None);
    }
}
