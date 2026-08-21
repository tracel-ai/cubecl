use core::time::Duration;

use alloc::vec::Vec;

use crate::throughput::{ThroughputKey, ThroughputValue};
use crate::tune::TuneInputs;

// A bound-builder constructs `AutotuneBound { resource: ResourceBound { .. }, .. }`
// alongside this module's own types, so the neutral record is re-exported
// here too, the same way `Work` is below.
pub use crate::throughput::ResourceBound;

/// A set of [`AutotuneBound`]s for a given key and reference inputs, with a launch overhead.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(autotune_persistence, derive(serde::Serialize, serde::Deserialize))]
pub struct Bounds {
    /// The bounds for autotuning.
    pub bounds: Vec<AutotuneBound>,
    /// The launch overhead for autotuning.
    pub launch_overhead: Duration,
}

// Sound because [`AutotuneBound`] compares its floats bitwise, so equality stays reflexive.
impl Eq for Bounds {}

/// Produces a set of [`AutotuneBound`]s for a given key and reference inputs.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid bounds generator",
    label = "invalid bounds generator"
)]
pub trait BoundsGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// Generate a set of bounds for a given key and reference inputs.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> Bounds;
}

/// `Fn(&K, &A) -> Bounds` acts as a [`BoundsGenerator`] when `A` is an owned type. For
/// multi-input kernels, `A` is a tuple that the closure destructures internally.
impl<K, Func, A> BoundsGenerator<K, A> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> Bounds,
{
    #[inline]
    fn generate<'a>(&self, key: &K, inputs: &<A as TuneInputs>::At<'a>) -> Bounds {
        (self)(key, inputs)
    }
}

/// A calculator that determines the time limit for autotune bounds.
pub trait TimeBound {
    /// Returns the time limit for autotune bounds.
    fn time_limit(&self) -> Option<Duration>;
}

/// A bound for autotuning a throughput kernel: a [`ResourceBound`] plus the
/// threshold over which the kernel is considered accurate.
#[derive(Debug, Clone)]
#[cfg_attr(autotune_persistence, derive(serde::Serialize, serde::Deserialize))]
pub struct AutotuneBound {
    /// How much work, against what peak throughput.
    pub resource: ResourceBound,
    /// The threshold for this bound, over which the kernel will be considered accurate.
    pub threshold: f32,
}

/// Bitwise comparison of the measured throughputs, so that equality is reflexive even if a
/// degenerate measurement ever produces a `NaN`, which is what makes the [`Eq`] below sound.
impl PartialEq for AutotuneBound {
    fn eq(&self, other: &Self) -> bool {
        self.resource.peak_per_s.to_bits() == other.resource.peak_per_s.to_bits()
            && self.threshold.to_bits() == other.threshold.to_bits()
            && self.resource.amount == other.resource.amount
    }
}

impl Eq for AutotuneBound {}

// `cubecl-common` cannot depend on `cubecl-runtime`, so `Work` lives there and
// autotune bounds and benchmark reporting share one definition.
pub use cubecl_common::work::Work;

/// Target fractions of modeled peak compute and memory roofline throughput.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct Thresholds {
    /// Fraction of peak compute throughput expected.
    pub compute: f32,
    /// Fraction of peak memory bandwidth expected.
    pub memory: f32,
}

impl Thresholds {
    /// The same fraction for both bounds.
    pub const fn uniform(fraction: f32) -> Self {
        Self {
            compute: fraction,
            memory: fraction,
        }
    }
}

impl Default for Thresholds {
    /// The roofline itself: a candidate is expected to reach 100% of the modeled peak, which
    /// is the only fraction that needs no justification.
    fn default() -> Self {
        Self::uniform(1.0)
    }
}

/// Standardizes the creation of compute and memory [`AutotuneBound`]s.
pub fn calculate_bounds(
    work: Work,
    thresholds: Thresholds,
    compute_throughput: &ThroughputValue,
    memory_throughput: &ThroughputValue,
    memory_key: &ThroughputKey,
) -> Vec<AutotuneBound> {
    alloc::vec![
        AutotuneBound {
            resource: ResourceBound {
                amount: work.compute_ops,
                peak_per_s: compute_throughput.ops_per_s(),
            },
            threshold: thresholds.compute,
        },
        AutotuneBound {
            resource: ResourceBound {
                amount: work.bytes,
                peak_per_s: memory_throughput.bytes_per_s(memory_key),
            },
            threshold: thresholds.memory,
        },
    ]
}

impl TimeBound for AutotuneBound {
    fn time_limit(&self) -> Option<Duration> {
        if !self.threshold.is_normal() {
            return None;
        }
        self.resource
            .time_at_peak()
            .map(|limit| limit.div_f64(self.threshold as f64))
    }
}

impl<B: TimeBound> TimeBound for Vec<B> {
    fn time_limit(&self) -> Option<Duration> {
        self.iter().filter_map(|b| b.time_limit()).max()
    }
}

impl TimeBound for Bounds {
    fn time_limit(&self) -> Option<Duration> {
        self.bounds
            .time_limit()
            .map(|limit| limit + self.launch_overhead)
    }
}

#[cfg(test)]
mod tests {
    use crate::throughput::ThroughputMode;

    use super::*;
    use alloc::vec;

    fn bound(ops_count: usize, throughput: f64, threshold: f32) -> AutotuneBound {
        AutotuneBound {
            resource: ResourceBound {
                amount: ops_count,
                peak_per_s: throughput,
            },
            threshold,
        }
    }

    #[test]
    fn time_limit_is_ops_over_throughput_scaled_by_threshold() {
        // (8 ops / 4 ops/s) / 0.5 = 4s. Powers of two keep the f64 math exact.
        let limit = bound(8, 4.0, 0.5).time_limit();
        assert_eq!(limit, Some(Duration::from_secs(4)));
    }

    #[test]
    fn time_limit_is_none_when_inputs_are_not_normal() {
        // A zero/NaN/inf throughput or a zero threshold would divide by zero or blow up,
        // so the bound disables the short-circuit instead of producing a garbage limit.
        assert_eq!(bound(8, 0.0, 0.5).time_limit(), None);
        assert_eq!(bound(8, f64::NAN, 0.5).time_limit(), None);
        assert_eq!(bound(8, f64::INFINITY, 0.5).time_limit(), None);
        assert_eq!(bound(8, 4.0, 0.0).time_limit(), None);
    }

    #[test]
    fn vec_time_limit_takes_the_roofline_max_not_min() {
        // Two simultaneous resource bounds (e.g. compute vs memory): the achievable floor
        // is the *slower* one, so the reduction must be `max`. `min` would pick the
        // unreachable 1s and the short-circuit would never fire.
        let compute = bound(8, 4.0, 1.0); // 2s
        let memory = bound(8, 8.0, 1.0); // 1s
        let limit = vec![compute, memory].time_limit();
        assert_eq!(limit, Some(Duration::from_secs(2)));
    }

    #[test]
    fn vec_time_limit_skips_non_normal_bounds_and_is_none_when_empty() {
        // A non-normal bound is filtered out rather than poisoning the reduction.
        let limit = vec![bound(8, 0.0, 1.0), bound(8, 4.0, 1.0)].time_limit();
        assert_eq!(limit, Some(Duration::from_secs(2)));

        assert_eq!(Vec::<AutotuneBound>::new().time_limit(), None);
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
    fn calculate_bounds_applies_a_threshold_per_resource() {
        let work = Work {
            compute_ops: 8,
            bytes: 16,
        };
        let thresholds = Thresholds {
            compute: 0.5,
            memory: 1.0,
        };
        let key = ThroughputKey {
            mode: ThroughputMode::Memory,
        };

        let bounds = calculate_bounds(
            work,
            thresholds,
            &ThroughputValue::ZERO,
            &ThroughputValue::ZERO,
            &key,
        );

        assert_eq!(bounds[0].resource.amount, 8);
        assert_eq!(bounds[0].threshold, 0.5);
        assert_eq!(bounds[1].resource.amount, 16);
        assert_eq!(bounds[1].threshold, 1.0);
    }

    #[test]
    fn the_default_threshold_is_the_roofline_itself() {
        assert_eq!(Thresholds::default(), Thresholds::uniform(1.0));
        assert_eq!(Thresholds::default().compute, 1.0);
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
