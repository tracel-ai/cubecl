use core::time::Duration;

use alloc::vec::Vec;

use crate::throughput::{ThroughputKey, ThroughputValue};
use crate::tune::TuneInputs;

/// A set of [`AutotuneBound`]s for a given key and reference inputs, with a launch overhead.
#[derive(Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct Bounds {
    /// The bounds for autotuning.
    pub bounds: Vec<AutotuneBound>,
    /// The launch overhead for autotuning.
    pub launch_overhead: Duration,
}

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

/// A bound for autotuning a throughput kernel, specifying the key, threshold, and number of operations.
#[derive(Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct AutotuneBound {
    /// Peak throughput of the reference kernel, in ops (or bytes) per second.
    pub throughput: f64,
    /// The threshold for this bound, over which the kernel will be considered accurate.
    pub threshold: f32,
    /// The number of operations the kernel will run.
    pub ops_count: usize,
}

/// Work required by a problem, specified in minimum compute operations and byte transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct Work {
    /// Compute operations required.
    pub compute_ops: usize,
    /// Memory bytes transferred (reads and writes).
    pub bytes: usize,
}

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
            ops_count: work.compute_ops,
            throughput: compute_throughput.ops_per_s(),
            threshold: thresholds.compute,
        },
        AutotuneBound {
            ops_count: work.bytes,
            throughput: memory_throughput.bytes_per_s(memory_key),
            threshold: thresholds.memory,
        },
    ]
}

impl TimeBound for AutotuneBound {
    fn time_limit(&self) -> Option<Duration> {
        if self.throughput.is_normal() && self.threshold.is_normal() {
            Some(Duration::from_secs_f64(
                (self.ops_count as f64 / self.throughput) / self.threshold as f64,
            ))
        } else {
            None
        }
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
            throughput,
            threshold,
            ops_count,
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

        assert_eq!(bounds[0].ops_count, 8);
        assert_eq!(bounds[0].threshold, 0.5);
        assert_eq!(bounds[1].ops_count, 16);
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
