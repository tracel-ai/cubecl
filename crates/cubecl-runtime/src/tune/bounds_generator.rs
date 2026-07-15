use core::time::Duration;

use alloc::vec::Vec;

use crate::throughput::{ThroughputKey, ThroughputValue};
use crate::tune::TuneInputs;

/// A set of [`AutotuneBound`]s for a given key and reference inputs, with a launch overhead.
#[derive(Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct Bounds<B: TimeBound> {
    /// The bounds for autotuning.
    pub bounds: Vec<B>,
    /// The launch overhead for autotuning.
    pub launch_overhead: Duration,
}

/// Produces a set of [`AutotuneBound`]s for a given key and reference inputs.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid bounds generator",
    label = "invalid bounds generator"
)]
pub trait BoundsGenerator<K, I: TuneInputs, B: TimeBound>: Send + Sync + 'static {
    /// Generate a set of bounds for a given key and reference inputs.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> Bounds<B>;
}

/// `Fn(&K, &A) -> Bounds<B>` acts as a [`BoundsGenerator`] when `A` is an owned type. For
/// multi-input kernels, `A` is a tuple that the closure destructures internally.
impl<K, Func, A, B: TimeBound> BoundsGenerator<K, A, B> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> Bounds<B>,
{
    #[inline]
    fn generate<'a>(&self, key: &K, inputs: &<A as TuneInputs>::At<'a>) -> Bounds<B> {
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

/// Standardizes the creation of compute and memory [`AutotuneBound`]s.
pub fn calculate_bounds(
    compute_throughput: &ThroughputValue,
    compute_ops: usize,
    compute_threshold: f32,
    memory_throughput: &ThroughputValue,
    memory_key: &ThroughputKey,
    memory_bytes: usize,
    memory_threshold: f32,
) -> Vec<AutotuneBound> {
    alloc::vec![
        AutotuneBound {
            ops_count: compute_ops,
            throughput: compute_throughput.ops_per_s(),
            threshold: compute_threshold,
        },
        AutotuneBound {
            ops_count: memory_bytes,
            throughput: memory_throughput.bytes_per_s(memory_key),
            threshold: memory_threshold,
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

impl<B: TimeBound> TimeBound for Bounds<B> {
    fn time_limit(&self) -> Option<Duration> {
        self.bounds
            .time_limit()
            .map(|limit| limit + self.launch_overhead)
    }
}

#[cfg(test)]
mod tests {
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
    fn bounds_time_limit_is_none_without_usable_bounds() {
        // No usable bound means no limit at all — the launch overhead is not a limit on
        // its own, so the short-circuit stays disabled.
        let bounds = Bounds::<AutotuneBound> {
            bounds: vec![],
            launch_overhead: Duration::from_millis(500),
        };
        assert_eq!(bounds.time_limit(), None);
    }
}
