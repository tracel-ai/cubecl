use crate::{
    config::CubeClRuntimeConfig,
    throughput::{ThroughputCache, ThroughputKey, ThroughputValue},
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use cubecl_common::profile::Duration;
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::sync::Mutex;

type Cache = Arc<Mutex<ThroughputCache>>;

/// Configuration and payload for a benchmarkable compute kernel.
pub struct KernelConfig {
    /// A closure that executes the kernel for the given number of iterations and returns the duration.
    pub sample: Box<dyn Fn(usize) -> Duration>,
    /// The number of operations processed in one iteration.
    pub ops_count: usize,
    /// Iterations a launch must carry however quickly they turn out to run.
    ///
    /// A duration target cannot express this. What makes a memory probe cold is
    /// that its window walks a whole pool before returning to bytes it already
    /// read, and the shorter the window the more iterations that same walk
    /// takes.
    pub min_iterations: usize,
}

/// A marker for measuring throughput of compute kernels.
pub struct ThroughputBenchmarker {
    cache: Cache,
    cache_enabled: bool,
}

impl ThroughputBenchmarker {
    /// Creates a new `ThroughputBenchmarker` with the given cache.
    pub fn new(cache: Cache) -> Self {
        let cache_enabled = !CubeClRuntimeConfig::get().throughput.disable_cache;
        Self {
            cache,
            cache_enabled,
        }
    }

    /// Measure the maximum compute throughput of the given kernel.
    /// Warms up the kernel until it plateaus,
    /// then measures the throughput over multiple iterations taking the minimum time per iteration (peak attained).
    pub fn measure(&mut self, key: ThroughputKey, kernel_config: KernelConfig) -> ThroughputValue {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.lock().get(&key)
        {
            return *cached_value;
        }

        let sample = kernel_config.sample;

        let iterations = self.warmup(kernel_config.min_iterations, &sample);
        let duration = self.sample_peak_duration(iterations, &sample);

        let value = ThroughputValue {
            ops_count: kernel_config.ops_count,
            duration,
        };

        if self.cache_enabled {
            self.cache.lock().insert(key, value);
        }

        value
    }

    /// Warms up the device by running the kernel multiple times
    /// and estimating the number of iterations needed to reach a stable duration.
    ///
    /// Never returns fewer than `min_iterations`, which the kernel needs to be
    /// measuring what it claims rather than merely to be timed accurately.
    fn warmup(&self, min_iterations: usize, sample: impl Fn(usize) -> Duration) -> usize {
        const MAX_WARMUP: usize = 50;
        // A guard on the branch below that doubles blind when the timer reads
        // zero, not a budget: what a launch costs is the duration target, and
        // capping the count instead leaves a cheap pass measuring the fixed
        // cost of the launch around it. A probe holding its working set still
        // has no walk to make it expensive, so it needs the whole range.
        const MAX_ITERATIONS: usize = 1 << 24;
        const PLATEAU_TOL: f64 = 0.03;
        const PATIENCE: usize = 3;
        const TARGET_DURATION_MS: f64 = 20.0;

        let mut best = f64::INFINITY;
        let mut stable = 0;
        let mut iterations = min_iterations.max(1);
        let ceiling = MAX_ITERATIONS.max(iterations);

        for _ in 0..MAX_WARMUP {
            let duration = sample(iterations).as_secs_f64() * 1000.0;
            if duration < TARGET_DURATION_MS {
                let extra_iters = if duration > 1e-6 {
                    let duration_per_iter = duration / iterations as f64;
                    ((TARGET_DURATION_MS - duration) / duration_per_iter).ceil() as usize
                } else {
                    iterations
                };
                if iterations == ceiling {
                    break;
                }
                iterations = (iterations + extra_iters.max(1)).min(ceiling);
                best = f64::INFINITY;
                stable = 0;
                continue;
            }

            let duration_per_iter = duration / iterations as f64;
            if duration_per_iter < best * (1.0 - PLATEAU_TOL) {
                best = duration_per_iter;
                stable = 0;
            } else {
                best = best.min(duration_per_iter);
                stable += 1;
                if stable >= PATIENCE {
                    break;
                }
            }
        }

        iterations
    }

    /// Sample the peak throughput of the kernel by running it multiple times
    /// and measuring the duration of each iteration.
    fn sample_peak_duration(
        &self,
        iterations: usize,
        sample_once: impl Fn(usize) -> Duration,
    ) -> Duration {
        debug_assert!(
            iterations > 0,
            "iterations must be positive to avoid division by zero"
        );

        const MIN_SAMPLES: usize = 20;
        const MAX_SAMPLES: usize = 200;
        const REL_TOL: f64 = 0.01;
        const PATIENCE: usize = 12;
        // Counting samples prices them all the same, and they are not: a probe
        // whose launch fills the duration target costs forty times one whose
        // pass is a few microseconds. Spending a fixed time instead leaves the
        // cheap probes their long survey and stops the dear ones once the
        // minimum has had a fair number of chances to fall.
        const SAMPLE_BUDGET: Duration = Duration::from_millis(200);

        let mut best = f64::INFINITY;
        let mut stale = 0;
        let mut spent = Duration::ZERO;

        for i in 0..MAX_SAMPLES {
            let sample = sample_once(iterations);
            spent += sample;

            let s = sample.as_secs_f64();
            if s < best * (1.0 - REL_TOL) {
                best = s;
                stale = 0;
            } else {
                best = best.min(s);
                stale += 1;
            }
            if (i > MIN_SAMPLES && stale >= PATIENCE) || spent >= SAMPLE_BUDGET {
                break;
            }
        }

        Duration::from_secs_f64(best / iterations as f64)
    }
}
