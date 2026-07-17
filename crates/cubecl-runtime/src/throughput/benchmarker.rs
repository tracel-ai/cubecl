use crate::{
    client::ComputeClient,
    config::CubeClRuntimeConfig,
    runtime::Runtime,
    throughput::{ThroughputCache, ThroughputKey, ThroughputValue},
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use cubecl_common::profile::{Duration, Instant};
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::future::block_on;
use cubecl_environment::sync::Mutex;

type Cache = Arc<Mutex<ThroughputCache>>;

/// Configuration and payload for a benchmarkable compute kernel.
pub struct KernelConfig {
    /// The executable kernel closure to be evaluated.
    pub kernel: Box<dyn Fn(usize)>,
    /// The number of operations processed in one iteration.
    pub ops_count: usize,
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

    /// Measure the maximum compute throughput of the given kernel on the given client.
    /// Warms up the kernel until it plateaus,
    /// then measures the throughput over multiple iterations taking the minimum time per iteration (peak attained).
    pub fn measure<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        key: ThroughputKey,
        kernel_config: KernelConfig,
    ) -> ThroughputValue {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.lock().unwrap().get(&key)
        {
            return *cached_value;
        }

        let kernel = kernel_config.kernel;

        let sample = |iterations: usize| {
            let start = Instant::now();
            kernel(iterations);
            let _ = block_on(client.sync());
            start.elapsed()
        };

        let iterations = self.warmup(sample);
        let duration = self.sample_peak_duration(iterations, sample);

        let value = ThroughputValue {
            ops_count: kernel_config.ops_count,
            duration,
        };

        if self.cache_enabled {
            self.cache.lock().unwrap().insert(key, value);
        }

        value
    }

    /// Warms up the device by running the kernel multiple times
    /// and estimating the number of iterations needed to reach a stable duration.
    fn warmup(&self, sample: impl Fn(usize) -> Duration) -> usize {
        const MAX_WARMUP: usize = 50;
        const PLATEAU_TOL: f64 = 0.03;
        const PATIENCE: usize = 3;
        const TARGET_DURATION_MS: f64 = 20.0;

        let mut best = f64::INFINITY;
        let mut stable = 0;
        let mut iterations = 1;

        for _ in 0..MAX_WARMUP {
            let duration = sample(iterations).as_secs_f64() * 1000.0;
            if duration < TARGET_DURATION_MS {
                let extra_iters = if duration > 1e-6 {
                    let duration_per_iter = duration / iterations as f64;
                    ((TARGET_DURATION_MS - duration) / duration_per_iter).ceil() as usize
                } else {
                    iterations
                };
                iterations += extra_iters.max(1);
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
        const MIN_SAMPLES: usize = 20;
        const MAX_SAMPLES: usize = 200;
        const REL_TOL: f64 = 0.01;
        const PATIENCE: usize = 12;

        let mut best = f64::INFINITY;
        let mut stale = 0;

        for i in 0..MAX_SAMPLES {
            let s = sample_once(iterations).as_secs_f64();
            if s < best * (1.0 - REL_TOL) {
                best = s;
                stale = 0;
            } else {
                best = best.min(s);
                stale += 1;
            }
            if i > MIN_SAMPLES && stale >= PATIENCE {
                break;
            }
        }

        Duration::from_secs_f64(best / iterations as f64)
    }
}
