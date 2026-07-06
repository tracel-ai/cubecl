use std::time::{Duration, Instant};
use std::vec::Vec;

use cubecl_common::{
    cache::{Cache, CacheOption},
    config::RuntimeConfig,
    future::block_on,
};
use cubecl_ir::ElemType;
use serde;

use crate::{client::ComputeClient, config::CubeClRuntimeConfig, runtime::Runtime};

#[derive(serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub enum ThroughputMode {
    Direct,
    TensorCore,
}

#[derive(serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub struct ThroughputKey {
    pub mode: ThroughputMode,
    pub dtype: ElemType,
}

#[derive(serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone, Copy, Debug)]
pub struct ThroughputValue {
    pub work_bytes: usize,
    pub duration: core::time::Duration,
}

/// One GB in bytes.
const ONE_GB: f64 = 1e9;

impl ThroughputValue {
    pub fn throughput_gb_s(&self) -> f64 {
        self.work_bytes as f64 / self.duration.as_secs_f64() / ONE_GB
    }
}

pub struct ThroughputCache {
    persistent_cache: Cache<ThroughputKey, ThroughputValue>,
}

impl ThroughputCache {
    pub fn new(name: &str) -> Self {
        let root = CubeClRuntimeConfig::get().throughput.cache.root();
        let options = CacheOption::default().root(root).name("throughput");

        Self {
            persistent_cache: Cache::new(name, options),
        }
    }
}

pub struct ThroughputBenchmarker {
    cache: ThroughputCache,
    cache_enabled: bool,
}

impl ThroughputBenchmarker {
    pub fn new(cache: ThroughputCache) -> Self {
        let cache_enabled = !CubeClRuntimeConfig::get().throughput.disable_cache;
        Self {
            cache,
            cache_enabled,
        }
    }

    /// Measure the maximum compute throughput of the given kernel on the given client.
    pub fn measure_throughput<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        key: ThroughputKey,
        work_bytes: usize,
        kernel: impl Fn() + Send + Sync,
    ) -> ThroughputValue {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.persistent_cache.get(&key)
        {
            return *cached_value;
        }

        // Take one wall-clock sample: enqueue the kernel and block until the GPU
        // has actually finished. Because `sync` waits for real completion, the
        // elapsed time reflects true device execution and can never collapse to
        // the near-zero readings the device-timestamp profiler sometimes returns.
        let sample_once = || {
            let start = Instant::now();
            kernel();
            let _ = block_on(client.sync());
            start.elapsed()
        };

        // --- Warmup to plateau -------------------------------------------------
        // The GPU ramps its clock (DVFS) under sustained load: the first launches
        // run in a low power state and get progressively faster until the clock
        // saturates. Keep warming until the samples stop getting meaningfully
        // faster (no improvement over the best-so-far for `PATIENCE` in a row),
        // so we measure boosted-clock time rather than the cold ramp.
        const MAX_WARMUP: usize = 50;
        const PLATEAU_TOL: f64 = 0.03; // treat <3% faster as "not improving"
        const PATIENCE: usize = 3;

        let mut best = f64::INFINITY;
        let mut stable = 0;
        for _ in 0..MAX_WARMUP {
            let s = sample_once().as_secs_f64();
            if s < best * (1.0 - PLATEAU_TOL) {
                best = s; // meaningful speedup => clock still ramping
                stable = 0;
            } else {
                best = best.min(s);
                stable += 1;
                if stable >= PATIENCE {
                    break; // clock has plateaued
                }
            }
        }

        // --- Measure -----------------------------------------------------------
        const N_SAMPLES: usize = 20;
        let mut samples: Vec<Duration> = (0..N_SAMPLES).map(|_| sample_once()).collect();
        samples.sort();

        let median = samples[samples.len() / 2];
        samples.retain(|d| *d >= median / 4);

        // --- Reduce ------------------------------------------------------------
        // Peak: the shortest valid sample => highest clock => maximum throughput.
        let duration = *samples.iter().min().expect("at least one valid sample");

        let value = ThroughputValue {
            work_bytes,
            duration,
        };

        if self.cache_enabled {
            self.cache
                .persistent_cache
                .insert(key, value)
                .expect("Should be able to insert new throughput value");
        }

        value
    }
}
