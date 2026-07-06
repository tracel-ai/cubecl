use alloc::{format, string::String};
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

/// Represents the mode of a throughput computation.
#[derive(serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone, Hash, Debug, Copy)]
pub enum ThroughputMode {
    /// Compute direct calculation without special hardware acceleration.
    ComputeDirect,
    /// Compute cmma calculation with CMMA hardware acceleration.
    ComputeCmma,
    /// Memory input reads and output writes.
    Memory,
}

/// Represents a key/configuration used to identify the throughput of a computation.
#[derive(serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone, Hash, Debug, Copy)]
pub struct ThroughputKey {
    /// The mode of the throughput computation.
    pub mode: ThroughputMode,
    /// The data type of the computation.
    pub dtype: ElemType,
}

/// Represents the throughput of a computation, including the number of operations and the duration.
#[derive(serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone, Copy, Debug)]
pub struct ThroughputValue {
    /// The number of operations performed or bytes moved depending of the mode during the computation.
    pub unit_count: usize,
    /// The duration of the computation.
    pub duration: core::time::Duration,
}

impl ThroughputValue {
    /// Returns the throughput per second.
    pub fn throughput_per_s(&self) -> f64 {
        self.unit_count as f64 / self.duration.as_secs_f64()
    }

    /// Formats the throughput as a human-readable string.
    pub fn format(&self, key: &ThroughputKey) -> String {
        let unit = match key.mode {
            ThroughputMode::ComputeDirect | ThroughputMode::ComputeCmma => "OPS",
            ThroughputMode::Memory => "bytes",
        };
        let mut op_s = self.throughput_per_s();
        let suffixes = ["", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"];
        for suffix in suffixes.iter() {
            if op_s <= 1000.0 {
                return format!("{op_s:.4} {suffix}{unit}/s");
            }
            op_s /= 1000.0;
        }
        format!("{op_s:.2} {unit}/s")
    }
}

///
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
    /// Warms up the kernel until it plateaus,
    /// then measures the throughput over multiple iterations taking the minimum time per iteration (peak attained).
    pub fn measure_throughput<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        key: ThroughputKey,
        unit_count: usize,
        kernel: alloc::boxed::Box<dyn Fn()>,
    ) -> ThroughputValue {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.persistent_cache.get(&key)
        {
            return *cached_value;
        }

        let sample_once = || {
            let start = Instant::now();
            kernel();
            let _ = block_on(client.sync());
            start.elapsed()
        };

        const MAX_WARMUP: usize = 50;
        const PLATEAU_TOL: f64 = 0.03;
        const PATIENCE: usize = 3;

        let mut best = f64::INFINITY;
        let mut stable = 0;
        for _ in 0..MAX_WARMUP {
            let s = sample_once().as_secs_f64();
            if s < best * (1.0 - PLATEAU_TOL) {
                best = s;
                stable = 0;
            } else {
                best = best.min(s);
                stable += 1;
                if stable >= PATIENCE {
                    break;
                }
            }
        }

        const N_SAMPLES: usize = 20;
        let mut samples: Vec<Duration> = (0..N_SAMPLES).map(|_| sample_once()).collect();
        samples.sort();

        let median = samples[samples.len() / 2];
        samples.retain(|d| *d >= median / 4);

        let duration = *samples.iter().min().expect("at least one valid sample");

        let value = ThroughputValue {
            unit_count,
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
