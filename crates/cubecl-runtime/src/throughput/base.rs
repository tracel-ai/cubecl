#[cfg(std_io)]
use cubecl_common::cache::{Cache, CacheOption};

#[cfg(not(std_io))]
use hashbrown::HashMap;

use crate::{client::ComputeClient, config::CubeClRuntimeConfig, runtime::Runtime};
use alloc::boxed::Box;
use alloc::vec::Vec;
use alloc::{format, string::String};
use cubecl_common::{
    config::RuntimeConfig,
    future::block_on,
    profile::{Duration, Instant},
};
use cubecl_ir::ElemType;
use serde;

#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct MatrixSizes {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Represents the mode of a throughput computation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ThroughputMode {
    /// Compute direct calculation without special hardware acceleration.
    ComputeDirect,
    /// Compute cmma calculation with CMMA hardware acceleration.
    ComputeCmma(MatrixSizes),
    /// Memory input reads and output writes.
    Memory,
}

/// Represents a key/configuration used to identify the throughput of a computation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct ThroughputKey {
    /// The mode of the throughput computation.
    pub mode: ThroughputMode,
    /// The data type of the computation.
    pub dtype: ElemType,
}

/// Represents the throughput of a computation, including the number of operations and the duration.
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
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
            ThroughputMode::ComputeDirect | ThroughputMode::ComputeCmma(_) => "OPS",
            ThroughputMode::Memory => "bytes",
        };

        let mut op_s = self.throughput_per_s();
        let suffixes = ["", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"];
        let mut suffix_idx = 0;

        for _ in 0..suffixes.len() - 1 {
            if op_s < 1000.0 {
                break;
            }
            op_s /= 1000.0;
            suffix_idx += 1;
        }

        format!("{op_s:.4} {}{unit}/s", suffixes[suffix_idx])
    }
}

/// Caches the [`ThroughputValue`] for a given [`ThroughputKey`].
///
/// This cache is used to avoid recomputing throughput values for the same key.
/// Stores on disk when std is available, otherwise stores in memory.
pub struct ThroughputCache {
    #[cfg(not(std_io))]
    cache: HashMap<ThroughputKey, ThroughputValue>,
    #[cfg(std_io)]
    cache: Cache<ThroughputKey, ThroughputValue>,
}

impl ThroughputCache {
    /// Creates a new `ThroughputCache` with the given name.
    pub fn new(#[cfg_attr(not(std_io), allow(unused_variables))] name: &str) -> Self {
        #[cfg(not(std_io))]
        {
            let in_memory_cache = HashMap::new();
            ThroughputCache {
                cache: in_memory_cache,
            }
        }

        #[cfg(std_io)]
        {
            let root = CubeClRuntimeConfig::get().throughput.cache.root();
            let options = CacheOption::default().root(root).name("throughput");

            Self {
                cache: Cache::new(name, options),
            }
        }
    }

    fn insert(&mut self, key: ThroughputKey, value: ThroughputValue) {
        #[cfg(std_io)]
        self.cache
            .insert(key, value)
            .expect("Should be able to insert new throughput value");
        #[cfg(not(std_io))]
        {
            self.cache.insert(key, value);
        }
    }

    fn get(&self, key: &ThroughputKey) -> Option<&ThroughputValue> {
        self.cache.get(key)
    }
}

/// Configuration and payload for a benchmarkable compute kernel.
pub struct KernelConfig {
    /// The executable kernel closure to be evaluated.
    pub kernel: Box<dyn Fn()>,
    /// The total number of units (e.g., bytes, operations, or elements) processed.
    pub unit_count: usize,
}

/// Hardware execution parameters for launching a compute kernel.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    /// The number of threads per cube.
    pub cube_dim: usize,
    /// The total number of cubes to dispatch.
    pub cube_count: usize,
    /// The vectorization factor (e.g., 4 for `vec4` operations).
    pub vector_size: usize,
    pub plane_size: usize,
}

/// A trait for running throughput benchmarks on compute kernels.
pub trait ThroughputRunner<R: Runtime> {
    /// Builds a kernel configuration for the given client, dtype, and launch config.
    fn build_kernel(
        client: &ComputeClient<R>,
        key: ThroughputKey,
        config: LaunchConfig,
    ) -> KernelConfig;
}

/// A marker for measuring throughput of compute kernels.
pub struct ThroughputBenchmarker {
    cache: ThroughputCache,
    cache_enabled: bool,
}

impl ThroughputBenchmarker {
    /// Creates a new `ThroughputBenchmarker` with the given cache.
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
        kernel_config: KernelConfig,
    ) -> ThroughputValue {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.get(&key)
        {
            return *cached_value;
        }

        let kernel = kernel_config.kernel;

        let sample_once = || {
            let start = Instant::now();
            kernel();
            let _ = block_on(client.sync());
            start.elapsed()
        };

        self.warmup(sample_once);
        let duration = self.estimate_throughput(sample_once);

        let value = ThroughputValue {
            unit_count: kernel_config.unit_count,
            duration,
        };

        if self.cache_enabled {
            self.cache.insert(key, value);
        }

        value
    }

    fn warmup(&mut self, sample_once: impl Fn() -> Duration) {
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
    }

    fn estimate_throughput(&mut self, sample_once: impl Fn() -> Duration) -> Duration {
        const N_SAMPLES: usize = 20;

        let mut samples: Vec<Duration> = (0..N_SAMPLES).map(|_| sample_once()).collect();
        samples.sort();

        let median = samples[samples.len() / 2];
        samples.retain(|d| *d >= median / 4);

        *samples.iter().min().expect("at least one valid sample")
    }
}
