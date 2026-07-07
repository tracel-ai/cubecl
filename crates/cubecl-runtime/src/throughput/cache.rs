#[cfg(std_io)]
use cubecl_common::cache::{Cache, CacheOption};

#[cfg(not(std_io))]
use hashbrown::HashMap;

use crate::{config::CubeClRuntimeConfig, throughput::ComputeCmmaConfig};
use cubecl_common::config::RuntimeConfig;
use cubecl_ir::ElemType;

use serde;

/// Represents the mode of a throughput computation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ThroughputMode {
    /// Compute direct calculation without special hardware acceleration.
    ComputeDirect,
    /// Compute cmma calculation with CMMA hardware acceleration.
    ComputeCmma(ComputeCmmaConfig),
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
    /// The number of operations performed depending of the mode during the computation.
    pub ops_count: usize,
    /// The duration of the computation.
    pub duration: core::time::Duration,
}

impl ThroughputValue {
    /// Returns the operations per second.
    pub fn ops_per_s(&self) -> f64 {
        self.ops_count as f64 / self.duration.as_secs_f64()
    }

    /// Returns the bytes per second.
    pub fn bytes_per_s(&self, key: &ThroughputKey) -> f64 {
        (self.ops_count * key.dtype.size()) as f64 / self.duration.as_secs_f64()
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

    /// Inserts a new [`ThroughputValue`] into the cache for the given [`ThroughputKey`].
    pub fn insert(&mut self, key: ThroughputKey, value: ThroughputValue) {
        #[cfg(std_io)]
        self.cache
            .insert(key, value)
            .expect("Should be able to insert new throughput value");

        #[cfg(not(std_io))]
        self.cache.insert(key, value);
    }

    /// Returns the [`ThroughputValue`] for the given [`ThroughputKey`], if it exists in the cache.
    pub fn get(&self, key: &ThroughputKey) -> Option<&ThroughputValue> {
        self.cache.get(key)
    }
}
