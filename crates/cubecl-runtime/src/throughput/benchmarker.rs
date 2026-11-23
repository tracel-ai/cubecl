use crate::{
    config::CubeClRuntimeConfig,
    throughput::{
        ThroughputCache, ThroughputCacheKey, ThroughputError, ThroughputKey, ThroughputValue,
    },
};
use alloc::sync::Arc;
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::sync::Mutex;

/// A marker for measuring throughput of compute kernels.
pub struct ThroughputBenchmarker<K = ThroughputKey> {
    cache: Arc<Mutex<ThroughputCache<K>>>,
    cache_enabled: bool,
}

impl<K: ThroughputCacheKey + Copy> ThroughputBenchmarker<K> {
    /// Creates a new `ThroughputBenchmarker` with the given cache.
    pub fn new(cache: Arc<Mutex<ThroughputCache<K>>>) -> Self {
        let cache_enabled = !CubeClRuntimeConfig::get().throughput.disable_cache;
        Self {
            cache,
            cache_enabled,
        }
    }

    /// The value already measured for `key`, where the cache is on and holds one.
    pub fn cached(&self, key: K) -> Option<ThroughputValue> {
        if !self.cache_enabled {
            return None;
        }

        self.cache.lock().get(&key).copied()
    }

    /// The value for `key`, measured by `probe` unless the cache holds one.
    ///
    /// # Errors
    ///
    /// Whatever `probe` reports. Only a measurement is cached.
    pub fn measure(
        &mut self,
        key: K,
        probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError>,
    ) -> Result<ThroughputValue, ThroughputError> {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.lock().get(&key)
        {
            return Ok(*cached_value);
        }

        let value = probe()?;

        if self.cache_enabled {
            self.cache.lock().insert(key, value);
        }

        Ok(value)
    }
}
