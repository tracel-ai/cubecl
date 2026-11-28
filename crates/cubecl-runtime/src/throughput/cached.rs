use crate::{
    config::CubeClRuntimeConfig,
    throughput::{
        ThroughputCache, ThroughputCacheKey, ThroughputError, ThroughputKey, ThroughputValue,
    },
};
use alloc::sync::Arc;
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::sync::Mutex;

/// The throughput measured for a device or a link, read from its [`ThroughputCache`] unless the
/// configuration turns the cache off, and measured where the cache holds nothing.
pub struct CachedThroughput<K = ThroughputKey> {
    cache: Arc<Mutex<ThroughputCache<K>>>,
    enabled: bool,
}

impl<K: ThroughputCacheKey + Copy> CachedThroughput<K> {
    /// The throughput `cache` holds, read through as the global configuration says.
    pub fn new(cache: Arc<Mutex<ThroughputCache<K>>>) -> Self {
        Self {
            cache,
            enabled: !CubeClRuntimeConfig::get().throughput.disable_cache,
        }
    }

    /// The value already measured for `key`, where the cache is on and holds one.
    pub fn cached(&self, key: K) -> Option<ThroughputValue> {
        cached_value(&self.cache, self.enabled, key)
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
        measure_into(&self.cache, self.enabled, key, probe)
    }
}

/// What `cache` holds for `key`, or `None` when it holds nothing or is not `enabled`.
fn cached_value<K: ThroughputCacheKey>(
    cache: &Mutex<ThroughputCache<K>>,
    enabled: bool,
    key: K,
) -> Option<ThroughputValue> {
    if !enabled {
        return None;
    }
    cache.lock().get(&key).copied()
}

/// What `cache` holds for `key`, or what `probe` measures, kept in `cache` when it is `enabled`.
fn measure_into<K: ThroughputCacheKey + Copy>(
    cache: &Mutex<ThroughputCache<K>>,
    enabled: bool,
    key: K,
    probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError>,
) -> Result<ThroughputValue, ThroughputError> {
    if let Some(value) = cached_value(cache, enabled, key) {
        return Ok(value);
    }
    let value = probe()?;
    if enabled {
        cache.lock().insert(key, value);
    }
    Ok(value)
}
