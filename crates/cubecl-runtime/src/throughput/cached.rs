use crate::throughput::{
    ThroughputCache, ThroughputCacheKey, ThroughputError, ThroughputKey, ThroughputValue,
};
use alloc::sync::Arc;
use cubecl_environment::sync::Mutex;

/// The throughput measured for a device or a link, read from its [`ThroughputCache`], and probed
/// where the cache holds nothing for its key.
pub struct CachedThroughput<K = ThroughputKey> {
    cache: Arc<Mutex<ThroughputCache<K>>>,
}

impl<K: ThroughputCacheKey + Copy> CachedThroughput<K> {
    /// The throughput `cache` holds.
    pub fn new(cache: Arc<Mutex<ThroughputCache<K>>>) -> Self {
        Self { cache }
    }

    /// The value already measured for `key`, where the cache holds one.
    pub fn cached(&self, key: K) -> Option<ThroughputValue> {
        self.cache.lock().get(&key)
    }

    /// The value for `key`, measured by `probe` unless the cache holds one.
    ///
    /// # Assumptions
    ///
    /// - The caller serializes the calls that may probe one key, so a key is probed once.
    ///
    /// # Errors
    ///
    /// Whatever `probe` reports. Only a measurement is cached.
    pub fn measure(
        &mut self,
        key: K,
        probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError>,
    ) -> Result<ThroughputValue, ThroughputError> {
        measure_into(&self.cache, key, probe)
    }
}

/// What `cache` holds for `key`, or what `probe` measures, kept in `cache`.
fn measure_into<K: ThroughputCacheKey + Copy>(
    cache: &Mutex<ThroughputCache<K>>,
    key: K,
    probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError>,
) -> Result<ThroughputValue, ThroughputError> {
    if let Some(value) = cache.lock().get(&key) {
        return Ok(value);
    }
    let value = probe()?;
    cache.lock().insert(key, value);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::throughput::{ThroughputMode, cache::tests::in_memory};
    use core::cell::Cell;

    const LAUNCH: ThroughputKey = ThroughputKey {
        mode: ThroughputMode::Launch,
    };

    fn throughputs() -> CachedThroughput {
        CachedThroughput::new(Arc::new(Mutex::new(in_memory())))
    }

    #[test]
    fn a_measured_value_is_not_probed_again_for_its_key() {
        let mut throughputs = throughputs();
        let probes = Cell::new(0);
        let measuring = || {
            probes.set(probes.get() + 1);
            Ok(ThroughputValue::ZERO)
        };

        throughputs
            .measure(LAUNCH, measuring)
            .expect("a probe that measures should answer its value");
        throughputs
            .measure(LAUNCH, measuring)
            .expect("a cached value should answer");

        assert_eq!(probes.get(), 1);
    }

    #[test]
    fn a_failed_probe_runs_again() {
        let mut throughputs = throughputs();
        let probes = Cell::new(0);
        let failing = || {
            probes.set(probes.get() + 1);
            Err(ThroughputError::Allocation)
        };

        let _ = throughputs.measure(LAUNCH, failing);
        let _ = throughputs.measure(LAUNCH, failing);

        assert_eq!(probes.get(), 2);
    }
}
