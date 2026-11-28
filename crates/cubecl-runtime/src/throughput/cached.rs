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

    /// What probing `key` already answered, a value or an error, if it was probed.
    pub fn cached(&self, key: K) -> Option<Result<ThroughputValue, ThroughputError>> {
        self.cache.lock().get(&key)
    }

    /// The value for `key`, probed by `probe` unless the cache holds what an earlier probe
    /// answered.
    ///
    /// # Guarantees
    ///
    /// Once a probe answered for `key`, a value or an error, no later call through the same
    /// cache probes it again in this process.
    ///
    /// # Errors
    ///
    /// Whatever `probe` reports, now or when it first ran for `key`.
    pub fn measure(
        &mut self,
        key: K,
        probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError>,
    ) -> Result<ThroughputValue, ThroughputError> {
        if let Some(outcome) = self.cached(key) {
            return outcome;
        }
        let outcome = probe();
        self.cache.lock().insert(key, outcome);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::throughput::ThroughputMode;
    use core::cell::Cell;

    const LAUNCH: ThroughputKey = ThroughputKey {
        mode: ThroughputMode::Launch,
    };

    fn throughputs() -> CachedThroughput {
        CachedThroughput::new(Arc::new(Mutex::new(ThroughputCache::in_memory())))
    }

    #[test]
    fn a_failed_probe_is_not_run_again_for_its_key() {
        let mut throughputs = throughputs();
        let probes = Cell::new(0);
        let failing = || {
            probes.set(probes.get() + 1);
            Err(ThroughputError::NoTiming)
        };

        assert_eq!(
            throughputs.measure(LAUNCH, failing),
            Err(ThroughputError::NoTiming)
        );
        assert_eq!(
            throughputs.measure(LAUNCH, failing),
            Err(ThroughputError::NoTiming)
        );
        assert_eq!(probes.get(), 1);
    }

    #[test]
    fn a_measured_value_is_not_probed_again_for_its_key() {
        let mut throughputs = throughputs();
        let probes = Cell::new(0);
        let measuring = || {
            probes.set(probes.get() + 1);
            Ok(ThroughputValue::ZERO)
        };

        throughputs.measure(LAUNCH, measuring).expect("a value");
        throughputs.measure(LAUNCH, measuring).expect("a value");

        assert_eq!(probes.get(), 1);
    }
}
