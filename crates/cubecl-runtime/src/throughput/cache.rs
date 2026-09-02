#[cfg(std_io)]
use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

use crate::throughput::{ThroughputKey, ThroughputValue};
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::Mutex;
use cubecl_ir::DeviceIdentity;

static GLOBAL_CACHE: Mutex<Option<HashMap<String, Arc<Mutex<ThroughputCache>>>>> = Mutex::new(None);

/// Caches the [`ThroughputValue`] for a given [`ThroughputKey`].
///
/// This cache is used to avoid recomputing throughput values for the same key.
/// Stores on disk when std is available, otherwise stores in memory.
pub struct ThroughputCache {
    #[cfg(not(std_io))]
    cache: HashMap<ThroughputKey, ThroughputValue>,
    #[cfg(std_io)]
    cache: Store<ThroughputKey, ThroughputValue>,
}

impl ThroughputCache {
    /// Gets or creates the global `ThroughputCache` holding what `runtime`
    /// measured on the device it reports as `identity`.
    pub fn get_for_device(runtime: &str, identity: &DeviceIdentity) -> Arc<Mutex<Self>> {
        let name = device_key(runtime, identity);
        let mut cache_map = GLOBAL_CACHE.lock();
        let cache_map = cache_map.get_or_insert_with(HashMap::new);

        cache_map
            .entry(name.clone())
            .or_insert_with(|| Arc::new(Mutex::new(Self::new(&name))))
            .clone()
    }

    /// Creates a new `ThroughputCache` with the given name.
    pub fn new(#[cfg_attr(not(std_io), allow(unused_variables))] name: &str) -> Self {
        #[cfg(not(std_io))]
        {
            ThroughputCache {
                cache: HashMap::new(),
            }
        }

        #[cfg(std_io)]
        {
            Self {
                cache: Store::new(StoreOptions::new().storage(namespace(name))),
            }
        }
    }

    /// Inserts a new [`ThroughputValue`] into the cache for the given [`ThroughputKey`].
    ///
    /// Throughput measurements are nondeterministic, so a concurrent process (or an
    /// earlier run) may have recorded a different value for the same key; the cache
    /// keeps the existing value in that case rather than failing.
    pub fn insert(&mut self, key: ThroughputKey, value: ThroughputValue) {
        #[cfg(std_io)]
        if let Err(err) = self.cache.insert(key, value) {
            log::warn!("Concurrent throughput measurement, keeping the existing value: {err}");
        }

        #[cfg(not(std_io))]
        self.cache.insert(key, value);
    }

    /// Returns the [`ThroughputValue`] for the given [`ThroughputKey`], if it exists in the cache.
    pub fn get(&self, key: &ThroughputKey) -> Option<&ThroughputValue> {
        self.cache.get(key)
    }
}

/// What separates one device's measurements from another's: the part, never the
/// device index, which `CUDA_VISIBLE_DEVICES` makes 0 for whichever card was
/// pinned. Two cards this cannot tell apart are the same part, and share a peak.
fn device_key(runtime: &str, identity: &DeviceIdentity) -> String {
    let DeviceIdentity { name, fingerprint } = identity;
    let part = name.replace(|c: char| !c.is_ascii_alphanumeric(), "-");

    format!("{runtime}_{fingerprint}_{part}")
}

#[cfg(std_io)]
fn namespace(device_key: &str) -> Namespace {
    Namespace::scoped(
        "throughput",
        format!("probe-v{}/{device_key}", crate::throughput::PROBE_VERSION),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// Two cards in one machine were served each other's peaks: pinning either
    /// one makes it index 0, and the index was all that told them apart.
    #[test]
    fn two_parts_of_one_architecture_do_not_share_an_entry() {
        let turing = |name: &str| DeviceIdentity {
            name: name.to_string(),
            fingerprint: "ptx_sm75".to_string(),
        };

        assert_ne!(
            device_key("cuda", &turing("NVIDIA GeForce RTX 2060")),
            device_key("cuda", &turing("NVIDIA GeForce GTX 1660 SUPER")),
        );
    }

    /// The crate version in the namespace does not move when a probe changes
    /// what it measures inside a release, so the probes carry their own.
    #[cfg(std_io)]
    #[test]
    fn the_namespace_carries_the_probe_version() {
        let expected = format!("/probe-v{}/", crate::throughput::PROBE_VERSION);

        assert!(namespace("cuda_ptx_sm75_part").as_str().contains(&expected));
    }
}
