#[cfg(persistence)]
use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

use crate::device::DevicePart;
use crate::throughput::{ThroughputKey, ThroughputValue};
#[cfg(persistence)]
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::Mutex;

/// The namespace segment naming which generation of probes wrote a value.
#[cfg(persistence)]
const GENERATION: &str = "probe-v";

static GLOBAL_CACHE: Mutex<Option<HashMap<String, Arc<Mutex<ThroughputCache>>>>> = Mutex::new(None);

/// Caches the [`ThroughputValue`] for a given [`ThroughputKey`].
///
/// This cache is used to avoid recomputing throughput values for the same key.
/// Stores on disk when std is available, otherwise stores in memory.
pub struct ThroughputCache {
    #[cfg(not(persistence))]
    cache: HashMap<ThroughputKey, ThroughputValue>,
    #[cfg(persistence)]
    cache: Store<ThroughputKey, ThroughputValue>,
}

impl ThroughputCache {
    /// Gets or creates the global `ThroughputCache` for one part.
    pub fn get_for_device(part: &DevicePart) -> Arc<Mutex<Self>> {
        let name = part.to_string();
        let mut cache_map = GLOBAL_CACHE.lock();
        let cache_map = cache_map.get_or_insert_with(HashMap::new);

        #[cfg(persistence)]
        drop_earlier_generations();

        cache_map
            .entry(name.clone())
            .or_insert_with(|| Arc::new(Mutex::new(Self::new(&name))))
            .clone()
    }

    /// Creates a new `ThroughputCache` with the given name.
    pub fn new(#[cfg_attr(not(persistence), allow(unused_variables))] name: &str) -> Self {
        #[cfg(not(persistence))]
        {
            ThroughputCache {
                cache: HashMap::new(),
            }
        }

        #[cfg(persistence)]
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
        #[cfg(persistence)]
        if let Err(err) = self.cache.insert(key, value) {
            log::warn!("Concurrent throughput measurement, keeping the existing value: {err}");
        }

        #[cfg(not(persistence))]
        self.cache.insert(key, value);
    }

    /// Returns the [`ThroughputValue`] for the given [`ThroughputKey`], if it exists in the cache.
    pub fn get(&self, key: &ThroughputKey) -> Option<&ThroughputValue> {
        self.cache.get(key)
    }
}

/// Namespaces this build wrote under an earlier [`PROBE_VERSION`].
///
/// Only this crate version's, and only earlier: another version may belong to a
/// cubecl still in use, and a later generation to a build running right now. A
/// namespace with no generation at all predates them.
#[cfg(persistence)]
fn is_earlier_generation(candidate: &str, scope: &str, current: u32) -> bool {
    let Some(device) = candidate.strip_prefix(scope) else {
        return false;
    };

    let generation = device
        .strip_prefix(GENERATION)
        .and_then(|device| device.split('/').next())
        .and_then(|generation| generation.parse::<u32>().ok())
        .unwrap_or(0);

    generation < current
}

#[cfg(persistence)]
fn drop_earlier_generations() {
    use core::sync::atomic::{AtomicBool, Ordering};

    static DROPPED: AtomicBool = AtomicBool::new(false);

    if DROPPED.swap(true, Ordering::Relaxed) {
        return;
    }

    let scope = String::from(Namespace::scoped("throughput", "").as_str());
    let current = crate::throughput::PROBE_VERSION;

    for candidate in cubecl_environment::persistence::namespaces() {
        if is_earlier_generation(&candidate, &scope, current) {
            cubecl_environment::persistence::open(&candidate).purge();
        }
    }
}

#[cfg(persistence)]
fn namespace(device_key: &str) -> Namespace {
    Namespace::scoped(
        "throughput",
        format!(
            "{GENERATION}{}/{device_key}",
            crate::throughput::PROBE_VERSION
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// A build drops what its own earlier probes wrote and nothing else:
    /// another crate version may be in use, and a later probe version is
    /// running right now.
    #[cfg(persistence)]
    #[test]
    fn only_this_crate_version_s_earlier_probes_are_dropped() {
        let stale = |ns: &str| is_earlier_generation(ns, "throughput/0.11.0/", 2);

        assert!(stale("throughput/0.11.0/probe-v1/cuda_ptx_sm75_part"));
        assert!(stale("throughput/0.11.0/cuda_dev0"));

        assert!(!stale("throughput/0.11.0/probe-v2/cuda_ptx_sm75_part"));
        assert!(!stale("throughput/0.11.0/probe-v3/cuda_ptx_sm75_part"));
        assert!(!stale("throughput/0.10.0/probe-v1/cuda_ptx_sm75_part"));
        assert!(!stale("autotune/0.11.0/device-0-0-cpu/matmul"));
    }

    /// The crate version in the namespace does not move when a probe changes
    /// what it measures inside a release, so the probes carry their own.
    #[cfg(persistence)]
    #[test]
    fn the_namespace_carries_the_probe_version() {
        let generation = namespace("").as_str().to_string();

        assert!(
            namespace("cuda_ptx_sm75_part")
                .as_str()
                .starts_with(&format!("{generation}/"))
        );
    }
}
