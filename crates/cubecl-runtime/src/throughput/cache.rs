#[cfg(std_io)]
use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

use crate::throughput::{ThroughputKey, ThroughputValue};
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::Mutex;
use cubecl_ir::DeviceIdentity;

/// The namespace segment naming which generation of probes wrote a value.
#[cfg(std_io)]
const GENERATION: &str = "probe-v";

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
    /// Gets or creates the global `ThroughputCache` for one device.
    pub fn get_for_device(runtime: &str, identity: &DeviceIdentity) -> Arc<Mutex<Self>> {
        let name = device_key(runtime, identity);
        let mut cache_map = GLOBAL_CACHE.lock();
        let cache_map = cache_map.get_or_insert_with(HashMap::new);

        #[cfg(std_io)]
        drop_earlier_generations();

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

/// The part, never the device index, which `CUDA_VISIBLE_DEVICES` makes 0 for
/// whichever card was pinned. Two cards this cannot separate are one part, and
/// share a peak.
fn device_key(runtime: &str, identity: &DeviceIdentity) -> String {
    let DeviceIdentity { name, fingerprint } = identity;
    // A namespace is a path, and a runtime names itself `wgpu<spirv>`.
    let segment = |text: &str| text.replace(|c: char| !c.is_ascii_alphanumeric(), "-");

    format!(
        "{}_{}_{}",
        segment(runtime),
        segment(fingerprint),
        segment(name)
    )
}

/// Namespaces this build wrote under an earlier [`PROBE_VERSION`].
///
/// Only this crate version's, and only earlier: another version may belong to a
/// cubecl still in use, and a later generation to a build running right now. A
/// namespace with no generation at all predates them.
#[cfg(std_io)]
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

#[cfg(std_io)]
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

#[cfg(std_io)]
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

    /// Two cards in one machine were served each other's peaks: pinning either
    /// makes it index 0, and the index was all that told them apart.
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

    /// A device key is one segment of a path, so a runtime that names itself
    /// `wgpu<spirv>` must not put brackets or a separator in it.
    #[test]
    fn a_device_key_is_one_path_segment() {
        let key = device_key(
            "wgpu<spirv>",
            &DeviceIdentity {
                name: "Intel(R) Arc(tm) B390 (PTL)".to_string(),
                fingerprint: "spirv_32902_45184".to_string(),
            },
        );

        assert!(
            key.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "{key}"
        );
    }

    /// A build drops what its own earlier probes wrote and nothing else:
    /// another crate version may be in use, and a later probe version is
    /// running right now.
    #[cfg(std_io)]
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
    #[cfg(std_io)]
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
