#[cfg(persistence)]
use cubecl_environment::persistence::{Store, StoreError, StoreOptions};

#[cfg(persistence)]
use crate::config::CubeClRuntimeConfig;
use crate::device::HardwareIdentity;
#[cfg(persistence)]
use crate::namespace_version::NamespaceVersion;
use crate::throughput::{LinkKey, ThroughputError, ThroughputKey, ThroughputValue};
use alloc::sync::Arc;
use alloc::{format, string::String, vec::Vec};
use cubecl_environment::collections::HashMap;
#[cfg(persistence)]
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::sync::Mutex;
use cubecl_ir::DeviceProperties;

/// The root of every throughput table's namespace.
pub const ROOT: &str = "throughput";

#[cfg(persistence)]
static PROBES: NamespaceVersion =
    NamespaceVersion::new(ROOT, "probe-v", crate::throughput::PROBE_VERSION);

type Caches<K> = Mutex<Option<HashMap<String, Arc<Mutex<ThroughputCache<K>>>>>>;

static GLOBAL_CACHE: Caches<ThroughputKey> = Mutex::new(None);

static LINK_CACHE: Caches<LinkKey> = Mutex::new(None);

/// A key of a [`ThroughputCache`]: one stored where the cache persists, hashed where it lives in
/// memory.
#[cfg(persistence)]
pub trait ThroughputCacheKey: cubecl_environment::persistence::StoreKey + core::fmt::Debug {}
#[cfg(persistence)]
impl<K: cubecl_environment::persistence::StoreKey + core::fmt::Debug> ThroughputCacheKey for K {}

/// A key of a [`ThroughputCache`]: one hashed, since this build keeps the cache in memory.
#[cfg(not(persistence))]
pub trait ThroughputCacheKey: core::hash::Hash + Eq + core::fmt::Debug {}
#[cfg(not(persistence))]
impl<K: core::hash::Hash + Eq + core::fmt::Debug> ThroughputCacheKey for K {}

/// Caches what probing a key answered: a [`ThroughputKey`] for one device, a [`LinkKey`] for the
/// devices a link joins.
///
/// A key is probed at most once a process: a measured value is kept, and so is the error of a
/// probe that failed. Values are also stored on disk where the build persists and the
/// configuration leaves the throughput cache on, so a later run reads them back. Errors are kept
/// in memory only, so the next run probes again.
pub struct ThroughputCache<K = ThroughputKey> {
    /// Each measured value, read and written by its key.
    #[cfg(not(persistence))]
    cache: HashMap<K, ThroughputValue>,
    /// Each measured value, read and written by its key.
    #[cfg(persistence)]
    cache: Store<K, ThroughputValue>,
    /// Each error a probe failed with in this process, read and written by its key.
    failures: HashMap<K, ThroughputError>,
}

impl ThroughputCache<LinkKey> {
    /// Gets or creates the global cache for the link between `members`, in order, or `None` where
    /// a member's place cannot be named.
    ///
    /// A link's rate moves with where its cards sit, which their hardware leaves out, so each is
    /// named by its PCI address too. That address is local to the host, and a stored cache can
    /// travel to another, so a stored one is also named for the host.
    pub fn get_for_link(runtime: &str, members: &[&DeviceProperties]) -> Option<Arc<Mutex<Self>>> {
        link_cache(runtime, members)
    }
}

fn link_cache(
    runtime: &str,
    members: &[&DeviceProperties],
) -> Option<Arc<Mutex<ThroughputCache<LinkKey>>>> {
    let places = members
        .iter()
        .map(|properties| HardwareIdentity::name_in_slot(runtime, properties))
        .collect::<Option<Vec<_>>>()?;

    Some(cached(&LINK_CACHE, link_name(&host_id()?, &places)))
}

/// The name of the cache on `host` for the link between cards named `places`, in order.
fn link_name(host: &str, places: &[String]) -> String {
    format!("link_{host}_{}", places.join("__"))
}

impl ThroughputCache {
    /// The process's cache for devices of `runtime` with the hardware `properties` describes.
    pub fn get_for_device(runtime: &str, properties: &DeviceProperties) -> Arc<Mutex<Self>> {
        cached(&GLOBAL_CACHE, device_name(runtime, properties))
    }
}

/// The name of the cache of a device of `runtime` reporting `properties`.
fn device_name(runtime: &str, properties: &DeviceProperties) -> String {
    // A memory ceiling moves with the installed capacity, which the identity leaves out.
    HardwareIdentity::new(runtime, properties)
        .name_with_memory_bytes(properties.memory.max_page_size)
}

fn cached<K: ThroughputCacheKey>(
    caches: &Caches<K>,
    name: String,
) -> Arc<Mutex<ThroughputCache<K>>> {
    let mut cache_map = caches.lock();
    let cache_map = cache_map.get_or_insert_with(HashMap::new);

    #[cfg(persistence)]
    PROBES.drop_earlier();

    cache_map
        .entry(name.clone())
        .or_insert_with(|| Arc::new(Mutex::new(ThroughputCache::new(&name))))
        .clone()
}

/// The host a stored cache was measured on, from the machine id systemd writes.
#[cfg(all(persistence, std_io))]
fn host_id() -> Option<String> {
    let id = std::fs::read_to_string("/etc/machine-id").ok()?;
    let id = id.trim();
    (!id.is_empty()).then(|| id.into())
}

/// No host has a stored cache to name.
#[cfg(all(persistence, not(std_io)))]
fn host_id() -> Option<String> {
    None
}

/// A cache that lives in memory never leaves the host.
#[cfg(not(persistence))]
fn host_id() -> Option<String> {
    Some("local".into())
}

impl<K: ThroughputCacheKey> ThroughputCache<K> {
    /// The cache named `name`: empty, or as an earlier run stored it where the configuration
    /// leaves the throughput cache on.
    pub fn new(#[cfg_attr(not(persistence), allow(unused_variables))] name: &str) -> Self {
        #[cfg(persistence)]
        if !CubeClRuntimeConfig::get().throughput.disable_cache {
            return Self {
                cache: Store::new(StoreOptions::new().storage(PROBES.namespace(name))),
                failures: HashMap::new(),
            };
        }

        Self::in_memory()
    }

    /// An empty cache that lives in memory alone.
    pub fn in_memory() -> Self {
        Self {
            #[cfg(not(persistence))]
            cache: HashMap::new(),
            #[cfg(persistence)]
            cache: Store::new(StoreOptions::new()),
            failures: HashMap::new(),
        }
    }

    /// Keeps what probing `key` answered.
    ///
    /// Throughput measurements are nondeterministic, so a concurrent process (or an
    /// earlier run) may have recorded a different value for the same key; the cache
    /// keeps the existing value in that case rather than failing. A value the storage
    /// refuses to write is still kept for the rest of the run.
    pub fn insert(&mut self, key: K, outcome: Result<ThroughputValue, ThroughputError>) {
        match outcome {
            Ok(value) => self.insert_value(key, value),
            Err(error) => self.insert_failure(key, error),
        }
    }

    /// What probing `key` answered, if it was probed.
    pub fn get(&self, key: &K) -> Option<Result<ThroughputValue, ThroughputError>> {
        match self.cache.get(key) {
            Some(value) => Some(Ok(*value)),
            None => self.failures.get(key).copied().map(Err),
        }
    }
}

impl<K: ThroughputCacheKey> ThroughputCache<K> {
    fn insert_value(&mut self, key: K, value: ThroughputValue) {
        #[cfg(persistence)]
        self.cache.insert(key, value).unwrap_or_else(warn_unstored);

        #[cfg(not(persistence))]
        self.cache.insert(key, value);
    }

    fn insert_failure(&mut self, key: K, error: ThroughputError) {
        self.failures.insert(key, error);
    }
}

/// Logs a value the storage refused, which stays in memory for the rest of the run.
#[cfg(persistence)]
fn warn_unstored<K: core::fmt::Debug>(err: StoreError<K, ThroughputValue>) {
    match err {
        StoreError::Backend { key, error } => log::warn!(
            "Throughput measurement for key {key:?} could not be stored, it will be measured again next run: {error}"
        ),
        err => log::warn!("Concurrent throughput measurement, keeping the existing value: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn a_link_is_named_in_the_order_its_bytes_move() {
        let (a, b) = (
            "cuda_l4_pci0000830".to_string(),
            "cuda_l4_pci0000860".to_string(),
        );

        assert_ne!(
            link_name("host", &[a.clone(), b.clone()]),
            link_name("host", &[b, a])
        );
    }
}
