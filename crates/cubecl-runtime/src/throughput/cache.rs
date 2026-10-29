#[cfg(persistence)]
use cubecl_environment::persistence::{Store, StoreError, StoreOptions};

use crate::device::HardwareIdentity;
#[cfg(persistence)]
use crate::namespace_version::NamespaceVersion;
use crate::throughput::{ThroughputKey, ThroughputValue};
use alloc::string::String;
use alloc::sync::Arc;
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::Mutex;
use cubecl_ir::DeviceProperties;

/// The root of every throughput table's namespace.
pub const ROOT: &str = "throughput";

#[cfg(persistence)]
static PROBES: NamespaceVersion =
    NamespaceVersion::new(ROOT, "probe-v", crate::throughput::PROBE_VERSION);

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
    /// Gets or creates the global `ThroughputCache` for one device.
    pub fn get_for_device(runtime: &str, properties: &DeviceProperties) -> Arc<Mutex<Self>> {
        // A memory ceiling moves with the installed capacity, which the identity leaves out.
        let name = HardwareIdentity::new(runtime, properties)
            .name_with_memory(properties.memory.max_page_size);
        let mut cache_map = GLOBAL_CACHE.lock();
        let cache_map = cache_map.get_or_insert_with(HashMap::new);

        #[cfg(persistence)]
        PROBES.drop_earlier();

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
                cache: Store::new(StoreOptions::new().storage(PROBES.namespace(name))),
            }
        }
    }

    /// Inserts a new [`ThroughputValue`] into the cache for the given [`ThroughputKey`].
    ///
    /// Throughput measurements are nondeterministic, so a concurrent process (or an
    /// earlier run) may have recorded a different value for the same key; the cache
    /// keeps the existing value in that case rather than failing. A value the storage
    /// refuses to write is still kept for the rest of the run.
    pub fn insert(&mut self, key: ThroughputKey, value: ThroughputValue) {
        #[cfg(persistence)]
        if let Err(err) = self.cache.insert(key, value) {
            match err {
                StoreError::Backend { key, error } => log::warn!(
                    "Throughput measurement for key {key:?} could not be stored, it will be measured again next run: {error}"
                ),
                err => log::warn!(
                    "Concurrent throughput measurement, keeping the existing value: {err}"
                ),
            }
        }

        #[cfg(not(persistence))]
        self.cache.insert(key, value);
    }

    /// Returns the [`ThroughputValue`] for the given [`ThroughputKey`], if it exists in the cache.
    pub fn get(&self, key: &ThroughputKey) -> Option<&ThroughputValue> {
        self.cache.get(key)
    }
}
