#[cfg(std_io)]
use cubecl_common::cache::{Cache, CacheOption};

use crate::{
    config::CubeClRuntimeConfig,
    throughput::{ThroughputKey, ThroughputValue},
};
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use cubecl_common::config::RuntimeConfig;
use hashbrown::HashMap;
use spin::Mutex;

static GLOBAL_CACHE: Mutex<Option<HashMap<String, Arc<Mutex<ThroughputCache>>>>> = Mutex::new(None);

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
    /// Gets or creates a global `ThroughputCache` for the given device name.
    pub fn get_for_device(name: &str) -> Arc<Mutex<Self>> {
        let mut cache_map = GLOBAL_CACHE.lock();
        let cache_map = cache_map.get_or_insert_with(HashMap::new);

        cache_map
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(Self::new(name))))
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
