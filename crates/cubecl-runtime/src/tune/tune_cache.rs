#[cfg(autotune_persistent_cache)]
mod std_imports {
    pub use std::io;
    pub use std::path::Path;
    pub use std::path::PathBuf;
}

use cubecl_common::cache::Cache;
#[cfg(autotune_persistent_cache)]
use std_imports::*;

#[cfg(autotune_persistent_cache)]
use serde::{Deserialize, Serialize};

use super::{AutotuneKey, AutotuneOutcome};
use hashbrown::HashMap;

#[cfg(autotune_persistent_cache)]
/// Return the file path for the persistent cache on disk
/// prefix should be the device id computed at the backend level
pub fn get_persistent_cache_file_path(prefix: &str) -> PathBuf {
    let home_dir = dirs::home_dir().expect("An home directory should exist");
    let path_dir = home_dir.join(".cache").join("cubecl").join("autotune");
    let path = Path::new(&path_dir);
    path.join(format!("{}-autotune-cache.json", prefix))
}

/// In-memory cache entry
#[derive(Debug)]
pub(crate) enum CacheEntry {
    Done {
        checksum_matches: Option<bool>,
        fastest_index: usize,
    },
    Pending,
}

/// Persistent cache key
#[cfg(autotune_persistent_cache)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(crate) struct PersistentCacheKey<K> {
    key: K,
    checksum: String,
}

/// Persistent cache entry
#[cfg(autotune_persistent_cache)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct PersistentCacheValue {
    fastest_index: usize,
    results: Vec<Result<AutotuneOutcome, String>>,
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    in_memory_cache: HashMap<K, CacheEntry>,
    #[cfg(autotune_persistent_cache)]
    persistent_cache: Cache<PersistentCacheKey<K>, PersistentCacheValue>,
}

/// Result of the cache try
#[derive(Debug)]
pub enum TuneCacheResult {
    /// An operation is found.
    Hit {
        /// The index of the fastest operation to execute.
        fastest_index: usize,
    },
    /// The operation might be cached, but we don't know yet whether the checksum is valid.
    Unchecked,
    /// We don't know yet what is fastest, but are waiting for a result to come in.
    Pending,
    /// No operation is found yet.
    Miss,
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new(
        #[cfg_attr(not(autotune_persistent_cache), allow(unused_variables))] name: &str,
        #[cfg_attr(not(autotune_persistent_cache), allow(unused_variables))] device_id: &str,
    ) -> Self {
        #[cfg(autotune_persistent_cache)]
        {
            let mut cache = TuneCache {
                in_memory_cache: HashMap::new(),
                persistent_cache: Cache::new(
                    format!("autotune/{device_id}/{name}"),
                    Default::default(),
                ),
            };
            if let Err(e) = cache.load() {
                log::warn!(
                    "Unable to load autotune cache. Cache will be ignored ({}).",
                    e
                );
            }
            cache
        }

        #[cfg(not(autotune_persistent_cache))]
        {
            TuneCache {
                in_memory_cache: HashMap::new(),
            }
        }
    }

    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        let result = self.in_memory_cache.get(key);

        let Some(val) = result else {
            return TuneCacheResult::Miss;
        };

        match val {
            CacheEntry::Done {
                checksum_matches,
                fastest_index,
            } => {
                if cfg!(autotune_persistent_cache) {
                    match checksum_matches {
                        None => TuneCacheResult::Unchecked,   // Don't know yet.
                        Some(false) => TuneCacheResult::Miss, // Can't use this.
                        Some(true) => TuneCacheResult::Hit {
                            fastest_index: *fastest_index,
                        },
                    }
                } else {
                    let _ = checksum_matches;
                    TuneCacheResult::Hit {
                        fastest_index: *fastest_index,
                    }
                }
            }
            CacheEntry::Pending => TuneCacheResult::Pending,
        }
    }

    #[cfg(autotune_persistent_cache)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) {
        let result = self.in_memory_cache.get_mut(key);
        let Some(val) = result else {
            return;
        };

        if let CacheEntry::Done {
            checksum_matches, ..
        } = val
        {
            if checksum_matches.is_none() {
                let persistent_key = PersistentCacheKey {
                    checksum: checksum.to_string(),
                    key: key.clone(),
                };
                let persistent_entry = self.persistent_cache.get(&persistent_key);
                *checksum_matches = Some(persistent_entry.is_some());
            }
        }
    }

    pub(crate) fn mark_pending(&mut self, key: K) {
        self.in_memory_cache.insert(key, CacheEntry::Pending);
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.in_memory_cache.insert(
            key,
            CacheEntry::Done {
                checksum_matches: Some(true),
                fastest_index,
            },
        );
    }
}

#[cfg(autotune_persistent_cache)]
impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        fastest_index: usize,
        results: Vec<Result<AutotuneOutcome, String>>,
    ) {
        self.persistent_cache.insert(
            PersistentCacheKey { key, checksum },
            PersistentCacheValue {
                fastest_index,
                results,
            },
        );
    }

    /// Load the persistent cache data from disk
    pub(crate) fn load(&mut self) -> Result<(), io::Error> {
        self.persistent_cache.for_each(|key, value| {
            self.in_memory_cache.insert(
                key.key.clone(),
                CacheEntry::Done {
                    checksum_matches: None,
                    fastest_index: value.fastest_index,
                },
            );
        });

        Ok(())
    }
}
