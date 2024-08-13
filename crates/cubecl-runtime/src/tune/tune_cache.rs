#[cfg(autotune_persistent_cache)]
mod std_imports {
    pub use std::fs;
    pub use std::fs::File;
    pub use std::io;
    pub use std::path::Path;
    pub use std::path::PathBuf;
}
#[cfg(autotune_persistent_cache)]
use std_imports::*;

#[cfg(autotune_persistent_cache)]
use serde::{Deserialize, Serialize};

use super::AutotuneKey;
use super::AutotuneOperation;
use super::AutotuneOperationSet;
use alloc::boxed::Box;
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
pub(crate) struct InMemoryCacheEntry {
    #[cfg(autotune_persistent_cache)]
    checksum_checked: bool,
    fastest_index: usize,
}

/// Persistent cache entry
#[cfg(autotune_persistent_cache)]
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PersistentCacheEntry {
    checksum: String,
    fastest_index: usize,
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    in_memory_cache: HashMap<K, InMemoryCacheEntry>,
    #[cfg(autotune_persistent_cache)]
    persistent_cache: HashMap<K, PersistentCacheEntry>,
    #[cfg(autotune_persistent_cache)]
    device_id: String,
    #[cfg(autotune_persistent_cache)]
    name: String,
}

/// Result of the cache try
pub enum TuneCacheResult<K> {
    /// An operation is found and given
    Hit(Box<dyn AutotuneOperation>),
    /// No operation is found and the set is given back for ownership
    Miss(Box<dyn AutotuneOperationSet<K>>),
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
                persistent_cache: HashMap::new(),
                device_id: device_id.to_string(),
                name: name.to_string(),
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

    pub(crate) fn find_fastest(&self, key: &K) -> Option<usize> {
        let val = self.in_memory_cache.get(key)?;

        #[cfg(autotune_persistent_cache)]
        if val.checksum_checked {
            Some(val.fastest_index)
        } else {
            None
        }

        #[cfg(not(autotune_persistent_cache))]
        Some(val.fastest_index)
    }

    pub(crate) fn try_cache(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K>>,
    ) -> TuneCacheResult<K> {
        let key = autotune_operation_set.key();
        let result = self.in_memory_cache.get_mut(&key);

        #[cfg(autotune_persistent_cache)]
        {
            if let Some(InMemoryCacheEntry {
                checksum_checked,
                fastest_index,
            }) = result
            {
                if !*checksum_checked {
                    let checksum = autotune_operation_set.compute_checksum();
                    let persistent_entry = self
                        .persistent_cache
                        .get(&key)
                        .expect("Both caches should be in sync");
                    if checksum != persistent_entry.checksum {
                        return TuneCacheResult::Miss(autotune_operation_set);
                    }
                    *checksum_checked = true;
                }
                return TuneCacheResult::Hit(autotune_operation_set.fastest(*fastest_index));
            }
        }

        #[cfg(not(autotune_persistent_cache))]
        {
            if let Some(InMemoryCacheEntry { fastest_index, .. }) = result {
                return TuneCacheResult::Hit(autotune_operation_set.fastest(*fastest_index));
            }
        }

        TuneCacheResult::Miss(autotune_operation_set)
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.in_memory_cache.insert(
            key,
            InMemoryCacheEntry {
                #[cfg(autotune_persistent_cache)]
                checksum_checked: true,
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
    ) {
        self.persistent_cache.insert(
            key,
            PersistentCacheEntry {
                checksum,
                fastest_index,
            },
        );
    }

    /// Load the persistent cache data from disk
    pub(crate) fn load(&mut self) -> Result<(), io::Error> {
        let file_path = self.get_persistent_cache_file_path();
        // note: reading file from memory is faster than using
        // serde from_reader with a buffered reader
        // see issue:
        // https://github.com/serde-rs/json/issues/160
        match fs::read_to_string(file_path) {
            Ok(data) => {
                let data: Vec<(K, PersistentCacheEntry)> = serde_json::from_str(&data)?;
                for (key, value) in data.into_iter() {
                    self.persistent_cache.insert(key, value);
                }
                Ok(())
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }?;
        for (key, entry) in self.persistent_cache.iter() {
            self.in_memory_cache.insert(
                key.clone(),
                InMemoryCacheEntry {
                    checksum_checked: false,
                    fastest_index: entry.fastest_index,
                },
            );
        }
        Ok(())
    }

    /// Save the persistent cache on disk
    pub(crate) fn save(&self) {
        let file_path = self.get_persistent_cache_file_path();
        if let Some(parent_dir) = file_path.parent() {
            if !parent_dir.exists() {
                fs::create_dir_all(parent_dir).unwrap_or_else(|_| {
                    panic!(
                    "Should be able to create directory '{}' for autotune persistent cache file",
                    parent_dir.to_str().unwrap())
                });
            }
        }
        let file = File::create(file_path.clone()).unwrap_or_else(|_| {
            panic!(
                "Should be able to open autotune persistent cache file '{}'",
                file_path.to_str().unwrap()
            )
        });
        let data = self.persistent_cache.iter().collect::<Vec<_>>();
        serde_json::to_writer_pretty(file, &data)
            .expect("Should be able to write to autotune persistent cache");
    }

    /// Return the file path for the persistent cache on disk
    pub fn get_persistent_cache_file_path(&self) -> PathBuf {
        get_persistent_cache_file_path(&format!("{}/{}", self.name, self.device_id))
    }
}
