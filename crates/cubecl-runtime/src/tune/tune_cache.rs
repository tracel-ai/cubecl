#[cfg(std_io)]
use super::AutotuneOutcome;
#[cfg(std_io)]
use cubecl_common::cache::Cache;
#[cfg(std_io)]
use cubecl_common::cache::CacheError;
#[cfg(std_io)]
use serde::{Deserialize, Serialize};

use super::AutotuneKey;
use alloc::string::String;
use hashbrown::HashMap;

/// In-memory cache entry
#[derive(Debug)]
pub(crate) enum CacheEntry {
    Done {
        checksum: ChecksumState,
        fastest_index: usize,
    },
    Pending,
}

#[derive(Debug)]
#[allow(dead_code)] // Some variants are not created when the cache isn't saved.
pub(crate) enum ChecksumState {
    Match,
    NoMatch,
    ToBeVerified(String),
}

/// Persistent cache key
#[cfg(std_io)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub(crate) struct PersistentCacheKey<K> {
    key: K,
    checksum: String,
}

/// Persistent cache entry
#[cfg(std_io)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub(crate) struct PersistentCacheValue {
    fastest_index: usize,
    results: Vec<Result<AutotuneOutcome, String>>,
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    in_memory_cache: HashMap<K, CacheEntry>,
    #[cfg(std_io)]
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
        #[cfg_attr(not(std_io), allow(unused_variables))] name: &str,
        #[cfg_attr(not(std_io), allow(unused_variables))] device_id: &str,
    ) -> Self {
        #[cfg(std_io)]
        {
            let root = crate::config::GlobalConfig::get().autotune.cache.root();
            let options = cubecl_common::cache::CacheOption::default();
            let mut cache = TuneCache {
                in_memory_cache: HashMap::new(),
                persistent_cache: Cache::new(
                    format!("{device_id}/{name}"),
                    options.root(root).name("autotune"),
                ),
            };
            cache.load();
            cache
        }

        #[cfg(not(std_io))]
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
                checksum,
                fastest_index,
            } => {
                if cfg!(std_io) {
                    match checksum {
                        ChecksumState::ToBeVerified(..) => TuneCacheResult::Unchecked, // Don't know yet.
                        ChecksumState::NoMatch => TuneCacheResult::Miss, // Can't use this.
                        ChecksumState::Match => TuneCacheResult::Hit {
                            fastest_index: *fastest_index,
                        },
                    }
                } else {
                    // Clippy;
                    let _ = checksum;
                    TuneCacheResult::Hit {
                        fastest_index: *fastest_index,
                    }
                }
            }
            CacheEntry::Pending => TuneCacheResult::Pending,
        }
    }

    #[cfg(std_io)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) {
        let result = self.in_memory_cache.get_mut(key);
        let Some(val) = result else {
            return;
        };

        if let CacheEntry::Done {
            checksum: checksum_state,
            ..
        } = val
        {
            if let ChecksumState::ToBeVerified(checksum_expected) = checksum_state {
                if checksum_expected == checksum {
                    *checksum_state = ChecksumState::Match;
                } else {
                    *checksum_state = ChecksumState::NoMatch;
                }
            }
        }
    }

    #[allow(unused)]
    pub(crate) fn mark_pending(&mut self, key: K) {
        self.in_memory_cache.insert(key, CacheEntry::Pending);
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.in_memory_cache.insert(
            key,
            CacheEntry::Done {
                checksum: ChecksumState::Match,
                fastest_index,
            },
        );
    }
}

#[cfg(std_io)]
impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        fastest_index: usize,
        results: Vec<Result<AutotuneOutcome, String>>,
    ) {
        if let Err(err) = self.persistent_cache.insert(
            PersistentCacheKey { key, checksum },
            PersistentCacheValue {
                fastest_index,
                results,
            },
        ) {
            match err {
                CacheError::DuplicatedKey {
                    key,
                    value_previous,
                    value_updated,
                } => {
                    log::warn!(
                        "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
                    );
                }
                CacheError::KeyOutOfSync { .. } => {
                    // This is OK.
                }
            }
        }
        // .expect();
    }

    /// Load the persistent cache data from disk
    pub(crate) fn load(&mut self) {
        log::info!("Load autotune cache ...");
        let mut loaded = 0;
        self.persistent_cache.for_each(|key, value| {
            loaded += 1;
            self.in_memory_cache.insert(
                key.key.clone(),
                CacheEntry::Done {
                    checksum: ChecksumState::ToBeVerified(key.checksum.clone()),
                    fastest_index: value.fastest_index,
                },
            );
        });
        log::info!("Loaded {loaded} autotune cached entries");
    }
}
