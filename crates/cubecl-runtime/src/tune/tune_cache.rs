#[cfg(autotune_persistence)]
use alloc::vec::Vec;

#[cfg(autotune_persistence)]
use cubecl_environment::persistence::Cache;
#[cfg(autotune_persistence)]
use cubecl_environment::persistence::CacheError;
#[cfg(autotune_persistence)]
use serde::{Deserialize, Serialize};

use super::{AutotuneError, AutotuneKey, AutotuneOutcome};
use alloc::string::String;
use cubecl_environment::collections::HashMap;

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
#[cfg(autotune_persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub(crate) struct PersistentCacheKey<K> {
    key: K,
    checksum: String,
}

/// Persistent cache entry
#[cfg(autotune_persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub(crate) struct PersistentCacheValue {
    fastest_index: usize,
    results: Vec<AutotuneResult>,
}

#[cfg_attr(autotune_persistence, derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// The result of an autotune job.
pub struct AutotuneResult {
    pub(crate) outcome: Result<AutotuneOutcome, AutotuneError>,
}

impl AutotuneResult {
    pub(crate) fn error(error: AutotuneError) -> Self {
        Self {
            outcome: Err(error),
        }
    }
    pub(crate) fn success(outcome: AutotuneOutcome) -> Self {
        Self {
            outcome: Ok(outcome),
        }
    }
}

impl Eq for AutotuneResult {}
impl PartialEq for AutotuneResult {
    fn eq(&self, other: &Self) -> bool {
        match (&self.outcome, &other.outcome) {
            (Ok(lhs), Ok(rhs)) => lhs == rhs,
            (Ok(_), Err(_)) => false,
            (Err(_), Ok(_)) => false,
            // We don't have to check the error
            (Err(_), Err(_)) => true,
        }
    }
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    in_memory_cache: HashMap<K, CacheEntry>,
    /// `None` when the persistent cache is disabled, so no cache file is ever touched.
    #[cfg(autotune_persistence)]
    persistent_cache: Option<Cache<PersistentCacheKey<K>, PersistentCacheValue>>,
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
    /// A tuning job is in flight for this key — the worker hasn't published a result yet.
    /// The receiver wakes (with `Err(RecvError)`) when the worker commits the result. Native
    /// callers `block_on` it and re-query; wasm callers drop it and fall back.
    Pending,
    /// No operation is found yet.
    Miss,
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new(
        #[cfg_attr(not(autotune_persistence), allow(unused_variables))] name: &str,
        #[cfg_attr(not(autotune_persistence), allow(unused_variables))] device_id: &str,
    ) -> Self {
        #[cfg(autotune_persistence)]
        {
            use crate::config::RuntimeConfig;
            use alloc::format;

            let config = crate::config::CubeClRuntimeConfig::get();

            if config.autotune.disable_cache {
                return TuneCache {
                    in_memory_cache: HashMap::new(),
                    persistent_cache: None,
                };
            }

            #[allow(unused_mut)]
            let mut options = cubecl_environment::persistence::CacheOption::default();
            #[cfg(std_io)]
            {
                options = options.root(config.autotune.cache.root());
            }
            let mut cache = TuneCache {
                in_memory_cache: HashMap::new(),
                persistent_cache: Some(Cache::open(
                    format!("{device_id}/{name}"),
                    options.name("autotune"),
                )),
            };
            cache.load();
            cache
        }

        #[cfg(not(autotune_persistence))]
        {
            TuneCache {
                in_memory_cache: HashMap::new(),
            }
        }
    }

    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        let Some(val) = self.in_memory_cache.get(key) else {
            return TuneCacheResult::Miss;
        };

        let CacheEntry::Done {
            checksum,
            fastest_index,
        } = val
        else {
            // Pending: clone the receiver so the caller can subscribe to the in-flight tune.
            let CacheEntry::Pending = val else {
                unreachable!()
            };
            return TuneCacheResult::Pending;
        };

        if cfg!(autotune_persistence) {
            match checksum {
                ChecksumState::ToBeVerified(..) => TuneCacheResult::Unchecked, // Don't know yet.
                ChecksumState::NoMatch => TuneCacheResult::Miss,               // Can't use this.
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

    #[cfg(autotune_persistence)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) -> TuneCacheResult {
        let Some(val) = self.in_memory_cache.get_mut(key) else {
            return TuneCacheResult::Miss;
        };

        if let CacheEntry::Done {
            checksum: checksum_state,
            ..
        } = val
            && let ChecksumState::ToBeVerified(checksum_expected) = checksum_state
        {
            if checksum_expected == checksum {
                *checksum_state = ChecksumState::Match;
            } else {
                *checksum_state = ChecksumState::NoMatch;
            }
        }

        self.fastest(key)
    }

    /// Mark a key as being tuned. Used by [`Tuner::tune`] under the cache mutex so that
    /// concurrent callers see [`TuneCacheResult::Pending`] and wait on the same job instead of
    /// starting a second one. Returns `(Sender, Receiver)`:
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

#[cfg(autotune_persistence)]
impl<K: AutotuneKey> TuneCache<K> {
    /// Ingest entries newly delivered by the persistent backend into the
    /// in-memory cache as unverified.
    ///
    /// On the browser backend the initial hydration is asynchronous: entries
    /// only become visible some time after construction, so this is called on
    /// a cache miss before starting a tune. On the file system backend it
    /// picks up entries appended by other processes.
    pub(crate) fn sync_persistent(&mut self) {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return;
        };

        // Only asynchronous backends (browser hydration) can have new
        // content after construction without a local insert. Synchronous
        // backends are fully ingested at open, and re-reading them here
        // would take the multi-process file lock on the autotune hot path
        // while the tuner mutex is held.
        if !persistent_cache.pending_hydration() {
            return;
        }

        persistent_cache.for_each(|key, value| {
            self.in_memory_cache
                .entry(key.key.clone())
                .or_insert(CacheEntry::Done {
                    checksum: ChecksumState::ToBeVerified(key.checksum.clone()),
                    fastest_index: value.fastest_index,
                });
        });
    }

    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        fastest_index: usize,
        results: Vec<AutotuneResult>,
    ) {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return;
        };

        if let Err(err) = persistent_cache.insert(
            PersistentCacheKey { key, checksum },
            PersistentCacheValue {
                fastest_index,
                results,
            },
        ) {
            let CacheError::DuplicatedKey {
                key,
                value_previous,
                value_updated,
            } = err;

            // Either this process tuned the same function twice, or another
            // process stored its own result first. Both are harmless: the
            // stored entry stays, and the two are equally valid.
            log::warn!(
                "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
            );
        }
    }

    /// Load the persistent cache data from disk
    pub(crate) fn load(&mut self) {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return;
        };

        log::info!("Load autotune cache ...");
        let mut loaded = 0;
        persistent_cache.for_each(|key, value| {
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
