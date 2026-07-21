#[cfg(autotune_persistence)]
use alloc::vec::Vec;

#[cfg(autotune_persistence)]
use cubecl_environment::persistence::StoreError;
#[cfg(autotune_persistence)]
use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};
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
    /// The single in-memory home of tuning state, keyed for the per-launch
    /// lookup: tuned picks, in-flight tunes and checksum verdicts. Hydrated
    /// from the store, which retains nothing itself, and rebuilt when the
    /// environment switches.
    in_memory_cache: HashMap<K, CacheEntry>,
    /// Write-through persistence, or `None` when the persistent cache is
    /// disabled, so no cache file is ever touched. Lazy: entries live in
    /// [`Self::in_memory_cache`] once hydrated, not here.
    #[cfg(autotune_persistence)]
    persistent_cache: Option<Store<PersistentCacheKey<K>, PersistentCacheValue>>,
    /// Whether everything the store holds has been ingested into
    /// [`Self::in_memory_cache`]. What makes an ordinary miss cost a bool
    /// check rather than a walk; `false` while an asynchronous storage
    /// (browser) is still loading, and again after an environment switch.
    #[cfg(autotune_persistence)]
    hydrated: bool,
    /// The environment generation [`Self::in_memory_cache`] was built under;
    /// see [`cubecl_environment::environment::generation`].
    #[cfg(autotune_persistence)]
    generation: u32,
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
    /// Callers that see this fall through to running the operation rather than blocking on
    /// the in-flight job.
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
                    hydrated: true,
                    generation: cubecl_environment::environment::generation(),
                };
            }

            // Sampled before the store opens, so a switch landing in between
            // reads as "rebuild", never as "this state belongs to the new
            // environment".
            let generation = cubecl_environment::environment::generation();
            let namespace = Namespace::scoped("autotune", format!("{device_id}/{name}"));
            let mut cache = TuneCache {
                in_memory_cache: HashMap::new(),
                persistent_cache: Some(Store::new(
                    StoreOptions::new()
                        .storage(namespace)
                        .cache(CacheOption::Lazy),
                )),
                hydrated: false,
                generation,
            };
            log::info!("Load autotune cache ...");
            let loaded = cache.sync_persistent();
            log::info!("Loaded {loaded} autotune cached entries");

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
    /// concurrent callers see [`TuneCacheResult::Pending`] instead of starting a second job
    /// for the same key.
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
    /// Drops tuning state belonging to a previous environment, so a switch
    /// re-hydrates and re-tunes rather than serving the old environment's
    /// picks. One relaxed atomic load when nothing switched.
    ///
    /// In-flight tunes are dropped with everything else: their completion
    /// still records a hardware-valid result, so the whole cost of the race
    /// is one duplicate tune per switch.
    pub(crate) fn reset_if_environment_switched(&mut self) {
        // Persistence disabled means the tuning state is process-local and
        // unbound, like a store without a storage: it survives switches.
        if self.persistent_cache.is_none() {
            return;
        }

        let generation = cubecl_environment::environment::generation();
        if generation == self.generation {
            return;
        }

        log::debug!("Environment switched, resetting the autotune cache");
        self.generation = generation;
        self.in_memory_cache.clear();
        self.hydrated = false;
    }

    /// Ingest everything the persistent store holds into the in-memory cache,
    /// as unverified entries.
    ///
    /// Runs at construction, and again whenever `hydrated` fell back to
    /// `false`: after an environment switch, and on the browser backend while
    /// its asynchronous hydration is still in flight. Once hydrated, a miss
    /// costs one bool check here — never a walk, and never a rescan of the
    /// database under the tuner mutex.
    ///
    /// Returns how many entries the store delivered.
    pub(crate) fn sync_persistent(&mut self) -> usize {
        if self.hydrated {
            return 0;
        }

        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return 0;
        };

        let mut delivered = 0;
        let complete = persistent_cache.scan(|key, value| {
            delivered += 1;
            self.in_memory_cache
                .entry(key.key)
                .or_insert(CacheEntry::Done {
                    checksum: ChecksumState::ToBeVerified(key.checksum),
                    fastest_index: value.fastest_index,
                });
        });
        self.hydrated = complete;

        delivered
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
            match err {
                StoreError::DuplicatedKey {
                    key,
                    value_previous,
                    value_updated,
                } => log::warn!(
                    "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
                ),
                // Another process sharing the cache root tuned this key first.
                // Routine with N training processes on a cold cache, and both
                // results are valid, so it stays quiet: warning here would
                // print a full result payload per key on every cold start.
                StoreError::KeyOutOfSync { key, .. } => {
                    log::debug!("Autotune result for key {key:?} was already stored concurrently")
                }
                StoreError::Backend { key, error } => log::warn!(
                    "Autotune result for key {key:?} could not be stored, it will be retuned: {error}"
                ),
            }
        }
    }
}
