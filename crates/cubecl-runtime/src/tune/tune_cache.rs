#[cfg(std_io)]
use std::vec::Vec;

#[cfg(std_io)]
use cubecl_common::cache::Cache;
#[cfg(std_io)]
use cubecl_common::cache::CacheError;
#[cfg(std_io)]
use serde::{Deserialize, Serialize};

use super::{AutotuneError, AutotuneKey, AutotuneOutcome};
use alloc::string::String;
use async_channel::{Receiver, Sender};
use hashbrown::HashMap;

/// In-memory cache entry.
///
/// `Pending` marks that a tuning job for this key has been handed off to the worker but hasn't
/// committed a result yet. It also carries a [`Receiver<()>`] that all concurrent callers can
/// clone and wait on: when the worker commits a result it drops its [`Sender`], which closes
/// the channel and wakes every waiter. That way a native caller who loses the race to start
/// the tune still blocks on the same job instead of kicking off redundant `try_all_operations`.
///
/// `Done` is a completed tuning result. Together these replace the old side-channel
/// `autotuning: HashSet<K>` — `Pending` is the single source of truth for "is anyone already
/// tuning this key?".
#[derive(Debug)]
pub(crate) enum CacheEntry {
    Done {
        checksum: ChecksumState,
        fastest_index: usize,
    },
    Pending {
        done_rx: Receiver<()>,
    },
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
    results: Vec<AutotuneResult>,
}

#[cfg_attr(std_io, derive(Serialize, Deserialize))]
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
    /// A tuning job is in flight for this key — the worker hasn't published a result yet.
    /// The receiver wakes (with `Err(RecvError)`) when the worker commits the result. Native
    /// callers `block_on` it and re-query; wasm callers drop it and fall back.
    Pending(Receiver<()>),
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
            use std::format;

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
        let Some(val) = self.in_memory_cache.get(key) else {
            return TuneCacheResult::Miss;
        };

        let CacheEntry::Done {
            checksum,
            fastest_index,
        } = val
        else {
            // Pending: clone the receiver so the caller can subscribe to the in-flight tune.
            let CacheEntry::Pending { done_rx } = val else {
                unreachable!()
            };
            return TuneCacheResult::Pending(done_rx.clone());
        };

        if cfg!(std_io) {
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

    #[cfg(std_io)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) {
        let Some(val) = self.in_memory_cache.get_mut(key) else {
            return;
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
    }

    /// Mark a key as being tuned. Used by [`Tuner::tune`] under the cache mutex so that
    /// concurrent callers see [`TuneCacheResult::Pending`] and wait on the same job instead of
    /// starting a second one. Returns `(Sender, Receiver)`:
    ///
    /// - The caller hands the `Sender` to the worker along with the tuning request; when the
    ///   worker drops it (after committing the result), the channel closes and every `Receiver`
    ///   wakes with `Err(RecvError)`.
    /// - The caller also gets the `Receiver` back so it can block on its own tune. A clone of
    ///   that receiver is stored inside the `Pending` entry so any concurrent caller that sees
    ///   `Pending` in `fastest()` can subscribe to the same signal.
    pub(crate) fn mark_pending(&mut self, key: K) -> (Sender<()>, Receiver<()>) {
        let (tx, rx) = async_channel::unbounded::<()>();
        self.in_memory_cache.insert(
            key,
            CacheEntry::Pending {
                done_rx: rx.clone(),
            },
        );
        (tx, rx)
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
        results: Vec<AutotuneResult>,
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
