use alloc::vec;
use alloc::vec::Vec;

use cubecl_common::device::ServiceId;
use cubecl_common::hash::StableHash;
#[cfg(persistence)]
use cubecl_environment::persistence::StoreError;
#[cfg(persistence)]
use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};
#[cfg(serializable)]
use serde::{Deserialize, Serialize};

use super::{AutotuneError, AutotuneKey, AutotuneOutcome};
use crate::device::HardwareIdentity;
#[cfg(persistence)]
use crate::execution::TuneOutcome;
use crate::execution::TuneRegistration;
use crate::launched::CandidateKernelMap;
#[cfg(persistence)]
use crate::namespace_version::NamespaceVersion;
#[cfg(persistence)]
use alloc::string::String;
use cubecl_environment::collections::HashMap;

/// Bumped when tables are named another way.
#[cfg(persistence)]
static TABLES: NamespaceVersion = NamespaceVersion::new(ROOT, LAYOUT_PREFIX, 1);

#[derive(Debug)]
pub(crate) enum CacheEntry {
    Done {
        fastest_index: usize,
    },
    Pending,
    /// A `CompileOnly` override queued the kernels of the key's candidates on `services` and
    /// decided nothing: a miss to anything that tunes, and done to the next `CompileOnly` override
    /// on one of `services`. Never persisted. `registration` is the key's tune as the override that
    /// gathered it first registered it, settled once its pick commits; `None` when it registered
    /// none. Replacing the entry other than through [`TuneCache::mark_pending`] or
    /// [`TuneCache::look_up`] drops the registration, which counts it abandoned. `kernels` are
    /// what each candidate launched while it was gathered, handed to the tune that measures the
    /// key: a candidate that tune never runs is still known by the kernels it compiled.
    Compiled {
        registration: Option<TuneRegistration>,
        services: Vec<ServiceId>,
        kernels: CandidateKernelMap,
    },
}

/// What a `CompileOnly` override left for the tune that measures a key it gathered.
#[derive(Debug, Default)]
pub(crate) struct GatheredTune {
    /// Where the key's tune was registered, if the override registered it.
    pub registration: Option<TuneRegistration>,
    /// The kernels each candidate launched while it was gathered.
    pub kernels: CandidateKernelMap,
}

/// Persistent cache key
#[cfg(persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub struct PersistentCacheKey<K> {
    /// The autotune key identifying the operation.
    pub key: K,
    /// The checksum of the candidate list the key was tuned under: an answer
    /// to another list is an answer to another question.
    pub checksum: String,
}

/// Persistent cache entry
///
/// Only [`fastest_index`](Self::fastest_index) is read back: a look-up seeds the in-memory cache
/// from it and nothing else. Everything below it is stored so a cache entry can be inspected after
/// the fact — why a kernel won, against which measurements, and under which bounds — which is the
/// question that cannot be answered from a live process once tuning is over. That is also why the
/// type is `pub`: reading an entry back is the point.
#[cfg(persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PersistentCacheValue {
    /// Index of the fastest candidate operation.
    pub fastest_index: usize,
    /// Benchmarking results for all autotune candidates.
    pub results: Vec<AutotuneResult>,
    /// Optional input size bounds for which the autotune result applies.
    ///
    /// Defaulted, so entries written before this field existed still decode. Without it every
    /// cached key on every existing installation would fail to read and re-tune from scratch.
    #[serde(default)]
    pub bounds: Option<crate::tune::Bounds>,
    /// Optional execution time limit for the autotune process.
    ///
    /// Defaulted for the same reason as [`bounds`](Self::bounds).
    #[serde(default)]
    pub limit: Option<core::time::Duration>,
}

#[cfg_attr(serializable, derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// The result of an autotune job.
pub struct AutotuneResult {
    /// The outcome of the benchmark.
    pub outcome: Result<AutotuneOutcome, AutotuneError>,
    /// The kernels the candidate launched, while it was gathered or measured, by the stable hash
    /// that names each one's stored artifact: what tells a losing candidate's kernels, which no
    /// workload loads, from the winner's.
    ///
    /// Defaulted, so entries written before this field existed still decode; such an entry names
    /// no kernel, and nothing is known to be a loser's.
    #[cfg_attr(serializable, serde(default))]
    pub kernels: Vec<StableHash>,
}

impl AutotuneResult {
    pub(crate) fn error(error: AutotuneError) -> Self {
        Self {
            outcome: Err(error),
            kernels: Vec::new(),
        }
    }
    pub(crate) fn success(outcome: AutotuneOutcome) -> Self {
        Self {
            outcome: Ok(outcome),
            kernels: Vec::new(),
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
    /// lookup: tuned picks and in-flight tunes. Cleared when the environment
    /// switches.
    in_memory_cache: HashMap<K, CacheEntry>,
    /// The cards a tune found lost. Kept across environment switches: a card stays lost.
    lost: Vec<ServiceId>,
    /// Write-through persistence, or `None` when the persistent cache is
    /// disabled, so no cache file is ever touched. Lazy: an entry is read
    /// when [`Self::in_memory_cache`] misses, then lives there.
    #[cfg(persistence)]
    persistent_cache: Option<Store<PersistentCacheKey<K>, PersistentCacheValue>>,
    /// The namespace the table is stored under, whether or not it is
    /// persisted: what a [record](super::TuneRecord) names it by.
    #[cfg(persistence)]
    table: String,
    /// The environment generation [`Self::in_memory_cache`] was built under;
    /// see [`cubecl_environment::environment::generation`].
    #[cfg(persistence)]
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
    /// A tuning job is in flight for this key — the worker hasn't published a result yet.
    /// Callers that see this fall through to running the operation rather than blocking on
    /// the in-flight job. A card a tune found lost answers this for every key it has not settled.
    Pending,
    /// A [`CompileOnly`](crate::execution::ProcessMode::CompileOnly) override queued the kernels of
    /// the key's candidates, and measured and decided nothing. Callers run the first candidate that
    /// serves the problem, which only queues its kernels too.
    Compiled,
    /// No operation is found yet.
    Miss,
}

/// The root of every tuner's table namespace.
pub const ROOT: &str = "autotune";

/// What the segment numbering a table layout starts with, as in `layout-v1`.
pub const LAYOUT_PREFIX: &str = "layout-v";

/// What names the table one tuner stores its answers in on one hardware.
#[cfg(persistence)]
#[derive(Clone, Copy, Debug)]
pub struct TuneTableName<'a> {
    /// The hardware the answers were measured on, as its [`HardwareIdentity`] displays.
    pub hardware: &'a str,
    /// The tuner, named by the code that declared it.
    pub tuner: &'a str,
}

#[cfg(persistence)]
impl TuneTableName<'_> {
    /// `autotune/<version>/layout-v<layout>/<hardware>/<tuner>`.
    pub fn namespace(&self) -> Namespace {
        let Self { hardware, tuner } = self;
        TABLES.namespace(&alloc::format!("{hardware}/{tuner}"))
    }
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new(
        #[cfg_attr(not(persistence), allow(unused_variables))] name: &str,
        #[cfg_attr(not(persistence), allow(unused_variables))] hardware: &HardwareIdentity,
    ) -> Self {
        #[cfg(persistence)]
        {
            use crate::config::RuntimeConfig;
            use alloc::string::ToString;

            let config = crate::config::CubeClRuntimeConfig::get();
            let namespace = TuneTableName {
                hardware: &hardware.to_string(),
                tuner: name,
            }
            .namespace();
            let table = namespace.as_str().into();

            if config.autotune.disable_cache {
                return TuneCache {
                    in_memory_cache: HashMap::new(),
                    lost: Vec::new(),
                    persistent_cache: None,
                    table,
                    generation: cubecl_environment::environment::generation(),
                };
            }

            // Sampled before the store opens, so a switch landing in between
            // reads as "rebuild", never as "this state belongs to the new
            // environment".
            let generation = cubecl_environment::environment::generation();
            TuneCache {
                in_memory_cache: HashMap::new(),
                lost: Vec::new(),
                persistent_cache: Some(Store::new(
                    StoreOptions::new()
                        .storage(namespace)
                        .cache(CacheOption::Lazy),
                )),
                table,
                generation,
            }
        }

        #[cfg(not(persistence))]
        {
            TuneCache {
                in_memory_cache: HashMap::new(),
                lost: Vec::new(),
            }
        }
    }

    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        match self.in_memory_cache.get(key) {
            Some(CacheEntry::Done { fastest_index }) => TuneCacheResult::Hit {
                fastest_index: *fastest_index,
            },
            Some(CacheEntry::Pending) => TuneCacheResult::Pending,
            // Nothing was decided: a tune of the key is still owed.
            Some(CacheEntry::Compiled { .. }) | None => TuneCacheResult::Miss,
        }
    }

    /// The index settled for `key`, reading only: `None` for anything but a settled result, and
    /// for every key once the environment has switched away from the one this cache was built
    /// under. The reset that switch owes is left to the next lookup that tunes.
    pub(crate) fn settled(&self, key: &K) -> Option<usize> {
        #[cfg(persistence)]
        if self.environment_switched() {
            return None;
        }

        match self.fastest(key) {
            TuneCacheResult::Hit { fastest_index } => Some(fastest_index),
            TuneCacheResult::Pending | TuneCacheResult::Compiled | TuneCacheResult::Miss => None,
        }
    }

    /// Mark a key as being tuned. Used by [`Tuner::check_tune`] under the cache mutex so that
    /// concurrent callers see [`TuneCacheResult::Pending`] instead of starting a second job
    /// for the same key. Returns what the override that gathered the key left for its tune.
    pub(crate) fn mark_pending(&mut self, key: K) -> GatheredTune {
        match self.in_memory_cache.insert(key, CacheEntry::Pending) {
            Some(CacheEntry::Compiled {
                registration,
                kernels,
                ..
            }) => GatheredTune {
                registration,
                kernels,
            },
            _ => GatheredTune::default(),
        }
    }

    /// Drop the [mark](Self::mark_pending) of a tune that stopped before it settled, so the next
    /// caller tunes the key.
    pub(crate) fn unmark_pending(&mut self, key: &K) {
        if let Some(CacheEntry::Pending) = self.in_memory_cache.get(key) {
            self.in_memory_cache.remove(key);
        }
    }

    /// Mark a key whose candidates' kernels a `CompileOnly` override is queueing on `service`, in
    /// place of a tune: nothing is decided. The first service to queue them registers the key's
    /// tune with `register`, and the others share that registration.
    pub(crate) fn mark_compiled(
        &mut self,
        key: K,
        service: ServiceId,
        register: impl FnOnce() -> Option<TuneRegistration>,
    ) {
        if let Some(CacheEntry::Compiled { services, .. }) = self.in_memory_cache.get_mut(&key) {
            services.push(service);
            return;
        }
        self.in_memory_cache.insert(
            key,
            CacheEntry::Compiled {
                registration: register(),
                services: vec![service],
                kernels: CandidateKernelMap::default(),
            },
        );
    }

    /// Add what a `CompileOnly` override's run of `key`'s candidates launched to what the key
    /// gathered: several cards gather one key, each launching its own candidates. Nothing is
    /// kept for a key no override is gathering any more.
    pub(crate) fn add_gathered(&mut self, key: &K, launched: CandidateKernelMap) {
        let Some(CacheEntry::Compiled { kernels, .. }) = self.in_memory_cache.get_mut(key) else {
            return;
        };
        for (candidate, launched) in launched {
            let gathered = kernels.entry(candidate).or_default();
            gathered.extend(launched);
            gathered.sort_unstable();
            gathered.dedup();
        }
    }

    /// Whether a `CompileOnly` override already queued the kernels of `key`'s candidates on
    /// `service`.
    pub(crate) fn is_compiled(&self, key: &K, service: ServiceId) -> bool {
        matches!(
            self.in_memory_cache.get(key),
            Some(CacheEntry::Compiled { services, .. }) if services.contains(&service)
        )
    }

    pub(crate) fn mark_lost(&mut self, service: ServiceId) {
        if !self.lost.contains(&service) {
            self.lost.push(service);
        }
    }

    pub(crate) fn is_lost(&self, service: ServiceId) -> bool {
        self.lost.contains(&service)
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.in_memory_cache
            .insert(key, CacheEntry::Done { fastest_index });
    }
}

#[cfg(persistence)]
impl<K: AutotuneKey> TuneCache<K> {
    /// Drops tuning state belonging to a previous environment, so a switch
    /// reads the new environment's store and tunes rather than serving the
    /// old environment's picks. One relaxed atomic load when nothing switched.
    ///
    /// In-flight tunes are dropped with everything else: their completion
    /// still records a hardware-valid result, so the whole cost of the race
    /// is one duplicate tune per switch. A key a `CompileOnly` override
    /// gathered for the old environment is dropped too, and its registration
    /// settles as abandoned: the key is tuned, if at all, for the new
    /// environment, under a registration of its own.
    pub(crate) fn reset_if_environment_switched(&mut self) {
        if !self.environment_switched() {
            return;
        }

        log::debug!("Environment switched, resetting the autotune cache");
        self.generation = cubecl_environment::environment::generation();
        self.in_memory_cache.clear();
    }

    /// The environment generation this cache was built under, or last reset to.
    pub(crate) fn generation(&self) -> u32 {
        self.generation
    }

    /// Whether the environment switched since this cache was built. Never with persistence
    /// disabled: the tuning state is then process-local and unbound, like a store without a
    /// storage, and survives switches.
    fn environment_switched(&self) -> bool {
        self.persistent_cache.is_some()
            && cubecl_environment::environment::generation() != self.generation
    }

    /// The store's answer for `key` under the set's `checksum`, read once memory missed. A miss
    /// costs one read of the store, under the tuner mutex, and a key read counts as loaded.
    pub(crate) fn look_up(&mut self, key: &K, checksum: &str) -> TuneCacheResult {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return TuneCacheResult::Miss;
        };
        let stored = PersistentCacheKey {
            key: key.clone(),
            checksum: checksum.into(),
        };
        let Some(value) = persistent_cache.remove(&stored) else {
            return TuneCacheResult::Miss;
        };
        let gathered = match self.in_memory_cache.remove(key) {
            Some(CacheEntry::Compiled { registration, .. }) => registration,
            _ => None,
        };
        if let Some(registration) = gathered.or_else(TuneRegistration::register) {
            registration.settle(TuneOutcome::Loaded);
        }
        self.cache_insert(key.clone(), value.fastest_index);
        TuneCacheResult::Hit {
            fastest_index: value.fastest_index,
        }
    }

    /// The namespace the table is stored under.
    pub(crate) fn table(&self) -> &str {
        &self.table
    }

    /// Store an answer in the table, and say whether it took it: never when
    /// the cache is disabled.
    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        value: PersistentCacheValue,
    ) -> bool {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return false;
        };

        let Err(err) = persistent_cache.insert(PersistentCacheKey { key, checksum }, value) else {
            return true;
        };
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
                "Autotune result for key {key:?} could not be stored, it will be re-tuned: {error}"
            ),
        }
        false
    }
}
