use cubecl_common::device::ServiceId;
use cubecl_environment::collections::{HashMap, HashSet};
#[cfg(persistence)]
use cubecl_environment::persistence::StoreError;
#[cfg(persistence)]
use cubecl_environment::persistence::{CacheOption, Store, StoreOptions};

use crate::device::HardwareIdentity;
#[cfg(persistence)]
use crate::execution::TuneOutcome;
use crate::execution::TuneRegistration;
use crate::tune::AutotuneKeyable;
#[cfg(persistence)]
use crate::tune::cache::table::{
    PersistentCacheKey, PersistentCacheValue, TuneTable, TuneTableName,
};
#[cfg(persistence)]
use alloc::string::String;

/// What a [`TuneTracker`] knows of one key.
#[derive(Debug)]
pub enum CacheEntry {
    Done {
        fastest_index: usize,
    },
    Pending,
    /// A `CompileOnly` override or another key's compile pass queued the kernels of the top of
    /// the key's plan on `services` and decided nothing: a miss to anything that tunes, and done
    /// to the next `CompileOnly` override on one of `services`. Never persisted. `registration` is
    /// the key's tune as the override that gathered it first registered it, settled once its pick
    /// commits; `None` when it registered none. Replacing the entry other than through
    /// [`TuneTracker::track_pending`] or [`TuneTracker::track_stored`] drops the registration, which
    /// counts it abandoned.
    Compiled {
        registration: Option<TuneRegistration>,
        /// The cards that queued the kernels (Membership Usage).
        services: HashSet<ServiceId>,
    },
}

impl CacheEntry {
    /// Where the gathered key's tune was registered, if it was gathered and registered.
    fn into_registration(self) -> Option<TuneRegistration> {
        match self {
            Self::Compiled { registration, .. } => registration,
            Self::Done { .. } | Self::Pending => None,
        }
    }
}

/// Tracks the answers one tuner holds: tuned picks and tunes in flight in memory, the cards found
/// lost, and, where tunes persist, the table every device of its hardware shares.
#[derive(Debug)]
pub struct TuneTracker<K> {
    /// The single in-memory home of tuning state, keyed for the per-launch lookup (Keyed Usage):
    /// tuned picks and in-flight tunes. Cleared when the environment switches.
    answers: HashMap<K, CacheEntry>,
    /// The cards a tune found lost (Membership Usage). Kept across environment switches: a card
    /// stays lost.
    lost: HashSet<ServiceId>,
    /// Write-through persistence, or `None` when the persistent cache is disabled, so no cache
    /// file is ever touched. Lazy: an entry is read when [`Self::answers`] misses, then lives
    /// there.
    #[cfg(persistence)]
    persistent_cache: Option<Store<PersistentCacheKey<K>, PersistentCacheValue>>,
    #[cfg(persistence)]
    table: TuneTable,
}

/// What a tuner answers for a key.
#[derive(Debug)]
pub enum TuneAnswer {
    /// The key is settled.
    Hit {
        /// The index of the fastest operation to execute.
        fastest_index: usize,
    },
    /// A tuning job is in flight for this key — the worker hasn't published a result yet.
    /// Callers that see this fall through to running the operation rather than blocking on
    /// the in-flight job. A card a tune found lost answers this for every key it has not settled.
    Pending,
    /// A [`CompileOnly`](crate::execution::ProcessMode::CompileOnly) override or another key's
    /// compile pass queued the kernels of the top of the key's plan, and measured and decided
    /// nothing. Callers run the first candidate that serves the problem, which only queues its
    /// kernels too.
    Compiled,
    /// Memory holds no decision for the key: it was never tuned, only gathered, or reset by an
    /// environment switch.
    Miss,
}

impl<K: AutotuneKeyable> TuneTracker<K> {
    /// A tracker of the answers of the tuner `name` on `hardware`, reading and writing that
    /// table unless the cache is disabled.
    pub fn new(
        #[cfg_attr(not(persistence), allow(unused_variables))] name: &str,
        #[cfg_attr(not(persistence), allow(unused_variables))] hardware: &HardwareIdentity,
    ) -> Self {
        #[cfg(persistence)]
        {
            use crate::config::RuntimeConfig;
            use alloc::string::ToString;

            let disabled = crate::config::CubeClRuntimeConfig::get()
                .autotune
                .disable_cache;
            let namespace = TuneTableName {
                hardware: &hardware.to_string(),
                tuner: name,
            }
            .namespace();
            // Sampled before the store opens, so a switch landing in between reads as "rebuild",
            // never as "this state belongs to the new environment".
            let table = TuneTable {
                namespace: namespace.as_str().into(),
                generation: cubecl_environment::environment::generation(),
            };
            let store = (!disabled).then(|| {
                Store::new(
                    StoreOptions::new()
                        .storage(namespace)
                        .cache(CacheOption::Lazy),
                )
            });
            Self::with_store(store, table)
        }

        #[cfg(not(persistence))]
        {
            TuneTracker {
                answers: HashMap::new(),
                lost: HashSet::new(),
            }
        }
    }

    /// What memory holds for `key`: [`Miss`](TuneAnswer::Miss) for a key only gathered, and,
    /// where tunes persist, for every key once the environment has switched away from the one the
    /// answers were built under. The reset that switch owes is left to the next lookup that tunes.
    pub fn fastest(&self, key: &K) -> TuneAnswer {
        #[cfg(persistence)]
        if self.environment_switched() {
            return TuneAnswer::Miss;
        }

        match self.answers.get(key) {
            Some(CacheEntry::Done { fastest_index }) => TuneAnswer::Hit {
                fastest_index: *fastest_index,
            },
            Some(CacheEntry::Pending) => TuneAnswer::Pending,
            // Nothing was decided: a tune of the key is still owed.
            Some(CacheEntry::Compiled { .. }) | None => TuneAnswer::Miss,
        }
    }

    /// What `key` answers on `service` without a tune, if anything: its pick or its tune in
    /// flight, [`Compiled`](TuneAnswer::Compiled) where `compiling` and `service` already
    /// queued its kernels, and [`Pending`](TuneAnswer::Pending) on a lost card, where every
    /// candidate would meet the loss and the key is left to a healthy card.
    pub fn answer(&self, key: &K, service: ServiceId, compiling: bool) -> Option<TuneAnswer> {
        // A key a `CompileOnly` override already queued the candidates of on this card answers at
        // once: a walk reaches the same key at every layer, and a miss would checksum the set and
        // read the store again each time for a tune that will not run.
        if compiling && self.is_compiled(key, service) {
            return Some(TuneAnswer::Compiled);
        }
        let known = self.fastest(key);
        if matches!(known, TuneAnswer::Hit { .. } | TuneAnswer::Pending) {
            return Some(known);
        }
        self.lost.contains(&service).then_some(TuneAnswer::Pending)
    }

    /// Track `key` as being tuned, so concurrent callers see [`TuneAnswer::Pending`] instead
    /// of starting a second job for the same key. Returns where the key's tune was registered, if
    /// the override that gathered it registered it.
    pub fn track_pending(&mut self, key: K) -> Option<TuneRegistration> {
        self.answers
            .insert(key, CacheEntry::Pending)
            .and_then(CacheEntry::into_registration)
    }

    /// Drop the [tracked](Self::track_pending) tune of `key` that stopped before it settled, so
    /// the next caller tunes the key. A settled key is left settled.
    pub fn untrack_pending(&mut self, key: &K) {
        if let Some(CacheEntry::Pending) = self.answers.get(key) {
            self.answers.remove(key);
        }
    }

    /// Track `key` as one whose candidates' kernels a `CompileOnly` override is queueing on
    /// `service`, in place of a tune: nothing is decided. The first service to queue them
    /// registers the key's tune with `register`, and the others share that registration.
    pub fn track_compiled(
        &mut self,
        key: K,
        service: ServiceId,
        register: impl FnOnce() -> Option<TuneRegistration>,
    ) {
        let compiled = || CacheEntry::Compiled {
            registration: register(),
            services: HashSet::new(),
        };
        if let CacheEntry::Compiled { services, .. } =
            self.answers.entry(key).or_insert_with(compiled)
        {
            services.insert(service);
        }
    }

    #[cfg(not(target_family = "wasm"))]
    /// Track `service` as a card a tune found lost.
    pub fn track_lost(&mut self, service: ServiceId) {
        self.lost.insert(service);
    }

    /// Track `fastest_index` as the pick of `key`, in memory, and answer it.
    pub fn track_answer(&mut self, key: K, fastest_index: usize) -> TuneAnswer {
        self.answers.insert(key, CacheEntry::Done { fastest_index });
        TuneAnswer::Hit { fastest_index }
    }
}

#[cfg(persistence)]
impl<K: AutotuneKeyable> TuneTracker<K> {
    /// A tracker over `persistent_cache`, its answers built for `table`.
    pub fn with_store(
        persistent_cache: Option<Store<PersistentCacheKey<K>, PersistentCacheValue>>,
        table: TuneTable,
    ) -> Self {
        Self {
            answers: HashMap::new(),
            lost: HashSet::new(),
            persistent_cache,
            table,
        }
    }

    /// Drops tuning state belonging to a previous environment, so a switch reads the new
    /// environment's store and tunes rather than serving the old environment's picks. One relaxed
    /// atomic load when nothing switched.
    ///
    /// In-flight tunes are dropped with everything else: their completion still records a
    /// hardware-valid result, so the whole cost of the race is one duplicate tune per switch. A
    /// key a `CompileOnly` override gathered for the old environment is dropped too, and its
    /// registration settles as abandoned: the key is tuned, if at all, for the new environment,
    /// under a registration of its own.
    pub fn reset_if_environment_switched(&mut self) {
        if !self.environment_switched() {
            return;
        }

        log::debug!("Environment switched, resetting the autotune cache");
        self.table.generation = cubecl_environment::environment::generation();
        self.answers.clear();
    }

    pub fn table(&self) -> &TuneTable {
        &self.table
    }

    /// Track the store's answer for `key` under the set's `checksum`, read once memory missed,
    /// and answer it. A miss costs one read of the store, under the tuner mutex, and a key read
    /// counts as loaded. A stored index outside the `len` candidates of the set is a miss: it was
    /// written by another build, or edited.
    pub fn track_stored(&mut self, key: &K, checksum: &str, len: usize) -> TuneAnswer {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return TuneAnswer::Miss;
        };
        let stored = PersistentCacheKey {
            key: key.clone(),
            checksum: checksum.into(),
        };
        let Some(value) = persistent_cache
            .remove(&stored)
            .filter(|value| value.decided_index < len)
        else {
            return TuneAnswer::Miss;
        };
        let gathered = self
            .answers
            .remove(key)
            .and_then(CacheEntry::into_registration);
        if let Some(registration) = gathered.or_else(TuneRegistration::register) {
            registration.settle(TuneOutcome::Loaded);
        }
        self.track_answer(key.clone(), value.decided_index)
    }

    /// Store an answer in the table, and say whether it took it: never when the cache is
    /// disabled.
    pub fn store_answer(&mut self, key: K, checksum: String, value: PersistentCacheValue) -> bool {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return false;
        };

        let Err(err) = persistent_cache.insert(PersistentCacheKey { key, checksum }, value) else {
            return true;
        };
        log_store_error(err);
        false
    }
}

impl<K: AutotuneKeyable> TuneTracker<K> {
    /// Whether a `CompileOnly` override already queued the kernels of the top of `key`'s plan on
    /// `service`.
    fn is_compiled(&self, key: &K, service: ServiceId) -> bool {
        matches!(
            self.answers.get(key),
            Some(CacheEntry::Compiled { services, .. }) if services.contains(&service)
        )
    }
}

#[cfg(persistence)]
impl<K: AutotuneKeyable> TuneTracker<K> {
    /// Whether the environment switched since the answers were built. Never with persistence
    /// disabled: the tuning state is then process-local and unbound, like a store without a
    /// storage, and survives switches.
    fn environment_switched(&self) -> bool {
        self.persistent_cache.is_some()
            && cubecl_environment::environment::generation() != self.table.generation
    }
}

/// Log why the table did not take an answer.
#[cfg(persistence)]
fn log_store_error<K: core::fmt::Debug, V: core::fmt::Debug>(err: StoreError<K, V>) {
    match err {
        StoreError::DuplicatedKey {
            key,
            value_previous,
            value_updated,
        } => log::warn!(
            "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
        ),
        // Another process sharing the cache root tuned this key first. Routine with N training
        // processes on a cold cache, and both results are valid, so it stays quiet: warning here
        // would print a full result payload per key on every cold start.
        StoreError::KeyOutOfSync { key, .. } => {
            log::debug!("Autotune result for key {key:?} was already stored concurrently")
        }
        StoreError::Backend { key, error } => log::warn!(
            "Autotune result for key {key:?} could not be stored, it will be re-tuned: {error}"
        ),
    }
}

#[cfg(test)]
mod small_tests {
    use super::*;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;
    use cubecl_common::device::DeviceId;

    fn tracker() -> TuneTracker<String> {
        #[cfg(persistence)]
        {
            TuneTracker::with_store(
                None,
                TuneTable {
                    namespace: "tune-cache-small".into(),
                    generation: cubecl_environment::environment::generation(),
                },
            )
        }
        #[cfg(not(persistence))]
        {
            TuneTracker {
                answers: HashMap::new(),
                lost: HashSet::new(),
            }
        }
    }

    fn key() -> String {
        "key".to_string()
    }

    fn service(index: u16) -> ServiceId {
        ServiceId::of::<()>(DeviceId::new(0, index))
    }

    #[test]
    fn an_untracked_key_is_a_miss_with_no_answer() {
        let tracker = tracker();

        assert!(matches!(tracker.fastest(&key()), TuneAnswer::Miss));
        assert!(tracker.answer(&key(), service(0), false).is_none());
    }

    #[test]
    fn a_tracked_answer_is_a_hit() {
        let mut tracker = tracker();
        tracker.track_answer(key(), 3);

        assert!(matches!(
            tracker.answer(&key(), service(0), false),
            Some(TuneAnswer::Hit { fastest_index: 3 })
        ));
    }

    #[test]
    fn a_pending_tune_answers_pending_until_untracked() {
        let mut tracker = tracker();
        tracker.track_pending(key());
        let pending = tracker.answer(&key(), service(0), false);
        tracker.untrack_pending(&key());

        assert!(matches!(pending, Some(TuneAnswer::Pending)));
        assert!(matches!(tracker.fastest(&key()), TuneAnswer::Miss));
    }

    #[test]
    fn untracking_a_pending_tune_leaves_a_settled_key() {
        let mut tracker = tracker();
        tracker.track_answer(key(), 1);
        tracker.untrack_pending(&key());

        assert!(matches!(
            tracker.fastest(&key()),
            TuneAnswer::Hit { fastest_index: 1 }
        ));
    }

    #[test]
    fn a_compiled_key_answers_compiled_only_to_a_compiling_card_that_queued_it() {
        let mut tracker = tracker();
        tracker.track_compiled(key(), service(0), || None);

        assert!(matches!(
            tracker.answer(&key(), service(0), true),
            Some(TuneAnswer::Compiled)
        ));
    }

    #[test]
    fn a_compiled_key_is_owed_a_tune_on_another_card() {
        let mut tracker = tracker();
        tracker.track_compiled(key(), service(0), || None);

        assert!(tracker.answer(&key(), service(1), true).is_none());
    }

    #[test]
    fn a_compiled_key_is_owed_a_tune_outside_a_compile_pass() {
        let mut tracker = tracker();
        tracker.track_compiled(key(), service(0), || None);

        assert!(tracker.answer(&key(), service(0), false).is_none());
    }

    #[test]
    fn a_second_card_compiling_a_key_joins_the_first_one_s_registration() {
        let mut tracker = tracker();
        tracker.track_compiled(key(), service(0), || None);
        tracker.track_compiled(key(), service(1), || {
            panic!("a key already compiled should not register again")
        });

        assert!(tracker.is_compiled(&key(), service(1)));
    }

    #[test]
    fn a_lost_card_answers_pending_for_an_unsettled_key() {
        let mut tracker = tracker();
        tracker.track_lost(service(0));
        tracker.track_lost(service(0));

        assert!(matches!(
            tracker.answer(&key(), service(0), false),
            Some(TuneAnswer::Pending)
        ));
    }

    #[test]
    fn a_healthy_card_is_owed_a_tune_beside_a_lost_one() {
        let mut tracker = tracker();
        tracker.track_lost(service(0));

        assert!(tracker.answer(&key(), service(1), false).is_none());
    }

    #[cfg(persistence)]
    #[test]
    fn a_tracker_that_persists_nothing_never_resets() {
        let mut tracker = tracker();
        tracker.track_answer(key(), 2);
        tracker.table.generation = tracker.table.generation.wrapping_add(1);

        tracker.reset_if_environment_switched();

        assert!(matches!(
            tracker.fastest(&key()),
            TuneAnswer::Hit { fastest_index: 2 }
        ));
    }

    #[cfg(persistence)]
    #[test]
    fn a_tracker_without_a_store_looks_nothing_up_and_stores_nothing() {
        let mut tracker = tracker();
        let value = PersistentCacheValue {
            decided_index: 0,
            results: Vec::new(),
            bounds: None,
            limit: None,
        };

        assert!(matches!(
            tracker.track_stored(&key(), "checksum", 1),
            TuneAnswer::Miss
        ));
        assert!(!tracker.store_answer(key(), "checksum".into(), value));
    }
}

#[cfg(all(test, persistence))]
mod tests;
