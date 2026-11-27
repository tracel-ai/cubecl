use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(not(target_family = "wasm"))]
use cubecl_common::device::ServiceId;
use cubecl_environment::sync::Mutex;

use crate::config::Logger;
use crate::execution::{TuneOutcome, TuneRegistration};
#[cfg(not(target_family = "wasm"))]
use crate::tune::cache::tracker::TuneCacheResult;
use crate::tune::cache::tracker::TuneTracker;
use crate::tune::cache::tune_logger::TuneLogger;
use crate::tune::drive::pending_unmarker::PendingUnmarker;
use crate::tune::{AutotuneKeyable, AutotuneResult};

/// Settles the key one tune answered: reports the answer, keeps it in memory, settles the tune's
/// registration, and, where tunes persist, stores and records it.
pub struct TuneSettler<K: AutotuneKeyable> {
    cache: Arc<Mutex<TuneTracker<K>>>,
    log_sink: Arc<Mutex<Logger>>,
    /// Where the tune settles once its pick is committed.
    registration: Option<TuneRegistration>,
    unmarker: PendingUnmarker<K>,
    #[cfg(persistence)]
    persister: crate::tune::drive::persister::TunePersister<K>,
}

impl<K: AutotuneKeyable> TuneSettler<K> {
    pub fn new(
        cache: Arc<Mutex<TuneTracker<K>>>,
        log_sink: Arc<Mutex<Logger>>,
        registration: Option<TuneRegistration>,
        unmarker: PendingUnmarker<K>,
        #[cfg(persistence)] persister: crate::tune::drive::persister::TunePersister<K>,
    ) -> Self {
        Self {
            cache,
            log_sink,
            registration,
            unmarker,
            #[cfg(persistence)]
            persister,
        }
    }

    /// Settle `key` on `fastest_index`, which is `unmeasured` where it was picked with nothing
    /// measured, `results` holding every candidate's result by tunable index.
    pub fn settle(
        self,
        key: K,
        results: Vec<AutotuneResult>,
        fastest_index: usize,
        unmeasured: bool,
        logger: &TuneLogger,
    ) {
        // Ranked for the report and the table: measured ones first, fastest first.
        let mut results = results;
        results.sort_by_key(AutotuneResult::score);
        logger.log_result(&mut self.log_sink.lock(), &key, &results, fastest_index);
        // In-memory regardless: without it this key re-tunes on every call, and a tune that
        // measured nothing would keep failing the same way.
        self.cache.lock().track_answer(key.clone(), fastest_index);
        // Where the tune was registered, whether or not that override is still open.
        let outcome = match unmeasured {
            true => TuneOutcome::Failed,
            false => TuneOutcome::Measured,
        };
        #[cfg_attr(not(persistence), allow(unused_variables))]
        let settled = self
            .registration
            .map(|registration| registration.settle(outcome));

        #[cfg(persistence)]
        {
            let stored = self.persister.persist(
                &self.cache,
                key,
                results,
                fastest_index,
                unmeasured,
                logger.context(),
            );
            if stored && let Some(settled) = settled.as_ref() {
                settled.record(crate::execution::TunePick::Persisted);
            }
        }
        #[cfg(not(persistence))]
        let _ = (key, results);
        self.unmarker.unmark();
    }

    #[cfg(not(target_family = "wasm"))]
    /// Stop the tune on a card found lost: answered pending, so the caller runs the first
    /// candidate that serves the problem and meets the loss in its own reads. The card is answered
    /// so for every key of this candidate list from now on, and the key is left to another card
    /// of the same hardware, which may be healthy.
    pub fn stop_on_lost_card(self, service: ServiceId) -> TuneCacheResult {
        self.cache.lock().track_lost(service);
        TuneCacheResult::Pending
    }
}
