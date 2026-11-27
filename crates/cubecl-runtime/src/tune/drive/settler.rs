use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(not(target_family = "wasm"))]
use cubecl_common::device::ServiceId;
use cubecl_environment::sync::Mutex;

use crate::config::Logger;
use crate::execution::{TuneOutcome, TuneRegistration};
#[cfg(not(target_family = "wasm"))]
use crate::tune::cache::tracker::TuneAnswer;
use crate::tune::cache::tracker::TuneTracker;
use crate::tune::drive::pending_unmarker::PendingUnmarker;
use crate::tune::log::tune_logger::TuneLogger;
use crate::tune::{AutotuneKeyable, AutotuneResult};

/// Settles the key one tune answered: reports the answer, keeps it in memory, settles the tune's
/// registration, and, where tunes persist, stores and records it.
pub struct TuneSettler<K: AutotuneKeyable> {
    tracker: Arc<Mutex<TuneTracker<K>>>,
    logger: Arc<Mutex<Logger>>,
    /// Where the tune settles once its pick is committed.
    registration: Option<TuneRegistration>,
    unmarker: PendingUnmarker<K>,
    #[cfg(persistence)]
    persister: crate::tune::drive::persister::TunePersister<K>,
}

impl<K: AutotuneKeyable> TuneSettler<K> {
    pub fn new(
        tracker: Arc<Mutex<TuneTracker<K>>>,
        logger: Arc<Mutex<Logger>>,
        registration: Option<TuneRegistration>,
        unmarker: PendingUnmarker<K>,
        #[cfg(persistence)] persister: crate::tune::drive::persister::TunePersister<K>,
    ) -> Self {
        Self {
            tracker,
            logger,
            registration,
            unmarker,
            #[cfg(persistence)]
            persister,
        }
    }

    /// Settle `key` on `decided_index`, which is `unmeasured` where it was picked with nothing
    /// measured, `results` holding every candidate's result by tunable index.
    pub fn settle(
        self,
        key: K,
        results: Vec<AutotuneResult>,
        decided_index: usize,
        unmeasured: bool,
        tune_logger: &TuneLogger,
    ) {
        // Ranked for the report and the table: measured ones first, fastest first.
        let mut results = results;
        results.sort_by_key(AutotuneResult::score);
        tune_logger.log_result(&mut self.logger.lock(), &key, &results, decided_index);
        // In-memory regardless: without it this key re-tunes on every call, and a tune that
        // measured nothing would keep failing the same way.
        self.tracker.lock().track_answer(key.clone(), decided_index);
        // Where the tune was registered, whether or not that override is still open.
        let outcome = if unmeasured {
            TuneOutcome::Failed
        } else {
            TuneOutcome::Measured
        };
        #[cfg_attr(not(persistence), allow(unused_variables))]
        let settled = self
            .registration
            .map(|registration| registration.settle(outcome));

        #[cfg(persistence)]
        {
            let stored = self.persister.persist(
                &self.tracker,
                key,
                results,
                decided_index,
                unmeasured,
                tune_logger.as_context(),
            );
            if stored && let Some(settled) = settled.as_ref() {
                settled.record(crate::execution::TunePick::Persisted);
            }
        }
        #[cfg(not(persistence))]
        let _ = (key, results);
        self.unmarker.unmark();
    }

    /// Stop the tune of `key` on the lost card `service`, tracked lost, with nothing decided:
    /// `tune_logger` reports the cross-checks it collected, and the answer is pending, with the
    /// tune's [unmarker](PendingUnmarker) and registration dropped.
    #[cfg(not(target_family = "wasm"))]
    pub fn stop_on_lost_card(
        self,
        service: ServiceId,
        key: &K,
        tune_logger: &TuneLogger,
    ) -> TuneAnswer {
        self.tracker.lock().track_lost(service);
        tune_logger.report_stopped(key);
        TuneAnswer::Pending
    }
}
