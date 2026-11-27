use alloc::string::String;
use alloc::vec::Vec;
use cubecl_environment::sync::Mutex;
use core::time::Duration;

use crate::tune::cache::record::{Answer, TuneRecorder};
use crate::tune::cache::tracker::TuneTracker;
use crate::tune::{
    AutotuneKeyable, AutotuneLogContext, AutotuneResult, Bounds, PersistentCacheValue,
};

/// Persists what one tune answered: its winner in the table, and the record of how it went.
pub struct TunePersister<K: AutotuneKeyable> {
    checksum: String,
    limit: Option<Duration>,
    bounds: Option<Bounds>,
    recorder: TuneRecorder<K>,
}

impl<K: AutotuneKeyable> TunePersister<K> {
    /// A persister of the tune of the candidate list `checksum` names, measured against `bounds`
    /// and their `limit`, and recorded by `recorder`.
    pub fn new(
        checksum: String,
        limit: Option<Duration>,
        bounds: Option<Bounds>,
        recorder: TuneRecorder<K>,
    ) -> Self {
        Self {
            checksum,
            limit,
            bounds,
            recorder,
        }
    }

    /// Persist `fastest_index` as the answer to `key`, with every candidate's `results`: in the
    /// table of the tracker `cache` holds unless it is `unmeasured`, locking it only for that
    /// insert, and in the tune's record either way, with no lock held. Answers whether the table
    /// took it.
    ///
    /// An unmeasured decision is a guess made to keep the device thread alive, and the failures
    /// that produce one — a profiling hiccup, timestamp query sets on a busy stream — are
    /// transient. Persisting it would freeze the guess into every later process and never measure
    /// the key again; letting it expire with this one costs a re-tune and buys a real measurement.
    pub fn persist(
        self,
        cache: &Mutex<TuneTracker<K>>,
        key: K,
        results: Vec<AutotuneResult>,
        fastest_index: usize,
        unmeasured: bool,
        log_context: Option<&AutotuneLogContext>,
    ) -> bool {
        let value = PersistentCacheValue {
            decided_index: fastest_index,
            results,
            bounds: self.bounds,
            limit: self.limit,
        };
        let stored = !unmeasured && cache.lock().store_answer(key, self.checksum, value);
        self.recorder.record_answer(
            Answer {
                winner: fastest_index,
                stored,
            },
            log_context,
        );
        stored
    }
}
