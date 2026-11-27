use alloc::sync::Arc;
use cubecl_environment::sync::Mutex;

use crate::tune::AutotuneKeyable;
use crate::tune::cache::tracker::TuneTracker;

/// Unmarks a key [tracked pending](TuneTracker::track_pending) for the tune holding this. Dropped
/// before the tune settled the key, by a tune that unwinds or stops on a lost device, it unmarks
/// the key so the next caller tunes it rather than every device of that hardware running a
/// fallback for good.
///
/// # Assumptions
///
/// - It drops while no guard of the tracker's lock is held on its thread: its drop takes the lock.
pub struct PendingUnmarker<K: AutotuneKeyable> {
    cache: Arc<Mutex<TuneTracker<K>>>,
    key: K,
    /// The tracker's generation when marked: once a switch reset it, the key may be a newer
    /// tune's.
    #[cfg(persistence)]
    generation: u32,
}

impl<K: AutotuneKeyable> PendingUnmarker<K> {
    /// An unmarker of `key`, which `tracker`, the tracker `cache` holds, just tracked pending.
    pub fn new(cache: Arc<Mutex<TuneTracker<K>>>, tracker: &TuneTracker<K>, key: K) -> Self {
        #[cfg(not(persistence))]
        let _ = tracker;
        Self {
            cache,
            key,
            #[cfg(persistence)]
            generation: tracker.table().generation,
        }
    }

    /// Unmark the key now: a no-op once the tune settled it.
    pub fn unmark(self) {
        core::mem::drop(self);
    }
}

impl<K: AutotuneKeyable> Drop for PendingUnmarker<K> {
    fn drop(&mut self) {
        let mut cache = self.cache.lock();
        #[cfg(persistence)]
        if cache.table().generation != self.generation {
            return;
        }
        cache.untrack_pending(&self.key);
    }
}

#[cfg(all(test, persistence))]
mod tests {
    use super::*;
    use crate::tune::cache::table::TuneTable;
    use crate::tune::cache::tracker::TuneAnswer;
    use alloc::string::{String, ToString};

    fn tracker(generation: u32) -> TuneTracker<String> {
        let mut tracker = TuneTracker::with_store(
            None,
            TuneTable {
                namespace: "pending-unmarker".into(),
                generation,
            },
        );
        tracker.track_pending("key".to_string());
        tracker
    }

    fn tracked() -> Arc<Mutex<TuneTracker<String>>> {
        Arc::new(Mutex::new(tracker(0)))
    }

    #[test]
    fn a_dropped_unmarker_leaves_the_key_to_the_next_caller() {
        let cache = tracked();
        let unmarker = PendingUnmarker::new(cache.clone(), &cache.lock(), "key".to_string());

        unmarker.unmark();

        assert!(matches!(
            cache.lock().fastest(&"key".to_string()),
            TuneAnswer::Miss
        ));
    }

    #[test]
    fn an_unmarker_of_an_earlier_generation_leaves_the_key() {
        // Marked under generation 1, while the tracker was reset to generation 0 since.
        let cache = tracked();
        let unmarker = PendingUnmarker::new(cache.clone(), &tracker(1), "key".to_string());

        unmarker.unmark();

        assert!(matches!(
            cache.lock().fastest(&"key".to_string()),
            TuneAnswer::Pending
        ));
    }
}
