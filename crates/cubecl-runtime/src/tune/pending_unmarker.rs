use alloc::sync::Arc;
use cubecl_environment::sync::Mutex;

use super::AutotuneKeyable;
use super::tune_cache::TuneTracker;

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
