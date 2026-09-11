use crossbeam_utils::CachePadded;
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// How long a wait spins before it moves to yielding. A park is a syscall,
/// and the waits on this counter are usually short.
const SPINS_BEFORE_YIELD: u32 = 1_000;

/// How long a wait yields before it parks.
const YIELDS_BEFORE_PARK: u32 = 64;

/// A unit-completion counter the pool advances and the client waits on.
///
/// The client parks on a condvar rather than polling, so waiting gives back
/// the logical CPU the pool owns instead of spinning on it.
pub struct CompletionCounter {
    value: CachePadded<AtomicU64>,
    wake_at: AtomicU64,
    lock: Mutex<()>,
    condvar: Condvar,
}

impl Default for CompletionCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionCounter {
    pub fn new() -> Self {
        Self {
            value: CachePadded::new(AtomicU64::new(0)),
            wake_at: AtomicU64::new(u64::MAX),
            lock: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }

    pub fn load(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Advances the counter by one completed unit and wakes the waiter if
    /// this completion crosses its target.
    ///
    /// The target is read before the lock is taken, so a counter whose
    /// waiter is not yet due costs one relaxed load and a compare, nothing
    /// more.
    pub fn add_done(&self) {
        let old = self.value.fetch_add(1, Ordering::Release);
        let new = old + 1;
        // Pairs with the fence in `wait_until`: without it, this load and
        // the waiter's `wake_at` write can both see stale values (the
        // store-buffer reordering), and the last unit's completion skips
        // the only wake.
        std::sync::atomic::fence(Ordering::SeqCst);
        if new >= self.wake_at.load(Ordering::Relaxed) {
            let _guard = self.lock.lock().unwrap();
            self.condvar.notify_all();
        }
    }

    /// Blocks the caller until the counter reaches `target`.
    pub fn wait_until(&self, target: u64) {
        let mut spins = 0u32;
        while self.load() < target && spins < SPINS_BEFORE_YIELD {
            spins += 1;
            std::hint::spin_loop();
        }
        if self.load() >= target {
            return;
        }

        let mut yields = 0u32;
        while self.load() < target && yields < YIELDS_BEFORE_PARK {
            yields += 1;
            std::thread::yield_now();
        }
        if self.load() >= target {
            return;
        }

        // Registering before the fence, and re-checking under the lock, is
        // what keeps a completion landing in this exact window from being
        // missed: `add_done` also notifies under the lock, so it either
        // runs before this re-check (and is seen) or after `wait` has
        // released it (and wakes it).
        self.wake_at.fetch_min(target, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);
        if self.load() >= target {
            self.wake_at.store(u64::MAX, Ordering::Relaxed);
            return;
        }
        let mut guard = self.lock.lock().unwrap();
        while self.load() < target {
            guard = self.condvar.wait(guard).unwrap();
        }
        // A CpuStream is owned by a single client thread, so a counter
        // never has more than one waiter to reset here.
        self.wake_at.store(u64::MAX, Ordering::Relaxed);
        drop(guard);
    }
}
