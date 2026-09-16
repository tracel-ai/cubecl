use crossbeam_utils::CachePadded;
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// A park is a syscall, and the waits on this counter are usually short.
const SPINS_BEFORE_YIELD: u32 = 1_000;

const YIELDS_BEFORE_PARK: u32 = 64;

/// The client is not pinned, so a busy wait lands on the SMT sibling of a worker and slows the
/// plane that worker runs; parking gives that logical CPU back.
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

    pub fn add_done(&self) {
        let old = self.value.fetch_add(1, Ordering::Release);
        let new = old + 1;
        // Without this fence and the one in `wait_until`, store-buffer reordering lets both sides
        // read stale values and the last completion skips the only wake.
        std::sync::atomic::fence(Ordering::SeqCst);
        if new >= self.wake_at.load(Ordering::Relaxed) {
            let _guard = self.lock.lock().unwrap();
            self.condvar.notify_all();
        }
    }

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

        // `add_done` notifies under the lock, so a completion in this window is either seen by the
        // re-check or wakes the wait.
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
        // A CpuStream has one client thread, so there is never a second waiter to keep a target for.
        self.wake_at.store(u64::MAX, Ordering::Relaxed);
        drop(guard);
    }
}
