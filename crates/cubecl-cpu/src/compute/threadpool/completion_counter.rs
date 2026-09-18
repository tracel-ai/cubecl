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
        // This fence and the one in `wait_until` must both be SeqCst: a weaker pair lets store-buffer
        // reordering hide each side's write from the other and skip the only wake.
        std::sync::atomic::fence(Ordering::SeqCst);
        if new >= self.wake_at.load(Ordering::Relaxed) {
            let _guard = self.lock.lock().unwrap();
            self.condvar.notify_one();
        }
    }

    /// At most one thread may wait on a counter: when the first waiter returns it resets the target
    /// a second one registered, and that one is never woken.
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

        // `add_done` takes the lock to notify, so a completion after the check under the lock cannot
        // notify before `wait` releases it.
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
        self.wake_at.store(u64::MAX, Ordering::Relaxed);
        drop(guard);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_reached_target_returns_at_once() {
        let counter = CompletionCounter::new();
        counter.wait_until(0);
        for _ in 0..3 {
            counter.add_done();
        }
        counter.wait_until(3);
        assert_eq!(counter.load(), 3);
    }

    #[test]
    fn a_parked_wait_wakes_on_the_last_completion() {
        let counter = CompletionCounter::new();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(50));
                for _ in 0..1_000 {
                    counter.add_done();
                }
            });
            counter.wait_until(1_000);
        });
        assert_eq!(counter.load(), 1_000);
    }

    #[test]
    fn no_completion_timing_loses_the_wake() {
        for round in 0..200u64 {
            let counter = CompletionCounter::new();
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    for unit in 0..8u64 {
                        std::thread::sleep(Duration::from_micros((round * 7 + unit * 13) % 400));
                        counter.add_done();
                    }
                });
                counter.wait_until(8);
            });
        }
    }
}
