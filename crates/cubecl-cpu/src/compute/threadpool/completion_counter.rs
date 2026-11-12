use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::Thread,
};

use crossbeam_utils::CachePadded;

/// The units a stream's launches have completed, and the client parked until they reach a step.
///
/// A waiting client parks instead of polling: it is not pinned, so a poll lands on the SMT
/// sibling of a busy worker and takes half of that core from the unit the launch waits for.
pub struct CompletionCounter {
    completed: CachePadded<AtomicU64>,
    /// The step the parked client waits for. Zero matches no unit, as the count is at least one
    /// after any completion, and a passed step never matches again since the count only grows.
    awaited: AtomicU64,
    waiter: Mutex<Option<Thread>>,
}

impl Default for CompletionCounter {
    fn default() -> Self {
        Self {
            completed: CachePadded::new(AtomicU64::new(0)),
            awaited: AtomicU64::new(0),
            waiter: Mutex::new(None),
        }
    }
}

/// How many polls a client spins before parking, so a launch done within a few microseconds is
/// not delayed by a park and unpark round trip.
const SPINS_BEFORE_PARK: u32 = 1_000;

impl CompletionCounter {
    /// The units completed so far, ordered after their writes.
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Records one completed unit, waking the client only if it waits for this exact step.
    pub fn complete_unit(&self) {
        // SeqCst pairs with `wait_until`: either the client sees this unit, or this unit sees
        // the step the client registered.
        let completed = self.completed.fetch_add(1, Ordering::SeqCst) + 1;
        if completed == self.awaited.load(Ordering::SeqCst)
            && let Some(waiter) = self.waiter.lock().unwrap().as_ref()
        {
            waiter.unpark();
        }
    }

    /// Blocks until `step` units have completed.
    ///
    /// One client waits at a time, which `LaunchProgress::wait` guarantees by taking `&mut self`.
    pub fn wait_until(&self, step: u64) {
        for _ in 0..SPINS_BEFORE_PARK {
            if self.completed() >= step {
                return;
            }
            core::hint::spin_loop();
        }
        *self.waiter.lock().unwrap() = Some(std::thread::current());
        self.awaited.store(step, Ordering::SeqCst);
        // SeqCst, unlike the spin's Acquire: this load must order after the `awaited` store.
        while self.completed.load(Ordering::SeqCst) < step {
            std::thread::park();
        }
    }
}

/// A stream's side of a [`CompletionCounter`]: the units it has issued, and the one waiter.
///
/// Waiting takes `&mut self`, so the counter never has two clients parked at once.
#[derive(Default)]
pub struct LaunchProgress {
    counter: Arc<CompletionCounter>,
    issued: u64,
}

impl LaunchProgress {
    /// Counts `units` as issued and returns the completed count before them, which their tasks
    /// wait for before running.
    pub fn issue(&mut self, units: u64) -> u64 {
        let first = self.issued;
        self.issued += units;
        first
    }

    /// The counter the issued tasks complete their units on.
    pub fn counter(&self) -> &Arc<CompletionCounter> {
        &self.counter
    }

    /// Blocks until every issued unit has completed.
    pub fn wait(&mut self) {
        self.counter.wait_until(self.issued);
    }
}

#[cfg(test)]
mod tests {
    use super::CompletionCounter;
    use std::{sync::Arc, thread, time::Duration};

    /// Long enough for a waiter to finish its spin and park.
    const PARK_DELAY: Duration = Duration::from_millis(50);

    #[test]
    fn a_waiter_parked_before_the_last_unit_wakes() {
        let counter = Arc::new(CompletionCounter::default());
        let waiter = thread::spawn({
            let counter = counter.clone();
            move || counter.wait_until(2)
        });
        counter.complete_unit();
        thread::sleep(PARK_DELAY);
        counter.complete_unit();
        waiter.join().unwrap();
    }

    #[test]
    fn a_waiter_arriving_after_the_last_unit_returns_at_once() {
        let counter = CompletionCounter::default();
        counter.complete_unit();
        counter.complete_unit();
        counter.wait_until(2);
        counter.wait_until(0);
    }

    #[test]
    fn a_spurious_wakeup_parks_again() {
        let counter = Arc::new(CompletionCounter::default());
        let waiter = thread::spawn({
            let counter = counter.clone();
            move || counter.wait_until(2)
        });
        counter.complete_unit();
        thread::sleep(PARK_DELAY);
        waiter.thread().unpark();
        thread::sleep(PARK_DELAY);
        assert!(!waiter.is_finished());
        counter.complete_unit();
        waiter.join().unwrap();
    }

    #[test]
    fn successive_waits_each_wake() {
        let counter = Arc::new(CompletionCounter::default());
        for step in 1..=3 {
            let waiter = thread::spawn({
                let counter = counter.clone();
                move || counter.wait_until(step)
            });
            thread::sleep(PARK_DELAY);
            counter.complete_unit();
            waiter.join().unwrap();
        }
    }

    #[test]
    fn racing_waits_never_lose_a_wakeup() {
        const UNITS: u64 = 10_000;
        let counter = Arc::new(CompletionCounter::default());
        let producer = thread::spawn({
            let counter = counter.clone();
            move || {
                for _ in 0..UNITS {
                    counter.complete_unit();
                }
            }
        });
        // Each wait is bounded by the producer, so a lost wakeup hangs the join below.
        for step in 1..=UNITS {
            counter.wait_until(step);
        }
        producer.join().unwrap();
    }
}
