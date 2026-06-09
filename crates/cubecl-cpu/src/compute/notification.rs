use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    thread::{self, Thread},
};

const MAX_SPIN_ITERATION: u16 = 8192;

#[derive(Clone)]
pub struct Notification {
    ready: Arc<AtomicBool>,
    current_thread: Thread,
}

impl Notification {
    #[inline]
    pub fn new() -> Self {
        let ready = Arc::new(AtomicBool::new(false));
        let current_thread = thread::current();
        Self {
            ready,
            current_thread,
        }
    }

    #[inline]
    pub fn send(&self) {
        self.ready.store(true, Ordering::Release);
        self.current_thread.unpark();
    }

    #[inline]
    pub fn wait(&self) {
        for _ in 0..MAX_SPIN_ITERATION {
            if self.ready.load(Ordering::Acquire) {
                return;
            }
            std::hint::spin_loop();
        }

        while !self.ready.load(Ordering::Acquire) {
            std::thread::park();
        }
    }
}

#[derive(Clone)]
pub struct Notifications {
    remaining: Arc<AtomicU32>,
    current_thread: Thread,
}

impl Notifications {
    #[inline]
    pub fn new(nb_notification: u32) -> Self {
        let remaining = Arc::new(AtomicU32::new(nb_notification));
        let current_thread = thread::current();
        Self {
            remaining,
            current_thread,
        }
    }

    #[inline]
    pub fn send(&self) {
        let value = self.remaining.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(value > 0); // Send should only be called nb_notification time
        if value == 1 {
            self.current_thread.unpark();
        }
    }

    #[inline]
    pub fn wait(&self) {
        for _ in 0..MAX_SPIN_ITERATION {
            if self.remaining.load(Ordering::Acquire) == 0 {
                return;
            }
            std::hint::spin_loop();
        }

        while self.remaining.load(Ordering::Acquire) != 0 {
            std::thread::park();
        }
    }
}
