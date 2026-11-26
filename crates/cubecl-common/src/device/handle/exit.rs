//! Closing services as the process exits.
//!
//! A native runtime frees itself in its libraries' exit teardown, which runs while the
//! process's other threads keep running, so a server call that reaches the runtime after that
//! point crashes the process once its work is done. The close has to run before any of that
//! teardown, and the teardown cannot be outrun by registering early or late: HIP's compiler
//! and queue code register their destructors lazily, on each first use, so some are always
//! registered after any handler of ours and run before it.
//!
//! What runs ahead of every exit handler, on Linux with glibc and on macOS, is the exiting
//! thread's thread-local destructors. A guard armed on the main thread before `main` closes the
//! services when the main thread exits, which is how a process ends when `main` returns or
//! calls [`std::process::exit`]. An exit handler covers an exit from another thread and the
//! other platforms, ahead of whatever teardown was registered before it.

use core::sync::atomic::{AtomicBool, Ordering};

/// Set once the services are being closed; no service is started after it.
static EXITING: AtomicBool = AtomicBool::new(false);

/// Whether the process is exiting, past the point a service is started.
pub(crate) fn is_exiting() -> bool {
    EXITING.load(Ordering::Acquire)
}

#[cfg(multi_threading)]
pub use handler::ExitGuard;
#[cfg(multi_threading)]
pub(crate) use handler::{close_at_exit, forget_device};

#[cfg(multi_threading)]
mod handler {
    use super::EXITING;
    use crate::device::{DeviceId, handle::ServiceCloseError};
    use alloc::vec::Vec;
    use core::any::TypeId;
    use core::sync::atomic::Ordering;
    use core::time::Duration;
    use std::time::Instant;

    /// The close of a service type: closes it on a device, waiting up to the given time for its
    /// runner.
    type Close = fn(DeviceId, Duration) -> Result<(), ServiceCloseError>;

    /// A service to close at exit: where it runs, and the close of its type.
    #[derive(Debug)]
    struct ServiceAtExit {
        device_id: DeviceId,
        type_id: TypeId,
        close: Close,
    }

    /// How long the exit waits, for all the services together, for their runners to reach the
    /// closes. A runner stuck in a call delays the exit by this much at most, however many
    /// services it holds.
    const EXIT_CLOSE_TIMEOUT: Duration = Duration::from_secs(30);

    /// How long the exit waits for the [`ExitGuard`]s still held once the services are closed,
    /// however long their closes took: the runtime's teardown must not overlap a guarded call.
    const EXIT_GUARD_TIMEOUT: Duration = Duration::from_secs(5);

    static SERVICES: spin::Mutex<Vec<ServiceAtExit>> = spin::Mutex::new(Vec::new());

    /// How many [`ExitGuard`]s are held, and the exit's wait for none to be.
    static EXIT_GUARDS: (std::sync::Mutex<usize>, std::sync::Condvar) =
        (std::sync::Mutex::new(0), std::sync::Condvar::new());

    /// Held across a runtime call made on a thread outside every service, such as releasing memory
    /// where its last owner drops it. The exit waits for the guards held before the runtime's
    /// teardown, as it does for the services, and like theirs the wait is bounded: a call stuck past
    /// it can still overlap the teardown.
    #[derive(Debug)]
    pub struct ExitGuard {
        _private: (),
    }

    impl ExitGuard {
        /// Holds the exit back until dropped, or returns `None` once the process is exiting, when
        /// the runtime may already be tearing down and the call must be skipped.
        pub fn acquire() -> Option<Self> {
            let mut calls = lock_exit_guards();
            // Read under the lock the exit takes after setting it: either this call is counted
            // before the exit waits, or it sees the exit.
            if EXITING.load(Ordering::Acquire) {
                return None;
            }
            *calls += 1;
            Some(Self { _private: () })
        }
    }

    impl Drop for ExitGuard {
        fn drop(&mut self) {
            let mut calls = lock_exit_guards();
            *calls -= 1;
            if *calls == 0 {
                EXIT_GUARDS.1.notify_all();
            }
        }
    }

    fn lock_exit_guards() -> std::sync::MutexGuard<'static, usize> {
        EXIT_GUARDS
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Waits until no [`ExitGuard`] is held, or until `deadline`.
    fn wait_exit_guards(deadline: Instant) {
        let calls = lock_exit_guards();
        let timeout = deadline.saturating_duration_since(Instant::now());
        let (calls, _) = EXIT_GUARDS
            .1
            .wait_timeout_while(calls, timeout, |calls| *calls > 0)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *calls > 0 {
            log::warn!("{} runtime calls were still running at exit", *calls);
        }
    }

    #[cfg(not(miri))]
    unsafe extern "C" {
        /// The C library's, which every target with threads and `std` links.
        fn atexit(handler: extern "C" fn()) -> core::ffi::c_int;
    }

    /// Closes the service `type_id` on `device_id`, with `close`, as the process exits.
    ///
    /// A service that started while the exit was already closing the others, past the check
    /// that refuses to start one, is closed right away: the exit has taken its list already.
    pub(crate) fn close_at_exit(device_id: DeviceId, type_id: TypeId, close: Close) {
        let mut services = SERVICES.lock();
        // Read under the lock `close_services` sets it under: either this service is in the
        // list the exit takes, or the exit has started and this one closes itself.
        if EXITING.load(Ordering::Acquire) {
            drop(services);
            close_service(device_id, close, EXIT_CLOSE_TIMEOUT);
            return;
        }
        let known = services
            .iter()
            .any(|service| service.device_id == device_id && service.type_id == type_id);
        if known {
            return;
        }
        let new_type = services.iter().all(|service| service.type_id != type_id);
        services.push(ServiceAtExit {
            device_id,
            type_id,
            close,
        });
        drop(services);
        // A runtime loads its libraries as its first service starts, so a handler per service
        // type keeps the newest-first order ahead of their teardown without one per device.
        if new_type {
            register_exit_handler();
        }
    }

    /// Forgets the services of `device_id`, which were shut down: none is left to close at
    /// exit.
    pub(crate) fn forget_device(device_id: DeviceId) {
        SERVICES
            .lock()
            .retain(|service| service.device_id != device_id);
    }

    /// Miri cannot call `atexit`, so under it no exit handler closes the services.
    #[cfg(miri)]
    fn register_exit_handler() {}

    #[cfg(not(miri))]
    fn register_exit_handler() {
        // Registered again for each new service type: exit handlers run newest first, so the
        // latest one runs ahead of the teardown of every library loaded before that service
        // started, whichever runtime started first. Only the first to run finds services to
        // close.
        // SAFETY: `close_on_exit` is a plain function taking nothing, which is what `atexit`
        // takes, and it lives as long as the process.
        if unsafe { atexit(close_on_exit) } != 0 {
            log::warn!(
                "no exit handler for the device services: an exit from a thread other than the \
                 main one may crash the process"
            );
        }
    }

    #[cfg(not(miri))]
    extern "C" fn close_on_exit() {
        close_services();
    }

    /// No service starts from here on, and every one that asked is closed, all within
    /// [`EXIT_CLOSE_TIMEOUT`], then the exit guards are waited for, at least
    /// [`EXIT_GUARD_TIMEOUT`]. Whichever of the guard and the handler runs second returns at once.
    fn close_services() {
        let services = {
            let mut services = SERVICES.lock();
            if EXITING.swap(true, Ordering::AcqRel) {
                return;
            }
            core::mem::take(&mut *services)
        };
        let deadline = Instant::now() + EXIT_CLOSE_TIMEOUT;
        for service in services {
            let remaining = deadline.saturating_duration_since(Instant::now());
            close_service(service.device_id, service.close, remaining);
        }
        wait_exit_guards(deadline.max(Instant::now() + EXIT_GUARD_TIMEOUT));
    }

    /// Closes one service with `close`, logging a close that failed or ran out of time.
    fn close_service(device_id: DeviceId, close: Close, timeout: Duration) {
        if let Err(error) = close(device_id, timeout) {
            log::warn!("a service on device {device_id:?} was not closed before exit: {error:?}");
        }
    }

    #[cfg(any(all(target_os = "linux", target_env = "gnu"), target_os = "macos"))]
    mod main_thread {
        /// Closes the services when the main thread's thread-locals are destroyed, which glibc
        /// and macOS do before they run any exit handler: past that point a close can no longer
        /// reach `std::thread::current()`.
        struct ExitGuard;

        impl Drop for ExitGuard {
            fn drop(&mut self) {
                super::close_services();
            }
        }

        std::thread_local! {
            static GUARD: ExitGuard = const { ExitGuard };
        }

        /// Arms the guard on the main thread only: another thread's thread-locals are destroyed
        /// when that thread ends, with the process still running. A library loaded later from
        /// another thread has no guard, and relies on the exit handler.
        extern "C" fn arm() {
            if is_main_thread() {
                GUARD.with(|_| ());
            }
        }

        #[cfg(target_os = "linux")]
        fn is_main_thread() -> bool {
            // SAFETY: both only read the calling process's and thread's ids.
            unsafe { libc::getpid() == libc::gettid() }
        }

        #[cfg(target_os = "macos")]
        fn is_main_thread() -> bool {
            // SAFETY: only reads whether the calling thread is the process's first.
            unsafe { libc::pthread_main_np() != 0 }
        }

        // Before `main`, on the main thread, so the guard is armed whichever thread opens a
        // device later.
        #[used]
        #[cfg_attr(target_os = "linux", unsafe(link_section = ".init_array"))]
        #[cfg_attr(target_os = "macos", unsafe(link_section = "__DATA,__mod_init_func"))]
        static ARM: extern "C" fn() = arm;
    }
}
