//! Closing services as the process exits.
//!
//! A native runtime frees itself in its libraries' exit teardown, which runs while the
//! process's other threads keep running, so a server call that reaches the runtime after that
//! point crashes the process once its work is done. The close has to run before any of that
//! teardown, and the teardown cannot be outrun by registering early or late: HIP's compiler
//! and queue code register their destructors lazily, on each first use, so some are always
//! registered after any handler of ours and run before it.
//!
//! What runs ahead of every exit handler, on Linux with glibc, is the exiting thread's
//! thread-local destructors. A guard armed on the main thread before `main` closes the
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
pub(crate) use handler::close_at_exit;

#[cfg(multi_threading)]
mod handler {
    use super::EXITING;
    use crate::device::{DeviceId, handle::ServiceCloseError};
    use alloc::vec::Vec;
    use core::any::TypeId;
    use core::sync::atomic::Ordering;
    use std::sync::Once;

    /// A service to close at exit: where it runs, and the close of its type.
    struct ServiceAtExit {
        device_id: DeviceId,
        type_id: TypeId,
        close: fn(DeviceId) -> Result<(), ServiceCloseError>,
    }

    static SERVICES: spin::Mutex<Vec<ServiceAtExit>> = spin::Mutex::new(Vec::new());

    static HANDLER: Once = Once::new();

    unsafe extern "C" {
        /// The C library's, which every target with threads and `std` links.
        fn atexit(handler: extern "C" fn()) -> core::ffi::c_int;
    }

    /// Closes the service `type_id` on `device_id`, with `close`, as the process exits.
    pub(crate) fn close_at_exit(
        device_id: DeviceId,
        type_id: TypeId,
        close: fn(DeviceId) -> Result<(), ServiceCloseError>,
    ) {
        HANDLER.call_once(|| {
            // SAFETY: `close_on_exit` is a plain function taking nothing, which is what
            // `atexit` takes, and it lives as long as the process.
            if unsafe { atexit(close_on_exit) } != 0 {
                log::warn!(
                    "no exit handler for the device services: an exit from a thread other than \
                     the main one may crash the process"
                );
            }
        });
        let mut services = SERVICES.lock();
        let known = services
            .iter()
            .any(|service| service.device_id == device_id && service.type_id == type_id);
        if !known {
            services.push(ServiceAtExit {
                device_id,
                type_id,
                close,
            });
        }
    }

    extern "C" fn close_on_exit() {
        close_services();
    }

    /// No service starts from here on, and every one that asked is closed. The list is taken,
    /// so whichever of the guard and the handler runs second finds nothing to close.
    fn close_services() {
        EXITING.store(true, Ordering::Release);
        let services = core::mem::take(&mut *SERVICES.lock());
        for service in services {
            if let Err(error) = (service.close)(service.device_id) {
                log::warn!(
                    "a service on device {:?} was not closed before exit: {error:?}",
                    service.device_id
                );
            }
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    mod main_thread {
        /// Closes the services when the main thread's thread-locals are destroyed, which glibc
        /// does before it runs any exit handler.
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
            // SAFETY: both only read the calling process's and thread's ids.
            let main_thread = unsafe { libc::getpid() == libc::gettid() };
            if main_thread {
                GUARD.with(|_| ());
            }
        }

        // Before `main`, on the main thread, so the guard is armed whichever thread opens a
        // device later.
        #[used]
        #[unsafe(link_section = ".init_array")]
        static ARM: extern "C" fn() = arm;
    }
}
