//! Closing every HIP device before the HIP runtime tears itself down at exit.
//!
//! The HIP library frees its runtime in its destructors, which run at exit while
//! the process's other threads keep running. A server call that reaches HIP after
//! that point, from whichever thread still holds a client, touches the freed
//! runtime and crashes the process once its work is done: a remote server's
//! session closing as its client exits frees pinned host memory exactly then.
//!
//! Exit handlers run in the reverse order they were registered. So the handler
//! here is registered once HIP is loaded and the first device is set up, after
//! whatever HIP registers for itself on the way, and runs before HIP's own: it
//! closes every opened device ([`DeviceHandle::close`]), which lets each runner
//! finish its batch in flight, refuses every later call, and drops the server,
//! joining the threads that still call into HIP while HIP is there to answer.
//!
//! A close reaches every service on the device's runners, whichever runtime
//! registered it: runtimes number their devices alike, so the CPU runtime's
//! device 0 shares a runner with HIP's. At exit that is what is wanted.

use super::HipServer;
use cubecl_common::device::DeviceId;
use cubecl_common::device_handle::DeviceHandle;
use std::sync::{Mutex, MutexGuard, Once};
use std::vec::Vec;

/// The HIP devices this process opened, closed together when it exits.
pub(crate) struct OpenedDevices {
    devices: Vec<DeviceId>,
    /// Set once the exit handler runs, under the same lock as `devices`: a device
    /// opened after it would reach HIP while the runtime is being torn down.
    exiting: bool,
}

static OPENED: Mutex<OpenedDevices> = Mutex::new(OpenedDevices {
    devices: Vec::new(),
    exiting: false,
});

/// Registers the exit handler once, with the first device.
static HANDLER: Once = Once::new();

impl OpenedDevices {
    fn lock() -> MutexGuard<'static, Self> {
        OPENED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Close `device_id` at exit, before the HIP runtime is torn down. Called once
    /// the device is set up, so the handler is registered after HIP's own.
    ///
    /// # Panics
    ///
    /// When the process is already exiting: the device would be opened on a
    /// runtime that is going away.
    pub(crate) fn close_at_exit(device_id: DeviceId) {
        HANDLER.call_once(|| {
            // SAFETY: registering a plain `extern "C"` function with no arguments,
            // which is what `atexit` takes; it lives as long as the process.
            let status = unsafe { libc::atexit(close_opened_devices) };
            if status != 0 {
                log::warn!(
                    "no exit handler for the HIP devices: a call racing the exit may crash the process"
                );
            }
        });
        let mut opened = Self::lock();
        assert!(
            !opened.exiting,
            "a HIP device cannot be opened while the process exits"
        );
        if !opened.devices.contains(&device_id) {
            opened.devices.push(device_id);
        }
    }
}

/// The exit handler: every opened device, closed.
extern "C" fn close_opened_devices() {
    let devices = {
        let mut opened = OpenedDevices::lock();
        opened.exiting = true;
        core::mem::take(&mut opened.devices)
    };
    for device_id in devices {
        // A panic may not unwind out of an `extern "C"` function, and a device
        // that cannot be closed (exit called from its own runner) is left as it
        // was rather than aborting the exit.
        let closed = std::panic::catch_unwind(|| {
            DeviceHandle::<HipServer>::close(device_id);
        });
        if closed.is_err() {
            log::warn!("HIP device {device_id:?} was not closed before exit");
        }
    }
}
