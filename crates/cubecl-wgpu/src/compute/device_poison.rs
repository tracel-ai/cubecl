//! Monitoring wgpu device poisoning.
//!
//! wgpu bounds-checks buffer accesses, so a kernel cannot fault the device the
//! way an illegal address faults a CUDA context. What remains is losing the
//! device outright — a driver reset, a timeout, a GPU that went away — which
//! wgpu reports once, through the device-lost callback, and never on the
//! operations that follow.
//!
//! The device is marked poisoned here when the callback fires, shared by every
//! stream on the device, and every sync point checks it

use core::fmt::Debug;
use core::time::Duration;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::channel::Sender;
use cubecl_server::driver::DevicePoison;
use cubecl_server::server::ServerError;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

/// How long one wait on the device blocks before it looks again.
const WAIT_SLICE: Duration = Duration::from_secs(1);

/// How long a wait goes without progress before it probes the device with an empty submission.
/// A driver that kills a hung context can leave its fences unsignaled for good, and report the
/// loss only to the next submission.
const PROBE_AFTER: Duration = Duration::from_secs(5);

/// Holds whether a wgpu device is poisoned. Shared by every stream on the device.
#[derive(Clone, Debug, Default)]
pub struct PoisonWatch {
    shared: Arc<Shared>,
}

#[derive(Debug, Default)]
struct Shared {
    reason: OnceLock<String>,
    waiters: Mutex<Vec<Box<dyn Waiter>>>,
}

impl PoisonWatch {
    /// Start watching `device`, installing the callbacks that poison it once wgpu loses it.
    ///
    /// Installs an uncaptured-error handler too. wgpu's default one panics,
    /// which on a healthy device is what surfaces a validation bug, so it
    /// still does; but once the device is poisoned every call on it errors.
    pub fn watch(device: &wgpu::Device) -> Self {
        let poison = Self::default();

        let lost = poison.clone();
        device.set_device_lost_callback(move |kind, message| {
            log::error!("the wgpu device was lost ({kind:?}): {message}");
            lost.poison(format!("{kind:?}: {message}"));
        });

        let watched = poison.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            if watched.is_poisoned() {
                log::debug!("error on a poisoned wgpu device: {error}");
            } else {
                panic!("wgpu error: {error}");
            }
        }));

        poison
    }

    /// Whether the device is poisoned.
    pub fn is_poisoned(&self) -> bool {
        self.shared.reason.get().is_some()
    }

    /// Returns a [ServerError::DevicePoisoned](ServerError::DevicePoisoned) once the device is poisoned.
    pub fn check(&self) -> Result<(), ServerError> {
        match self.shared.reason.get() {
            Some(reason) => Err(DevicePoison::new(reason.clone()).into()),
            None => Ok(()),
        }
    }

    /// Closes `sender` once the device is lost, or now if it already is, so whatever awaits its
    /// receiver wakes to find the device poisoned. wgpu completes the maps and work-done
    /// callbacks of a lost device only once its queue empties, which a hung one never does.
    pub fn wake_on_loss<T: Send + 'static>(&self, sender: Sender<T>) {
        let mut waiters = self.waiters();
        if self.is_poisoned() {
            sender.close();
            return;
        }
        waiters.retain(|waiter| !waiter.is_closed());
        waiters.push(Box::new(sender));
    }

    /// Waits until `submission` completes, or everything submitted when `None`, and fails once
    /// the device is lost instead of waiting on work that will never complete.
    pub fn wait(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        submission: Option<wgpu::SubmissionIndex>,
    ) -> Result<(), ServerError> {
        let mut stalled = Duration::ZERO;
        loop {
            self.check()?;
            let polled = device.poll(wgpu::PollType::Wait {
                submission_index: submission.clone(),
                timeout: Some(WAIT_SLICE),
            });
            match polled {
                Ok(_) => return Ok(()),
                Err(wgpu::PollError::Timeout) => {
                    stalled += WAIT_SLICE;
                    if stalled >= PROBE_AFTER {
                        queue.submit(core::iter::empty());
                        stalled = Duration::ZERO;
                    }
                }
                Err(err) => {
                    return Err(ServerError::Generic {
                        reason: format!("wgpu: waiting on the device failed ({err})"),
                        backtrace: BackTrace::capture(),
                    });
                }
            }
        }
    }

    fn poison(&self, reason: String) {
        let _ = self.shared.reason.set(reason);
        for waiter in self.waiters().drain(..) {
            waiter.close();
        }
    }

    fn waiters(&self) -> MutexGuard<'_, Vec<Box<dyn Waiter>>> {
        self.shared
            .waiters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// A channel a pending wait listens on.
trait Waiter: Send + Debug {
    /// Whether the wait is over, so there is nothing left to wake.
    fn is_closed(&self) -> bool;
    fn close(&self);
}

impl<T: Send> Waiter for Sender<T> {
    fn is_closed(&self) -> bool {
        Sender::is_closed(self)
    }

    fn close(&self) {
        Sender::close(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_environment::future::{block_on, channel::bounded};

    #[test]
    fn a_wait_pending_when_the_device_is_lost_wakes_poisoned() {
        let watch = PoisonWatch::default();
        let (sender, receiver) = bounded::<()>(1);
        watch.wake_on_loss(sender);

        watch.poison("lost".into());

        assert!(block_on(receiver.recv()).is_err());
        assert!(watch.check().is_err());
    }

    #[test]
    fn a_wait_started_after_the_loss_wakes_at_once() {
        let watch = PoisonWatch::default();
        watch.poison("lost".into());
        let (sender, receiver) = bounded::<()>(1);

        watch.wake_on_loss(sender);

        assert!(block_on(receiver.recv()).is_err());
    }
}
