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
#[cfg(not(target_family = "wasm"))]
use core::time::Duration;
#[cfg(not(target_family = "wasm"))]
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::channel::{Sender, WeakSender};
use cubecl_server::driver::DevicePoison;
use cubecl_server::server::ServerError;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

/// How long one wait on the device blocks before it looks again.
#[cfg(not(target_family = "wasm"))]
const WAIT_SLICE: Duration = Duration::from_secs(1);

/// How long a wait goes without progress before it probes the device with an empty submission.
/// A driver that kills a hung context can leave its fences unsignaled for good, and report the
/// loss only to the next submission.
#[cfg(not(target_family = "wasm"))]
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
            lost.poison(format!("{kind:?}: {message}"));
            log::error!("the wgpu device was lost ({kind:?}): {message}");
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
    pub fn wake_on_loss<T: Send + 'static>(&self, sender: &Sender<T>) {
        let mut waiters = self.waiters();
        if self.is_poisoned() {
            sender.close();
            return;
        }
        // Pruned only when full, and the reserve keeps the next prune at least as far away.
        if waiters.len() == waiters.capacity() {
            waiters.retain(|waiter| waiter.is_pending());
            let pending = waiters.len();
            waiters.reserve(pending);
        }
        waiters.push(Box::new(sender.downgrade()));
    }

    /// Waits until `submission` completes, or everything submitted when `None`, and fails once
    /// the device is lost instead of waiting on work that will never complete.
    #[cfg(not(target_family = "wasm"))]
    pub fn wait_unless_lost(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        submission: Option<wgpu::SubmissionIndex>,
    ) -> Result<(), ServerError> {
        let poll_slice = |submission: &Option<wgpu::SubmissionIndex>| {
            device.poll(wgpu::PollType::Wait {
                submission_index: submission.clone(),
                timeout: Some(WAIT_SLICE),
            })
        };

        self.check()?;
        let mut submission = submission;
        let mut polled = poll_slice(&submission);
        let mut stalled = Duration::ZERO;
        while let Err(wgpu::PollError::Timeout) = polled {
            self.check()?;
            // Pinned on the first slice, so the wait never chases what other streams submit later.
            submission.get_or_insert_with(|| queue.submit(core::iter::empty()));
            stalled += WAIT_SLICE;
            if stalled >= PROBE_AFTER {
                queue.submit(core::iter::empty());
                stalled = Duration::ZERO;
            }
            polled = poll_slice(&submission);
        }

        match polled {
            Ok(_) => self.check(),
            Err(err) => Err(ServerError::Generic {
                reason: format!("wgpu: waiting on the device failed ({err})"),
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// The browser drives the device, so there is nothing to block on.
    #[cfg(target_family = "wasm")]
    pub fn wait_unless_lost(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _submission: Option<wgpu::SubmissionIndex>,
    ) -> Result<(), ServerError> {
        self.check()
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

/// A channel a pending wait listens on. Held weakly, so a callback dropped unrun still closes it.
trait Waiter: Send + Debug {
    /// Whether anything can still send on it, so there is a wait left to wake.
    fn is_pending(&self) -> bool;
    fn close(&self);
}

impl<T: Send> Waiter for WeakSender<T> {
    fn is_pending(&self) -> bool {
        self.upgrade().is_some()
    }

    fn close(&self) {
        if let Some(sender) = self.upgrade() {
            sender.close();
        }
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
        watch.wake_on_loss(&sender);

        watch.poison("lost".into());

        assert!(block_on(receiver.recv()).is_err());
        assert!(watch.check().is_err());
    }

    #[test]
    fn a_wait_started_after_the_loss_wakes_at_once() {
        let watch = PoisonWatch::default();
        watch.poison("lost".into());
        let (sender, receiver) = bounded::<()>(1);

        watch.wake_on_loss(&sender);

        assert!(block_on(receiver.recv()).is_err());
    }

    #[test]
    fn a_sender_dropped_unsent_still_closes_its_channel() {
        let watch = PoisonWatch::default();
        let (sender, receiver) = bounded::<()>(1);
        watch.wake_on_loss(&sender);

        drop(sender);

        assert!(block_on(receiver.recv()).is_err());
        assert!(watch.check().is_ok());
    }
}
