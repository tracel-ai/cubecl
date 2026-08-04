mod base;

pub use base::*;

use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle};

#[cfg(feature = "std")]
#[allow(dead_code)]
mod channel;

#[allow(dead_code)]
mod mutex;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod reentrant;

#[cfg(all(feature = "std", multi_threading))]
type Inner<S> = channel::ChannelDeviceHandle<S>;
// type Inner<S> = mutex::MutexDeviceHandle<S>;
#[cfg(all(feature = "std", not(multi_threading)))]
type Inner<S> = reentrant::ReentrantMutexDeviceHandle<S>;
#[cfg(all(not(feature = "std"), not(multi_threading)))]
type Inner<S> = mutex::MutexDeviceHandle<S>;

/// TODO: Docs
pub struct DeviceHandle<S: DeviceService> {
    handle: Inner<S>,
}

impl<S: DeviceService> Clone for DeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

#[allow(missing_docs)]
impl<S: DeviceService> DeviceHandle<S> {
    pub const fn is_blocking() -> bool {
        Inner::<S>::BLOCKING
    }

    pub fn insert(device_id: super::DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Ok(Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::insert(device_id, service)?,
        })
    }

    pub fn new(device_id: super::DeviceId) -> Self {
        Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::new(device_id),
        }
    }

    pub fn device_id(&self) -> DeviceId {
        self.handle.device_id()
    }

    pub fn utilities(&self) -> ServerUtilitiesHandle {
        self.handle.utilities()
    }

    pub fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        self.handle.submit_blocking(task)
    }

    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.handle.submit(task)
    }

    pub fn flush_queue(&self) {
        self.handle.flush_queue();
    }

    pub fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        self.handle.exclusive(task)
    }

    /// Stops the background runner threads for `device_id`, blocking until they
    /// exit. Queued tasks run before the threads stop. Live handles keep their
    /// runner alive, so all handles for the device should be dropped first.
    ///
    /// Only meaningful for handle implementations with background threads; a
    /// no-op otherwise.
    ///
    /// # Scope
    ///
    /// **This is device-wide, and `S` is ignored.** It shuts down every
    /// [`DeviceService`] registered on `device_id`, across both service stages, not
    /// only `S`. The type parameter selects the handle implementation to dispatch
    /// through, nothing more.
    ///
    /// [`DeviceId`] is not unique across runtimes either: `type_id` is assigned per
    /// runtime, so distinct backends can hand out the same id. Shutting down a
    /// device from one runtime can therefore tear down another runtime's services
    /// on the colliding id, and block while that runtime's handles are still live.
    pub fn shutdown(device_id: DeviceId) {
        <Inner<S> as DeviceHandleSpec<S>>::shutdown(device_id)
    }
}

/// Shuts a device's runner down when dropped. Use [`DeviceFixture`] rather than
/// this directly: the guard alone still requires getting the device id and the
/// declaration order right.
#[cfg(test)]
struct ShutdownGuard {
    device_id: DeviceId,
    shutdown: fn(DeviceId),
}

#[cfg(test)]
impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        (self.shutdown)(self.device_id);
    }
}

/// Hands out a device id no other test is using.
///
/// The channel implementation keys global registries by device id, and the whole
/// crate's tests share one binary, so a hardcoded id collides with whatever test
/// happens to run alongside: one test would shut down another's live runner.
#[cfg(test)]
fn next_test_device_id() -> DeviceId {
    use core::sync::atomic::{AtomicU16, Ordering};

    static NEXT: AtomicU16 = AtomicU16::new(0);

    DeviceId {
        type_id: 0,
        index_id: NEXT.fetch_add(1, Ordering::Relaxed),
    }
}

/// A handle on a device of its own, whose runner is shut down when the fixture drops.
///
/// This is the only correct way to spell the pattern, so it is the only one tests
/// should use. The three hazards are all handled structurally: the device id comes
/// from [`next_test_device_id`] so it cannot collide, the guard is created with it
/// so it cannot be forgotten, and `handle` is declared before `_guard` so it drops
/// first, meaning the shutdown never waits on a handle the fixture itself owns.
///
/// Handles a test creates on top of this one (a second service on the same device,
/// clones) must be locals declared *after* the fixture, which drop in reverse order
/// and so are gone before the shutdown runs.
#[cfg(test)]
pub(crate) struct DeviceFixture<H> {
    handle: H,
    _guard: ShutdownGuard,
    device_id: DeviceId,
}

#[cfg(test)]
impl<H> DeviceFixture<H> {
    pub(crate) fn new(build: fn(DeviceId) -> H, shutdown: fn(DeviceId)) -> Self {
        let device_id = next_test_device_id();

        Self {
            handle: build(device_id),
            _guard: ShutdownGuard {
                device_id,
                shutdown,
            },
            device_id,
        }
    }

    pub(crate) fn device_id(&self) -> DeviceId {
        self.device_id
    }
}

#[cfg(test)]
impl<H> core::ops::Deref for DeviceFixture<H> {
    type Target = H;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[cfg(test)]
mod tests_channel {
    type DeviceHandle<S> = channel::ChannelDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}

#[cfg(test)]
mod tests_mutex {
    type DeviceHandle<S> = mutex::MutexDeviceHandle<S>;

    include!("./tests.rs");
}

#[cfg(test)]
mod tests_reentrant {
    type DeviceHandle<S> = reentrant::ReentrantMutexDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}
