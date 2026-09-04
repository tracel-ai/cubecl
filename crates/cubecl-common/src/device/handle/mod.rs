mod base;

pub use base::*;

use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle, ServiceId};
use core::any::Any;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod channel;

#[allow(dead_code)]
mod mutex;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod reentrant;

#[cfg(all(feature = "std", multi_threading))]
type Inner = channel::ChannelDeviceHandle;
// type Inner = mutex::MutexDeviceHandle;
#[cfg(all(feature = "std", not(multi_threading)))]
type Inner = reentrant::ReentrantMutexDeviceHandle;
#[cfg(all(not(feature = "std"), not(multi_threading)))]
type Inner = mutex::MutexDeviceHandle;

/// A handle to one service, reached as `S`.
///
/// `S` is the concrete service for a handle built with [`insert`](Self::insert)
/// or [`new`](Self::new), and a trait object for one built with
/// [`seen_as`](Self::seen_as). Either way the service lives where the
/// transport `I` put it; the handle only knows how to see it as `S`.
pub struct DeviceHandle<S: ?Sized, I: DeviceHandleSpec = Inner> {
    handle: I,
    service: ServiceId,
    cast: fn(&mut dyn Any) -> &mut S,
}

impl<S: ?Sized, I: DeviceHandleSpec> Clone for DeviceHandle<S, I> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            service: self.service,
            cast: self.cast,
        }
    }
}

/// The state the transport holds is `S`, or the registry is broken.
fn downcast<S: 'static>(state: &mut dyn Any) -> &mut S {
    state
        .downcast_mut::<S>()
        .expect("State type mismatch in the device registry")
}

#[allow(missing_docs)]
impl<S: DeviceService, I: DeviceHandleSpec> DeviceHandle<S, I> {
    pub fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Ok(Self {
            handle: I::insert::<S>(device_id, service)?,
            service: ServiceId::of::<S>(device_id),
            cast: downcast::<S>,
        })
    }

    pub fn new(device_id: DeviceId) -> Self {
        Self {
            handle: I::new::<S>(device_id),
            service: ServiceId::of::<S>(device_id),
            cast: downcast::<S>,
        }
    }

    pub fn try_new(device_id: DeviceId) -> Result<Self, ServiceCreationError> {
        Ok(Self {
            handle: I::try_new::<S>(device_id)?,
            service: ServiceId::of::<S>(device_id),
            cast: downcast::<S>,
        })
    }
}

#[allow(missing_docs)]
impl<S: ?Sized + 'static, I: DeviceHandleSpec> DeviceHandle<S, I> {
    pub const fn is_blocking() -> bool {
        I::BLOCKING
    }

    /// The same service, seen as `T`: `cast` turns the state this handle
    /// already reaches into a `T`, once per task, on the thread that runs it.
    pub fn seen_as<T: ?Sized>(self, cast: fn(&mut dyn Any) -> &mut T) -> DeviceHandle<T, I> {
        DeviceHandle {
            handle: self.handle,
            service: self.service,
            cast,
        }
    }

    pub fn device_id(&self) -> DeviceId {
        self.handle.device_id()
    }

    /// The service this handle reaches: its device and its concrete type,
    /// whatever it is seen as.
    pub fn service_id(&self) -> ServiceId {
        self.service
    }

    pub fn utilities(&self) -> ServerUtilitiesHandle {
        self.handle.utilities()
    }

    pub fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let cast = self.cast;
        self.handle.submit_blocking(move |state| task(cast(state)))
    }

    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let cast = self.cast;
        self.handle.submit(move |state| task(cast(state)))
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
        I::shutdown(device_id)
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
    type DeviceHandle<S> = super::DeviceHandle<S, channel::ChannelDeviceHandle>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}

#[cfg(test)]
mod tests_mutex {
    type DeviceHandle<S> = super::DeviceHandle<S, mutex::MutexDeviceHandle>;

    include!("./tests.rs");
}

#[cfg(test)]
mod tests_reentrant {
    type DeviceHandle<S> = super::DeviceHandle<S, reentrant::ReentrantMutexDeviceHandle>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}
