use alloc::boxed::Box;
use core::any::{Any, TypeId};
use core::cell::{Cell, RefCell};
use cubecl_environment::sync::reentrant::{ReentrantMutex, ReentrantMutexGuard};
use hashbrown::HashMap;
use std::sync::Arc;

use crate::device::handle::{DeviceHandleSpec, ServerUtilitiesHandle, ServiceCreationError};
use crate::device::{DeviceId, DeviceService};

/// Handle for accessing a [`DeviceState`] associated with a specific device.
pub struct ReentrantMutexDeviceHandle {
    lock: DeviceStateLock,
    device_id: DeviceId,
    /// The service type this transport reaches, as the state map keys it.
    type_id: TypeId,
    /// Its name, for the report when the state is already borrowed.
    type_name: &'static str,
    /// Builds the service on first use, since the map is filled lazily.
    init: fn(DeviceId) -> ReentrantMutexDeviceState,
}

/// The state for a service of type `S` on `device_id`, built on first use.
fn init_state<S: DeviceService>(device_id: DeviceId) -> ReentrantMutexDeviceState {
    let service = S::init(device_id);
    let utilities = service.utilities();
    ReentrantMutexDeviceState {
        service: Cell::new(Some(Box::new(service))),
        utilities,
    }
}

impl DeviceHandleSpec for ReentrantMutexDeviceHandle {
    const BLOCKING: bool = true;

    fn insert<S: DeviceService>(
        device_id: DeviceId,
        service: S,
    ) -> Result<Self, ServiceCreationError> {
        Self::insert::<S>(device_id, service).map_err(ServiceCreationError::new)
    }

    fn new<S: DeviceService>(device_id: DeviceId) -> Self {
        Self::locate::<S>(device_id)
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        let state = self.lock.lock.lock();
        state
            .map
            .borrow()
            .get(&self.type_id)
            .expect("Service not yet initialized — call init() before load()")
            .utilities
            .clone()
    }

    fn flush_queue(&self) {}

    fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, super::CallError> {
        Ok(self.with_lock(task))
    }

    fn submit<T: FnOnce(&mut dyn Any) + Send + 'static>(&self, task: T) {
        self.with_lock(task);
    }

    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, super::CallError> {
        let guard = self.lock_device();
        let result = task();
        core::mem::drop(guard);
        Ok(result)
    }
}

impl Clone for ReentrantMutexDeviceHandle {
    fn clone(&self) -> Self {
        Self {
            lock: self.lock.clone(),
            device_id: self.device_id,
            type_id: self.type_id,
            type_name: self.type_name,
            init: self.init,
        }
    }
}

/// Guard making sure only the locked device can be used.
pub struct DeviceGuard<'a> {
    _guard_mutex: Option<ReentrantMutexGuard<'a, DeviceStateMap>>,
}

impl<'a> Drop for DeviceGuard<'a> {
    fn drop(&mut self) {
        self._guard_mutex = None;
    }
}

impl ReentrantMutexDeviceHandle {
    /// Creates a handle reaching a service of type `S` on the given device.
    ///
    /// Registers the device-type combination globally if needed.
    pub fn locate<S: DeviceService>(device: DeviceId) -> Self {
        DeviceStateLock::locate::<S>(device)
    }

    /// Inserts a new state associated with the device.
    ///
    /// # Returns
    ///
    /// An error if the device already has a registered state.
    pub fn insert<S: DeviceService>(
        device: DeviceId,
        state_new: S,
    ) -> Result<Self, alloc::string::String> {
        let lock = Self::locate::<S>(device);
        let id = TypeId::of::<S>();

        let state = lock.lock.lock.lock();
        let mut map = state.map.borrow_mut();

        if map.contains_key(&id) {
            return Err(alloc::format!(
                "A server is still registered for device {device:?}"
            ));
        }

        let utilities = state_new.utilities();
        let any: Box<dyn Any + Send + 'static> = Box::new(state_new);
        map.insert(
            id,
            ReentrantMutexDeviceState {
                service: Cell::new(Some(any)),
                utilities,
            },
        );

        core::mem::drop(map);
        core::mem::drop(state);

        Ok(lock)
    }

    /// Locks the current device making sure this device can be used.
    pub fn lock_device(&self) -> DeviceGuard<'_> {
        let state = self.lock.lock.lock();

        DeviceGuard {
            _guard_mutex: Some(state),
        }
    }

    /// Acquires exclusive mutable access to the state and passes it to `f`.
    ///
    /// The same device can lock multiple types at the same time.
    ///
    /// # Panics
    ///
    /// If the same state type is locked multiple times on the same thread.
    fn with_lock<R>(&self, f: impl FnOnce(&mut dyn Any) -> R) -> R {
        let key = self.type_id;
        let state = self.lock.lock.lock();

        // Take the entry out of the map. This gives us owned data with
        // no lifetime tied to the map borrow, so re-entrant calls for
        // different service types can access the map freely.
        let entry = {
            let mut map = state.map.borrow_mut();
            map.entry(key)
                .or_insert_with(|| (self.init)(self.device_id))
                .service
                .take()
        };

        let entry = entry.unwrap_or_else(|| {
            panic!(
                "State {} is already borrowed by the current thread",
                self.type_name,
            )
        });

        // Put the entry back when `f` returns, and just the same when it
        // unwinds: a panic inside a task is the task's to report, not a
        // reason for every later call on this device to find the service
        // "already borrowed". The panic itself keeps propagating.
        let mut restore = Restore {
            map: &state.map,
            key,
            entry: Some(entry),
        };

        f(&mut **restore.entry.as_mut().expect("taken back only on drop"))
    }
}

/// Returns a service taken out of its [`DeviceStateMap`] on drop, whether the
/// task holding it returned or unwound.
struct Restore<'a> {
    map: &'a RefCell<HashMap<TypeId, ReentrantMutexDeviceState>>,
    key: TypeId,
    entry: Option<Box<dyn Any + Send>>,
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        let entry = self.entry.take().expect("restored once");
        self.map
            .borrow()
            .get(&self.key)
            .expect("Entry still exists")
            .service
            .replace(Some(entry));
    }
}

static GLOBAL: spin::Mutex<DeviceLocator> = spin::Mutex::new(DeviceLocator { state: None });

#[derive(Default)]
struct DeviceLocatorState {
    states: HashMap<DeviceId, DeviceStateLock>,
}

struct DeviceLocator {
    state: Option<DeviceLocatorState>,
}

#[derive(Clone)]
struct DeviceStateLock {
    lock: Arc<ReentrantMutex<DeviceStateMap>>,
}

struct DeviceStateMap {
    map: RefCell<HashMap<TypeId, ReentrantMutexDeviceState>>,
}

struct ReentrantMutexDeviceState {
    /// `None` means the state is currently borrowed by a `with_lock` call.
    service: Cell<Option<Box<dyn Any + Send + 'static>>>,
    utilities: ServerUtilitiesHandle,
}

impl DeviceStateLock {
    fn locate<S: DeviceService>(device: DeviceId) -> ReentrantMutexDeviceHandle {
        let mut global = GLOBAL.lock();

        let locator_state = match &mut global.state {
            Some(state) => state,
            None => {
                global.state = Some(Default::default());
                global.state.as_mut().expect("Just created Option::Some")
            }
        };

        let lock = match locator_state.states.get(&device) {
            Some(value) => value.clone(),
            None => {
                let state = DeviceStateMap::new();

                let value = DeviceStateLock {
                    lock: Arc::new(ReentrantMutex::new(state)),
                };

                locator_state.states.insert(device, value);
                locator_state
                    .states
                    .get(&device)
                    .expect("Just inserted the key/value")
                    .clone()
            }
        };

        ReentrantMutexDeviceHandle {
            lock,
            device_id: device,
            type_id: TypeId::of::<S>(),
            type_name: core::any::type_name::<S>(),
            init: init_state::<S>,
        }
    }
}

impl DeviceStateMap {
    fn new() -> Self {
        Self {
            map: RefCell::new(HashMap::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    macro_rules! make_service {
        ($name:ident) => {
            struct $name;
            impl DeviceService for $name {
                fn init(_: DeviceId) -> Self {
                    $name
                }
                fn utilities(&self) -> ServerUtilitiesHandle {
                    Arc::new(())
                }
            }
        };
    }

    make_service!(Svc1);
    make_service!(Svc2);
    make_service!(Svc3);
    make_service!(Svc4);
    make_service!(Svc5);
    make_service!(Svc6);
    make_service!(Svc7);
    make_service!(Svc8);

    /// Lock many service types on the same device to force `HashMap` resizes
    /// while earlier services are still locked. Pre-fix, `borrow_mut_split`
    /// transmuted a `RefMut` lifetime, and `HashMap` resize moved entries out
    /// from under those `RefMuts`. Miri can catch this use-after-free.
    #[test]
    fn test_many_services_reentrant_resize() {
        let device = DeviceId {
            type_id: 99,
            index_id: 99,
        };

        let h1 = ReentrantMutexDeviceHandle::new::<Svc1>(device);
        h1.with_lock(|_| {
            let h2 = ReentrantMutexDeviceHandle::new::<Svc2>(device);
            h2.with_lock(|_| {
                let h3 = ReentrantMutexDeviceHandle::new::<Svc3>(device);
                h3.with_lock(|_| {
                    let h4 = ReentrantMutexDeviceHandle::new::<Svc4>(device);
                    h4.with_lock(|_| {
                        let h5 = ReentrantMutexDeviceHandle::new::<Svc5>(device);
                        h5.with_lock(|_| {
                            let h6 = ReentrantMutexDeviceHandle::new::<Svc6>(device);
                            h6.with_lock(|_| {
                                let h7 = ReentrantMutexDeviceHandle::new::<Svc7>(device);
                                h7.with_lock(|_| {
                                    let h8 = ReentrantMutexDeviceHandle::new::<Svc8>(device);
                                    h8.with_lock(|_| {});
                                });
                            });
                        });
                    });
                });
            });
        });
    }
}
