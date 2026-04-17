use alloc::boxed::Box;
use core::cell::{Cell, RefCell};
use core::{
    any::{Any, TypeId},
    marker::PhantomData,
};
use hashbrown::HashMap;
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use std::sync::Arc;

use crate::device::handle::{DeviceHandleSpec, ServerUtilitiesHandle, ServiceCreationError};
use crate::device::{DeviceId, DeviceService};

/// Handle for accessing a [`DeviceState`] associated with a specific device.
pub struct ReentrantMutexDeviceHandle<S: DeviceService> {
    lock: DeviceStateLock,
    device_id: DeviceId,
    // fn(S) makes this Send+Sync regardless of S, since the handle
    // never holds an S — it only accesses it through the lock.
    _phantom: PhantomData<fn(S)>,
}

impl<S: DeviceService> DeviceHandleSpec<S> for ReentrantMutexDeviceHandle<S> {
    const BLOCKING: bool = true;

    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Self::insert(device_id, service).map_err(ServiceCreationError::new)
    }

    fn new(device_id: DeviceId) -> Self {
        Self::locate(device_id)
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        let state = self.lock.lock.lock();
        state
            .map
            .borrow()
            .get(&TypeId::of::<S>())
            .expect("Service not yet initialized — call init() before load()")
            .utilities
            .clone()
    }

    fn flush_queue(&self) {}

    fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, super::CallError> {
        Ok(self.with_lock(task))
    }

    fn submit_blocking_scoped<'a, R: Send + 'a, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> R {
        self.with_lock(task)
    }

    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.with_lock(task);
    }

    fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, super::CallError> {
        let guard = self.lock_device();
        let result = task();
        core::mem::drop(guard);
        Ok(result)
    }

    fn exclusive_scoped<R: Send, T: FnOnce() -> R + Send>(
        &self,
        task: T,
    ) -> Result<R, super::CallError> {
        let guard = self.lock_device();
        let result = task();
        core::mem::drop(guard);
        Ok(result)
    }
}

impl<S: DeviceService> Clone for ReentrantMutexDeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            lock: self.lock.clone(),
            _phantom: self._phantom,
            device_id: self.device_id,
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

impl<S: DeviceService> ReentrantMutexDeviceHandle<S> {
    /// Creates a [`DeviceContext<S>`] handle for the given device.
    ///
    /// Registers the device-type combination globally if needed.
    pub fn locate(device: DeviceId) -> Self {
        DeviceStateLock::locate(device)
    }

    /// Inserts a new state associated with the device.
    ///
    /// # Returns
    ///
    /// An error if the device already has a registered state.
    pub fn insert(device: DeviceId, state_new: S) -> Result<Self, alloc::string::String> {
        let lock = Self::locate(device);
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
    fn with_lock<R>(&self, f: impl FnOnce(&mut S) -> R) -> R {
        let key = TypeId::of::<S>();
        let state = self.lock.lock.lock();

        // Take the entry out of the map. This gives us owned data with
        // no lifetime tied to the map borrow, so re-entrant calls for
        // different service types can access the map freely.
        let entry = {
            let mut map = state.map.borrow_mut();
            map.entry(key)
                .or_insert_with(|| {
                    let service = S::init(self.device_id);
                    let utilities = service.utilities();
                    ReentrantMutexDeviceState {
                        service: Cell::new(Some(Box::new(service))),
                        utilities,
                    }
                })
                .service
                .take()
        };

        let mut entry = entry.unwrap_or_else(|| {
            panic!(
                "State {} is already borrowed by the current thread",
                core::any::type_name::<S>(),
            )
        });

        let s = entry.downcast_mut::<S>().expect("The type to be correct");
        let result = f(s);

        // Put the entry back.
        state
            .map
            .borrow()
            .get(&key)
            .expect("Entry still exists")
            .service
            .replace(Some(entry));

        result
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
    fn locate<S: DeviceService>(device: DeviceId) -> ReentrantMutexDeviceHandle<S> {
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
            _phantom: PhantomData,
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

        let h1 = ReentrantMutexDeviceHandle::<Svc1>::new(device);
        h1.with_lock(|_| {
            let h2 = ReentrantMutexDeviceHandle::<Svc2>::new(device);
            h2.with_lock(|_| {
                let h3 = ReentrantMutexDeviceHandle::<Svc3>::new(device);
                h3.with_lock(|_| {
                    let h4 = ReentrantMutexDeviceHandle::<Svc4>::new(device);
                    h4.with_lock(|_| {
                        let h5 = ReentrantMutexDeviceHandle::<Svc5>::new(device);
                        h5.with_lock(|_| {
                            let h6 = ReentrantMutexDeviceHandle::<Svc6>::new(device);
                            h6.with_lock(|_| {
                                let h7 = ReentrantMutexDeviceHandle::<Svc7>::new(device);
                                h7.with_lock(|_| {
                                    let h8 = ReentrantMutexDeviceHandle::<Svc8>::new(device);
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
