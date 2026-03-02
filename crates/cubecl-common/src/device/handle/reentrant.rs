use alloc::boxed::Box;
use core::cell::{RefCell, RefMut};
use core::ops::DerefMut;
use core::{
    any::{Any, TypeId},
    marker::PhantomData,
};
use hashbrown::HashMap;
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use std::sync::Arc;

use crate::device::handle::{DeviceHandleSpec, ServiceCreationError};
use crate::device::{DeviceId, DeviceService};

type MutCell<T> = RefCell<T>;
type MutGuard<'a, T> = RefMut<'a, T>;

/// Handle for accessing a [`DeviceState`] associated with a specific device.
pub struct ReentrantMutexDeviceHandle<S: DeviceService> {
    lock: DeviceStateLock,
    device_id: DeviceId,
    _phantom: PhantomData<S>,
}

impl<S: DeviceService> DeviceHandleSpec<S> for ReentrantMutexDeviceHandle<S> {
    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Self::insert(device_id, service).map_err(ServiceCreationError::new)
    }

    fn new(device_id: DeviceId) -> Self {
        Self::locate(device_id)
    }

    fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, super::CallError> {
        let mut guard = self.lock();
        Ok(task(&mut guard))
    }

    fn submit_blocking_scoped<'a, R: Send + 'a, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> R {
        let mut guard = self.lock();
        task(&mut guard)
    }

    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let mut guard = self.lock();
        task(&mut guard)
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

/// There is nothing to read without a lock, and it's fine to allow locking a context reference.
unsafe impl<S: DeviceService> Sync for ReentrantMutexDeviceHandle<S> {}

impl<S: DeviceService> Clone for ReentrantMutexDeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            lock: self.lock.clone(),
            _phantom: self._phantom,
            device_id: self.device_id,
        }
    }
}

/// Guard providing mutable access to [`DeviceState`].
///
/// Automatically releases the lock when dropped.
pub struct DeviceStateGuard<'a, S: DeviceService> {
    guard_ref: Option<MutGuard<'a, Box<dyn Any + Send + 'static>>>,
    guard_mutex: Option<ReentrantMutexGuard<'a, DeviceStateMap>>,
    _phantom: PhantomData<S>,
}

/// Guard making sure only the locked device can be used.
///
/// Automatically releases the lock when dropped.
pub struct DeviceGuard<'a> {
    guard_mutex: Option<ReentrantMutexGuard<'a, DeviceStateMap>>,
}

impl<'a, S: DeviceService> Drop for DeviceStateGuard<'a, S> {
    fn drop(&mut self) {
        // Important to drop the ref before.
        self.guard_ref = None;
        self.guard_mutex = None;
    }
}

impl<'a> Drop for DeviceGuard<'a> {
    fn drop(&mut self) {
        self.guard_mutex = None;
    }
}

impl<'a, S: DeviceService> core::ops::Deref for DeviceStateGuard<'a, S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        self.guard_ref
            .as_ref()
            .expect("The guard to not be dropped")
            .downcast_ref()
            .expect("The type to be correct")
    }
}

impl<'a, S: DeviceService> core::ops::DerefMut for DeviceStateGuard<'a, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard_ref
            .as_mut()
            .expect("The guard to not be dropped")
            .downcast_mut()
            .expect("The type to be correct")
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

        // It is safe for the same reasons enumerated in the lock function.
        let (map, map_guard) = unsafe { borrow_mut_split(&state.map) };

        if map.contains_key(&id) {
            return Err(alloc::format!(
                "A server is still registered for device {device:?}"
            ));
        }

        let any: Box<dyn Any + Send + 'static> = Box::new(state_new);
        let cell = MutCell::new(any);

        map.insert(id, cell);

        core::mem::drop(map_guard);
        core::mem::drop(state);

        Ok(lock)
    }

    /// Locks the current device making sure this device can be used.
    pub fn lock_device(&self) -> DeviceGuard<'_> {
        let state = self.lock.lock.lock();

        DeviceGuard {
            guard_mutex: Some(state),
        }
    }

    /// Acquires exclusive mutable access to the [`DeviceState`].
    ///
    /// The same device can lock multiple types at the same time.
    ///
    /// # Panics
    ///
    /// If the same state type is locked multiple times on the same thread.
    /// This can only happen with recursive locking of the same state, which isn't allowed
    /// since having multiple mutable references to the same state isn't valid.
    pub fn lock(&self) -> DeviceStateGuard<'_, S> {
        let key = TypeId::of::<S>();
        let state = self.lock.lock.lock();

        // It is safe for multiple reasons.
        //
        // 1. The mutability of the map is handled by each map entry with a RefCell.
        //    Therefore, multiple mutable references to a map entry are checked.
        // 2. Map items are never cleaned up, therefore it's impossible to remove the validity of
        //    an entry.
        // 3. Because of the lock, no race condition is possible.
        //
        // The reason why unsafe is necessary is that the [DeviceStateGuard] doesn't keep track
        // of the borrowed map entry lifetime. But since it keeps track of both the [RefCell]
        // and the [ReentrantMutex] guards, it is fine to erase the lifetime here.
        let (map, map_guard) = unsafe { borrow_mut_split(&state.map) };

        if !map.contains_key(&key) {
            let state_default = S::init(self.device_id);
            let any: Box<dyn Any + Send + 'static> = Box::new(state_default);
            let cell = MutCell::new(any);

            map.insert(key, cell);
        }

        let value = map
            .get(&key)
            .expect("Just validated the map contains the key.");
        let ref_guard = match value.try_borrow_mut() {
            Ok(guard) => guard,
            #[cfg(feature = "std")]
            Err(_) => panic!(
                "State {} is already borrowed by the current thread {:?}",
                core::any::type_name::<S>(),
                std::thread::current().id()
            ),
            #[cfg(not(feature = "std"))]
            Err(_) => panic!("State {} is already borrowed", core::any::type_name::<S>(),),
        };

        core::mem::drop(map_guard);

        DeviceStateGuard {
            guard_ref: Some(ref_guard),
            guard_mutex: Some(state),
            _phantom: PhantomData,
        }
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
    map: MutCell<HashMap<TypeId, MutCell<Box<dyn Any + Send + 'static>>>>,
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
            map: MutCell::new(HashMap::new()),
        }
    }
}

unsafe fn borrow_mut_split<'a, T>(cell: &MutCell<T>) -> (&'a mut T, MutGuard<'_, T>) {
    let mut guard = cell.borrow_mut();
    let item = guard.deref_mut();
    let item: &'a mut T = unsafe { core::mem::transmute(item) };

    (item, guard)
}
