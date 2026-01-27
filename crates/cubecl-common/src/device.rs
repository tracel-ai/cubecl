use core::cmp::Ordering;

/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
}

/// Device trait for all cubecl devices.
pub trait Device: Default + Clone + core::fmt::Debug + Send + Sync + 'static {
    /// Create a device from its [id](DeviceId).
    fn from_id(device_id: DeviceId) -> Self;
    /// Retrieve the [device id](DeviceId) from the device.
    fn to_id(&self) -> DeviceId;
    /// Returns the number of devices available under the provided type id.
    fn device_count(type_id: u16) -> usize;
    /// Returns the total number of devices that can be handled by the runtime.
    fn device_count_total() -> usize {
        Self::device_count(0)
    }
}

impl core::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

impl Ord for DeviceId {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.type_id.cmp(&other.type_id) {
            Ordering::Equal => self.index_id.cmp(&other.index_id),
            other => other,
        }
    }
}

impl PartialOrd for DeviceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub use context::*;

#[cfg(feature = "std")]
mod reentrant {
    pub use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
}

// MutCell and MutGuard differs in implementation whether `std` is activated.

#[cfg(feature = "std")]
mod cell {
    use core::cell::{RefCell, RefMut};
    use core::ops::DerefMut;

    pub type MutCell<T> = RefCell<T>;
    pub type MutGuard<'a, T> = RefMut<'a, T>;

    pub unsafe fn borrow_mut_split<'a, T>(cell: &MutCell<T>) -> (&'a mut T, MutGuard<'_, T>) {
        let mut guard = cell.borrow_mut();
        let item = guard.deref_mut();
        let item: &'a mut T = unsafe { core::mem::transmute(item) };

        (item, guard)
    }
}

#[cfg(not(feature = "std"))]
mod cell {
    use core::ops::{Deref, DerefMut};

    pub struct MutGuard<'a, T> {
        guard: spin::MutexGuard<'a, T>,
    }

    pub struct MutCell<T> {
        lock: spin::Mutex<T>,
    }

    impl<T> MutCell<T> {
        pub fn new(item: T) -> Self {
            Self {
                lock: spin::Mutex::new(item),
            }
        }
    }

    impl<'a, T> Deref for MutGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            self.guard.deref()
        }
    }

    impl<'a, T> DerefMut for MutGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.guard.deref_mut()
        }
    }

    impl<T> MutCell<T> {
        pub fn try_borrow_mut(&self) -> Result<MutGuard<'_, T>, ()> {
            match self.lock.try_lock() {
                Some(guard) => Ok(MutGuard { guard }),
                None => Err(()),
            }
        }
    }

    pub unsafe fn borrow_mut_split<'a, T>(
        cell: &MutCell<T>,
    ) -> (&'a mut T, spin::MutexGuard<'_, T>) {
        let mut guard = cell.lock.lock();
        let item = guard.deref_mut();
        let item: &'a mut T = unsafe { core::mem::transmute(item) };

        (item, guard)
    }
}

#[cfg(not(feature = "std"))]
mod reentrant {
    use core::ops::Deref;

    pub struct ReentrantMutex<T> {
        inner: spin::RwLock<T>,
    }

    impl<T> ReentrantMutex<T> {
        pub fn new(item: T) -> Self {
            Self {
                inner: spin::RwLock::new(item),
            }
        }
    }

    pub struct ReentrantMutexGuard<'a, T> {
        guard: spin::RwLockReadGuard<'a, T>,
    }

    impl<'a, T> Deref for ReentrantMutexGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            self.guard.deref()
        }
    }

    impl<T> ReentrantMutex<T> {
        pub fn lock(&self) -> ReentrantMutexGuard<'_, T> {
            let guard = self.inner.read();
            ReentrantMutexGuard { guard }
        }
    }
}

mod context {
    use super::cell::{MutCell, MutGuard};
    use alloc::boxed::Box;
    use core::{
        any::{Any, TypeId},
        marker::PhantomData,
    };
    use hashbrown::HashMap;

    use super::reentrant::{ReentrantMutex, ReentrantMutexGuard};

    use crate::{device::cell::borrow_mut_split, stub::Arc};

    use super::{Device, DeviceId};

    /// A state that can be saved inside the [`DeviceContext`].
    pub trait DeviceState: Send + 'static {
        /// Initialize a new state on the given device.
        fn init(device_id: DeviceId) -> Self;
    }

    /// Handle for accessing a [`DeviceState`] associated with a specific device.
    pub struct DeviceContext<S: DeviceState> {
        lock: DeviceStateLock,
        lock_kind: Arc<ReentrantMutex<()>>,
        device_id: DeviceId,
        _phantom: PhantomData<S>,
    }

    /// There is nothing to read without a lock, and it's fine to allow locking a context reference.
    unsafe impl<S: DeviceState> Sync for DeviceContext<S> {}

    impl<S: DeviceState> Clone for DeviceContext<S> {
        fn clone(&self) -> Self {
            Self {
                lock: self.lock.clone(),
                lock_kind: self.lock_kind.clone(),
                _phantom: self._phantom,
                device_id: self.device_id,
            }
        }
    }

    /// Guard providing mutable access to [`DeviceState`].
    ///
    /// Automatically releases the lock when dropped.
    pub struct DeviceStateGuard<'a, S: DeviceState> {
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

    impl<'a, S: DeviceState> Drop for DeviceStateGuard<'a, S> {
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

    impl<'a, S: DeviceState> core::ops::Deref for DeviceStateGuard<'a, S> {
        type Target = S;

        fn deref(&self) -> &Self::Target {
            self.guard_ref
                .as_ref()
                .expect("The guard to not be dropped")
                .downcast_ref()
                .expect("The type to be correct")
        }
    }

    impl<'a, S: DeviceState> core::ops::DerefMut for DeviceStateGuard<'a, S> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.guard_ref
                .as_mut()
                .expect("The guard to not be dropped")
                .downcast_mut()
                .expect("The type to be correct")
        }
    }

    impl<S: DeviceState> DeviceContext<S> {
        /// Creates a [`DeviceContext<S>`] handle for the given device.
        ///
        /// Registers the device-type combination globally if needed.
        pub fn locate<D: Device + 'static>(device: &D) -> Self {
            DeviceStateLock::locate(device)
        }

        /// Inserts a new state associated with the device.
        ///
        /// # Returns
        ///
        /// An error if the device already has a registered state.
        pub fn insert<D: Device + 'static>(
            device: &D,
            state_new: S,
        ) -> Result<Self, alloc::string::String> {
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

        /// Locks all devices under the same kind.
        ///
        /// This is useful when you need mutable access to multiple devices at once, which can lead
        /// to deadlocks.
        pub fn lock_device_kind(&self) -> ReentrantMutexGuard<'_, ()> {
            self.lock_kind.lock()
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

    type Key = (DeviceId, TypeId);

    static GLOBAL: spin::Mutex<DeviceLocator> = spin::Mutex::new(DeviceLocator { state: None });

    #[derive(Default)]
    struct DeviceLocatorState {
        device: HashMap<Key, DeviceStateLock>,
        device_kind: HashMap<TypeId, Arc<ReentrantMutex<()>>>,
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
        fn locate<D: Device + 'static, S: DeviceState>(device: &D) -> DeviceContext<S> {
            let id = device.to_id();
            let kind = TypeId::of::<D>();
            let key = (id, TypeId::of::<D>());
            let mut global = GLOBAL.lock();

            let locator_state = match &mut global.state {
                Some(state) => state,
                None => {
                    global.state = Some(Default::default());
                    global.state.as_mut().expect("Just created Option::Some")
                }
            };

            let lock = match locator_state.device.get(&key) {
                Some(value) => value.clone(),
                None => {
                    let state = DeviceStateMap::new();

                    let value = DeviceStateLock {
                        lock: Arc::new(ReentrantMutex::new(state)),
                    };

                    locator_state.device.insert(key, value);
                    locator_state
                        .device
                        .get(&key)
                        .expect("Just inserted the key/value")
                        .clone()
                }
            };
            let lock_kind = match locator_state.device_kind.get(&kind) {
                Some(value) => value.clone(),
                None => {
                    locator_state
                        .device_kind
                        .insert(kind, Arc::new(ReentrantMutex::new(())));
                    locator_state
                        .device_kind
                        .get(&kind)
                        .expect("Just inserted the key/value")
                        .clone()
                }
            };

            DeviceContext {
                lock,
                lock_kind,
                device_id: id,
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

    #[cfg(test)]
    mod tests {
        use core::{
            ops::{Deref, DerefMut},
            time::Duration,
        };

        use super::*;

        #[test_log::test]
        fn can_have_multiple_mutate_state() {
            let device1 = TestDevice::<0>::new(0);
            let device2 = TestDevice::<1>::new(0);

            let state1_usize = DeviceContext::<usize>::locate(&device1);
            let state1_u32 = DeviceContext::<u32>::locate(&device1);
            let state2_usize = DeviceContext::<usize>::locate(&device2);

            let mut guard_usize = state1_usize.lock();
            let mut guard_u32 = state1_u32.lock();

            let val_usize = guard_usize.deref_mut();
            let val_u32 = guard_u32.deref_mut();

            *val_usize += 1;
            *val_u32 += 2;

            assert_eq!(*val_usize, 1);
            assert_eq!(*val_u32, 2);

            core::mem::drop(guard_usize);
            core::mem::drop(guard_u32);

            let mut guard_usize = state2_usize.lock();

            let val_usize = guard_usize.deref_mut();
            *val_usize += 1;

            assert_eq!(*val_usize, 1);

            core::mem::drop(guard_usize);

            let guard_usize = state1_usize.lock();
            let guard_u32 = state1_u32.lock();

            let val_usize = guard_usize.deref();
            let val_u32 = guard_u32.deref();

            assert_eq!(*val_usize, 1);
            assert_eq!(*val_u32, 2);
        }

        #[test_log::test]
        #[should_panic]
        fn can_not_have_multiple_mut_ref_to_same_state() {
            let device1 = TestDevice::<0>::new(0);

            struct DummyState;

            impl DeviceState for DummyState {
                fn init(_device_id: DeviceId) -> Self {
                    DummyState
                }
            }

            fn recursive(total: usize, state: &DeviceContext<DummyState>) {
                let _guard = state.lock();

                if total > 0 {
                    recursive(total - 1, state);
                }
            }

            recursive(5, &DeviceContext::locate(&device1));
        }

        #[test_log::test]
        #[ignore = "Ignore for now because it breaks CI"]
        fn work_with_many_threads() {
            let num_threads = 32;
            let handles: Vec<_> = (0..num_threads)
                .map(|i| std::thread::spawn(move || thread_main((num_threads * 4) - i)))
                .collect();

            handles.into_iter().for_each(|h| h.join().unwrap());

            let device1 = TestDevice::<0>::new(0);
            let device2 = TestDevice::<1>::new(0);

            let state1_i64 = DeviceContext::<i64>::locate(&device1);
            let state1_i32 = DeviceContext::<i32>::locate(&device1);
            let state2_i32 = DeviceContext::<i32>::locate(&device2);

            let guard_i64 = state1_i64.lock();
            let guard_i32 = state1_i32.lock();

            assert_eq!(*guard_i64, num_threads as i64);
            assert_eq!(*guard_i32, num_threads as i32 * 2);

            core::mem::drop(guard_i64);
            core::mem::drop(guard_i32);

            let guard_i32 = state2_i32.lock();
            assert_eq!(*guard_i32, num_threads as i32);
        }

        fn thread_main(sleep: u64) {
            let device1 = TestDevice::<0>::new(0);
            let device2 = TestDevice::<1>::new(0);

            let state1_i64 = DeviceContext::<i64>::locate(&device1);
            let state1_i32 = DeviceContext::<i32>::locate(&device1);
            let state2_i32 = DeviceContext::<i32>::locate(&device2);

            let mut guard_i64 = state1_i64.lock();
            let mut guard_i32 = state1_i32.lock();

            let val_i64 = guard_i64.deref_mut();
            let val_i32 = guard_i32.deref_mut();

            *val_i64 += 1;
            *val_i32 += 2;

            core::mem::drop(guard_i64);
            core::mem::drop(guard_i32);

            std::thread::sleep(Duration::from_millis(sleep));

            let mut guard_i32 = state2_i32.lock();

            let val_i32 = guard_i32.deref_mut();
            *val_i32 += 1;

            core::mem::drop(guard_i32);
        }

        #[derive(Debug, Clone, Default, new)]
        /// Type is only to create different type ids.
        pub struct TestDevice<const TYPE: u8> {
            index: u32,
        }

        impl<const TYPE: u8> Device for TestDevice<TYPE> {
            fn from_id(device_id: DeviceId) -> Self {
                Self {
                    index: device_id.index_id,
                }
            }

            fn to_id(&self) -> DeviceId {
                DeviceId {
                    type_id: 0,
                    index_id: self.index,
                }
            }

            fn device_count(_type_id: u16) -> usize {
                TYPE as usize + 1
            }
        }

        impl DeviceState for usize {
            fn init(_device_id: DeviceId) -> Self {
                0
            }
        }

        impl DeviceState for u32 {
            fn init(_device_id: DeviceId) -> Self {
                0
            }
        }
        impl DeviceState for i32 {
            fn init(_device_id: DeviceId) -> Self {
                0
            }
        }
        impl DeviceState for i64 {
            fn init(_device_id: DeviceId) -> Self {
                0
            }
        }
    }
}
