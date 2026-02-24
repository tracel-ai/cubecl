use crate::stub::Mutex;
use crate::{
    device::{
        DeviceId, DeviceService,
        handle::{CallError, DeviceHandleSpec},
    },
    stream_id::StreamId,
    stub::RwLock,
};
use alloc::{boxed::Box, sync::Arc};
use core::{
    any::{Any, TypeId},
    marker::PhantomData,
};
use hashbrown::HashMap;

/// A handle to a specific device context (no-std version).
pub struct MutexDeviceHandle<S: DeviceService> {
    service: Arc<Mutex<Box<dyn Any + Send>>>,
    device_id: DeviceId,
    _phantom: PhantomData<S>,
}

/// The global storage for all device services.
/// In no-std, we use a global registry protected by a Mutex.
static DEVICE_REGISTRY: spin::Mutex<Option<HashMap<DeviceId, DeviceRegistry>>> =
    spin::Mutex::new(None);

/// Maps `TypeId` to the actual Service instance.
type DeviceRegistry = HashMap<TypeId, Arc<Mutex<Box<dyn Any + Send>>>>;

impl<S: DeviceService + 'static> DeviceHandleSpec<S> for MutexDeviceHandle<S> {
    fn new(device_id: DeviceId) -> Self {
        let mut guard = DEVICE_REGISTRY.lock();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        };
        let device_map: &mut HashMap<_, _> = match guard.as_mut() {
            Some(val) => val.entry(device_id).or_insert_with(HashMap::new),
            None => unreachable!(),
        };

        let type_id = TypeId::of::<S>();

        let service = device_map
            .entry(type_id)
            .or_insert_with(|| {
                let state = S::init(device_id);
                Arc::new(Mutex::new(Box::new(state)))
            })
            .clone();

        Self {
            service,
            device_id,
            _phantom: PhantomData,
        }
    }

    fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        self.submit(move |state| {
            let returned = task(state);
            sender.send(returned).unwrap();
        });

        recv.try_recv().map_err(|_| CallError)
    }

    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let mut guard = self.service.lock().unwrap();
        let state = guard
            .downcast_mut::<S>()
            .expect("State type mismatch in Thread Local Storage");

        task(state);
    }

    fn insert(device_id: DeviceId, service: S) -> Result<Self, ()> {
        let mut guard = DEVICE_REGISTRY.lock();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        };
        let device_map: &mut HashMap<_, _> = match guard.as_mut() {
            Some(val) => val.entry(device_id).or_insert_with(HashMap::new),
            None => unreachable!(),
        };

        let type_id = TypeId::of::<S>();

        if device_map.contains_key(&type_id) {
            return Err(());
        }

        let service = device_map
            .entry(type_id)
            .or_insert_with(|| Arc::new(Mutex::new(Box::new(service))))
            .clone();

        Ok(Self {
            service,
            device_id,
            _phantom: PhantomData,
        })
    }

    fn submit_blocking_scoped<'a, R: Send + 'a, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> R {
        let (sender, recv) = oneshot::channel();

        // 1. Wrap the task and sender into a single closure
        let wrapper = move |state: &mut S| {
            let returned = task(state);
            sender.send(returned).unwrap();
        };

        // 2. Erase the lifetime using transmute to make it 'static
        // We use Box first to get a stable pointer size
        //
        // This is safe if we ensure the function is actually called BEFORE the end of this
        // function. Which is the case if we don't have any error.
        let boxed: Box<dyn for<'s> FnOnce(&'s mut S) + Send> = Box::new(wrapper);

        let static_task: Box<dyn for<'s> FnOnce(&'s mut S) + Send + 'static> =
            unsafe { core::mem::transmute(boxed) };

        self.submit(static_task);

        recv.try_recv().unwrap()
    }

    fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let lock = self.device_lock();
        let guard = lock.lock();
        let result = Ok(task());
        core::mem::drop(guard);
        result
    }

    fn exclusive_scoped<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        let lock = self.device_lock();
        let guard = lock.lock();
        let result = Ok(task());
        core::mem::drop(guard);
        result
    }
}

static DEVICE_LOCK: spin::Mutex<Option<HashMap<DeviceId, Arc<DeviceLock>>>> =
    spin::Mutex::new(None);

struct DeviceLock {
    lock: RwLock<Option<StreamId>>,
    main: spin::Mutex<()>,
}

enum Guard<'a> {
    Reentrant,
    Main(spin::MutexGuard<'a, ()>, &'a DeviceLock),
}

impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        match self {
            Guard::Reentrant => {}
            Guard::Main(_mutex_guard, thread_mutex) => {
                let mut state = thread_mutex.lock.write().unwrap();
                *state = None;
            }
        }
    }
}

impl DeviceLock {
    pub fn lock(&self) -> Guard<'_> {
        // TODO: Use thread id when we can.
        let stream_id = StreamId::current();

        loop {
            let mut state = self.lock.write().unwrap();

            let is_ok = match state.as_ref() {
                Some(value) => *value == stream_id,
                None => {
                    *state = Some(stream_id);
                    let guard = self.main.lock();
                    return Guard::Main(guard, self);
                }
            };

            match is_ok {
                true => {
                    core::mem::drop(state);
                    return Guard::Reentrant;
                }
                false => {
                    // spin.
                }
            };
        }
    }
}

impl<S: DeviceService> MutexDeviceHandle<S> {
    fn device_lock(&self) -> Arc<DeviceLock> {
        let mut guard = DEVICE_LOCK.lock();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        };

        let device_map = match guard.as_mut() {
            Some(val) => val.entry(self.device_id),
            None => unreachable!(),
        };

        device_map
            .or_insert_with(|| {
                Arc::new(DeviceLock {
                    lock: RwLock::new(None),
                    main: spin::Mutex::new(()),
                })
            })
            .clone()
    }
}
impl<S: DeviceService> Clone for MutexDeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            service: self.service.clone(),
            device_id: self.device_id,
            _phantom: PhantomData,
        }
    }
}
