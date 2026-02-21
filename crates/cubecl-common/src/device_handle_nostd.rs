use super::device_handle_shared::*;
use crate::device::{DeviceId, DeviceService};
use alloc::{boxed::Box, sync::Arc};
use core::{
    any::{Any, TypeId},
    marker::PhantomData,
};
use hashbrown::HashMap;
use spin::Mutex; // Using spin crate for no-std locks

/// A handle to a specific device context (no-std version).
pub struct DeviceHandle<S: DeviceService> {
    device_id: DeviceId,
    _phantom: PhantomData<S>,
}

/// The global storage for all device services.
/// In no-std, we use a global registry protected by a Mutex.
static DEVICE_REGISTRY: Mutex<Option<HashMap<DeviceId, DeviceRegistry>>> = spin::Mutex::new(None);

/// Maps `TypeId` to the actual Service instance.
type DeviceRegistry = HashMap<TypeId, Arc<Mutex<Box<dyn Any + Send>>>>;

impl<S: DeviceService + 'static> DeviceHandle<S> {
    /// Creates or retrieves a context for the given device ID.
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            _phantom: PhantomData,
        }
    }

    /// In no-std, call executes immediately under a lock.
    pub fn call<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
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

    /// In no-std, submit executes immediately on the current thread.
    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let mut guard = DEVICE_REGISTRY.lock();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        };
        let device_map: &mut HashMap<_, _> = match guard.as_mut() {
            Some(val) => val.entry(self.device_id).or_insert_with(HashMap::new),
            None => unreachable!(),
        };

        let type_id = TypeId::of::<S>();

        let state = device_map
            .entry(type_id)
            .or_insert_with(|| {
                let state = S::init(self.device_id);
                Arc::new(Mutex::new(Box::new(state)))
            })
            .clone();
        core::mem::drop(guard);

        let mut state = state.lock();
        let state = state
            .downcast_mut::<S>()
            .expect("State type mismatch in Thread Local Storage");

        task(state);
    }
}

impl<S: DeviceService> Clone for DeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            device_id: self.device_id,
            _phantom: PhantomData,
        }
    }
}
