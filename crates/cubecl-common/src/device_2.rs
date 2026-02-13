use crate::device::{DeviceId, DeviceState};
use alloc::boxed::Box;
use core::{
    any::{Any, TypeId},
    cell::RefCell,
    marker::PhantomData,
};
use hashbrown::HashMap;
use std::{rc::Rc, sync::mpsc::SyncSender};

/// A handle to a specific device context.
///
/// This struct allows sending closures to be executed on a dedicated
/// thread for the specific device, ensuring thread-safe access to
/// the device's state (`S`).
pub struct DeviceContext2<S: DeviceState> {
    sender: SyncSender<Message>,
    device_id: DeviceId,
    _phantom: PhantomData<S>,
}

impl<S: DeviceState> Clone for DeviceContext2<S> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            device_id: self.device_id.clone(),
            _phantom: self._phantom.clone(),
        }
    }
}

impl<S: DeviceState + 'static> DeviceContext2<S> {
    /// Creates or retrieves a context for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    pub fn new(device_id: DeviceId) -> Self {
        let mut guard = RUNNERS.lock();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        };
        let runners: &mut HashMap<_, _> = match guard.as_mut() {
            Some(val) => val,
            None => unreachable!(),
        };

        let device_sender = runners.get(&device_id);
        let sender = match device_sender {
            Some(sender) => sender.clone(),
            None => {
                let sender = DeviceRunner::start(device_id);
                runners.insert(device_id, sender.clone());
                sender
            }
        };

        Self {
            sender,
            device_id,
            _phantom: PhantomData,
        }
    }

    /// Executes a task on the dedicated device thread.
    ///
    /// This provides mutable access to the device's state `S`. The state is
    /// lazily initialized on the device thread if it hasn't been accessed yet.
    pub fn execute<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let device_id = self.device_id.clone();

        // The inner closure that will run on the device thread
        let func_init = move || {
            let type_id = TypeId::of::<S>();

            // Access or initialize the state inside Thread Local Storage
            let state = STATES.with_borrow_mut(|map| {
                map.entry(type_id)
                    .or_insert_with(|| {
                        let state = S::init(device_id);
                        Rc::new(RefCell::new(Box::new(state)))
                    })
                    .clone()
            });

            let mut state_borrow = match state.try_borrow_mut() {
                Ok(val) => val,
                Err(_) => {
                    panic!(
                        "State '{}' is already borrowed by another task.",
                        core::any::type_name::<S>()
                    )
                }
            };
            let state = state_borrow
                .downcast_mut::<S>()
                .expect("State type mismatch in Thread Local Storage");

            task(state);
        };

        self.send(func_init);
    }

    /// Internal helper to route the task.
    ///
    /// If we are already on the device runner thread, it executes immediately
    /// to allow for recursion. Otherwise, it sends the task to the runner.
    fn send<T: FnOnce() + Send + 'static>(&self, task: T) {
        match is_device_runner_thread(&self.device_id) {
            false => {
                self.sender
                    .send(Message {
                        task: Box::new(task),
                    })
                    .unwrap();
            }
            true => {
                task();
            }
        }
    }
}

/// Checks if the current thread is the dedicated runner for the given device.
fn is_device_runner_thread(device_id: &DeviceId) -> bool {
    SERVER_THREAD.with_borrow(|state| match state {
        Some(id) => id == device_id,
        None => false,
    })
}

std::thread_local! {
    /// Stores the DeviceId associated with the current runner thread.
    static SERVER_THREAD: RefCell<Option<DeviceId>> = RefCell::new(None);

    /// Stores the various states (indexed by TypeId) owned by the current thread.
    static STATES: RefCell<HashMap<TypeId, Rc<RefCell<Box<dyn Any + 'static>>>>> = RefCell::new(HashMap::new());
}

struct DeviceRunner {}

/// Message packet containing the closure to be executed.
struct Message {
    task: Box<dyn FnOnce() + Send>,
}

/// Global registry of device runners.
static RUNNERS: spin::Mutex<Option<HashMap<DeviceId, SyncSender<Message>>>> =
    spin::Mutex::new(None);

impl DeviceRunner {
    /// Spawns a new background thread to handle tasks for a specific device.
    pub fn start(device_id: DeviceId) -> SyncSender<Message> {
        let (sender, recv) = std::sync::mpsc::sync_channel::<Message>(100);

        let (sender_init, recv_init) = std::sync::mpsc::sync_channel::<()>(1);

        std::thread::spawn(move || {
            // Mark this thread as belonging to the device
            SERVER_THREAD.with_borrow_mut(move |cell| *cell = Some(device_id));
            sender_init.send(()).unwrap();

            for message in recv {
                let task = message.task;
                task();
            }
        });

        // Waits for the device thread to be init.
        if !recv_init.recv().is_ok() {
            panic!("Can't start the new runner thread");
        }

        sender
    }
}

#[cfg(test)]
mod tests {
    use crate::device::Device;

    use super::*;

    #[test]
    fn test_concurrent_increment() {
        let device = TestDevice::<1>::new(0);
        let context = DeviceContext2::<TestDeviceState<1>>::new(device.to_id());

        let thread_count = 10;
        let mut handles = Vec::new();

        for _ in 0..thread_count {
            let ctx = context.clone();
            handles.push(std::thread::spawn(move || {
                ctx.execute(|state| {
                    state.counter += 1;
                });
            }));
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify final result
        let (tx, rx) = std::sync::mpsc::channel();
        context.execute(move |state| {
            tx.send(state.counter).unwrap();
        });

        assert_eq!(rx.recv().unwrap(), thread_count);
    }

    #[test]
    #[should_panic]
    fn test_recursive_execution_same_state() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 5,
        };
        let context = DeviceContext2::<TestDeviceState<1>>::new(device_id);
        let context_cloned = context.clone();

        // Verify final result
        let (tx, rx) = std::sync::mpsc::channel();

        context.execute(move |state| {
            state.counter += 1;
            context_cloned.execute(move |state| {
                tx.send(state.counter).unwrap();
            });
        });

        rx.recv().unwrap();
    }

    #[test]
    fn test_recursive_execution_different_state() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 5,
        };
        let context = DeviceContext2::<TestDeviceState<1>>::new(device_id);
        let context_second = DeviceContext2::<TestDeviceState<2>>::new(device_id);

        context.execute(move |_state| {
            context_second.execute(move |_inner_state| {});
        });
    }

    #[derive(Debug, Clone, Default, new)]
    /// Type is only to create different type ids.
    pub struct TestDevice<const TYPE: u8> {
        index: u32,
    }

    pub struct TestDeviceState<const T: usize> {
        counter: usize,
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

    impl<const T: usize> DeviceState for TestDeviceState<T> {
        fn init(_device_id: DeviceId) -> Self {
            TestDeviceState { counter: 0 }
        }
    }
}
