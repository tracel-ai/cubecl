use crate::device::{
    DeviceId, DeviceService,
    handle::{CallError, DeviceHandleSpec, ServiceCreationError},
};
use hashbrown::HashMap;
use oneshot::Sender;
use std::{
    any::{Any, TypeId},
    boxed::Box,
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::mpsc::SyncSender,
};

/// A handle to a specific device context.
///
/// This struct allows sending closures to be executed on a dedicated
/// thread for the specific device, ensuring thread-safe access to
/// the device's state (`S`).
pub struct ChannelDeviceHandle<S: DeviceService> {
    sender: SyncSender<Message>,
    device_id: DeviceId,
    _phantom: PhantomData<S>,
}

/// We can sync the device handle, but not the service, which is why we have to unsafly
/// implement the trait.
unsafe impl<S: DeviceService> Sync for ChannelDeviceHandle<S> {}

impl<S: DeviceService + 'static> DeviceHandleSpec<S> for ChannelDeviceHandle<S> {
    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        let this = Self::new(device_id);
        let (sender, recv) = oneshot::channel();
        this.sender
            .send(Message::Init(TypeId::of::<S>(), Box::new(service), sender))
            .unwrap();

        if recv.recv().is_err() {
            return Err(ServiceCreationError::new("The service is dead".into()));
        };

        Ok(this)
    }

    fn new(device_id: DeviceId) -> Self {
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

    fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        // 1. Wrap the task and sender into a single closure
        self.submit(move |state: &mut S| {
            let returned = task(state);
            sender.send(returned).unwrap();
        });

        // 4. Block until finished
        recv.recv().map_err(|_| CallError)
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
            unsafe { std::mem::transmute(boxed) };

        self.submit(static_task);

        recv.recv().unwrap()
    }

    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let device_id = self.device_id;

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

    fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        self.send(move || {
            let returned = task();
            sender.send(returned).unwrap();
        });

        recv.recv().map_err(|_| CallError)
    }

    /// TODO: Docs.
    fn exclusive_scoped<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        // 1. Wrap the task and sender into a single closure
        let wrapper = move || {
            let returned = task();
            sender.send(returned).unwrap();
        };

        // 2. Erase the lifetime using transmute to make it 'static
        // We use Box first to get a stable pointer size
        //
        // This is safe if we ensure the function is actually called BEFORE the end of this
        // function. Which is the case if we don't have any error.
        let boxed: Box<dyn FnOnce() + Send> = Box::new(wrapper);

        let static_task: Box<dyn FnOnce() + Send + 'static> = unsafe { std::mem::transmute(boxed) };

        self.send(static_task);

        recv.recv().map_err(|_| CallError)
    }
}

impl<S: DeviceService + 'static> ChannelDeviceHandle<S> {
    /// Send a task to the device working thread.
    ///
    /// If we are already on the device runner thread, it executes immediately
    /// to allow for recursion. Otherwise, it sends the task to the runner.
    fn send<T: FnOnce() + Send + 'static>(&self, task: T) {
        match is_device_runner_thread(&self.device_id) {
            false => {
                self.sender.send(Message::Task(Box::new(task))).unwrap();
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
    static SERVER_THREAD: RefCell<Option<DeviceId>> = const { RefCell::new(None) };

    /// Stores the various states (indexed by TypeId) owned by the current thread.
    #[allow(clippy::type_complexity)]
    static STATES: RefCell<HashMap<TypeId, Rc<RefCell<Box<dyn Any + 'static>>>>> = RefCell::new(HashMap::new());
}

struct DeviceRunner {}

/// Message packet containing the closure to be executed.
enum Message {
    Task(Box<dyn FnOnce() + Send>),
    Init(TypeId, Box<dyn Any + Send>, Sender<Result<(), ()>>),
}

/// Global registry of device runners.
static RUNNERS: spin::Mutex<Option<HashMap<DeviceId, SyncSender<Message>>>> =
    spin::Mutex::new(None);

impl DeviceRunner {
    /// Spawns a new background thread to handle tasks for a specific device.
    pub fn start(device_id: DeviceId) -> SyncSender<Message> {
        // A maximum of 1024 tasks can be queued at the same time on a channel.
        let (sender, recv) = std::sync::mpsc::sync_channel::<Message>(128);
        let (sender_init, recv_init) = oneshot::channel();

        std::thread::spawn(move || {
            // Marks this thread as belonging to the device
            SERVER_THREAD.with_borrow_mut(move |cell| *cell = Some(device_id));
            sender_init.send(()).unwrap();

            loop {
                if let Ok(message) = recv.try_recv() {
                    match message {
                        Message::Task(task) => task(),
                        Message::Init(type_id, any, sender) => {
                            // Access or initialize the state inside Thread Local Storage
                            STATES.with_borrow_mut(|map| {
                                if map.contains_key(&type_id) {
                                    sender.send(Err(())).unwrap();
                                } else {
                                    map.insert(type_id, Rc::new(RefCell::new(any)));
                                    sender.send(Ok(())).unwrap();
                                }
                            });
                        }
                    }
                }
            }
        });

        // Waits for the device thread to be init.
        if recv_init.recv().is_err() {
            panic!("Can't start the new runner thread");
        }

        sender
    }
}

impl<S: DeviceService> Clone for ChannelDeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            device_id: self.device_id,
            _phantom: self._phantom,
        }
    }
}
