use crate::device::{
    DeviceId, DeviceService,
    handle::{
        CallError, DeviceHandleSpec, ServiceCreationError,
        channel::{
            channel::{CHANNEL_MAX_TASK, ChannelClient, ChannelTask},
            tasks::Task,
        },
    },
};
use hashbrown::HashMap;
use oneshot::Sender;
use std::{
    any::{Any, TypeId},
    boxed::Box,
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
};

/// A handle to a specific device context.
///
/// This struct allows sending closures to be executed on a dedicated
/// thread for the specific device, ensuring thread-safe access to
/// the device's state (`S`).
pub struct ChannelDeviceHandle<S: DeviceService> {
    sender: ChannelClient<Message>,
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
            .enqueue(Message::Init(TypeId::of::<S>(), Box::new(service), sender));

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
        self.flush();

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
        self.flush();

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
        self.flush();

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
        self.flush();

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
                self.sender.enqueue(Message::Task(Task::new(task)));
            }
            true => {
                task();
            }
        }
    }
    fn flush(&self) {
        let num_tasks_required = CHANNEL_MAX_TASK - self.sender.size();

        for _ in 0..num_tasks_required {
            self.sender.enqueue(Message::Empty);
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
    Task(tasks::Task),
    Init(TypeId, Box<dyn Any + Send>, Sender<Result<(), ()>>),
    Empty,
}

impl ChannelTask for Message {
    fn run(self) {
        match self {
            Message::Task(task) => task.run(),
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
            Message::Empty => {}
        }
    }

    fn empty() -> Self {
        Self::Empty
    }
}

/// Global registry of device runners.
static RUNNERS: spin::Mutex<Option<HashMap<DeviceId, ChannelClient<Message>>>> =
    spin::Mutex::new(None);

impl DeviceRunner {
    /// Spawns a new background thread to handle tasks for a specific device.
    pub fn start(device_id: DeviceId) -> ChannelClient<Message> {
        let (sender_init, recv_init) = oneshot::channel();
        let channel = ChannelClient::new(move || {
            // Marks this thread as belonging to the device
            SERVER_THREAD.with_borrow_mut(move |cell| *cell = Some(device_id));
            sender_init.send(()).unwrap();
        });

        // Waits for the device thread to be init.
        if recv_init.recv().is_err() {
            panic!("Can't start the new runner thread");
        }

        channel
        // // A maximum of 1024 tasks can be queued at the same time on a channel.
        // let (sender, recv) = std::sync::mpsc::sync_channel::<Message>(128);
        // let (sender_init, recv_init) = oneshot::channel();

        // std::thread::spawn(move || {
        //     // Marks this thread as belonging to the device
        //     SERVER_THREAD.with_borrow_mut(move |cell| *cell = Some(device_id));
        //     sender_init.send(()).unwrap();

        //     loop {
        //         if let Ok(message) = recv.try_recv() {
        //             match message {
        //                 Message::Task(task) => task.run(),
        //                 Message::Init(type_id, any, sender) => {
        //                     // Access or initialize the state inside Thread Local Storage
        //                     STATES.with_borrow_mut(|map| {
        //                         if map.contains_key(&type_id) {
        //                             sender.send(Err(())).unwrap();
        //                         } else {
        //                             map.insert(type_id, Rc::new(RefCell::new(any)));
        //                             sender.send(Ok(())).unwrap();
        //                         }
        //                     });
        //                 }
        //             }
        //         }
        //     }
        // });

        // // Waits for the device thread to be init.
        // if recv_init.recv().is_err() {
        //     panic!("Can't start the new runner thread");
        // }

        // sender
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

mod tasks {
    use super::*;
    use std::mem::size_of;

    pub static TASK_MAX_SIZE: usize = 264;
    pub type TaskData = [u8; TASK_MAX_SIZE];

    /// A task that can hold a closure, either inline (no allocation) or on the heap.
    pub enum Task {
        Inline {
            // inline storage, aligned to usize.
            data: TaskData,
            // Shim to call the closure.
            call_fn: unsafe fn(*mut u8),
        },
        Boxed(Box<dyn FnOnce() + Send>),
    }

    impl Task {
        pub fn new<F: FnOnce() + 'static + Send>(f: F) -> Self {
            if size_of::<F>() <= TASK_MAX_SIZE {
                unsafe fn call_shim<F: FnOnce() + 'static>(ptr: *mut u8) {
                    // SAFETY: F is guaranteed to be stored at ptr and moved only once.
                    unsafe {
                        let f = std::ptr::read(ptr as *mut F);
                        f();
                    }
                }

                let mut data = [0; TASK_MAX_SIZE];
                // TODO: That could be inlined inside the channel and we could write the data
                // directly into dynamic memory (the channel queue) instead.
                unsafe {
                    std::ptr::write(data.as_mut_ptr() as *mut F, f);
                }

                Task::Inline {
                    data,
                    call_fn: call_shim::<F>,
                }
            } else {
                Task::Boxed(Box::new(f))
            }
        }

        pub fn run(self) {
            match self {
                Task::Inline { mut data, call_fn } => unsafe {
                    (call_fn)(data.as_mut_ptr() as *mut u8);
                },
                Task::Boxed(func) => {
                    func();
                }
            }
        }
    }
}

mod channel {
    use alloc::boxed::Box;
    use core::{
        hint::spin_loop,
        sync::atomic::{AtomicBool, AtomicU32, Ordering},
    };
    use std::{sync::Arc, vec::Vec};

    pub const CHANNEL_MAX_TASK: usize = 32;

    pub trait ChannelTask: Send + 'static {
        fn run(self);
        fn empty() -> Self;
    }

    pub struct ChannelClient<T: ChannelTask> {
        state: Arc<State<T>>,
    }

    unsafe impl<T: ChannelTask> Send for ChannelClient<T> {}
    unsafe impl<T: ChannelTask> Sync for ChannelClient<T> {}

    impl<T: ChannelTask> Clone for ChannelClient<T> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<T: ChannelTask> ChannelClient<T> {
        pub fn new<I: FnOnce() + Send + 'static>(init: I) -> Self {
            let mut server = Server::new();
            let state = server.state.clone();

            std::thread::spawn(move || {
                init();
                server.run();
            });

            Self { state }
        }

        pub fn enqueue(&self, task: T) {
            loop {
                let index = self.state.index.fetch_add(1, Ordering::Acquire) as usize;
                if index >= CHANNEL_MAX_TASK {
                    spin_loop();
                    continue;
                }

                let ptr = if self.state.data_index.load(Ordering::Relaxed) {
                    self.state.data_1
                } else {
                    self.state.data_2
                };
                unsafe {
                    ptr.offset(index as isize).write(task);
                };

                self.state.success.fetch_add(1, Ordering::Acquire);
                return;
            }
        }

        pub fn size(&self) -> usize {
            self.state.success.load(Ordering::Relaxed) as usize
        }
    }

    struct State<T: ChannelTask> {
        data_1: *mut T,
        data_2: *mut T,
        index: AtomicU32,
        data_index: AtomicBool,
        success: AtomicU32,
    }

    struct Server<T: ChannelTask> {
        state: Arc<State<T>>,
        total: usize,
        drop: Box<dyn FnOnce()>,
        data_index: bool,
    }

    unsafe impl<T: ChannelTask> Send for Server<T> {}

    impl<T: ChannelTask> Server<T> {
        pub fn new() -> Self {
            let mut data_1_vec: Vec<T> = Vec::with_capacity(CHANNEL_MAX_TASK);
            let mut data_2_vec: Vec<T> = Vec::with_capacity(CHANNEL_MAX_TASK);

            for _ in 0..CHANNEL_MAX_TASK {
                data_1_vec.push(T::empty());
                data_2_vec.push(T::empty());
            }

            let data_1 = data_1_vec.as_mut_ptr();
            let data_2 = data_2_vec.as_mut_ptr();
            let state = Arc::new(State {
                data_1,
                data_2,
                index: AtomicU32::new(0),
                success: AtomicU32::new(0),
                data_index: AtomicBool::new(false),
            });

            Self {
                state,
                total: 0,
                drop: Box::new(move || {
                    core::mem::drop(data_1_vec);
                    core::mem::drop(data_2_vec);
                }),
                data_index: true,
            }
        }

        fn run(&mut self) {
            loop {
                if self.total != 0 {
                    self.iteration();
                } else if self.state.success.load(Ordering::Relaxed) as usize >= CHANNEL_MAX_TASK {
                    self.fetch();
                } else {
                    spin_loop();
                }
            }
        }

        fn iteration(&mut self) {
            let ptr = if self.data_index {
                self.state.data_1
            } else {
                self.state.data_2
            };

            for cursor in 0..self.total {
                let task = unsafe { ptr.offset(cursor as isize).read() };
                task.run();
            }
            self.total = 0;
        }

        fn fetch(&mut self) {
            let total = self.state.success.load(Ordering::Acquire);
            self.state
                .data_index
                .store(self.data_index, Ordering::Relaxed);

            self.data_index = !self.data_index;

            self.total = total as usize;

            self.state.success.store(0, Ordering::Release);
            self.state.index.store(0, Ordering::Release);
        }
    }
}
