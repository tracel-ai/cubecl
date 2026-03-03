use crate::device::{
    DeviceId, DeviceService,
    handle::{CallError, DeviceHandleSpec, ServiceCreationError, channel::channel::ChannelClient},
};
use hashbrown::HashMap;
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
///
/// The `ChannelDeviceHandle` acts as a proxy; it doesn't hold the state `S`
/// itself, but rather a communication channel to the thread where `S` lives.
pub struct ChannelDeviceHandle<S: DeviceService> {
    sender: ChannelClient,
    device_id: DeviceId,
    _phantom: PhantomData<S>,
}

/// SAFETY: While `DeviceService` (S) may not be `Sync`, the `ChannelDeviceHandle`
/// only sends tasks to a dedicated thread where `S` is accessed in a single-threaded
/// manner. Thus, the handle itself is safe to share across threads.
unsafe impl<S: DeviceService> Sync for ChannelDeviceHandle<S> {}

impl<S: DeviceService + 'static> DeviceHandleSpec<S> for ChannelDeviceHandle<S> {
    /// Registers a new service instance for the device and returns a handle.
    ///
    /// If the service type `S` is already initialized on the device's runner thread,
    /// this will return an error.
    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        let this = Self::new(device_id);
        let (sender, recv) = oneshot::channel();

        let any = Box::new(service);
        let init = move || {
            let type_id = TypeId::of::<S>();
            // Access or initialize the state inside Thread Local Storage on the runner thread
            STATES.with_borrow_mut(|map| {
                if map.contains_key(&type_id) {
                    let _ = sender.send(Err(()));
                } else {
                    map.insert(type_id, Rc::new(RefCell::new(any)));
                    let _ = sender.send(Ok(()));
                }
            });
        };
        this.sender.enqueue(init);
        this.sender.flush();

        if recv.recv().is_err() {
            return Err(ServiceCreationError::new(
                "The service initialization failed or the runner is dead".into(),
            ));
        };

        Ok(this)
    }

    /// Creates a handle for an existing device or starts a new `DeviceRunner` if one
    /// does not exist for the given `device_id`.
    fn new(device_id: DeviceId) -> Self {
        let mut guard = RUNNERS.lock();
        let runners = guard.get_or_insert_with(HashMap::new);

        let sender = runners
            .entry(device_id)
            .or_insert_with(|| DeviceRunner::start(device_id))
            .clone();

        Self {
            sender,
            device_id,
            _phantom: PhantomData,
        }
    }

    /// Submits a task to the device thread and blocks the current thread until
    /// the task returns a result.
    fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        self.submit(move |state: &mut S| {
            let returned = task(state);
            let _ = sender.send(returned);
        });
        self.sender.flush();

        recv.recv().map_err(|_| CallError)
    }

    /// Submits a task with a non-static lifetime.
    ///
    /// # Safety
    /// This uses `transmute` to erase lifetimes. This is safe only because the
    /// current thread blocks on `recv.recv()`, ensuring the task is executed
    /// and finished before the scope of the task's captured variables ends.
    fn submit_blocking_scoped<'a, R: Send + 'a, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> R {
        let (sender, recv) = oneshot::channel();

        let wrapper = move |state: &mut S| {
            let returned = task(state);
            let _ = sender.send(returned);
        };

        let boxed: Box<dyn for<'s> FnOnce(&'s mut S) + Send> = Box::new(wrapper);

        // SAFETY: The recv.recv() below ensures the 'static requirement is
        // effectively met by blocking the caller's stack frame.
        let static_task: Box<dyn for<'s> FnOnce(&'s mut S) + Send + 'static> =
            unsafe { std::mem::transmute(boxed) };

        self.submit(static_task);
        self.sender.flush();

        recv.recv()
            .expect("Scoped task failed: Runner disconnected")
    }

    /// Asynchronously dispatches a task to the device thread.
    ///
    /// This method retrieves the service state `S` from the runner's TLS.
    /// If `S` is not yet initialized, it calls `S::init`.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        let device_id = self.device_id;

        let func_init = move || {
            let type_id = TypeId::of::<S>();

            let state_rc = STATES.with_borrow_mut(|map| {
                map.entry(type_id)
                    .or_insert_with(|| {
                        let state = S::init(device_id);
                        Rc::new(RefCell::new(Box::new(state)))
                    })
                    .clone()
            });

            let mut state_borrow = state_rc.try_borrow_mut().unwrap_or_else(|_| {
                panic!(
                    "State '{}' is already borrowed.",
                    core::any::type_name::<S>()
                )
            });

            let state = state_borrow
                .downcast_mut::<S>()
                .expect("State type mismatch in Thread Local Storage");

            task(state);
        };

        self.send(func_init);
    }

    /// Executes a task on the device thread that does not require direct
    /// access to the `DeviceService` state, blocking until completion.
    fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        self.send(move || {
            let returned = task();
            let _ = sender.send(returned);
        });
        self.sender.flush();

        recv.recv().map_err(|_| CallError)
    }

    /// Executes a closure with a captured scope on the device thread.
    ///
    /// Blocks until the task is complete. Useful for operations that need to
    /// run on the dedicated thread but reference data on the caller's stack.
    fn exclusive_scoped<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        let wrapper = move || {
            let returned = task();
            let _ = sender.send(returned);
        };

        let boxed: Box<dyn FnOnce() + Send> = Box::new(wrapper);
        // SAFETY: Blocking on `recv` guarantees the closure finishes before the scope ends.
        let static_task: Box<dyn FnOnce() + Send + 'static> = unsafe { std::mem::transmute(boxed) };

        self.send(static_task);
        self.sender.flush();

        recv.recv().map_err(|_| CallError)
    }
}

impl<S: DeviceService + 'static> ChannelDeviceHandle<S> {
    /// Dispatches a task to the runner.
    ///
    /// If the current thread is already the runner for this device, it executes
    /// immediately to prevent deadlocks and allow for recursive calls.
    fn send<T: FnOnce() + Send + 'static>(&self, task: T) {
        if is_device_runner_thread(&self.device_id) {
            task();
        } else {
            self.sender.enqueue(task);
        }
    }
}

/// Helper to verify if the current execution context is the device's runner thread.
fn is_device_runner_thread(device_id: &DeviceId) -> bool {
    SERVER_THREAD.with_borrow(|state| state.as_ref() == Some(device_id))
}

std::thread_local! {
    /// The ID of the device this thread is responsible for.
    static SERVER_THREAD: RefCell<Option<DeviceId>> = const { RefCell::new(None) };

    /// Heterogeneous map of service states owned by this thread.
    #[allow(clippy::type_complexity)]
    static STATES: RefCell<HashMap<TypeId, Rc<RefCell<Box<dyn Any + 'static>>>>> = RefCell::new(HashMap::new());
}

/// Internal runner logic to manage background thread spawning.
struct DeviceRunner {}

static RUNNERS: spin::Mutex<Option<HashMap<DeviceId, ChannelClient>>> = spin::Mutex::new(None);

impl DeviceRunner {
    /// Spawns a new thread, marks it with the `device_id`, and returns a `ChannelClient`.
    pub fn start(device_id: DeviceId) -> ChannelClient {
        let (sender_init, recv_init) = oneshot::channel();
        let channel = ChannelClient::new(move || {
            SERVER_THREAD.with_borrow_mut(|cell| *cell = Some(device_id));
            sender_init.send(()).unwrap();
        });

        if recv_init.recv().is_err() {
            panic!("Failed to synchronize device runner thread initialization");
        }

        channel
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

mod task {
    use super::*;
    use std::mem::size_of;

    /// The maximum size of a closure that can be stored without heap allocation.
    pub static TASK_MAX_SIZE: usize = 64 * 5;
    pub type TaskData = [u8; TASK_MAX_SIZE];

    /// A type-erased task container.
    ///
    /// If the closure fits within `TASK_MAX_SIZE`, it is stored inline.
    /// Otherwise, it is boxed and the box is stored inline.
    pub struct Task {
        call_fn_ptr: unsafe fn(*mut u8),
        data_ptr: TaskData,
    }

    unsafe fn call_shim<F: FnOnce() + 'static>(ptr: *mut u8) {
        let f = unsafe { std::ptr::read(ptr as *mut F) };
        f();
    }

    impl Task {
        /// Initializes the task with a closure.
        pub fn init<F: FnOnce() + 'static + Send>(&mut self, f: F) {
            if size_of::<F>() <= TASK_MAX_SIZE {
                self.init_inner(f);
            } else {
                let func: Box<dyn FnOnce() + 'static + Send> = Box::new(f);
                self.init_inner(move || {
                    func();
                });
            }
        }

        fn init_inner<F: FnOnce() + 'static + Send>(&mut self, f: F) {
            unsafe {
                std::ptr::write(self.data_ptr.as_mut_ptr() as *mut F, f);
                self.call_fn_ptr = call_shim::<F>;
            };
        }

        /// Executes the stored task.
        pub fn run(&mut self) {
            unsafe {
                (self.call_fn_ptr)(self.data_ptr.as_mut_ptr() as *mut u8);
            }
        }
    }
}

mod channel {
    use crate::device::handle::channel::task::Task;
    use alloc::boxed::Box;
    use core::{
        hint::spin_loop,
        sync::atomic::{AtomicPtr, AtomicU32, Ordering},
    };
    use std::{sync::Arc, vec::Vec};

    /// Buffer size for the command channel.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// The client-side handle used to enqueue tasks.
    pub struct ChannelClient {
        state: Arc<State>,
    }

    unsafe impl Send for ChannelClient {}
    unsafe impl Sync for ChannelClient {}

    impl Clone for ChannelClient {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl ChannelClient {
        /// Creates a new channel and spawns a server thread to process it.
        pub fn new<I: FnOnce() + Send + 'static>(init: I) -> Self {
            let mut server = Server::new();
            let state = server.state.clone();

            std::thread::spawn(move || {
                init();
                server.run();
            });

            Self { state }
        }

        /// Atomically reserves a slot in the buffer and writes the task.
        pub fn enqueue<F: FnOnce() + Send + 'static>(&self, func: F) {
            loop {
                let index = self.state.index.fetch_add(1, Ordering::Acquire) as usize;
                if index >= CHANNEL_MAX_TASK {
                    // Buffer full; spin until the server flushes/swaps buffers.
                    spin_loop();
                    continue;
                }

                let ptr = self.state.ptr.load(Ordering::Relaxed);
                unsafe {
                    let task = ptr.add(index).as_mut().unwrap();
                    task.init(func);
                };

                self.state.success.fetch_add(1, Ordering::Release);
                return;
            }
        }

        /// Forces a flush by filling the remaining buffer with no-op tasks.
        pub fn flush(&self) {
            let success_count = self.state.success.load(Ordering::Acquire) as usize;
            if success_count == 0 {
                return;
            }

            let num_tasks_required = CHANNEL_MAX_TASK - success_count;
            let index_start = self
                .state
                .index
                .fetch_add(num_tasks_required as u32, Ordering::Acquire)
                as usize;

            if index_start >= CHANNEL_MAX_TASK {
                return;
            }

            let ptr = self.state.ptr.load(Ordering::Relaxed);
            for index in index_start..CHANNEL_MAX_TASK {
                unsafe {
                    let task = ptr.add(index).as_mut().unwrap();
                    task.init(|| {});
                };
            }

            let actual_added = CHANNEL_MAX_TASK - index_start;
            self.state
                .success
                .fetch_add(actual_added as u32, Ordering::Release);
        }
    }

    struct State {
        /// Pointer to the current active client buffer.
        ptr: AtomicPtr<Task>,
        /// Next available index for writing.
        index: AtomicU32,
        /// Number of tasks successfully written and ready for processing.
        success: AtomicU32,
    }

    /// The server-side runner that processes tasks.
    struct Server {
        state: Arc<State>,
        ptr_client: *mut Task,
        ptr_server: *mut Task,
        num_remaining: usize,
        _drop_guard: Box<dyn FnOnce()>, // Ensures Vecs are cleaned up
        data_index: bool,
    }

    unsafe impl Send for Server {}

    impl Server {
        fn new() -> Self {
            let mut data_1_vec: Vec<Task> = Vec::with_capacity(CHANNEL_MAX_TASK);
            let mut data_2_vec: Vec<Task> = Vec::with_capacity(CHANNEL_MAX_TASK);

            let data_client = data_1_vec.as_mut_ptr();
            let data_server = data_2_vec.as_mut_ptr();

            let state = Arc::new(State {
                ptr: AtomicPtr::new(data_client),
                index: AtomicU32::new(0),
                success: AtomicU32::new(0),
            });

            Self {
                state,
                num_remaining: 0,
                ptr_client: data_client,
                ptr_server: data_server,
                _drop_guard: Box::new(move || {
                    drop(data_1_vec);
                    drop(data_2_vec);
                }),
                data_index: true,
            }
        }

        /// Main execution loop for the device thread.
        fn run(&mut self) {
            loop {
                if self.num_remaining != 0 {
                    self.execute_tasks();
                } else if self.state.success.load(Ordering::Acquire) as usize >= CHANNEL_MAX_TASK {
                    // Swap buffers when the client has filled the current one.
                    self.fetch();
                } else {
                    spin_loop();
                }
            }
        }

        fn execute_tasks(&mut self) {
            for cursor in 0..self.num_remaining {
                let mut task = unsafe { self.ptr_server.add(cursor).read() };
                task.run();
            }
            self.num_remaining = 0;
        }

        /// Swaps the client and server pointers, allowing the client to start
        /// filling the next buffer while the server processes the current one.
        fn fetch(&mut self) {
            let remaining = self.state.success.load(Ordering::Acquire);
            core::mem::swap(&mut self.ptr_client, &mut self.ptr_server);

            self.state.ptr.store(self.ptr_client, Ordering::Release);
            self.data_index = !self.data_index;
            self.num_remaining = remaining as usize;

            // Reset indices for the new client buffer
            self.state.success.store(0, Ordering::Release);
            self.state.index.store(0, Ordering::Release);
        }
    }
}
