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
pub struct ChannelDeviceHandle<S: DeviceService> {
    sender: ChannelClient,
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

        let any = Box::new(service);
        let init = move || {
            let type_id = TypeId::of::<S>();
            // Access or initialize the state inside Thread Local Storage
            STATES.with_borrow_mut(|map| {
                if map.contains_key(&type_id) {
                    sender.send(Err(())).unwrap();
                } else {
                    map.insert(type_id, Rc::new(RefCell::new(any)));
                    sender.send(Ok(())).unwrap();
                }
            });
        };
        this.sender.enqueue(init);
        this.sender.flush();

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
        self.sender.flush();

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
        self.sender.flush();

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
        self.sender.flush();

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
        self.sender.flush();

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
                self.sender.enqueue(task);
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

/// Global registry of device runners.
static RUNNERS: spin::Mutex<Option<HashMap<DeviceId, ChannelClient>>> = spin::Mutex::new(None);

impl DeviceRunner {
    /// Spawns a new background thread to handle tasks for a specific device.
    pub fn start(device_id: DeviceId) -> ChannelClient {
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

    pub static TASK_MAX_SIZE: usize = 64 * 5;
    pub type TaskData = [u8; TASK_MAX_SIZE];

    /// A task that can hold a closure, either inline (no allocation) or on the heap.
    pub struct Task {
        // Shim to call the closure.
        call_fn_ptr: unsafe fn(*mut u8),
        data_ptr: TaskData,
    }

    unsafe fn call_shim<F: FnOnce() + 'static>(ptr: *mut u8) {
        // SAFETY: F is guaranteed to be stored at ptr and moved only once.
        unsafe {
            let f = std::ptr::read(ptr as *mut F);
            f();
        }
    }

    impl Task {
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
            // TODO: That could be inlined inside the channel and we could write the data
            // directly into dynamic memory (the channel queue) instead.
            unsafe {
                std::ptr::write(self.data_ptr.as_mut_ptr() as *mut F, f);
                self.call_fn_ptr = call_shim::<F>;
            };
        }

        pub fn run(&mut self) {
            unsafe {
                (self.call_fn_ptr)(self.data_ptr.as_mut_ptr() as *mut u8);
            }
        }
    }
}

mod channel {
    use alloc::boxed::Box;
    use core::{
        hint::spin_loop,
        sync::atomic::{AtomicPtr, AtomicU32, Ordering},
    };
    use std::{sync::Arc, vec::Vec};

    use crate::device::handle::channel::task::Task;

    pub const CHANNEL_MAX_TASK: usize = 32;

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
        pub fn new<I: FnOnce() + Send + 'static>(init: I) -> Self {
            let mut server = Server::new();
            let state = server.state.clone();

            std::thread::spawn(move || {
                init();
                server.run();
            });

            Self { state }
        }

        pub fn enqueue<F: FnOnce() + Send + 'static>(&self, func: F) {
            loop {
                let index = self.state.index.fetch_add(1, Ordering::Acquire) as usize;
                if index >= CHANNEL_MAX_TASK {
                    spin_loop();
                    continue;
                }

                let ptr = self.state.ptr.load(Ordering::Relaxed);
                unsafe {
                    let task = ptr.offset(index as isize).as_mut().unwrap();
                    task.init(func);
                };

                self.state.success.fetch_add(1, Ordering::Acquire);
                return;
            }
        }

        pub fn flush(&self) {
            // We use `state.success` since it is guaranteed to be max `CHANNEL_MAX_TASK`.
            //
            // `state.index` can be higher.
            let num_tasks_required =
                CHANNEL_MAX_TASK - self.state.success.load(Ordering::Acquire) as usize;

            // We fetch the index start based on `state.index`.
            let index_start = self
                .state
                .index
                .fetch_add(num_tasks_required as u32, Ordering::Acquire)
                as usize;

            // Since the call to flush, other threads might have enqueue enough tasks to trigger a
            // flush, in this case we can exit.
            if index_start >= CHANNEL_MAX_TASK {
                return;
            }

            // We may have required too many task to be sent, we trim the number of tasks.
            let index_end = CHANNEL_MAX_TASK;

            let ptr = self.state.ptr.load(Ordering::Relaxed);
            for index in index_start..index_end {
                unsafe {
                    let task = ptr.offset(index as isize).as_mut().unwrap();
                    task.init(|| {});
                };
            }

            // We add the number of empty tasks that we sent to trigger the flushing.
            let num_tasks_required_true = index_end - index_start;
            self.state
                .success
                .fetch_add(num_tasks_required_true as u32, Ordering::Acquire);
        }
    }

    struct State {
        ptr: AtomicPtr<Task>,
        index: AtomicU32,
        success: AtomicU32,
    }

    struct Server {
        state: Arc<State>,
        ptr_client: *mut Task,
        ptr_server: *mut Task,
        num_remaining: usize,
        drop: Box<dyn FnOnce()>,
        data_index: bool,
    }

    unsafe impl Send for Server {}

    impl Server {
        fn new() -> Self {
            // We use `Vec` because it is convenient.
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
                drop: Box::new(move || {
                    core::mem::drop(data_1_vec);
                    core::mem::drop(data_2_vec);
                }),
                data_index: true,
            }
        }

        fn run(&mut self) {
            loop {
                if self.num_remaining != 0 {
                    self.execute_tasks();
                } else if self.state.success.load(Ordering::Relaxed) as usize >= CHANNEL_MAX_TASK {
                    self.fetch();
                } else {
                    spin_loop();
                }
            }
        }

        fn execute_tasks(&mut self) {
            for cursor in 0..self.num_remaining {
                let mut task = unsafe { self.ptr_server.offset(cursor as isize).read() };
                task.run();
            }
            self.num_remaining = 0;
        }

        fn fetch(&mut self) {
            let remaining = self.state.success.load(Ordering::Acquire);
            core::mem::swap(&mut self.ptr_client, &mut self.ptr_server);

            self.state.ptr.store(self.ptr_client, Ordering::Release);
            self.data_index = !self.data_index;
            self.num_remaining = remaining as usize;

            self.state.success.store(0, Ordering::Release);
            self.state.index.store(0, Ordering::Release);
        }
    }
}
