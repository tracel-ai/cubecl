use crate::device::{
    DeviceId, DeviceService,
    handle::{
        CallError, DeviceHandleSpec, ServiceCreationError,
        channel::task::{TaskError, TaskResult},
    },
};
use hashbrown::HashMap;
use std::{
    any::{Any, TypeId},
    boxed::Box,
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
};

use custom_channel::DeviceClient;

/// A handle to a specific device context.
///
/// This struct allows sending closures to be executed on a dedicated
/// thread for the specific device, ensuring thread-safe access to
/// the device's state (`S`).
///
/// The `ChannelDeviceHandle` acts as a proxy; it doesn't hold the state `S`
/// itself, but rather a communication channel to the thread where `S` lives.
pub struct ChannelDeviceHandle<S: DeviceService> {
    state: ChannelDeviceState,
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
        let state = ChannelDeviceState::init(device_id, Some(service))?;

        Ok(Self {
            state,
            _phantom: PhantomData,
        })
    }

    /// Creates a handle for an existing device or starts a new `DeviceRunner` if one
    /// does not exist for the given `device_id`.
    fn new(device_id: DeviceId) -> Self {
        let state = ChannelDeviceState::init::<S>(device_id, None).unwrap();

        Self {
            state,
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

        self.submit_inner::<_, true>(move |state: &mut S| {
            let returned = task(state);
            sender.send(returned).unwrap();
        });

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

        self.submit_inner::<_, true>(static_task);

        recv.recv()
            .expect("Scoped task failed: Runner disconnected")
    }

    /// Asynchronously dispatches a task to the device thread.
    ///
    /// This method retrieves the service state `S` from the runner's TLS.
    /// If `S` is not yet initialized, it calls `S::init`.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.submit_inner::<_, false>(task);
    }

    /// Executes a task on the device thread that does not require direct
    /// access to the `DeviceService` state, blocking until completion.
    fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        self.send::<_, true>(move || {
            let returned = task();
            let _ = sender.send(returned);
            Ok(())
        })?;

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
            Ok(())
        };

        let boxed: Box<dyn FnOnce() -> TaskResult + Send> = Box::new(wrapper);
        // SAFETY: Blocking on `recv` guarantees the closure finishes before the scope ends.
        let static_task: Box<dyn FnOnce() -> TaskResult + Send + 'static> =
            unsafe { std::mem::transmute(boxed) };

        self.send::<_, true>(static_task)?;

        recv.recv().map_err(|_| CallError)
    }
}

impl<S: DeviceService + 'static> ChannelDeviceHandle<S> {
    /// Asynchronously dispatches a task to the device thread.
    ///
    /// This method retrieves the service state `S` from the runner's TLS.
    /// If `S` is not yet initialized, it calls `S::init`.
    fn submit_inner<T: FnOnce(&mut S) + Send + 'static, const FLUSH: bool>(&self, task: T) {
        let state = self.state.service.clone();

        let func_init = move || {
            let state = state.as_ref();

            let mut state_borrow = match state.try_borrow_mut() {
                Ok(state) => state,
                Err(_) => {
                    log::error!(
                        "State '{}' is already borrowed.",
                        core::any::type_name::<S>()
                    );
                    return Err(TaskError);
                }
            };

            let state = state_borrow
                .downcast_mut::<S>()
                .expect("State type mismatch in Thread Local Storage");

            task(state);
            Ok(())
        };

        self.send::<_, FLUSH>(func_init).unwrap();
    }

    /// Dispatches a task to the runner.
    ///
    /// If the current thread is already the runner for this device, it executes
    /// immediately to prevent deadlocks and allow for recursive calls.
    fn send<T: FnOnce() -> TaskResult + Send + 'static, const FLUSH: bool>(
        &self,
        task: T,
    ) -> Result<(), CallError> {
        if is_device_runner_thread(self.state.client.device_id()) {
            task().expect("Task to not have an error");
            Ok(())
        } else {
            self.state.client.enqueue(task).unwrap();

            if FLUSH {
                self.state.client.flush();
            }

            Ok(())
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

#[derive(Clone)]
/// A simple wrapper over a client and a service that is cached with [`CHANNELS`].
struct ChannelDeviceState {
    client: DeviceClient,
    service: ChannelService,
}

/// SAFETY: This is safe since we ensure the service will only be accessed from the device thread.
/// We use this ptr to avoid an hashmap lookup in thread local memory for every submission.
struct ChannelService {
    ptr: *const RefCell<Box<dyn Any + 'static>>,
}

unsafe impl Send for ChannelService {}
unsafe impl Sync for ChannelService {}

static RUNNERS: spin::Mutex<Option<HashMap<DeviceId, DeviceClient>>> = spin::Mutex::new(None);
static CHANNELS: spin::Mutex<Option<HashMap<(DeviceId, TypeId), ChannelDeviceState>>> =
    spin::Mutex::new(None);

impl ChannelDeviceState {
    pub fn init<S: DeviceService>(
        device_id: DeviceId,
        service: Option<S>,
    ) -> Result<Self, ServiceCreationError> {
        let type_id = TypeId::of::<S>();
        let key = (device_id, type_id);
        let mut guard_channel = CHANNELS.lock();
        let channels = guard_channel.get_or_insert_with(HashMap::new);

        // Most of the time the channel state is already initialized.
        if let Some(value) = channels.get(&key) {
            return Ok(value.clone());
        };

        core::mem::drop(guard_channel);

        // When initializing a service, we first need to make sure the device runner is
        // initialized.
        //
        // # Notes
        //
        // A single device runner can serve multiple [`DeviceService`].
        let mut guard = RUNNERS.lock();
        let runners = guard.get_or_insert_with(HashMap::new);

        let device_client = runners
            .entry(device_id)
            .or_insert_with(|| DeviceRunner::start(device_id))
            .clone();

        core::mem::drop(guard);

        let (callback, recv) = oneshot::channel();

        // The service initialization function.
        let initialize_service = move || {
            STATES.with_borrow_mut(|map| {
                // If a service state is passed as parameter, we enforce it being used.
                if service.is_some() && map.contains_key(&type_id) {
                    callback.send(Err(())).unwrap();
                } else {
                    let service = service.unwrap_or_else(|| S::init(device_id));
                    let state_rc = map
                        .entry(type_id)
                        .or_insert_with(|| Rc::new(RefCell::new(Box::new(service))))
                        .clone();
                    let state = Rc::as_ptr(&state_rc);
                    callback.send(Ok(ChannelService { ptr: state })).unwrap();
                }
            });

            Ok(())
        };

        if is_device_runner_thread(&device_id) {
            initialize_service().unwrap();
        } else {
            device_client.enqueue(initialize_service).unwrap();
            device_client.flush();
        };

        let service = recv.recv().unwrap();

        let service = match service {
            Ok(service) => service,
            Err(_) => {
                return Err(ServiceCreationError::new(
                    "Service already initialized.".into(),
                ));
            }
        };

        let channel = Self {
            client: device_client,
            service,
        };

        let mut guard_channel = CHANNELS.lock();
        let channels = guard_channel.get_or_insert_with(HashMap::new);
        channels.insert(key, channel.clone());

        Ok(channel)
    }
}

impl Clone for ChannelService {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl ChannelService {
    fn as_ref(&self) -> &RefCell<Box<dyn Any + 'static>> {
        unsafe { self.ptr.as_ref() }.unwrap()
    }
}

impl DeviceRunner {
    /// Spawns a new thread, marks it with the `device_id`, and returns a `ChannelClient`.
    pub fn start(device_id: DeviceId) -> DeviceClient {
        let (sender_init, recv_init) = oneshot::channel();
        let channel = DeviceClient::new(device_id, move || {
            SERVER_THREAD.with_borrow_mut(|cell| *cell = Some(device_id));
            sender_init.send(()).unwrap();
            Ok(())
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
            state: self.state.clone(),
            _phantom: self._phantom,
        }
    }
}

mod task {
    use super::*;
    use std::{mem::size_of, vec::Vec};

    /// The maximum size of a closure that can be stored without heap allocation.
    pub const GLOBAL_TASK_MAX_SIZE: usize = 4096;
    /// The maximum size of a closure that can be stored using inlined memory.
    const INLINE_TASK_MAX_SIZE: usize = 48;

    // We use u128 to force aligment.
    pub type TaskData = [u128; GLOBAL_TASK_MAX_SIZE / 16];
    pub type InlineData = [u128; INLINE_TASK_MAX_SIZE / 16];

    #[derive(Debug)]
    pub struct TaskError;

    pub type TaskResult = Result<(), TaskError>;

    #[repr(C, align(64))]
    pub struct Task {
        data: InlineData,
        data_large_ptr: *mut u8,
        fn_ptr: unsafe fn(*mut Task) -> TaskResult,
    }

    /// Used when the closure is stored inside the Task struct.
    unsafe fn small_shim<F: FnOnce() -> TaskResult + 'static>(ptr: *mut Task) -> TaskResult {
        // We know it's small, so we read from the 'data' field.
        let f = unsafe { std::ptr::read((*ptr).data.as_mut_ptr() as *mut F) };
        f()
    }

    /// Used when the closure is stored in the 4KB Arena.
    unsafe fn large_shim<F: FnOnce() -> TaskResult + 'static>(ptr: *mut Task) -> TaskResult {
        // We know it's large, so we read from the 'data_large_ptr'.
        let f = unsafe { std::ptr::read((*ptr).data_large_ptr as *mut F) };
        f()
    }

    impl Task {
        pub fn new(index: usize, arena: &mut Vec<u8>) -> Self {
            // We want the task to take 64 bytes of memory to not violate cache lines.
            debug_assert!(size_of::<Self>() == 64usize);

            let offset = index * GLOBAL_TASK_MAX_SIZE;
            let large_data_ptr = unsafe { arena.as_mut_ptr().add(offset) };
            Self::new_inner(|| Ok(()), large_data_ptr)
        }

        fn new_inner<F: FnOnce() -> TaskResult + Send + 'static>(
            func: F,
            large_data_ptr: *mut u8,
        ) -> Self {
            let data = unsafe {
                let mut data: InlineData = [0; INLINE_TASK_MAX_SIZE / 16];
                std::ptr::write(data.as_mut_ptr() as *mut F, func);
                data
            };

            Self {
                data,
                fn_ptr: small_shim::<F>,
                data_large_ptr: large_data_ptr,
            }
        }

        pub fn init<F: FnOnce() -> TaskResult + Send + 'static>(&mut self, func: F) {
            if size_of::<F>() <= size_of::<InlineData>() {
                Self::init_small(self, func)
            } else if size_of::<F>() <= size_of::<TaskData>() {
                Self::init_large(self, func)
            } else {
                let boxed = Box::new(func);
                Self::init_small(self, move || {
                    let func = boxed;
                    func()
                });
            }
        }

        fn init_small<F: FnOnce() -> TaskResult + Send + 'static>(&mut self, func: F) {
            unsafe {
                std::ptr::write(self.data.as_mut_ptr() as *mut F, func);
            };
            self.fn_ptr = small_shim::<F>;
        }

        fn init_large<F: FnOnce() -> TaskResult + Send + 'static>(&mut self, func: F) {
            unsafe {
                std::ptr::write(self.data_large_ptr as *mut F, func);
            };
            self.fn_ptr = large_shim::<F>;
        }

        pub fn run(&mut self) -> Result<(), super::task::TaskError> {
            let func = self.fn_ptr;
            unsafe { (func)(self) }
        }
    }
}

/// Use a normal channel instead.
mod normal_channel {
    use crate::device::{
        DeviceId,
        handle::{CallError, channel::task::TaskResult},
    };
    use alloc::boxed::Box;
    use std::sync::mpsc::SyncSender;

    /// Buffer size for the command channel.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// The client-side handle used to enqueue tasks.
    pub struct DeviceClient {
        state: SyncSender<Box<dyn FnOnce() -> TaskResult + Send + 'static>>,
        device_id: DeviceId,
    }

    unsafe impl Send for DeviceClient {}
    unsafe impl Sync for DeviceClient {}

    impl Clone for DeviceClient {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                device_id: self.device_id,
            }
        }
    }

    impl DeviceClient {
        /// Gets the device id associated to the channel.
        pub fn device_id(&self) -> &DeviceId {
            &self.device_id
        }
        /// Creates a new channel and spawns a server thread to process it.
        pub fn new<I: FnOnce() -> TaskResult + Send + 'static>(
            device_id: DeviceId,
            init: I,
        ) -> Self {
            let (sender, recv) = std::sync::mpsc::sync_channel::<
                Box<dyn FnOnce() -> TaskResult + Send + 'static>,
            >(CHANNEL_MAX_TASK);

            std::thread::spawn(move || {
                init().unwrap();
                loop {
                    if let Ok(item) = recv.recv()
                        && let Err(err) = item()
                    {
                        panic!("{err:?}");
                    }
                }
            });

            Self {
                state: sender,
                device_id,
            }
        }

        /// Atomically reserves a slot in the buffer and writes the task.
        pub fn enqueue<F: FnOnce() -> TaskResult + Send + 'static>(
            &self,
            func: F,
        ) -> Result<(), CallError> {
            self.state.send(Box::new(func)).map_err(|_| CallError)
        }

        /// Forces a flush by filling the remaining buffer with no-op tasks.
        pub fn flush(&self) {
            // Nothing to do.
        }
    }
}

/// We implement a custom channel with automatic batching, no locking and no allocation (most of the time).
mod custom_channel {
    use crate::device::{
        DeviceId,
        handle::{
            CallError,
            channel::task::{GLOBAL_TASK_MAX_SIZE, Task, TaskResult},
        },
    };
    use alloc::boxed::Box;
    use core::{
        hint::spin_loop,
        sync::atomic::{AtomicPtr, AtomicU32, Ordering},
    };
    use std::{sync::Arc, vec::Vec};

    /// Buffer size for the command channel.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// The client-side handle used to enqueue tasks.
    pub struct DeviceClient {
        state: Arc<State>,
    }

    unsafe impl Send for DeviceClient {}
    unsafe impl Sync for DeviceClient {}

    impl Clone for DeviceClient {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl DeviceClient {
        /// Gets the device id associated to the channel.
        pub fn device_id(&self) -> &DeviceId {
            &self.state.device_id
        }
        /// Creates a new channel and spawns a server thread to process it.
        pub fn new<I: FnOnce() -> TaskResult + Send + 'static>(
            device_id: DeviceId,
            init: I,
        ) -> Self {
            let mut server = Server::new(device_id);
            let state = server.state.clone();

            std::thread::Builder::new()
                .name(std::format!(
                    "device-{}-{}",
                    device_id.type_id,
                    device_id.index_id
                ))
                .spawn(move || {
                    init().unwrap();
                    server.run();
                })
                .unwrap();

            Self { state }
        }

        /// Atomically reserves a slot in the buffer and writes the task.
        pub fn enqueue<F: FnOnce() -> TaskResult + Send + 'static>(
            &self,
            func: F,
        ) -> Result<(), CallError> {
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

                self.state.success.fetch_add(1, Ordering::SeqCst);
                return Ok(());
            }
        }

        /// Forces a flush by filling the remaining buffer with no-op tasks.
        pub fn flush(&self) {
            let index_start =
                self.state
                    .index
                    .fetch_add(CHANNEL_MAX_TASK as u32, Ordering::Acquire) as usize;

            if index_start >= CHANNEL_MAX_TASK {
                return;
            }

            let ptr = self.state.ptr.load(Ordering::Relaxed);
            for index in index_start..CHANNEL_MAX_TASK {
                unsafe {
                    let task = ptr.add(index).as_mut().unwrap();
                    task.init(move || Ok(()));
                };
            }

            let actual_added = CHANNEL_MAX_TASK - index_start;
            self.state
                .success
                .fetch_add(actual_added as u32, Ordering::SeqCst);
        }
    }

    struct State {
        /// Pointer to the current active client buffer.
        ptr: AtomicPtr<Task>,
        /// Next available index for writing.
        index: AtomicU32,
        /// Number of tasks successfully written and ready for processing.
        success: AtomicU32,
        /// The device id (for debugging purposes).
        device_id: DeviceId,
    }

    /// The server-side runner that processes tasks.
    struct Server {
        state: Arc<State>,
        ptr_client: *mut Task,
        ptr_server: *mut Task,
        num_remaining: usize,
        _drop_guard: Box<dyn FnOnce()>, // Ensures Vecs are cleaned up
    }

    unsafe impl Send for Server {}

    impl Server {
        fn new(device_id: DeviceId) -> Self {
            let mut arena_1_vec =
                Vec::from_iter((0..CHANNEL_MAX_TASK * GLOBAL_TASK_MAX_SIZE).map(|_| 0u8));
            let mut arena_2_vec =
                Vec::from_iter((0..CHANNEL_MAX_TASK * GLOBAL_TASK_MAX_SIZE).map(|_| 0u8));
            let mut data_1_vec = Vec::from_iter(
                (0..CHANNEL_MAX_TASK).map(|index| Task::new(index, &mut arena_1_vec)),
            );
            let mut data_2_vec = Vec::from_iter(
                (0..CHANNEL_MAX_TASK).map(|index| Task::new(index, &mut arena_2_vec)),
            );

            let data_client = data_1_vec.as_mut_ptr();
            let data_server = data_2_vec.as_mut_ptr();

            let state = Arc::new(State {
                ptr: AtomicPtr::new(data_client),
                index: AtomicU32::new(0),
                success: AtomicU32::new(0),
                device_id,
            });

            Self {
                state,
                num_remaining: 0,
                ptr_client: data_client,
                ptr_server: data_server,
                _drop_guard: Box::new(move || {
                    log::error!("Dropping the server");
                    drop(arena_1_vec);
                    drop(arena_2_vec);
                    drop(data_1_vec);
                    drop(data_2_vec);
                }),
            }
        }

        /// Main execution loop for the device thread.
        fn run(&mut self) {
            loop {
                if self.num_remaining != 0 {
                    self.execute_tasks();
                }

                let success_count = self.state.success.load(Ordering::Acquire) as usize;

                if success_count >= CHANNEL_MAX_TASK {
                    self.fetch();
                } else {
                    spin_loop();
                }
            }
        }

        fn execute_tasks(&mut self) {
            for cursor in 0..self.num_remaining {
                let mut task = unsafe { self.ptr_server.add(cursor).read() };

                if task.run().is_err() {
                    panic!("Server doesn't handle task errors.")
                }
            }
            self.num_remaining = 0;
        }

        /// Swaps the client and server pointers, allowing the client to start
        /// filling the next buffer while the server processes the current one.
        fn fetch(&mut self) {
            let remaining = self.state.success.load(Ordering::Acquire);
            core::mem::swap(&mut self.ptr_client, &mut self.ptr_server);

            self.state.ptr.store(self.ptr_client, Ordering::SeqCst);
            self.num_remaining = remaining as usize;

            // Reset indices for the new client buffer
            self.state.success.store(0, Ordering::SeqCst);
            self.state.index.store(0, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::device::handle::channel::custom_channel::CHANNEL_MAX_TASK;

    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    // A mock service to track state changes and initialization
    struct MockService {
        counter: usize,
        id: DeviceId,
    }

    impl DeviceService for MockService {
        fn init(id: DeviceId) -> Self {
            Self { counter: 0, id }
        }
    }

    #[test]
    fn test_basic_execution_and_state_persistence() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 1,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        // Task 1: Increment the counter
        let res = handle
            .submit_blocking(|state| {
                state.counter += 1;
                state.counter
            })
            .unwrap();

        // Task 2: Increment again to ensure it's the same state instance
        let res2 = handle
            .submit_blocking(|state| {
                state.counter += 1;
                state.counter
            })
            .unwrap();

        assert_eq!(res, 1);
        assert_eq!(res2, 2);
    }

    #[test]
    fn test_scoped_tasks_and_lifetimes() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 3,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        let local_val = 42; // This lives on the test stack

        // Test exclusive_scoped
        let result = handle.exclusive_scoped(|| local_val + 8).unwrap();

        assert_eq!(result, 50);

        // Test submit_blocking_scoped
        let result_mut = handle.submit_blocking_scoped(|state| {
            state.counter = local_val;
            state.counter
        });

        assert_eq!(result_mut, 42);
    }

    #[test]
    fn test_buffer_flushing_at_limit() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 4,
        };

        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let completed_count = Arc::new(AtomicUsize::new(0));

        // We fill exactly CHANNEL_MAX_TASK
        // The last task should trigger a buffer swap/fetch.
        for _ in 0..CHANNEL_MAX_TASK {
            let counter = Arc::clone(&completed_count);
            handle.submit(move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }

        // Give the server a moment to process the batch
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(completed_count.load(Ordering::SeqCst), 32);
    }

    #[test]
    fn test_manual_flush_for_partial_buffer() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 5,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let (tx, rx) = oneshot::channel();

        // Send only 1 task (buffer is not full)
        handle.submit(move |_| {
            tx.send(true).unwrap();
        });

        // This would hang forever if flush() didn't fill the buffer with no-ops
        handle.state.client.flush();

        let received = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("Task was not flushed and processed in time");
        assert!(received);
    }

    #[test]
    fn test_closure_captures_are_dropped_after_execution() {
        let device_id = DeviceId {
            type_id: 0,
            index_id: 6,
        };

        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        // This atomic counter will track how many times our "Spy" is dropped.
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy(Arc<AtomicUsize>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy(Arc::clone(&drop_count));

        // We capture `spy` in the closure.
        // 1. It is moved into the Task buffer (or a Box if too large).
        // 2. The runner thread's shim uses ptr::read to move it into a local variable.
        // 3. The closure finishes, the local variable goes out of scope, and drop() is called.
        handle
            .submit_blocking(move |_state| {
                // Accessing spy here to ensure it's captured.
                let _ = &spy;
            })
            .expect("Task execution failed");

        // At this point, the blocking call has returned.
        // Because the shim moved the closure and let it go out of scope,
        // the drop count should be exactly 1.
        assert_eq!(
            drop_count.load(Ordering::SeqCst),
            1,
            "Capture was not dropped after execution"
        );
    }
}
