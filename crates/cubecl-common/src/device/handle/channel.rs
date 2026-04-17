use crate::{
    device::{
        DeviceId, DeviceService, ServerUtilitiesHandle,
        handle::{CallError, DeviceHandleSpec, ServiceCreationError, channel::task::TaskResult},
    },
    stream_id::StreamId,
};
use hashbrown::HashMap;
use std::{
    any::{Any, TypeId},
    boxed::Box,
    cell::RefCell,
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
};

use custom_channel::DeviceClient;

// For debugging and benchmarking.
//
// use normal_channel::DeviceClient;

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
    // fn(S) makes this Send+Sync regardless of S, since the handle
    // never actually holds an S — it only sends closures to the runner thread.
    _phantom: PhantomData<fn(S)>,
}

impl<S: DeviceService + 'static> DeviceHandleSpec<S> for ChannelDeviceHandle<S> {
    const BLOCKING: bool = false;

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

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.state.utilities()
    }

    /// Submits a task to the device thread and blocks the current thread until
    /// the task returns a result.
    fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        self.submit_inner::<_, SEND_FLUSH>(move |state: &mut S| {
            let returned = task(state);
            sender.send(returned).unwrap();
        })?;

        recv.recv().map_err(|_| CallError)
    }

    /// Submits a task with a non-static lifetime.
    ///
    /// # Safety
    ///
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

        self.submit_inner::<_, SEND_FLUSH>(static_task)
            .expect("Can't have an error when submitting a scoped task");

        recv.recv()
            .expect("Scoped task failed: Runner disconnected")
    }

    /// Asynchronously dispatches a task to the device thread.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.submit_inner::<_, SEND_NO_FLUSH>(task)
            .expect("Can't have an error when submitting a task");
    }

    /// Executes a task on the device thread that does not require direct
    /// access to the `DeviceService` state, blocking until completion.
    fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();
        let current = StreamId::current();

        self.send::<_, SEND_FLUSH>(move || {
            let returned = current.executes(task);
            let _ = sender.send(returned);
        })?;

        recv.recv().map_err(|_| CallError)
    }

    fn flush_queue(&self) {
        if !is_device_runner_thread(self.state.client.device_id()) {
            self.state.client.flush();
        }
    }

    /// Executes a closure with a captured scope on the device thread.
    ///
    /// Blocks until the task is complete. Useful for operations that need to
    /// run on the dedicated thread but reference data on the caller's stack.
    fn exclusive_scoped<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        let (sender, recv) = oneshot::channel();

        let current = StreamId::current();

        let wrapper = move || {
            let returned = current.executes(task);
            let _ = sender.send(returned);
        };

        let boxed: Box<dyn FnOnce() -> TaskResult + Send> = Box::new(wrapper);
        // SAFETY: Blocking on `recv` guarantees the closure finishes before the scope ends.
        let static_task: Box<dyn FnOnce() -> TaskResult + Send + 'static> =
            unsafe { std::mem::transmute(boxed) };

        self.send::<_, SEND_FLUSH>(static_task)?;

        recv.recv().map_err(|_| CallError)
    }
}

const SEND_FLUSH: bool = true;
const SEND_NO_FLUSH: bool = false;

impl<S: DeviceService + 'static> ChannelDeviceHandle<S> {
    /// Asynchronously dispatches a task to the device thread.
    fn submit_inner<T: FnOnce(&mut S) + Send + 'static, const FLUSH: bool>(
        &self,
        task: T,
    ) -> Result<(), CallError> {
        let state = self.state.service.clone();

        let current = StreamId::current();

        let func_init = move || {
            state.act_on(|state| {
                let state = state
                    .downcast_mut::<S>()
                    .expect("State type mismatch in Thread Local Storage");

                current.executes(|| task(state));
            });
        };

        self.send::<_, FLUSH>(func_init)
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
            if let Err(err) = catch_unwind(AssertUnwindSafe(task)) {
                log::warn!("Task failed: {err:?}");
                return Err(CallError);
            }
        } else {
            self.state.client.enqueue(task)?;

            // We use const boolean to avoid branching in a hot loop.
            if FLUSH {
                self.state.client.flush();
            }
        };

        Ok(())
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
    static STATES: RefCell<HashMap<TypeId, RefCell<Box<dyn Any + 'static>>>> = RefCell::new(HashMap::new());
}

/// Internal runner logic to manage background thread spawning.
struct DeviceRunner {}

/// A simple wrapper over a client and a service that is cached with [`CHANNELS`].
#[derive(Clone)]
struct ChannelDeviceState {
    client: DeviceClient,
    service: ChannelService,
}

/// Cached reference to a device service's state.
///
/// Holds only the `TypeId` for looking up the state in thread-local STATES.
/// This avoids sharing the state across threads.
#[derive(Clone)]
struct ChannelService {
    type_id: TypeId,
    utilities: ServerUtilitiesHandle,
}

static RUNNERS: spin::Mutex<Option<HashMap<DeviceId, DeviceClient>>> = spin::Mutex::new(None);
/// Device/service map. The lock is held across the entire `init` sequence so `S::init` runs
/// once per `(DeviceId, TypeId)` pair. This serializes channel creation across all
/// backends.
static CHANNELS: spin::Mutex<Option<HashMap<(DeviceId, TypeId), ChannelDeviceState>>> =
    spin::Mutex::new(None);

impl ChannelDeviceState {
    pub fn init<S: DeviceService>(
        device_id: DeviceId,
        service: Option<S>,
    ) -> Result<Self, ServiceCreationError> {
        let type_id = TypeId::of::<S>();
        let key = (device_id, type_id);

        // Hold the `CHANNELS` lock across the entire init sequence so that the
        // "check missing, insert new" transition is atomic. Without this, two
        // concurrent callers for the same key would both observe a missing entry,
        // both run `S::init`, and race to insert.
        let mut guard_channel = CHANNELS.lock();
        let channels = guard_channel.get_or_insert_with(HashMap::new);

        if let Some(existing) = channels.get(&key) {
            if service.is_some() {
                // `insert(device, service)` cannot replace an existing state.
                return Err(ServiceCreationError::new(
                    "Service already initialized.".into(),
                ));
            }
            return Ok(existing.clone());
        }

        // A single device runner can serve multiple [`DeviceService`].
        let device_client = {
            let mut guard = RUNNERS.lock();
            let runners = guard.get_or_insert_with(HashMap::new);
            runners
                .entry(device_id)
                .or_insert_with(|| DeviceRunner::start(device_id))
                .clone()
        };

        let (callback, recv) = oneshot::channel();

        // The service initialization function.
        let initialize_service = move || {
            STATES.with_borrow_mut(|map| {
                // If a service state is passed as parameter, we enforce it being used, by
                // returning an error to the callback.
                if service.is_some() && map.contains_key(&type_id) {
                    callback.send(Err(())).unwrap();
                } else {
                    let service = service.unwrap_or_else(|| S::init(device_id));
                    let utilities = service.utilities();

                    map.entry(type_id)
                        .or_insert_with(|| RefCell::new(Box::new(service)));
                    callback
                        .send(Ok(ChannelService { type_id, utilities }))
                        .unwrap();
                }
            });
        };

        // Same reason in [`send]` we need to call the function directly if we are on the runner
        // thread.
        if is_device_runner_thread(&device_id) {
            if let Err(err) = catch_unwind(AssertUnwindSafe(initialize_service)) {
                return Err(ServiceCreationError::new(std::format!(
                    "Service initialization failed: {err:?}"
                )));
            };
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

        channels.insert(key, channel.clone());

        Ok(channel)
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.service.utilities.clone()
    }
}

impl ChannelService {
    /// Borrows the service state from thread-local storage and passes it to `f`.
    /// Panics if the state is already borrowed (re-entrant access).
    fn act_on<R>(&self, f: impl FnOnce(&mut Box<dyn Any + 'static>) -> R) -> R {
        STATES.with_borrow(|map| {
            let cell = map.get(&self.type_id).expect("Service state not found");
            let mut guard = cell
                .try_borrow_mut()
                .expect("Service state is already borrowed");
            f(&mut guard)
        })
    }
}

impl DeviceRunner {
    /// Spawns a new thread, marks it with the `device_id`, and returns a `DeviceClient`.
    pub fn start(device_id: DeviceId) -> DeviceClient {
        let (sender_init, recv_init) = oneshot::channel();
        let channel = DeviceClient::new(device_id, move || {
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
            state: self.state.clone(),
            _phantom: self._phantom,
        }
    }
}

mod task {
    use super::*;
    use core::sync::atomic::{AtomicPtr, Ordering};
    use std::{
        mem::{align_of, size_of},
        panic::{AssertUnwindSafe, catch_unwind},
    };

    /// The maximum size of a closure that can be stored without heap allocation.
    pub const GLOBAL_TASK_MAX_SIZE: usize = 4096;

    /// The maximum size of a closure that can be stored using inlined memory.
    const INLINE_TASK_MAX_SIZE: usize = 48;

    /// One arena slot. `#[repr(C, align(64))]` makes every slot 64-byte
    /// aligned on its own, so the slot alignment does not depend on the layout of any
    /// enclosing type. `GLOBAL_TASK_MAX_SIZE` is a multiple of 64, so there is no
    /// per-slot padding.
    #[repr(C, align(64))]
    pub struct ArenaSlot {
        pub data: [u8; GLOBAL_TASK_MAX_SIZE],
    }

    /// The return type of wrapped closures.
    pub type TaskResult = ();

    #[repr(C, align(64))]
    /// A task is how we represent closures in memory without extra allocations.
    ///
    /// It fits in 64 bytes, ensuring multiple threads can initialize tasks at the same time
    /// without causing false sharing.
    pub struct Task {
        // 48 bytes; 64-aligned because it is the first field of a 64-aligned struct.
        data: [u8; INLINE_TASK_MAX_SIZE],
        // 8 bytes (usize/u64 ptr)
        data_large_ptr: AtomicPtr<u8>,
        // 8 bytes (usize/u64 ptr)
        fn_ptr: fn(&mut Task) -> TaskResult,
    }

    const _: () = {
        // ArenaSlot is 4096 bytes and 64-aligned on its own.
        assert!(core::mem::size_of::<ArenaSlot>() == GLOBAL_TASK_MAX_SIZE);
        // `Task::data` lives at offset 0 of a 64-aligned 64-byte struct, which is
        // what lets the router assume the inline slot has `SLOT_ALIGN`-byte alignment.
        assert!(core::mem::size_of::<Task>() == 64);
        assert!(core::mem::align_of::<Task>() == core::mem::align_of::<ArenaSlot>());
        assert!(core::mem::offset_of!(Task, data) == 0);
    };

    impl Task {
        pub fn new(large_data_ptr: *mut u8) -> Self {
            Self {
                data: [0u8; INLINE_TASK_MAX_SIZE],
                data_large_ptr: AtomicPtr::new(large_data_ptr),
                fn_ptr: |_| {},
            }
        }

        /// Store `func` in the inline slot, the arena slot, or on the heap depending on
        /// its size and alignment. Both checks are required: writing into a slot whose
        /// alignment is smaller than `align_of::<F>()` would produce a misaligned
        /// `ptr::write` (UB). The boxed fallback uses `Box::new`, whose allocation
        /// satisfies any alignment.
        pub fn init<F: FnOnce() -> TaskResult + Send + 'static>(&mut self, func: F) {
            let fits_inline = size_of::<F>() <= INLINE_TASK_MAX_SIZE
                && align_of::<F>() <= align_of::<ArenaSlot>();
            let fits_arena = size_of::<F>() <= GLOBAL_TASK_MAX_SIZE
                && align_of::<F>() <= align_of::<ArenaSlot>();

            if fits_inline {
                // SAFETY: size + align checked above, read back exactly once by fn_ptr.
                unsafe { std::ptr::write(self.data.as_mut_ptr() as *mut F, func) };
                self.fn_ptr = |task| {
                    // SAFETY: Paired with the ptr::write to data above.
                    let f = unsafe { std::ptr::read(task.data.as_mut_ptr() as *mut F) };
                    if let Err(err) = catch_unwind(AssertUnwindSafe(f)) {
                        log::warn!("Task failed: {err:?}");
                    }
                };
            } else if fits_arena {
                // SAFETY: size + align checked above, read back exactly once by fn_ptr.
                unsafe {
                    std::ptr::write(self.data_large_ptr.load(Ordering::Relaxed) as *mut F, func)
                };
                self.fn_ptr = |task| {
                    // SAFETY: Paired with the ptr::write to data_large_ptr above.
                    let f = unsafe {
                        std::ptr::read(task.data_large_ptr.load(Ordering::Relaxed) as *mut F)
                    };
                    if let Err(err) = catch_unwind(AssertUnwindSafe(f)) {
                        log::warn!("Task failed: {err:?}");
                    }
                };
            } else {
                // Size or alignment exceeds both slots. Heap-allocate to get a
                // properly-aligned, pointer-sized handle, then recurse as an inline
                // task (the Box is a pointer so it trivially fits inline).
                let boxed: Box<dyn FnOnce() -> TaskResult + Send> = Box::new(func);
                self.init(boxed);
            }
        }

        /// Runs the task.
        ///
        /// The task must be initialized and run only once per initialization.
        /// Tasks must run, otherwise we will create memory leaks since we don't
        /// drop tasks that aren't executed.
        pub fn run(&mut self) {
            (self.fn_ptr)(self)
        }
    }
}

/// A normal channel implementation, use for debugging.
#[allow(dead_code)]
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
                init();
                loop {
                    if let Ok(item) = recv.recv() {
                        item()
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

/// We implement a custom channel with automatic batching, no locking and
/// no allocation (most of the time, see [`task`] for more details.
mod custom_channel {
    use crate::device::{
        DeviceId,
        handle::{
            CallError,
            channel::task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task, TaskResult},
        },
    };
    use core::{
        hint::spin_loop,
        sync::atomic::{AtomicPtr, AtomicU32, Ordering},
    };
    use std::{sync::Arc, vec::Vec};

    /// Maximum number of [`Task`] that can be queued.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// The client-side handle used to enqueue tasks.
    pub struct DeviceClient {
        state: Arc<State>,
    }

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
                    "Device-{}-{}",
                    device_id.type_id,
                    device_id.index_id
                ))
                .spawn(move || {
                    init();
                    server.start();
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
                let index = self.state.available_index.fetch_add(1, Ordering::Acquire) as usize;
                if index >= CHANNEL_MAX_TASK {
                    // The queue is full; spin until the server flushes/swaps buffers.
                    spin_loop();
                    continue;
                }

                self.state.init_task_at(index, func);
                self.state.enqueued_count.fetch_add(1, Ordering::SeqCst);
                return Ok(());
            }
        }

        /// Forces a flush by filling the remaining buffer with no-op tasks.
        pub fn flush(&self) {
            let index_start =
                self.state
                    .available_index
                    .fetch_add(CHANNEL_MAX_TASK as u32, Ordering::Acquire) as usize;

            // The queue is already flushed.
            if index_start >= CHANNEL_MAX_TASK {
                return;
            }

            // We clamp the number of no-op to the required amount.
            //
            // # Notes
            //
            // index_end != index_start + CHANNEL_MAX_TASK;
            let index_end = CHANNEL_MAX_TASK;

            for index in index_start..index_end {
                self.state.init_task_at(index, || ());
            }

            let actual_added = index_end - index_start;
            self.state
                .enqueued_count
                .fetch_add(actual_added as u32, Ordering::SeqCst);
        }
    }

    struct State {
        /// Pointer to the current active queue buffer.
        ///
        /// Written by the server thread (Release) after swapping buffers,
        /// read by client threads (Acquire) before writing tasks.
        queue_ptr: AtomicPtr<Task>,
        /// Next available index for writing.
        available_index: AtomicU32,
        /// Number of tasks successfully written and ready for processing.
        enqueued_count: AtomicU32,
        /// The device id (for debugging purposes).
        device_id: DeviceId,
    }

    impl State {
        /// Initializes the task at `index` in the current queue with `func`.
        /// Exclusive access per slot is guaranteed by `available_index.fetch_add`.
        fn init_task_at<F: FnOnce() -> TaskResult + Send + 'static>(&self, index: usize, func: F) {
            assert!(index < CHANNEL_MAX_TASK, "task index {index} out of bounds");
            // SAFETY: queue_ptr points to a valid buffer of CHANNEL_MAX_TASK tasks,
            // bounds checked above, and the &mut doesn't escape.
            unsafe { &mut *self.queue_ptr.load(Ordering::Acquire).add(index) }.init(func);
        }
    }

    /// Owns a task buffer and its associated large-closure arena.
    struct TaskBuffer {
        tasks: Vec<Task>,
        _arena: Vec<ArenaSlot>,
    }

    impl TaskBuffer {
        fn new() -> Self {
            let mut arena: Vec<ArenaSlot> =
                Vec::from_iter((0..CHANNEL_MAX_TASK).map(|_| ArenaSlot {
                    data: [0u8; GLOBAL_TASK_MAX_SIZE],
                }));

            let arena_ptr = arena.as_mut_ptr() as *mut u8;
            let tasks = Vec::from_iter((0..CHANNEL_MAX_TASK).map(|index| {
                // SAFETY: Each task owns a non-overlapping `ArenaSlot` region.
                Task::new(unsafe { arena_ptr.add(index * GLOBAL_TASK_MAX_SIZE) })
            }));
            Self {
                tasks,
                _arena: arena,
            }
        }
    }

    /// The server-side runner that processes tasks.
    struct Server {
        state: Arc<State>,
        /// Index into `buffers`: which buffer clients are currently writing to.
        client_buf: usize,
        buffers: [TaskBuffer; 2],
        ready_to_execute: bool,
    }

    impl Server {
        fn new(device_id: DeviceId) -> Self {
            let mut buffers = [TaskBuffer::new(), TaskBuffer::new()];

            let state = Arc::new(State {
                queue_ptr: AtomicPtr::new(buffers[0].tasks.as_mut_ptr()),
                available_index: AtomicU32::new(0),
                enqueued_count: AtomicU32::new(0),
                device_id,
            });

            Self {
                state,
                client_buf: 0,
                buffers,
                ready_to_execute: false,
            }
        }

        /// Main execution loop for the device thread.
        fn start(&mut self) {
            loop {
                if self.ready_to_execute {
                    self.execute_tasks();
                }

                let queue_size = self.state.enqueued_count.load(Ordering::Acquire) as usize;

                if queue_size >= CHANNEL_MAX_TASK {
                    self.fetch();
                } else {
                    spin_loop();
                }
            }
        }

        fn execute_tasks(&mut self) {
            let server_buf = 1 - self.client_buf;
            for task in &mut self.buffers[server_buf].tasks {
                task.run();
            }
            self.ready_to_execute = false;
        }

        /// Swaps the client and server buffers, allowing the client to start
        /// filling the next buffer while the server processes the current one.
        fn fetch(&mut self) {
            self.client_buf = 1 - self.client_buf;

            self.state.queue_ptr.store(
                self.buffers[self.client_buf].tasks.as_mut_ptr(),
                Ordering::Release,
            );

            self.ready_to_execute = true;

            // Reset indices for the new client buffer
            self.state.enqueued_count.store(0, Ordering::SeqCst);

            // This is what is used for the spin loop on the client size.
            //
            // It is very important to be the last thing to reset.
            self.state.available_index.store(0, Ordering::SeqCst);
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

        fn utilities(&self) -> ServerUtilitiesHandle {
            Arc::new(())
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
    #[cfg(not(miri))]
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

        // Wait for tasks to complete. Miri is very slow with this test, so sleeping fails here.
        let _ = handle.submit_blocking(|_| {});

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

    #[test]
    fn test_large_closure_uses_arena() {
        // Closure captures > 48 bytes (InlineSlot), forcing the arena path.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 7,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        let big_data = [42u8; 128]; // 128 bytes > 48 byte inline limit
        let result = handle
            .submit_blocking(move |_state| {
                // Use big_data to prevent it from being optimized away.
                big_data[0] + big_data[127]
            })
            .unwrap();

        assert_eq!(result, 84);
    }

    #[test]
    fn test_extra_large_closure_uses_box() {
        // Closure captures > 4096 bytes (GLOBAL_TASK_MAX_SIZE), forcing the Box fallback.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 8,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);

        let huge_data = [7u8; 8192]; // 8KB > 4096 byte arena limit
        let result = handle
            .submit_blocking(move |_state| huge_data[0] + huge_data[8191])
            .unwrap();

        assert_eq!(result, 14);
    }

    #[test]
    fn test_large_closure_drop_is_called() {
        // Verify that Drop runs correctly for closures stored in the arena.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 9,
        };
        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy {
            counter: Arc<AtomicUsize>,
            _padding: [u8; 128], // Force arena path (> 48 bytes)
        }
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy {
            counter: Arc::clone(&drop_count),
            _padding: [0; 128],
        };

        handle
            .submit_blocking(move |_state| {
                let _ = &spy;
            })
            .unwrap();

        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    /// Concurrent callers racing on the same `(DeviceId, TypeId)` must share a single
    /// `S::init` invocation.
    #[test]
    fn test_init_runs_exactly_once_under_contention() {
        use alloc::vec::Vec;
        use std::sync::Barrier;
        use std::sync::atomic::AtomicUsize;
        use std::thread;

        static INIT_CALLS: AtomicUsize = AtomicUsize::new(0);

        struct CountingService;
        impl DeviceService for CountingService {
            fn init(_: DeviceId) -> Self {
                INIT_CALLS.fetch_add(1, Ordering::SeqCst);
                CountingService
            }
            fn utilities(&self) -> ServerUtilitiesHandle {
                Arc::new(())
            }
        }

        INIT_CALLS.store(0, Ordering::SeqCst);

        const THREADS: usize = 4;
        // Unique device_id so the global `CHANNELS` entry is independent of other tests.
        let device_id = DeviceId {
            type_id: 0,
            index_id: 77,
        };

        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                ChannelDeviceHandle::<CountingService>::new(device_id)
            }));
        }
        for h in handles {
            let _ = h.join().unwrap();
        }

        assert_eq!(
            INIT_CALLS.load(Ordering::SeqCst),
            1,
            "CountingService::init must run exactly once across {THREADS} racing callers"
        );
    }

    /// A closure that spills to the arena (size > 48) and carries the maximum arena
    /// alignment (64) must be stored and executed soundly.
    #[test]
    fn test_task_init_arena_aligned_closure() {
        use super::task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task};

        #[repr(align(64))]
        #[derive(Clone, Copy)]
        struct A64 {
            data: [u8; 128],
        }

        // Mirror `TaskBuffer::new`: a 64-aligned 4KB region per slot.
        let mut arena = alloc::boxed::Box::new(ArenaSlot {
            data: [0u8; GLOBAL_TASK_MAX_SIZE],
        });
        let arena_ptr = arena.data.as_mut_ptr();
        let mut task = Task::new(arena_ptr);

        let data = A64 { data: [0xCD; 128] };
        task.init(move || {
            let d = core::hint::black_box(data);
            let _: usize = d.data.iter().map(|&b| b as usize).sum();
        });
        task.run();
    }

    /// A closure whose alignment exceeds the arena slot alignment must take the
    /// boxed fallback.
    #[test]
    fn test_task_init_extremely_over_aligned_closure_uses_box() {
        use super::task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task};

        #[repr(align(256))]
        #[derive(Clone, Copy)]
        struct A256 {
            data: [u8; 256],
        }

        let mut arena = alloc::boxed::Box::new(ArenaSlot {
            data: [0u8; GLOBAL_TASK_MAX_SIZE],
        });
        let arena_ptr = arena.data.as_mut_ptr();
        let mut task = Task::new(arena_ptr);

        let data = A256 { data: [0xAA; 256] };
        task.init(move || {
            let d = core::hint::black_box(data);
            let _: usize = d.data.iter().map(|&b| b as usize).sum();
        });
        task.run();
    }
}
