use crate::device::{
    DeviceId, DeviceService, DeviceServiceStage, ServerUtilitiesHandle,
    handle::{CallError, DeviceHandleSpec, ServiceCreationError},
};
use core::time::Duration;
use cubecl_environment::future::channel::oneshot;
use cubecl_environment::stream::StreamId;
use hashbrown::{HashMap, HashSet};
use std::{
    any::{Any, TypeId},
    boxed::Box,
    cell::RefCell,
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
    vec::Vec,
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

    fn device_id(&self) -> DeviceId {
        self.state.client.runner_id().device
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.state.utilities()
    }

    /// Runs `task` on the device thread, blocking until it returns.
    fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        let state = self.state.service.clone();
        let current = StreamId::current();
        self.run_scoped(move || {
            state.act_on(|s| {
                let s = s
                    .downcast_mut::<S>()
                    .expect("State type mismatch in Thread Local Storage");
                current.executes(|| task(s))
            })
        })
    }

    /// Asynchronously dispatches a task to the device thread.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.submit_inner::<_, SEND_NO_FLUSH>(task)
            .expect("Can't have an error when submitting a task");
    }

    fn flush_queue(&self) {
        if !is_device_runner_thread(self.state.client.runner_id()) {
            self.state.client.flush();
        }
    }

    /// Runs `task` on the device thread while propagating the caller's
    /// `StreamId`, blocking until it returns.
    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        let current = StreamId::current();
        self.run_scoped(move || current.executes(task))
    }

    fn shutdown(device_id: DeviceId) {
        shutdown_device(device_id);
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

    /// Dispatches a `FnOnce() -> R` on the device thread and blocks on
    /// its result. `task` may borrow from the caller's stack.
    fn run_scoped<'a, R: Send, T: FnOnce() -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        /// What the runner sends back: the task's value, or the payload of the
        /// panic it died on.
        type Outcome<R> = Result<R, Box<dyn Any + Send>>;

        /// Builds a `'static` shim that consumes `*slot` on the device
        /// thread. The caller has to keep `*slot` alive until the shim has run,
        /// `run_scoped` does this by blocking on `recv.recv()`.
        ///
        /// The shim runs the task and sends the outcome in two separate steps,
        /// which is what keeps the second half of that contract honest. `task`
        /// holds references into the caller's frame, and a reference passed by
        /// value into a call is protected for as long as that call is running,
        /// so the frame it points into may not be freed before the call
        /// returns. Sending from inside the call would hand that frame back to
        /// the caller to pop while the protector is still live. Running the
        /// task under its own `catch_unwind` frame releases the protectors
        /// first, leaving nothing caller-derived alive when the send wakes it.
        fn create_shim<R: Send, F: FnOnce() -> R + Send>(
            slot: &mut Option<(F, oneshot::Sender<Outcome<R>>)>,
        ) -> impl FnOnce() + Send + 'static {
            // `*mut ()` so the shim is `'static`.
            struct Ptr(*mut ());
            // SAFETY: pointee is `Send` by the bounds on `F` and `R`;
            // uniqueness of access is upheld by the deref below.
            unsafe impl Send for Ptr {}

            let ptr = Ptr(slot as *mut _ as *mut ());
            move || {
                let _ = &ptr; // capture whole ptr so the closure is Send.
                // SAFETY:
                // - Caller keeps `*slot` alive through the shim's run.
                // - `unwrap_unchecked`: the shim is `FnOnce` run at most
                //   once, so `*slot` is always `Some` on entry.
                // `Option::take` flips `*slot` to `None`, keeping drop
                // correct if the shim ran, panicked, or was never enqueued.
                let (task, sender) = unsafe {
                    (*(ptr.0 as *mut Option<(F, oneshot::Sender<Outcome<R>>)>))
                        .take()
                        .unwrap_unchecked()
                };
                // Both moved out of the caller's slot and owned by this frame,
                // so the caller's stack is only reachable through `task`, and
                // only until `catch_unwind` returns.
                let outcome = catch_unwind(AssertUnwindSafe(task));
                let _ = sender.send(outcome);
            }
        }

        let (sender, recv) = oneshot::channel();
        let mut slot = Some((task, sender));
        // Send the erased shim to the device thread.
        self.send::<_, SEND_FLUSH>(create_shim(&mut slot))?;
        match recv.recv() {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(payload)) => Err(CallError::from_panic(payload)),
            // The sender was dropped without sending (e.g. the runner thread died)
            // so no panic payload is available.
            Err(_) => Err(CallError::disconnected()),
        }
    }

    /// Dispatches a task to the runner.
    ///
    /// If the current thread is already the runner for this device, it executes
    /// immediately to prevent deadlocks and allow for recursive calls.
    fn send<T: FnOnce() + Send + 'static, const FLUSH: bool>(
        &self,
        task: T,
    ) -> Result<(), CallError> {
        if is_device_runner_thread(self.state.client.runner_id()) {
            // Capture the panic payload so it surfaces in the returned `CallError`.
            if let Err(payload) = catch_unwind(AssertUnwindSafe(task)) {
                let err = CallError::from_panic(payload);
                log::warn!("Task failed: {err:?}");
                return Err(err);
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
fn is_device_runner_thread(runner_key: &RunnerId) -> bool {
    SERVER_THREAD.with_borrow(|state| state.as_ref() == Some(runner_key))
}

/// Whether the current thread is a runner for `device_id`, on any stage.
fn is_device_thread(device_id: DeviceId) -> bool {
    SERVER_THREAD.with_borrow(|state| state.as_ref().is_some_and(|id| id.device == device_id))
}

std::thread_local! {
    /// The ID of the device this thread is responsible for.
    static SERVER_THREAD: RefCell<Option<RunnerId>> = const { RefCell::new(None) };

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

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct RunnerId {
    device: DeviceId,
    stage: DeviceServiceStage,
}

/// A registered runner: the client used to reach it plus the join handle of
/// its server thread, kept so [`shutdown_device`] can wait for the exit.
struct RunnerEntry {
    client: DeviceClient,
    thread: std::thread::JoinHandle<()>,
}

static RUNNERS: spin::Mutex<Option<HashMap<RunnerId, RunnerEntry>>> = spin::Mutex::new(None);
/// Device/service map. The lock is held across the entire `init` sequence so `S::init` runs
/// once per `(DeviceId, TypeId)` pair. This serializes channel creation across all
/// backends.
///
/// Lock order is `CHANNELS` then `RUNNERS`, never the reverse.
static CHANNELS: spin::Mutex<Option<Registry>> = spin::Mutex::new(None);

/// The cached device/service states, plus the runners currently being wound down.
///
/// One lock covers both because [`ChannelDeviceState::init`] and [`shutdown_device`] have
/// to agree on a single ordering: a shutdown that swept the map has to be visible to
/// every `init` that has not yet inserted into it. Sharing the lock `init` already takes
/// also means the check costs it a set lookup rather than another acquisition, and
/// handle creation is warm enough to care.
#[derive(Default)]
struct Registry {
    /// Cached state per `(RunnerId, TypeId)`.
    channels: HashMap<(RunnerId, TypeId), ChannelDeviceState>,
    /// Runners [`shutdown_device`] has swept out of both registries but not yet joined.
    /// An `init` that lands on one waits for the join instead of caching a client of a
    /// runner that is about to go away, which would park it forever, or spawning a
    /// second runner with the same [`RunnerId`] while the first is still draining.
    shutting_down: HashSet<RunnerId>,
}

/// How long [`shutdown_device`] waits for a runner thread before giving up on it.
///
/// A runner only exits once every client is gone, so a task parked in another device's
/// queue, or a cycle between two runners shutting each other down, can hold it forever.
/// Leaking the thread with a warning beats hanging the process with no diagnostic.
const SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Number of `yield_now` calls before the join wait drops to sleeping, so a normal
/// shutdown still returns promptly.
const SHUTDOWN_JOIN_YIELD_BUDGET: u32 = 1024;
/// Poll interval once the yield budget is exhausted.
const SHUTDOWN_JOIN_POLL: Duration = Duration::from_micros(200);

impl ChannelDeviceState {
    pub fn init<S: DeviceService>(
        device_id: DeviceId,
        service: Option<S>,
    ) -> Result<Self, ServiceCreationError> {
        let type_id = TypeId::of::<S>();
        let runner_id = RunnerId {
            device: device_id,
            stage: S::stage(),
        };
        let key = (runner_id, type_id);

        // Hold the `CHANNELS` lock across the entire init sequence so that the
        // "check missing, insert new" transition is atomic. Without this, two
        // concurrent callers for the same key would both observe a missing entry,
        // both run `S::init`, and race to insert.
        //
        // Taking the lock also means waiting out any in-flight `shutdown_device` for
        // this runner. The check belongs under the lock: `shutdown_device` marks the
        // runner while holding it, so testing it here is what closes the window where
        // we would otherwise cache a client of a runner that is already being joined,
        // parking that runner forever.
        let mut guard_channel = loop {
            let mut guard = CHANNELS.lock();
            let shutting_down = guard
                .get_or_insert_with(Registry::default)
                .shutting_down
                .contains(&runner_id);

            if !shutting_down {
                break guard;
            }

            if is_device_thread(runner_id.device) {
                // Waiting is not an option: the shutdown covers every runner of this
                // device, including the one running this very code, and it cannot finish
                // until we return. Device-scoped for the same reason the self-join assert
                // in `shutdown_device` is: the runners of a device are joined in an
                // unspecified order, so waiting on a sibling stage can deadlock too.
                return Err(ServiceCreationError::new(
                    "Cannot create a device handle from a runner thread of a device that \
                     is shutting down."
                        .into(),
                ));
            }

            drop(guard);
            std::thread::yield_now();
        };
        let channels = &mut guard_channel.get_or_insert_with(Registry::default).channels;

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
                .entry(runner_id)
                .or_insert_with(|| DeviceRunner::start(runner_id))
                .client
                .clone()
        };

        let (callback, recv) = oneshot::channel();

        // The service initialization function.
        let initialize_service = move || {
            STATES.with(|state| {
                let mut map = match state.try_borrow_mut() {
                    Ok(map) => map,
                    Err(err) => panic!(
                        "The device service {:?} is already borrowed: {err}",
                        core::any::type_name::<S>()
                    ),
                };

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
        if is_device_runner_thread(&runner_id) {
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
    /// Spawns a new thread, marks it with the `device_id`, and returns the
    /// client together with the thread's join handle.
    pub fn start(runner_id: RunnerId) -> RunnerEntry {
        let (sender_init, recv_init) = oneshot::channel();
        let (client, thread) = DeviceClient::new(runner_id, move || {
            SERVER_THREAD.with_borrow_mut(|cell| *cell = Some(runner_id));
            sender_init.send(()).unwrap();
        });

        if recv_init.recv().is_err() {
            panic!("Failed to synchronize device runner thread initialization");
        }

        RunnerEntry { client, thread }
    }
}

/// Stops every runner thread of `device_id`, blocking until they exit.
///
/// Queued tasks run before the threads stop. Live handles keep their runner alive, so
/// this blocks until the last handle for the device is dropped. New handles created for
/// the device afterwards spawn fresh runner threads.
///
/// # Scope
///
/// This is device-wide, not service-wide: it tears down *every* [`DeviceService`] on
/// `device_id` and both [`DeviceServiceStage`]s, not just the service the caller went
/// through. [`DeviceId`] is also not unique across runtimes, `type_id` is assigned per
/// runtime, so two backends can name the same id. Shutting down "device 0" therefore
/// reaches every service any runtime registered under that id.
///
/// # Notes
///
/// If a runner does not stop within [`SHUTDOWN_JOIN_TIMEOUT`] the thread is leaked with a
/// warning rather than blocking forever. That bounds the failure modes the ownership
/// rules cannot rule out: a task holding this device's handle parked in *another*
/// device's unflushed queue, or two runners shutting each other down.
pub(crate) fn shutdown_device(device_id: DeviceId) {
    // A runner joining itself would deadlock. Cycles through another device cannot be
    // caught here, the join timeout is what bounds those.
    SERVER_THREAD.with_borrow(|current| {
        if let Some(runner) = current {
            assert_ne!(
                runner.device, device_id,
                "cannot shut down a device from its own runner thread"
            );
        }
    });

    // Both registries are swept under both locks, and the runners are marked as shutting
    // down before either lock is released. A gap between the sweeps would let a
    // concurrent `init` observe a half-swept state: clone a client from a `RunnerEntry`
    // that is about to be joined and cache it in the registry, which parks the runner
    // forever since nothing but this function removes that entry.
    let (channels, runners) = {
        let mut guard_channel = CHANNELS.lock();
        let registry = guard_channel.get_or_insert_with(Registry::default);

        let runners: Vec<(RunnerId, RunnerEntry)> = match RUNNERS.lock().as_mut() {
            Some(map) => map
                .extract_if(|runner_id, _| runner_id.device == device_id)
                .collect(),
            None => Vec::new(),
        };

        let channels: Vec<ChannelDeviceState> = registry
            .channels
            .extract_if(|(runner_id, _), _| runner_id.device == device_id)
            .map(|(_, state)| state)
            .collect();

        // Marked before the lock is released, so an `init` racing with this sweep either
        // got in before it, and its entry is in `channels` above, or waits for the join.
        for (runner_id, _) in runners.iter() {
            registry.shutting_down.insert(*runner_id);
        }

        (channels, runners)
    };

    // Dropped outside the locks: each cached state holds a client clone that must go away
    // before the servers can exit.
    drop(channels);

    for (runner_id, runner) in runners {
        runner.client.request_shutdown();
        drop(runner.client);
        join_runner(runner_id, runner.thread);

        // Cleared even on a timeout, otherwise every later handle for this device would
        // wait on a runner that is never coming back.
        if let Some(registry) = CHANNELS.lock().as_mut() {
            registry.shutting_down.remove(&runner_id);
        }
    }

    // A concurrent `shutdown_device` for the same device may still be joining. This call
    // promises the runners are gone once it returns, so wait that one out too.
    wait_for_device_shutdown(device_id);
}

/// Waits for `thread` to exit, giving up after [`SHUTDOWN_JOIN_TIMEOUT`].
///
/// `JoinHandle::join` has no timed variant, hence the poll on `is_finished`.
fn join_runner(runner_id: RunnerId, thread: std::thread::JoinHandle<()>) {
    let start = std::time::Instant::now();
    let mut yields: u32 = 0;

    while !thread.is_finished() {
        if start.elapsed() >= SHUTDOWN_JOIN_TIMEOUT {
            log::warn!(
                "Device runner {runner_id:?} did not stop within {SHUTDOWN_JOIN_TIMEOUT:?}, \
                 leaking the thread. Something still holds a client for it: a task parked in \
                 another device's unflushed queue, or two runners shutting each other down."
            );
            return;
        }

        if yields < SHUTDOWN_JOIN_YIELD_BUDGET {
            std::thread::yield_now();
            yields += 1;
        } else {
            std::thread::sleep(SHUTDOWN_JOIN_POLL);
        }
    }

    if thread.join().is_err() {
        log::warn!("Device runner {runner_id:?} panicked during shutdown");
    }
}

/// Blocks while any runner of `device_id` is still being wound down, by this call or a
/// concurrent one. Bounded by the same timeout as the join it is waiting on, doubled so a
/// timing-out joiner gets to clear its own mark first.
fn wait_for_device_shutdown(device_id: DeviceId) {
    let start = std::time::Instant::now();

    loop {
        let pending = CHANNELS.lock().as_ref().is_some_and(|registry| {
            registry
                .shutting_down
                .iter()
                .any(|runner_id| runner_id.device == device_id)
        });

        if !pending {
            return;
        }

        if start.elapsed() >= SHUTDOWN_JOIN_TIMEOUT * 2 {
            log::warn!(
                "A concurrent shutdown of {device_id:?} is still in flight after \
                 {SHUTDOWN_JOIN_TIMEOUT:?}, returning anyway."
            );
            return;
        }

        std::thread::sleep(SHUTDOWN_JOIN_POLL);
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
        fn_ptr: fn(&mut Task),
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
        pub fn init<F: FnOnce() + Send + 'static>(&mut self, func: F) {
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
                    if let Err(payload) = catch_unwind(AssertUnwindSafe(f)) {
                        log::warn!("{:?}", CallError::from_panic(payload));
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
                    if let Err(payload) = catch_unwind(AssertUnwindSafe(f)) {
                        log::warn!("{:?}", CallError::from_panic(payload));
                    }
                };
            } else {
                // Size or alignment exceeds both slots. Heap-allocate to get a
                // properly-aligned, pointer-sized handle, then recurse as an inline
                // task (the Box is a pointer so it trivially fits inline).
                let boxed: Box<dyn FnOnce() + Send> = Box::new(func);
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
    use super::RunnerId;
    use crate::device::handle::CallError;
    use alloc::boxed::Box;
    use std::sync::mpsc::SyncSender;

    /// Buffer size for the command channel.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// The client-side handle used to enqueue tasks.
    pub struct DeviceClient {
        state: SyncSender<Box<dyn FnOnce() + Send + 'static>>,
        runner_id: RunnerId,
    }

    impl Clone for DeviceClient {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                runner_id: self.runner_id,
            }
        }
    }

    impl DeviceClient {
        /// Gets the device id associated to the channel.
        pub fn runner_id(&self) -> &RunnerId {
            &self.runner_id
        }
        /// Creates a new channel and spawns a server thread to process it.
        pub fn new<I: FnOnce() + Send + 'static>(
            runner_id: RunnerId,
            init: I,
        ) -> (Self, std::thread::JoinHandle<()>) {
            let (sender, recv) = std::sync::mpsc::sync_channel::<Box<dyn FnOnce() + Send + 'static>>(
                CHANNEL_MAX_TASK,
            );

            let thread = std::thread::spawn(move || {
                init();
                // `Err` means every sender is gone and the queue is drained:
                // no task can ever arrive again.
                while let Ok(item) = recv.recv() {
                    item()
                }
            });

            (
                Self {
                    state: sender,
                    runner_id,
                },
                thread,
            )
        }

        /// Nothing to do: the server stops once every client is dropped and
        /// the queue is drained.
        pub fn request_shutdown(&self) {}

        /// Atomically reserves a slot in the buffer and writes the task.
        pub fn enqueue<F: FnOnce() + Send + 'static>(&self, func: F) -> Result<(), CallError> {
            self.state
                .send(Box::new(func))
                .map_err(|_| CallError::disconnected())
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
    use crate::device::handle::{
        CallError,
        channel::{
            RunnerId,
            task::{ArenaSlot, GLOBAL_TASK_MAX_SIZE, Task},
        },
    };
    use core::{
        hint::spin_loop,
        sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering},
        time::Duration,
    };
    use std::{sync::Arc, vec::Vec};

    /// Maximum number of [`Task`] that can be queued.
    pub const CHANNEL_MAX_TASK: usize = 32;

    /// Number of `spin_loop` iterations before the server starts yielding.
    /// Gives a hot window to absorb back-to-back submits without any syscall.
    const SPIN_BUDGET_SERVER: u32 = 8192;
    /// Number of `thread::yield_now` calls after the spin budget is exhausted,
    /// before the server drops to sleeping.
    const YIELD_BUDGET_SERVER: u32 = 64;
    /// Sleep duration once both the spin and yield budgets are exhausted.
    /// Bounds the wake-up latency from a fully idle state.
    const SLEEP_STEP_SERVER: Duration = Duration::from_micros(150);

    /// The client has the buffer to fill plus we add a factor of two to account for the double
    /// buffering approach.
    const CLIENT_BUDGET_FACTOR: u32 = CHANNEL_MAX_TASK as u32 * 2u32;

    /// Number of `spin_loop` iterations on the client before yielding when the
    /// queue is full. Longer than the server budget because a producer stall is
    /// expected to resolve quickly (the server only needs to swap buffers).
    const SPIN_BUDGET_CLIENT: u32 = SPIN_BUDGET_SERVER * CLIENT_BUDGET_FACTOR;
    /// Number of `thread::yield_now` calls on the client after the spin budget
    /// is exhausted, before dropping to sleeping.
    const YIELD_BUDGET_CLIENT: u32 = YIELD_BUDGET_SERVER * CLIENT_BUDGET_FACTOR;
    /// Sleep duration on the client once both budgets are exhausted. Kept short
    /// to avoid stalling the producer's critical path.
    const SLEEP_STEP_CLIENT: Duration = Duration::from_micros(75);

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
        /// Gets the runner id associated to the channel.
        pub fn runner_id(&self) -> &RunnerId {
            &self.state.runner_id
        }
        /// Creates a new channel and spawns a server thread to process it.
        pub fn new<I: FnOnce() + Send + 'static>(
            runner_id: RunnerId,
            init: I,
        ) -> (Self, std::thread::JoinHandle<()>) {
            let mut server = Server::new(runner_id);
            let state = server.state.clone();

            let thread = std::thread::Builder::new()
                .name(std::format!(
                    "DS{}-{}-{}",
                    match runner_id.stage {
                        crate::device::DeviceServiceStage::Upstream => "U",
                        crate::device::DeviceServiceStage::Downstream => "D",
                    },
                    runner_id.device.type_id,
                    runner_id.device.index_id
                ))
                .spawn(move || {
                    init();
                    server.start();
                })
                .unwrap();

            (Self { state }, thread)
        }

        /// Signals the server to run any remaining tasks and stop once every
        /// client is dropped. Queued tasks may themselves hold clients, so the
        /// server keeps draining until the last one is gone.
        pub fn request_shutdown(&self) {
            self.state.shutdown.store(true, Ordering::Release);
        }

        /// Atomically reserves a slot in the buffer and writes the task.
        pub fn enqueue<F: FnOnce() + Send + 'static>(&self, func: F) -> Result<(), CallError> {
            let mut idle_count: u32 = 0;
            loop {
                let index = self.state.available_index.fetch_add(1, Ordering::Acquire) as usize;
                if index >= CHANNEL_MAX_TASK {
                    // The queue is full; back off until the server flushes/swaps buffers.
                    if idle_count < SPIN_BUDGET_CLIENT {
                        spin_loop();
                    } else if idle_count < SPIN_BUDGET_CLIENT + YIELD_BUDGET_CLIENT {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(SLEEP_STEP_CLIENT);
                    }
                    idle_count = idle_count.saturating_add(1);
                    continue;
                }

                self.state.init_task_at(index, func);
                self.state.enqueued_count.fetch_add(1, Ordering::SeqCst);
                return Ok(());
            }
        }

        /// Forces a flush by filling the remaining buffer with no-op tasks.
        pub fn flush(&self) {
            self.state.pad_with_noops();
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
        /// Set by [`DeviceClient::request_shutdown`]; the server winds down
        /// once it is set and every client is dropped.
        shutdown: AtomicBool,
        /// The runner id (for debugging purposes).
        runner_id: RunnerId,
    }

    impl State {
        /// Fills the rest of the current queue with no-op tasks so a partially
        /// filled buffer reaches [`CHANNEL_MAX_TASK`] and the server swaps it in.
        ///
        /// Both sides of the protocol go through here, [`DeviceClient::flush`] and
        /// [`Server::try_shutdown`], so the two copies cannot drift apart.
        ///
        /// `inline` keeps the client side, which is on the submit path, exactly as it was
        /// when the body was spelled out in [`DeviceClient::flush`].
        #[inline]
        fn pad_with_noops(&self) {
            let index_start =
                self.available_index
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
                self.init_task_at(index, || ());
            }

            let actual_added = index_end - index_start;
            self.enqueued_count
                .fetch_add(actual_added as u32, Ordering::SeqCst);
        }

        /// Initializes the task at `index` in the current queue with `func`.
        /// Exclusive access per slot is guaranteed by `available_index.fetch_add`.
        fn init_task_at<F: FnOnce() + Send + 'static>(&self, index: usize, func: F) {
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
        fn new(runner_id: RunnerId) -> Self {
            let mut buffers = [TaskBuffer::new(), TaskBuffer::new()];

            let state = Arc::new(State {
                queue_ptr: AtomicPtr::new(buffers[0].tasks.as_mut_ptr()),
                available_index: AtomicU32::new(0),
                enqueued_count: AtomicU32::new(0),
                shutdown: AtomicBool::new(false),
                runner_id,
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
            let mut idle_count: u32 = 0;
            loop {
                if self.ready_to_execute {
                    self.execute_tasks();
                    idle_count = 0;
                }

                let queue_size = self.state.enqueued_count.load(Ordering::Acquire) as usize;

                if queue_size >= CHANNEL_MAX_TASK {
                    self.fetch();
                    idle_count = 0;
                    continue;
                }

                if idle_count < SPIN_BUDGET_SERVER {
                    spin_loop();
                } else {
                    // Past the hot window, so this costs nothing on the fast
                    // path. A strong count of one means this server owns the
                    // only reference to the state: every client is gone and no
                    // new task can arrive.
                    if (self.state.shutdown.load(Ordering::Acquire)
                        || Arc::strong_count(&self.state) == 1)
                        && self.try_shutdown()
                    {
                        return;
                    }
                    if idle_count < SPIN_BUDGET_SERVER + YIELD_BUDGET_SERVER {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(SLEEP_STEP_SERVER);
                    }
                }
                idle_count = idle_count.saturating_add(1);
            }
        }

        /// Winds the server down once no more work can arrive.
        ///
        /// Returns `true` when this server holds the last reference to its state
        /// and the queue is empty, meaning the thread can stop. Otherwise pads the
        /// buffer with no-ops exactly like a client flush so the main loop executes
        /// what is queued, and returns `false`. Queued tasks may hold client clones,
        /// which is why draining must happen before the strong count can reach one.
        ///
        /// # Notes
        ///
        /// The strong count is read *before* `enqueued_count`, and that order is
        /// load-bearing. Once the count is one there is no producer left and none can
        /// appear, which is what makes the queue read that follows authoritative.
        /// Reading the queue first would let the last client enqueue a task and drop
        /// its handle in between the two loads: the server would see a stale empty
        /// queue next to a fresh count of one and exit with a written-but-never-run
        /// task, leaking it and everything it captured (see [`Task::run`]).
        fn try_shutdown(&mut self) -> bool {
            let is_last_ref = Arc::strong_count(&self.state) == 1;

            if is_last_ref {
                // `Arc::strong_count` is a relaxed load. This fence pairs with the
                // release on the last client's `Arc` drop, so every task those
                // clients wrote is visible to the `enqueued_count` load below.
                core::sync::atomic::fence(Ordering::Acquire);
            }

            if self.state.enqueued_count.load(Ordering::Acquire) > 0 {
                self.state.pad_with_noops();
                return false;
            }

            is_last_ref
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
    use crate::device::handle::CallResultExt;
    use crate::device::handle::DeviceFixture;
    // Only the `cfg(not(miri))` flushing test uses this.
    #[cfg(not(miri))]
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

    /// A [`MockService`] handle on a device of its own, whose runner is shut down when
    /// the returned value drops.
    fn mock_fixture() -> DeviceFixture<ChannelDeviceHandle<MockService>> {
        DeviceFixture::new(ChannelDeviceHandle::<MockService>::new, shutdown_device)
    }

    #[test]
    fn test_basic_execution_and_state_persistence() {
        let handle = mock_fixture();

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
        let handle = mock_fixture();

        let local_val = 42; // This lives on the test stack

        // Test exclusive_scoped
        let result = handle.exclusive(|| local_val + 8).unwrap();

        assert_eq!(result, 50);

        // Test submit_blocking_scoped
        let result_mut = handle
            .submit_blocking(|state| {
                state.counter = local_val;
                state.counter
            })
            .unwrap();

        assert_eq!(result_mut, 42);
    }

    #[test]
    #[cfg(not(miri))]
    fn test_buffer_flushing_at_limit() {
        let handle = mock_fixture();
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
        let handle = mock_fixture();
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

    /// A task the last client enqueued without flushing still runs before the runner
    /// exits. Nothing but the shutdown drain can execute it: the buffer is partial, so
    /// the main loop never swaps it in on its own.
    #[test]
    fn test_shutdown_drains_a_task_the_last_client_left_queued() {
        let ran = Arc::new(AtomicUsize::new(0));

        {
            let handle = mock_fixture();
            let counter = Arc::clone(&ran);
            handle.submit(move |_state| {
                counter.fetch_add(1, Ordering::SeqCst);
            });
            // The handle drops first, then the fixture's guard shuts the runner down.
        }

        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "the queued task must run before the runner thread exits"
        );
    }

    /// A handle created for a device that was shut down gets a fresh runner with fresh
    /// state, rather than hanging on the old one or reviving it.
    #[test]
    fn test_handle_created_after_shutdown_gets_a_fresh_runner() {
        let device_id = {
            let handle = mock_fixture();
            handle.submit_blocking(|state| state.counter += 1).unwrap();
            handle.device_id()
        };

        let handle = ChannelDeviceHandle::<MockService>::new(device_id);
        let counter = handle.submit_blocking(|state| state.counter).unwrap();
        assert_eq!(counter, 0, "the new runner must start from a fresh service");

        drop(handle);
        shutdown_device(device_id);
    }

    #[test]
    fn test_closure_captures_are_dropped_after_execution() {
        let handle = mock_fixture();

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
        let handle = mock_fixture();

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
        let handle = mock_fixture();

        let huge_data = [7u8; 8192]; // 8KB > 4096 byte arena limit
        let result = handle
            .submit_blocking(move |_state| huge_data[0] + huge_data[8191])
            .unwrap();

        assert_eq!(result, 14);
    }

    #[test]
    fn test_large_closure_drop_is_called() {
        // Verify that Drop runs correctly for closures stored in the arena.
        let handle = mock_fixture();
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
        // The fixture's own handle is one of the racing callers, so it is created after
        // `INIT_CALLS` is reset and counts towards the single expected init.
        let fixture =
            DeviceFixture::new(ChannelDeviceHandle::<CountingService>::new, shutdown_device);
        let device_id = fixture.device_id();

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

    /// If the task panics, `submit_blocking` must return `Err`, drop the
    /// task's captures exactly once, and leave the channel usable.
    #[test]
    fn test_submit_blocking_panic_drops_and_returns_err() {
        let handle = mock_fixture();
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy(Arc<AtomicUsize>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy(Arc::clone(&drop_count));
        let result = handle.submit_blocking(move |_state| {
            let _ = &spy;
            panic!("boom");
        });

        let err = result.expect_err("panicking task must return Err");
        assert_eq!(
            err.message(),
            Some("boom"),
            "the panic message must be preserved in the CallError"
        );
        assert_eq!(
            drop_count.load(Ordering::SeqCst),
            1,
            "captures must be dropped exactly once on panic"
        );

        // Channel survives: next task still runs.
        let ok = handle.submit_blocking(|state| state.counter).unwrap();
        assert_eq!(ok, 0);
    }

    /// Same guarantees for `exclusive`.
    #[test]
    fn test_exclusive_panic_drops_and_returns_err() {
        let handle = mock_fixture();
        let drop_count = Arc::new(AtomicUsize::new(0));

        struct DropSpy(Arc<AtomicUsize>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let spy = DropSpy(Arc::clone(&drop_count));
        let result: Result<(), _> = handle.exclusive(move || {
            let _ = &spy;
            panic!("boom");
        });

        let err = result.expect_err("panicking task must return Err");
        assert_eq!(
            err.message(),
            Some("boom"),
            "the panic message must be preserved in the CallError"
        );
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
        let ok = handle.exclusive(|| 7).unwrap();
        assert_eq!(ok, 7);
    }

    /// A.1 — Submit a panic payload as task, capture the payload and return to caller.
    #[test]
    fn test_submit_blocking_preserves_formatted_string_payload() {
        let handle = mock_fixture();

        let result = handle.submit_blocking(|_state| {
            panic!("value {}", 99);
        });

        let err = result.expect_err("panicking task must return Err");
        assert_eq!(err.message(), Some("value 99"));
    }

    /// A.2 — Submit a custom panic payload, capture the payload and return to caller.
    #[test]
    fn test_submit_blocking_preserves_non_string_payload() {
        let handle = mock_fixture();

        #[derive(Debug, PartialEq)]
        struct Boom {
            code: u32,
            what: &'static str,
        }

        let result = handle.submit_blocking(|_state| {
            std::panic::panic_any(Boom {
                code: 7,
                what: "kaboom",
            });
        });

        let err = result.expect_err("panicking task must return Err");
        assert_eq!(err.message(), None, "a non-string payload has no message");
        let payload = err.into_panic().expect("the payload must be preserved");
        let boom = *payload
            .downcast::<Boom>()
            .expect("payload must downcast to the original type");
        assert_eq!(
            boom,
            Boom {
                code: 7,
                what: "kaboom",
            }
        );
    }

    #[test]
    fn test_submit_blocking_preserves_scalar_payload() {
        let handle = mock_fixture();

        let result = handle.submit_blocking(|_state| {
            std::panic::panic_any(42i32);
        });

        let err = result.expect_err("panicking task must return Err");
        assert_eq!(err.message(), None);
        let payload = err.into_panic().expect("the payload must be preserved");
        assert_eq!(
            *payload.downcast::<i32>().expect("payload must be an i32"),
            42
        );
    }

    /// B.1 — A real index-out-of-bounds panic (the symptom from the issue) keeps its
    /// message instead of being erased into a generic error.
    #[test]
    fn test_submit_blocking_preserves_index_out_of_bounds_message() {
        let handle = mock_fixture();

        let result = handle.submit_blocking(|_state| {
            let data = [10u8, 20u8];
            // `black_box` hides the index so the access happens at runtime rather than
            // tripping the const-eval `unconditional_panic` lint.
            let idx = core::hint::black_box(5usize);
            let _ = data[idx];
        });

        let err = result.expect_err("panicking task must return Err");
        let message = err
            .message()
            .expect("an index panic carries a string message");
        assert!(
            message.contains("index out of bounds"),
            "unexpected message: {message}"
        );
    }

    /// B.2 — A real `unwrap()` panic keeps its message (the autotune symptom).
    #[test]
    fn test_submit_blocking_preserves_unwrap_message() {
        let handle = mock_fixture();

        let result = handle.submit_blocking(|_state| {
            let value: Result<(), &str> = Err("nope");
            #[allow(clippy::unnecessary_literal_unwrap)]
            value.unwrap();
        });

        let err = result.expect_err("panicking task must return Err");
        let message = err
            .message()
            .expect("an unwrap panic carries a string message");
        assert!(message.contains("unwrap"), "unexpected message: {message}");
    }

    /// C.1 — The captured payload can re-raise the original panic via `resume_unwind`,
    /// proving it is the genuine payload and not a lossy copy.
    #[test]
    fn test_into_panic_can_be_resumed() {
        let handle = mock_fixture();

        let result = handle.submit_blocking(|_state| {
            panic!("re-raise me");
        });

        let err = result.expect_err("panicking task must return Err");
        assert_eq!(err.message(), Some("re-raise me"));

        let payload = err.into_panic().expect("the payload must be preserved");
        // Re-raise on this thread and catch it again: the round-tripped payload must
        // still hold the original message.
        let recaught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            std::panic::resume_unwind(payload);
        }));
        let payload = recaught.expect_err("resume_unwind must re-panic");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("re-raise me"),
            "the re-raised panic must carry the original message"
        );
    }

    /// E.1 — `exclusive` preserves a non-string payload just like `submit_blocking`.
    #[test]
    fn test_exclusive_preserves_non_string_payload() {
        let handle = mock_fixture();

        #[derive(Debug, PartialEq)]
        struct Boom {
            code: u32,
            what: &'static str,
        }

        let result: Result<(), _> = handle.exclusive(|| {
            std::panic::panic_any(Boom {
                code: 9,
                what: "exclusive",
            });
        });

        let err = result.expect_err("panicking task must return Err");
        assert_eq!(err.message(), None);
        let payload = err.into_panic().expect("the payload must be preserved");
        let boom = *payload
            .downcast::<Boom>()
            .expect("payload must downcast to the original type");
        assert_eq!(
            boom,
            Boom {
                code: 9,
                what: "exclusive",
            }
        );
    }

    /// E.3 — The runner thread survives repeated panics, each call surfaces its own
    /// payload, and a later normal task still succeeds.
    #[test]
    fn test_channel_survives_repeated_panics_each_preserved() {
        let handle = mock_fixture();

        let first = handle.submit_blocking(|_state| panic!("first"));
        assert_eq!(
            first.expect_err("first panic must return Err").message(),
            Some("first")
        );

        let second = handle.submit_blocking(|_state| panic!("second"));
        assert_eq!(
            second.expect_err("second panic must return Err").message(),
            Some("second")
        );

        // The runner thread is still alive: a normal task runs and returns Ok.
        let counter = handle.submit_blocking(|state| state.counter).unwrap();
        assert_eq!(counter, 0);
    }

    /// 2-E1 — `unwrap_or_resume` re-raises a runner-thread panic as the *original*
    /// panic on the caller (faithful message), and the runner thread survives.
    #[test]
    fn test_unwrap_or_resume_reraises_submit_blocking_panic() {
        let handle = mock_fixture();

        let reraised = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle
                .submit_blocking(|_state| {
                    panic!("device boom");
                })
                .unwrap_or_resume()
        }));

        let payload = reraised.expect_err("the original panic must be re-raised at the caller");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("device boom"),
            "the re-raised panic must carry the original message"
        );

        // The runner thread survived the captured-and-re-raised panic.
        let counter = handle.submit_blocking(|state| state.counter).unwrap();
        assert_eq!(counter, 0);
    }

    /// 2-E2 — `unwrap_or_resume` re-raises a non-string payload from `exclusive`
    /// end to end, preserving the exact payload object.
    #[test]
    fn test_unwrap_or_resume_reraises_exclusive_non_string_payload() {
        let handle = mock_fixture();

        #[derive(Debug, PartialEq)]
        struct Boom {
            code: u32,
            what: &'static str,
        }

        let reraised = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle
                .exclusive(|| {
                    std::panic::panic_any(Boom {
                        code: 9,
                        what: "exclusive",
                    });
                })
                .unwrap_or_resume()
        }));

        let payload = reraised.expect_err("the original panic must be re-raised at the caller");
        let boom = *payload
            .downcast::<Boom>()
            .expect("the re-raised payload must be the original object");
        assert_eq!(
            boom,
            Boom {
                code: 9,
                what: "exclusive",
            }
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
