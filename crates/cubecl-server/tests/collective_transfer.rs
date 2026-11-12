use cubecl_common::{
    bytes::Bytes,
    device::{DeviceId, ServiceId},
    profile::ProfileDuration,
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_ir::{ElemType, MemoryDeviceProperties};
use cubecl_server::{
    command::{
        CollectiveDriver, CollectiveServer, Collectives, Command, CopyLayout, DeviceResource,
        DeviceStream, Driver, Elements, HostResource, StreamSignal,
    },
    device_events::{EventApi, EventProfiler},
    driver::DriverError,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, FailureId, ManagedMemoryBinding, MemoryConfiguration, MemoryManagement,
        MemoryManagementOptions,
        drop_queue::{Fence, PendingDropQueue},
    },
    metadata_cache::{MetadataCachePolicy, MetadataInfoCache},
    server::{
        BufferBinding, CommunicationId, CommunicationTiming, CopyDescriptor, DeviceCaptures,
        Handle, IoError, LaunchError, ReduceOperation, SendOutcome, ServerError,
    },
    storage::BytesStorage,
    stream::{EventStreamBackend, FailureStore, MultiStream, StreamCapture, StreamMemory},
};
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;

const SIZE: u64 = 4;
const STREAM: StreamId = StreamId { value: 1 };

/// What the driver has been asked to post between two devices.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Posted {
    sends: usize,
    recvs: usize,
}

struct Wire {
    posted: Mutex<Posted>,
    recv_posted: Condvar,
    refused_rank: AtomicUsize,
}

static WIRES: LazyLock<Mutex<HashMap<CommunicationId, &'static Wire>>> =
    LazyLock::new(Default::default);

impl Wire {
    fn named(id: &CommunicationId) -> &'static Self {
        WIRES.lock().unwrap().entry(id.clone()).or_insert_with(|| {
            Box::leak(Box::new(Wire {
                posted: Default::default(),
                recv_posted: Condvar::new(),
                refused_rank: AtomicUsize::new(usize::MAX),
            }))
        })
    }

    fn between(a: DeviceId, b: DeviceId) -> &'static Self {
        Self::named(&CommunicationId::transfers_between(a, b))
    }

    fn posted(&self) -> Posted {
        *self.posted.lock().unwrap()
    }

    /// The first transfer of a pair connects it, which holds the send until its recv is posted.
    fn post_send(&self) {
        let mut posted = self.posted.lock().unwrap();
        posted.sends += 1;
        let sends = posted.sends;
        let (posted, timeout) = self
            .recv_posted
            .wait_timeout_while(posted, Duration::from_secs(10), |posted| {
                posted.recvs < sends
            })
            .unwrap();
        drop(posted);
        assert!(!timeout.timed_out(), "the send never saw its recv posted");
    }

    fn post_recv(&self) {
        self.posted.lock().unwrap().recvs += 1;
        self.recv_posted.notify_all();
    }
}

#[derive(Debug)]
struct Fake;

#[derive(Debug)]
struct FakeBackend;

struct FakeFence;

impl Fence for FakeFence {
    fn wait(self) -> Result<(), ServerError> {
        Ok(())
    }
}

struct FakeStream {
    device_memory: MemoryManagement<BytesStorage>,
    host_memory: MemoryManagement<BytesStorage>,
    drop_queue: PendingDropQueue<FakeFence>,
    capturing: StreamCapture,
    info_cache: MetadataInfoCache<Handle>,
}

impl fmt::Debug for FakeStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FakeStream")
    }
}

impl StreamMemory for FakeStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.device_memory.failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.device_memory
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.device_memory
            .written(&binding.memory, binding.range(), failures)
    }
}

impl DeviceStream for FakeStream {
    type Fence = FakeFence;
    type DeviceStorage = BytesStorage;
    type HostStorage = BytesStorage;
    type Signal = ();

    fn device_memory(&mut self) -> &mut MemoryManagement<Self::DeviceStorage> {
        &mut self.device_memory
    }

    fn host_memory(&mut self) -> &mut MemoryManagement<Self::HostStorage> {
        &mut self.host_memory
    }

    fn drop_queue(&mut self) -> &mut PendingDropQueue<Self::Fence> {
        &mut self.drop_queue
    }

    fn capturing(&mut self) -> &mut StreamCapture {
        &mut self.capturing
    }

    fn info_cache(&mut self) -> &mut MetadataInfoCache<Handle> {
        &mut self.info_cache
    }

    fn signal(&self) -> Self::Signal {}

    fn fence(_signal: Self::Signal) -> Self::Fence {
        FakeFence
    }
}

fn memory(name: &'static str) -> MemoryManagement<BytesStorage> {
    MemoryManagement::from_configuration(
        BytesStorage::default(),
        &MemoryDeviceProperties::new(1024 * 1024, 32),
        MemoryConfiguration::default(),
        Arc::new(ServerLogger::default()),
        MemoryManagementOptions::new(name),
    )
}

impl EventStreamBackend for FakeBackend {
    type Stream = FakeStream;
    type Event = ();

    fn create_stream(&self) -> Result<Self::Stream, ServerError> {
        Ok(FakeStream {
            device_memory: memory("device"),
            host_memory: memory("host"),
            drop_queue: Default::default(),
            capturing: StreamCapture::new(&DeviceCaptures::default()),
            info_cache: MetadataInfoCache::new(MetadataCachePolicy::default()),
        })
    }

    fn handle_cursor(_stream: &Self::Stream, _handle: &BufferBinding) -> u64 {
        0
    }

    fn flush(_stream: &mut Self::Stream, _failures: &mut ErrorGraph) -> Self::Event {}

    fn wait_event(_stream: &mut Self::Stream, _event: Self::Event) -> Result<(), ServerError> {
        Ok(())
    }

    fn wait_event_sync(_event: Self::Event) -> Result<(), ServerError> {
        Ok(())
    }
}

impl Driver for Fake {
    type Backend = FakeBackend;
    type Stream = FakeStream;
    type Context = ();
    type LaunchArgs = ();
    type Loaded = ();

    unsafe fn pinned_bytes(
        _binding: ManagedMemoryBinding,
        _resource: HostResource<Self>,
        _size: usize,
    ) -> Bytes {
        unreachable!("a transfer stages nothing")
    }

    unsafe fn copy_to_host(
        _resource: &DeviceResource<Self>,
        _layout: &CopyLayout<'_>,
        _bytes: &mut Bytes,
        _stream: &Self::Stream,
    ) -> Result<(), IoError> {
        unreachable!("a transfer stages nothing")
    }

    unsafe fn copy_to_device(
        _resource: &DeviceResource<Self>,
        _layout: &CopyLayout<'_>,
        _data: &[u8],
        _stream: &Self::Stream,
    ) -> Result<(), IoError> {
        unreachable!("a transfer stages nothing")
    }

    unsafe fn copy_on_device(
        _source: &DeviceResource<Self>,
        _target: &DeviceResource<Self>,
        _queue: (),
    ) -> Result<(), IoError> {
        unreachable!("a transfer relocates nothing")
    }

    fn launch(
        _ctx: &mut Self::Context,
        _stream: &mut Self::Stream,
        _id: &KernelId,
        _kernel: &Self::Loaded,
        _count: (u32, u32, u32),
        _args: &mut Self::LaunchArgs,
    ) -> Result<(), LaunchError> {
        unreachable!("a transfer launches nothing")
    }
}

impl CollectiveDriver for Fake {
    type Communicator = &'static Wire;
    type UniqueId = &'static Wire;
    type DataType = ();

    const LIBRARY: &'static str = "the fake";

    fn wait_for(_stream: (), _queued_on: ()) -> Result<(), ServerError> {
        Ok(())
    }

    fn group_id(id: &CommunicationId) -> Result<Self::UniqueId, ServerError> {
        Ok(Wire::named(id))
    }

    fn join(
        id: Self::UniqueId,
        _ranks: usize,
        rank: usize,
    ) -> Result<Self::Communicator, ServerError> {
        if id.refused_rank.load(Ordering::Relaxed) == rank {
            return Err(error("the source could not join"));
        }
        Ok(id)
    }

    fn data_type(_dtype: ElemType) -> Option<Self::DataType> {
        Some(())
    }

    fn all_reduce(
        _comm: &Self::Communicator,
        _src: &DeviceResource<Self>,
        _dst: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _op: ReduceOperation,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        unreachable!("only transfers run here")
    }

    fn all_gather(
        _comm: &Self::Communicator,
        _src: &DeviceResource<Self>,
        _dst: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        unreachable!("only transfers run here")
    }

    fn reduce_scatter(
        _comm: &Self::Communicator,
        _src: &DeviceResource<Self>,
        _dst: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _op: ReduceOperation,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        unreachable!("only transfers run here")
    }

    fn broadcast(
        _comm: &Self::Communicator,
        _src: Option<&DeviceResource<Self>>,
        _dst: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _root: usize,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        unreachable!("only transfers run here")
    }

    fn all_to_all(
        _comm: &Self::Communicator,
        _src: &DeviceResource<Self>,
        _dst: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _ranks: usize,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        unreachable!("only transfers run here")
    }

    fn send(
        comm: &Self::Communicator,
        _src: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _peer: usize,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        comm.post_send();
        Ok(())
    }

    fn recv(
        comm: &Self::Communicator,
        _dst: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _peer: usize,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        comm.post_recv();
        Ok(())
    }
}

/// The device clock every fake stream shares, a tick per event recorded.
static TICKS: AtomicU64 = AtomicU64::new(0);

std::thread_local! {
    /// The events this thread has taken from the driver, and recorded.
    static EVENTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

fn events_on_this_thread() -> (usize, usize) {
    EVENTS.with(Cell::get)
}

struct FakeEvents;

struct FakeEvent(Mutex<Option<u64>>);

impl EventApi for FakeEvents {
    type Event = FakeEvent;
    type Stream = ();

    const BACKEND: &'static str = "fake";

    fn event_create() -> Result<Self::Event, DriverError> {
        EVENTS.with(|events| events.set((events.get().0 + 1, events.get().1)));
        Ok(FakeEvent(Mutex::new(None)))
    }

    fn event_destroy(_event: &mut Self::Event) -> Result<(), DriverError> {
        Ok(())
    }

    fn event_record(event: &Self::Event, _stream: ()) -> Result<(), DriverError> {
        EVENTS.with(|events| events.set((events.get().0, events.get().1 + 1)));
        *event.0.lock().unwrap() = Some(TICKS.fetch_add(1, Ordering::Relaxed));
        Ok(())
    }

    fn event_wait(_event: &Self::Event) -> Result<(), DriverError> {
        Ok(())
    }

    fn event_elapsed(
        start: &Self::Event,
        end: &Self::Event,
    ) -> Result<cubecl_common::profile::Duration, DriverError> {
        let start = start.0.lock().unwrap().expect("recorded");
        let end = end.0.lock().unwrap().expect("recorded");
        Ok(cubecl_common::profile::Duration::from_micros(
            end.saturating_sub(start),
        ))
    }

    fn stream_wait_event(_stream: (), _event: &Self::Event) -> Result<(), DriverError> {
        Ok(())
    }

    fn stream_synchronize(_stream: ()) -> Result<(), DriverError> {
        Ok(())
    }

    fn stream_create_non_blocking() -> Result<(), DriverError> {
        Ok(())
    }

    fn stream_destroy(_stream: ()) -> Result<(), DriverError> {
        Ok(())
    }
}

struct FakeServer {
    device: DeviceId,
    service: ServiceId,
    ctx: (),
    collectives: Collectives<Fake>,
    streams: MultiStream<FakeBackend>,
    profiler: EventProfiler<FakeEvents>,
}

impl CollectiveServer for FakeServer {
    type Driver = Fake;
    type Events = FakeEvents;

    fn device_id(&self) -> DeviceId {
        self.device
    }

    fn collectives(&mut self) -> &mut Collectives<Fake> {
        &mut self.collectives
    }

    fn profiler(&mut self) -> &mut EventProfiler<FakeEvents> {
        &mut self.profiler
    }

    fn streams(&mut self) -> &mut MultiStream<FakeBackend> {
        &mut self.streams
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<Command<'_, Fake>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles)?;
        Ok(Command::new(&mut self.ctx, streams, self.service))
    }

    fn comm_stream(&self) -> StreamSignal<Fake> {}

    fn transfer_stream(&self) -> StreamSignal<Fake> {}

    fn connect_element(&self) -> &DeviceResource<Fake> {
        unreachable!("only transfers run here")
    }
}

impl FakeServer {
    fn new(device: DeviceId) -> Self {
        Self {
            device,
            service: ServiceId::of::<Self>(device),
            ctx: (),
            collectives: Collectives::new(device),
            streams: MultiStream::new(Arc::new(ServerLogger::default()), FakeBackend, 4),
            profiler: EventProfiler::default(),
        }
    }

    fn unallocated(&self) -> Handle {
        Handle::new(self.service, STREAM, SIZE)
    }

    /// A buffer no failure claims.
    fn allocated(&mut self) -> Handle {
        let handle = self.unallocated();
        let mut command = self.command(STREAM, [].into_iter()).unwrap();
        let memory = command.reserve(SIZE).unwrap();
        command.bind(memory, handle.memory.clone()).unwrap();
        drop(command);
        handle
    }

    /// A buffer whose allocation failed with `reason`.
    fn out_of_memory(&mut self, reason: &str) -> Handle {
        let handle = self.unallocated();
        self.streams.fail_unallocated(&handle.memory, error(reason));
        handle
    }

    fn read_failure(&mut self, handle: &Handle) -> String {
        let error = self
            .streams
            .ensure_written([&handle.clone().binding()].into_iter())
            .expect_err("the read should fail");
        format!("{error}")
    }
}

fn error(reason: &str) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}

fn descriptor(handle: Handle) -> CopyDescriptor {
    CopyDescriptor::new(handle.binding(), [SIZE as usize].into(), [1].into(), 1)
}

/// Device indices no other test uses, since a wire outlives its test.
fn devices(first: u16) -> (DeviceId, DeviceId) {
    (DeviceId::new(0, first), DeviceId::new(0, first + 1))
}

#[test_log::test]
fn a_refused_send_posts_nothing_and_fails_the_read_of_its_recv() {
    let (src, dst) = devices(10);
    let (mut source, mut destination) = (FakeServer::new(src), FakeServer::new(dst));
    let refused = source.out_of_memory("out of device memory");
    let received = destination.unallocated();
    let (report, outcome) = SendOutcome::channel();

    let sent = source.send(descriptor(refused), STREAM, dst, report);
    let recv = destination.recv(received.clone(), STREAM, src, outcome);

    assert!(sent.is_err());
    assert!(recv.is_err());
    assert_eq!(Wire::between(src, dst).posted(), Posted::default());
    let failure = destination.read_failure(&received);
    assert!(failure.contains("out of device memory"), "{failure}");
}

#[test_log::test]
fn a_send_that_cannot_set_up_tells_its_recv_why() {
    let (src, dst) = devices(20);
    Wire::between(src, dst)
        .refused_rank
        .store(0, Ordering::Relaxed);
    let (mut source, mut destination) = (FakeServer::new(src), FakeServer::new(dst));
    let written = source.allocated();
    let received = destination.unallocated();
    let (report, outcome) = SendOutcome::channel();

    source
        .send(descriptor(written), STREAM, dst, report)
        .unwrap_err();
    destination
        .recv(received.clone(), STREAM, src, outcome)
        .unwrap_err();

    assert_eq!(Wire::between(src, dst).posted(), Posted::default());
    let failure = destination.read_failure(&received);
    assert!(failure.contains("the source could not join"), "{failure}");
}

#[test_log::test]
fn a_recv_is_posted_while_its_send_connects() {
    let (src, dst) = devices(30);
    let (report, outcome) = SendOutcome::channel();

    std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut source = FakeServer::new(src);
            let written = source.allocated();
            source
                .send(descriptor(written), STREAM, dst, report)
                .unwrap();
        });
        scope.spawn(move || {
            let mut destination = FakeServer::new(dst);
            let received = destination.unallocated();
            destination.recv(received, STREAM, src, outcome).unwrap();
        });
    });

    assert_eq!(
        Wire::between(src, dst).posted(),
        Posted { sends: 1, recvs: 1 }
    );
}

/// Runs a send and its recv between two fresh servers, each on its own thread since the send
/// waits for its recv, and returns the events each thread took and recorded.
fn transfer(
    devices: (DeviceId, DeviceId),
    timings: [Option<CommunicationTiming>; 2],
) -> [(usize, usize); 2] {
    let (src, dst) = devices;
    let (report, outcome) = SendOutcome::channel();
    let [send_timing, recv_timing] = timings;

    std::thread::scope(|scope| {
        let sent = scope.spawn(move || {
            let mut source = FakeServer::new(src);
            let written = source.allocated();
            if let Some(timing) = send_timing {
                source.time_next_communication(timing);
            }
            source
                .send(descriptor(written), STREAM, dst, report)
                .unwrap();
            events_on_this_thread()
        });
        let received = scope.spawn(move || {
            let mut destination = FakeServer::new(dst);
            let received = destination.unallocated();
            if let Some(timing) = recv_timing {
                destination.time_next_communication(timing);
            }
            destination.recv(received, STREAM, src, outcome).unwrap();
            events_on_this_thread()
        });
        [sent.join().unwrap(), received.join().unwrap()]
    })
}

fn read(window: ProfileDuration) -> cubecl_common::profile::Duration {
    cubecl_environment::future::block_on(window.resolve())
        .expect("the server sent the window")
        .duration()
}

#[test_log::test]
fn an_untimed_transfer_takes_and_records_no_event() {
    let events = transfer(devices(40), [None, None]);

    assert_eq!(events, [(0, 0), (0, 0)]);
}

#[test_log::test]
fn a_timed_transfer_sends_the_window_of_each_half_and_of_its_wait() {
    let mut timings = [
        CommunicationTiming::default(),
        CommunicationTiming::default(),
    ];
    let windows: Vec<_> = timings
        .iter_mut()
        .flat_map(|timing| [timing.part_window(), timing.wait_window()])
        .collect();

    let events = transfer(devices(50), timings.map(Some));

    // Three marks per half, and the anchor that places them.
    assert_eq!(events, [(4, 4), (4, 4)]);
    let read: Vec<_> = windows.into_iter().map(read).collect();
    let [send, send_wait, recv, recv_wait] = read[..] else {
        unreachable!("two windows per half");
    };
    assert!(send > cubecl_common::profile::Duration::ZERO);
    assert!(recv > cubecl_common::profile::Duration::ZERO);
    // Each compute stream reaches its wait where it hands the half over, before the half starts.
    assert!(send_wait > send);
    assert!(recv_wait > recv);
}

#[test_log::test]
fn a_timing_its_refused_send_never_used_is_dropped_rather_than_given_to_the_next() {
    let (src, dst) = devices(60);
    let mut timing = CommunicationTiming::default();
    let window = timing.part_window();
    let (report, outcome) = SendOutcome::channel();

    let untimed = std::thread::scope(|scope| {
        let sent = scope.spawn(move || {
            let mut source = FakeServer::new(src);
            let refused = source.out_of_memory("out of device memory");
            let (refused_report, _) = SendOutcome::channel();
            source.time_next_communication(timing);
            source
                .send(descriptor(refused), STREAM, dst, refused_report)
                .unwrap_err();

            let before = events_on_this_thread();
            let written = source.allocated();
            source
                .send(descriptor(written), STREAM, dst, report)
                .unwrap();
            before == events_on_this_thread()
        });
        scope.spawn(move || {
            let mut destination = FakeServer::new(dst);
            let received = destination.unallocated();
            destination.recv(received, STREAM, src, outcome).unwrap();
        });
        sent.join().unwrap()
    });

    assert!(untimed, "the next send took timing events");
    assert!(cubecl_environment::future::block_on(window.resolve()).is_none());
}
