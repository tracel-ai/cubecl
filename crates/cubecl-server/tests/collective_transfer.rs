mod fake;

use cubecl_common::{
    device::{DeviceId, ServiceId},
    profile::ProfileDuration,
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_ir::{ElemType, FloatKind};
use cubecl_server::{
    command::{
        CollectiveDriver, CollectiveServer, Collectives, Command, DeviceResource, DeviceStream,
        Elements, StreamSignal,
    },
    device_events::{EventApi, EventProfiler},
    driver::DriverError,
    logging::ServerLogger,
    server::{
        BufferBinding, CommunicationId, CommunicationTiming, ComputeWait, CopyDescriptor, Handle,
        ReduceOperation, Reduction, SendOutcome, ServerError,
    },
    stream::{FailureStore, MultiStream, StreamCapture, WriteScoped},
};
use fake::{Fake, FakeBackend};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
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

    /// The wire a communicator of [`CommunicationId`] `id` posts on.
    fn of(id: u64) -> &'static Self {
        Self::named(&CommunicationId { id })
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

impl CollectiveDriver for Fake {
    /// The wire's [`CommunicationId`], which [`Wire::of`] looks up.
    type Communicator = u64;
    type UniqueId = u64;
    type DataType = ();

    const LIBRARY: &'static str = "the fake";

    fn wait_for(_stream: (), _queued_on: ()) -> Result<(), ServerError> {
        count_wait();
        Ok(())
    }

    fn group_id(id: &CommunicationId) -> Result<Self::UniqueId, ServerError> {
        Ok(id.id)
    }

    fn join(
        id: Self::UniqueId,
        _ranks: usize,
        rank: usize,
    ) -> Result<Self::Communicator, ServerError> {
        if Wire::of(id).refused_rank.load(Ordering::Relaxed) == rank {
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
        Ok(())
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
        Wire::of(*comm).post_send();
        Ok(())
    }

    fn recv(
        comm: &Self::Communicator,
        _dst: &DeviceResource<Self>,
        _elements: Elements<Self::DataType>,
        _peer: usize,
        _stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        Wire::of(*comm).post_recv();
        Ok(())
    }
}

std::thread_local! {
    /// The events this thread has taken from the driver, and recorded.
    static EVENTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    /// The device clock this thread's fake streams share, a tick per event recorded.
    static CLOCK: Cell<u64> = const { Cell::new(0) };
    /// The waits this thread has ordered one stream after another with.
    static WAITS: Cell<usize> = const { Cell::new(0) };
}

fn count_wait() {
    WAITS.with(|waits| waits.set(waits.get() + 1));
}

/// The waits across streams `queue` orders.
fn waits_across_streams(queue: impl FnOnce()) -> usize {
    let before = WAITS.with(Cell::get);
    queue();
    WAITS.with(Cell::get) - before
}

/// Sets this thread's device clock to `micros`.
fn clock_at(micros: u64) {
    CLOCK.with(|clock| clock.set(micros));
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
        *event.0.lock().unwrap() = Some(CLOCK.with(|clock| clock.replace(clock.get() + 1)));
        Ok(())
    }

    fn event_wait(_event: &Self::Event) -> Result<(), DriverError> {
        Ok(())
    }

    fn event_query(_event: &Self::Event) -> Result<bool, DriverError> {
        Ok(true)
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
        count_wait();
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
    collectives: Collectives<Fake, FakeEvents>,
    streams: MultiStream<FakeBackend>,
    profiler: EventProfiler<FakeEvents>,
}

impl CollectiveServer for FakeServer {
    type Driver = Fake;
    type Events = FakeEvents;

    fn device_id(&self) -> DeviceId {
        self.device
    }

    fn collectives(&mut self) -> &mut Collectives<Fake, FakeEvents> {
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

impl WriteScoped for FakeServer {
    type Streams = MultiStream<FakeBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.streams
    }

    fn capturing(&mut self, stream: StreamId) -> Option<&mut StreamCapture> {
        self.streams
            .try_stream_mut(&stream)
            .map(DeviceStream::capturing)
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
        command.allocate(SIZE, handle.memory.clone()).unwrap();
        drop(command);
        handle
    }

    /// A buffer whose allocation failed with `reason`.
    fn out_of_memory(&mut self, reason: &str) -> Handle {
        let handle = self.unallocated();
        self.streams
            .fail_unallocated(&handle.memory.to_binding(), error(reason));
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

fn all_reduce(server: &mut FakeServer, device: DeviceId, timing: Option<CommunicationTiming>) {
    all_reduce_on(server, device, timing, STREAM, ComputeWait::AtSync);
}

fn all_reduce_on(
    server: &mut FakeServer,
    device: DeviceId,
    timing: Option<CommunicationTiming>,
    stream: StreamId,
    wait: ComputeWait,
) {
    let (src, dst) = (server.allocated(), server.allocated());
    if let Some(timing) = timing {
        server.time_next_communication(timing);
    }
    let reduction = Reduction {
        dtype: ElemType::Float(FloatKind::F32),
        op: ReduceOperation::Sum,
    };
    server
        .all_reduce(
            src.binding(),
            dst.binding(),
            reduction,
            stream,
            vec![device],
            wait,
        )
        .unwrap();
}

/// A timed sync on `stream`, and the window of its wait.
fn timed_sync(server: &mut FakeServer, stream: StreamId) -> ProfileDuration {
    let mut timing = CommunicationTiming::default();
    let wait = timing.wait_window();
    server.time_next_communication(timing);
    server.sync_collective(stream).unwrap();
    wait
}

fn joined_alone(index: u16) -> (FakeServer, DeviceId) {
    let device = DeviceId::new(0, index);
    let mut server = FakeServer::new(device);
    server.collectives().join(vec![device]).unwrap();
    (server, device)
}

#[test_log::test]
fn a_sync_after_its_collectives_ended_waits_for_nothing() {
    let (mut server, device) = joined_alone(70);

    clock_at(3);
    all_reduce(&mut server, device, Some(CommunicationTiming::default()));
    clock_at(9);
    let wait = timed_sync(&mut server, STREAM);

    assert_eq!(read(wait), cubecl_common::profile::Duration::ZERO);
    let nothing_queued = timed_sync(&mut server, StreamId { value: 2 });
    assert!(cubecl_environment::future::block_on(nothing_queued.resolve()).is_none());
}

#[test_log::test]
fn a_sync_after_an_untimed_collective_reads_back_empty() {
    let (mut server, device) = joined_alone(71);

    all_reduce(&mut server, device, Some(CommunicationTiming::default()));
    all_reduce(&mut server, device, None);
    let wait = timed_sync(&mut server, STREAM);

    assert!(cubecl_environment::future::block_on(wait.resolve()).is_none());
}

#[test_log::test]
fn a_collective_waited_for_at_once_runs_on_its_compute_stream() {
    let (mut server, device) = joined_alone(73);

    let waits = waits_across_streams(|| {
        all_reduce_on(&mut server, device, None, STREAM, ComputeWait::Now);
    });

    assert_eq!(waits, 0);
}

#[test_log::test]
fn a_collective_left_for_the_sync_runs_on_the_collective_stream() {
    let (mut server, device) = joined_alone(74);

    let waits = waits_across_streams(|| {
        all_reduce_on(&mut server, device, None, STREAM, ComputeWait::AtSync);
    });

    assert_eq!(
        waits, 1,
        "the collective stream waits for the compute stream"
    );
}

#[test_log::test]
fn a_collective_after_one_left_for_the_sync_follows_it_on_the_collective_stream() {
    let (mut server, device) = joined_alone(75);
    all_reduce_on(&mut server, device, None, STREAM, ComputeWait::AtSync);

    let follows = waits_across_streams(|| {
        all_reduce_on(&mut server, device, None, STREAM, ComputeWait::Now);
    });
    let then = waits_across_streams(|| {
        all_reduce_on(&mut server, device, None, STREAM, ComputeWait::Now);
    });

    assert_eq!(follows, 2, "both streams wait for each other");
    assert_eq!(
        then, 0,
        "the compute stream already follows every collective"
    );
}

#[test_log::test]
fn a_collective_from_another_compute_stream_waits_for_the_last_one() {
    let (mut server, device) = joined_alone(76);
    all_reduce_on(&mut server, device, None, STREAM, ComputeWait::Now);
    let other = StreamId { value: 2 };

    let waits = waits_across_streams(|| {
        all_reduce_on(&mut server, device, None, other, ComputeWait::Now);
    });

    // The collective stream waits for the other compute stream, then for this one, and this one
    // waits for it.
    assert_eq!(waits, 3);
}

#[test_log::test]
fn a_sync_from_another_compute_stream_waits_for_the_last_collective() {
    let (mut server, device) = joined_alone(79);
    all_reduce_on(&mut server, device, None, STREAM, ComputeWait::Now);

    let waits = waits_across_streams(|| {
        server.sync_collective(StreamId { value: 2 }).unwrap();
    });

    // Only the compute stream that ran the collective: nothing ran on the collective stream.
    assert_eq!(waits, 1);
}

#[test_log::test]
fn a_sync_on_the_stream_that_ran_the_collectives_waits_on_no_other() {
    let (mut server, device) = joined_alone(80);
    all_reduce_on(&mut server, device, None, STREAM, ComputeWait::Now);

    let waits = waits_across_streams(|| server.sync_collective(STREAM).unwrap());

    assert_eq!(waits, 0);
}

#[test_log::test]
fn a_sync_after_a_collective_left_for_it_waits_for_the_collective_stream() {
    let (mut server, device) = joined_alone(81);
    all_reduce_on(&mut server, device, None, STREAM, ComputeWait::AtSync);

    let waits = waits_across_streams(|| server.sync_collective(STREAM).unwrap());

    assert_eq!(waits, 1);
}

#[test_log::test]
fn a_collective_on_its_compute_stream_is_exposed_for_its_whole_part() {
    let (mut server, device) = joined_alone(77);
    let mut timing = CommunicationTiming::default();
    let (part, wait) = (timing.part_window(), timing.wait_window());

    all_reduce_on(&mut server, device, Some(timing), STREAM, ComputeWait::Now);

    let (part, wait) = (read(part), read(wait));
    assert!(part > cubecl_common::profile::Duration::ZERO);
    assert_eq!(wait, part);
}

#[test_log::test]
fn a_sync_after_a_collective_on_its_compute_stream_waits_for_nothing() {
    let (mut server, device) = joined_alone(78);
    all_reduce_on(
        &mut server,
        device,
        Some(CommunicationTiming::default()),
        STREAM,
        ComputeWait::Now,
    );

    let wait = timed_sync(&mut server, STREAM);

    assert_eq!(read(wait), cubecl_common::profile::Duration::ZERO);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "left its timing untaken")]
fn a_timing_left_untaken_fails_the_next_one() {
    let mut collectives = Collectives::<Fake, FakeEvents>::new(DeviceId::new(0, 72));
    collectives.time_next(CommunicationTiming::default());
    collectives.time_next(CommunicationTiming::default());
}

/// Opens a recording window on `stream`, as `begin_capture` does once the work is planned.
fn recording(server: &mut FakeServer, stream: StreamId) {
    let capture = server
        .capturing(stream)
        .expect("the stream should exist before recording");
    capture.prepare(stream).unwrap();
    capture.begin().unwrap();
}

fn reduction() -> Reduction {
    Reduction {
        dtype: ElemType::Float(FloatKind::F32),
        op: ReduceOperation::Sum,
    }
}

#[test_log::test]
fn a_collective_recorded_in_a_graph_is_in_its_write_set() {
    let (mut server, device) = joined_alone(82);
    let (src, dst) = (server.allocated(), server.allocated());
    recording(&mut server, STREAM);

    server
        .all_reduce(
            src.to_binding(),
            dst.to_binding(),
            reduction(),
            STREAM,
            vec![device],
            ComputeWait::Now,
        )
        .unwrap();

    let capture = server.capturing(STREAM).unwrap();
    capture.end(STREAM).unwrap();
    let written: Vec<_> = capture
        .take_recorded()
        .iter()
        .map(|binding| binding.memory.id())
        .collect();
    assert_eq!(written, vec![dst.to_binding().memory.id()]);
}

#[test_log::test]
fn a_collective_refused_in_a_graph_dooms_it() {
    let (mut server, device) = joined_alone(83);
    let src = server.allocated();
    let dst = Handle::new(server.service, STREAM, SIZE / 2);
    let mut command = server.command(STREAM, [].into_iter()).unwrap();
    command.allocate(SIZE / 2, dst.memory.clone()).unwrap();
    drop(command);
    recording(&mut server, STREAM);

    server
        .all_reduce(
            src.binding(),
            dst.binding(),
            reduction(),
            STREAM,
            vec![device],
            ComputeWait::Now,
        )
        .unwrap_err();

    let capture = server.capturing(STREAM).unwrap();
    capture.end(STREAM).unwrap();
    assert!(capture.take_failure().is_some());
}

#[test_log::test]
fn a_recv_recorded_in_a_graph_is_in_its_write_set() {
    let (src, dst) = devices(84);
    let (report, outcome) = SendOutcome::channel();

    let (written, received) = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut source = FakeServer::new(src);
            let sent = source.allocated();
            source.send(descriptor(sent), STREAM, dst, report).unwrap();
        });
        scope
            .spawn(move || {
                let mut destination = FakeServer::new(dst);
                let _stream = destination.allocated();
                let received = destination.unallocated();
                recording(&mut destination, STREAM);
                destination
                    .recv(received.clone(), STREAM, src, outcome)
                    .unwrap();
                let capture = destination.capturing(STREAM).unwrap();
                capture.end(STREAM).unwrap();
                let written: Vec<_> = capture
                    .take_recorded()
                    .iter()
                    .map(|binding| binding.memory.id())
                    .collect();
                (written, received.to_binding().memory.id())
            })
            .join()
            .unwrap()
    });

    assert_eq!(written, vec![received]);
}

#[test_log::test]
fn a_recv_refused_in_a_graph_dooms_it() {
    let (src, dst) = devices(86);
    let (mut source, mut destination) = (FakeServer::new(src), FakeServer::new(dst));
    let refused = source.out_of_memory("out of device memory");
    let _stream = destination.allocated();
    let received = destination.unallocated();
    let (report, outcome) = SendOutcome::channel();
    source
        .send(descriptor(refused), STREAM, dst, report)
        .unwrap_err();
    recording(&mut destination, STREAM);

    destination
        .recv(received, STREAM, src, outcome)
        .unwrap_err();

    let capture = destination.capturing(STREAM).unwrap();
    capture.end(STREAM).unwrap();
    assert!(capture.take_failure().is_some());
}
