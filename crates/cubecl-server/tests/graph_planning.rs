//! A graph capture's planning runs and recording, on a fake driver whose device memory is host
//! bytes: where the work's allocations land, and who may end a planning run.

use cubecl_common::{
    bytes::Bytes,
    device::{DeviceId, ServiceId},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::{
    command::{
        Command, CopyLayout, DeviceResource, DeviceStream, Driver, Graph, GraphDriver,
        HostResource, Window,
    },
    id::{GraphId, KernelId},
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, FailureId, ManagedMemoryBinding, MemoryConfiguration, MemoryLocation,
        MemoryManagement, MemoryManagementOptions,
        drop_queue::{Fence, PendingDropQueue},
    },
    metadata_cache::{MetadataCachePolicy, MetadataInfoCache},
    server::{BufferBinding, DeviceCaptures, Handle, IoError, LaunchError, ServerError},
    storage::{BytesStorage, ComputeStorage, StorageHandle, StorageId},
    stream::{EventStreamBackend, MultiStream, StreamCapture, StreamMemory},
};
use std::cell::Cell;
use std::fmt;
use std::sync::Arc;

const MIB: u64 = 1024 * 1024;
/// Backend streams the fake pools: logical streams `n` and `n + STREAMS` share one.
const STREAMS: u8 = 4;
const OWNER: StreamId = StreamId { value: 1 };
const NEIGHBOUR: StreamId = StreamId {
    value: 1 + STREAMS as u64,
};

std::thread_local! {
    /// How many of this thread's next device allocations the storage refuses, as a full device
    /// would.
    static REFUSALS: Cell<usize> = const { Cell::new(0) };
}

/// Have the device refuse this thread's next allocation of a page.
fn refuse_next_page() {
    REFUSALS.with(|refusals| refusals.set(1));
}

/// Host bytes standing in for device memory, out of memory whenever [`refuse_next_page`] says.
#[derive(Default)]
struct FlakyStorage {
    bytes: BytesStorage,
}

impl ComputeStorage for FlakyStorage {
    type Resource = <BytesStorage as ComputeStorage>::Resource;

    fn alignment(&self) -> usize {
        self.bytes.alignment()
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        self.bytes.get(handle)
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let refused = REFUSALS.with(|refusals| refusals.replace(refusals.get().saturating_sub(1)));
        if refused > 0 {
            return Err(IoError::OutOfMemory {
                size,
                backtrace: BackTrace::capture(),
            });
        }
        self.bytes.alloc(size)
    }

    fn dealloc(&mut self, id: StorageId) {
        self.bytes.dealloc(id)
    }

    fn flush(&mut self) {
        self.bytes.flush()
    }

    fn bytes_allocated(&self) -> u64 {
        self.bytes.bytes_allocated()
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
    device_memory: MemoryManagement<FlakyStorage>,
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
    type DeviceStorage = FlakyStorage;
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

fn memory<S: ComputeStorage + Default>(name: &'static str) -> MemoryManagement<S> {
    MemoryManagement::from_configuration(
        S::default(),
        &MemoryDeviceProperties::new(128 * MIB, 32),
        MemoryConfiguration::Adaptive,
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
        unreachable!("the work stages nothing")
    }

    unsafe fn copy_to_host(
        _resource: &DeviceResource<Self>,
        _layout: &CopyLayout<'_>,
        _bytes: &mut Bytes,
        _stream: &Self::Stream,
    ) -> Result<(), IoError> {
        unreachable!("the work reads nothing")
    }

    unsafe fn copy_to_device(
        _resource: &DeviceResource<Self>,
        _layout: &CopyLayout<'_>,
        _data: &[u8],
        _stream: &Self::Stream,
    ) -> Result<(), IoError> {
        unreachable!("the work writes nothing from the host")
    }

    unsafe fn copy_on_device(
        _source: &DeviceResource<Self>,
        _target: &DeviceResource<Self>,
        _queue: (),
    ) -> Result<(), IoError> {
        unreachable!("nothing is relocated")
    }

    fn launch(
        _ctx: &mut Self::Context,
        _stream: &mut Self::Stream,
        _id: &KernelId,
        _kernel: &Self::Loaded,
        _count: (u32, u32, u32),
        _args: &mut Self::LaunchArgs,
    ) -> Result<(), LaunchError> {
        unreachable!("the work launches nothing")
    }
}

impl GraphDriver for Fake {
    type Executable = ();

    fn begin(_stream: &mut Self::Stream) -> Result<(), ServerError> {
        Ok(())
    }

    fn instantiate(
        _stream: &mut Self::Stream,
        doomed: Option<ServerError>,
    ) -> Result<Self::Executable, ServerError> {
        doomed.map_or(Ok(()), Err)
    }

    fn upload(_exec: &Self::Executable, _stream: &mut Self::Stream) {}

    fn replay(_exec: &Self::Executable, _stream: &mut Self::Stream) -> Result<(), ServerError> {
        Ok(())
    }
}

struct FakeServer {
    ctx: (),
    service: ServiceId,
    streams: MultiStream<FakeBackend>,
}

impl FakeServer {
    fn new() -> Self {
        Self {
            ctx: (),
            service: ServiceId::of::<Self>(DeviceId::new(0, 0)),
            streams: MultiStream::new(Arc::new(ServerLogger::default()), FakeBackend, STREAMS),
        }
    }

    fn command(&mut self, stream: StreamId) -> Command<'_, Fake> {
        let streams = self.streams.resolve(stream, [].into_iter()).unwrap();
        Command::new(&mut self.ctx, streams, self.service)
    }

    fn prepare(&mut self, stream: StreamId) {
        Window::on(&mut self.command(stream))
            .prepare(stream)
            .unwrap();
    }

    fn end_run(&mut self, stream: StreamId) -> Result<bool, ServerError> {
        Window::on(&mut self.command(stream)).end_run(stream)
    }

    fn begin(&mut self, stream: StreamId) {
        Window::on(&mut self.command(stream)).begin().unwrap();
    }

    fn seal(&mut self, stream: StreamId) -> Result<Graph<Fake>, ServerError> {
        Window::on(&mut self.command(stream))
            .instantiate(stream, GraphId::new())
            .map_err(|refused| refused.error)
    }

    /// The work every capture here records: one buffer of `MIB` bytes, which is its output.
    fn work(&mut self, stream: StreamId) -> Result<Handle, IoError> {
        self.command(stream).empty(MIB)
    }

    /// Plan [`work`](Self::work) until two runs in a row allocate alike, each run's output held
    /// until the run ends.
    fn plan(&mut self, stream: StreamId) {
        while !{
            let output = self.work(stream).unwrap();
            let stable = self.end_run(stream).unwrap();
            drop(output);
            stable
        } {}
    }
}

fn location(handle: &Handle) -> MemoryLocation {
    handle.memory.descriptor().location()
}

/// A recorded run that lets go of what its planning found to be an output still writes it on
/// every replay: the graph holds that room, so another graph's output is never placed where
/// this graph's replay writes.
#[test]
fn an_output_the_recording_let_go_stays_its_graphs() {
    let mut server = FakeServer::new();
    server.prepare(OWNER);
    server.plan(OWNER);
    server.begin(OWNER);
    let let_go = server.work(OWNER).unwrap();
    let written = location(&let_go);
    drop(let_go);
    let _first = server
        .seal(OWNER)
        .expect("an output let go is no divergence");

    server.prepare(OWNER);
    server.plan(OWNER);
    server.begin(OWNER);
    let kept = server.work(OWNER).unwrap();
    let _second = server.seal(OWNER).unwrap();
    assert_ne!(
        location(&kept),
        written,
        "the second graph's output sits where the first graph's replay writes"
    );
}

/// A planning run's output that the device could not place at once is placed where the plan puts
/// it once memory is reclaimed: on the room apart from the shared one, which the recording then
/// finds.
#[test]
fn an_output_placed_after_a_reclaim_is_where_the_recording_finds_it() {
    let mut server = FakeServer::new();
    server.prepare(OWNER);
    let first = server.work(OWNER).unwrap();
    assert!(!server.end_run(OWNER).unwrap(), "a first run");
    drop(first);

    refuse_next_page();
    let second = server.work(OWNER).expect("placed after a reclaim");
    assert!(server.end_run(OWNER).unwrap(), "two runs allocated alike");
    drop(second);

    server.begin(OWNER);
    let recorded = server.work(OWNER);
    let graph = server.seal(OWNER);
    assert!(
        recorded.is_ok(),
        "the recorded output found no room: {:?}",
        recorded.err()
    );
    graph.expect("the recording seals");
}

/// Another logical stream folded onto the same backend stream cannot end the owner's planning
/// run: its log stays the owner's, and the owner's run ends as it would have.
#[test]
fn only_the_owner_ends_a_planning_run() {
    let mut server = FakeServer::new();
    server.prepare(OWNER);
    let output = server.work(OWNER).unwrap();
    assert!(
        server.end_run(NEIGHBOUR).is_err(),
        "a neighbour ended the owner's planning run"
    );
    assert!(!server.end_run(OWNER).unwrap(), "the owner's first run");
    drop(output);
}
