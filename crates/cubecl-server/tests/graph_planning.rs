//! A graph capture's planning runs and recording, on a fake driver whose device memory is host
//! bytes: where the work's allocations land, and who may end a planning run.

mod fake;

use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_environment::stream::StreamId;
use cubecl_server::{
    command::{Command, Graph, Window},
    id::GraphId,
    logging::ServerLogger,
    memory_management::MemoryLocation,
    server::{Handle, IoError, ServerError},
    stream::MultiStream,
};
use fake::{Fake, FakeBackend, refusals_left, refuse_next_page};
use std::sync::Arc;

const MIB: u64 = 1024 * 1024;
/// Backend streams the fake pools: logical streams `n` and `n + STREAMS` share one.
const STREAMS: u8 = 4;
const OWNER: StreamId = StreamId { value: 1 };
const NEIGHBOUR: StreamId = StreamId {
    value: 1 + STREAMS as u64,
};

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

/// Record [`FakeServer::work`] as a graph on `stream`: the graph, and the output its recorded
/// run allocated.
fn record(server: &mut FakeServer, stream: StreamId) -> (Graph<Fake>, Handle) {
    server.prepare(stream);
    server.plan(stream);
    server.begin(stream);
    let output = server.work(stream).unwrap();
    (server.seal(stream).unwrap(), output)
}

/// A recorded run that lets go of what its planning found to be an output still writes it on
/// every replay: the graph holds that room, so another graph's output is never placed where
/// this graph's replay writes. Once the graph is gone, the room serves the next output again.
#[test]
fn an_output_the_recording_let_go_stays_its_graphs() {
    let mut server = FakeServer::new();
    let (first, let_go) = record(&mut server, OWNER);
    let written = location(&let_go);
    drop(let_go);

    let (second, kept) = record(&mut server, OWNER);
    assert_ne!(
        location(&kept),
        written,
        "the second graph's output sits where the first graph's replay writes"
    );

    drop(first);
    let (_third, next) = record(&mut server, OWNER);
    assert_eq!(
        location(&next),
        written,
        "the room the first graph held serves the next output once it is gone"
    );
    drop((second, kept));
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
    assert_eq!(
        refusals_left(),
        0,
        "the output took a page of its own, which the device refused"
    );
    let planned = location(&second);
    assert!(server.end_run(OWNER).unwrap(), "two runs allocated alike");
    drop(second);

    server.begin(OWNER);
    let recorded = server.work(OWNER);
    let graph = server.seal(OWNER);
    let recorded = recorded.expect("the recorded output finds the room the planning run grew");
    assert_eq!(
        location(&recorded),
        planned,
        "the recorded output lands where the last planning run's did"
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
