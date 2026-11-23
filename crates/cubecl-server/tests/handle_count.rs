//! A buffer's handle count, which `can_mut` reads, is its users' handles plus one while the
//! device thread sets the buffer up: the allocation task's handle moves into the pool, and no
//! clone of it shows on the way. Every backend allocates through
//! [`Command::initialize_memory`], so this checks them all on a fake driver.

mod fake;

use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_environment::stream::StreamId;
use cubecl_server::{command::Command, logging::ServerLogger, server::Handle, stream::MultiStream};
use fake::{Fake, FakeBackend};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

const STREAM: StreamId = StreamId { value: 1 };
const SIZE: u64 = 4096;
/// Allocations watched. A clone taken on the way is out for only part of each one, so the
/// sampler needs many of them to land in that window.
const ROUNDS: usize = 20_000;

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
            streams: MultiStream::new(Arc::new(ServerLogger::default()), FakeBackend, 4),
        }
    }

    fn command(&mut self) -> Command<'_, Fake> {
        let streams = self.streams.resolve(STREAM, [].into_iter()).unwrap();
        Command::new(&mut self.ctx, streams, self.service)
    }
}

#[test]
fn an_allocation_never_counts_a_handle_of_the_device_thread() {
    let mut server = FakeServer::new();
    let handles: Vec<Handle> = (0..ROUNDS)
        .map(|_| Handle::new(server.service, STREAM, SIZE))
        .collect();
    let watched = &handles;
    let (current, done, above) = (
        std::sync::atomic::AtomicUsize::new(0),
        AtomicBool::new(false),
        AtomicBool::new(false),
    );

    thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(Ordering::Acquire) {
                let handle = &watched[current.load(Ordering::Acquire)];
                if !handle.can_mut() {
                    above.store(true, Ordering::Release);
                }
            }
        });
        for (round, handle) in handles.iter().enumerate() {
            current.store(round, Ordering::Release);
            // The task's handle, which the command takes by value as every backend passes it.
            server
                .command()
                .initialize_memory(handle.memory.clone(), SIZE)
                .unwrap();
        }
        done.store(true, Ordering::Release);
    });

    assert!(
        !above.load(Ordering::Acquire),
        "the device thread held a handle of its own while it set a buffer up"
    );
    assert!(
        handles.iter().all(Handle::can_mut),
        "one handle besides the user's: the pool's"
    );
}
