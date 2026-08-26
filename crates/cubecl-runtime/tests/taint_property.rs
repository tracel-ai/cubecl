//! The one oracle this area answers to, checked over random interleavings.
//!
//! > A read returns bytes if and only if every byte it returns was last
//! > written by work that succeeded.
//!
//! The harness drives the same machinery every backend server drives — a
//! [`StreamPool`] of streams each owning a [`MemoryManagement`], the
//! device-wide [`ErrorGraph`], and the taint calls a launch, a host write and
//! a read make — over random sequences of allocate, launch, write, read,
//! flush, free and cleanup across several logical streams. The model keeps
//! its own answer per buffer and compares after every read.
//!
//! Two invariants ride along, checked continuously:
//!
//! - the graph never holds more failures than there are stale live buffers,
//!   which is the bound the design rests on: the graph leaks if and only if
//!   the program leaks memory;
//! - once every buffer is dropped and the pools are swept, the graph is
//!   empty.

use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{
    ErrorGraph, FailureId, ManagedMemoryBinding, MemoryConfiguration, MemoryManagement,
    MemoryManagementOptions,
};
use cubecl_runtime::server::{BufferBinding, Handle, ServerError};
use cubecl_runtime::storage::BytesStorage;
use cubecl_runtime::stream::{
    StreamErrorSink, StreamErrors, StreamFactory, StreamMemory, StreamPool,
};
use std::sync::Arc;

const MAX_STREAMS: u8 = 4;
const OPS_PER_RUN: usize = 300;
const SEEDS: u64 = 40;

/// What the model believes about a live buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// The last work given this buffer succeeded (or nothing wrote it yet,
    /// which reads as trustworthy: fresh memory carries no failure).
    Trusted,
    /// The last work that was going to write this buffer failed, and nothing
    /// has written it since.
    Stale,
}

struct Buffer {
    /// Owns the allocation; dropping it frees the slice.
    _handle: Handle,
    binding: BufferBinding,
    expect: Expect,
}

struct TestStream {
    memory: MemoryManagement<BytesStorage>,
    errors: StreamErrors,
}

impl core::fmt::Debug for TestStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TestStream").finish()
    }
}

impl StreamErrorSink for TestStream {
    fn errors(&self) -> impl core::ops::Deref<Target = StreamErrors> + '_ {
        &self.errors
    }
}

impl StreamMemory for TestStream {
    fn failure(&self, binding: &ManagedMemoryBinding) -> Option<FailureId> {
        self.memory.failure(binding)
    }

    fn taint(
        &mut self,
        binding: &ManagedMemoryBinding,
        failure: FailureId,
        failures: &mut ErrorGraph,
    ) {
        self.memory.taint(binding, failure, failures)
    }

    fn written(&mut self, binding: &ManagedMemoryBinding, failures: &mut ErrorGraph) {
        self.memory.written(binding, failures)
    }
}

struct Factory {
    config: MemoryConfiguration,
    properties: MemoryDeviceProperties,
    logger: Arc<ServerLogger>,
}

impl StreamFactory for Factory {
    type Stream = TestStream;

    fn create(&mut self) -> Self::Stream {
        TestStream {
            memory: MemoryManagement::from_configuration(
                BytesStorage::default(),
                &self.properties,
                self.config.clone(),
                self.logger.clone(),
                MemoryManagementOptions::new("property harness"),
            ),
            errors: StreamErrors::default(),
        }
    }
}

/// A tiny deterministic generator, so a failing seed reproduces exactly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound.max(1) as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

struct Harness {
    pool: StreamPool<Factory>,
    failures: ErrorGraph,
    buffers: Vec<Buffer>,
    rng: Rng,
}

fn error(reason: &str) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: Default::default(),
    }
}

impl Harness {
    fn new(config: MemoryConfiguration, seed: u64) -> Self {
        let properties = MemoryDeviceProperties {
            max_page_size: 128 * 1024,
            alignment: 32,
        };
        Self {
            pool: StreamPool::new(
                Factory {
                    config,
                    properties,
                    logger: Arc::new(ServerLogger::default()),
                },
                MAX_STREAMS,
                0,
            ),
            failures: ErrorGraph::default(),
            buffers: Vec::new(),
            rng: Rng(seed),
        }
    }

    fn stream_id(&mut self) -> StreamId {
        StreamId {
            value: self.rng.next() % (MAX_STREAMS as u64 * 2),
        }
    }

    /// Allocate on a random stream: reserve, then bind, as every server does.
    fn alloc(&mut self) {
        let id = self.stream_id();
        let size = 32 * (1 + self.rng.below(64)) as u64;
        let handle = Handle::new(id, size);

        let stream = self.pool.get_mut(&id);
        let reserved = match stream.memory.reserve(size, &mut self.failures) {
            Ok(reserved) => reserved,
            Err(err) => panic!("the harness never outgrows its pools: {err}"),
        };
        stream
            .memory
            .bind(reserved, handle.memory.clone(), 0, &mut self.failures)
            .unwrap();

        self.buffers.push(Buffer {
            binding: handle.clone().binding(),
            _handle: handle,
            expect: Expect::Trusted,
        });
    }

    /// A launch on a random stream, writing a random subset of live buffers,
    /// succeeding or failing. Exactly what a server's launch path does with
    /// its write set: taint it on failure, release it on success.
    fn launch(&mut self, fail: bool) {
        if self.buffers.is_empty() {
            return;
        }
        let id = self.stream_id();
        let count = 1 + self.rng.below(3.min(self.buffers.len()));
        let mut written = Vec::new();
        for _ in 0..count {
            let index = self.rng.below(self.buffers.len());
            if !written.contains(&index) {
                written.push(index);
            }
        }

        if fail {
            let bindings: Vec<&BufferBinding> =
                written.iter().map(|i| &self.buffers[*i].binding).collect();
            self.pool
                .taint(error("launch"), bindings.into_iter(), &mut self.failures);
            let stream = self.pool.get_mut(&id);
            stream.errors.push(id, error("launch"));
            for index in written {
                self.buffers[index].expect = Expect::Stale;
            }
        } else {
            let bindings: Vec<&BufferBinding> =
                written.iter().map(|i| &self.buffers[*i].binding).collect();
            self.pool.written(bindings.into_iter(), &mut self.failures);
            for index in written {
                self.buffers[index].expect = Expect::Trusted;
            }
        }
    }

    /// A host write: fills one buffer, whatever state it was in.
    fn host_write(&mut self) {
        if self.buffers.is_empty() {
            return;
        }
        let index = self.rng.below(self.buffers.len());
        let binding = self.buffers[index].binding.clone();
        self.pool
            .written([&binding].into_iter(), &mut self.failures);
        self.buffers[index].expect = Expect::Trusted;
    }

    /// The oracle: a read fails if and only if the model says the bytes are
    /// stale.
    fn read(&mut self) {
        if self.buffers.is_empty() {
            return;
        }
        let index = self.rng.below(self.buffers.len());
        let buffer = &self.buffers[index];
        let result = self
            .pool
            .ensure_written(&self.failures, [&buffer.binding].into_iter());

        match buffer.expect {
            Expect::Trusted => assert!(
                result.is_ok(),
                "a buffer whose last writer succeeded must read: {result:?}"
            ),
            Expect::Stale => assert!(
                result.is_err(),
                "a buffer whose last writer failed must not read"
            ),
        }
    }

    /// A flush drains what the stream is owed and changes nothing about any
    /// buffer: reported is not written.
    fn flush(&mut self) {
        let id = self.stream_id();
        let stream = self.pool.get_mut(&id);
        let _ = stream.errors.take(id);
    }

    /// Drop a live buffer. Its slice frees and later sheds any taint through
    /// bind, coalesce, tombstone or sweep — never through anything the model
    /// has to do here.
    fn free(&mut self) {
        if self.buffers.is_empty() {
            return;
        }
        let index = self.rng.below(self.buffers.len());
        self.buffers.swap_remove(index);
    }

    fn cleanup(&mut self) {
        let id = self.stream_id();
        let explicit = self.rng.chance(50);
        let stream = self.pool.get_mut(&id);
        stream.memory.cleanup(explicit, &mut self.failures);
    }

    fn sweep(&mut self) {
        for value in 0..MAX_STREAMS as u64 {
            let id = StreamId { value };
            let stream = self.pool.get_mut(&id);
            stream.memory.cleanup(true, &mut self.failures);
        }
    }
}

fn run(config: MemoryConfiguration, seed: u64) {
    let mut harness = Harness::new(config, seed);

    for _ in 0..OPS_PER_RUN {
        match harness.rng.below(100) {
            0..=19 => harness.alloc(),
            20..=39 => harness.launch(true),
            40..=59 => harness.launch(false),
            60..=69 => harness.host_write(),
            70..=84 => harness.read(),
            85..=89 => harness.flush(),
            90..=95 => harness.free(),
            _ => harness.cleanup(),
        }
    }

    // Read every live buffer once more: the oracle at rest.
    for index in 0..harness.buffers.len() {
        let buffer = &harness.buffers[index];
        let result = harness
            .pool
            .ensure_written(&harness.failures, [&buffer.binding].into_iter());
        match buffer.expect {
            Expect::Trusted => assert!(result.is_ok(), "trusted buffer failed at rest: {result:?}"),
            Expect::Stale => assert!(result.is_err(), "stale buffer read clean at rest"),
        }
    }

    // Drop everything and sweep: with no live buffer and no free slice left
    // carrying a failure, the graph must be empty. This is the design's
    // retention bound — the graph leaks if and only if the program leaks
    // memory — checked mechanically.
    harness.buffers.clear();
    harness.sweep();
    assert!(
        harness.failures.is_empty(),
        "the graph held {} failure(s) after every buffer was dropped and every pool swept",
        harness.failures.len()
    );
}

#[test]
fn a_read_returns_bytes_iff_their_last_writer_succeeded_subslices() {
    #[cfg(not(exclusive_memory_only))]
    for seed in 0..SEEDS {
        run(MemoryConfiguration::SubSlices, seed);
    }
}

#[test]
fn a_read_returns_bytes_iff_their_last_writer_succeeded_exclusive_pages() {
    for seed in 0..SEEDS {
        run(MemoryConfiguration::ExclusivePages, seed);
    }
}
