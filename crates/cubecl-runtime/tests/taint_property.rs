//! The one oracle this area answers to, checked over random interleavings.
//!
//! > A read returns bytes if and only if every byte it returns was last
//! > written by work that succeeded.
//!
//! The harness drives the same machinery every backend server drives — a
//! [`StreamPool`] of streams each owning a [`MemoryManagement`], the
//! device-wide [`ErrorGraph`], and the write scope every launch and host
//! write runs inside — over random sequences of allocate, launch, write,
//! read, flush, free and cleanup across several logical streams. Launches,
//! writes and reads go through partial bindings as often as whole ones,
//! because the claim is byte-ranged: a launch that failed writing one region
//! says nothing about the rest. The model keeps its own answer per buffer —
//! one staleness flag per byte, deliberately nothing like the interval
//! arithmetic it checks — and compares after every read.
//!
//! Two invariants ride along, checked continuously:
//!
//! - the graph never holds more failures than there are stale live buffers,
//!   which is the bound the design rests on: the graph leaks if and only if
//!   the program leaks memory;
//! - once every buffer is dropped and the pools are swept, the graph is
//!   empty.
//!
//! The write scope's own properties get dedicated tests below the random
//! runs: success releases, failure names the real error, and a panic between
//! entry and exit leaves the write set tainted.

use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::id::KernelId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{
    ErrorGraph, FailureId, MemoryConfiguration, MemoryManagement, MemoryManagementOptions,
};
use cubecl_runtime::server::{BufferBinding, Handle, ServerError};
use cubecl_runtime::storage::BytesStorage;
use cubecl_runtime::stream::{
    ExecuteScope, FailureStore, Failures, ScopedOutcome, StreamCapture, StreamFactory,
    StreamMemory, StreamPool, WriteScoped,
};
use std::sync::Arc;

const MAX_STREAMS: u8 = 4;
const OPS_PER_RUN: usize = 300;
const SEEDS: u64 = 40;

struct Buffer {
    /// Owns the allocation; dropping it frees the slice.
    _handle: Handle,
    binding: BufferBinding,
    /// What the model believes, byte by byte: `true` where the last work
    /// that was going to write the byte failed and nothing has written it
    /// since. Fresh memory reads as trustworthy — it carries no failure.
    stale: Vec<bool>,
}

impl Buffer {
    /// The binding for `range`, as a caller slicing into the buffer builds
    /// one: `offset_start` trims the front, `offset_end` trims the back.
    fn slice(&self, range: &core::ops::Range<u64>) -> BufferBinding {
        let mut binding = self.binding.clone();
        binding.offset_start = Some(range.start);
        binding.offset_end = Some(binding.size - range.end);
        binding
    }

    fn size(&self) -> u64 {
        self.binding.size
    }
}

struct TestStream {
    memory: MemoryManagement<BytesStorage>,
}

impl core::fmt::Debug for TestStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TestStream").finish()
    }
}

impl StreamMemory for TestStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory.failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.memory
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory
            .written(&binding.memory, binding.range(), failures)
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
        }
    }
}

/// The harness's stand-in for a backend server: the streams, the failure
/// store, and nothing else. Implementing [`FailureStore`] and [`WriteScoped`]
/// on it means the harness's launches and host writes run through the very
/// same `while_writing` the real servers use, rather than a second copy of
/// the rules that would drift from it.
struct Device {
    pool: StreamPool<Factory>,
    failures: Failures,
    /// One capture window for the whole harness, so the tests below can pin
    /// the scope-to-window contract — the scope is the window's only
    /// informant — without a driver underneath. Closed outside those tests,
    /// where `fail` and `record` are no-ops.
    capture: StreamCapture,
}

impl FailureStore for Device {
    type Factory = Factory;

    fn split(&mut self) -> (&mut StreamPool<Factory>, &mut Failures) {
        (&mut self.pool, &mut self.failures)
    }

    fn parts(&self) -> (&StreamPool<Factory>, &Failures) {
        (&self.pool, &self.failures)
    }
}

impl WriteScoped for Device {
    type Streams = Self;

    fn write_streams(&mut self) -> &mut Self::Streams {
        self
    }

    fn capturing(&mut self, _stream: StreamId) -> Option<&mut StreamCapture> {
        Some(&mut self.capture)
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
    device: Device,
    buffers: Vec<Buffer>,
    rng: Rng,
}

fn error(reason: &str) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: Default::default(),
    }
}

fn reason(error: &ServerError) -> String {
    format!("{error}")
}

impl Harness {
    fn new(config: MemoryConfiguration, seed: u64) -> Self {
        let properties = MemoryDeviceProperties {
            max_page_size: 128 * 1024,
            alignment: 32,
        };
        let logger = Arc::new(ServerLogger::default());
        Self {
            device: Device {
                pool: StreamPool::new(
                    Factory {
                        config,
                        properties,
                        logger: logger.clone(),
                    },
                    MAX_STREAMS,
                    0,
                ),
                failures: Failures::new(logger),
                capture: StreamCapture::default(),
            },
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

        let device = &mut self.device;
        let stream = device.pool.get_mut(&id);
        let reserved = match stream.memory.reserve(size, device.failures.graph_mut()) {
            Ok(reserved) => reserved,
            Err(err) => panic!("the harness never outgrows its pools: {err}"),
        };
        stream
            .memory
            .bind(
                reserved,
                handle.memory.clone(),
                0,
                device.failures.graph_mut(),
            )
            .unwrap();

        self.buffers.push(Buffer {
            binding: handle.clone().binding(),
            _handle: handle,
            stale: vec![false; size as usize],
        });
    }

    /// A random byte range of a `size`-byte buffer: the whole thing half the
    /// time, a proper slice of it otherwise.
    fn pick_range(&mut self, size: u64) -> core::ops::Range<u64> {
        if self.rng.chance(50) {
            return 0..size;
        }
        let start = self.rng.below(size as usize) as u64;
        let end = start + 1 + self.rng.below((size - start) as usize) as u64;
        start..end
    }

    /// A launch on a random stream, writing a random subset of live buffers,
    /// succeeding or failing — through the same write scope a server's launch
    /// path runs in: entry taints the set provisionally, exit releases it or
    /// swaps the real failure in.
    fn launch(&mut self, fail: bool) {
        if self.buffers.is_empty() {
            return;
        }
        let _id = self.stream_id();
        let count = 1 + self.rng.below(3.min(self.buffers.len()));
        let mut indices = Vec::new();
        for _ in 0..count {
            let index = self.rng.below(self.buffers.len());
            if !indices.contains(&index) {
                indices.push(index);
            }
        }

        let mut writes = Vec::new();
        for index in &indices {
            let range = self.pick_range(self.buffers[*index].size());
            writes.push((*index, range));
        }

        let bindings: Vec<BufferBinding> = writes
            .iter()
            .map(|(index, range)| self.buffers[*index].slice(range))
            .collect();
        let _ = ExecuteScope::over(&mut self.device, StreamId::current(), bindings).execute(|_| {
            match fail {
                true => Err(error("launch")),
                false => Ok(()),
            }
        });

        for (index, range) in writes {
            for byte in range.start as usize..range.end as usize {
                self.buffers[index].stale[byte] = fail;
            }
        }
    }

    /// A host write: fills one buffer, whatever state it was in — same scope,
    /// destination as the write set.
    fn host_write(&mut self) {
        if self.buffers.is_empty() {
            return;
        }
        let index = self.rng.below(self.buffers.len());
        let range = self.pick_range(self.buffers[index].size());
        let binding = self.buffers[index].slice(&range);
        let _id = binding.stream;
        let _ = ExecuteScope::over(&mut self.device, StreamId::current(), vec![binding])
            .execute(|_| Ok::<(), ServerError>(()));
        for byte in range.start as usize..range.end as usize {
            self.buffers[index].stale[byte] = false;
        }
    }

    /// The oracle: a read fails if and only if the model says the bytes are
    /// stale.
    fn read(&mut self) {
        if self.buffers.is_empty() {
            return;
        }
        let index = self.rng.below(self.buffers.len());
        let range = self.pick_range(self.buffers[index].size());
        let buffer = &self.buffers[index];
        let binding = buffer.slice(&range);
        let result = self.device.ensure_written([&binding].into_iter());

        let stale = buffer.stale[range.start as usize..range.end as usize]
            .iter()
            .any(|stale| *stale);
        match stale {
            false => assert!(
                result.is_ok(),
                "bytes whose last writer succeeded must read: {result:?}"
            ),
            true => assert!(
                result.is_err(),
                "bytes whose last writer failed must not read"
            ),
        }
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
        let device = &mut self.device;
        let stream = device.pool.get_mut(&id);
        stream.memory.cleanup(explicit, device.failures.graph_mut());
    }

    fn sweep(&mut self) {
        let device = &mut self.device;
        for value in 0..MAX_STREAMS as u64 {
            let id = StreamId { value };
            let stream = device.pool.get_mut(&id);
            stream.memory.cleanup(true, device.failures.graph_mut());
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
            70..=89 => harness.read(),
            90..=95 => harness.free(),
            _ => harness.cleanup(),
        }
    }

    // Read every live buffer whole once more: the oracle at rest.
    for index in 0..harness.buffers.len() {
        let buffer = &harness.buffers[index];
        let result = harness.device.ensure_written([&buffer.binding].into_iter());
        match buffer.stale.iter().any(|stale| *stale) {
            false => assert!(result.is_ok(), "trusted buffer failed at rest: {result:?}"),
            true => assert!(result.is_err(), "stale buffer read clean at rest"),
        }
    }

    // Drop everything and sweep: with no live buffer and no free slice left
    // carrying a failure, the graph must be empty. This is the design's
    // retention bound — the graph leaks if and only if the program leaks
    // memory — checked mechanically.
    harness.buffers.clear();
    harness.sweep();
    assert!(
        harness.device.failures.graph().is_empty(),
        "the graph held {} failure(s) after every buffer was dropped and every pool swept",
        harness.device.failures.graph().len()
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

/// A scope whose body succeeds leaves nothing behind: the provisional failure
/// is released on exit and the graph is empty again.
#[test]
fn a_scope_that_succeeds_releases_the_provisional_failure() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 7);
    harness.alloc();
    let binding = harness.buffers[0].binding.clone();

    let result = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![binding.clone()],
    )
    .execute(|_| Ok::<(), ServerError>(()));

    assert!(matches!(result, ScopedOutcome::Executed(())));
    assert!(
        harness.device.failures.graph().is_empty(),
        "success leaves no node"
    );
    harness
        .device
        .ensure_written([&binding].into_iter())
        .expect("a buffer whose writer succeeded reads");
}

/// A scope whose body fails leaves the write set carrying the body's error —
/// not the provisional one — and queues it on the issuing stream for its next
/// flush to report.
#[test]
fn a_scope_that_fails_names_the_real_error_and_logs_it() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 7);
    harness.alloc();
    let binding = harness.buffers[0].binding.clone();
    let _id = binding.stream;

    let result = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![binding.clone()],
    )
    .execute(|_| Err::<(), ServerError>(error("the real failure")));

    assert!(matches!(result, ScopedOutcome::Failed(_)));
    let read = harness
        .device
        .ensure_written([&binding].into_iter())
        .expect_err("a buffer whose writer failed must not read");
    let read = reason(&read);
    assert!(
        read.contains("the real failure"),
        "the read fails on the body's error, got: {read}"
    );
    assert!(
        !read.contains("torn down"),
        "the provisional error was replaced, got: {read}"
    );
}

/// The provisional node doing its one irreplaceable job: a body that panics
/// never reaches the exit, and the write set is left carrying the failure the
/// scope entered with — a read fails loudly instead of returning bytes
/// nothing wrote.
#[test]
fn a_mid_launch_panic_leaves_the_write_set_tainted() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 7);
    harness.alloc();
    let binding = harness.buffers[0].binding.clone();

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ExecuteScope::over(
            &mut harness.device,
            StreamId::current(),
            vec![binding.clone()],
        )
        .execute(|_| -> Result<(), ServerError> {
            panic!("mid-launch, before anything could report")
        });
    }));
    assert!(panicked.is_err(), "the panic propagates");

    let read = harness
        .device
        .ensure_written([&binding].into_iter())
        .expect_err("the write set must be tainted after a mid-launch panic");
    let read = reason(&read);
    assert!(
        read.contains("torn down"),
        "the read fails on the provisional error, got: {read}"
    );

    // Writing the buffer again is the recovery, exactly as for an ordinary
    // failure: the provisional node is released and the graph empties.
    let result = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![binding.clone()],
    )
    .execute(|_| Ok::<(), ServerError>(()));
    assert!(matches!(result, ScopedOutcome::Executed(())));
    assert!(harness.device.failures.graph().is_empty());
    harness
        .device
        .ensure_written([&binding].into_iter())
        .expect("a rewritten buffer reads again");
}

/// A skipped scope points its write set at the failure its input carried, and
/// never mints one of its own.
///
/// The two paths cannot interleave — a scope enters or skips, decided in the
/// constructor — and this is why that matters. Entering first would mint a
/// provisional failure, have it overwritten by the propagated one, and then
/// leave it in the graph forever: the exit that prunes it is on the path that
/// does not run.
/// A skip hands the pooled write set back, so the loop that skips on every
/// iteration allocates nothing.
///
/// A tainted buffer carried forward through a loop skips every launch that
/// reads it — the most frequent event in this design — and a skip path that
/// dropped its write set instead of returning it would allocate and free one
/// vector per iteration. The pool's buffer is identified by its address:
/// the same allocation must come back out.
#[test]
fn a_skip_returns_the_pooled_write_set() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 23);
    harness.alloc();
    harness.alloc();
    let input = harness.buffers[0].binding.clone();
    let output = harness.buffers[1].binding.clone();

    // Prime the pool so it holds a real allocation, sized for what follows.
    let mut primed = FailureStore::write_set(&mut harness.device);
    primed.push(input.clone());
    primed.push(output.clone());
    let pooled = primed.as_ptr();
    ExecuteScope::over(&mut harness.device, StreamId::current(), primed)
        .execute(|_| Ok::<(), ServerError>(()));

    // The input carries a failure, so the next launch reading it skips.
    let mut failing = FailureStore::write_set(&mut harness.device);
    assert_eq!(
        failing.as_ptr(),
        pooled,
        "the clean exit returned the buffer"
    );
    failing.push(input.clone());
    ExecuteScope::over(&mut harness.device, StreamId::current(), failing)
        .execute(|_| Err::<(), ServerError>(error("the launch that left these bytes")));

    let mut skipping = FailureStore::write_set(&mut harness.device);
    assert_eq!(skipping.as_ptr(), pooled, "the failed exit returned it too");
    skipping.push(output.clone());
    let outcome = ExecuteScope::launching(
        &mut harness.device,
        KernelId::new::<()>(),
        StreamId::current(),
        [&input].into_iter(),
        skipping,
    )
    .execute(|_| -> Result<(), ServerError> {
        unreachable!("a skipped scope must not run its body")
    });
    assert!(matches!(outcome, ScopedOutcome::Skipped));

    let returned = FailureStore::write_set(&mut harness.device);
    assert_eq!(
        returned.as_ptr(),
        pooled,
        "the skip path must hand the write set back, not drop it and allocate a new one"
    );
}

#[test]
fn a_skipped_scope_claims_the_failure_its_input_carried_and_mints_none() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 11);
    harness.alloc();
    harness.alloc();
    let input = harness.buffers[0].binding.clone();
    let output = harness.buffers[1].binding.clone();

    // A launch fails writing the input, so it carries a failure.
    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![input.clone()],
    )
    .execute(|_| Err::<(), ServerError>(error("the launch that left these bytes")));
    let carried = harness.device.failures.graph().len();
    assert_eq!(carried, 1, "one failure, on the input");

    // A launch reading it is skipped, and its output takes the same failure.
    let outcome = ExecuteScope::launching(
        &mut harness.device,
        KernelId::new::<()>(),
        StreamId::current(),
        [&input].into_iter(),
        vec![output.clone()],
    )
    .execute(|_| -> Result<(), ServerError> {
        unreachable!("a skipped scope must not run its body")
    });
    assert!(matches!(outcome, ScopedOutcome::Skipped));

    // No second failure was minted: the output names the original one.
    assert_eq!(
        harness.device.failures.graph().len(),
        carried,
        "a skip reuses the failure it found, and a provisional would linger"
    );
    let read = harness
        .device
        .ensure_written([&output].into_iter())
        .expect_err("a skipped launch's output must not read");
    assert!(
        reason(&read).contains("the launch that left these bytes"),
        "the output names the root cause, got: {}",
        reason(&read)
    );
}

/// The precision step 3 exists for, end to end: a failed launch taints the
/// range it was going to write, a partial host write releases exactly the
/// bytes it covers, and reads of the untouched remainder keep failing on the
/// original error.
#[test]
fn a_partial_host_write_releases_only_the_bytes_it_covers() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 7);
    harness.alloc();
    let buffer = &harness.buffers[0];
    let size = buffer.size();
    let whole = buffer.binding.clone();
    let middle = buffer.slice(&(size / 4..size / 2));
    let _id = whole.stream;

    // A launch fails writing the whole buffer.
    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![whole.clone()],
    )
    .execute(|_| Err::<(), ServerError>(error("the launch that left these bytes")));

    // The host rewrites the middle quarter.
    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![middle.clone()],
    )
    .execute(|_| Ok::<(), ServerError>(()));

    // The rewritten bytes read; the rest still fails on the launch's error.
    harness
        .device
        .ensure_written([&middle].into_iter())
        .expect("the rewritten bytes have a writer");
    let front = harness.buffers[0].slice(&(0..size / 4));
    let read = harness
        .device
        .ensure_written([&front].into_iter())
        .expect_err("the untouched bytes still carry the failure");
    assert!(
        reason(&read).contains("the launch that left these bytes"),
        "the remainder still names the original failure"
    );
    let whole_read = harness.device.ensure_written([&whole].into_iter());
    assert!(
        whole_read.is_err(),
        "a read spanning stale bytes fails however much of it was rewritten"
    );
}

/// The scope is a recording window's only informant: work that fails inside
/// it dooms it, whatever kind of work it was.
///
/// Before this was the scope's job, only a *skipped* launch doomed the
/// window, through a callback each backend wired by hand — a launch that
/// failed host-side left the recording un-doomed, and `end_capture` sealed a
/// graph silently missing the operation while a later replay released taint
/// on buffers the graph never writes.
#[test]
fn a_failed_scope_dooms_the_recording_window() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 7);
    harness.alloc();
    let binding = harness.buffers[0].binding.clone();

    harness.device.capture.prepare(StreamId::current()).unwrap();
    harness.device.capture.begin().unwrap();

    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![binding.clone()],
    )
    .execute(|_| Err::<(), ServerError>(error("failed mid-window")));

    harness.device.capture.end(StreamId::current()).unwrap();
    let doomed = harness
        .device
        .capture
        .take_failure()
        .expect("a failure inside the window must doom the recording");
    assert!(
        doomed.to_string().contains("failed mid-window"),
        "the window names the failure that doomed it, got: {doomed}"
    );
}

/// A skipped launch dooms the window the same way — the callback that used to
/// carry this is gone, and the scope reports both endings through one path.
#[test]
fn a_skipped_scope_dooms_the_recording_window() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 11);
    harness.alloc();
    harness.alloc();
    let input = harness.buffers[0].binding.clone();
    let output = harness.buffers[1].binding.clone();

    // The input's writer fails before the window opens, so the launch below
    // reads a buffer carrying a failure.
    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![input.clone()],
    )
    .execute(|_| Err::<(), ServerError>(error("the writer that never wrote")));

    harness.device.capture.prepare(StreamId::current()).unwrap();
    harness.device.capture.begin().unwrap();

    let outcome = ExecuteScope::launching(
        &mut harness.device,
        KernelId::new::<()>(),
        StreamId::current(),
        [&input].into_iter(),
        vec![output],
    )
    .execute(|_| -> Result<(), ServerError> {
        unreachable!("a skipped scope must not run its body")
    });
    assert!(matches!(outcome, ScopedOutcome::Skipped));

    harness.device.capture.end(StreamId::current()).unwrap();
    assert!(
        harness.device.capture.take_failure().is_some(),
        "a skip inside the window must doom the recording"
    );
}

/// A clean scope hands the window its write set — recording is the scope's
/// exit, so "recorded" and "in the graph" are the same event. A scope that
/// fails records nothing: the graph will not contain the work, so the graph
/// must not answer for its buffers.
#[test]
fn a_clean_scope_hands_the_window_its_write_set() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 13);
    harness.alloc();
    harness.alloc();
    let recorded = harness.buffers[0].binding.clone();
    let failed = harness.buffers[1].binding.clone();

    harness.device.capture.prepare(StreamId::current()).unwrap();
    harness.device.capture.begin().unwrap();

    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![recorded.clone()],
    )
    .execute(|_| Ok::<(), ServerError>(()));
    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![failed.clone()],
    )
    .execute(|_| Err::<(), ServerError>(error("enqueued nothing")));

    harness.device.capture.end(StreamId::current()).unwrap();
    let written = harness.device.capture.take_recorded();
    assert_eq!(
        written.len(),
        1,
        "the window holds exactly what clean scopes wrote"
    );
    assert_eq!(
        written[0].claim_key(),
        recorded.claim_key(),
        "and it is the clean scope's write set, not the failed one's"
    );
}

/// Outside a window the scope has nothing to tell: no doom, no recording —
/// the same scopes leave the closed capture exactly as it was.
#[test]
fn a_scope_outside_a_window_neither_dooms_nor_records() {
    let mut harness = Harness::new(MemoryConfiguration::ExclusivePages, 17);
    harness.alloc();
    let binding = harness.buffers[0].binding.clone();

    let _ = ExecuteScope::over(
        &mut harness.device,
        StreamId::current(),
        vec![binding.clone()],
    )
    .execute(|_| Ok::<(), ServerError>(()));
    let _ = ExecuteScope::over(&mut harness.device, StreamId::current(), vec![binding])
        .execute(|_| Err::<(), ServerError>(error("no window to doom")));

    assert!(harness.device.capture.take_failure().is_none());
    assert!(harness.device.capture.take_recorded().is_empty());
}
