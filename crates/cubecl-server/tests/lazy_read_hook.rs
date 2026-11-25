//! The write scope starts the lazy reads its server keeps of the memory it writes, and no others.

use cubecl_common::bytes::Bytes;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::logging::ServerLogger;
use cubecl_server::memory_management::{
    ErrorGraph, FailureId, MemoryConfiguration, MemoryManagement, MemoryManagementOptions,
    PageUpdate,
};
use cubecl_server::server::{
    BufferBinding, CopyDescriptor, Handle, LazyRead, LazyReadHold, ServerError,
};
use cubecl_server::storage::BytesStorage;
use cubecl_server::stream::{
    ExecuteScope, FailureStore, Failures, StreamCapture, StreamFactory, StreamFence, StreamMemory,
    StreamPool, WriteScoped, failed_writing, keep_lazy_read, lazy_read_accessed,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const SIZE: u64 = 256;

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
    logger: Arc<ServerLogger>,
}

impl StreamFactory for Factory {
    type Stream = TestStream;

    fn create(&mut self) -> Result<Self::Stream, ServerError> {
        Ok(TestStream {
            memory: MemoryManagement::from_configuration(
                BytesStorage::default(),
                &MemoryDeviceProperties::new(128 * 1024, 32),
                MemoryConfiguration::ExclusivePages,
                self.logger.clone(),
                MemoryManagementOptions::new("lazy read hook"),
            ),
        })
    }
}

/// A server that records the stream of each read a write starts, each fence past its copy,
/// and each wait a write makes on another stream's copy, as `(writer, reader)`: on the copy's
/// fence in `waits`, on all the reader's work in `stream_waits`.
struct Device {
    pool: StreamPool<Factory>,
    failures: Failures,
    capture: StreamCapture,
    started: Vec<StreamId>,
    copied: Vec<core::ops::Range<u64>>,
    /// How many started copies were let go of.
    copies_dropped: Arc<AtomicUsize>,
    fences: Vec<Arc<Fence>>,
    waits: Vec<(StreamId, StreamId)>,
    stream_waits: Vec<(StreamId, StreamId)>,
}

#[derive(Debug)]
struct Fence {
    stream: StreamId,
    reached: AtomicBool,
}

impl StreamFence for Fence {
    fn is_reached(&self) -> bool {
        self.reached.load(Ordering::Relaxed)
    }
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

    fn read_before_write(
        &mut self,
        sources: Vec<CopyDescriptor>,
        stream: StreamId,
    ) -> Option<DynFut<Result<Vec<Bytes>, ServerError>>> {
        self.started.push(stream);
        let ranges: Vec<_> = sources.iter().map(|source| source.handle.range()).collect();
        self.copied.extend(ranges.iter().cloned());
        let dropped = CountsDrop(self.copies_dropped.clone());
        Some(Box::pin(async move {
            let _dropped = dropped;
            Ok(ranges
                .into_iter()
                .map(|range| Bytes::from_bytes_vec(vec![0; (range.end - range.start) as usize]))
                .collect())
        }))
    }

    fn fence_stream(&mut self, stream: StreamId) -> Option<Arc<dyn StreamFence>> {
        let fence = Arc::new(Fence {
            stream,
            reached: AtomicBool::new(false),
        });
        self.fences.push(fence.clone());
        Some(fence)
    }

    fn wait_for_fence(&mut self, writer: StreamId, fence: &dyn StreamFence) {
        let fence = (fence as &dyn core::any::Any)
            .downcast_ref::<Fence>()
            .expect("a fence of this device");
        self.waits.push((writer, fence.stream));
    }

    fn wait_before_write(&mut self, writer: StreamId, reader: StreamId) {
        self.stream_waits.push((writer, reader));
    }
}

impl Device {
    fn new() -> Self {
        let logger = Arc::new(ServerLogger::default());
        Self {
            pool: StreamPool::new(
                Factory {
                    logger: logger.clone(),
                },
                4,
                0,
            ),
            failures: Failures::new(logger),
            capture: StreamCapture::new(&cubecl_runtime::server::DeviceCaptures::default()),
            started: Vec::new(),
            copied: Vec::new(),
            copies_dropped: Arc::new(AtomicUsize::new(0)),
            fences: Vec::new(),
            waits: Vec::new(),
            stream_waits: Vec::new(),
        }
    }

    /// A buffer of [`SIZE`] bytes on `stream`, and its handle to keep it alive.
    fn buffer(&mut self, stream: StreamId) -> (Handle, BufferBinding) {
        let service =
            cubecl_common::device::ServiceId::of::<()>(cubecl_common::device::DeviceId::new(0, 0));
        let handle = Handle::new(service, stream, SIZE);
        let memory = &mut self.pool.get_mut(&stream).memory;
        let reserved = memory
            .reserve(SIZE, PageUpdate::Allow, self.failures.graph_mut())
            .expect("a fresh pool should fit one buffer");
        memory
            .bind(
                reserved,
                handle.memory.clone(),
                0,
                self.failures.graph_mut(),
            )
            .expect("a reserved slice should bind");
        let binding = handle.clone().binding();
        (handle, binding)
    }

    fn write(&mut self, stream: StreamId, written: BufferBinding) {
        ExecuteScope::over(self, stream, vec![written])
            .execute(|_| Ok::<(), ServerError>(()))
            .into_result()
            .expect("a scope over clean memory should run");
    }
}

struct CountsDrop(Arc<AtomicUsize>);

impl Drop for CountsDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn reach_every_fence(device: &Device) {
    for fence in &device.fences {
        fence.reached.store(true, Ordering::Relaxed);
    }
}

/// A lazy read of all of `handle`, held whole as its bytes hold it.
fn lazy_read(handle: &Handle) -> LazyReadHold {
    Arc::new(LazyRead::new(CopyDescriptor::new(
        handle.clone(),
        [SIZE as usize].into(),
        [1].into(),
        1,
    )))
    .hold(0..SIZE as usize)
}

/// An access's copy of the first four bytes of `read` on `stream`, in flight until dropped.
fn access(
    device: &mut Device,
    read: &Arc<LazyRead>,
    binding: &BufferBinding,
    stream: StreamId,
) -> DynFut<Result<Vec<Bytes>, ServerError>> {
    let copy: DynFut<Result<Vec<Bytes>, ServerError>> = Box::pin(async { Ok(Vec::new()) });
    lazy_read_accessed(device, read, bytes(0..4, binding).range(), stream, copy)
}

fn bytes(range: core::ops::Range<u64>, binding: &BufferBinding) -> BufferBinding {
    let mut binding = binding.clone();
    binding.offset_start = Some(range.start);
    binding.offset_end = Some(binding.size - range.end);
    binding
}

#[test]
fn a_write_over_memory_no_lazy_read_holds_starts_no_read() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (_handle, binding) = device.buffer(stream);

    device.write(stream, binding);

    assert!(device.started.is_empty());
}

#[test]
fn a_write_over_a_kept_lazy_read_starts_it_once() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), stream).expect("clean memory should be readable");

    device.write(stream, binding.clone());
    device.write(stream, binding);

    assert_eq!(device.started, vec![stream]);
}

#[test]
fn a_write_on_another_stream_copies_behind_the_reads_position_on_its_own() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");

    device.write(writer, binding);

    assert_eq!(
        (device.started, device.waits, device.stream_waits),
        (vec![writer], vec![(writer, reader)], vec![])
    );
}

#[test]
fn a_write_after_the_reader_passed_the_read_waits_on_nothing() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    device.fences[0].reached.store(true, Ordering::Relaxed);

    device.write(writer, binding);

    assert_eq!(
        (device.started, device.waits, device.stream_waits),
        (vec![writer], vec![], vec![])
    );
}

#[test]
fn a_write_between_views_that_outlived_their_read_leaves_it_kept() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    keep_lazy_read(&mut device, held.read().clone(), stream)
        .expect("clean memory should be readable");
    let _views = (
        held.narrowed(0..4),
        held.narrowed(SIZE as usize - 4..SIZE as usize),
    );
    drop(held);

    device.write(stream, bytes(SIZE / 2..SIZE / 2 + 4, &binding));

    assert!(device.started.is_empty());
}

#[test]
fn views_that_outlived_their_read_copy_their_own_bytes_alone() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    keep_lazy_read(&mut device, held.read().clone(), stream)
        .expect("clean memory should be readable");
    let _views = (
        held.narrowed(0..4),
        held.narrowed(SIZE as usize - 4..SIZE as usize),
    );
    drop(held);

    device.write(stream, bytes(0..4, &binding));

    assert_eq!(device.copied, vec![0..4, SIZE - 4..SIZE]);
}

#[test]
fn a_write_beside_a_kept_lazy_read_leaves_it_kept() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let mut front = handle.clone();
    front.offset_end = Some(SIZE / 2);
    let read = Arc::new(LazyRead::new(CopyDescriptor::new(
        front,
        [SIZE as usize / 2].into(),
        [1].into(),
        1,
    )));
    let _held = read.hold(0..SIZE as usize / 2);
    keep_lazy_read(&mut device, read.clone(), stream).expect("clean memory should be readable");

    device.write(stream, bytes(SIZE / 2..SIZE, &binding));

    assert!(device.started.is_empty() && read.is_kept());
}

#[test]
fn a_dropped_lazy_read_is_not_started() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    keep_lazy_read(&mut device, lazy_read(&handle).read().clone(), stream)
        .expect("clean memory should be readable");

    device.write(stream, binding);

    assert!(device.started.is_empty());
}

#[test]
fn a_write_from_another_stream_waits_for_a_copy_in_flight() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(writer);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    device.write(reader, bytes(0..SIZE / 2, &binding));

    device.write(writer, bytes(SIZE / 2..SIZE, &binding));

    assert_eq!(device.waits, vec![(writer, reader)]);
}

#[test]
fn a_failed_write_starts_the_read_its_claim_would_fail() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), stream).expect("clean memory should be readable");

    failed_writing(
        &mut device,
        stream,
        vec![binding],
        ServerError::Generic {
            reason: "never compiled".into(),
            backtrace: Default::default(),
        },
    );

    assert_eq!(device.started, vec![stream]);
}

#[test]
fn a_kept_lazy_read_leaves_its_memory_free_to_move() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    drop(binding);
    let held = lazy_read(&handle);
    let read = held.read().clone();

    keep_lazy_read(&mut device, read.clone(), stream).expect("clean memory should be readable");

    assert!(!handle.memory.is_bound());
}

#[test]
fn a_write_from_another_stream_waits_on_the_copy_alone() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(writer);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    device.write(reader, bytes(0..SIZE / 2, &binding));

    device.write(writer, bytes(SIZE / 2..SIZE, &binding));

    assert!(device.stream_waits.is_empty());
}

#[test]
fn a_write_after_the_copy_is_done_waits_on_nothing() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(writer);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    device.write(reader, bytes(0..SIZE / 2, &binding));
    reach_every_fence(&device);

    device.write(writer, bytes(SIZE / 2..SIZE, &binding));

    assert!(device.waits.is_empty() && device.stream_waits.is_empty());
}

#[test]
fn a_write_beside_a_view_that_outlived_its_read_leaves_it_kept() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    keep_lazy_read(&mut device, held.read().clone(), stream)
        .expect("clean memory should be readable");
    let _view = held.narrowed(0..4);
    drop(held);

    device.write(stream, bytes(SIZE / 2..SIZE, &binding));

    assert!(device.started.is_empty());
}

#[test]
fn a_view_that_outlived_its_read_copies_only_its_bytes() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    keep_lazy_read(&mut device, held.read().clone(), stream)
        .expect("clean memory should be readable");
    let _view = held.narrowed(4..8);
    drop(held);

    device.write(stream, binding);

    assert_eq!(device.copied, vec![4..8]);
}

#[test]
fn a_write_after_an_access_from_another_stream_waits_on_its_copy() {
    let mut device = Device::new();
    let (reader, accessor) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    let _copy = access(&mut device, &read, &binding, accessor);

    device.write(reader, binding);

    assert_eq!(device.waits, vec![(reader, accessor)]);
}

#[test]
fn an_access_leaves_the_rest_of_its_read_kept() {
    let mut device = Device::new();
    let (reader, accessor) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");

    let _copy = access(&mut device, &read, &binding, accessor);

    assert!(device.started.is_empty() && read.is_kept());
}

#[test]
fn a_write_beside_an_access_waits_on_nothing() {
    let mut device = Device::new();
    let (reader, accessor) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let mut front = handle.clone();
    front.offset_end = Some(SIZE / 2);
    let read = Arc::new(LazyRead::new(CopyDescriptor::new(
        front,
        [SIZE as usize / 2].into(),
        [1].into(),
        1,
    )));
    let _held = read.hold(0..SIZE as usize / 2);
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    let _copy = access(&mut device, &read, &binding, accessor);

    device.write(reader, bytes(SIZE / 2..SIZE, &binding));

    assert!(device.waits.is_empty());
}

#[test]
fn a_write_after_a_shared_read_copied_waits_on_nothing() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(writer);
    handle.memory.descriptor().share_with_peers();
    let held = lazy_read(&handle);
    keep_lazy_read(&mut device, held.read().clone(), reader)
        .expect("clean memory should be readable");
    reach_every_fence(&device);

    device.write(writer, binding);

    assert!(device.waits.is_empty() && device.stream_waits.is_empty());
}

#[test]
fn a_write_after_an_access_has_its_bytes_waits_on_nothing() {
    let mut device = Device::new();
    let (reader, accessor) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    let copy = access(&mut device, &read, &binding, accessor);
    cubecl_environment::future::block_on(copy).expect("the fake copy should succeed");

    device.write(reader, binding);

    assert!(device.waits.is_empty() && device.stream_waits.is_empty());
}

#[test]
fn an_access_in_flight_is_found_after_its_read_lets_go() {
    let mut device = Device::new();
    let (reader, accessor) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");
    let _copy = access(&mut device, &read, &binding, accessor);
    // As the read resolving through another holder lets go of its count.
    read.descriptor().lazy_read_let_go();

    device.write(reader, binding);

    assert_eq!(device.waits, vec![(reader, accessor)]);
}

#[test]
fn a_read_dropped_unread_lets_go_of_its_copy_once_it_is_done() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    keep_lazy_read(&mut device, held.read().clone(), stream)
        .expect("clean memory should be readable");
    device.write(stream, binding);
    drop(held);
    let kept_while_running = device.copies_dropped.load(Ordering::Relaxed);
    reach_every_fence(&device);

    let next = lazy_read(&handle);
    keep_lazy_read(&mut device, next.read().clone(), stream)
        .expect("clean memory should be readable");

    assert_eq!(
        (
            kept_while_running,
            device.copies_dropped.load(Ordering::Relaxed)
        ),
        (0, 1)
    );
}

#[test]
fn a_write_over_a_rank_zero_read_copies_its_binding() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let scalar = CopyDescriptor::new(
        handle.clone(),
        [0usize; 0].into(),
        [0usize; 0].into(),
        SIZE as usize,
    );
    let held = Arc::new(LazyRead::new(scalar)).hold(0..SIZE as usize);
    keep_lazy_read(&mut device, held.read().clone(), stream)
        .expect("clean memory should be readable");

    device.write(stream, binding);

    assert_eq!(device.copied, vec![0..SIZE]);
}

#[test]
fn a_read_dropped_unread_after_its_copy_is_done_lets_go_of_it_at_once() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let held = lazy_read(&handle);
    keep_lazy_read(&mut device, held.read().clone(), stream)
        .expect("clean memory should be readable");
    device.write(stream, binding);
    reach_every_fence(&device);

    drop(held);

    assert_eq!(device.copies_dropped.load(Ordering::Relaxed), 1);
}

#[test]
fn a_write_past_a_contiguous_read_of_a_longer_handle_leaves_it_kept() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (handle, binding) = device.buffer(stream);
    let read = Arc::new(LazyRead::new(CopyDescriptor::new(
        handle,
        [4].into(),
        [1].into(),
        1,
    )));
    let _held = read.hold(0..4);
    keep_lazy_read(&mut device, read.clone(), stream).expect("clean memory should be readable");

    device.write(stream, bytes(SIZE - 4..SIZE, &binding));

    assert!(device.started.is_empty() && read.is_kept());
}
