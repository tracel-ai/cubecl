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
    CopyFence, ExecuteScope, FailureStore, Failures, StreamCapture, StreamFactory, StreamMemory,
    StreamPool, WriteScoped, failed_writing, keep_lazy_read,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    fences: Vec<Arc<Fence>>,
    waits: Vec<(StreamId, StreamId)>,
    stream_waits: Vec<(StreamId, StreamId)>,
}

#[derive(Debug)]
struct Fence {
    stream: StreamId,
    reached: AtomicBool,
}

impl CopyFence for Fence {
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
        source: CopyDescriptor,
        stream: StreamId,
    ) -> Option<DynFut<Result<Vec<Bytes>, ServerError>>> {
        self.started.push(stream);
        self.copied.push(source.handle.range());
        Some(Box::pin(async {
            Ok(vec![Bytes::from_bytes_vec(vec![0; SIZE as usize])])
        }))
    }

    fn fence_copy(&mut self, stream: StreamId) -> Option<Arc<dyn CopyFence>> {
        let fence = Arc::new(Fence {
            stream,
            reached: AtomicBool::new(false),
        });
        self.fences.push(fence.clone());
        Some(fence)
    }

    fn wait_for_copy(&mut self, writer: StreamId, fence: &dyn CopyFence) {
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
fn a_write_on_another_stream_starts_the_read_on_its_own() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (handle, binding) = device.buffer(reader);
    let held = lazy_read(&handle);
    let read = held.read().clone();
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");

    device.write(writer, binding);

    assert_eq!(
        (device.started, device.waits),
        (vec![reader], vec![(writer, reader)])
    );
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
    device.fences[0].reached.store(true, Ordering::Relaxed);

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
