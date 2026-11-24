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
use cubecl_server::server::{BufferBinding, CopyDescriptor, Handle, LazyRead, ServerError};
use cubecl_server::storage::BytesStorage;
use cubecl_server::stream::{
    ExecuteScope, FailureStore, Failures, StreamCapture, StreamFactory, StreamMemory, StreamPool,
    WriteScoped, keep_lazy_read,
};
use std::sync::Arc;

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

/// A server that records each read a write starts, as `(read's stream, writer's stream)`.
struct Device {
    pool: StreamPool<Factory>,
    failures: Failures,
    capture: StreamCapture,
    started: Vec<(StreamId, StreamId)>,
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
        _source: CopyDescriptor,
        stream: StreamId,
        writer: StreamId,
    ) -> Option<DynFut<Result<Vec<Bytes>, ServerError>>> {
        self.started.push((stream, writer));
        Some(Box::pin(async {
            Ok(vec![Bytes::from_bytes_vec(vec![0; SIZE as usize])])
        }))
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

fn lazy_read(binding: &BufferBinding) -> Arc<LazyRead> {
    Arc::new(LazyRead::new(CopyDescriptor::new(
        binding.clone(),
        [SIZE as usize].into(),
        [1].into(),
        1,
    )))
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
    let (_handle, binding) = device.buffer(stream);
    let read = lazy_read(&binding);
    keep_lazy_read(&mut device, read.clone(), stream).expect("clean memory should be readable");

    device.write(stream, binding.clone());
    device.write(stream, binding);

    assert_eq!(device.started, vec![(stream, stream)]);
}

#[test]
fn a_write_on_another_stream_starts_the_read_on_its_own() {
    let mut device = Device::new();
    let (reader, writer) = (StreamId { value: 0 }, StreamId { value: 1 });
    let (_handle, binding) = device.buffer(reader);
    let read = lazy_read(&binding);
    keep_lazy_read(&mut device, read.clone(), reader).expect("clean memory should be readable");

    device.write(writer, binding);

    assert_eq!(device.started, vec![(reader, writer)]);
}

#[test]
fn a_write_beside_a_kept_lazy_read_leaves_it_kept() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (_handle, binding) = device.buffer(stream);
    let read = Arc::new(LazyRead::new(CopyDescriptor::new(
        bytes(0..SIZE / 2, &binding),
        [SIZE as usize / 2].into(),
        [1].into(),
        1,
    )));
    keep_lazy_read(&mut device, read.clone(), stream).expect("clean memory should be readable");

    device.write(stream, bytes(SIZE / 2..SIZE, &binding));

    assert!(device.started.is_empty() && read.is_kept());
}

#[test]
fn a_dropped_lazy_read_is_not_started() {
    let mut device = Device::new();
    let stream = StreamId { value: 0 };
    let (_handle, binding) = device.buffer(stream);
    keep_lazy_read(&mut device, lazy_read(&binding), stream)
        .expect("clean memory should be readable");

    device.write(stream, binding);

    assert!(device.started.is_empty());
}
