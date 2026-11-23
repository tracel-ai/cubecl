//! A driver whose device memory is host bytes, for tests of what a command does with memory:
//! nothing is copied, launched or staged, and the device can be told to refuse an allocation.
//!
//! A test file adds what it needs on top, a server and any other driver trait.

use cubecl_common::bytes::Bytes;
use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::{
    command::{CopyLayout, DeviceResource, DeviceStream, Driver, GraphDriver, HostResource},
    id::KernelId,
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, FailureId, ManagedMemoryBinding, MemoryConfiguration, MemoryManagement,
        MemoryManagementOptions,
        drop_queue::{Fence, PendingDropQueue},
    },
    metadata_cache::{MetadataCachePolicy, MetadataInfoCache},
    server::{BufferBinding, DeviceCaptures, Handle, IoError, LaunchError, ServerError},
    storage::{BytesStorage, StorageAllocator, StorageHandle, StorageId},
    stream::{EventStreamBackend, StreamCapture, StreamMemory},
};
use std::cell::Cell;
use std::fmt;
use std::sync::Arc;

const MIB: u64 = 1024 * 1024;

std::thread_local! {
    /// How many of this thread's next device allocations [`FlakyStorage`] refuses, as a full
    /// device would: set it to have the device refuse, and read it to see the refusals spent.
    pub static REFUSALS: Cell<usize> = const { Cell::new(0) };
}

/// Host bytes standing in for device memory, out of memory whenever [`REFUSALS`] says.
#[derive(Default)]
pub struct FlakyStorage {
    bytes: BytesStorage,
}

impl StorageAllocator for FlakyStorage {
    type Resource = <BytesStorage as StorageAllocator>::Resource;

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        self.bytes.get(handle)
    }

    fn allocate_bytes(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let refused = REFUSALS.with(|refusals| refusals.replace(refusals.get().saturating_sub(1)));
        if refused > 0 {
            return Err(IoError::OutOfMemory {
                size,
                backtrace: BackTrace::capture(),
            });
        }
        self.bytes.allocate_bytes(size)
    }

    fn deallocate(&mut self, id: StorageId) {
        self.bytes.deallocate(id)
    }

    fn flush(&mut self) {
        self.bytes.flush()
    }

    fn bytes_allocated(&self) -> u64 {
        self.bytes.bytes_allocated()
    }
}

#[derive(Debug)]
pub struct Fake;

#[derive(Debug)]
pub struct FakeBackend;

pub struct FakeFence;

impl Fence for FakeFence {
    fn wait(self) -> Result<(), ServerError> {
        Ok(())
    }
}

pub struct FakeStream {
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

fn memory<S: StorageAllocator + Default>(name: &'static str) -> MemoryManagement<S> {
    MemoryManagement::from_configuration(
        S::default(),
        &MemoryDeviceProperties::new(512 * MIB, 32),
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
        source: &DeviceResource<Self>,
        target: &DeviceResource<Self>,
        _queue: (),
    ) -> Result<(), IoError> {
        let (from, len) = source.get_write_ptr_and_length();
        let (into, room) = target.get_write_ptr_and_length();
        assert_eq!(
            len, room,
            "a relocation copies between storages of one size"
        );
        // SAFETY: both resources are live slices of host bytes, `len` long, and a relocation
        // never copies a slice onto itself.
        unsafe { core::ptr::copy_nonoverlapping(from, into, len) };
        Ok(())
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
