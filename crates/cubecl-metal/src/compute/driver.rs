//! What the shared [`Command`](cubecl_server::command::Command) cannot do
//! itself: Metal's device calls, and where a Metal stream keeps the state
//! they move in step with.
//!
//! Metal's buffers are shared with the host, so a host copy is the host's own
//! `memcpy` once the stream has drained, and the host memory it lands in needs
//! no pinning.

use crate::compute::capture::RecordedDispatch;
use crate::compute::context::MetalContext;
use crate::compute::pipelines::MetalCompiledKernel;
use crate::compute::stream::{MetalEvent, MetalStream, MetalStreamBackend, QueueRef};
use crate::memory::{MetalResource, MetalStorage};
use cubecl_common::bytes::Bytes;
use cubecl_server::command::{CopyLayout, DeviceStream, Driver};
use cubecl_server::id::KernelId;
use cubecl_server::memory_management::drop_queue::PendingDropQueue;
use cubecl_server::memory_management::{ManagedMemoryBinding, MemoryManagement};
use cubecl_server::metadata_cache::MetadataInfoCache;
use cubecl_server::server::{Handle, IoError, LaunchError};
use cubecl_server::storage::{
    BytesResource, BytesStorage, PinnedMemoryAllocController, PinnedMemoryResource,
};
use cubecl_server::stream::StreamCapture;
use objc2_metal::MTLBlitCommandEncoder;

impl DeviceStream for MetalStream {
    type Fence = MetalEvent;
    type DeviceStorage = MetalStorage;
    type HostStorage = BytesStorage;
    type Signal = QueueRef;

    fn device_memory(&mut self) -> &mut MemoryManagement<MetalStorage> {
        &mut self.memory_management
    }

    fn host_memory(&mut self) -> &mut MemoryManagement<BytesStorage> {
        &mut self.host_memory
    }

    fn drop_queue(&mut self) -> &mut PendingDropQueue<MetalEvent> {
        &mut self.drop_queue
    }

    fn capturing(&mut self) -> &mut StreamCapture {
        &mut self.capturing
    }

    fn info_cache(&mut self) -> &mut MetadataInfoCache<Handle> {
        &mut self.info_cache
    }

    fn signal(&self) -> QueueRef {
        self.queue_ref()
    }

    fn fence(signal: QueueRef) -> MetalEvent {
        signal.get().flush()
    }
}

/// What a Metal dispatch binds: its buffers in order, then the launch's
/// metadata words after them.
#[derive(Default)]
pub struct MetalBindings {
    /// The kernel's buffers, bound at their position.
    pub buffers: Vec<MetalResource>,
    /// The launch's metadata, bound right after the buffers.
    pub info: Vec<u64>,
}

/// Metal, as the shared command sees it.
pub(crate) struct Metal;

impl Driver for Metal {
    type Backend = MetalStreamBackend;
    type Stream = MetalStream;
    type Context = MetalContext;
    type LaunchArgs = MetalBindings;
    type Loaded = MetalCompiledKernel;

    unsafe fn pinned_bytes(
        binding: ManagedMemoryBinding,
        resource: BytesResource,
        size: usize,
    ) -> Bytes {
        // Plain host memory serves as pinned: the copies into and out of it
        // are the host's own.
        let (ptr, len) = resource.get_write_ptr_and_length();
        let resource = PinnedMemoryResource { ptr, size: len };
        let controller = Box::new(PinnedMemoryAllocController::init(binding, resource));
        // SAFETY: the caller guarantees `binding` names initialized memory of
        // at least `size` bytes.
        unsafe { Bytes::from_controller(controller, size) }
    }

    unsafe fn copy_to_host(
        resource: &MetalResource,
        layout: &CopyLayout<'_>,
        bytes: &mut Bytes,
        stream: &MetalStream,
    ) -> Result<(), IoError> {
        stream.host_access("read")?;
        let source = resource.host_ptr(layout, bytes.len())?;
        let target = bytes.as_mut_ptr();
        match layout.pitch {
            // SAFETY: the stream has drained, so nothing writes the source,
            // `host_ptr` checked the source holds the copy, and the
            // destination was sized for it.
            None => unsafe { std::ptr::copy_nonoverlapping(source, target, bytes.len()) },
            Some(pitch) => {
                for row in 0..pitch.height {
                    // SAFETY: as above, a row at a time: `width_bytes` apart
                    // in the destination, `stride_bytes` in the source.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            source.add(row * pitch.stride_bytes),
                            target.add(row * pitch.width_bytes),
                            pitch.width_bytes,
                        )
                    };
                }
            }
        }
        Ok(())
    }

    unsafe fn copy_to_device(
        resource: &MetalResource,
        layout: &CopyLayout<'_>,
        data: &[u8],
        stream: &MetalStream,
    ) -> Result<(), IoError> {
        stream.host_access("write")?;
        let target = resource.host_ptr(layout, data.len())?;
        match layout.pitch {
            // SAFETY: the stream has drained, so nothing reads or writes the
            // target, and `host_ptr` checked it holds the copy.
            None => unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), target, data.len()) },
            Some(pitch) => {
                for row in 0..pitch.height {
                    // SAFETY: as above, a row at a time: `width_bytes` apart
                    // in the source, `stride_bytes` in the destination.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr().add(row * pitch.width_bytes),
                            target.add(row * pitch.stride_bytes),
                            pitch.width_bytes,
                        )
                    };
                }
            }
        }
        Ok(())
    }

    unsafe fn copy_on_device(
        source: &MetalResource,
        target: &MetalResource,
        queue: QueueRef,
    ) -> Result<(), IoError> {
        let mut queue = queue.get();
        // SAFETY: both are live slices of the same size on distinct buffers,
        // left alone until the queue's next fence has passed.
        unsafe {
            queue
                .batch()
                .blit()
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source.buffer(),
                    source.offset(),
                    target.buffer(),
                    target.offset(),
                    // Blits copy whole words; the slices are padded to the
                    // device alignment, a multiple of four, so the rounded
                    // size stays inside both.
                    source.size().next_multiple_of(4),
                );
        }
        // Counted as a launch is, so the batch holding it is work a profile
        // window measures and the batch limits see.
        queue.encoded(1, source.size());
        Ok(())
    }

    fn launch(
        _ctx: &mut MetalContext,
        stream: &mut MetalStream,
        _id: &KernelId,
        kernel: &MetalCompiledKernel,
        count: (u32, u32, u32),
        bindings: &mut MetalBindings,
    ) -> Result<(), LaunchError> {
        if stream.capturing.is_recording() {
            let bindings = core::mem::take(bindings);
            stream
                .recording
                .push(RecordedDispatch::new(kernel, count, bindings));
            return Ok(());
        }
        stream.queue().dispatch(kernel, count, bindings)
    }
}
