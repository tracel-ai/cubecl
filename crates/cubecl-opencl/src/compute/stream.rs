use crate::compute::cl::{self, ClError, Event};
use crate::memory::{Memory, OpenClStorage};
use cubecl_core::{
    MemoryConfiguration,
    server::{DeviceCaptures, Handle, ServerError},
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, FailureId, MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
        drop_queue::{Fence, PendingDropQueue},
    },
    metadata_cache::{MetadataCachePolicy, MetadataInfoCache},
    server::BufferBinding,
    storage::BytesStorage,
    stream::{EventStreamBackend, StreamCapture, StreamMemory},
};
use opencl_sys::{cl_command_queue, cl_context, cl_device_id};
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Arc;

/// Launches before a stream's first flush after it synced (a read or a sync, not the flushes other
/// streams and released memory ask for); each flush doubles the batch, up to [`LAST_BATCH`]: work
/// reaches the device soon after a sync, and a host running ahead submits few, large batches (a
/// submission can leave the device idle a while).
const FIRST_BATCH: usize = 32;
const LAST_BATCH: usize = 1024;

/// A stream: a command queue and the memory it allocates.
#[derive(Debug)]
pub struct OpenClStream {
    pub(crate) queue: Queue,
    pub(crate) memory_management: MemoryManagement<OpenClStorage>,
    /// Where reads land and writes are staged: plain host memory, which every copy takes.
    pub(crate) host_memory: MemoryManagement<BytesStorage>,
    /// Host data of copies still in flight, until a marker after them completes.
    pub(crate) drop_queue: PendingDropQueue<Marker>,
    /// Never recording: `OpenCL` has no graphs.
    pub(crate) capturing: StreamCapture,
    pub(crate) info_cache: MetadataInfoCache<Handle>,
    batch_ops: Cell<usize>,
    /// Commands the current batch takes before it is flushed.
    batch: Cell<usize>,
}

// SAFETY: the queue is a thread-safe OpenCL handle the stream owns a reference to, and the server owns the stream.
unsafe impl Send for OpenClStream {}

impl Drop for OpenClStream {
    fn drop(&mut self) {
        if cl::finish(self.queue.raw).is_err() {
            // Copies from the host data may still run: leaked rather than freed under them.
            core::mem::forget(core::mem::take(&mut self.drop_queue));
        }
        // The storage holds its own reference to the queue.
        cl::api().clReleaseCommandQueue(self.queue.raw);
    }
}

impl StreamMemory for OpenClStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory_management
            .failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.memory_management
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory_management
            .written(&binding.memory, binding.range(), failures)
    }
}

impl OpenClStream {
    /// Orders what follows after the commands just enqueued, and submits the batch when it is full.
    pub(crate) fn enqueued(&self) -> Result<(), ClError> {
        self.queue.barrier()?;
        let ops = self.batch_ops.get() + 1;
        if ops < self.batch.get() {
            self.batch_ops.set(ops);
            return Ok(());
        }
        self.batch_ops.set(0);
        self.batch.set((2 * self.batch.get()).min(LAST_BATCH));
        cl::flush(self.queue.raw)
    }

    /// The host waits for the stream: its next commands start the device early again.
    pub(crate) fn synced(&self) {
        self.batch.set(FIRST_BATCH);
    }
}

/// A stream's queue, as its commands are enqueued: with how its memory is copied, and whether it is
/// an out-of-order queue kept in order by a barrier after every command (see
/// [`OpenClStreamBackend::new`]).
#[derive(Debug, Clone, Copy)]
pub struct Queue {
    pub(crate) raw: cl_command_queue,
    pub(crate) memory: Memory,
    barriers: bool,
}

impl Queue {
    /// Orders the queue's later commands after its earlier ones.
    pub(crate) fn barrier(self) -> Result<(), ClError> {
        // SAFETY: a live stream's queue.
        if self.barriers
            && let Err(err) = unsafe { cl::barrier(self.raw) }
        {
            // Nothing orders the command before the next one: wait for it instead.
            let _ = cl::finish(self.raw);
            return Err(err);
        }
        Ok(())
    }

    /// Enqueues the copies `(destination, source, bytes)`; where one fails, waits for the ones
    /// enqueued before it, which use memory the caller then frees.
    ///
    /// # Safety
    ///
    /// Both ranges of each copy are valid for its bytes until the queue has run it.
    pub(crate) unsafe fn copies(
        self,
        copies: impl IntoIterator<Item = (*mut c_void, *const c_void, usize)>,
    ) -> Result<(), ClError> {
        let enqueued = copies.into_iter().try_for_each(|(dst, src, len)| {
            // SAFETY: as the function's.
            unsafe { self.memory.copy(self.raw, false, dst, src, len) }
        });
        if enqueued.is_err() {
            let _ = cl::finish(self.raw);
        }
        enqueued
    }
}

/// A marker after everything enqueued on a queue, which it submits (or why it couldn't be made):
/// the fence of the stream's held memory, and what other streams wait on.
#[derive(Debug)]
pub struct Marker(Result<Event, ClError>);

impl Marker {
    pub(crate) fn on(queue: cl_command_queue) -> Self {
        // SAFETY: a live stream's queue.
        Self(unsafe { cl::marker(queue) }.and_then(|event| cl::flush(queue).map(|()| event)))
    }
}

impl Fence for Marker {
    fn wait(self) -> Result<(), ServerError> {
        self.0?.wait().map_err(ClError::into_sync)
    }
}

/// Creates the streams of one device.
#[derive(Debug)]
pub struct OpenClStreamBackend {
    memory: Memory,
    context: cl_context,
    device: cl_device_id,
    out_of_order: bool,
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
    /// The device's captures, which every stream takes its (never recording) capture state from.
    captures: DeviceCaptures,
}

// SAFETY: OpenCL contexts and devices are thread-safe handles; the server keeps the context alive.
unsafe impl Send for OpenClStreamBackend {}

impl OpenClStreamBackend {
    /// `out_of_order`: streams on out-of-order queues kept in order by barriers (Intel's driver submits an in-order
    /// queue's every command on its own; out of order it batches them).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        memory: Memory,
        context: cl_context,
        device: cl_device_id,
        out_of_order: bool,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        captures: DeviceCaptures,
    ) -> Self {
        Self {
            memory,
            context,
            device,
            out_of_order,
            mem_props,
            mem_config,
            logger,
            captures,
        }
    }
}

impl EventStreamBackend for OpenClStreamBackend {
    type Stream = OpenClStream;
    type Event = Marker;

    fn create_stream(&self) -> Result<Self::Stream, ServerError> {
        // Profiled, for the device-timed profiling windows.
        let mut flags = opencl_sys::CL_QUEUE_PROFILING_ENABLE;
        if self.out_of_order {
            flags |= opencl_sys::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        }
        let properties = [
            opencl_sys::CL_QUEUE_PROPERTIES as opencl_sys::cl_queue_properties,
            flags,
            0,
        ];
        // SAFETY: a queue on this backend's context and device, both alive while the server is.
        let queue = unsafe {
            cl3::command_queue::create_command_queue_with_properties(
                self.context,
                self.device,
                properties.as_ptr(),
            )
        }
        .map_err(ClError::of("clCreateCommandQueueWithProperties"))?;
        let storage = OpenClStorage::new(
            self.memory,
            self.context,
            queue,
            self.mem_props.alignment as usize,
            self.mem_props.max_page_size,
        );
        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("OpenCL device memory"),
        );
        let host_memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties::new(
                self.mem_props.max_page_size,
                BytesStorage::ALIGNMENT as u64,
            ),
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("OpenCL host memory").mode(MemoryAllocationMode::Auto),
        );
        Ok(OpenClStream {
            queue: Queue {
                raw: queue,
                memory: self.memory,
                barriers: self.out_of_order,
            },
            memory_management,
            host_memory,
            drop_queue: Default::default(),
            capturing: StreamCapture::new(&self.captures),
            info_cache: MetadataInfoCache::new(MetadataCachePolicy::default()),
            batch_ops: Cell::new(0),
            batch: Cell::new(FIRST_BATCH),
        })
    }

    fn flush(stream: &mut Self::Stream, _failures: &mut ErrorGraph) -> Self::Event {
        stream.batch_ops.set(0);
        Marker::on(stream.queue.raw)
    }

    fn handle_cursor(stream: &Self::Stream, handle: &BufferBinding) -> u64 {
        stream
            .memory_management
            .get_cursor(handle.memory.clone())
            .unwrap_or(u64::MAX)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) -> Result<(), ServerError> {
        // SAFETY: the stream's own queue.
        Ok(unsafe { cl::wait_on(stream.queue.raw, &event.0?) }?)
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait()
    }
}
