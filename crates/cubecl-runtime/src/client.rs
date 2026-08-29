use crate::{
    config::memory::MemoryPoolsConfig,
    config::{TypeNameFormatLevel, type_name_format},
    id::GraphId,
    kernel::KernelMetadata,
    logging::ProfileLevel,
    memory_management::{
        InstallMemoryPoolsError, MemoryAllocationMode, MemoryConfiguration, MemoryReport,
        MemoryUsage,
    },
    runtime::Runtime,
    server::{
        BufferBinding, CommunicationId, ComputeServer, CopyDescriptor, CubeCount, Handle,
        KernelArguments, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy,
        MemoryLayoutStrategy, ProfileError, ReduceOperation, ServerCommunication, ServerError,
        ServerUtilities,
    },
    storage::{ComputeStorage, ManagedResource},
    throughput::{
        KernelConfig, ThroughputBenchmarker, ThroughputCache, ThroughputKey, ThroughputValue,
    },
};
use alloc::{format, string::String, sync::Arc, vec, vec::Vec};

#[cfg(not(target_family = "wasm"))]
mod lazy;
use cubecl_common::{
    bytes::{AllocationProperty, Bytes},
    device::{Device, DeviceId},
    device_handle::{CallResultExt, DeviceHandle},
    profile::ProfileDuration,
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_ir::{DeviceProperties, ElemType, VectorSize, features::Features};
use cubecl_zspace::Shape;

#[allow(unused)]
use cubecl_common::profile::TimingMethod;
use cubecl_environment::stream::StreamId;

/// The `ComputeClient` is the entry point to require tasks from the `ComputeServer`.
/// It should be obtained for a specific device via the Compute struct.
pub struct ComputeClient<R: Runtime> {
    device: DeviceHandle<R::Server>,
    utilities: Arc<ServerUtilities<R::Server>>,
    stream_id: Option<StreamId>,
}

/// A captured graph produced by [`ComputeClient::stop_capture`]: a recorded
/// launch sequence that [`replay`](Graph::replay) re-runs against its original
/// buffers, skipping the launch path it was recorded from. Cheap to clone
/// (shares one backend graph).
///
/// The graph itself lives in the backend server, referenced here only by
/// [`GraphId`]; this handle holds a reference-counted owner that releases the
/// backend graph once the last clone drops. The graph replays against the exact
/// device buffers used during capture. The caller keeps those input/output
/// [`Handle`]s alive and, each iteration, writes fresh inputs into the input
/// handles (same device pointers) and reads the output handles after replaying —
/// see [`ComputeClient::stop_capture`].
///
/// **Stream ordering.** [`replay`](Graph::replay) always dispatches on the
/// stream the graph was captured on, but input writes and output reads go on the
/// *writing client's* current stream. They are ordered against the replay only
/// when they land on that same stream, so keep the client pinned to the capture
/// stream (via [`set_stream`](ComputeClient::set_stream)) — or issue all writes,
/// replays, and reads from the same unpinned client — for the whole decode loop.
/// Refreshing inputs from a client on a different stream races the replay and
/// silently feeds it stale data.
pub struct Graph<R: Runtime> {
    inner: Arc<GraphHandle<R>>,
}

impl<R: Runtime> core::fmt::Debug for Graph<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graph")
            .field("id", &self.inner.id)
            .field("stream_id", &self.inner.stream_id)
            .finish()
    }
}

/// Reference-counted owner of a backend graph. Its [`Drop`] ships the release to
/// the server actor, so the last [`Graph`] clone frees the backend graph on the
/// thread that owns it.
struct GraphHandle<R: Runtime> {
    id: GraphId,
    device: DeviceHandle<R::Server>,
    stream_id: StreamId,
}

impl<R: Runtime> Graph<R> {
    /// Replay the captured launch sequence — every recorded kernel re-run
    /// against the buffers it was captured with, on the stream it was captured
    /// on. Self-contained (the handle owns its device handle); no client
    /// needed.
    ///
    /// How much of the launch path this skips depends on the backend: a
    /// hardware graph (CUDA, HIP) replays as one dispatch, while a software
    /// graph (wgpu) re-encodes the recorded dispatches from prebuilt state.
    /// Either way pipeline lookup, binding resolution and metadata upload
    /// happened once, at capture.
    ///
    /// Blocking only on the enqueue: [`replay`](Self::replay) waits for the
    /// device thread to accept the dispatch and hands back what that enqueue
    /// said — an unknown or destroyed graph, a refusal — then returns without
    /// waiting for the device. A failure also leaves the graph's write set
    /// carrying it, so a read of those buffers keeps failing until a replay
    /// lands.
    ///
    /// The wait costs end-to-end throughput nothing: the device-thread work
    /// happens either way, and blocking here only stops deferring it to the
    /// next sync. What it does move is the caller-visible latency of this
    /// call, from the cost of posting to a channel to the real cost of
    /// enqueuing the pass — so a benchmark reading this column is reading
    /// latency, not throughput.
    ///
    /// # Safety
    ///
    /// The dispatch re-runs the recorded kernels against the raw device pointers
    /// captured with them; nothing validates those buffers still exist or are
    /// unshared. The caller must guarantee, until the replay's work completes on
    /// the stream:
    ///
    /// - **Liveness** — every [`Handle`] the captured kernels read or wrote is
    ///   still allocated. Freeing one returns its memory to the pool, and a
    ///   later replay reads or corrupts whatever the allocator has since placed
    ///   there.
    /// - **No concurrent use** — no other stream or thread touches buffers the
    ///   graph reads or writes while the replay executes; the replay is ordered
    ///   only against work on its capture stream.
    /// - **Same-stream refreshes** — input writes and output reads are issued on
    ///   the capture stream (keep the client pinned to it via
    ///   [`set_stream`](ComputeClient::set_stream), or do everything from the
    ///   one client), so they order against the replay instead of racing it.
    pub unsafe fn replay(&self) -> Result<(), ServerError> {
        let id = self.inner.id;
        let stream_id = self.inner.stream_id;
        self.inner
            .device
            .submit_blocking(move |server| server.replay(id, stream_id))
            .unwrap_or_resume()
    }
}

impl<R: Runtime> Clone for Graph<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<R: Runtime> Drop for GraphHandle<R> {
    fn drop(&mut self) {
        let id = self.id;
        let stream_id = self.stream_id;
        // Destroying the raw executable must happen on the server actor (the
        // only thread allowed to touch it) and only once in-flight replays have
        // completed — `replay` returns at enqueue time, not completion. Ship the
        // release to the actor; the backend syncs the stream before it destroys.
        self.device
            .submit(move |server| server.graph_destroy(id, stream_id));
    }
}

impl<R: Runtime> Clone for ComputeClient<R> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            utilities: self.utilities.clone(),
            stream_id: self.stream_id,
        }
    }
}

impl<R: Runtime> ComputeClient<R> {
    /// Get the info of the current backend.
    pub fn info(&self) -> &<R::Server as ComputeServer>::Info {
        &self.utilities.info
    }

    /// Create a new client with a new server.
    pub fn init<D: Device>(device: &D, server: R::Server) -> Self {
        let utilities = server.utilities();
        let context = DeviceHandle::<R::Server>::insert(device.to_id(), server)
            .expect("Can't create a new client on an already registered server");

        Self {
            device: context,
            utilities,
            stream_id: None,
        }
    }

    /// Load the client for the given device.
    pub fn load<D: Device>(device: &D) -> Self {
        let context = DeviceHandle::<R::Server>::new(device.to_id());

        // This is safe because we now know the return type of [`DeviceHandle::utilities()`].
        let utilities = context
            .utilities()
            .downcast::<ServerUtilities<R::Server>>()
            .expect("Can downcast to `ServerUtilities`");

        Self {
            device: context,
            utilities,
            stream_id: None,
        }
    }

    fn stream_id(&self) -> StreamId {
        match self.stream_id {
            Some(val) => val,
            None => StreamId::current(),
        }
    }

    /// Set the stream in which the current client is operating on.
    ///
    /// # Safety
    ///
    /// This is highly unsafe and should probably only be used by the CubeCL/Burn projects for now.
    pub unsafe fn set_stream(&mut self, stream_id: StreamId) {
        self.stream_id = Some(stream_id);
    }

    fn do_read(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.read(descriptors, stream_id))
            .unwrap_or_resume()
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_async(
        &self,
        handles: Vec<Handle>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send {
        let shapes = handles
            .iter()
            .map(|it| [it.size_in_used() as usize].into())
            .collect::<Vec<Shape>>();
        let descriptors = handles
            .into_iter()
            .zip(shapes)
            .map(|(handle, shape)| CopyDescriptor::new(handle.binding(), shape, [1].into(), 1))
            .collect();

        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes> {
        cubecl_environment::future::reader::read_sync(self.read_async(handles)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    pub fn read_one(&self, handle: Handle) -> Result<Bytes, ServerError> {
        Ok(cubecl_environment::future::reader::read_sync(self.read_async(vec![handle]))?.remove(0))
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails. Useful for tests.
    pub fn read_one_unchecked(&self, handle: Handle) -> Bytes {
        cubecl_environment::future::reader::read_sync(self.read_async(vec![handle]))
            .unwrap()
            .remove(0)
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_tensor_async(
        &self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send {
        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    ///
    /// The tensor must be in the same layout as created by the runtime, or more strict.
    /// Contiguous tensors are always fine, strided tensors are only ok if the stride is similar to
    /// the one created by the runtime (i.e. padded on only the last dimension). A way to check
    /// stride compatibility on the runtime will be added in the future.
    ///
    /// Also see [`ComputeClient::create_tensor`].
    pub fn read_tensor(&self, descriptors: Vec<CopyDescriptor>) -> Vec<Bytes> {
        cubecl_environment::future::reader::read_sync(self.read_tensor_async(descriptors))
            .expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_tensor_async(
        &self,
        descriptor: CopyDescriptor,
    ) -> impl Future<Output = Result<Bytes, ServerError>> + Send {
        let fut = self.read_tensor_async(vec![descriptor]);

        async { Ok(fut.await?.remove(0)) }
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_unchecked_tensor(&self, descriptor: CopyDescriptor) -> Bytes {
        self.read_tensor(vec![descriptor]).remove(0)
    }

    /// Reads the device resource described by `descriptor` lazily.
    ///
    /// The returned [`Bytes`] only performs the device-to-host copy on first access (e.g. during
    /// serialization), keeping the source allocation alive until then. This lets a large number of
    /// device tensors be serialized without materializing them all in host memory at once: drain
    /// the [`Bytes`] sequentially rather than holding them all alive.
    ///
    /// The data reflects the device state at first access, so the buffer must not be mutated
    /// between this call and the first read.
    #[cfg(not(target_family = "wasm"))]
    pub fn read_lazy(&self, descriptor: CopyDescriptor) -> Bytes {
        let len = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let controller = lazy::LazyDeviceController::new(self.clone(), Arc::new(descriptor));
        // SAFETY: the controller materializes exactly `len` bytes on first access.
        unsafe { Bytes::from_controller(alloc::boxed::Box::new(controller), len) }
    }

    /// Reads the device resource described by `descriptor` lazily, async variant.
    ///
    /// On native targets the returned future is immediately ready and yields a lazy [`Bytes`]
    /// whose device-to-host copy is deferred to first access (see [`read_lazy`](Self::read_lazy)).
    #[cfg(not(target_family = "wasm"))]
    pub fn read_lazy_async(
        &self,
        descriptor: CopyDescriptor,
    ) -> impl Future<Output = Result<Bytes, ServerError>> + Send {
        let len = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let controller = lazy::LazyDeviceController::new(self.clone(), Arc::new(descriptor));
        // SAFETY: the controller materializes exactly `len` bytes on first access.
        let bytes = unsafe { Bytes::from_controller(alloc::boxed::Box::new(controller), len) };
        core::future::ready(Ok(bytes))
    }

    /// Reads the device resource described by `descriptor` lazily, async variant.
    ///
    /// On `wasm` the deferred copy cannot run inside the synchronous access path, so awaiting
    /// performs the read eagerly and yields a materialized [`Bytes`]. Awaiting one tensor at a
    /// time still bounds peak host memory, which is the point of the lazy API.
    #[cfg(target_family = "wasm")]
    pub fn read_lazy_async(
        &self,
        descriptor: CopyDescriptor,
    ) -> impl Future<Output = Result<Bytes, ServerError>> + Send {
        self.read_one_tensor_async(descriptor)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        handle: Handle,
    ) -> Result<
        ManagedResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource>,
        ServerError,
    > {
        let stream_id = self.stream_id();
        let binding = handle.binding();

        self.device
            .submit_blocking(move |state| state.get_resource(binding, stream_id))
            .unwrap_or_resume()
    }

    fn do_create_from_slices(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        slices: Vec<Vec<u8>>,
    ) -> Vec<MemoryLayout> {
        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(stream_id, &descriptors);

        let descriptors = descriptors
            .into_iter()
            .zip(layouts.iter())
            .zip(slices)
            .map(|((desc, alloc), data)| {
                (
                    CopyDescriptor::new(
                        alloc.memory.clone().binding(),
                        desc.shape,
                        alloc.strides.clone(),
                        desc.elem_size,
                    ),
                    Bytes::from_bytes_vec(data.to_vec()),
                )
            })
            .collect::<Vec<_>>();

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(descriptors, stream_id);
        });

        layouts
    }

    fn do_create(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        data: Vec<Bytes>,
    ) -> Vec<MemoryLayout> {
        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(stream_id, &descriptors);

        let descriptors = descriptors
            .into_iter()
            .zip(layouts.iter())
            .zip(data)
            .map(|((desc, layout), data)| {
                (
                    CopyDescriptor::new(
                        layout.memory.clone().binding(),
                        desc.shape,
                        layout.strides.clone(),
                        desc.elem_size,
                    ),
                    data,
                )
            })
            .collect::<Vec<_>>();

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(descriptors, stream_id);
        });

        layouts
    }

    /// Returns a resource handle containing the given data.
    ///
    /// # Notes
    ///
    /// Prefer using the more efficient [`Self::create`] function.
    pub fn create_from_slice(&self, slice: &[u8]) -> Handle {
        let shape: Shape = [slice.len()].into();

        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![slice.to_vec()],
        )
        .remove(0)
        .memory
    }

    /// Run `task` with this device to itself, so nothing else is scheduled
    /// against it for the duration.
    ///
    /// # Errors
    ///
    /// The device could not be taken exclusively — another holder has it, or
    /// its runner is gone. Nothing ran, so the caller may retry.
    pub fn exclusive<'a, Re: Send + 'static, F: FnOnce() -> Re + Send + 'a>(
        &'a self,
        task: F,
    ) -> Result<Re, ServerError> {
        // We then launch the task.
        self.device
            .exclusive(task)
            .map_err(|err| ServerError::Generic {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    /// Run `task` with every allocation it makes routed to the persistent
    /// pool, then restore the previous mode.
    ///
    /// Persistent slices are exact-fit and are not reclaimed by the ordinary
    /// sweep, which is what weights want: allocated once, alive for the
    /// process, and stable enough for a graph capture to record against.
    pub fn memory_persistent_allocation<
        'a,
        Re: Send,
        Input: Send,
        F: FnOnce(Input) -> Re + Send + 'a,
    >(
        &'a self,
        input: Input,
        task: F,
    ) -> Re {
        let stream_id = StreamId::current();

        self.device.submit(move |server| {
            server.allocation_mode(MemoryAllocationMode::Persistent, stream_id);
        });

        // All tasks created on the same stream will have persistent memory.
        let output = task(input);

        self.device.submit(move |server| {
            server.allocation_mode(MemoryAllocationMode::Auto, stream_id);
        });

        output
    }

    /// Write `data` into an existing allocation, in place (same device pointer).
    ///
    /// This is how a captured [`Graph`]'s inputs are refreshed between replays:
    /// the graph records raw device pointers, so new input bytes must land in
    /// the very buffer the capture read from. Issue it from the capture stream
    /// (see the stream-ordering notes on [`Graph`]) so the write orders against
    /// the replays instead of racing them.
    ///
    /// Non-blocking: the write is enqueued on this client's current stream.
    pub fn write(&self, handle: &Handle, data: Bytes) {
        let stream_id = self.stream_id();
        let descriptor =
            CopyDescriptor::new(handle.clone().binding(), [data.len()].into(), [1].into(), 1);
        self.device.submit(move |server| {
            server.write(vec![(descriptor, data)], stream_id);
        });
    }

    /// Returns a resource handle containing the given [Bytes].
    pub fn create(&self, data: Bytes) -> Handle {
        let shape = [data.len()].into();

        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![data],
        )
        .remove(0)
        .memory
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensor`] for better performance.
    pub fn create_tensor_from_slice(
        &self,
        slice: &[u8],
        shape: Shape,
        elem_size: usize,
    ) -> MemoryLayout {
        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![slice.to_vec()],
        )
        .remove(0)
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    pub fn create_tensor(&self, bytes: Bytes, shape: Shape, elem_size: usize) -> MemoryLayout {
        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![bytes],
        )
        .remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensors`] for better performance.
    pub fn create_tensors_from_slices(
        &self,
        descriptors: Vec<(MemoryLayoutDescriptor, &[u8])>,
    ) -> Vec<MemoryLayout> {
        let mut data = Vec::with_capacity(descriptors.len());
        let mut descriptors_ = Vec::with_capacity(descriptors.len());
        for (a, b) in descriptors {
            data.push(b.to_vec());
            descriptors_.push(a);
        }

        self.do_create_from_slices(descriptors_, data)
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn create_tensors(
        &self,
        descriptors: Vec<(MemoryLayoutDescriptor, Bytes)>,
    ) -> Vec<MemoryLayout> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create(descriptors, data)
    }

    fn do_empty(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Vec<MemoryLayout> {
        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(stream_id, &descriptors);

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
        });

        layouts
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle {
        let shape: Shape = [size].into();
        let descriptor = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, 1);
        self.do_empty(vec![descriptor]).remove(0).memory
    }

    /// Reserves `shape` in the storage, and returns a tensor handle for it.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensor(&self, shape: Shape, elem_size: usize) -> MemoryLayout {
        let descriptor =
            MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size);
        self.do_empty(vec![descriptor]).remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensors(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Vec<MemoryLayout> {
        self.do_empty(descriptors)
    }

    /// Marks the given [Bytes] as being a staging buffer, maybe transferring it to pinned memory
    /// for faster data transfer with compute device.
    ///
    /// TODO: This blocks the compute queue, so it will drop the compute utilization.
    pub fn staging<'a, I>(&self, bytes: I, file_only: bool)
    where
        I: Iterator<Item = &'a mut Bytes>,
    {
        let has_staging = |b: &Bytes| match b.property() {
            AllocationProperty::Pinned => false,
            AllocationProperty::File => true,
            // A lazily device-backed buffer materializes on access and is staged (if needed)
            // by the backend write path, so don't force it into a host staging buffer here.
            AllocationProperty::Device => false,
            AllocationProperty::Native | AllocationProperty::Other => !file_only,
        };

        let mut to_be_updated = Vec::new();
        let sizes = bytes
            .filter_map(|b| match has_staging(b) {
                true => {
                    let len = b.len();
                    to_be_updated.push(b);
                    Some(len)
                }
                false => None,
            })
            .collect::<Vec<usize>>();

        if sizes.is_empty() {
            return;
        }

        let stream_id = self.stream_id();
        let sizes = sizes.to_vec();
        let stagings = self
            .device
            .submit_blocking(move |server| server.staging(&sizes, stream_id))
            .unwrap_or_resume();

        let stagings = match stagings {
            Ok(val) => val,
            Err(_) => return,
        };

        to_be_updated
            .into_iter()
            .zip(stagings)
            .for_each(|(b, mut staging)| {
                b.copy_into(&mut staging);
                core::mem::swap(b, &mut staging);
            });
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src, dst_server))
    )]
    pub fn to_client(&mut self, src: Handle, dst_server: &Self, dtype: ElemType) -> Handle {
        let shape = [src.size_in_used() as usize];
        let src_descriptor = src.copy_descriptor(shape.into(), [1].into(), 1);

        if R::Server::SERVER_COMM_ENABLED {
            self.to_client_tensor(src_descriptor, dst_server, dtype)
        } else {
            let alloc_desc = MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                src_descriptor.shape.clone(),
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
                .memory
        }
    }

    /// Perform an `all_reduce` operation on the given devices.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, device_ids))
    )]
    pub fn ensure_init_collective(&mut self, device_ids: Vec<DeviceId>) {
        let comm_id = CommunicationId::from(device_ids.clone());
        let is_comms_init = self.utilities.initialized_comms.read().contains(&comm_id);
        if !is_comms_init {
            self.device
                .submit(move |server| server.comm_init(device_ids).unwrap());
            let mut initialized_comms = self.utilities.initialized_comms.write();
            initialized_comms.insert(comm_id);
            // Flush immediately so other devices aren't blocked waiting on this initialization.
            self.device.flush_queue();
        }
    }

    /// Wait on the communication stream.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn sync_collective(&self) {
        if DeviceHandle::<R::Server>::is_blocking() {
            panic!("Can't use `sync_collective` with a blocking device handle");
        }
        let stream_id = self.stream_id();

        self.device.submit(move |server| {
            // Logged rather than unwrapped: a panic on the server thread is
            // reduced to a log line by the channel's catch_unwind anyway, so
            // report deliberately instead of through a swallowed unwind.
            if let Err(err) = server.sync_collective(stream_id) {
                log::error!("sync_collective failed: {err}");
            }
        });

        // We don't actually need or want to sync the server here, but we need to make sure any
        // task enqueued on the communication channel is done.
        self.device.flush_queue();
    }

    /// Perform an `all_reduce` operation on the given devices.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src, dst, dtype, device_ids, op))
    )]
    pub fn all_reduce(
        &mut self,
        src: Handle,
        dst: Handle,
        dtype: ElemType,
        device_ids: Vec<DeviceId>,
        op: ReduceOperation,
    ) {
        if DeviceHandle::<R::Server>::is_blocking() {
            panic!("Can't use `all_reduce` with a blocking device handle");
        }

        let stream_id = self.stream_id();
        let src = src.binding();
        let dst = dst.binding();

        self.ensure_init_collective(device_ids.clone());

        self.device.submit(move |server| {
            // The report lives on the buffers: a refused or failed reduce has
            // tainted the destination, so the read that consumes it fails on
            // the root cause. The log is the eager half of that report — an
            // unwrap here would only be reduced to a warn by the channel's
            // catch_unwind, with the taint doing the real work either way.
            if let Err(err) = server.all_reduce(src, dst, dtype, stream_id, op, device_ids) {
                log::error!("all_reduce failed; the destination carries the failure: {err}");
            }
        });
    }

    /// Transfer data from one client to another
    ///
    /// Make sure the source description can be read in a contiguous manner.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src_descriptor, dst_server))
    )]
    pub fn to_client_tensor(
        &mut self,
        src_descriptor: CopyDescriptor,
        dst_server: &Self,
        dtype: ElemType,
    ) -> Handle {
        let stream_id_src = self.stream_id();
        let stream_id_dst = dst_server.stream_id();

        let device_id_src = self.device.device_id();
        let device_id_dst = dst_server.device.device_id();

        let mut dst_server = dst_server.clone();
        let handle = Handle::new(stream_id_dst, src_descriptor.handle.size_in_used());
        let handle_cloned = handle.clone();

        let device_ids = vec![device_id_src, device_id_dst];
        self.ensure_init_collective(device_ids.clone());
        dst_server.ensure_init_collective(device_ids);

        self.device.submit(move |server_src| {
            // A refused send has no local buffer to answer for, so the log is
            // the whole local report. The peer's posted recv is left waiting
            // on its communication stream — the recv cannot be recalled from
            // here, and cross-device failure propagation needs a design pass
            // of its own — so the wedge is named loudly rather than hidden
            // behind a swallowed unwrap.
            if let Err(err) = server_src.send(src_descriptor, dtype, stream_id_src, device_id_dst) {
                log::error!(
                    "send to {device_id_dst:?} failed; the peer's recv is left waiting: {err}"
                );
            }
        });

        dst_server.device.submit(move |server_dst| {
            // A failed recv taints the destination handle, so the read that
            // consumes this transfer fails on the cause.
            if let Err(err) = server_dst.recv(handle_cloned, dtype, stream_id_dst, device_id_src) {
                log::error!(
                    "recv from {device_id_src:?} failed; the destination carries the failure: {err}"
                );
                return;
            }
            if let Err(err) = server_dst.sync_collective(stream_id_dst) {
                log::error!("sync_collective failed: {err}");
            }
        });

        // `ServerCommunication::send` and`ServerCommunication::recv` are blocking: they each wait for the corresponding recv/send
        // call to be made. We flush the operations right away so that the neither server ends up in a deadlock.
        // The actual data transfer is still executed asynchronously on the communication stream.
        self.device.flush_queue();
        dst_server.device.flush_queue();

        handle
    }

    #[track_caller]
    #[cfg_attr(feature = "tracing", tracing::instrument(level="trace",
        skip(self, kernel, bindings),
        fields(
            kernel.name = %kernel.name(),
            kernel.id = %kernel.id(),
        )
    ))]
    unsafe fn launch_inner(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
    ) {
        // No work, and some drivers reject a zero grid dim.
        if let CubeCount::Static(x, y, z) = &count
            && (*x == 0 || *y == 0 || *z == 0)
        {
            return;
        }

        // Decided here, on the issuing thread, because that is the only place
        // that still knows whether this launch is an autotune measurement — by
        // the time it reaches the server thread, that context is gone.
        let launch_mode = crate::dry_run::launch_mode();

        let level = self.utilities.logger.profile_level();

        // Before the submit, on the issuing thread: this is the last point at
        // which the caller's own context still exists, and attributing a
        // launch to what caused it is the whole reason the hook is here rather
        // than beside the logger's aggregation.
        crate::logging::notify_launch(kernel.name());

        // An observer asking for timing gets the profiled path even with the
        // profiling logger off — the two are separate readers of the same
        // measurement, and making one depend on the other's configuration
        // would mean a caller could not time launches without also logging
        // them somewhere it did not choose.
        let observed_timing = crate::logging::timing_wanted();

        match level {
            None | Some(ProfileLevel::ExecutionOnly) if !observed_timing => {
                let utilities = self.utilities.clone();
                self.device.submit(move |state| {
                    let name = kernel.name();
                    unsafe { state.launch(kernel, count, bindings, stream_id, launch_mode) };

                    if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                        let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                        utilities.logger.register_execution(info);
                    }
                });
            }
            level => {
                let name = kernel.name();
                let kernel_id = kernel.id();
                let context = self.device.clone();
                // The arguments travel through a slot the profiled closure
                // empties, because a profile can be refused — a graph capture
                // window refuses one on the spot — and a refusal must hand the
                // launch back: dropping a kernel because its measurement could
                // not start would turn a missing timing into a missing
                // computation.
                let slot = Arc::new(cubecl_environment::sync::Mutex::new(Some((
                    kernel,
                    count.clone(),
                    bindings,
                ))));
                let to_launch = slot.clone();
                let profiled = self.profile(
                    move || {
                        let (kernel, count, bindings) = to_launch
                            .lock()
                            .take()
                            .expect("filled right above, emptied only here");
                        context
                            .submit_blocking(move |state| unsafe {
                                state.launch(kernel, count, bindings, stream_id, launch_mode)
                            })
                            .unwrap_or_resume()
                    },
                    name,
                );
                let profile = match profiled {
                    Ok(((), profile)) => profile,
                    Err(err) => {
                        // The logger's timing levels opted into profiling and
                        // keep their loud failure. Only the observer's timing
                        // degrades: it asked for a measurement, and a refused
                        // measurement must not take the launch down with it.
                        if !matches!(level, None | Some(ProfileLevel::ExecutionOnly)) {
                            panic!("{err:?}");
                        }
                        match slot.lock().take() {
                            // The refusal came before the closure ran, so the
                            // kernel was never submitted. Launch it the way an
                            // unobserved run would have.
                            Some((kernel, count, bindings)) => {
                                let utilities = self.utilities.clone();
                                self.device.submit(move |state| {
                                    unsafe {
                                        state.launch(
                                            kernel,
                                            count,
                                            bindings,
                                            stream_id,
                                            launch_mode,
                                        )
                                    };
                                    if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                                        let info =
                                            type_name_format(name, TypeNameFormatLevel::Balanced);
                                        utilities.logger.register_execution(info);
                                    }
                                });
                            }
                            // The closure ran, so the kernel was submitted;
                            // only its measurement was lost.
                            None => {
                                if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                                    let info =
                                        type_name_format(name, TypeNameFormatLevel::Balanced);
                                    self.utilities.logger.register_execution(info);
                                }
                            }
                        }
                        log::warn!(
                            "Skipped timing a launch of `{name}` for its observer: the profile was refused ({err:?})"
                        );
                        return;
                    }
                };
                // The observer is told first, because resolving the profile
                // consumes it: the logger's copy is the one that can be
                // deferred, an observer's cannot be recovered afterwards.
                let profile = if observed_timing {
                    let method = profile.timing_method();
                    let ticks = cubecl_environment::future::block_on(profile.resolve());
                    crate::logging::notify_timed(name, ticks.duration(), method);
                    // Handed on already resolved rather than measured again:
                    // the logger and the observer are two readers of one
                    // measurement, and a second would not be the same launch.
                    ProfileDuration::new(alloc::boxed::Box::pin(async move { ticks }), method)
                } else {
                    profile
                };
                match level {
                    // An observer does not change what the logger writes.
                    // `ExecutionOnly` is documented as the kernels that ran
                    // without their timings, and it reaches here only because
                    // an observer asked for the profiled path — registering
                    // the profile would turn a log the caller configured into
                    // one it did not.
                    Some(ProfileLevel::ExecutionOnly) => {
                        let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                        self.utilities.logger.register_execution(info);
                    }
                    Some(level) => {
                        let info = match level {
                            ProfileLevel::Full => {
                                format!("{name}: {kernel_id} CubeCount {count:?}")
                            }
                            _ => type_name_format(name, TypeNameFormatLevel::Balanced),
                        };
                        self.utilities.logger.register_profiled(info, profile);
                    }
                    None => {}
                }
            }
        }
    }

    /// Launches the `kernel` with the given `bindings`.
    #[track_caller]
    pub fn launch(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
    ) {
        unsafe { self.launch_inner(kernel, count, bindings, self.stream_id()) }
    }

    /// Whether the bytes behind `handles` can be trusted, right now and with
    /// no barrier: the claim check a read makes, without the read. One lookup
    /// per handle, so a fusion layer or an autotuner can recover per tensor
    /// instead of tearing down a device.
    ///
    /// Instant means enqueue-time failures only — a compile or binding
    /// failure is visible here immediately, a device fault is not until the
    /// queue drains. [`sync_buffers`](Self::sync_buffers) is the complete
    /// answer; [`read_one`](Self::read_one) is that plus the copy.
    ///
    /// # Errors
    ///
    /// [`ServerError::Several`] naming every failure these buffers carry, each
    /// once however many carry it. The bytes are gone, so there is nothing to
    /// retry: this is the answer, not a hint.
    pub fn check<'a>(
        &self,
        handles: impl IntoIterator<Item = &'a Handle>,
    ) -> Result<(), ServerError> {
        let bindings = Self::bindings(handles);
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.check(bindings, stream_id))
            .unwrap_or_resume()
    }

    /// Flush all outstanding commands.
    pub fn flush(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();

        self.device
            .submit_blocking(move |server| server.flush(stream_id))
            .unwrap_or_resume()
    }

    /// Prepare this client's stream for a graph capture (see
    /// [`ComputeServer::graph_prepare`]) — enable the persistent pool + capture
    /// recording. Call this **before** the warmup run, then
    /// [`start_capture`](Self::start_capture) around the run to record.
    pub fn graph_prepare(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.graph_prepare(stream_id))
            .unwrap_or_resume()
    }

    /// Begin recording launches on this client's stream into a graph rather
    /// than executing them (see [`ComputeServer::begin_capture`]). Pin the
    /// client to a dedicated stream with [`set_stream`](Self::set_stream), then
    /// [`graph_prepare`](Self::graph_prepare) and warm up first.
    ///
    /// Between this and [`stop_capture`](Self::stop_capture) the window records
    /// launches and nothing else: reading, syncing or profiling the stream is
    /// refused, and so is writing to a handle — a recorded graph cannot carry a
    /// host copy, so feed fresh inputs by writing *between* replays instead. A
    /// refused write is reported late, by failing `stop_capture`, rather than
    /// handing back a graph that silently skips it. Fresh allocation inside the
    /// window is fatal on a hardware-graph backend and merely wasteful on a
    /// software-graph one, which is what the warmup run exists to avoid.
    ///
    /// Returns an error on backends without graph support.
    pub fn start_capture(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.begin_capture(stream_id))
            .unwrap_or_resume()
    }

    /// Stop recording and return the captured graph, ready to
    /// [`replay`](Graph::replay).
    pub fn stop_capture(&self) -> Result<Graph<R>, ServerError> {
        let stream_id = self.stream_id();
        let id = self
            .device
            .submit_blocking(move |server| server.end_capture(stream_id))
            .unwrap_or_resume()?;

        Ok(Graph {
            inner: Arc::new(GraphHandle {
                id,
                device: self.device.clone(),
                stream_id,
            }),
        })
    }

    /// Wait for the completion of every task in the server.
    ///
    /// The barrier alone, which also reports a device fault — the only failure
    /// left that no buffer can report. A launch failure is not this sync's to
    /// report: it lives on the buffers the launch never wrote and surfaces on
    /// any read, [`check`](Self::check) or
    /// [`sync_buffers`](Self::sync_buffers) of those.
    pub fn sync(&self) -> DynFut<Result<(), ServerError>> {
        self.sync_buffers([])
    }

    /// The barrier, and then an answer for `handles`.
    ///
    /// [`sync`](Self::sync) first, so a device fault counts, and then the
    /// claim check a read would have made — a read without the read, for the
    /// caller that needs to know its work produced something trustworthy and
    /// does not want to pull it to the host to find out.
    ///
    /// # Errors
    ///
    /// The device fault the barrier found, or [`ServerError::Several`] naming
    /// every failure these buffers carry.
    pub fn sync_buffers<'a>(
        &self,
        handles: impl IntoIterator<Item = &'a Handle>,
    ) -> DynFut<Result<(), ServerError>> {
        let stream_id = self.stream_id();
        let bindings = Self::bindings(handles);

        let fut = self
            .device
            .submit_blocking(move |server| server.sync(bindings, stream_id))
            .unwrap_or_resume();

        self.utilities.logger.profile_summary();

        fut
    }

    /// The bindings `handles` name, which is what crosses to the device
    /// thread: a `Handle` borrows, and the closure that answers for it runs
    /// somewhere else.
    fn bindings<'a>(handles: impl IntoIterator<Item = &'a Handle>) -> Vec<BufferBinding> {
        handles
            .into_iter()
            .map(|handle| handle.clone().binding())
            .collect()
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties {
        &self.utilities.properties
    }

    /// Get the features supported by the compute server.
    pub fn features(&self) -> &Features {
        &self.utilities.properties.features
    }

    /// # Warning
    ///
    /// For private use only.
    pub fn properties_mut(&mut self) -> Option<&mut DeviceProperties> {
        Arc::get_mut(&mut self.utilities).map(|state| &mut state.properties)
    }

    /// Total memory usage across all streams on this client's device.
    ///
    /// The closure iterates the server's `stream_ids()` and folds each
    /// per-stream `memory_usage(id)` with `MemoryUsage::combine`, so the
    /// result is correct regardless of which thread queries it.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.device
            .submit_blocking(move |server| {
                server
                    .stream_ids()
                    .into_iter()
                    .fold(MemoryUsage::default(), |acc, id| {
                        acc.combine(server.memory_usage(id))
                    })
            })
            .unwrap_or_resume()
    }

    /// Structured per-pool report of the **calling stream's** main GPU memory:
    /// each pool's shape, usage, and high-water marks, in allocation-routing
    /// order.
    ///
    /// The read side of a measured memory plan — install a layout with
    /// [`install_memory_pools`](Self::install_memory_pools), measure under a
    /// [`DryRun`](crate::dry_run::DryRun), cap at the observed peaks; the full
    /// cycle is on [`MemoryReport`].
    ///
    /// Unlike [`memory_usage`](Self::memory_usage), which aggregates across
    /// streams, this reads one stream: pools are per stream, and a plan is
    /// measured and installed on the stream that runs the workload.
    pub fn memory_report(&self) -> MemoryReport {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.memory_report(stream_id))
            .unwrap_or_resume()
    }

    /// Get all devices of a specific type available to this runtime
    pub fn enumerate_devices(&self, type_id: u16) -> Vec<DeviceId> {
        R::enumerate_devices(type_id, self.info())
    }

    /// Get all devices available to this runtime
    pub fn enumerate_all_devices(&self) -> Vec<DeviceId> {
        R::enumerate_all_devices(self.info())
    }

    /// Get the number of devices of a specific type available to this runtime
    pub fn device_count(&self, type_id: u16) -> usize {
        self.enumerate_devices(type_id).len()
    }

    /// Get the number of devices of a specific type available to this runtime
    pub fn device_count_total(&self) -> usize {
        self.enumerate_all_devices().len()
    }

    /// Change the memory allocation mode.
    ///
    /// # Safety
    ///
    /// This function isn't thread safe and might create memory leaks.
    pub unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        let stream_id = self.stream_id();
        self.device
            .submit(move |server| server.allocation_mode(mode, stream_id));
    }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        self.device.submit(move |server| {
            for id in server.stream_ids() {
                server.memory_cleanup(id);
            }
        });
    }

    /// Install a new dynamic-pool layout for the device's main GPU memory.
    ///
    /// This replaces the pools themselves, not just a setting they read. It
    /// lands in two places:
    ///
    /// - **The calling stream's pools are rebuilt in place**, discarding the
    ///   old ones — which is why it only happens when nothing is live in them,
    ///   and why the high-water marks in
    ///   [`memory_report`](Self::memory_report) start over.
    /// - **The layout becomes the one every stream created afterwards is
    ///   built with.** Other streams that already exist keep theirs; memory is
    ///   per stream, and rebuilding a stream this call is not synchronized
    ///   with would swap pools under its live slices.
    ///
    /// Pool layouts are a purely programmatic, runtime setting — there is no
    /// config-file pathway — sized per workload (e.g. per model, just before
    /// loading it), so install at a quiescent point such as right after
    /// unloading a model. Auxiliary pools (pinned CPU, staging, uniforms) and
    /// the persistent pool are never affected.
    ///
    /// # Errors
    ///
    /// [`PoolsInUse`](InstallMemoryPoolsError::PoolsInUse) when the current
    /// stream kept its old layout because something was still live in its
    /// pools — e.g. a garbage-collection task that has not released its
    /// cross-stream pins yet, which can lag behind an explicit
    /// [`memory_cleanup`](Self::memory_cleanup). Nothing is disturbed, the
    /// layout still applies to streams created afterwards, and retrying after
    /// the remaining work drains rebuilds the current stream too.
    ///
    /// [`Unsupported`](InstallMemoryPoolsError::Unsupported) from a runtime
    /// with no configurable pools, where retrying will never succeed.
    ///
    /// # Panics
    ///
    /// Panics if the layout is invalid (empty list, too many pools, zero page
    /// size, slice larger than page, cap smaller than page, unavailable
    /// preset) — that is a bad layout literal rather than a runtime condition,
    /// and an explicit layout that cannot be honored must not be silently
    /// replaced.
    pub fn install_memory_pools(
        &self,
        pools: &MemoryPoolsConfig,
    ) -> Result<(), InstallMemoryPoolsError> {
        let config =
            match MemoryConfiguration::default().resolve(Some(pools), &self.properties().memory) {
                Ok(config) => config,
                Err(err) => panic!("Invalid memory pools configuration: {err}"),
            };
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.install_memory_pools(config, stream_id))
            .unwrap_or_resume()
    }

    /// Measure the execution time of some inner operations.
    #[track_caller]
    pub fn profile<O: Send + 'static>(
        &self,
        func: impl FnOnce() -> O + Send,
        #[allow(unused)] func_name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        // Get the outer caller. For execute() this points straight to the
        // cube kernel. For general profiling it points to whoever calls profile.
        #[cfg(feature = "profile-tracy")]
        let location = std::panic::Location::caller();

        // Make a CPU span. If the server has system profiling this is all you need.
        #[cfg(feature = "profile-tracy")]
        let _span = tracy_client::Client::running().unwrap().span_alloc(
            None,
            func_name,
            location.file(),
            location.line(),
            0,
        );

        let stream_id = self.stream_id();

        #[cfg(feature = "profile-tracy")]
        let gpu_span = if self.utilities.properties.timing_method == TimingMethod::Device {
            let gpu_span = self
                .utilities
                .gpu_client
                .span_alloc(func_name, "profile", location.file(), location.line())
                .unwrap();
            Some(gpu_span)
        } else {
            None
        };

        let device = self.device.clone();
        #[allow(unused_mut, reason = "Used in profile-tracy")]
        let mut result = self
            .device
            .exclusive(move || {
                // We first get mut access to the server to create a token.
                // Then we free to server, since it's going to be accessed in `func()`.
                let token =
                    match device.submit_blocking(move |server| server.start_profile(stream_id)) {
                        Ok(token) => match token {
                            Ok(token) => token,
                            Err(err) => return Err(err),
                        },
                        Err(err) => {
                            return Err(ServerError::Generic {
                                reason: alloc::format!(
                                    "Can't start profiling because of a call error: {err:?}"
                                ),
                                backtrace: BackTrace::capture(),
                            });
                        }
                    };

                // We execute `func()` which will recursibly access the server.
                let out = func();

                // Finally we get the result from the token.
                let result = device
                    .submit_blocking(move |server| {
                        let mut result = server.end_profile(stream_id, token);

                        match result {
                            Ok(result) => Ok((out, result)),
                            Err(err) => Err(err),
                        }
                    })
                    .unwrap_or_resume();

                Ok(result)
            })
            .unwrap_or_resume()
            .map_err(|err| ProfileError::from(&err))?;

        #[cfg(feature = "profile-tracy")]
        if let Some(mut gpu_span) = gpu_span {
            gpu_span.end_zone();
            let epoch = self.utilities.epoch_time;
            // Add in the work to upload the timestamp data.
            result = result.map(|(o, result)| {
                (
                    o,
                    ProfileDuration::new(
                        alloc::boxed::Box::pin(async move {
                            let ticks = result.resolve().await;
                            let start_duration =
                                ticks.start_duration_since(epoch).as_nanos() as i64;
                            let end_duration = ticks.end_duration_since(epoch).as_nanos() as i64;
                            gpu_span.upload_timestamp_start(start_duration);
                            gpu_span.upload_timestamp_end(end_duration);
                            ticks
                        }),
                        TimingMethod::Device,
                    ),
                )
            });
        }

        result
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "trace",
            skip(self, src_descriptor, alloc_descriptor, dst_server)
        )
    )]
    fn change_client_sync(
        &self,
        src_descriptor: CopyDescriptor,
        alloc_descriptor: MemoryLayoutDescriptor,
        dst_server: &Self,
    ) -> MemoryLayout {
        let shape = src_descriptor.shape.clone();
        let elem_size = src_descriptor.elem_size;
        let stream_id = self.stream_id();

        let read = self
            .device
            .submit_blocking(move |server| server.read(vec![src_descriptor], stream_id))
            .unwrap_or_resume();

        let mut data = cubecl_environment::future::block_on(read).unwrap();

        let (handle_base, mut layouts) = self
            .utilities
            .layout_policy
            .apply(stream_id, &[alloc_descriptor]);
        let alloc = layouts.remove(0);

        let desc_descriptor = CopyDescriptor {
            handle: handle_base.clone().binding(),
            shape,
            strides: alloc.strides.clone(),
            elem_size,
        };

        let (size, memory) = (handle_base.size(), handle_base.memory);
        dst_server.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(vec![(desc_descriptor, data.remove(0))], stream_id)
        });

        alloc
    }

    /// Returns all vector sizes that are useful to perform optimal IO operation on the given element.
    pub fn io_optimized_vector_sizes(
        &self,
        size: usize,
    ) -> impl Iterator<Item = VectorSize> + Clone {
        let load_width = self.properties().hardware.load_width as usize;
        let size_bits = size * 8;
        let max = load_width / size_bits;
        let max = usize::min(self.properties().hardware.max_vector_size, max);

        // If the max is 8, we want to test 1, 2, 4, 8 which is log2(8) + 1.
        let num_candidates = max.trailing_zeros() + 1;

        (0..num_candidates).map(|i| 2usize.pow(i)).rev()
    }

    /// Stable per-device identity, used to key device-level measurement caches.
    fn device_key(&self) -> String {
        format!("{}_dev{}", R::name(self), self.device.device_id().index_id)
    }

    /// Calculates the maximum throughput of the device given the given config (like tensor core with certain sizes and dtypes, or just arithmetic by dtype)
    pub fn measure_throughput(
        &self,
        key: ThroughputKey,
        kernel_config: KernelConfig,
    ) -> ThroughputValue {
        let cache = ThroughputCache::get_for_device(&self.device_key());
        let mut throughputs = ThroughputBenchmarker::new(cache);
        throughputs.measure(key, kernel_config)
    }
}
