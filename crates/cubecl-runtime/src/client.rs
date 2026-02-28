use crate::{
    config::{TypeNameFormatLevel, type_name_format},
    kernel::KernelMetadata,
    logging::ProfileLevel,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    runtime::Runtime,
    server::{
        ComputeServer, CopyDescriptor, CubeCount, ExecutionMode, Handle, HandleId, IoError,
        KernelArguments, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy,
        MemoryLayoutStrategy, ProfileError, ServerCommunication, ServerError, ServerUtilities,
    },
    storage::{ComputeStorage, ManagedResource},
};
use alloc::{boxed::Box, format, sync::Arc, vec, vec::Vec};
use cubecl_common::{
    backtrace::BackTrace,
    bytes::{AllocationProperty, Bytes},
    device::Device,
    device_handle::DeviceHandle,
    future::DynFut,
    profile::ProfileDuration,
};
use cubecl_ir::{DeviceProperties, LineSize};
use cubecl_zspace::Shape;

#[cfg(feature = "profile-tracy")]
use alloc::boxed::Box;

#[allow(unused)]
use cubecl_common::profile::TimingMethod;
use cubecl_common::stream_id::StreamId;

/// The `ComputeClient` is the entry point to require tasks from the `ComputeServer`.
/// It should be obtained for a specific device via the Compute struct.
pub struct ComputeClient<R: Runtime> {
    device: DeviceHandle<R::Server>,
    utilities: Arc<ServerUtilities<R::Server>>,
    stream_id: Option<StreamId>,
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
        let utilities = context.submit_blocking(|state| state.utilities()).unwrap();

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
            .unwrap()
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_async(
        &self,
        handles: Vec<Handle<R>>,
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
    pub fn read(&self, handles: Vec<Handle<R>>) -> Vec<Bytes> {
        cubecl_common::reader::read_sync(self.read_async(handles)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    pub fn read_one(&self, handle: Handle<R>) -> Result<Bytes, ServerError> {
        Ok(cubecl_common::reader::read_sync(self.read_async(vec![handle]))?.remove(0))
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails. Useful for tests.
    pub fn read_one_unchecked(&self, handle: Handle<R>) -> Bytes {
        cubecl_common::reader::read_sync(self.read_async(vec![handle]))
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
        cubecl_common::reader::read_sync(self.read_tensor_async(descriptors)).expect("TODO")
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

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        handle: Handle<R>,
    ) -> Result<
        ManagedResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource>,
        ServerError,
    > {
        let stream_id = self.stream_id();
        let binding = handle.binding();

        self.device
            .submit_blocking(move |state| state.get_resource(binding, stream_id))
            .unwrap()
    }

    fn do_create_from_slices(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        slices: Vec<Vec<u8>>,
    ) -> Result<Vec<MemoryLayout<R>>, IoError> {
        let stream_id = self.stream_id();
        let layouts = descriptors
            .iter()
            .map(|descriptor| {
                self.utilities
                    .layout_policy
                    .apply(self.clone(), stream_id, descriptor)
            })
            .collect::<Vec<_>>();

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

        self.device.submit(move |server| {
            server.initialize_bindings(
                descriptors
                    .iter()
                    .map(|(desc, _bytes)| desc.handle.clone())
                    .collect(),
                stream_id,
            );
            server.write(descriptors, stream_id);
        });

        Ok(layouts)
    }

    fn do_create(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        mut data: Vec<Bytes>,
    ) -> Result<Vec<MemoryLayout<R>>, IoError> {
        self.staging(data.iter_mut(), true);

        let stream_id = self.stream_id();
        let layouts = descriptors
            .iter()
            .map(|descriptor| {
                self.utilities
                    .layout_policy
                    .apply(self.clone(), stream_id, descriptor)
            })
            .collect::<Vec<_>>();

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
                    Bytes::from_bytes_vec(data.to_vec()),
                )
            })
            .collect::<Vec<_>>();
        let handles = layouts
            .iter()
            .map(|desc| desc.memory.clone().binding())
            .collect();

        self.device.submit(move |server| {
            server.initialize_bindings(handles, stream_id);
            server.write(descriptors, stream_id);
        });

        Ok(layouts)
    }

    /// Returns a resource handle containing the given data.
    ///
    /// # Notes
    ///
    /// Prefer using the more efficient [`Self::create`] function.
    pub fn create_from_slice(&self, slice: &[u8]) -> Handle<R> {
        let shape: Shape = [slice.len()].into();

        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![slice.to_vec()],
        )
        .unwrap()
        .remove(0)
        .memory
    }

    /// Executes a task that has exclusive access to the current device.
    pub fn exclusive<Re: Send + 'static, F: FnOnce() -> Re + Send + 'static>(
        &self,
        task: F,
    ) -> Result<Re, ServerError> {
        // We first flush current tasks enqueued on the device.
        self.flush()?;

        // We then launch the task.
        self.device
            .exclusive(task)
            .map_err(|err| ServerError::ServerUnhealthy {
                reason: format!("Communication channel with the server is down: {err:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    /// todo: docs
    pub fn scoped<'a, Re: Send, F: FnOnce() -> Re + Send + 'a>(
        &'a self,
        task: F,
    ) -> Result<Re, ServerError> {
        // We then launch the task.
        self.device
            .exclusive_scoped(task)
            .map_err(|err| ServerError::ServerUnhealthy {
                reason: format!("Communication channel with the server is down: {err:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    /// dodo: Docs
    pub fn memory_persistent_allocation<
        'a,
        Re: Send,
        Input: Send,
        F: FnOnce(Input) -> Re + Send + 'a,
    >(
        &'a self,
        input: Input,
        task: F,
    ) -> Result<Re, ServerError> {
        // We then launch the task.
        self.device
            .exclusive_scoped(move || task(input))
            .map_err(|err| ServerError::ServerUnhealthy {
                reason: format!("Communication channel with the server is down: {err:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    /// Returns a resource handle containing the given [Bytes].
    pub fn create(&self, data: Bytes) -> Handle<R> {
        let shape = [data.len()].into();

        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![data],
        )
        .unwrap()
        .remove(0)
        .memory
    }

    /// Free a handle.
    pub(crate) fn free(&self, handle: HandleId, stream_id: StreamId) {
        self.device
            .submit(move |server| server.free(handle, stream_id));
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
    ) -> MemoryLayout<R> {
        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![slice.to_vec()],
        )
        .unwrap()
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
    pub fn create_tensor(&self, bytes: Bytes, shape: Shape, elem_size: usize) -> MemoryLayout<R> {
        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![bytes],
        )
        .unwrap()
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
    ) -> Vec<MemoryLayout<R>> {
        let mut data = Vec::with_capacity(descriptors.len());
        let mut descriptors_ = Vec::with_capacity(descriptors.len());
        for (a, b) in descriptors {
            data.push(b.to_vec());
            descriptors_.push(a);
        }

        self.do_create_from_slices(descriptors_, data).unwrap()
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn create_tensors(
        &self,
        descriptors: Vec<(MemoryLayoutDescriptor, Bytes)>,
    ) -> Vec<MemoryLayout<R>> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create(descriptors, data).unwrap()
    }

    fn do_empty(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
    ) -> Result<Vec<MemoryLayout<R>>, IoError> {
        let stream_id = self.stream_id();
        let layouts = descriptors
            .iter()
            .map(|descriptor| {
                self.utilities
                    .layout_policy
                    .apply(self.clone(), stream_id, descriptor)
            })
            .collect::<Vec<_>>();
        let bindings = layouts
            .iter()
            .map(|desc| desc.memory.clone().binding())
            .collect();

        self.device.submit(move |server| {
            server.initialize_bindings(bindings, stream_id);
        });

        Ok(layouts)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle<R> {
        let shape: Shape = [size].into();
        let descriptor = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, 1);
        self.do_empty(vec![descriptor]).unwrap().remove(0).memory
    }

    /// Reserves `shape` in the storage, and returns a tensor handle for it.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensor(&self, shape: Shape, elem_size: usize) -> MemoryLayout<R> {
        let descriptor =
            MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size);
        self.do_empty(vec![descriptor]).unwrap().remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensors(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Vec<MemoryLayout<R>> {
        self.do_empty(descriptors).unwrap()
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
            .unwrap();

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
    pub fn to_client(&self, src: Handle<R>, dst_server: &Self) -> Handle<R> {
        let shape = [src.size_in_used() as usize];
        let src_descriptor = src.copy_descriptor(shape.into(), [1].into(), 1);

        if R::Server::SERVER_COMM_ENABLED {
            self.to_client_tensor(src_descriptor, dst_server)
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

    /// Transfer data from one client to another
    ///
    /// Make sure the source description can be read in a contiguous manner.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src_descriptor, dst_server))
    )]
    pub fn to_client_tensor(&self, src_descriptor: CopyDescriptor, dst_server: &Self) -> Handle<R> {
        if R::Server::SERVER_COMM_ENABLED {
            let stream_id_src = self.stream_id();
            let stream_id_dst = dst_server.stream_id();

            let dst_server = dst_server.clone();
            let handle = Handle::new(
                self.clone(),
                stream_id_dst,
                src_descriptor.handle.size_in_used(),
            );
            let binding = handle.clone().binding();
            // TODO: This should be made in a non-blocking API.
            self.device
                .submit_blocking_scoped(move |server_src| {
                    dst_server.device.submit_blocking_scoped(|server_dst| {
                        R::Server::copy(
                            binding,
                            server_src,
                            server_dst,
                            src_descriptor,
                            stream_id_src,
                            stream_id_dst,
                        )
                    })
                })
                .unwrap();

            handle
        } else {
            let alloc_desc = MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                src_descriptor.shape.clone(),
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
                .memory
        }
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
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let level = self.utilities.logger.profile_level();

        match level {
            None | Some(ProfileLevel::ExecutionOnly) => {
                let utilities = self.utilities.clone();
                self.device.submit(move |state| {
                    let name = kernel.name();
                    unsafe { state.launch(kernel, count, bindings, mode, stream_id) };

                    if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                        let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                        utilities.logger.register_execution(info);
                    }
                });
            }
            Some(level) => {
                let name = kernel.name();
                let kernel_id = kernel.id();
                let context = self.device.clone();
                let count_moved = count.clone();
                let (result, profile) = self
                    .profile(
                        move || {
                            context
                                .submit_blocking(move |state| unsafe {
                                    state.launch(kernel, count_moved, bindings, mode, stream_id)
                                })
                                .unwrap()
                        },
                        name,
                    )
                    .unwrap();
                let info = match level {
                    ProfileLevel::Full => {
                        format!("{name}: {kernel_id} CubeCount {count:?}")
                    }
                    _ => type_name_format(name, TypeNameFormatLevel::Balanced),
                };
                self.utilities.logger.register_profiled(info, profile);
                result
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
        // SAFETY: Using checked execution mode.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                ExecutionMode::Checked,
                self.stream_id(),
            )
        }
    }

    /// Launches the `kernel` with the given `bindings` without performing any bound checks.
    ///
    /// # Safety
    ///
    /// To ensure this is safe, you must verify your kernel:
    /// - Has no out-of-bound reads and writes that can happen.
    /// - Has no infinite loops that might never terminate.
    #[track_caller]
    pub unsafe fn launch_unchecked(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
    ) {
        // SAFETY: Caller has to uphold kernel being safe.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                ExecutionMode::Unchecked,
                self.stream_id(),
            )
        }
    }

    /// Flush all outstanding commands.
    pub fn flush(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.flush(stream_id))
            .unwrap()
    }

    /// Flush all outstanding errors.
    pub fn flush_errors(&self) -> Vec<ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| {
                let _result = server.flush(stream_id);
                server.flush_errors(stream_id)
            })
            .unwrap()
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self) -> DynFut<Result<(), ServerError>> {
        let stream_id = self.stream_id();
        let fut = self
            .device
            .submit_blocking(move |server| server.sync(stream_id))
            .unwrap();

        self.utilities.logger.profile_summary();

        fut
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties {
        &self.utilities.properties
    }

    /// # Warning
    ///
    /// For private use only.
    pub fn properties_mut(&mut self) -> Option<&mut DeviceProperties> {
        Arc::get_mut(&mut self.utilities).map(|state| &mut state.properties)
    }

    /// Get the current memory usage of this client.
    pub fn memory_usage(&self) -> Result<MemoryUsage, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.memory_usage(stream_id))
            .unwrap()
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

    // TODO: Remove that or rework for performance.
    //
    // /// Use a persistent memory strategy to execute the provided function.
    // ///
    // /// # Notes
    // ///
    // /// - Using that memory strategy is beneficial for stating model parameters and similar workflows.
    // /// - You can call [`Self::memory_cleanup()`] if you want to free persistent memory.
    // pub fn memory_persistent_allocation<
    //     Input: Send + 'static,
    //     Output: Send + 'static,
    //     Func: Fn(Input) -> Output + Send + 'static,
    // >(
    //     &self,
    //     input: Input,
    //     func: Func,
    // ) -> Output {
    //     let stream_id = self.stream_id();
    //     self.device
    //         .submit_blocking(move |server| {
    //             server.allocation_mode(MemoryAllocationMode::Persistent, stream_id);
    //             let output = func(input);
    //             server.allocation_mode(MemoryAllocationMode::Auto, stream_id);
    //             output
    //         })
    //         .unwrap()
    // }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        let stream_id = self.stream_id();
        self.device
            .submit(move |server| server.memory_cleanup(stream_id));
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
        let result = self
            .device
            .exclusive_scoped(move || {
                // We first get mut access to the server to create a token.
                // Then we free to server, since it's going to be accessed in `func()`.
                let token = device
                    .submit_blocking(move |server| server.start_profile(stream_id))
                    .unwrap();

                let token = match token {
                    Ok(token) => token,
                    Err(err) => {
                        // This should never happened in general since we check after every
                        // profiling is the state is OK.
                        //
                        // But if for some obsure reason the state of the server isn't right before
                        // we start a profiling, we clear the errors and log them, hoping that
                        // future executions won't be affected by it.
                        let token = device
                            .submit_blocking(move |server| {
                                let errors = server.flush_errors(stream_id);
                                if !errors.is_empty() {
                                    log::warn!("An error hapenned while profiling: {err}\nResetted server error state: {errors:?}");
                                }
                                server.start_profile(stream_id)
                            })
                            .unwrap();

                        token.expect("The state of the server to be healthy after flushing.")
                    }
                };

                // We execute `func()` which will recursibly access the server.
                let out = func();

                // Finaly we get the result from the token.
                device
                    .submit_blocking(move |server| {
                        #[allow(unused_mut, reason = "Used in profile-tracy")]
                        let mut result = server.end_profile(stream_id, token);

                        // Better be safe than story, we validate the state of the server after the
                        // profiling. If the state is in errors, we free the server from those
                        // errors and make the profiling fail even if the result is sucessful,
                        // since we can't trust durations profiled from a server in an invalid
                        // state.
                        let errors = server.flush_errors(stream_id);

                        if !errors.is_empty() {
                            log::warn!("Resetted server error state: {errors:?}");
                            if result.is_ok() {
                                result = Err(ProfileError::Server(Box::new(ServerError::ServerUnhealthy { reason: format!("Server error state: {errors:?}"), backtrace: BackTrace::capture() })));
                            }
                        }

                        match result {
                            Ok(result) => Ok((out, result)),
                            Err(err) => Err(err),
                        }
                    })
                    .unwrap()
            })
            .unwrap();

        #[cfg(feature = "profile-tracy")]
        if let Some(mut gpu_span) = gpu_span {
            gpu_span.end_zone();
            let epoch = self.utilities.epoch_time;
            // Add in the work to upload the timestamp data.
            result = result.map(|result| {
                ProfileDuration::new(
                    Box::pin(async move {
                        let ticks = result.resolve().await;
                        let start_duration = ticks.start_duration_since(epoch).as_nanos() as i64;
                        let end_duration = ticks.end_duration_since(epoch).as_nanos() as i64;
                        gpu_span.upload_timestamp_start(start_duration);
                        gpu_span.upload_timestamp_end(end_duration);
                        ticks
                    }),
                    TimingMethod::Device,
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
    ) -> MemoryLayout<R> {
        let shape = src_descriptor.shape.clone();
        let elem_size = src_descriptor.elem_size;
        let stream_id = self.stream_id();

        let read = self
            .device
            .submit_blocking(move |server| server.read(vec![src_descriptor], stream_id))
            .unwrap();

        let mut data = cubecl_common::future::block_on(read).unwrap();

        let alloc = self
            .utilities
            .layout_policy
            .apply(self.clone(), stream_id, &alloc_descriptor);
        let handle_binding = alloc.memory.clone().binding();
        let desc_descriptor = CopyDescriptor {
            handle: handle_binding.clone(),
            shape,
            strides: alloc.strides.clone(),
            elem_size,
        };

        dst_server.device.submit(move |server| {
            server.initialize_bindings(vec![handle_binding], stream_id);
            server.write(vec![(desc_descriptor, data.remove(0))], stream_id)
        });

        alloc
    }

    /// Returns all line sizes that are useful to perform optimal IO operation on the given element.
    pub fn io_optimized_line_sizes(&self, size: usize) -> impl Iterator<Item = LineSize> + Clone {
        let load_width = self.properties().hardware.load_width as usize;
        let size_bits = size * 8;
        let max = load_width / size_bits;
        let max = usize::min(self.properties().hardware.max_line_size, max);

        // If the max is 8, we want to test 1, 2, 4, 8 which is log2(8) + 1.
        let num_candidates = max.trailing_zeros() + 1;

        (0..num_candidates).map(|i| 2usize.pow(i)).rev()
    }
}
