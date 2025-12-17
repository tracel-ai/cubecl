use crate::{
    DeviceProperties,
    config::{TypeNameFormatLevel, type_name_format},
    kernel::KernelMetadata,
    logging::ProfileLevel,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    runtime::Runtime,
    server::{
        Allocation, AllocationDescriptor, AllocationKind, Binding, Bindings, ComputeServer,
        CopyDescriptor, CubeCount, ExecutionError, Handle, IoError, LaunchError, ProfileError,
        ServerCommunication, ServerUtilities,
    },
    storage::{BindingResource, ComputeStorage},
};
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::DerefMut;
use cubecl_common::{
    ExecutionMode,
    bytes::{AllocationProperty, Bytes},
    device::{Device, DeviceContext},
    future::DynFut,
    profile::ProfileDuration,
};
use cubecl_ir::StorageType;

#[allow(unused)]
use cubecl_common::profile::TimingMethod;
use cubecl_common::stream_id::StreamId;

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
pub struct ComputeClient<R: Runtime> {
    context: DeviceContext<R::Server>,
    utilities: Arc<ServerUtilities<R::Server>>,
    stream_id: Option<StreamId>,
}

impl<R: Runtime> Clone for ComputeClient<R> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
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

        let context = DeviceContext::<R::Server>::insert(device, server)
            .expect("Can't create a new client on an already registered server");

        Self {
            context,
            utilities,
            stream_id: None,
        }
    }

    /// Load the client for the given device.
    pub fn load<D: Device>(device: &D) -> Self {
        let context = DeviceContext::<R::Server>::locate(device);
        let utilities = context.lock().utilities();

        Self {
            context,
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

    fn do_read(&self, descriptors: Vec<CopyDescriptor<'_>>) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let stream_id = self.stream_id();
        let mut state = self.context.lock();
        let fut = state.read(descriptors, stream_id);
        core::mem::drop(state);
        fut
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_async(
        &self,
        handles: Vec<Handle>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send {
        let strides = [1];
        let shapes = handles
            .iter()
            .map(|it| [it.size() as usize])
            .collect::<Vec<_>>();
        let bindings = handles
            .into_iter()
            .map(|it| it.binding())
            .collect::<Vec<_>>();
        let descriptors = bindings
            .into_iter()
            .zip(shapes.iter())
            .map(|(binding, shape)| CopyDescriptor::new(binding, shape, &strides, 1))
            .collect();

        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes> {
        cubecl_common::reader::read_sync(self.read_async(handles)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    pub fn read_one(&self, handle: Handle) -> Bytes {
        cubecl_common::reader::read_sync(self.read_async(vec![handle]))
            .expect("TODO")
            .remove(0)
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_tensor_async(
        &self,
        descriptors: Vec<CopyDescriptor<'_>>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send {
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
    /// Also see [ComputeClient::create_tensor].
    pub fn read_tensor(&self, descriptors: Vec<CopyDescriptor<'_>>) -> Vec<Bytes> {
        cubecl_common::reader::read_sync(self.read_tensor_async(descriptors)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    /// See [ComputeClient::read_tensor]
    pub fn read_one_tensor_async(
        &self,
        descriptor: CopyDescriptor<'_>,
    ) -> impl Future<Output = Result<Bytes, IoError>> + Send {
        let fut = self.read_tensor_async(vec![descriptor]);

        async { Ok(fut.await?.remove(0)) }
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    /// See [ComputeClient::read_tensor]
    pub fn read_one_tensor(&self, descriptor: CopyDescriptor) -> Bytes {
        self.read_tensor(vec![descriptor]).remove(0)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource> {
        let stream_id = self.stream_id();
        self.context.lock().get_resource(binding, stream_id)
    }

    fn do_create_from_slices(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        slices: Vec<&[u8]>,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut state = self.context.lock();
        let allocations = state.create(descriptors.clone(), self.stream_id())?;
        let descriptors = descriptors
            .into_iter()
            .zip(allocations.iter())
            .zip(slices)
            .map(|((desc, alloc), data)| {
                (
                    CopyDescriptor::new(
                        alloc.handle.clone().binding(),
                        desc.shape,
                        &alloc.strides,
                        desc.elem_size,
                    ),
                    Bytes::from_bytes_vec(data.to_vec()),
                )
            })
            .collect();
        let stream_id = self.stream_id();
        state.write(descriptors, stream_id)?;
        Ok(allocations)
    }

    fn do_create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        mut data: Vec<Bytes>,
    ) -> Result<Vec<Allocation>, IoError> {
        self.staging(data.iter_mut(), true);

        let mut state = self.context.lock();
        let allocations = state.create(descriptors.clone(), self.stream_id())?;
        let descriptors = descriptors
            .into_iter()
            .zip(allocations.iter())
            .zip(data)
            .map(|((desc, alloc), data)| {
                (
                    CopyDescriptor::new(
                        alloc.handle.clone().binding(),
                        desc.shape,
                        &alloc.strides,
                        desc.elem_size,
                    ),
                    data,
                )
            })
            .collect();
        let stream_id = self.stream_id();
        state.write(descriptors, stream_id)?;
        Ok(allocations)
    }

    /// Returns a resource handle containing the given data.
    ///
    /// # Notes
    ///
    /// Prefer using the more efficient [Self::create] function.
    pub fn create_from_slice(&self, slice: &[u8]) -> Handle {
        let shape = [slice.len()];

        self.do_create_from_slices(
            vec![AllocationDescriptor::new(
                AllocationKind::Contiguous,
                &shape,
                1,
            )],
            vec![slice],
        )
        .unwrap()
        .remove(0)
        .handle
    }

    /// Returns a resource handle containing the given [Bytes].
    pub fn create(&self, data: Bytes) -> Handle {
        let shape = [data.len()];

        self.do_create(
            vec![AllocationDescriptor::new(
                AllocationKind::Contiguous,
                &shape,
                1,
            )],
            vec![data],
        )
        .unwrap()
        .remove(0)
        .handle
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
    /// (also see [ComputeClient::read_tensor]).
    ///
    /// # Notes
    ///
    /// Prefer using [Self::create_tensor] for better performance.
    pub fn create_tensor_from_slice(
        &self,
        slice: &[u8],
        shape: &[usize],
        elem_size: usize,
    ) -> Allocation {
        self.do_create_from_slices(
            vec![AllocationDescriptor::new(
                AllocationKind::Optimized,
                shape,
                elem_size,
            )],
            vec![slice],
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
    /// (also see [ComputeClient::read_tensor]).
    pub fn create_tensor(&self, bytes: Bytes, shape: &[usize], elem_size: usize) -> Allocation {
        self.do_create(
            vec![AllocationDescriptor::new(
                AllocationKind::Optimized,
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
    /// See [ComputeClient::create_tensor]
    ///
    /// # Notes
    ///
    /// Prefer using [Self::create_tensors] for better performance.
    pub fn create_tensors_from_slices(
        &self,
        descriptors: Vec<(AllocationDescriptor<'_>, &[u8])>,
    ) -> Vec<Allocation> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create_from_slices(descriptors, data).unwrap()
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [ComputeClient::create_tensor]
    pub fn create_tensors(
        &self,
        descriptors: Vec<(AllocationDescriptor<'_>, Bytes)>,
    ) -> Vec<Allocation> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create(descriptors, data).unwrap()
    }

    fn do_empty(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut state = self.context.lock();
        state.create(descriptors, self.stream_id())
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle {
        let shape = [size];
        let descriptor = AllocationDescriptor::new(AllocationKind::Contiguous, &shape, 1);
        self.do_empty(vec![descriptor]).unwrap().remove(0).handle
    }

    /// Reserves `shape` in the storage, and returns a tensor handle for it.
    /// See [ComputeClient::create_tensor]
    pub fn empty_tensor(&self, shape: &[usize], elem_size: usize) -> Allocation {
        let descriptor = AllocationDescriptor::new(AllocationKind::Optimized, shape, elem_size);
        self.do_empty(vec![descriptor]).unwrap().remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, and returns the handles for them.
    /// See [ComputeClient::create_tensor]
    pub fn empty_tensors(&self, descriptors: Vec<AllocationDescriptor<'_>>) -> Vec<Allocation> {
        self.do_empty(descriptors).unwrap()
    }

    /// Marks the given [Bytes] as being a staging buffer, maybe transferring it to pinned memory
    /// for faster data transfer with compute device.
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
        let mut context = self.context.lock();
        let stagings = match context.staging(&sizes, stream_id) {
            Ok(val) => val,
            Err(_) => return,
        };
        core::mem::drop(context);

        to_be_updated
            .into_iter()
            .zip(stagings)
            .for_each(|(b, mut staging)| {
                b.copy_into(&mut staging);
                core::mem::swap(b, &mut staging);
            });
    }

    /// Transfer data from one client to another
    pub fn to_client(&self, src: Handle, dst_server: &Self) -> Allocation {
        let shape = [src.size() as usize];
        let src_descriptor = src.copy_descriptor(&shape, &[1], 1);

        if R::Server::SERVER_COMM_ENABLED {
            self.to_client_tensor(src_descriptor, dst_server)
        } else {
            let alloc_desc = AllocationDescriptor::new(
                AllocationKind::Contiguous,
                src_descriptor.shape,
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
        }
    }

    /// Transfer data from one client to another
    ///
    /// Make sure the source description can be read in a contiguous manner.
    pub fn to_client_tensor(
        &self,
        src_descriptor: CopyDescriptor<'_>,
        dst_server: &Self,
    ) -> Allocation {
        if R::Server::SERVER_COMM_ENABLED {
            let guard = self.context.lock_device_kind();
            let mut server_src = self.context.lock();
            let mut server_dst = dst_server.context.lock();

            let copied = R::Server::copy(
                server_src.deref_mut(),
                server_dst.deref_mut(),
                src_descriptor,
                self.stream_id(),
                dst_server.stream_id(),
            )
            .unwrap();
            core::mem::drop(server_src);
            core::mem::drop(server_dst);
            core::mem::drop(guard);
            copied
        } else {
            let alloc_desc = AllocationDescriptor::new(
                AllocationKind::Optimized,
                src_descriptor.shape,
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
        }
    }

    #[track_caller]
    unsafe fn launch_inner(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let level = self.utilities.logger.profile_level();

        match level {
            None | Some(ProfileLevel::ExecutionOnly) => {
                let mut state = self.context.lock();
                let name = kernel.name();

                let result = unsafe { state.launch(kernel, count, bindings, mode, stream_id) };

                if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                    let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                    self.utilities.logger.register_execution(info);
                }
                result
            }
            Some(level) => {
                let name = kernel.name();
                let kernel_id = kernel.id();
                let (result, profile) = self
                    .profile(
                        || unsafe {
                            let mut state = self.context.lock();
                            state.launch(kernel, count.clone(), bindings, mode, stream_id)
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
        bindings: Bindings,
    ) -> Result<(), LaunchError> {
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
        bindings: Bindings,
    ) -> Result<(), LaunchError> {
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
    pub fn flush(&self) {
        let stream_id = self.stream_id();
        self.context.lock().flush(stream_id)
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self) -> DynFut<Result<(), ExecutionError>> {
        let stream_id = self.stream_id();
        let mut state = self.context.lock();
        let fut = state.sync(stream_id);
        core::mem::drop(state);
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
    pub fn memory_usage(&self) -> MemoryUsage {
        self.context.lock().memory_usage(self.stream_id())
    }

    /// Change the memory allocation mode.
    ///
    /// # Safety
    ///
    /// This function isn't thread safe and might create memory leaks.
    pub unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        self.context.lock().allocation_mode(mode, self.stream_id())
    }

    /// Use a persistent memory strategy to execute the provided function.
    ///
    /// # Notes
    ///
    /// - Using that memory strategy is beneficial for stating model parameters and similar workflows.
    /// - You can call [Self::memory_cleanup()] if you want to free persistent memory.
    pub fn memory_persistent_allocation<Input, Output, Func: Fn(Input) -> Output>(
        &self,
        input: Input,
        func: Func,
    ) -> Output {
        let device_guard = self.context.lock_device();

        self.context
            .lock()
            .allocation_mode(MemoryAllocationMode::Persistent, self.stream_id());

        let output = func(input);

        self.context
            .lock()
            .allocation_mode(MemoryAllocationMode::Auto, self.stream_id());

        core::mem::drop(device_guard);

        output
    }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        self.context.lock().memory_cleanup(self.stream_id())
    }

    /// Measure the execution time of some inner operations.
    #[track_caller]
    pub fn profile<O>(
        &self,
        func: impl FnOnce() -> O,
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

        let device_guard = self.context.lock_device();

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

        let token = self.context.lock().start_profile(self.stream_id());

        let out = func();

        #[allow(unused_mut, reason = "Used in profile-tracy")]
        let mut result = self.context.lock().end_profile(self.stream_id(), token);

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
        core::mem::drop(device_guard);

        match result {
            Ok(result) => Ok((out, result)),
            Err(err) => Err(err),
        }
    }

    /// Transfer data from one client to another
    fn change_client_sync(
        &self,
        src_descriptor: CopyDescriptor<'_>,
        alloc_descriptor: AllocationDescriptor<'_>,
        dst_server: &Self,
    ) -> Allocation {
        let shape = src_descriptor.shape;
        let elem_size = src_descriptor.elem_size;
        let stream_id = self.stream_id();

        // Allocate destination
        let alloc = dst_server
            .context
            .lock()
            .create(vec![alloc_descriptor], self.stream_id())
            .unwrap()
            .remove(0);

        let read = self.context.lock().read(vec![src_descriptor], stream_id);
        let mut data = cubecl_common::future::block_on(read).unwrap();

        let desc_descriptor = CopyDescriptor {
            binding: alloc.handle.clone().binding(),
            shape,
            strides: &alloc.strides,
            elem_size,
        };

        dst_server
            .context
            .lock()
            .write(vec![(desc_descriptor, data.remove(0))], stream_id)
            .unwrap();

        alloc
    }

    /// Returns all line sizes that are useful to perform optimal IO operation on the given element.
    pub fn io_optimized_line_sizes(&self, elem: &StorageType) -> impl Iterator<Item = u8> + Clone {
        let load_width = self.properties().hardware.load_width as usize;
        let max = (load_width / elem.size_bits()) as u8;
        let supported = R::supported_line_sizes();
        supported.iter().filter(move |v| **v <= max).cloned()
    }

    /// Returns all line sizes that are useful to perform optimal IO operation on the given element.
    /// Ignores native support, and allows all line sizes. This means the returned size may be
    /// unrolled, and may not support dynamic indexing.
    pub fn io_optimized_line_sizes_unchecked(
        &self,
        size: usize,
    ) -> impl Iterator<Item = u8> + Clone {
        let load_width = self.properties().hardware.load_width as usize;
        let size_bits = size * 8;
        let max = load_width / size_bits;
        // This makes this effectively the same as checked, if it doesn't work it's a problem with
        // unroll that should be investigated instead. But separate PR.
        let max = usize::min(R::max_global_line_size() as usize, max);

        // If the max is 8, we want to test 1, 2, 4, 8 which is log2(8) + 1.
        let num_candidates = max.trailing_zeros() + 1;

        (0..num_candidates).map(|i| 2u8.pow(i)).rev()
    }
}
