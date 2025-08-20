use crate::{
    channel::ComputeChannel, config::{type_name_format, TypeNameFormatLevel}, kernel::KernelMetadata, logging::{ProfileLevel, ServerLogger}, memory_management::{MemoryAllocationMode, MemoryUsage}, server::{
        Allocation, AllocationDescriptor, AllocationKind, Binding, Bindings, ComputeServer,
        CopyDescriptor, CubeCount, Handle, IoError, ProfileError,
    }, storage::{BindingResource, ComputeStorage}, transfer::ComputeDataTransferId, DeviceProperties
};
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::{ExecutionMode, profile::ProfileDuration};

#[allow(unused)]
use cubecl_common::profile::TimingMethod;

#[cfg(multi_threading)]
use cubecl_common::stream_id::StreamId;

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
pub struct ComputeClient<Server: ComputeServer, Channel> {
    channel: Channel,
    state: Arc<ComputeClientState<Server>>,
}

#[derive(new)]
struct ComputeClientState<Server: ComputeServer> {
    #[cfg(feature = "profile-tracy")]
    epoch_time: web_time::Instant,

    #[cfg(feature = "profile-tracy")]
    gpu_client: tracy_client::GpuContext,

    properties: DeviceProperties<Server::Feature>,
    info: Server::Info,
    logger: Arc<ServerLogger>,

    #[cfg(multi_threading)]
    current_profiling: spin::RwLock<Option<StreamId>>,
}

impl<S, C> Clone for ComputeClient<S, C>
where
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
            state: self.state.clone(),
        }
    }
}

impl<Server, Channel> ComputeClient<Server, Channel>
where
    Server: ComputeServer,
    Channel: ComputeChannel<Server>,
{
    /// Get the info of the current backend.
    pub fn info(&self) -> &Server::Info {
        &self.state.info
    }

    /// Create a new client.
    pub fn new(
        channel: Channel,
        properties: DeviceProperties<Server::Feature>,
        info: Server::Info,
    ) -> Self {
        let logger = ServerLogger::default();

        // Start a tracy client if needed.
        #[cfg(feature = "profile-tracy")]
        let client = tracy_client::Client::start();

        let state = ComputeClientState {
            properties,
            logger: Arc::new(logger),
            #[cfg(multi_threading)]
            current_profiling: spin::RwLock::new(None),
            // Create the GPU client if needed.
            #[cfg(feature = "profile-tracy")]
            gpu_client: client
                .clone()
                .new_gpu_context(
                    Some(&format!("{info:?}")),
                    // In the future should ask the server what makes sense here. 'Invalid' atm is a generic stand-in (Tracy doesn't have CUDA/RocM atm anyway).
                    tracy_client::GpuContextType::Invalid,
                    0,   // Timestamps are manually aligned to this epoch so start at 0.
                    1.0, // Timestamps are manually converted to be nanoseconds so period is 1.
                )
                .unwrap(),
            #[cfg(feature = "profile-tracy")]
            epoch_time: web_time::Instant::now(),
            info,
        };

        Self {
            channel,
            state: Arc::new(state),
        }
    }

    async fn do_read(&self, descriptors: Vec<CopyDescriptor<'_>>) -> Result<Vec<Vec<u8>>, IoError> {
        self.profile_guard();

        self.channel.read(descriptors).await
    }

    /// Given bindings, returns owned resources as bytes.
    pub async fn read_async(&self, handles: Vec<Handle>) -> Vec<Vec<u8>> {
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

        self.do_read(descriptors).await.unwrap()
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    pub fn read(&self, handles: Vec<Handle>) -> Vec<Vec<u8>> {
        cubecl_common::reader::read_sync(self.read_async(handles))
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    pub fn read_one(&self, handle: Handle) -> Vec<u8> {
        cubecl_common::reader::read_sync(self.read_async(vec![handle])).remove(0)
    }

    /// Given bindings, returns owned resources as bytes.
    pub async fn read_tensor_async(&self, descriptors: Vec<CopyDescriptor<'_>>) -> Vec<Vec<u8>> {
        self.do_read(descriptors).await.unwrap()
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
    pub fn read_tensor(&self, descriptors: Vec<CopyDescriptor<'_>>) -> Vec<Vec<u8>> {
        cubecl_common::reader::read_sync(self.read_tensor_async(descriptors))
    }

    /// Given a binding, returns owned resource as bytes.
    /// See [ComputeClient::read_tensor]
    pub async fn read_one_tensor_async(&self, descriptor: CopyDescriptor<'_>) -> Vec<u8> {
        self.read_tensor_async(vec![descriptor]).await.remove(0)
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    /// See [ComputeClient::read_tensor]
    pub fn read_one_tensor(&self, descriptor: CopyDescriptor) -> Vec<u8> {
        self.read_tensor(vec![descriptor]).remove(0)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        self.profile_guard();

        self.channel.get_resource(binding)
    }

    fn do_create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        data: Vec<&[u8]>,
    ) -> Result<Vec<Allocation>, IoError> {
        self.profile_guard();

        let allocations = self.channel.create(descriptors.clone())?;
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
        self.channel.write(descriptors)?;
        Ok(allocations)
    }

    /// Given a resource, stores it and returns the resource handle.
    pub fn create(&self, data: &[u8]) -> Handle {
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
    pub fn create_tensor(&self, data: &[u8], shape: &[usize], elem_size: usize) -> Allocation {
        self.do_create(
            vec![AllocationDescriptor::new(
                AllocationKind::Optimized,
                shape,
                elem_size,
            )],
            vec![data],
        )
        .unwrap()
        .remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [ComputeClient::create_tensor]
    pub fn create_tensors(
        &self,
        descriptors: Vec<(AllocationDescriptor<'_>, &[u8])>,
    ) -> Vec<Allocation> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create(descriptors, data).unwrap()
    }

    fn do_empty(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError> {
        self.profile_guard();

        self.channel.create(descriptors)
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

    pub fn to_client(&self, src: Handle, dst_server: Self) -> Allocation {
        let strides = [1];
        let size = src.size() as usize;
        let shape = [size];
        let descriptor = src.copy_descriptor(&shape, &strides, 1);

        let id = ComputeDataTransferId::new();
        
        self.channel.send_to_peer(id, descriptor).unwrap();

        let alloc_desc = AllocationDescriptor::new(AllocationKind::Contiguous, &shape, 1);
        let alloc = self.channel.create(vec![alloc_desc]).unwrap().remove(0);
        let cpy_desc = CopyDescriptor::new(
            alloc.handle.clone().binding(),
            alloc_desc.shape,
            &alloc.strides,
            alloc_desc.elem_size,
        );

        dst_server.channel.recv_from_peer(id, cpy_desc).unwrap();

        alloc
    }

    #[track_caller]
    unsafe fn execute_inner(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
    ) {
        let level = self.state.logger.profile_level();

        match level {
            None | Some(ProfileLevel::ExecutionOnly) => {
                self.profile_guard();

                let name = kernel.name();

                unsafe {
                    self.channel
                        .execute(kernel, count, bindings, mode, self.state.logger.clone())
                };

                if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                    let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                    self.state.logger.register_execution(info);
                }
            }
            Some(level) => {
                let name = kernel.name();
                let kernel_id = kernel.id();
                let profile = self
                    .profile(
                        || unsafe {
                            self.channel.execute(
                                kernel,
                                count.clone(),
                                bindings,
                                mode,
                                self.state.logger.clone(),
                            )
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
                self.state.logger.register_profiled(info, profile);
            }
        }
    }

    /// Executes the `kernel` over the given `bindings`.
    #[track_caller]
    pub fn execute(&self, kernel: Server::Kernel, count: CubeCount, bindings: Bindings) {
        // SAFETY: Using checked execution mode.
        unsafe {
            self.execute_inner(kernel, count, bindings, ExecutionMode::Checked);
        }
    }

    /// Executes the `kernel` over the given `bindings` without performing any bound checks.
    ///
    /// # Safety
    ///
    /// To ensure this is safe, you must verify your kernel:
    /// - Has no out-of-bound reads and writes that can happen.
    /// - Has no infinite loops that might never terminate.
    #[track_caller]
    pub unsafe fn execute_unchecked(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
    ) {
        // SAFETY: Caller has to uphold kernel being safe.
        unsafe {
            self.execute_inner(kernel, count, bindings, ExecutionMode::Unchecked);
        }
    }

    /// Flush all outstanding commands.
    pub fn flush(&self) {
        self.profile_guard();

        self.channel.flush();
    }

    /// Wait for the completion of every task in the server.
    pub async fn sync(&self) {
        self.profile_guard();

        self.channel.sync().await;
        self.state.logger.profile_summary();
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties<Server::Feature> {
        &self.state.properties
    }

    /// # Warning
    ///
    /// For private use only.
    pub fn properties_mut(&mut self) -> Option<&mut DeviceProperties<Server::Feature>> {
        Arc::get_mut(&mut self.state).map(|state| &mut state.properties)
    }

    /// Get the current memory usage of this client.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.profile_guard();

        self.channel.memory_usage()
    }

    /// Change the memory allocation mode.
    ///
    /// # Safety
    ///
    /// This function isn't thread safe and might create memory leaks.
    pub unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        self.profile_guard();

        self.channel.allocation_mode(mode)
    }

    /// Use a static memory strategy to execute the provided function.
    ///
    /// # Notes
    ///
    /// Using that memory strategy is beneficial for weights loading and similar workflows.
    /// However make sure to call [Self::memory_cleanup()] if you want to free the allocated
    /// memory.
    pub fn memory_static_allocation<Input, Output, Func: Fn(Input) -> Output>(
        &self,
        input: Input,
        func: Func,
    ) -> Output {
        // We use the same profiling lock to make sure no other task is currently using the current
        // device. Meaning that the current static memory strategy will only be used for the
        // provided function.

        #[cfg(multi_threading)]
        let stream_id = self.profile_acquire();

        self.channel.allocation_mode(MemoryAllocationMode::Static);
        let output = func(input);
        self.channel.allocation_mode(MemoryAllocationMode::Auto);

        #[cfg(multi_threading)]
        self.profile_release(stream_id);

        output
    }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        self.profile_guard();

        self.channel.memory_cleanup()
    }

    /// Measure the execution time of some inner operations.
    #[track_caller]
    pub fn profile<O>(
        &self,
        func: impl FnOnce() -> O,
        #[allow(unused)] func_name: &str,
    ) -> Result<ProfileDuration, ProfileError> {
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

        #[cfg(multi_threading)]
        let stream_id = self.profile_acquire();

        #[cfg(feature = "profile-tracy")]
        let gpu_span = if self.state.properties.timing_method == TimingMethod::Device {
            let gpu_span = self
                .state
                .gpu_client
                .span_alloc(func_name, "profile", location.file(), location.line())
                .unwrap();
            Some(gpu_span)
        } else {
            None
        };

        let token = self.channel.start_profile();

        let out = func();

        #[allow(unused_mut)]
        let mut result = self.channel.end_profile(token);

        core::mem::drop(out);

        #[cfg(feature = "profile-tracy")]
        if let Some(mut gpu_span) = gpu_span {
            gpu_span.end_zone();
            let epoch = self.state.epoch_time;
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

        #[cfg(multi_threading)]
        self.profile_release(stream_id);

        result
    }

    #[cfg(not(multi_threading))]
    fn profile_guard(&self) {}

    #[cfg(multi_threading)]
    fn profile_guard(&self) {
        let current = self.state.current_profiling.read();

        if let Some(current_stream_id) = current.as_ref() {
            let stream_id = StreamId::current();

            if current_stream_id == &stream_id {
                return;
            }

            core::mem::drop(current);

            loop {
                std::thread::sleep(core::time::Duration::from_millis(10));

                let current = self.state.current_profiling.read();
                match current.as_ref() {
                    Some(current_stream_id) => {
                        if current_stream_id == &stream_id {
                            return;
                        }
                    }
                    None => {
                        return;
                    }
                }
            }
        }
    }

    #[cfg(multi_threading)]
    fn profile_acquire(&self) -> Option<StreamId> {
        let stream_id = StreamId::current();
        let mut current = self.state.current_profiling.write();

        match current.as_mut() {
            Some(current_stream_id) => {
                if current_stream_id == &stream_id {
                    return None;
                }

                core::mem::drop(current);

                loop {
                    std::thread::sleep(core::time::Duration::from_millis(10));

                    let mut current = self.state.current_profiling.write();

                    match current.as_mut() {
                        Some(current_stream_id) => {
                            if current_stream_id == &stream_id {
                                return None;
                            }
                        }
                        None => {
                            *current = Some(stream_id);
                            return Some(stream_id);
                        }
                    }
                }
            }
            None => {
                *current = Some(stream_id);
                Some(stream_id)
            }
        }
    }

    #[cfg(multi_threading)]
    fn profile_release(&self, stream_id: Option<StreamId>) {
        let stream_id = match stream_id {
            Some(val) => val,
            None => return, // No releasing
        };
        let mut current = self.state.current_profiling.write();

        match current.as_mut() {
            Some(current_stream_id) => {
                if current_stream_id != &stream_id {
                    panic!("Can't release a different profiling guard.");
                } else {
                    *current = None;
                }
            }
            None => panic!("Can't release an empty profiling guard"),
        }
    }
}
