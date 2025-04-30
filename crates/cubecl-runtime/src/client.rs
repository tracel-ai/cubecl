use crate::{
    DeviceProperties, TimeMeasurement,
    channel::ComputeChannel,
    memory_management::MemoryUsage,
    server::{Binding, BindingWithMeta, Bindings, ComputeServer, CubeCount, Handle},
    storage::{BindingResource, ComputeStorage},
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::{ExecutionMode, benchmark::ProfileDuration};
use spin::Mutex;

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
#[derive(Debug)]
pub struct ComputeClient<Server: ComputeServer, Channel> {
    channel: Channel,
    state: Arc<ComputeClientState<Server>>,
}

#[derive(new, Debug)]
struct ComputeClientState<Server: ComputeServer> {
    properties: DeviceProperties<Server::Feature>,
    profile_lock: Mutex<()>,

    info: Server::Info,
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
        let state = ComputeClientState::new(properties, Mutex::new(()), info);
        Self {
            channel,
            state: Arc::new(state),
        }
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_async(
        &self,
        bindings: Vec<Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + use<Server, Channel> {
        self.channel.read(bindings)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    pub fn read(&self, bindings: Vec<Binding>) -> Vec<Vec<u8>> {
        cubecl_common::reader::read_sync(self.channel.read(bindings))
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    pub fn read_one(&self, binding: Binding) -> Vec<u8> {
        cubecl_common::reader::read_sync(self.channel.read([binding].into())).remove(0)
    }

    /// Given bindings, returns owned resources as bytes.
    pub async fn read_tensor_async(&self, bindings: Vec<BindingWithMeta>) -> Vec<Vec<u8>> {
        self.channel.read_tensor(bindings).await
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
    /// stride compatiblity on the runtime will be added in the future.
    ///
    /// Also see [ComputeClient::create_tensor].
    pub fn read_tensor(&self, bindings: Vec<BindingWithMeta>) -> Vec<Vec<u8>> {
        cubecl_common::reader::read_sync(self.channel.read_tensor(bindings))
    }

    /// Given a binding, returns owned resource as bytes.
    /// See [ComputeClient::read_tensor]
    pub async fn read_one_tensor_async(&self, binding: BindingWithMeta) -> Vec<u8> {
        self.channel.read_tensor([binding].into()).await.remove(0)
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    /// See [ComputeClient::read_tensor]
    pub fn read_one_tensor(&self, binding: BindingWithMeta) -> Vec<u8> {
        cubecl_common::reader::read_sync(self.channel.read_tensor([binding].into())).remove(0)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        self.channel.get_resource(binding)
    }

    /// Given a resource, stores it and returns the resource handle.
    pub fn create(&self, data: &[u8]) -> Handle {
        self.channel.create(data)
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
    pub fn create_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        elem_size: usize,
    ) -> (Handle, Vec<usize>) {
        self.channel.create_tensor(data, shape, elem_size)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle {
        self.channel.empty(size)
    }

    /// Reserves `shape` in the storage, and returns a tensor handle for it.
    /// See [ComputeClient::create_tensor]
    pub fn empty_tensor(&self, shape: &[usize], elem_size: usize) -> (Handle, Vec<usize>) {
        self.channel.empty_tensor(shape, elem_size)
    }

    /// Executes the `kernel` over the given `bindings`.
    pub fn execute(&self, kernel: Server::Kernel, count: CubeCount, bindings: Bindings) {
        unsafe {
            self.channel
                .execute(kernel, count, bindings, ExecutionMode::Checked)
        }
    }

    /// Executes the `kernel` over the given `bindings` without performing any bound checks.
    ///
    /// # Safety
    ///
    /// Without checks, the out-of-bound reads and writes can happen.
    pub unsafe fn execute_unchecked(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
    ) {
        unsafe {
            self.channel
                .execute(kernel, count, bindings, ExecutionMode::Unchecked)
        }
    }

    /// Flush all outstanding commands.
    pub fn flush(&self) {
        self.channel.flush();
    }

    /// Wait for the completion of every task in the server.
    pub async fn sync(&self) {
        self.channel.sync().await
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties<Server::Feature> {
        &self.state.properties
    }

    /// Get the current memory usage of this client.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.channel.memory_usage()
    }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        self.channel.memory_cleanup()
    }

    /// Measure the execution time of some inner operations.
    ///
    /// Nb: this function will only allow one function at a time to be submitted when multithrading.
    /// Recursive measurements are not allowed and will deadlock.
    pub fn profile(&self, func: impl FnOnce()) -> ProfileDuration {
        let guard = self.state.profile_lock.lock();
        self.channel.start_profile();
        func();
        let result = self.channel.end_profile();
        let result = match self.properties().time_measurement() {
            TimeMeasurement::Device => result,
            TimeMeasurement::System => {
                #[cfg(target_family = "wasm")]
                panic!("Can't use system timing mode on wasm");

                #[cfg(not(target_family = "wasm"))]
                {
                    // It is important to wait for the profiling to be done, since we're actually
                    // measuring its execution timing using 'real' time.
                    let duration = cubecl_common::future::block_on(result.resolve());
                    ProfileDuration::from_duration(duration)
                }
            }
        };
        core::mem::drop(guard);
        result
    }
}
