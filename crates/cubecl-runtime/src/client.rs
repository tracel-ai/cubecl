use core::future::Future;

use crate::{
    channel::ComputeChannel,
    memory_management::MemoryUsage,
    server::{Binding, ComputeServer, CubeCount, Handle},
    storage::BindingResource,
    DeviceProperties, ExecutionMode,
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::benchmark::TimestampsResult;

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
    // #[cfg(not(target_family = "wasm"))]
    // timestamp_lock: cubecl_common::stub::Mutex<()>,
    // #[cfg(target_family = "wasm")]
    timestamp_lock: async_lock::Mutex<()>,
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
    /// Create a new client.
    pub fn new(channel: Channel, properties: DeviceProperties<Server::Feature>) -> Self {
        // let state = ComputeClientState::new(properties, cubecl_common::stub::Mutex::new(()));
        let state = ComputeClientState::new(properties, async_lock::Mutex::new(()));
        Self {
            channel,
            state: Arc::new(state),
        }
    }

    /// Given a binding, returns owned resource as bytes.
    pub async fn read_async(&self, binding: Binding) -> Vec<u8> {
        self.channel.read(binding).await
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    pub fn read(&self, binding: Binding) -> Vec<u8> {
        cubecl_common::reader::read_sync(self.channel.read(binding))
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(&self, binding: Binding) -> BindingResource<Server> {
        self.channel.get_resource(binding)
    }

    /// Given a resource, stores it and returns the resource handle.
    pub fn create(&self, data: &[u8]) -> Handle {
        self.channel.create(data)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle {
        self.channel.empty(size)
    }

    /// Executes the `kernel` over the given `bindings`.
    pub fn execute(&self, kernel: Server::Kernel, count: CubeCount, bindings: Vec<Binding>) {
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
        bindings: Vec<Binding>,
    ) {
        self.channel
            .execute(kernel, count, bindings, ExecutionMode::Unchecked)
    }

    /// Flush all outstanding commands.
    pub fn flush(&self) {
        self.channel.flush();
    }

    /// Wait for the completion of every task in the server.
    pub async fn sync(&self) {
        self.channel.sync().await
    }

    /// Wait for the completion of every task in the server.
    pub async fn sync_elapsed(&self) -> TimestampsResult {
        self.channel.sync_elapsed().await
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties<Server::Feature> {
        &self.state.properties
    }

    /// Get the current memory usage of this client.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.channel.memory_usage()
    }

    /// Profile a function.
    pub async fn profile<O, Fut, Func>(&self, func: Func) -> O
    where
        Fut: Future<Output = O>,
        Func: FnOnce() -> Fut,
    {
        let lock = &self.state.timestamp_lock;
        // let guard = lock.lock().unwrap();
        let guard = lock.lock().await;

        self.channel.enable_timestamps();

        let fut = func();
        let output = fut.await;

        self.channel.disable_timestamps();

        core::mem::drop(guard);
        output
    }

    /// Eanble timestamps collection on the server.
    ///
    /// # Warning
    ///
    /// Only use it when performing benchmarks since it's going to slow down a lot throughput.
    pub fn enable_timestamps(&self) {
        self.channel.enable_timestamps();
    }
}
