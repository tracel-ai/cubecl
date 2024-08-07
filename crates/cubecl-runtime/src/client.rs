use crate::{
    channel::ComputeChannel,
    server::{Binding, ComputeServer, Handle},
    storage::ComputeStorage,
    ExecutionMode,
};
use alloc::sync::Arc;
use alloc::vec::Vec;

pub use cubecl_common::sync_type::SyncType;

/// The ComputeClient is the entry point to require tasks from the ComputeServer.
/// It should be obtained for a specific device via the Compute struct.
#[derive(Debug)]
pub struct ComputeClient<Server: ComputeServer, Channel> {
    channel: Channel,
    features: Arc<Server::FeatureSet>,
}

impl<S, C> Clone for ComputeClient<S, C>
where
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
            features: self.features.clone(),
        }
    }
}

impl<Server, Channel> ComputeClient<Server, Channel>
where
    Server: ComputeServer,
    Channel: ComputeChannel<Server>,
{
    /// Create a new client.
    pub fn new(channel: Channel, features: Arc<Server::FeatureSet>) -> Self {
        Self { channel, features }
    }

    /// Given a binding, returns owned resource as bytes.
    pub async fn read_async(&self, binding: Binding<Server>) -> Vec<u8> {
        self.channel.read(binding).await
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    pub fn read(&self, binding: Binding<Server>) -> Vec<u8> {
        cubecl_common::reader::read_sync(self.channel.read(binding))
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        binding: Binding<Server>,
    ) -> <Server::Storage as ComputeStorage>::Resource {
        self.channel.get_resource(binding)
    }

    /// Given a resource, stores it and returns the resource handle.
    pub fn create(&self, data: &[u8]) -> Handle<Server> {
        self.channel.create(data)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle<Server> {
        self.channel.empty(size)
    }

    /// Executes the `kernel` over the given `bindings`.
    pub fn execute(
        &self,
        kernel: Server::Kernel,
        count: Server::DispatchOptions,
        bindings: Vec<Binding<Server>>,
    ) {
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
        count: Server::DispatchOptions,
        bindings: Vec<Binding<Server>>,
    ) {
        self.channel
            .execute(kernel, count, bindings, ExecutionMode::Unchecked)
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self, sync_type: SyncType) {
        self.channel.sync(sync_type)
    }

    /// Get the features supported by the compute server.
    pub fn features(&self) -> &Server::FeatureSet {
        self.features.as_ref()
    }
}
