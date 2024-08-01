use crate::{
    server::{Binding, ComputeServer, Handle},
    storage::ComputeStorage,
};
use alloc::vec::Vec;
use cubecl_common::{reader::Reader, sync_type::SyncType};

/// The kind of execution to be performed.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum KernelExecutionStrategy {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Unchecked kernels are unsafe.
    Unchecked,
}

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug + Send + Sync {
    /// Given a binding, returns owned resource as bytes
    fn read(&self, binding: Binding<Server>) -> Reader;

    /// Given a resource handle, return the storage resource.
    fn get_resource(
        &self,
        binding: Binding<Server>,
    ) -> <Server::Storage as ComputeStorage>::Resource;

    /// Given a resource as bytes, stores it and returns the resource handle
    fn create(&self, data: &[u8]) -> Handle<Server>;

    /// Reserves `size` bytes in the storage, and returns a handle over them
    fn empty(&self, size: usize) -> Handle<Server>;

    /// Executes the `kernel` over the given `bindings`.
    fn execute(
        &self,
        kernel: Server::Kernel,
        count: Server::DispatchOptions,
        bindings: Vec<Binding<Server>>,
        strategy: KernelExecutionStrategy,
    );

    /// Perform some synchronization of commands on the server.
    fn sync(&self, sync_type: SyncType);
}
