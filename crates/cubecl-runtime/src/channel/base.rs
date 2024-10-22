use core::future::Future;
use cubecl_common::benchmark::TimestampsResult;

use crate::{
    server::{Binding, ComputeServer, CubeCount, Handle},
    storage::BindingResource,
    ExecutionMode,
};
use alloc::vec::Vec;

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug + Send + Sync {
    /// Given a binding, returns owned resource as bytes
    fn read(&self, binding: Binding) -> impl Future<Output = Vec<u8>> + Send;

    /// Given a resource handle, return the storage resource.
    fn get_resource(&self, binding: Binding) -> BindingResource<Server>;

    /// Given a resource as bytes, stores it and returns the resource handle
    fn create(&self, data: &[u8]) -> Handle;

    /// Reserves `size` bytes in the storage, and returns a handle over them
    fn empty(&self, size: usize) -> Handle;

    /// Executes the `kernel` over the given `bindings`.
    ///
    /// # Safety
    ///
    /// When executing with mode [ExecutionMode::Unchecked], out-of-bound reads and writes can happen.
    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        bindings: Vec<Binding>,
        mode: ExecutionMode,
    );

    /// Flush outstanding work of the server.
    fn flush(&self);

    /// Wait for the completion of every task in the server.
    fn sync(&self) -> impl Future<Output = ()> + Send;

    /// Wait for the completion of every task in the server.
    ///
    /// Returns the (approximate) total amount of GPU work done since the last sync.
    fn sync_elapsed(&self) -> impl Future<Output = TimestampsResult> + Send;

    /// Get the current memory usage of the server.
    fn memory_usage(&self) -> crate::memory_management::MemoryUsage;

    /// Enable collecting timestamps.
    fn enable_timestamps(&self);

    /// Disable collecting timestamps.
    fn disable_timestamps(&self);
}
