use cubecl_common::{ExecutionMode, benchmark::ProfileDuration, future::DynFut};

use crate::{
    logging::ServerLogger,
    server::{
        Binding, BindingWithMeta, Bindings, ComputeServer, CubeCount, Handle, ProfilingToken,
    },
    storage::{BindingResource, ComputeStorage},
};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug + Send + Sync {
    /// Given bindings, returns owned resources as bytes
    fn read(&self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>>;

    /// Given bindings, returns owned resources as bytes
    fn read_tensor(&self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>>;

    /// Wait for the completion of every task in the server.
    fn sync(&self) -> DynFut<()>;

    /// Given a resource handle, return the storage resource.
    fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource>;

    /// Given a resource as bytes, stores it and returns the resource handle
    fn create(&self, data: &[u8]) -> Handle;

    /// Given a resource as bytes and a shape, stores it and returns the tensor handle
    fn create_tensors(
        &self,
        data: Vec<&[u8]>,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)>;

    /// Reserves `size` bytes in the storage, and returns a handle over them
    fn empty(&self, size: usize) -> Handle;

    /// Reserves a tensor with `shape` in the storage, and returns a handle to it
    fn empty_tensors(
        &self,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)>;

    /// Executes the `kernel` over the given `bindings`.
    ///
    /// Optionally returns some debug information about the compilation to be logged.
    /// # Safety
    ///
    /// When executing with mode [ExecutionMode::Unchecked], out-of-bound reads and writes can happen.
    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    );

    /// Flush outstanding work of the server.
    fn flush(&self);

    /// Get the current memory usage of the server.
    fn memory_usage(&self) -> crate::memory_management::MemoryUsage;

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&self);

    /// Start a profile on the server. This allows you to profile kernels.
    ///
    /// This will measure execution time either by measuring the 'full' execution time by synchronizing
    /// the execution at the start and the end of the profile, or 'device' time by using device timestamps.
    /// This function will handle any required synchronization.
    ///
    /// Recursive profiling is not allowed and will panic.
    fn start_profile(&self) -> ProfilingToken;

    /// End the profile and return a [`ProfileDuration`].
    ///
    /// You can retrieve the Duration of the client profile asynchronously. This function will handle any required synchronization.
    fn end_profile(&self, token: ProfilingToken) -> ProfileDuration;
}
