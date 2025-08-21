use cubecl_common::{ExecutionMode, future::DynFut, profile::ProfileDuration};

use crate::{
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, ComputeServer, CopyDescriptor,
        CubeCount, IoError, ProfileError, ProfilingToken,
    },
    storage::{BindingResource, ComputeStorage}, data_service::ComputeDataTransferId,
};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug + Send + Sync {
    /// Create a new handle given a set of descriptors
    fn create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError>;

    /// Given bindings, returns owned resources as bytes
    fn read(&self, descriptors: Vec<CopyDescriptor<'_>>) -> DynFut<Result<Vec<Vec<u8>>, IoError>>;

    /// Write bytes to each binding
    fn write(&self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError>;

    /// Send data to another server
    fn send_to_peer(&self, id: ComputeDataTransferId, src: CopyDescriptor<'_>) -> Result<(), IoError>;

    /// Receive data from another server
    fn recv_from_peer(&self, id: ComputeDataTransferId, dst: CopyDescriptor<'_>) -> Result<(), IoError>;

    /// Wait for the completion of every task in the server.
    fn sync(&self) -> DynFut<()>;

    /// Given a resource handle, return the storage resource.
    fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource>;

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

    /// Change the memory allocation mode.
    fn allocation_mode(&self, mode: MemoryAllocationMode);

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&self);

    /// Start a profile on the server. This allows you to profile kernels.
    ///
    /// This will measure execution time either by measuring the 'full' execution time by synchronizing
    /// the execution at the start and the end of the profile, or 'device' time by using device timestamps.
    /// This function will handle any required synchronization.
    fn start_profile(&self) -> ProfilingToken;

    /// End the profile and return a [`ProfileDuration`].
    ///
    /// You can retrieve the Duration of the client profile asynchronously. This function will handle any required synchronization.
    fn end_profile(&self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError>;
}
