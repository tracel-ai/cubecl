use cubecl_common::{
    ExecutionMode, bytes::Bytes, future::DynFut, profile::ProfileDuration, stream_id::StreamId,
};

use crate::{
    data_service::DataTransferId,
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, ComputeServer, CopyDescriptor,
        CubeCount, IoError, ProfileError, ProfilingToken,
    },
    storage::{BindingResource, ComputeStorage},
};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// The ComputeChannel trait links the ComputeClient to the ComputeServer
/// while ensuring thread-safety
pub trait ComputeChannel<Server: ComputeServer>: Clone + core::fmt::Debug + Send + Sync {
    const CHANGE_SERVER: bool;

    /// Retrieve the server logger.
    fn logger(&self) -> Arc<ServerLogger>;

    /// Create a new handle given a set of descriptors
    fn create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError>;

    /// Given bindings, returns owned resources as bytes
    fn read(
        &self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>>;

    /// Write bytes to each binding
    fn write(
        &self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        stream_id: StreamId,
    ) -> Result<(), IoError>;

    /// Send data to another server.
    fn data_transfer_send(&self, id: DataTransferId, src: CopyDescriptor<'_>, stream_id: StreamId);

    /// Receive data from another server. Returns when the transfer has been registered.
    fn data_transfer_recv(&self, id: DataTransferId, dst: CopyDescriptor<'_>, stream_id: StreamId);

    fn change_server(
        server_src: &Self,
        server_dst: &Self,
        desc_src: CopyDescriptor<'_>,
        desc_dst: CopyDescriptor<'_>,
    ) -> Result<(), IoError>;

    fn change_server_v2(
        server_src: &Self,
        server_dst: &Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        todo!()
    }

    /// Wait for the completion of every task in the server.
    fn sync(&self, stream_id: StreamId) -> DynFut<()>;

    /// Given a resource handle, return the storage resource.
    fn get_resource(
        &self,
        binding: Binding,
        stream_id: StreamId,
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
        stream_id: StreamId,
    );

    /// Flush outstanding work of the server.
    fn flush(&self, stream_id: StreamId);

    /// Get the current memory usage of the server.
    fn memory_usage(&self, stream_id: StreamId) -> crate::memory_management::MemoryUsage;

    /// Change the memory allocation mode.
    fn allocation_mode(&self, mode: MemoryAllocationMode, stream_id: StreamId);

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&self, stream_id: StreamId);

    /// Start a profile on the server. This allows you to profile kernels.
    ///
    /// This will measure execution time either by measuring the 'full' execution time by synchronizing
    /// the execution at the start and the end of the profile, or 'device' time by using device timestamps.
    /// This function will handle any required synchronization.
    fn start_profile(&self, stream_id: StreamId) -> ProfilingToken;

    /// End the profile and return a [`ProfileDuration`].
    ///
    /// You can retrieve the Duration of the client profile asynchronously. This function will handle any required synchronization.
    fn end_profile(
        &self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError>;
}
