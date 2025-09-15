use super::ComputeChannel;
use crate::data_service::DataTransferId;
use crate::memory_management::MemoryAllocationMode;
use crate::server::{
    Binding, Bindings, ComputeServer, CopyDescriptor, CubeCount, ProfileError, ProfilingToken,
};
use crate::storage::{BindingResource, ComputeStorage};
use crate::{
    logging::ServerLogger,
    server::{Allocation, AllocationDescriptor, IoError},
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::ExecutionMode;
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use spin::Mutex;

/// The MutexComputeChannel ensures thread-safety by locking the server
/// on every operation
#[derive(Debug)]
pub struct MutexComputeChannel<Server> {
    server: Arc<Mutex<Server>>,
}

impl<S> Clone for MutexComputeChannel<S> {
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
        }
    }
}
impl<Server> MutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    /// Create a new mutex compute channel.
    pub fn new(server: Server) -> Self {
        Self {
            server: Arc::new(Mutex::new(server)),
        }
    }
}

impl<Server> ComputeChannel<Server> for MutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    fn create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut server = self.server.lock();
        server.create(descriptors, stream_id)
    }

    fn read(
        &self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut server = self.server.lock();
        server.read(descriptors, stream_id)
    }

    fn write(
        &self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let mut server = self.server.lock();
        server.write(descriptors, stream_id)
    }

    fn data_transfer_send(&self, id: DataTransferId, src: CopyDescriptor<'_>, stream_id: StreamId) {
        let mut server = self.server.lock();
        server.register_src(id, src);
    }

    fn data_transfer_recv(&self, id: DataTransferId, dst: CopyDescriptor<'_>, stream_id: StreamId) {
        let mut server = self.server.lock();
        server.register_dest(id, dst);
    }

    fn sync(&self, stream_id: StreamId) -> DynFut<()> {
        let mut server = self.server.lock();
        server.sync(stream_id)
    }

    fn get_resource(
        &self,
        binding: Binding,
        stream_id: StreamId,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        self.server.lock().get_resource(binding)
    }

    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        handles: Bindings,
        kind: ExecutionMode,
        logger: Arc<ServerLogger>,
        stream_id: StreamId,
    ) {
        unsafe {
            self.server
                .lock()
                .execute(kernel, count, handles, kind, logger, stream_id)
        }
    }

    fn flush(&self, stream_id: StreamId) {
        self.server.lock().flush(stream_id);
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        self.server.lock().memory_usage()
    }

    fn memory_cleanup(&self) {
        self.server.lock().memory_cleanup();
    }

    fn start_profile(&self, stream_id: StreamId) -> ProfilingToken {
        self.server.lock().start_profile(stream_id)
    }

    fn end_profile(
        &self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.server.lock().end_profile(stream_id, token)
    }

    fn allocation_mode(&self, mode: MemoryAllocationMode) {
        let mut server = self.server.lock();
        server.allocation_mode(mode)
    }
}
