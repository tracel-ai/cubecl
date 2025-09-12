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

    fn read(&self, descriptors: Vec<CopyDescriptor<'_>>) -> DynFut<Result<Vec<Vec<u8>>, IoError>> {
        let mut server = self.server.lock();
        server.read(descriptors)
    }

    fn write(&self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError> {
        let mut server = self.server.lock();
        server.write(descriptors)
    }

    fn data_transfer_send(&self, id: DataTransferId, src: CopyDescriptor<'_>) {
        let mut server = self.server.lock();
        server.register_src(id, src);
    }

    fn data_transfer_recv(&self, id: DataTransferId, dst: CopyDescriptor<'_>) {
        let mut server = self.server.lock();
        server.register_dest(id, dst);
    }

    fn sync(&self) -> DynFut<()> {
        let mut server = self.server.lock();
        server.sync()
    }

    fn get_resource(
        &self,
        binding: Binding,
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
    ) {
        unsafe {
            self.server
                .lock()
                .execute(kernel, count, handles, kind, logger)
        }
    }

    fn flush(&self) {
        self.server.lock().flush();
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        self.server.lock().memory_usage()
    }

    fn memory_cleanup(&self) {
        self.server.lock().memory_cleanup();
    }

    fn start_profile(&self) -> ProfilingToken {
        self.server.lock().start_profile()
    }

    fn end_profile(&self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        self.server.lock().end_profile(token)
    }

    fn allocation_mode(&self, mode: MemoryAllocationMode) {
        let mut server = self.server.lock();
        server.allocation_mode(mode)
    }
}
