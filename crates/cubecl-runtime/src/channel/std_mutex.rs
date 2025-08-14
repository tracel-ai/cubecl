use std::sync::Mutex;

use super::ComputeChannel;
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

/// The MutexComputeChannel ensures thread-safety by locking the server
/// on every operation
#[derive(Debug)]
pub struct StdMutexComputeChannel<Server> {
    server: Arc<Mutex<Server>>,
}

impl<S> Clone for StdMutexComputeChannel<S> {
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
        }
    }
}
impl<Server> StdMutexComputeChannel<Server>
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

impl<Server> ComputeChannel<Server> for StdMutexComputeChannel<Server>
where
    Server: ComputeServer,
{
    fn create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut server = self.server.lock().unwrap();
        server.create(descriptors)
    }

    fn read(&self, descriptors: Vec<CopyDescriptor<'_>>) -> DynFut<Result<Vec<Vec<u8>>, IoError>> {
        let mut server = self.server.lock().unwrap();
        server.read(descriptors)
    }

    fn write(&self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError> {
        let mut server = self.server.lock().unwrap();
        server.write(descriptors)
    }

    fn sync(&self) -> DynFut<()> {
        let mut server = self.server.lock().unwrap();
        server.sync()
    }

    fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        self.server.lock().unwrap().get_resource(binding)
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
                .unwrap()
                .execute(kernel, count, handles, kind, logger)
        }
    }

    fn flush(&self) {
        self.server.lock().unwrap().flush();
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        self.server.lock().unwrap().memory_usage()
    }

    fn memory_cleanup(&self) {
        self.server.lock().unwrap().memory_cleanup();
    }

    fn start_profile(&self) -> ProfilingToken {
        self.server.lock().unwrap().start_profile()
    }

    fn end_profile(&self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        self.server.lock().unwrap().end_profile(token)
    }

    fn allocation_mode(&self, mode: MemoryAllocationMode) {
        let mut server = self.server.lock().unwrap();
        server.allocation_mode(mode)
    }
}
