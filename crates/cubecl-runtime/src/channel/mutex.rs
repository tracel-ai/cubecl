use super::ComputeChannel;
use crate::logging::ServerLogger;
use crate::server::{
    Binding, BindingWithMeta, Bindings, ComputeServer, CubeCount, Handle, ProfileError,
    ProfilingToken,
};
use crate::storage::{BindingResource, ComputeStorage};
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::ExecutionMode;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
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
    fn read(&self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        let mut server = self.server.lock();
        server.read(bindings)
    }

    fn read_tensor(&self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let mut server = self.server.lock();
        server.read_tensor(bindings)
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

    fn create(&self, data: &[u8]) -> Handle {
        self.server.lock().create(data)
    }

    fn create_tensors(
        &self,
        data: Vec<&[u8]>,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        self.server.lock().create_tensors(data, shape, elem_size)
    }

    fn empty(&self, size: usize) -> Handle {
        self.server.lock().empty(size)
    }

    fn empty_tensors(
        &self,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        self.server.lock().empty_tensors(shape, elem_size)
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
}
