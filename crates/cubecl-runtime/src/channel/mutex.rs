use super::ComputeChannel;
use crate::server::{Binding, ComputeServer, Handle};
use crate::storage::BindingResource;
use crate::ExecutionMode;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::reader::Reader;
use cubecl_common::sync_type::SyncType;
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
    fn read(&self, handle: Binding<Server>) -> Reader {
        self.server.lock().read(handle)
    }

    fn get_resource(&self, binding: Binding<Server>) -> BindingResource<Server> {
        self.server.lock().get_resource(binding)
    }

    fn create(&self, data: &[u8]) -> Handle<Server> {
        self.server.lock().create(data)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        self.server.lock().empty(size)
    }

    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: Server::DispatchOptions,
        handles: Vec<Binding<Server>>,
        kind: ExecutionMode,
    ) {
        self.server.lock().execute(kernel, count, handles, kind)
    }

    fn sync(&self, sync_type: SyncType) {
        self.server.lock().sync(sync_type)
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        self.server.lock().memory_usage()
    }
}
