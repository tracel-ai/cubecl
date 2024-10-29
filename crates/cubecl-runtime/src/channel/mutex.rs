use super::ComputeChannel;
use crate::server::{Binding, ComputeServer, CubeCount, Handle};
use crate::storage::BindingResource;
use crate::ExecutionMode;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::benchmark::TimestampsResult;
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
    async fn read(&self, handle: Binding) -> Vec<u8> {
        // Nb: The order here is really important - the mutex guard has to be dropped before
        // the future is polled. Just calling lock().read().await can deadlock.
        let fut = {
            let mut server = self.server.lock();
            server.read(handle)
        };
        fut.await
    }

    fn get_resource(&self, binding: Binding) -> BindingResource<Server> {
        self.server.lock().get_resource(binding)
    }

    fn create(&self, data: &[u8]) -> Handle {
        self.server.lock().create(data)
    }

    fn empty(&self, size: usize) -> Handle {
        self.server.lock().empty(size)
    }

    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        handles: Vec<Binding>,
        kind: ExecutionMode,
    ) {
        self.server.lock().execute(kernel, count, handles, kind)
    }

    fn flush(&self) {
        self.server.lock().flush();
    }

    async fn sync(&self) {
        // Nb: The order here is really important - the mutex guard has to be dropped before
        // the future is polled. Just calling lock().sync().await can deadlock.
        let fut = {
            let mut server = self.server.lock();
            server.sync()
        };
        fut.await
    }

    async fn sync_elapsed(&self) -> TimestampsResult {
        // Nb: The order here is really important - the mutex guard has to be dropped before
        // the future is polled. Just calling lock().sync().await can deadlock.
        let fut = {
            let mut server = self.server.lock();
            server.sync_elapsed()
        };
        fut.await
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        self.server.lock().memory_usage()
    }

    fn enable_timestamps(&self) {
        self.server.lock().enable_timestamps();
    }

    fn disable_timestamps(&self) {
        self.server.lock().disable_timestamps();
    }
}
