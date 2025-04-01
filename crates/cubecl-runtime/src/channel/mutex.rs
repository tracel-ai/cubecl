use super::ComputeChannel;
use crate::server::{Binding, BindingWithMeta, ComputeServer, ConstBinding, CubeCount, Handle};
use crate::storage::{BindingResource, ComputeStorage};
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::{ExecutionMode, benchmark::TimestampsResult};
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
    async fn read(&self, bindings: Vec<Binding>) -> Vec<Vec<u8>> {
        // Nb: The order here is really important - the mutex guard has to be dropped before
        // the future is polled. Just calling lock().read().await can deadlock.
        let fut = {
            let mut server = self.server.lock();
            server.read(bindings)
        };
        fut.await
    }

    async fn read_tensor(&self, bindings: Vec<BindingWithMeta>) -> Vec<Vec<u8>> {
        // Nb: The order here is really important - the mutex guard has to be dropped before
        // the future is polled. Just calling lock().read().await can deadlock.
        let fut = {
            let mut server = self.server.lock();
            server.read_tensor(bindings)
        };
        fut.await
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

    fn create_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        elem_size: usize,
    ) -> (Handle, Vec<usize>) {
        self.server.lock().create_tensor(data, shape, elem_size)
    }

    fn empty(&self, size: usize) -> Handle {
        self.server.lock().empty(size)
    }

    fn empty_tensor(&self, shape: &[usize], elem_size: usize) -> (Handle, Vec<usize>) {
        self.server.lock().empty_tensor(shape, elem_size)
    }

    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        constants: Vec<ConstBinding>,
        handles: Vec<Binding>,
        kind: ExecutionMode,
    ) {
        unsafe {
            self.server
                .lock()
                .execute(kernel, count, constants, handles, kind)
        }
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

    fn memory_cleanup(&self) {
        self.server.lock().memory_cleanup();
    }

    fn enable_timestamps(&self) {
        self.server.lock().enable_timestamps();
    }

    fn disable_timestamps(&self) {
        self.server.lock().disable_timestamps();
    }
}
