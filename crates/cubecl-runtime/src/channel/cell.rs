use super::ComputeChannel;
use crate::server::{Binding, BindingWithMeta, Bindings, ComputeServer, CubeCount, Handle};
use crate::storage::{BindingResource, ComputeStorage};
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::ExecutionMode;
use cubecl_common::benchmark::ProfileDuration;
use cubecl_common::future::DynFut;

/// A channel using a [ref cell](core::cell::RefCell) to access the server with mutability.
///
/// # Important
///
/// Only use this channel if you don't use any threading in your application, otherwise it will
/// panic or cause undefined behaviors.
///
/// This is mosly useful for `no-std` environments where threads aren't supported, otherwise prefer
/// the [mutex](super::MutexComputeChannel) or the [mpsc](super::MpscComputeChannel) channels.
#[derive(Debug)]
pub struct RefCellComputeChannel<Server> {
    server: Arc<core::cell::RefCell<Server>>,
}

impl<S> Clone for RefCellComputeChannel<S> {
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
        }
    }
}

impl<Server> RefCellComputeChannel<Server>
where
    Server: ComputeServer,
{
    /// Create a new cell compute channel.
    pub fn new(server: Server) -> Self {
        Self {
            server: Arc::new(core::cell::RefCell::new(server)),
        }
    }
}

impl<Server> ComputeChannel<Server> for RefCellComputeChannel<Server>
where
    Server: ComputeServer + Send,
{
    fn read(&self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        let mut server = self.server.borrow_mut();
        server.read(bindings)
    }

    fn read_tensor(&self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let mut server = self.server.borrow_mut();
        server.read_tensor(bindings)
    }

    fn sync(&self) -> DynFut<()> {
        let mut server = self.server.borrow_mut();
        server.sync()
    }

    fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        self.server.borrow_mut().get_resource(binding)
    }

    fn create(&self, resource: &[u8]) -> Handle {
        self.server.borrow_mut().create(resource)
    }

    fn create_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        elem_size: usize,
    ) -> (Handle, Vec<usize>) {
        self.server
            .borrow_mut()
            .create_tensor(data, shape, elem_size)
    }

    fn empty(&self, size: usize) -> Handle {
        self.server.borrow_mut().empty(size)
    }

    fn empty_tensor(&self, shape: &[usize], elem_size: usize) -> (Handle, Vec<usize>) {
        self.server.borrow_mut().empty_tensor(shape, elem_size)
    }

    unsafe fn execute(
        &self,
        kernel_description: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
    ) {
        unsafe {
            self.server
                .borrow_mut()
                .execute(kernel_description, count, bindings, kind)
        }
    }

    fn flush(&self) {
        self.server.borrow_mut().flush()
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        self.server.borrow_mut().memory_usage()
    }

    fn memory_cleanup(&self) {
        self.server.borrow_mut().memory_cleanup();
    }

    fn start_profile(&self) {
        self.server.borrow_mut().start_profile()
    }

    fn end_profile(&self) -> ProfileDuration {
        self.server.borrow_mut().end_profile()
    }
}

/// This is unsafe, since no concurrency is supported by the `RefCell` channel.
/// However using this channel should only be done in single threaded environments such as `no-std`.
unsafe impl<Server: ComputeServer> Send for RefCellComputeChannel<Server> {}
unsafe impl<Server: ComputeServer> Sync for RefCellComputeChannel<Server> {}
