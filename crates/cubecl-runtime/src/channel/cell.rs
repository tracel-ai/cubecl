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

    fn create_tensors(
        &self,
        data: Vec<&[u8]>,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        self.server
            .borrow_mut()
            .create_tensors(data, shape, elem_size)
    }

    fn empty(&self, size: usize) -> Handle {
        self.server.borrow_mut().empty(size)
    }

    fn empty_tensors(
        &self,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        self.server.borrow_mut().empty_tensors(shape, elem_size)
    }

    unsafe fn execute(
        &self,
        kernel_description: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) {
        unsafe {
            self.server
                .borrow_mut()
                .execute(kernel_description, count, bindings, kind, logger)
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

    fn start_profile(&self) -> ProfilingToken {
        self.server.borrow_mut().start_profile()
    }

    fn end_profile(&self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        self.server.borrow_mut().end_profile(token)
    }

    fn allocation_mode(&self, mode: crate::memory_management::MemoryAllocationMode) {
        self.server.borrow_mut().allocation_mode(mode)
    }
}

/// This is unsafe, since no concurrency is supported by the `RefCell` channel.
/// However using this channel should only be done in single threaded environments such as `no-std`.
unsafe impl<Server: ComputeServer> Send for RefCellComputeChannel<Server> {}
unsafe impl<Server: ComputeServer> Sync for RefCellComputeChannel<Server> {}
