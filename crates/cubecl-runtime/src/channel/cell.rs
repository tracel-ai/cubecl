use super::ComputeChannel;
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
    const SERVER_COMM_SUPPORTED: bool = true;

    fn logger(&self) -> Arc<ServerLogger> {
        todo!();
    }

    fn create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut server = self.server.borrow_mut();
        server.create(descriptors, stream_id)
    }

    fn read(
        &self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut server = self.server.borrow_mut();
        server.read(descriptors, stream_id)
    }

    fn write(
        &self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let mut server = self.server.borrow_mut();
        server.write(descriptors, stream_id)
    }

    fn sync(&self, stream_id: StreamId) -> DynFut<()> {
        let mut server = self.server.borrow_mut();
        server.sync(stream_id)
    }

    fn get_resource(
        &self,
        binding: Binding,
        stream_id: StreamId,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        self.server.borrow_mut().get_resource(binding, stream_id)
    }

    unsafe fn execute(
        &self,
        kernel_description: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        stream_id: StreamId,
    ) {
        unsafe {
            self.server
                .borrow_mut()
                .execute(kernel_description, count, bindings, kind, stream_id)
        }
    }

    fn flush(&self, stream_id: StreamId) {
        self.server.borrow_mut().flush(stream_id)
    }

    fn memory_usage(&self, stream_id: StreamId) -> crate::memory_management::MemoryUsage {
        self.server.borrow_mut().memory_usage(stream_id)
    }

    fn memory_cleanup(&self, stream_id: StreamId) {
        self.server.borrow_mut().memory_cleanup(stream_id);
    }

    fn start_profile(&self, stream_id: StreamId) -> ProfilingToken {
        self.server.borrow_mut().start_profile(stream_id)
    }

    fn end_profile(
        &self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.server.borrow_mut().end_profile(stream_id, token)
    }

    fn allocation_mode(
        &self,
        mode: crate::memory_management::MemoryAllocationMode,
        stream_id: StreamId,
    ) {
        self.server.borrow_mut().allocation_mode(mode, stream_id)
    }

    fn copy(
        server_src: &Self,
        server_dst: &Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        let mut server_src = server_src.server.borrow_mut();
        let mut server_dst = server_dst.server.borrow_mut();

        Server::copy(
            &mut server_src,
            &mut server_dst,
            src,
            stream_id_src,
            stream_id_dst,
        )
    }
}

/// This is unsafe, since no concurrency is supported by the `RefCell` channel.
/// However using this channel should only be done in single threaded environments such as `no-std`.
unsafe impl<Server: ComputeServer> Send for RefCellComputeChannel<Server> {}
unsafe impl<Server: ComputeServer> Sync for RefCellComputeChannel<Server> {}
