use std::sync::Arc;

use cubecl_common::{
    ExecutionMode,
    future::{DynFut, spawn_detached_fut},
    profile::ProfileDuration,
};

use super::ComputeChannel;
use crate::{
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    server::{
        Allocation, AllocationDescriptor, AllocationType, Binding, Bindings, ComputeServer,
        CopyDescriptor, CubeCount, Handle, IoError, ProfileError, ProfilingToken,
    },
    storage::{BindingResource, ComputeStorage},
};

/// Create a channel using a [multi-producer, single-consumer channel to communicate with
/// the compute server spawn on its own thread.
#[derive(Debug)]
pub struct MpscComputeChannel<Server>
where
    Server: ComputeServer,
{
    state: Arc<MpscComputeChannelState<Server>>,
}

#[derive(Debug)]
struct MpscComputeChannelState<Server>
where
    Server: ComputeServer,
{
    sender: async_channel::Sender<Message<Server>>,
}

type Callback<Response> = async_channel::Sender<Response>;

struct AllocationDescriptorOwned {
    type_: AllocationType,
    shape: Vec<usize>,
    elem_size: usize,
}

impl From<AllocationDescriptor<'_>> for AllocationDescriptorOwned {
    fn from(value: AllocationDescriptor) -> Self {
        AllocationDescriptorOwned {
            type_: value.type_,
            shape: value.shape.to_vec(),
            elem_size: value.elem_size,
        }
    }
}

impl AllocationDescriptorOwned {
    fn as_ref(&self) -> AllocationDescriptor<'_> {
        AllocationDescriptor::new(self.type_, &self.shape, self.elem_size)
    }
}

struct CopyDescriptorOwned {
    handle: Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    elem_size: usize,
}

impl From<CopyDescriptor<'_>> for CopyDescriptorOwned {
    fn from(value: CopyDescriptor<'_>) -> Self {
        CopyDescriptorOwned {
            handle: value.handle,
            shape: value.shape.to_vec(),
            strides: value.strides.to_vec(),
            elem_size: value.elem_size,
        }
    }
}

impl CopyDescriptorOwned {
    fn as_ref(&self) -> CopyDescriptor<'_> {
        CopyDescriptor::new(
            self.handle.clone(),
            &self.shape,
            &self.strides,
            self.elem_size,
        )
    }
}

enum Message<Server>
where
    Server: ComputeServer,
{
    Create(
        Vec<AllocationDescriptorOwned>,
        Callback<Result<Vec<Allocation>, IoError>>,
    ),
    Read(
        Vec<CopyDescriptorOwned>,
        Callback<Result<Vec<Vec<u8>>, IoError>>,
    ),
    Write(
        Vec<(CopyDescriptorOwned, Vec<u8>)>,
        Callback<Result<(), IoError>>,
    ),
    GetResource(
        Binding,
        Callback<BindingResource<<Server::Storage as ComputeStorage>::Resource>>,
    ),
    ExecuteKernel(
        (Server::Kernel, CubeCount, ExecutionMode),
        Bindings,
        Arc<ServerLogger>,
    ),
    Flush,
    Sync(Callback<()>),
    MemoryUsage(Callback<MemoryUsage>),
    MemoryCleanup,
    AllocationMode(MemoryAllocationMode),
    StartProfile(Callback<ProfilingToken>),
    StopMeasure(
        Callback<Result<ProfileDuration, ProfileError>>,
        ProfilingToken,
    ),
}

impl<Server> MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    /// Create a new mpsc compute channel.
    pub fn new(mut server: Server) -> Self {
        let (sender, receiver) = async_channel::unbounded();

        spawn_detached_fut(async move {
            while let Ok(message) = receiver.recv().await {
                match message {
                    Message::Create(descriptors, callback) => {
                        let descriptors = descriptors.iter().map(|it| it.as_ref()).collect();
                        let data = server.create(descriptors);
                        callback.send(data).await.unwrap();
                    }
                    Message::Read(descriptors, callback) => {
                        let descriptors = descriptors.iter().map(|it| it.as_ref()).collect();
                        let data = server.read(descriptors).await;
                        callback.send(data).await.unwrap();
                    }
                    Message::Write(descriptors, callback) => {
                        let descriptors = descriptors
                            .iter()
                            .map(|(desc, data)| (desc.as_ref(), data.as_slice()))
                            .collect();
                        let data = server.write(descriptors);
                        callback.send(data).await.unwrap();
                    }
                    Message::GetResource(binding, callback) => {
                        let data = server.get_resource(binding);
                        callback.send(data).await.unwrap();
                    }
                    Message::ExecuteKernel(kernel, bindings, logger) => unsafe {
                        server.execute(kernel.0, kernel.1, bindings, kernel.2, logger);
                    },
                    Message::Sync(callback) => {
                        server.sync().await;
                        callback.send(()).await.unwrap();
                    }
                    Message::Flush => {
                        server.flush();
                    }
                    Message::MemoryUsage(callback) => {
                        callback.send(server.memory_usage()).await.unwrap();
                    }
                    Message::MemoryCleanup => {
                        server.memory_cleanup();
                    }
                    Message::StartProfile(callback) => {
                        let token = server.start_profile();
                        callback.send(token).await.unwrap();
                    }
                    Message::StopMeasure(callback, token) => {
                        callback.send(server.end_profile(token)).await.unwrap();
                    }
                    Message::AllocationMode(mode) => {
                        server.allocation_mode(mode);
                    }
                };
            }
        });

        Self {
            state: Arc::new(MpscComputeChannelState { sender }),
        }
    }
}

impl<Server: ComputeServer> Clone for MpscComputeChannel<Server> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<Server> ComputeChannel<Server> for MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    fn create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError> {
        let descriptors = descriptors.into_iter().map(|it| it.into()).collect();

        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::Create(descriptors, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn read(&self, descriptors: Vec<CopyDescriptor<'_>>) -> DynFut<Result<Vec<Vec<u8>>, IoError>> {
        let sender = self.state.sender.clone();
        let descriptors = descriptors.into_iter().map(|it| it.into()).collect();

        Box::pin(async move {
            let (callback, response) = async_channel::unbounded();
            sender
                .send(Message::Read(descriptors, callback))
                .await
                .unwrap();
            handle_response(response.recv().await)
        })
    }

    fn write(&self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError> {
        let descriptors = descriptors
            .into_iter()
            .map(|(desc, data)| (desc.into(), data.to_vec()))
            .collect();

        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::Write(descriptors, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::GetResource(binding, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) {
        self.state
            .sender
            .send_blocking(Message::ExecuteKernel(
                (kernel, count, kind),
                bindings,
                logger,
            ))
            .unwrap();
    }

    fn flush(&self) {
        self.state.sender.send_blocking(Message::Flush).unwrap()
    }

    fn sync(&self) -> DynFut<()> {
        let sender = self.state.sender.clone();

        Box::pin(async move {
            let (callback, response) = async_channel::unbounded();
            sender.send(Message::Sync(callback)).await.unwrap();
            handle_response(response.recv().await)
        })
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::MemoryUsage(callback))
            .unwrap();
        handle_response(response.recv_blocking())
    }

    fn memory_cleanup(&self) {
        self.state
            .sender
            .send_blocking(Message::MemoryCleanup)
            .unwrap()
    }

    fn start_profile(&self) -> ProfilingToken {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::StartProfile(callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn end_profile(&self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::StopMeasure(callback, token))
            .unwrap();
        handle_response(response.recv_blocking())
    }

    fn allocation_mode(&self, mode: crate::memory_management::MemoryAllocationMode) {
        self.state
            .sender
            .send_blocking(Message::AllocationMode(mode))
            .unwrap()
    }
}

fn handle_response<Response, Err: core::fmt::Debug>(response: Result<Response, Err>) -> Response {
    match response {
        Ok(val) => val,
        Err(err) => panic!("Can't connect to the server correctly {err:?}"),
    }
}
