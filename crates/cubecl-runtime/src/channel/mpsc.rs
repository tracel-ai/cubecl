use std::sync::Arc;

use cubecl_common::{
    ExecutionMode,
    future::{DynFut, spawn_detached_fut},
    profile::ProfileDuration,
};

use super::ComputeChannel;
use crate::{
    logging::ServerLogger,
    memory_management::MemoryUsage,
    server::{
        Binding, BindingWithMeta, Bindings, ComputeServer, CubeCount, Handle, ProfileError,
        ProfilingToken,
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

enum Message<Server>
where
    Server: ComputeServer,
{
    Read(Vec<Binding>, Callback<Vec<Vec<u8>>>),
    ReadTensor(Vec<BindingWithMeta>, Callback<Vec<Vec<u8>>>),
    GetResource(
        Binding,
        Callback<BindingResource<<Server::Storage as ComputeStorage>::Resource>>,
    ),
    Create(Vec<u8>, Callback<Handle>),
    CreateTensor(
        Vec<Vec<u8>>,
        Vec<Vec<usize>>,
        Vec<usize>,
        Callback<Vec<(Handle, Vec<usize>)>>,
    ),
    Empty(usize, Callback<Handle>),
    EmptyTensor(
        Vec<Vec<usize>>,
        Vec<usize>,
        Callback<Vec<(Handle, Vec<usize>)>>,
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
                    Message::Read(bindings, callback) => {
                        let data = server.read(bindings).await;
                        callback.send(data).await.unwrap();
                    }
                    Message::ReadTensor(bindings, callback) => {
                        let data = server.read_tensor(bindings).await;
                        callback.send(data).await.unwrap();
                    }
                    Message::GetResource(binding, callback) => {
                        let data = server.get_resource(binding);
                        callback.send(data).await.unwrap();
                    }
                    Message::Create(data, callback) => {
                        let handle = server.create(&data);
                        callback.send(handle).await.unwrap();
                    }
                    Message::CreateTensor(data, shape, elem_size, callback) => {
                        let data = data.iter().map(|it| it.as_slice()).collect();
                        let shape = shape.iter().map(|it| it.as_slice()).collect();
                        let handle = server.create_tensors(data, shape, elem_size);
                        callback.send(handle).await.unwrap();
                    }
                    Message::Empty(size, callback) => {
                        let handle = server.empty(size);
                        callback.send(handle).await.unwrap();
                    }
                    Message::EmptyTensor(shape, elem_size, callback) => {
                        let shape = shape.iter().map(|it| it.as_slice()).collect();
                        let handle = server.empty_tensors(shape, elem_size);
                        callback.send(handle).await.unwrap();
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
    fn read(&self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        let sender = self.state.sender.clone();

        Box::pin(async move {
            let (callback, response) = async_channel::unbounded();
            sender
                .send(Message::Read(bindings, callback))
                .await
                .unwrap();
            handle_response(response.recv().await)
        })
    }

    fn read_tensor(&self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let sender = self.state.sender.clone();
        Box::pin(async move {
            let (callback, response) = async_channel::unbounded();
            sender
                .send(Message::ReadTensor(bindings, callback))
                .await
                .unwrap();
            handle_response(response.recv().await)
        })
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

    fn create(&self, data: &[u8]) -> Handle {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::Create(data.to_vec(), callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn create_tensors(
        &self,
        data: Vec<&[u8]>,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::CreateTensor(
                data.into_iter().map(|it| it.to_vec()).collect(),
                shape.into_iter().map(|it| it.to_vec()).collect(),
                elem_size,
                callback,
            ))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn empty(&self, size: usize) -> Handle {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::Empty(size, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn empty_tensors(
        &self,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::EmptyTensor(
                shape.into_iter().map(|it| it.to_vec()).collect(),
                elem_size,
                callback,
            ))
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
}

fn handle_response<Response, Err: core::fmt::Debug>(response: Result<Response, Err>) -> Response {
    match response {
        Ok(val) => val,
        Err(err) => panic!("Can't connect to the server correctly {err:?}"),
    }
}
