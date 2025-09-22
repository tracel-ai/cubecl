use super::ComputeChannel;
use crate::{
    data_service::DataTransferId,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    server::{
        Allocation, AllocationDescriptor, AllocationKind, Binding, Bindings, ComputeServer,
        CopyDescriptor, CubeCount, IoError, ProfileError, ProfilingToken,
    },
    storage::{BindingResource, ComputeStorage},
    stream::scheduler::{SchedulerMultiStream, SchedulerStreamBackend, SchedulerTask},
};
use async_channel::Receiver;
use cubecl_common::{
    ExecutionMode,
    bytes::Bytes,
    future::{DynFut, spawn_detached_fut},
    profile::ProfileDuration,
    stream_id::StreamId,
};
use std::sync::Arc;

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
    type_: AllocationKind,
    shape: Vec<usize>,
    elem_size: usize,
}

impl From<AllocationDescriptor<'_>> for AllocationDescriptorOwned {
    fn from(value: AllocationDescriptor) -> Self {
        AllocationDescriptorOwned {
            type_: value.kind,
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
    binding: Binding,
    shape: Vec<usize>,
    strides: Vec<usize>,
    elem_size: usize,
}

impl From<CopyDescriptor<'_>> for CopyDescriptorOwned {
    fn from(value: CopyDescriptor<'_>) -> Self {
        CopyDescriptorOwned {
            binding: value.binding,
            shape: value.shape.to_vec(),
            strides: value.strides.to_vec(),
            elem_size: value.elem_size,
        }
    }
}

impl CopyDescriptorOwned {
    fn as_ref(&self) -> CopyDescriptor<'_> {
        CopyDescriptor::new(
            self.binding.clone(),
            &self.shape,
            &self.strides,
            self.elem_size,
        )
    }
}

impl<S: ComputeServer> core::fmt::Debug for Message<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Create(..) => f.write_str("Create"),
            Self::Read(..) => f.write_str("Read"),
            Self::Write(..) => f.write_str("Write"),
            Self::GetResource(..) => f.write_str("GetResource"),
            Self::ExecuteKernel(..) => f.write_str("ExecuteKernel"),
            Self::Flush(..) => f.write_str("Flush"),
            Self::Sync(..) => f.write_str("Sync"),
            Self::MemoryUsage(..) => f.write_str("MemoryUsage"),
            Self::MemoryCleanup(..) => f.write_str("MemoryCleanup"),
            Self::AllocationMode(..) => f.write_str("AllocationMode"),
            Self::StartProfile(..) => f.write_str("StartProfile"),
            Self::StopMeasure(..) => f.write_str("StopMeasure"),
            Self::DataTransferSend(..) => f.write_str("DataTransferSend"),
            Self::DataTransferRecv(..) => f.write_str("DataTransferRecv"),
        }
    }
}

impl<S: ComputeServer> SchedulerTask for Message<S> {
    fn stream_id(&self) -> StreamId {
        match self {
            Message::Create(_, stream_id, ..) => *stream_id,
            Message::Read(_, stream_id, ..) => *stream_id,
            Message::Write(_, stream_id, ..) => *stream_id,
            Message::GetResource(_, stream_id, ..) => *stream_id,
            Message::ExecuteKernel((_, _, _, stream_id), ..) => *stream_id,
            Message::Flush(stream_id) => *stream_id,
            Message::Sync(stream_id, ..) => *stream_id,
            Message::MemoryUsage(stream_id, ..) => *stream_id,
            Message::MemoryCleanup(stream_id) => *stream_id,
            Message::AllocationMode(stream_id, ..) => *stream_id,
            Message::StartProfile(stream_id, ..) => *stream_id,
            Message::StopMeasure(_, stream_id, ..) => *stream_id,
            Message::DataTransferSend(stream_id, ..) => *stream_id,
            Message::DataTransferRecv(stream_id, ..) => *stream_id,
        }
    }

    fn bindings<'a>(&'a self) -> Vec<&'a Binding> {
        match self {
            Message::Create(..) => Vec::new(),
            Message::Read(copy_descriptor_owneds, ..) => {
                copy_descriptor_owneds.iter().map(|c| &c.binding).collect()
            }
            Message::Write(items, ..) => items.iter().map(|item| &item.0.binding).collect(),
            Message::GetResource(binding, ..) => {
                vec![&binding]
            }
            Message::ExecuteKernel(_, bindings, ..) => bindings.buffers.iter().collect(),
            Message::Flush(..) => vec![],
            Message::Sync(..) => vec![],
            Message::MemoryUsage(..) => vec![],
            Message::MemoryCleanup(..) => vec![],
            Message::AllocationMode(..) => vec![],
            Message::StartProfile(..) => vec![],
            Message::StopMeasure(..) => vec![],
            Message::DataTransferSend(_, _, copy_descriptor_owned) => {
                vec![&copy_descriptor_owned.binding]
            }
            Message::DataTransferRecv(_, _, copy_descriptor_owned) => {
                vec![&copy_descriptor_owned.binding]
            }
        }
    }
}

enum Message<Server>
where
    Server: ComputeServer,
{
    Create(
        Vec<AllocationDescriptorOwned>,
        StreamId,
        Callback<Result<Vec<Allocation>, IoError>>,
    ),
    Read(
        Vec<CopyDescriptorOwned>,
        StreamId,
        Callback<Result<Vec<Bytes>, IoError>>,
    ),
    Write(
        Vec<(CopyDescriptorOwned, Vec<u8>)>,
        StreamId,
        Callback<Result<(), IoError>>,
    ),
    GetResource(
        Binding,
        StreamId,
        Callback<BindingResource<<Server::Storage as ComputeStorage>::Resource>>,
    ),
    ExecuteKernel(
        (Server::Kernel, CubeCount, ExecutionMode, StreamId),
        Bindings,
        Arc<ServerLogger>,
    ),
    Flush(StreamId),
    Sync(StreamId, Callback<()>),
    MemoryUsage(StreamId, Callback<MemoryUsage>),
    MemoryCleanup(StreamId),
    AllocationMode(StreamId, MemoryAllocationMode),
    StartProfile(StreamId, Callback<ProfilingToken>),
    StopMeasure(
        Callback<Result<ProfileDuration, ProfileError>>,
        StreamId,
        ProfilingToken,
    ),
    DataTransferSend(StreamId, DataTransferId, CopyDescriptorOwned),
    DataTransferRecv(StreamId, DataTransferId, CopyDescriptorOwned),
}

impl<Server> MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    /// Create a new mpsc compute channel.
    pub fn new(server: Server) -> Self {
        let (sender, receiver) = async_channel::unbounded();

        let server = MpscServer { server };
        spawn_detached_fut(server.run(receiver));

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
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let descriptors = descriptors.into_iter().map(|it| it.into()).collect();

        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::Create(descriptors, stream_id, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn read(
        &self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let sender = self.state.sender.clone();
        let descriptors = descriptors.into_iter().map(|it| it.into()).collect();

        Box::pin(async move {
            let (callback, response) = async_channel::unbounded();
            sender
                .send(Message::Read(descriptors, stream_id, callback))
                .await
                .unwrap();
            handle_response(response.recv().await)
        })
    }

    fn write(
        &self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let descriptors = descriptors
            .into_iter()
            .map(|(desc, data)| (desc.into(), data.to_vec()))
            .collect();

        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::Write(descriptors, stream_id, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn data_transfer_send(&self, id: DataTransferId, src: CopyDescriptor<'_>, stream_id: StreamId) {
        let sender = self.state.sender.clone();
        let src = src.into();

        sender
            .send_blocking(Message::DataTransferSend(stream_id, id, src))
            .unwrap();
    }

    fn data_transfer_recv(&self, id: DataTransferId, dst: CopyDescriptor<'_>, stream_id: StreamId) {
        let sender = self.state.sender.clone();
        let dst = dst.into();

        sender
            .send_blocking(Message::DataTransferRecv(stream_id, id, dst))
            .unwrap();
    }

    fn get_resource(
        &self,
        binding: Binding,
        stream_id: StreamId,
    ) -> BindingResource<<Server::Storage as ComputeStorage>::Resource> {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::GetResource(binding, stream_id, callback))
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
        stream_id: StreamId,
    ) {
        self.state
            .sender
            .send_blocking(Message::ExecuteKernel(
                (kernel, count, kind, stream_id),
                bindings,
                logger,
            ))
            .unwrap();
    }

    fn flush(&self, stream_id: StreamId) {
        self.state
            .sender
            .send_blocking(Message::Flush(stream_id))
            .unwrap()
    }

    fn sync(&self, stream_id: StreamId) -> DynFut<()> {
        let sender = self.state.sender.clone();

        Box::pin(async move {
            let (callback, response) = async_channel::unbounded();
            sender
                .send(Message::Sync(stream_id, callback))
                .await
                .unwrap();
            handle_response(response.recv().await)
        })
    }

    fn memory_usage(&self, stream_id: StreamId) -> crate::memory_management::MemoryUsage {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::MemoryUsage(stream_id, callback))
            .unwrap();
        handle_response(response.recv_blocking())
    }

    fn memory_cleanup(&self, stream_id: StreamId) {
        self.state
            .sender
            .send_blocking(Message::MemoryCleanup(stream_id))
            .unwrap()
    }

    fn start_profile(&self, stream_id: StreamId) -> ProfilingToken {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::StartProfile(stream_id, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn end_profile(
        &self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::StopMeasure(callback, stream_id, token))
            .unwrap();
        handle_response(response.recv_blocking())
    }

    fn allocation_mode(
        &self,
        mode: crate::memory_management::MemoryAllocationMode,
        stream_id: StreamId,
    ) {
        self.state
            .sender
            .send_blocking(Message::AllocationMode(stream_id, mode))
            .unwrap()
    }
}

fn handle_response<Response, Err: core::fmt::Debug>(response: Result<Response, Err>) -> Response {
    match response {
        Ok(val) => val,
        Err(err) => panic!("Can't connect to the server correctly {err:?}"),
    }
}

struct MpscServer<S: ComputeServer> {
    server: S,
}

impl<S: ComputeServer + 'static> SchedulerStreamBackend for MpscServer<S> {
    type Task = Message<S>;

    async fn execute(&mut self, tasks: impl Iterator<Item = Self::Task>) {
        for task in tasks {
            self.process(task).await;
        }
    }
}

impl<S: ComputeServer + 'static> MpscServer<S> {
    async fn run(self, receiver: Receiver<Message<S>>) {
        let mut sm = SchedulerMultiStream::new(self, 8);

        while let Ok(message) = receiver.recv().await {
            sm.register(message).await;
        }
    }

    async fn process(&mut self, message: Message<S>) {
        match message {
            Message::Create(descriptors, stream_id, callback) => {
                let descriptors = descriptors.iter().map(|it| it.as_ref()).collect();
                let data = self.server.create(descriptors, stream_id);
                callback.send(data).await.unwrap();
            }
            Message::Read(descriptors, stream, callback) => {
                let descriptors = descriptors.iter().map(|it| it.as_ref()).collect();
                let data = self.server.read(descriptors, stream).await;
                callback.send(data).await.unwrap();
            }
            Message::Write(descriptors, stream, callback) => {
                let descriptors = descriptors
                    .iter()
                    .map(|(desc, data)| (desc.as_ref(), data.as_slice()))
                    .collect();
                let data = self.server.write(descriptors, stream);
                callback.send(data).await.unwrap();
            }
            Message::GetResource(binding, stream, callback) => {
                let data = self.server.get_resource(binding, stream);
                callback.send(data).await.unwrap();
            }
            Message::ExecuteKernel(kernel, bindings, logger) => unsafe {
                self.server
                    .execute(kernel.0, kernel.1, bindings, kernel.2, logger, kernel.3);
            },
            Message::Sync(stream, callback) => {
                self.server.sync(stream).await;
                callback.send(()).await.unwrap();
            }
            Message::Flush(stream) => {
                self.server.flush(stream);
            }
            Message::MemoryUsage(stream, callback) => {
                callback
                    .send(self.server.memory_usage(stream))
                    .await
                    .unwrap();
            }
            Message::MemoryCleanup(stream) => {
                self.server.memory_cleanup(stream);
            }
            Message::StartProfile(stream_id, callback) => {
                let token = self.server.start_profile(stream_id);
                callback.send(token).await.unwrap();
            }
            Message::StopMeasure(callback, stream_id, token) => {
                callback
                    .send(self.server.end_profile(stream_id, token))
                    .await
                    .unwrap();
            }
            Message::AllocationMode(stream, mode) => {
                self.server.allocation_mode(mode, stream);
            }
            Message::DataTransferSend(stream, id, src) => {
                self.server.register_src(stream, id, src.as_ref());
            }
            Message::DataTransferRecv(stream, id, dst) => {
                self.server.register_dest(stream, id, dst.as_ref());
            }
        };
    }
}
