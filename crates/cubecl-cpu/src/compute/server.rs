use crate::{
    CpuCompiler,
    compute::stream::{AllocationDescriptorOwned, CopyDescriptorOwned, CpuStream, CpuTask},
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    CubeCount, ExecutionMode, MemoryConfiguration, MemoryUsage,
    compute::CubeTask,
    future::DynFut,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, ComputeServer, CopyDescriptor,
        IoError, ProfileError, ProfilingToken, ServerCommunication, ServerUtilities,
    },
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryDeviceProperties, MemoryManagement},
    storage::{BindingResource, BytesStorage, ComputeStorage},
    timestamp_profiler::TimestampProfiler,
};
use std::sync::{Arc, mpsc::SyncSender};

#[derive(Debug)]
pub struct CpuServer {
    stream: SyncSender<CpuTask>,
    utilities: Arc<ServerUtilities<Self>>,
}

impl CpuServer {
    pub fn new(
        config: MemoryConfiguration,
        properties: &MemoryDeviceProperties,
        logger: Arc<ServerLogger>,
        utilities: Arc<ServerUtilities<Self>>,
    ) -> Self {
        let stream = CpuStream::new(config, properties, logger, utilities.clone());
        let stream = stream.start(StreamId::current());
        Self { utilities, stream }
    }
}

#[derive(Debug)]
pub struct CpuContext {
    pub(crate) memory_management: MemoryManagement<BytesStorage>,
    pub(crate) timestamps: TimestampProfiler,
}

impl CpuContext {
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self {
            memory_management,
            timestamps: TimestampProfiler::default(),
        }
    }
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn CubeTask<CpuCompiler>>;
    type Storage = BytesStorage;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        _stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut descriptors_owned = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            descriptors_owned.push(AllocationDescriptorOwned {
                kind: desc.kind,
                shape: desc.shape.to_vec(),
                elem_size: desc.elem_size,
            });
        }

        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::Create {
                descriptors: descriptors_owned,
                callback: sender,
            })
            .unwrap();

        rec.recv().unwrap()
    }

    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut descriptors_owned = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            descriptors_owned.push(CopyDescriptorOwned {
                strides: desc.strides.to_vec(),
                shape: desc.shape.to_vec(),
                elem_size: desc.elem_size,
                binding: desc.binding,
            });
        }

        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::Read {
                descriptors: descriptors_owned,
                callback: sender,
            })
            .unwrap();

        rec.recv().unwrap()
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        _stream_id: StreamId,
    ) -> Result<(), IoError> {
        let mut descriptors_owned = Vec::with_capacity(descriptors.len());
        for (desc, data) in descriptors {
            descriptors_owned.push((
                CopyDescriptorOwned {
                    strides: desc.strides.to_vec(),
                    shape: desc.shape.to_vec(),
                    elem_size: desc.elem_size,
                    binding: desc.binding,
                },
                data.to_vec(),
            ));
        }

        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::Write {
                descriptors: descriptors_owned,
                callback: sender,
            })
            .unwrap();

        rec.recv().unwrap()
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> MemoryUsage {
        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::MemoryUsage { callback: sender })
            .unwrap();
        rec.recv().unwrap()
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {
        self.stream.send(CpuTask::MemoryCleanup).unwrap();
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        _stream_id: StreamId,
    ) {
        self.stream
            .send(CpuTask::Compute {
                kernel,
                count,
                bindings,
                kind,
            })
            .unwrap();
    }

    fn flush(&mut self, _stream_id: StreamId) {
        self.stream.send(CpuTask::Flush).unwrap();
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<()> {
        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::Sync { callback: sender })
            .unwrap();
        Box::pin(async move {
            rec.recv().unwrap();
        })
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> ProfilingToken {
        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::StartProfile { callback: sender })
            .unwrap();
        rec.recv().unwrap()
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::EndProfile {
                token,
                callback: sender,
            })
            .unwrap();
        rec.recv().unwrap()
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        _stream_id: StreamId,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.stream
            .send(CpuTask::GetResource {
                binding,
                callback: sender,
            })
            .unwrap();
        rec.recv().unwrap()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, _stream_id: StreamId) {
        self.stream.send(CpuTask::AllocMode { mode }).unwrap();
    }
}

impl ServerCommunication for CpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
