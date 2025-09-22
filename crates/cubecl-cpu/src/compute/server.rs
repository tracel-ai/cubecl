use std::sync::Arc;

use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    CubeCount, ExecutionMode, MemoryUsage,
    compute::CubeTask,
    future::DynFut,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, ComputeServer, CopyDescriptor,
        DataTransferService, Handle, IoError, ProfileError, ProfilingToken,
    },
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryManagement, offset_handles},
    storage::{BindingResource, BytesStorage, ComputeStorage},
    timestamp_profiler::TimestampProfiler,
};

use crate::{CpuCompiler, compute::alloc_controller::CpuAllocController};

use super::scheduler::Scheduler;

#[derive(Debug)]
pub struct CpuServer {
    ctx: CpuContext,
    scheduler: Scheduler,
    logger: ServerLogger,
}

impl DataTransferService for CpuServer {}

impl CpuServer {
    pub fn new(ctx: CpuContext) -> Self {
        Self {
            logger: ServerLogger::default(),
            scheduler: Scheduler::default(),
            ctx,
        }
    }
}

#[derive(Debug)]
pub struct CpuContext {
    memory_management: MemoryManagement<BytesStorage>,
    timestamps: TimestampProfiler,
}

impl CpuContext {
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self {
            memory_management,
            timestamps: TimestampProfiler::default(),
        }
    }
}

impl CpuServer {
    fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        fn inner(
            ctx: &mut CpuContext,
            descriptors: Vec<CopyDescriptor>,
        ) -> Result<Vec<Bytes>, IoError> {
            let mut result = Vec::with_capacity(descriptors.len());
            for desc in descriptors {
                let len = desc.binding.size() as usize;
                let (controller, alloc) =
                    CpuAllocController::init(desc.binding, &mut ctx.memory_management)?;
                result.push(unsafe { Bytes::from_raw_parts(alloc, len, Box::new(controller)) });
            }
            Ok(result)
        }

        let res = inner(&mut self.ctx, descriptors);

        async move { res }
    }
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn CubeTask<CpuCompiler>>;
    type Storage = BytesStorage;
    type Info = ();

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let align = 8;
        let strides = descriptors
            .iter()
            .map(|desc| contiguous_strides(desc.shape))
            .collect::<Vec<_>>();
        let sizes = descriptors
            .iter()
            .map(|desc| desc.shape.iter().product::<usize>() * desc.elem_size)
            .collect::<Vec<_>>();
        let total_size = sizes
            .iter()
            .map(|it| it.next_multiple_of(align))
            .sum::<usize>();

        let handle = self.ctx.memory_management.reserve(total_size as u64)?;
        let mem_handle = Handle::new(handle, None, None, stream_id, 0, total_size as u64);
        let handles = offset_handles(mem_handle, &sizes, align);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        Box::pin(self.read_async(descriptors))
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        _stream_id: StreamId,
    ) -> Result<(), IoError> {
        for (desc, data) in descriptors {
            if desc.strides != contiguous_strides(desc.shape) {
                return Err(IoError::UnsupportedStrides);
            }

            self.copy_to_binding(desc.binding, data);
        }
        Ok(())
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> MemoryUsage {
        self.ctx.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {
        self.ctx.memory_management.cleanup(true)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        _logger: Arc<ServerLogger>,
        _stream_id: StreamId,
    ) {
        let cube_count = match count {
            CubeCount::Static(x, y, z) => [x, y, z],
            CubeCount::Dynamic(binding) => {
                let handle = self
                    .ctx
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .expect("Failed to find resource");
                let bytes = handle.read();
                let x = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
                let y = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                let z = u32::from_ne_bytes(bytes[8..12].try_into().unwrap());
                [x, y, z]
            }
        };
        self.scheduler.dispatch_execute(
            kernel,
            cube_count,
            bindings,
            kind,
            &mut self.ctx.memory_management,
        );
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<()> {
        self.logger.profile_summary();
        Box::pin(async move {})
    }

    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken {
        cubecl_common::future::block_on(self.sync(stream_id));
        self.ctx.timestamps.start()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.logger.profile_summary();
        cubecl_common::future::block_on(self.sync(stream_id));
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        _stream_id: StreamId,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        BindingResource::new(
            binding.clone(),
            self.ctx
                .memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Can't find resource"),
        )
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, _stream_id: StreamId) {
        self.ctx.memory_management.mode(mode);
    }
}

impl CpuServer {
    fn copy_to_binding(&mut self, binding: Binding, data: &[u8]) {
        let resource = self
            .ctx
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .unwrap();

        resource.write().copy_from_slice(data);
    }
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
