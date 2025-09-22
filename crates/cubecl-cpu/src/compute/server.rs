use std::sync::Arc;

use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
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
    memory_management::{MemoryManagement, offset_handles},
    storage::{BindingResource, BytesStorage, ComputeStorage},
    timestamp_profiler::TimestampProfiler,
};

use crate::{CpuCompiler, compute::alloc_controller::CpuAllocController};
use cubecl_runtime::stride::{contiguous_strides, pitched_rows_layout, row_pitch_elems};

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
                let binding = desc.binding;
                let elem = desc.elem_size;
                let size = desc.shape.iter().product::<usize>() * elem;

                // Contiguous: return zero-copy Bytes over the binding with logical len
                if contiguous_strides(desc.shape) == desc.strides {
                    let (controller, alloc) =
                        CpuAllocController::init(binding, &mut ctx.memory_management)?;
                    result
                        .push(unsafe { Bytes::from_raw_parts(alloc, size, Box::new(controller)) });
                    continue;
                }

                // Inner-contiguous rows: reconstruct rows into contiguous buffer
                if let Some(row_pitch_elems) = row_pitch_elems(desc.shape, desc.strides) {
                    let resource = ctx
                        .memory_management
                        .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                        .ok_or(IoError::InvalidHandle)?;
                    let last = desc.shape.len() - 1;
                    let rows = desc.shape[..last].iter().product::<usize>();
                    let cols = desc.shape[last];
                    let row_bytes = cols * elem;
                    let row_pitch = row_pitch_elems * elem;
                    let src = resource.read();
                    let mut out = vec![0u8; rows * row_bytes];
                    for r in 0..rows {
                        let src_off = r * row_pitch;
                        let dst_off = r * row_bytes;
                        out[dst_off..dst_off + row_bytes]
                            .copy_from_slice(&src[src_off..src_off + row_bytes]);
                    }
                    result.push(Bytes::from_bytes_vec(out));
                    continue;
                }

                return Err(IoError::UnsupportedStrides);
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
    ) -> Result<Vec<Allocation>, IoError> {
        let align = 8;
        let mut strides = Vec::with_capacity(descriptors.len());
        let mut sizes = Vec::with_capacity(descriptors.len());

        use cubecl_core::server::AllocationKind;

        for desc in &descriptors {
            let rank = desc.shape.len();
            if matches!(desc.kind, AllocationKind::Optimized) && rank > 1 {
                let (s, size) = pitched_rows_layout(desc.shape, desc.elem_size, align);
                strides.push(s);
                sizes.push(size);
            } else {
                strides.push(contiguous_strides(desc.shape));
                sizes.push(desc.shape.iter().product::<usize>() * desc.elem_size);
            }
        }
        let total_size = sizes
            .iter()
            .map(|it| it.next_multiple_of(align))
            .sum::<usize>();

        let handle = self.ctx.memory_management.reserve(total_size as u64)?;
        let mem_handle = Handle::new(handle, None, None, total_size as u64);
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
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        Box::pin(self.read_async(descriptors))
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError> {
        for (desc, data) in descriptors {
            // Contiguous path
            if contiguous_strides(desc.shape) == desc.strides {
                self.copy_to_binding(desc.binding, data);
                continue;
            }

            // Inner-contiguous rows: copy into pitched destination row-by-row
            if let Some(row_pitch_elems) = row_pitch_elems(desc.shape, desc.strides) {
                let last = desc.shape.len() - 1;
                let rows = desc.shape[..last].iter().product::<usize>();
                let cols = desc.shape[last];
                let elem = desc.elem_size;
                let row_bytes = cols * elem;
                let row_pitch = row_pitch_elems * elem;

                let resource = self
                    .ctx
                    .memory_management
                    .get_resource(
                        desc.binding.memory,
                        desc.binding.offset_start,
                        desc.binding.offset_end,
                    )
                    .ok_or(IoError::InvalidHandle)?;

                let dst = resource.write();
                for r in 0..rows {
                    let dst_off = r * row_pitch;
                    let src_off = r * row_bytes;
                    dst[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&data[src_off..src_off + row_bytes]);
                }
                continue;
            }

            return Err(IoError::UnsupportedStrides);
        }
        Ok(())
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.ctx.memory_management.cleanup(true)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        _logger: Arc<ServerLogger>,
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

    fn flush(&mut self) {}

    fn sync(&mut self) -> DynFut<()> {
        self.logger.profile_summary();
        Box::pin(async move {})
    }

    fn start_profile(&mut self) -> ProfilingToken {
        cubecl_common::future::block_on(self.sync());
        self.ctx.timestamps.start()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        self.logger.profile_summary();
        cubecl_common::future::block_on(self.sync());
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: Binding,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        BindingResource::new(
            binding.clone(),
            self.ctx
                .memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Can't find resource"),
        )
    }

    fn allocation_mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
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
