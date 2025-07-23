use std::sync::Arc;

use cubecl_common::profile::ProfileDuration;
use cubecl_core::{
    CubeCount, ExecutionMode, Feature, MemoryUsage,
    compute::CubeTask,
    future::DynFut,
    server::{
        Binding, BindingWithMeta, Bindings, ComputeServer, Handle, ProfileError, ProfilingToken,
    },
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{MemoryManagement, offset_handles},
    storage::{BindingResource, BytesStorage, ComputeStorage},
    timestamp_profiler::TimestampProfiler,
};

use crate::CpuCompiler;

use super::scheduler::Scheduler;

#[derive(Debug)]
pub struct CpuServer {
    ctx: CpuContext,
    scheduler: Scheduler,
    logger: ServerLogger,
}

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
        bindings: Vec<Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + use<> {
        let mut result = Vec::with_capacity(bindings.len());

        for binding in bindings {
            let resource = self
                .ctx
                .memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource");

            let data = resource.read().to_vec();

            result.push(data);
        }
        async move { result }
    }
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn CubeTask<CpuCompiler>>;
    type Storage = BytesStorage;
    type Feature = Feature;
    type Info = ();

    fn read(&mut self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        Box::pin(self.read_async(bindings))
    }

    fn read_tensor(&mut self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let bindings = bindings.into_iter().map(|it| it.binding).collect();
        Box::pin(self.read_async(bindings))
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.ctx.memory_management.cleanup(true)
    }

    fn create(&mut self, data: &[u8]) -> Handle {
        let handle = self.empty(data.len());
        let binding = handle.clone().binding();
        self.copy_to_binding(binding, data);

        handle
    }

    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let handles_strides = self.empty_tensors(shapes.clone(), elem_sizes);
        for i in 0..data.len() {
            let data = data[i];
            let (handle, _) = &handles_strides[i];
            let binding = handle.clone().binding();
            self.copy_to_binding(binding, data);
        }
        handles_strides
    }

    fn empty(&mut self, size: usize) -> Handle {
        let handle = self.ctx.memory_management.reserve(size as u64, None);
        Handle::new(handle, None, None, size as u64)
    }

    fn empty_tensors(
        &mut self,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let align = 8;
        let strides = shape
            .iter()
            .map(|shape| contiguous_strides(shape))
            .collect::<Vec<_>>();
        let sizes = shape
            .iter()
            .map(|it| it.iter().product::<usize>())
            .zip(elem_size)
            .map(|(size, elem_size)| (size * elem_size).next_multiple_of(align))
            .collect::<Vec<_>>();
        let total_size = sizes.iter().sum::<usize>();

        let mem_handle = self.empty(total_size);
        let handles = offset_handles(mem_handle, &sizes);

        handles.into_iter().zip(strides).collect()
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

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
