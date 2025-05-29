use std::sync::Arc;

use cubecl_core::{
    CubeCount, ExecutionMode, Feature, MemoryUsage,
    benchmark::ProfileDuration,
    compute::CubeTask,
    future::DynFut,
    server::{Binding, BindingWithMeta, Bindings, ComputeServer, Handle, ProfilingToken},
};
use cubecl_runtime::{
    kernel_timestamps::KernelTimestamps,
    logging::ServerLogger,
    memory_management::MemoryManagement,
    storage::{BindingResource, BytesStorage, ComputeStorage},
};

use crate::{CpuCompiler, compiler::MlirCompilerOptions};

#[derive(Debug)]
pub struct CpuServer {
    ctx: CpuContext,
    logger: ServerLogger,
}

impl CpuServer {
    pub fn new(ctx: CpuContext) -> Self {
        Self {
            logger: ServerLogger::default(),
            ctx,
        }
    }
}

#[derive(Debug)]
pub struct CpuContext {
    memory_management: MemoryManagement<BytesStorage>,
    timestamps: KernelTimestamps,
}

impl CpuContext {
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self {
            memory_management,
            timestamps: KernelTimestamps::default(),
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
        _shapes: Vec<&[usize]>,
        _elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        todo!("Check how strides should be done on CPU")
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        _logger: Arc<ServerLogger>,
    ) {
        // TODO implement the runtime
        let kernel = kernel.compile(
            &mut Default::default(),
            &MlirCompilerOptions::default(),
            kind,
        );
        let mut execution_engine = kernel.repr.unwrap();
        let buffers = bindings.buffers.clone();
        for binding in buffers.into_iter() {
            let handle = self
                .ctx
                .memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource");
            let ptr = handle.write();
            unsafe {
                execution_engine.push_buffer(ptr);
            }
        }
        let cube_count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            CubeCount::Dynamic(_binding) => todo!("Needs to figure it later"),
        };
        execution_engine.push_builtin();
        execution_engine.builtin.set_cube_count(cube_count);
        let (cube_dim_x, cube_dim_y, cube_dim_z) = execution_engine.builtin.get_cube_dim();
        // Will be multithreaded later
        for unit_pos_x in 0..cube_dim_x {
            execution_engine.builtin.set_unit_pos_x(unit_pos_x);
            for unit_pos_y in 0..cube_dim_y {
                execution_engine.builtin.set_unit_pos_y(unit_pos_y);
                for unit_pos_z in 0..cube_dim_z {
                    execution_engine.builtin.set_unit_pos_z(unit_pos_z);
                    unsafe {
                        execution_engine.run_kernel();
                    }
                }
            }
        }
    }

    fn flush(&mut self) {}

    // TODO find when task are finish to be scheduled
    fn sync(&mut self) -> DynFut<()> {
        self.logger.profile_summary();
        Box::pin(async move {})
    }

    fn start_profile(&mut self) -> ProfilingToken {
        cubecl_common::future::block_on(self.sync());
        self.ctx.timestamps.start()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> ProfileDuration {
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
