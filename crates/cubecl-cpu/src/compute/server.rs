use cubecl_core::{
    Feature, MemoryUsage,
    future::DynFut,
    prelude::CubeTask,
    server::{Binding, BindingWithMeta, Bindings, ComputeServer, Handle, ProfilingToken},
};
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::MemoryManagement,
    storage::{BindingResource, BytesStorage, ComputeStorage},
};

use crate::CpuCompiler;

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
}

impl CpuContext {
    pub fn new(memory_management: MemoryManagement<BytesStorage>) -> Self {
        Self { memory_management }
    }
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn CubeTask<CpuCompiler>>;
    type Storage = BytesStorage;
    type Feature = Feature;
    type Info = ();

    fn read(&mut self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        todo!()
    }

    fn read_tensor(&mut self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        todo!()
    }

    fn sync(&mut self) -> DynFut<()> {
        todo!()
    }

    fn get_resource(
        &mut self,
        binding: Binding,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        todo!()
    }

    fn create(&mut self, data: &[u8]) -> Handle {
        todo!()
    }

    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        todo!()
    }

    fn empty(&mut self, size: usize) -> Handle {
        todo!()
    }

    fn empty_tensors(
        &mut self,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        todo!()
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: cubecl_core::CubeCount,
        bindings: Bindings,
        kind: cubecl_core::ExecutionMode,
    ) {
        todo!()
    }

    fn flush(&mut self) {
        todo!()
    }

    fn memory_usage(&self) -> MemoryUsage {
        todo!()
    }

    fn memory_cleanup(&mut self) {
        todo!()
    }

    fn start_profile(&mut self) -> ProfilingToken {
        todo!()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> cubecl_core::benchmark::ProfileDuration {
        todo!()
    }
}
