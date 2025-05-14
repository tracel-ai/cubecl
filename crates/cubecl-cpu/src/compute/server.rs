use cubecl_core::{Feature, prelude::CubeTask, server::ComputeServer};
use cubecl_runtime::logging::ServerLogger;

use crate::CpuCompiler;

use super::storage::CPUStorage;

#[derive(Debug)]
pub struct CpuServer {
    logger: ServerLogger,
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn CubeTask<CpuCompiler>>;
    type Storage = CPUStorage;
    type Feature = Feature;
    type Info = ();

    fn read(
        &mut self,
        bindings: Vec<cubecl_core::server::Binding>,
    ) -> cubecl_core::future::DynFut<Vec<Vec<u8>>> {
        todo!()
    }

    fn read_tensor(
        &mut self,
        bindings: Vec<cubecl_core::server::BindingWithMeta>,
    ) -> cubecl_core::future::DynFut<Vec<Vec<u8>>> {
        todo!()
    }

    fn sync(&mut self) -> cubecl_core::future::DynFut<()> {
        todo!()
    }

    fn get_resource(
        &mut self,
        binding: cubecl_core::server::Binding,
    ) -> cubecl_runtime::storage::BindingResource<
        <Self::Storage as cubecl_runtime::storage::ComputeStorage>::Resource,
    > {
        todo!()
    }

    fn create(&mut self, data: &[u8]) -> cubecl_core::server::Handle {
        todo!()
    }

    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(cubecl_core::server::Handle, Vec<usize>)> {
        todo!()
    }

    fn empty(&mut self, size: usize) -> cubecl_core::server::Handle {
        todo!()
    }

    fn empty_tensors(
        &mut self,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(cubecl_core::server::Handle, Vec<usize>)> {
        todo!()
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: cubecl_core::CubeCount,
        bindings: cubecl_core::server::Bindings,
        kind: cubecl_core::ExecutionMode,
    ) {
        todo!()
    }

    fn flush(&mut self) {
        todo!()
    }

    fn memory_usage(&self) -> cubecl_core::MemoryUsage {
        todo!()
    }

    fn memory_cleanup(&mut self) {
        todo!()
    }

    fn start_profile(&mut self) -> cubecl_core::server::ProfilingToken {
        todo!()
    }

    fn end_profile(
        &mut self,
        token: cubecl_core::server::ProfilingToken,
    ) -> cubecl_core::benchmark::ProfileDuration {
        todo!()
    }
}
