use std::future::Future;

use cranelift::prelude::FunctionBuilderContext;
use cubecl_core::{
    compute::DebugInformation,
    prelude::*,
    server::{Binding, Handle},
    Feature, KernelId, MemoryConfiguration, WgpuCompilationOptions,
};
use cubecl_runtime::{
    debug::{DebugLogger, ProfileLevel},
    memory_management::MemoryDeviceProperties,
    server::{self, ComputeServer},
    storage::BindingResource,
    TimestampsError, TimestampsResult,
};

struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    // JITModule::get_finalized_function()?
    func: *const u8,
}

use hashbrown::HashMap;

use crate::compiler::FunctionCompiler;

use super::storage::CraneliftStorage;
#[derive(Debug)]
pub struct CraneliftServer {
    context: CraneLiftContext,
    logger: DebugLogger,
}

pub(crate) struct CraneLiftContext {
    builder_context: FunctionBuilderContext,
    codegen_context: cranelift_codegen::Context,
    modules: HashMap<KernelId, cranelift_jit::JITModule>,
}

impl ComputeServer for CraneliftServer {
    type Kernel = Box<dyn for<'a> CubeTask<FunctionCompiler<'a>>>;
    type Storage = CraneliftStorage;
    type Feature = Feature;

    fn read(
        &mut self,
        bindings: Vec<Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + 'static {
        todo!()
    }

    fn get_resource(&mut self, binding: Binding) -> BindingResource<Self> {
        todo!()
    }

    fn create(&mut self, data: &[u8]) -> Handle {
        todo!()
    }

    fn empty(&mut self, size: usize) -> Handle {
        todo!()
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Vec<Binding>,
        kind: ExecutionMode,
    ) {
        todo!()
    }

    fn flush(&mut self) {
        todo!()
    }

    fn sync(&mut self) -> impl std::future::Future<Output = ()> + Send + 'static {
        todo!()
    }

    fn sync_elapsed(
        &mut self,
    ) -> impl std::future::Future<Output = TimestampsResult> + Send + 'static {
        todo!()
    }

    fn memory_usage(&self) -> cubecl_core::MemoryUsage {
        todo!()
    }

    fn enable_timestamps(&mut self) {
        todo!()
    }

    fn disable_timestamps(&mut self) {
        todo!()
    }
}

impl alloc::fmt::Debug for CraneLiftContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //None of the fields implement Debug. It might be possible to
        //display some state, but I haven't worked out how to do that yet.
        f.debug_struct("CraneliftContext").finish()
    }
}
