use cranelift::prelude::FunctionBuilderContext;
use cubecl_core::{
    compute::DebugInformation,
    future,
    prelude::*,
    server::{Binding, Handle},
    Feature, KernelId, MemoryConfiguration, WgpuCompilationOptions,
};
use cubecl_runtime::{
    debug::{DebugLogger, ProfileLevel},
    memory_management::{MemoryDeviceProperties, MemoryManagement},
    server::{self, ComputeServer},
    storage::BindingResource,
    TimestampsError, TimestampsResult,
};
use std::future::Future;
use web_time::Instant;

#[derive(Debug)]
enum KernelTimestamps {
    Inferred { start_time: Instant },
    Disabled,
}

impl KernelTimestamps {
    fn enable(&mut self) {
        if !matches!(self, Self::Disabled) {
            return;
        }

        *self = Self::Inferred {
            start_time: Instant::now(),
        };
    }

    fn disable(&mut self) {
        *self = Self::Disabled;
    }
}

struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    // JITModule::get_finalized_function()?
    func: *const u8,
}

use hashbrown::HashMap;

use crate::compiler::FunctionCompiler;

use super::{storage::CraneliftStorage, CraneliftResource};
#[derive(Debug)]
pub struct CraneliftServer {
    context: CraneliftContext,
    logger: DebugLogger,
}

//Contains the state for
pub(crate) struct CraneliftContext {
    builder_context: FunctionBuilderContext,
    codegen_context: cranelift_codegen::Context,
    timestamp: KernelTimestamps,
    memory_management: MemoryManagement<CraneliftStorage>,
    modules: HashMap<KernelId, cranelift_jit::JITModule>,
}

impl CraneliftContext {
    fn execute_task(&mut self, kernel_id: KernelId, resources: Vec<CraneliftResource>) {
        //let kernel: &CompiledKernel = self.modules.get(&kernel_id);
        todo!()
    }
}

impl ComputeServer for CraneliftServer {
    type Kernel = Box<dyn CubeTask<FunctionCompiler>>;
    type Storage = CraneliftStorage;
    type Feature = Feature;

    fn read(
        &mut self,
        bindings: Vec<Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + 'static {
        let mut result = Vec::with_capacity(bindings.len());
        result.extend(bindings.into_iter().map(|binding| {
            let rb = self.get_resource(binding);
            let resource = rb.resource();
            Vec::<u8>::from(resource)
        }));
        async move { result }
    }

    fn get_resource(&mut self, binding: Binding) -> BindingResource<Self> {
        BindingResource::new(
            binding.clone(),
            self.context
                .memory_management
                .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                .expect("Failed to find resource"),
        )
    }

    fn create(&mut self, data: &[u8]) -> Handle {
        let alloc_handle = self.empty(data.len());
        let alloc_binding = alloc_handle.clone().binding();
        // maybe use rayon here?
        let resource_dest = self
            .context
            .memory_management
            .get_resource(
                alloc_binding.memory,
                alloc_binding.offset_start,
                alloc_binding.offset_end,
            )
            .expect("Failed to find resource");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), resource_dest.ptr, data.len());
        }
        alloc_handle
    }

    fn empty(&mut self, size: usize) -> Handle {
        let alloc_handle = self.context.memory_management.reserve(size as u64);
        Handle::new(alloc_handle, None, None, size as u64)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Vec<Binding>,
        kind: ExecutionMode,
    ) {
        // Note: Maybe this can be a function/trait in cubecl-core?
        // Check for any profiling work to be done before execution.
        let profile_level = self.logger.profile_level();
        let profile_info = if profile_level.is_some() {
            Some((kernel.name(), kernel.id()))
        } else {
            None
        };

        if let Some(level) = profile_level {
            todo!()
        } else {
        }

        //match count if needed
    }

    fn flush(&mut self) {
        todo!()
    }

    fn sync(&mut self) -> impl std::future::Future<Output = ()> + Send + 'static {
        self.logger.profile_summary();
        async move { todo!() }
    }

    fn sync_elapsed(
        &mut self,
    ) -> impl std::future::Future<Output = TimestampsResult> + Send + 'static {
        async move { todo!() }
    }

    fn memory_usage(&self) -> cubecl_core::MemoryUsage {
        self.context.memory_management.memory_usage()
    }

    fn enable_timestamps(&mut self) {
        self.context.timestamp.enable();
    }

    fn disable_timestamps(&mut self) {
        self.context.timestamp.disable();
    }
}

impl alloc::fmt::Debug for CraneliftContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //None of the fields implement Debug. It might be possible to
        //display some state, but I haven't worked out how to do that yet.
        f.debug_struct("CraneliftContext").finish()
    }
}
