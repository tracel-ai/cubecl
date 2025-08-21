use super::WgpuResource;
use super::{WgpuStorage, stream::WgpuStream};
use crate::AutoCompiler;
use alloc::sync::Arc;
use cubecl_common::profile::{ProfileDuration, TimingMethod};
use cubecl_core::future::DynFut;
use cubecl_core::server::{ProfileError, ProfilingToken};
use cubecl_core::{
    Feature, MemoryConfiguration, WgpuCompilationOptions,
    prelude::*,
    server::{Binding, Bindings, CopyDescriptor},
};
use cubecl_core::{
    compute::{CubeTask, DebugInformation},
    server::{Allocation, AllocationDescriptor, IoError},
};
use cubecl_runtime::data_service::ComputeDataTransferId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::offset_handles;
use cubecl_runtime::{
    memory_management::MemoryDeviceProperties, server::ComputeServer, storage::BindingResource,
};
use hashbrown::HashMap;
use wgpu::ComputePipeline;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    stream: WgpuStream,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
}

impl WgpuServer {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
    ) -> Self {
        let stream = WgpuStream::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            tasks_max,
        );

        Self {
            compilation_options,
            device,
            pipelines: HashMap::new(),
            stream,
            backend,
        }
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Arc<ComputePipeline> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return pipeline.clone();
        }

        let mut compiler = compiler(self.backend);
        let mut compile = compiler.compile(self, kernel, mode);

        if logger.compilation_activated() {
            compile.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        logger.log_compilation(&compile);
        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // println!("SOURCE:\n{}", compile.source);
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compile.source).expect("should write to file");
        //     let _status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     // std::process::exit(status.code().unwrap());
        // }
        let pipeline = self.create_pipeline(compile, mode);
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        pipeline
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn CubeTask<AutoCompiler>>;
    type Storage = WgpuStorage;
    type Feature = Feature;
    type Info = wgpu::Backend;

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError> {
        let align = self.device.limits().min_storage_buffer_offset_alignment as usize;
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

        let mem_handle = self.stream.empty(total_size as u64)?;
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
    ) -> DynFut<Result<Vec<Vec<u8>>, IoError>> {
        for desc in &descriptors {
            if contiguous_strides(desc.shape) != desc.strides {
                return Box::pin(async { Err(IoError::UnsupportedStrides) });
            }
        }
        self.stream.read_buffers(descriptors)
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor<'_>, &[u8])>) -> Result<(), IoError> {
        for (desc, data) in descriptors {
            if contiguous_strides(desc.shape) != desc.strides {
                return Err(IoError::UnsupportedStrides);
            }
            self.stream.write(desc.binding, data);
        }
        Ok(())
    }

    fn get_resource(&mut self, binding: Binding) -> BindingResource<WgpuResource> {
        let resource = self.stream.mem_manage.get_resource(binding.clone());
        BindingResource::new(binding, resource)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) {
        let pipeline = self.pipeline(kernel, mode, logger);
        self.stream.register(pipeline, bindings, &count);
    }

    fn flush(&mut self) {
        // End the current compute pass.
        self.stream.flush();
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self) -> DynFut<()> {
        self.stream.sync()
    }

    fn start_profile(&mut self) -> ProfilingToken {
        self.stream.start_profile()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        self.stream.end_profile(token)
    }

    fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.stream.mem_manage.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.stream.mem_manage.memory_cleanup(true);
    }

    fn allocation_mode(&mut self, mode: cubecl_runtime::memory_management::MemoryAllocationMode) {
        self.stream.mem_manage.mode(mode);
    }

    fn send_to_peer(
        &mut self,
        _id: ComputeDataTransferId,
        _src: CopyDescriptor<'_>,
    ) -> Result<(), IoError> {
        todo!("Peer-to-peer data service unimplemented for WGPU backend")
    }

    fn recv_from_peer(
        &mut self,
        _id: ComputeDataTransferId,
        _dst: CopyDescriptor<'_>,
    ) -> Result<(), IoError> {
        todo!("Peer-to-peer data service unimplemented for WGPU backend")
    }
}

fn compiler(backend: wgpu::Backend) -> AutoCompiler {
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan => AutoCompiler::SpirV(Default::default()),
        #[cfg(feature = "msl")]
        wgpu::Backend::Metal => AutoCompiler::Msl(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
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
