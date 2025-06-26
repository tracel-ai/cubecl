use super::WgpuResource;
use super::{WgpuStorage, stream::WgpuStream};
use crate::AutoCompiler;
use alloc::sync::Arc;
use cubecl_common::profile::{ProfileDuration, TimingMethod};
use cubecl_core::compute::{CubeTask, DebugInformation};
use cubecl_core::future::DynFut;
use cubecl_core::server::{ProfileError, ProfilingToken};
use cubecl_core::{
    Feature, MemoryConfiguration, WgpuCompilationOptions,
    prelude::*,
    server::{Binding, BindingWithMeta, Bindings, Handle},
};
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::offset_handles;
use cubecl_runtime::{
    memory_management::MemoryDeviceProperties,
    server::{self, ComputeServer},
    storage::BindingResource,
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

    fn read(&mut self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        self.stream.read_buffers(bindings)
    }

    fn get_resource(&mut self, binding: Binding) -> BindingResource<WgpuResource> {
        let resource = self.stream.mem_manage.get_resource(binding.clone());
        BindingResource::new(binding, resource)
    }

    /// When we create a new handle from existing data, we use custom allocations so that we don't
    /// have to execute the current pending tasks.
    ///
    /// This is important, otherwise the compute passes are going to be too small and we won't be able to
    /// fully utilize the GPU.
    fn create(&mut self, data: &[u8]) -> server::Handle {
        self.stream.create(data)
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        self.stream.empty(size as u64)
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

    fn read_tensor(&mut self, bindings: Vec<BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let expected_sizes = bindings
            .iter()
            .map(|it| it.shape.iter().product::<usize>() * it.elem_size)
            .collect::<Vec<_>>();
        let bindings = bindings.into_iter().map(|it| it.binding).collect();
        let data = self.read(bindings);
        Box::pin(async move {
            let mut data = data.await;
            for (data, expected_size) in data.iter_mut().zip(expected_sizes) {
                data.truncate(expected_size);
            }
            data
        })
    }

    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shapes: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let handles_strides = self.empty_tensors(shapes.clone(), elem_size);

        for i in 0..data.len() {
            let data = data[i];
            let (handle, _) = &handles_strides[i];
            self.stream.copy_to_handle(handle.clone(), data);
        }

        handles_strides
    }

    fn empty_tensors(
        &mut self,
        shape: Vec<&[usize]>,
        elem_size: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let align = self.device.limits().min_storage_buffer_offset_alignment as usize;
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
