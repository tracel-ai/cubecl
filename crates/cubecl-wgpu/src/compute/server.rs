use std::{future::Future, time::Duration};

use super::WgpuResource;
use super::{WgpuStorage, stream::WgpuStream};
use crate::AutoCompiler;
use alloc::sync::Arc;
use cubecl_common::future;
use cubecl_core::benchmark::ProfileDuration;
use cubecl_core::{
    Feature, KernelId, MemoryConfiguration, WgpuCompilationOptions,
    compute::DebugInformation,
    prelude::*,
    server::{Binding, BindingWithMeta, Bindings, Handle},
};
use cubecl_runtime::TimeMeasurement;
use cubecl_runtime::{
    debug::{DebugLogger, ProfileLevel},
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
    logger: DebugLogger,
    duration_profiled: Option<Duration>,
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
        time_measurement: TimeMeasurement,
    ) -> Self {
        let logger = DebugLogger::default();

        let stream = WgpuStream::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            tasks_max,
            time_measurement,
        );

        Self {
            compilation_options,
            device,
            pipelines: HashMap::new(),
            logger,
            duration_profiled: None,
            stream,
            backend,
        }
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Arc<ComputePipeline> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return pipeline.clone();
        }

        let mut compiler = compiler(self.backend);
        let mut compile = compiler.compile(self, kernel, mode);

        if self.logger.is_activated() {
            compile.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        let compile = self.logger.debug(compile);
        // /!\ Do not delete the following commented code.
        // This is usefull while working on the metal compiler.
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

    fn read(
        &mut self,
        bindings: Vec<Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + 'static {
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
    ) {
        // Check for any profiling work to be done before execution.
        let profile_level = self.logger.profile_level();
        let profile_info = if profile_level.is_some() {
            Some((kernel.name(), kernel.id()))
        } else {
            None
        };

        let currently_profiling = self.stream.is_profiling();

        if profile_level.is_some() {
            // Add in current time if currently profiling.
            if currently_profiling {
                let profile = self.stream.stop_profile();
                let duration = future::block_on(profile.resolve());
                self.duration_profiled =
                    Some(self.duration_profiled.unwrap_or_default() + duration);
            }

            self.stream.start_profile();
        }

        // Start execution.
        let pipeline = self.pipeline(kernel, mode);
        self.stream.register(pipeline, bindings, &count);

        // If profiling, write out results.
        if let Some(level) = profile_level {
            let profile = self.stream.stop_profile();
            // Execute the task.
            let duration = future::block_on(profile.resolve());

            let (name, kernel_id) = profile_info.unwrap();

            self.duration_profiled = Some(self.duration_profiled.unwrap_or_default() + duration);

            let info = match level {
                ProfileLevel::Basic | ProfileLevel::Medium => {
                    if let Some(val) = name.split("<").next() {
                        val.split("::").last().unwrap_or(name).to_string()
                    } else {
                        name.to_string()
                    }
                }
                ProfileLevel::Full => {
                    format!("{name}: {kernel_id} CubeCount {count:?}")
                }
            };
            self.logger.register_profiled(info, duration);

            // Restart profile if currently profiling.
            if currently_profiling {
                self.stream.start_profile();
            }
        }
    }

    fn flush(&mut self) {
        // End the current compute pass.
        self.stream.flush();
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self) -> impl Future<Output = ()> + 'static {
        self.logger.profile_summary();
        self.stream.sync()
    }

    fn start_profile(&mut self) {
        self.stream.start_profile();
    }

    fn end_profile(&mut self) -> ProfileDuration {
        self.logger.profile_summary();

        // TODO: Deal with BS recursive profile thing...
        let profile = self.stream.stop_profile();
        let duration_profiled = self.duration_profiled;
        self.duration_profiled = None;

        // Add in profiled duration if needed.
        ProfileDuration::from_future(async move {
            profile.resolve().await + duration_profiled.unwrap_or(Duration::from_secs(0))
        })
    }

    fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.stream.mem_manage.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.stream.mem_manage.memory_cleanup(true);
    }

    fn read_tensor(
        &mut self,
        bindings: Vec<BindingWithMeta>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + 'static {
        let bindings = bindings.into_iter().map(|it| it.binding).collect();
        self.read(bindings)
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
        let align = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        let mut strides = Vec::new();
        let mut total_size = 0;

        for (shape, elem_size) in shape.into_iter().zip(elem_size) {
            let size = shape.iter().product::<usize>();
            let size_bytes = (size * elem_size).next_multiple_of(align);
            total_size += size_bytes;
            strides.push((size_bytes, contiguous_strides(shape)));
        }

        let mut offset = 0;
        let mut out = Vec::new();
        let mem_handle = self.empty(total_size);

        for (size, strides) in strides {
            let handle = mem_handle
                .clone()
                .offset_start(offset as u64)
                .offset_end((total_size - offset - size) as u64);
            out.push((handle, strides));
            offset += size;
        }

        out
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
