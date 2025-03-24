use std::{future::Future, time::Duration};

use super::WgpuResource;
use super::{WgpuStorage, stream::WgpuStream};
use crate::AutoCompiler;
use crate::timestamps::KernelTimestamps;
use alloc::sync::Arc;
use cubecl_common::future;
use cubecl_core::{
    Feature, KernelId, MemoryConfiguration, WgpuCompilationOptions,
    compute::DebugInformation,
    prelude::*,
    server::{Binding, BindingWithMeta, ConstBinding, Handle},
};
use cubecl_runtime::{
    TimestampsError, TimestampsResult,
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
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
    ) -> Self {
        let logger = DebugLogger::default();
        let mut timestamps = KernelTimestamps::Disabled;

        if logger.profile_level().is_some() {
            timestamps.enable(&device);
        }

        let stream = WgpuStream::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timestamps,
            tasks_max,
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
            compile.debug_info = Some(DebugInformation::new("wgsl", kernel_id.clone()));
        }

        let compile = self.logger.debug(compile);
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
        constants: Vec<ConstBinding>,
        bindings: Vec<Binding>,
        mode: ExecutionMode,
    ) {
        // Check for any profiling work to be done before execution.
        let profile_level = self.logger.profile_level();
        let profile_info = if profile_level.is_some() {
            Some((kernel.name(), kernel.id()))
        } else {
            None
        };

        if profile_level.is_some() {
            let fut = self.stream.sync_elapsed();
            if let Ok(duration) = future::block_on(fut) {
                if let Some(profiled) = &mut self.duration_profiled {
                    *profiled += duration;
                } else {
                    self.duration_profiled = Some(duration);
                }
            }
        }

        // Start execution.
        let pipeline = self.pipeline(kernel, mode);
        self.stream.register(pipeline, constants, bindings, &count);

        // If profiling, write out results.
        if let Some(level) = profile_level {
            let (name, kernel_id) = profile_info.unwrap();

            // Execute the task.
            if let Ok(duration) = future::block_on(self.stream.sync_elapsed()) {
                if let Some(profiled) = &mut self.duration_profiled {
                    *profiled += duration;
                } else {
                    self.duration_profiled = Some(duration);
                }

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

    /// Returns the total time of GPU work this sync completes.
    fn sync_elapsed(&mut self) -> impl Future<Output = TimestampsResult> + 'static {
        self.logger.profile_summary();

        let fut = self.stream.sync_elapsed();

        let profiled = self.duration_profiled;
        self.duration_profiled = None;

        async move {
            match fut.await {
                Ok(duration) => match profiled {
                    Some(profiled) => Ok(duration + profiled),
                    None => Ok(duration),
                },
                Err(err) => match err {
                    TimestampsError::Disabled => Err(err),
                    TimestampsError::Unavailable => match profiled {
                        Some(profiled) => Ok(profiled),
                        None => Err(err),
                    },
                    TimestampsError::Unknown(_) => Err(err),
                },
            }
        }
    }

    fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.stream.mem_manage.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.stream.mem_manage.memory_cleanup(true);
    }

    fn enable_timestamps(&mut self) {
        self.stream.timestamps.enable(&self.device);
    }

    fn disable_timestamps(&mut self) {
        // Only disable timestamps if profiling isn't enabled.
        if self.logger.profile_level().is_none() {
            self.stream.timestamps.disable();
        }
    }

    fn read_tensor(
        &mut self,
        bindings: Vec<BindingWithMeta>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + 'static {
        let bindings = bindings.into_iter().map(|it| it.binding).collect();
        self.read(bindings)
    }

    fn create_tensor(
        &mut self,
        data: &[u8],
        shape: &[usize],
        _elem_size: usize,
    ) -> (Handle, Vec<usize>) {
        let strides = contiguous_strides(shape);
        let handle = self.create(data);
        (handle, strides)
    }

    fn empty_tensor(&mut self, shape: &[usize], elem_size: usize) -> (Handle, Vec<usize>) {
        let strides = contiguous_strides(shape);
        let size = shape.iter().product::<usize>() * elem_size;
        let handle = self.empty(size);
        (handle, strides)
    }
}

fn compiler(backend: wgpu::Backend) -> AutoCompiler {
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan => AutoCompiler::SpirV(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
