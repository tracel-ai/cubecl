use std::{future::Future, marker::PhantomData, num::NonZero, time::Duration};

use super::{
    stream::{PipelineDispatch, WgpuStream},
    WgpuStorage,
};
use crate::compiler::base::WgpuCompiler;
use crate::timestamps::KernelTimestamps;
use alloc::sync::Arc;
use cubecl_common::future;
use cubecl_core::{compute::DebugInformation, prelude::*, server::Handle, Feature, KernelId};
use cubecl_runtime::{
    debug::{DebugLogger, ProfileLevel},
    memory_management::{MemoryHandle, MemoryLock, MemoryManagement},
    server::{self, ComputeServer},
    storage::{BindingResource, ComputeStorage},
    ExecutionMode, TimestampsError, TimestampsResult,
};
use hashbrown::HashMap;
use wgpu::ComputePipeline;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<C: WgpuCompiler> {
    memory_management: MemoryManagement<WgpuStorage>,
    pub(crate) device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    logger: DebugLogger,
    storage_locked: MemoryLock,
    duration_profiled: Option<Duration>,
    stream: WgpuStream,
    pub compilation_options: C::CompilationOptions,
    _compiler: PhantomData<C>,
}

impl<C: WgpuCompiler> WgpuServer<C> {
    /// Create a new server.
    pub fn new(
        memory_management: MemoryManagement<WgpuStorage>,
        compilation_options: C::CompilationOptions,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        tasks_max: usize,
    ) -> Self {
        let logger = DebugLogger::default();
        let mut timestamps = KernelTimestamps::Disabled;

        if logger.profile_level().is_some() {
            timestamps.enable(&device);
        }

        let stream = WgpuStream::new(device.clone(), queue.clone(), timestamps, tasks_max);

        // Low estimate, but it makes sure there is no memory error from allocating too much
        // at the same time.
        let estimated_buffers_per_task = 4;
        let storage_locked = MemoryLock::new(tasks_max * estimated_buffers_per_task);

        Self {
            memory_management,
            compilation_options,
            device: device.clone(),
            queue: queue.clone(),
            storage_locked,
            pipelines: HashMap::new(),
            logger,
            duration_profiled: None,
            stream,
            _compiler: PhantomData,
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

        let mut compile = <C as WgpuCompiler>::compile(self, kernel, mode);

        if self.logger.is_activated() {
            compile.debug_info = Some(DebugInformation::new("wgsl", kernel_id.clone()));
        }

        let compile = self.logger.debug(compile);
        let pipeline = C::create_pipeline(self, compile);

        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        pipeline
    }

    fn on_flushed(&mut self) {
        self.storage_locked.clear_locked();

        // Cleanup allocations and deallocations.
        self.memory_management.cleanup();
        self.memory_management.storage().perform_deallocations();
    }
}

impl<C: WgpuCompiler> ComputeServer for WgpuServer<C> {
    type Kernel = Box<dyn CubeTask<C>>;
    type Storage = WgpuStorage;
    type Feature = Feature;

    fn read(
        &mut self,
        bindings: Vec<server::Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + Send + 'static {
        let resources = bindings
            .into_iter()
            .map(|binding| {
                let rb = self.get_resource(binding);
                let resource = rb.resource();

                (resource.buffer.clone(), resource.offset(), resource.size())
            })
            .collect();

        // Clear compute pass.
        let fut = self.stream.read_buffers(resources);
        self.on_flushed();

        fut
    }

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<Self> {
        let handle = self.memory_management.get(binding.memory.clone());

        // Keep track of any buffer that might be used in the wgpu queue, as we cannot copy into them
        // after they have any outstanding compute work. Calling get_resource repeatedly
        // will add duplicates to this, but that is ok.
        self.storage_locked.add_locked(handle.id);

        let handle = match binding.offset_start {
            Some(offset) => handle.offset_start(offset),
            None => handle,
        };
        let handle = match binding.offset_end {
            Some(offset) => handle.offset_end(offset),
            None => handle,
        };
        let resource = self.memory_management.storage().get(&handle);
        BindingResource::new(binding, resource)
    }

    /// When we create a new handle from existing data, we use custom allocations so that we don't
    /// have to execute the current pending tasks.
    ///
    /// This is important, otherwise the compute passes are going to be too small and we won't be able to
    /// fully utilize the GPU.
    fn create(&mut self, data: &[u8]) -> server::Handle {
        let num_bytes = data.len() as u64;

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is 32 bytes aligned (see WgpuStorage).
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let aligned_len = num_bytes.div_ceil(align) * align;

        // Reserve memory on some storage we haven't yet used this command queue for compute
        // or copying.
        let memory = self
            .memory_management
            .reserve(aligned_len, Some(&self.storage_locked));

        if let Some(len) = NonZero::new(aligned_len) {
            let resource_handle = self.memory_management.get(memory.clone().binding());

            // Dont re-use this handle for writing until the queue is flushed. All writes
            // happen at the start of the submission.
            self.storage_locked.add_locked(resource_handle.id);

            let resource = self.memory_management.storage().get(&resource_handle);

            // Write to the staging buffer. Next queue submission this will copy the data to the GPU.
            self.queue
                .write_buffer_with(&resource.buffer, resource.offset(), len)
                .expect("Failed to write to staging buffer.")[0..data.len()]
                .copy_from_slice(data);

            // If too many handles are locked, we flush.
            if self.storage_locked.should_flush() {
                self.flush();
            }
        }

        Handle::new(memory, None, None, aligned_len)
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        server::Handle::new(
            self.memory_management.reserve(size as u64, None),
            None,
            None,
            size as u64,
        )
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Vec<server::Binding>,
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

        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources: Vec<_> = bindings
            .iter()
            .map(|binding| self.get_resource(binding.clone()).into_resource())
            .collect();

        // First resolve the dispatch buffer if needed. The weird ordering is because the lifetime of this
        // needs to be longer than the compute pass, so we can't do this just before dispatching.
        let dispatch = match count.clone() {
            CubeCount::Dynamic(binding) => {
                PipelineDispatch::Dynamic(self.get_resource(binding).into_resource())
            }
            CubeCount::Static(x, y, z) => PipelineDispatch::Static(x, y, z),
        };

        if self
            .stream
            .register(pipeline, resources, dispatch, &self.storage_locked)
        {
            self.on_flushed();
        }

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
        self.on_flushed();
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self) -> impl Future<Output = ()> + 'static {
        self.logger.profile_summary();
        let fut = self.stream.sync();
        self.on_flushed();

        fut
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync_elapsed(&mut self) -> impl Future<Output = TimestampsResult> + 'static {
        self.logger.profile_summary();

        let fut = self.stream.sync_elapsed();
        self.on_flushed();

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
        self.memory_management.memory_usage()
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
}
