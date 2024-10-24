use std::{future::Future, marker::PhantomData, num::NonZero, pin::Pin, time::Duration};

use super::poll::WgpuPoll;
use super::WgpuStorage;
use crate::compiler::base::WgpuCompiler;
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
use web_time::Instant;
use wgpu::{CommandEncoder, ComputePass, ComputePipeline, QuerySet, QuerySetDescriptor, QueryType};

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<C: WgpuCompiler> {
    memory_management: MemoryManagement<WgpuStorage>,
    pub(crate) device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    encoder: CommandEncoder,
    current_pass: Option<ComputePass<'static>>,
    tasks_count: usize,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    tasks_max: usize,
    logger: DebugLogger,
    poll: WgpuPoll,
    storage_locked: MemoryLock,
    duration_profiled: Option<Duration>,
    timestamps: KernelTimestamps,
    _compiler: PhantomData<C>,
}

#[derive(Debug)]
enum KernelTimestamps {
    Native { query_set: QuerySet, init: bool },
    Inferred { start_time: Instant },
    Disabled,
}

impl KernelTimestamps {
    fn enable(&mut self, device: &wgpu::Device) {
        if !matches!(self, Self::Disabled) {
            return;
        }

        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let query_set = device.create_query_set(&QuerySetDescriptor {
                label: Some("CubeCL profile queries"),
                ty: QueryType::Timestamp,
                count: 2,
            });

            *self = Self::Native {
                query_set,
                init: false,
            };
        } else {
            *self = Self::Inferred {
                start_time: Instant::now(),
            };
        };
    }

    fn disable(&mut self) {
        *self = Self::Disabled;
    }
}

fn create_encoder(device: &wgpu::Device) -> CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("CubeCL Command Encoder"),
    })
}

impl<C: WgpuCompiler> WgpuServer<C> {
    /// Create a new server.
    pub fn new(
        memory_management: MemoryManagement<WgpuStorage>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        tasks_max: usize,
    ) -> Self {
        let logger = DebugLogger::default();
        let mut timestamps = KernelTimestamps::Disabled;

        if logger.profile_level().is_some() {
            timestamps.enable(&device);
        }

        Self {
            memory_management,
            device: device.clone(),
            queue: queue.clone(),
            encoder: create_encoder(&device),
            current_pass: None,
            tasks_count: 0,
            storage_locked: MemoryLock::default(),
            pipelines: HashMap::new(),
            tasks_max,
            logger,
            poll: WgpuPoll::new(device.clone()),
            duration_profiled: None,
            timestamps,
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
        let pipeline = C::create_pipeline(self, compile, mode);

        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        pipeline
    }

    fn clear_compute_pass(&mut self) {
        self.current_pass = None;
    }

    fn read_wgpu_buffer(
        &mut self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> impl Future<Output = Vec<u8>> + 'static {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.encoder
            .copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, size);

        // Flush all commands to the queue, so GPU gets started on copying to the staging buffer.
        self.flush();

        let (sender, receiver) = async_channel::bounded(1);
        staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |v| {
                sender
                    .try_send(v)
                    .expect("Unable to send buffer slice result to async channel.");
            });
        let poll = self.poll.start_polling();
        async move {
            receiver
                .recv()
                .await
                .expect("Unable to receive buffer slice result.")
                .expect("Failed to map buffer");
            // Can stop polling now.
            drop(poll);

            let result = {
                let data = staging_buffer.slice(..).get_mapped_range();
                bytemuck::cast_slice(&data).to_vec()
            };
            staging_buffer.unmap();
            result
        }
    }

    fn sync_queue(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        self.flush();

        #[cfg(target_family = "wasm")]
        {
            // TODO: This should work queue.on_submitted_work_done() but that
            // is not yet implemented on wgpu https://github.com/gfx-rs/wgpu/issues/6395
            //
            // For now, instead do a dummy readback. This *seems* to wait for the entire
            // queue to be done.

            let dummy = self.empty(32);
            let fut = self.read(dummy.binding());

            Box::pin(async move {
                fut.await;
            })
        }

        #[cfg(not(target_family = "wasm"))]
        {
            self.device.poll(wgpu::MaintainBase::Wait);
            Box::pin(async move {})
        }
    }

    fn sync_queue_elapsed(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = TimestampsResult> + Send + 'static>> {
        self.clear_compute_pass();

        enum TimestampMethod {
            Buffer(wgpu::Buffer, u64),
            StartTime(Instant),
        }

        let method = match &mut self.timestamps {
            KernelTimestamps::Native { query_set, init } => {
                if !*init {
                    let fut = self.sync_queue();

                    return Box::pin(async move {
                        fut.await;
                        Err(TimestampsError::Unavailable)
                    });
                } else {
                    let size = 2 * size_of::<u64>() as u64;
                    let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size,
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                        mapped_at_creation: false,
                    });

                    self.encoder
                        .resolve_query_set(query_set, 0..2, &resolved, 0);
                    *init = false;
                    TimestampMethod::Buffer(resolved, size)
                }
            }
            KernelTimestamps::Inferred { start_time } => {
                let mut instant = Instant::now();
                core::mem::swap(&mut instant, start_time);
                TimestampMethod::StartTime(instant)
            }
            KernelTimestamps::Disabled => {
                let fut = self.sync_queue();

                return Box::pin(async move {
                    fut.await;
                    Err(TimestampsError::Disabled)
                });
            }
        };

        match method {
            TimestampMethod::Buffer(resolved, size) => {
                let period = self.queue.get_timestamp_period() as f64 * 1e-9;
                let fut = self.read_wgpu_buffer(&resolved, 0, size);

                Box::pin(async move {
                    let data = fut
                        .await
                        .chunks_exact(8)
                        .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let delta = u64::checked_sub(data[1], data[0]).unwrap_or(1);
                    let duration = Duration::from_secs_f64(delta as f64 * period);

                    Ok(duration)
                })
            }
            TimestampMethod::StartTime(start_time) => {
                let fut = self.sync_queue();

                Box::pin(async move {
                    fut.await;
                    Ok(start_time.elapsed())
                })
            }
        }
    }
}

impl<C: WgpuCompiler> ComputeServer for WgpuServer<C> {
    type Kernel = Box<dyn CubeTask<C>>;
    type Storage = WgpuStorage;
    type Feature = Feature;

    fn read(&mut self, binding: server::Binding) -> impl Future<Output = Vec<u8>> + Send + 'static {
        let rb = self.get_resource(binding);
        let resource = rb.resource();
        self.clear_compute_pass();
        self.read_wgpu_buffer(&resource.buffer, resource.offset(), resource.size())
    }

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<Self> {
        // Keep track of any buffer that might be used in the wgpu queue, as we cannot copy into them
        // after they have any outstanding compute work. Calling get_resource repeatedly
        // will add duplicates to this, but that is ok.
        let handle = self.memory_management.get(binding.memory.clone());
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
        }

        Handle::new(memory, None, None)
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        server::Handle::new(
            self.memory_management.reserve(size as u64, None),
            None,
            None,
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
            let fut = self.sync_queue_elapsed();
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
        let group_layout = pipeline.get_bind_group_layout(0);

        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources: Vec<_> = bindings
            .iter()
            .map(|binding| self.get_resource(binding.clone()))
            .collect();
        let entries = &resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.resource().as_wgpu_bind_resource(),
            })
            .collect::<Vec<_>>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries,
        });

        // First resolve the dispatch buffer if needed. The weird ordering is because the lifetime of this
        // needs to be longer than the compute pass, so we can't do this just before dispatching.
        let dispatch_br = match count.clone() {
            CubeCount::Dynamic(binding) => Some(self.get_resource(binding)),
            _ => None,
        };

        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        let pass = self.current_pass.get_or_insert_with(|| {
            // Write out timestamps. The first compute pass writes both a start and end timestamp.
            // the second timestamp writes out only an end stamp.
            let timestamps =
                if let KernelTimestamps::Native { query_set, init } = &mut self.timestamps {
                    let result = Some(wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: if !*init { Some(0) } else { None },
                        end_of_pass_write_index: Some(1),
                    });
                    *init = true;
                    result
                } else {
                    None
                };

            self.encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: timestamps,
                })
                .forget_lifetime()
        });

        self.tasks_count += 1;

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        match count {
            CubeCount::Static(x, y, z) => {
                pass.dispatch_workgroups(x, y, z);
            }
            CubeCount::Dynamic(_) => {
                let binding_resource = dispatch_br.as_ref().unwrap();
                pass.dispatch_workgroups_indirect(
                    &binding_resource.resource().buffer,
                    binding_resource.resource().offset(),
                );
            }
        }

        if self.tasks_count >= self.tasks_max {
            self.flush();
        }

        // If profiling, write out results.
        if let Some(level) = profile_level {
            let (name, kernel_id) = profile_info.unwrap();

            // Execute the task.
            if let Ok(duration) = future::block_on(self.sync_queue_elapsed()) {
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
        self.clear_compute_pass();
        let new_encoder = create_encoder(&self.device);
        let encoder = std::mem::replace(&mut self.encoder, new_encoder);
        self.queue.submit([encoder.finish()]);

        self.tasks_count = 0;
        self.storage_locked.clear_locked();

        // Cleanup allocations and deallocations.
        self.memory_management.cleanup();
        self.memory_management.storage().perform_deallocations();
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self) -> impl Future<Output = ()> + 'static {
        self.logger.profile_summary();

        self.sync_queue()
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync_elapsed(&mut self) -> impl Future<Output = TimestampsResult> + 'static {
        self.logger.profile_summary();

        let future = self.sync_queue_elapsed();
        let profiled = self.duration_profiled;
        self.duration_profiled = None;

        async move {
            match future.await {
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
        self.timestamps.enable(&self.device);
    }

    fn disable_timestamps(&mut self) {
        // Only disable timestamps if profiling isn't enabled.
        if self.logger.profile_level().is_none() {
            self.timestamps.disable();
        }
    }
}
