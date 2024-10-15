use std::time::Duration;
use std::{future::Future, num::NonZero};

use super::WgpuStorage;
use alloc::{borrow::Cow, sync::Arc};
use cubecl_common::future;
use cubecl_core::{compute::DebugInformation, prelude::*, server::Handle, Feature, KernelId};
use cubecl_runtime::{
    debug::DebugLogger,
    memory_management::{MemoryHandle, MemoryLock, MemoryManagement, MemoryUsage},
    server::{self, ComputeServer},
    storage::{BindingResource, ComputeStorage},
    ExecutionMode,
};
use cubecl_spirv::SpirvCompiler;
use cubecl_wgpu::WgpuPoll;
use hashbrown::HashMap;
use wgpu::{
    hal::{self, vulkan},
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    CommandEncoder, ComputePass, ComputePipeline, PipelineLayout, PipelineLayoutDescriptor,
    QuerySet, QuerySetDescriptor, QueryType, ShaderModuleDescriptorSpirV, ShaderStages,
};

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuSpirvServer {
    memory_management: MemoryManagement<WgpuStorage>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    encoder: CommandEncoder,
    current_pass: Option<ComputePass<'static>>,
    tasks_count: usize,
    poll: WgpuPoll,
    storage_locked: MemoryLock,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    tasks_max: usize,
    logger: DebugLogger,

    duration_profiled: Duration,
    query_set: QuerySet,
    query_started: bool,
}

fn create_encoder(device: &wgpu::Device) -> CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("CubeCL Command Encoder"),
    })
}

impl WgpuSpirvServer {
    /// Create a new server.
    pub fn new(
        memory_management: MemoryManagement<WgpuStorage>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        tasks_max: usize,
    ) -> Self {
        Self {
            memory_management,
            device: device.clone(),
            queue: queue.clone(),
            encoder: create_encoder(&device),
            current_pass: None,
            tasks_count: 0,
            pipelines: HashMap::new(),
            tasks_max,
            logger: DebugLogger::new(),
            storage_locked: MemoryLock::default(),
            poll: WgpuPoll::new(device.clone()),
            query_set: device.create_query_set(&QuerySetDescriptor {
                label: Some("CubeCL profile queries"),
                ty: QueryType::Timestamp,
                count: 2,
            }),
            query_started: false,
            duration_profiled: Duration::from_secs(0),
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

        // `wgpu` currently always enables `robustness2` on Vulkan if available, so default to
        // unchecked execution if robustness is enabled and let Vulkan handle it
        let mode = if is_robust(&self.device) {
            ExecutionMode::Unchecked
        } else {
            mode
        };
        let mut compile = kernel.compile(mode);
        if self.logger.is_activated() {
            compile.debug_info = Some(DebugInformation::new("spv", kernel_id.clone()));
        }

        let compile = self.logger.debug(compile);

        let repr = compile
            .repr
            .expect("Need compiled repr to assemble to spirv");

        let num_bindings = repr.num_bindings as u32;
        let bindings = (0..num_bindings)
            .map(|i| BindGroupLayoutEntry {
                binding: i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &bindings,
            });
        let layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

        let pipeline = self.compile_source(&repr.assemble(), &layout);

        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        pipeline
    }

    fn compile_source(&self, spirv: &[u32], layout: &PipelineLayout) -> Arc<ComputePipeline> {
        let module = unsafe {
            self.device
                .create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
                    label: None,
                    source: Cow::Borrowed(spirv),
                })
        };

        Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(layout),
                    module: &module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: None,
                }),
        )
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

    fn sync_queue(&mut self) -> impl Future<Output = Duration> + 'static {
        self.clear_compute_pass();

        let fut = if self.query_started {
            let size = 2 * size_of::<u64>() as u64;
            let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            });

            self.encoder
                .resolve_query_set(&self.query_set, 0..2, &resolved, 0);

            let period = self.queue.get_timestamp_period() as f64 * 1e-9;
            Some((self.read_wgpu_buffer(&resolved, 0, size), period))
        } else {
            None
        };

        self.query_started = false;
        let duration_profiled = self.duration_profiled;

        async move {
            if let Some((fut, period)) = fut {
                let data = fut
                    .await
                    .chunks_exact(8)
                    .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let delta = data[1] - data[0];
                Duration::from_secs_f64(delta as f64 * period) + duration_profiled
            } else {
                duration_profiled
            }
        }
    }
}

impl ComputeServer for WgpuSpirvServer {
    type Kernel = Box<dyn CubeTask<SpirvCompiler>>;
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
        let num_bytes = data.len();

        // Handle empty tensors (must bind at minimum 4 bytes)
        let reserve_size = core::cmp::max(num_bytes, 4);

        // Reserve memory on some storage we haven't yet used this command queue for compute
        // or copying.
        let memory = self
            .memory_management
            .reserve(reserve_size, Some(&self.storage_locked));

        if let Some(len) = NonZero::new(num_bytes as u64) {
            let resource_handle = self.memory_management.get(memory.clone().binding());

            // Dont re-use this handle for writing until the queue is flushed. All writes
            // happen at the start of the submission.
            self.storage_locked.add_locked(resource_handle.id);

            let resource = self.memory_management.storage().get(&resource_handle);

            // Write to the staging buffer. Next queue submission this will copy the data to the GPU.
            self.queue
                .write_buffer_with(&resource.buffer, resource.offset(), len)
                .expect("Failed to write to staging buffer.")
                .copy_from_slice(data);
        }

        Handle::new(memory, None, None)
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        server::Handle::new(self.memory_management.reserve(size, None), None, None)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Vec<server::Binding>,
        mode: ExecutionMode,
    ) {
        let profile_level = self.logger.profile_level();
        let profile_info = if profile_level.is_some() {
            Some((kernel.name(), kernel.id()))
        } else {
            None
        };

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

        if profile_level.is_some() {
            let duration = future::block_on(self.sync_queue());
            self.duration_profiled = duration;
        }

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

        self.tasks_count += 1;

        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        let pass = self.current_pass.get_or_insert_with(|| {
            let start_write = if !self.query_started { Some(0) } else { None };
            let end_write = Some(1);
            self.query_started = true;

            let pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &self.query_set,
                        beginning_of_pass_write_index: start_write,
                        end_of_pass_write_index: end_write,
                    }),
                })
                .forget_lifetime();
            pass
        });

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

        if let Some(level) = profile_level {
            let (name, kernel_id) = profile_info.unwrap();

            // Execute the task.
            let duration_previous = self.duration_profiled;
            self.duration_profiled = Duration::from_secs(0);
            let duration = future::block_on(self.sync_queue());
            self.duration_profiled = duration_previous + duration;

            let info = match level {
                cubecl_runtime::debug::ProfileLevel::Basic => {
                    if let Some(val) = name.split("<").next() {
                        val.split("::").last().unwrap_or(name).to_string()
                    } else {
                        name.to_string()
                    }
                }
                cubecl_runtime::debug::ProfileLevel::Full => {
                    format!("{name}: {kernel_id} CubeCount {count:?}")
                }
            };
            self.logger.register_profiled(info, duration);
        } else if self.tasks_count >= self.tasks_max {
            self.flush();
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
        self.memory_management.storage().perform_deallocations();
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self) -> impl Future<Output = Duration> + 'static {
        let future = self.sync_queue();
        self.duration_profiled = Duration::from_secs(0);
        future
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }
}

fn is_robust(device: &wgpu::Device) -> bool {
    fn is_robust(device: &vulkan::Device) -> bool {
        device
            .enabled_device_extensions()
            .contains(&c"VK_EXT_robustness2")
    }
    unsafe {
        device
            .as_hal::<hal::api::Vulkan, _, _>(|device| device.map(is_robust).unwrap_or(false))
            .unwrap_or(false)
    }
}
