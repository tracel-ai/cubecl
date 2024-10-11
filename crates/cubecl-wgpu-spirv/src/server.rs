use std::{fs, num::NonZero};

use super::WgpuStorage;
use alloc::{borrow::Cow, sync::Arc};
use cubecl_common::{reader::Reader, sync_type::SyncType};
use cubecl_core::{compute::DebugInformation, prelude::*, server::Handle, Feature, KernelId};
use cubecl_runtime::{
    debug::DebugLogger,
    memory_management::{MemoryHandle, MemoryLock, MemoryManagement, MemoryUsage},
    server::{self, ComputeServer},
    storage::{BindingResource, ComputeStorage},
    ExecutionMode,
};
use cubecl_spirv::SpirvCompiler;
use hashbrown::HashMap;
use wgpu::{
    hal::{self, vulkan},
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    CommandEncoder, ComputePass, ComputePipeline, Device, PipelineLayout, PipelineLayoutDescriptor,
    ShaderModuleDescriptorSpirV, ShaderStages,
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
    storage_locked: MemoryLock,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    tasks_max: usize,
    logger: DebugLogger,
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

        println!("Compiling {}", kernel.name());

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

        let file_name = sanitize_filename::sanitize(
            kernel
                .name()
                .split("<")
                .next()
                .unwrap()
                .split("::")
                .last()
                .unwrap(),
        );
        fs::write(
            format!("out/{}.opt.txt", file_name),
            format!("{}", repr.optimizer),
        )
        .unwrap();
        fs::write(
            format!("out/{}.spv", file_name),
            repr.assemble()
                .into_iter()
                .flat_map(|it| it.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
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
}

impl ComputeServer for WgpuSpirvServer {
    type Kernel = Box<dyn CubeTask<SpirvCompiler>>;
    type Storage = WgpuStorage;
    type Feature = Feature;

    fn read(&mut self, binding: server::Binding) -> Reader {
        let br = self.get_resource(binding);
        let resource = br.resource();

        let size = resource.size();
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.clear_compute_pass();

        self.encoder.copy_buffer_to_buffer(
            &resource.buffer,
            resource.offset(),
            &read_buffer,
            0,
            size,
        );

        // Flush all commands to the queue, so GPU gets started on copying to the staging buffer.
        self.sync(SyncType::Flush);

        let (sender, receiver) = async_channel::bounded(1);
        let slice = read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |v| {
            sender
                .try_send(v)
                .expect("Unable to send buffer slice result to async channel.");
        });

        let device = self.device.clone();

        Box::pin(async move {
            // Now wait for the GPU to finish.
            device.poll(wgpu::Maintain::Wait);

            let slice = read_buffer.slice(..);

            receiver
                .recv()
                .await
                .expect("Unable to receive buffer slice result.")
                .expect("Failed to map buffer");

            let data = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            read_buffer.unmap();

            result
        })
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

        let start = if profile_level.is_some() {
            self.sync(SyncType::Wait);
            Some(std::time::SystemTime::now())
        } else {
            None
        };

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
            self.encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                })
                .forget_lifetime()
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
            self.sync(SyncType::Wait);

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
            self.logger
                .register_profiled(info, start.unwrap().elapsed().unwrap());
        } else if self.tasks_count >= self.tasks_max {
            self.sync(SyncType::Flush);
        }
    }

    fn sync(&mut self, sync_type: SyncType) {
        // End the current compute pass.
        self.clear_compute_pass();
        let new_encoder = create_encoder(&self.device);
        let encoder = std::mem::replace(&mut self.encoder, new_encoder);
        self.queue.submit([encoder.finish()]);

        self.tasks_count = 0;
        self.storage_locked.clear_locked();

        if sync_type == SyncType::Wait {
            self.device.poll(wgpu::Maintain::Wait);
        }

        // Cleanup allocations and deallocations.
        self.memory_management.storage().perform_deallocations();
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }
}

fn is_robust(device: &Device) -> bool {
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
