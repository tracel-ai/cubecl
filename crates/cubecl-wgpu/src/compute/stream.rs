use cubecl_core::MemoryConfiguration;
use std::{future::Future, num::NonZeroU64, pin::Pin, sync::Arc, time::Duration};
use web_time::Instant;

use super::{poll::WgpuPoll, timestamps::KernelTimestamps, WgpuResource, WgpuStorage};
use cubecl_runtime::{
    memory_management::{
        self, MemoryDeviceProperties, MemoryHandle, MemoryManagement, MemoryPoolOptions,
        SliceBinding, SliceHandle,
    },
    TimestampsError, TimestampsResult,
};
use wgpu::{util::StagingBelt, BufferDescriptor, BufferUsages, ComputePipeline};

const MAX_UNIFORM_BUFFER_SIZE: u64 = 8192;

#[derive(Debug)]
pub struct WgpuStream {
    memory_management: MemoryManagement<WgpuStorage>,
    memory_management_uniforms: MemoryManagement<WgpuStorage>,
    locked_copy_handles: Vec<SliceBinding>,

    encoder: wgpu::CommandEncoder,
    copy_encoder: wgpu::CommandEncoder,

    staging_belt: StagingBelt,

    compute_pass: Option<wgpu::ComputePass<'static>>,

    pub timestamps: KernelTimestamps,
    tasks_count: usize,
    tasks_max: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    poll: WgpuPoll,
    sync_buffer: Option<Arc<wgpu::Buffer>>,
    submission_load: SubmissionLoad,
}

pub enum PipelineDispatch {
    Static(u32, u32, u32),
    Dynamic(WgpuResource),
}

impl WgpuStream {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timestamps: KernelTimestamps,
        tasks_max: usize,
    ) -> Self {
        let poll = WgpuPoll::new(device.clone());

        #[cfg(target_family = "wasm")]
        let sync_buffer = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        })));
        #[cfg(not(target_family = "wasm"))]
        let sync_buffer = None;

        let memory_management_uniforms = MemoryManagement::from_configuration(
            WgpuStorage::new(
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT
                    | BufferUsages::UNIFORM,
            ),
            &memory_properties,
            MemoryConfiguration::Custom {
                neighbour_candidates: 1,
                pool_options: vec![MemoryPoolOptions {
                    pool_type: memory_management::PoolType::ExclusivePages,
                    page_size: 8192,
                    chunk_num_prealloc: 32,
                    dealloc_period: Some(1000),
                }],
            },
        );

        let memory_management = MemoryManagement::from_configuration(
            WgpuStorage::new(
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT,
            ),
            &memory_properties,
            memory_config,
        );

        Self {
            memory_management,
            memory_management_uniforms,
            locked_copy_handles: Vec::new(),
            compute_pass: None,
            timestamps,
            staging_belt: StagingBelt::new(MAX_UNIFORM_BUFFER_SIZE),
            encoder: create_encoder(&device),
            copy_encoder: create_encoder(&device),
            device,
            queue,
            tasks_count: 0,
            tasks_max,
            poll,
            sync_buffer,
            submission_load: SubmissionLoad::default(),
        }
    }

    pub fn register(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: Vec<WgpuResource>,
        dispatch: PipelineDispatch,
    ) {
        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        let pass = self.compute_pass.get_or_insert_with(|| {
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

        let entries = &resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.as_wgpu_bind_resource(),
            })
            .collect::<Vec<_>>();

        let group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        match dispatch {
            PipelineDispatch::Static(x, y, z) => {
                pass.dispatch_workgroups(x, y, z);
            }
            PipelineDispatch::Dynamic(binding_resource) => {
                pass.dispatch_workgroups_indirect(
                    &binding_resource.buffer,
                    binding_resource.offset(),
                );
            }
        }

        self.flush_if_needed();
    }

    pub fn read_buffers(
        &mut self,
        buffers: Vec<(Arc<wgpu::Buffer>, u64, u64)>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + 'static {
        self.compute_pass = None;
        let mut staging_buffers = Vec::with_capacity(buffers.len());
        let mut callbacks = Vec::with_capacity(buffers.len());

        for (buffer, offset, size) in buffers {
            // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
            // memory is 32 bytes aligned (see WgpuStorage).
            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let aligned_len = size.div_ceil(align) * align;

            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: aligned_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.encoder
                .copy_buffer_to_buffer(&buffer, offset, &staging_buffer, 0, aligned_len);
            staging_buffers.push((staging_buffer, size));
        }

        // Flush all commands to the queue, so GPU gets started on copying to the staging buffer.
        self.flush();

        for (staging_buffer, _size) in staging_buffers.iter() {
            let (sender, receiver) = async_channel::bounded(1);
            staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |v| {
                    // This might fail if the channel is closed (eg. the future is dropped).
                    // This is fine, just means results aren't needed anymore.
                    let _ = sender.try_send(v);
                });

            callbacks.push(receiver);
        }

        let poll = self.poll.start_polling();

        async move {
            for callback in callbacks {
                callback
                    .recv()
                    .await
                    .expect("Unable to receive buffer slice result.")
                    .expect("Failed to map buffer");
            }

            // Can stop polling now.
            core::mem::drop(poll);

            let result = {
                staging_buffers
                    .iter()
                    .map(|(staging_buffer, size)| {
                        let data = staging_buffer.slice(..).get_mapped_range();
                        bytemuck::cast_slice(&data[0..(*size as usize)]).to_vec()
                    })
                    .collect()
            };

            for (staging_buffer, _size) in staging_buffers {
                staging_buffer.unmap();
            }
            result
        }
    }

    pub fn read_buffer(
        &mut self,
        buffer: Arc<wgpu::Buffer>,
        offset: u64,
        size: u64,
    ) -> impl Future<Output = Vec<u8>> + 'static {
        let fut = self.read_buffers(vec![(buffer, offset, size)]);
        async move { fut.await.remove(0) }
    }

    pub fn sync_elapsed(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = TimestampsResult> + Send + 'static>> {
        self.compute_pass = None;

        enum TimestampMethod {
            Buffer(wgpu::Buffer, u64),
            StartTime(Instant),
        }

        let method = match &mut self.timestamps {
            KernelTimestamps::Native { query_set, init } => {
                if !*init {
                    let fut = self.sync();

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
                let fut = self.sync();

                return Box::pin(async move {
                    fut.await;
                    Err(TimestampsError::Disabled)
                });
            }
        };

        match method {
            TimestampMethod::Buffer(resolved, size) => {
                let period = self.queue.get_timestamp_period() as f64 * 1e-9;
                let fut = self.read_buffer(Arc::new(resolved), 0, size);

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
                let fut = self.sync();

                Box::pin(async move {
                    fut.await;
                    Ok(start_time.elapsed())
                })
            }
        }
    }

    pub fn sync(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        self.flush();

        let mut buffer = None;
        core::mem::swap(&mut buffer, &mut self.sync_buffer);

        match buffer.as_mut() {
            Some(buf) => {
                // TODO: This should work queue.on_submitted_work_done() but that
                // is not yet implemented on wgpu https://github.com/gfx-rs/wgpu/issues/6395
                //
                // For now, instead do a dummy readback. This *seems* to wait for the entire
                // queue to be done.
                let fut = self.read_buffer(buf.clone(), 0, 32);
                core::mem::swap(&mut buffer, &mut self.sync_buffer);

                Box::pin(async move {
                    fut.await;
                })
            }
            None => {
                #[cfg(not(target_family = "wasm"))]
                {
                    self.device.poll(wgpu::MaintainBase::Wait);
                    Box::pin(async move {})
                }
                #[cfg(target_family = "wasm")]
                {
                    panic!("Only synching from a buffer is supported.");
                }
            }
        }
    }

    pub fn get_resource(
        &mut self,
        binding: SliceBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> WgpuResource {
        if let Some(res) =
            self.memory_management
                .get_resource(binding.clone(), offset_start, offset_end)
        {
            res
        } else if let Some(res) =
            self.memory_management_uniforms
                .get_resource(binding.clone(), offset_start, offset_end)
        {
            // This resource now might be used for computations, so we _cannot_ use it anymore for create() calls.
            // This keeps the binding alive which means create() won't try to use it.
            self.locked_copy_handles.push(binding);
            res
        } else {
            panic!("Failed to find resource");
        }
    }

    pub fn empty(&mut self, size: u64) -> SliceHandle {
        self.memory_management.reserve(size)
    }

    pub fn create(&mut self, data: &[u8]) -> SliceHandle {
        let num_bytes = data.len() as u64;

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is 32 bytes aligned (see WgpuStorage).
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let aligned_len = num_bytes.div_ceil(align) * align;

        // We'd like to keep operations as one long ComputePass. To do so, this tries to submit
        // all copy operations & all execute operations in one batch. To do so, we do all copy operations
        // at the start of the encoder, and all execute operations afterwards. For this re-ordering to be valid,
        // a buffer we copy to MUST not have any outstanding compute work associated with it.
        // - For small uniform buffers, any small handles with compute work are kept
        //   in self.locked_copy_handles which means the allocation won't try to use them.
        // - For bigger buffers, we do break the compute stream.

        let allocator = if aligned_len < MAX_UNIFORM_BUFFER_SIZE {
            &mut self.memory_management_uniforms
        } else {
            &mut self.memory_management
        };

        let slice = allocator.reserve(aligned_len);
        let resource = allocator
            .get_resource(slice.clone().binding(), None, None)
            .unwrap();

        if let Some(size) = NonZeroU64::new(aligned_len) {
            // Small buffers are 'locked' if they are used for compute, so we know we can
            // safely write these with the initial copy_encoder.
            if aligned_len < MAX_UNIFORM_BUFFER_SIZE {
                let mut staging = self.staging_belt.write_buffer(
                    &mut self.copy_encoder,
                    &resource.buffer,
                    resource.offset(),
                    size,
                    &self.device,
                );
                staging[0..data.len()].copy_from_slice(data);
            } else {
                // For bigger buffers, end the compute pass as we need to submit a copy operation
                // inline with the current encoder.
                //
                // Note: It is possible to use the staging belt here, but, that would
                // allocate staging buffers in the staging belt which stick around forever.
                self.compute_pass = None;

                let staging = self.device.create_buffer(&BufferDescriptor {
                    label: Some("(wgpu internal) StagingBelt staging buffer"),
                    size: aligned_len,
                    usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                staging
                    .slice(0..data.len() as u64)
                    .get_mapped_range_mut()
                    .copy_from_slice(data);
                staging.unmap();

                self.encoder.copy_buffer_to_buffer(
                    &staging,
                    0,
                    &resource.buffer,
                    resource.offset(),
                    aligned_len,
                );
            };
        }

        self.flush_if_needed();

        slice
    }

    pub fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_management
            .memory_usage()
            .combine(self.memory_management_uniforms.memory_usage())
    }

    fn flush_if_needed(&mut self) {
        // For now we only consider the number of handles locked, but we may consider the amount in
        // bytes at some point.
        if self.tasks_count >= self.tasks_max || self.locked_copy_handles.len() >= 32 {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        // End the current compute pass.
        self.compute_pass = None;

        self.staging_belt.finish();
        let copy_encoder = std::mem::replace(&mut self.copy_encoder, create_encoder(&self.device));
        let encoder = std::mem::replace(&mut self.encoder, create_encoder(&self.device));

        let index = self.queue.submit([copy_encoder.finish(), encoder.finish()]);

        self.staging_belt.recall();

        self.submission_load
            .regulate(&self.device, self.tasks_count, index);

        self.tasks_count = 0;
        self.locked_copy_handles.clear();

        // Cleanup allocations and deallocations.
        self.memory_management.cleanup();
        self.memory_management.storage().perform_deallocations();

        self.memory_management_uniforms.cleanup();
        self.memory_management_uniforms
            .storage()
            .perform_deallocations();
    }
}

#[cfg(not(target_family = "wasm"))]
mod __submission_load {

    #[derive(Default, Debug)]
    pub enum SubmissionLoad {
        Init {
            last_index: wgpu::SubmissionIndex,
            tasks_count_submitted: usize,
        },
        #[default]
        Empty,
    }

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            device: &wgpu::Device,
            tasks_count: usize,
            mut index: wgpu::SubmissionIndex,
        ) {
            match self {
                SubmissionLoad::Init {
                    last_index,
                    tasks_count_submitted,
                } => {
                    *tasks_count_submitted += tasks_count;

                    // Enough to keep the GPU busy.
                    //
                    // - Too much can hang the GPU and create slowdown.
                    // - Too little and GPU utilization is really bad.
                    //
                    // TODO: Could be smarter and dynamic based on stats.
                    const MAX_TOTAL_TASKS: usize = 512;

                    if *tasks_count_submitted >= MAX_TOTAL_TASKS {
                        core::mem::swap(last_index, &mut index);
                        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));

                        *tasks_count_submitted = 0;
                    }
                }
                SubmissionLoad::Empty => {
                    *self = Self::Init {
                        last_index: index,
                        tasks_count_submitted: 0,
                    }
                }
            }
        }
    }
}

#[cfg(target_family = "wasm")]
mod __submission_load_wasm {
    #[derive(Default, Debug)]
    pub struct SubmissionLoad;

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            _device: &wgpu::Device,
            _tasks_count: usize,
            _index: wgpu::SubmissionIndex,
        ) {
            // Nothing to do.
        }
    }
}

#[cfg(not(target_family = "wasm"))]
use __submission_load::*;
#[cfg(target_family = "wasm")]
use __submission_load_wasm::*;

fn create_encoder(device: &wgpu::Device) -> wgpu::CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("CubeCL Command Encoder"),
    })
}
