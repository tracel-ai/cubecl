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

// When uploading data smaller than this size, consider the data
// as a special 'small uniform' buffer which we can handle more efficiently.
const SMALL_UNIFORMS_BUFFER_SIZE: u64 = 8192;

#[derive(Debug)]
pub struct WgpuStream {
    memory_main: MemoryManagement<WgpuStorage>,

    uniforms_staging_pool: StagingBelt,
    memory_uniforms: MemoryManagement<WgpuStorage>,

    locked_copy_handles: Vec<SliceBinding>,

    tasks_encoder: wgpu::CommandEncoder,
    copy_uniforms_encoder: wgpu::CommandEncoder,

    compute_pass: Option<wgpu::ComputePass<'static>>,

    pub timestamps: KernelTimestamps,
    tasks_count: usize,
    tasks_max: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    poll: WgpuPoll,
    sync_buffer: Option<wgpu::Buffer>,
    submission_load: SubmissionLoad,
}

pub enum PipelineDispatch {
    Static(u32, u32, u32),
    Dynamic(WgpuResource),
}

impl WgpuStream {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timestamps: KernelTimestamps,
        tasks_max: usize,
    ) -> Self {
        let poll = WgpuPoll::new(device.clone());

        #[cfg(target_family = "wasm")]
        let sync_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
        #[cfg(not(target_family = "wasm"))]
        let sync_buffer = None;

        // Allocate storage & a memory management for buffers.
        let main_memory_management = MemoryManagement::from_configuration(
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

        // Allocate a separate storage & memory management for 'uniforms' (small bits of data
        // that need to be uploaded quickly). We allocate these with the BufferUsages::UNIFORM flag
        // to allow binding them as uniforms.
        let uniforms_memory_management = MemoryManagement::from_configuration(
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
                pool_options: vec![MemoryPoolOptions {
                    pool_type: memory_management::PoolType::ExclusivePages {
                        max_alloc_size: SMALL_UNIFORMS_BUFFER_SIZE,
                    },
                    dealloc_period: Some(5000),
                }],
            },
        );

        // Create a staging belt that can re-use staging buffers we use to upload
        // small uniform buffers. These are then uploaded to the buffers
        // we allocate with the uniforms_memory_management.
        let uniforms_staging_pool = StagingBelt::new(SMALL_UNIFORMS_BUFFER_SIZE);

        Self {
            memory_main: main_memory_management,
            memory_uniforms: uniforms_memory_management,
            locked_copy_handles: Vec::new(),
            compute_pass: None,
            timestamps,
            uniforms_staging_pool,
            tasks_encoder: {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Tasks Encoder"),
                })
            },
            copy_uniforms_encoder: {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Uniforms Copy Encoder"),
                })
            },
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

            self.tasks_encoder
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
        buffers: Vec<(wgpu::Buffer, u64, u64)>,
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
            self.tasks_encoder.copy_buffer_to_buffer(
                &buffer,
                offset,
                &staging_buffer,
                0,
                aligned_len,
            );
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
        buffer: wgpu::Buffer,
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

                    self.tasks_encoder
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
                let fut = self.read_buffer(resolved, 0, size);

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
        if let Some(res) = self
            .memory_main
            .get_resource(binding.clone(), offset_start, offset_end)
        {
            res
        } else if let Some(res) =
            self.memory_uniforms
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
        // For empty buffers we always use the main memory even if they are small, as
        // we don't need to upload any data to them.
        self.memory_main.reserve(size)
    }

    pub fn create(&mut self, data: &[u8]) -> SliceHandle {
        // We'd like to keep operations as one long ComputePass. To do so, this tries to submit
        // all copy operations & all execute operations in one batch. To do so, we do all copy operations
        // at the start of the encoder, and all execute operations afterwards. For this re-ordering to be valid,
        // a buffer we copy to MUST not have any outstanding compute work associated with it.
        // - For small uniform buffers, any small handles with compute work are kept
        //   in self.locked_copy_handles which means the allocation won't try to use them.
        // - For bigger buffers, we do restart the compute stream. These bigger allocations
        //   don't happen frequently so this cost isn't big.

        let num_bytes = data.len() as u64;

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is 32 bytes aligned (see WgpuStorage).
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let aligned_len = num_bytes.div_ceil(align) * align;

        // If the data is small enough, we assume we're creating some kind of uniform buffer,
        // that we can handle specially.
        let uniform_alloc = aligned_len < SMALL_UNIFORMS_BUFFER_SIZE;

        let allocator = if uniform_alloc {
            &mut self.memory_uniforms
        } else {
            &mut self.memory_main
        };

        let slice = allocator.reserve(aligned_len);
        let resource = allocator
            .get_resource(slice.clone().binding(), None, None)
            .unwrap();

        if let Some(size) = NonZeroU64::new(aligned_len) {
            // Small buffers are 'locked' if they were previously used for compute operations,
            // so we can safely do the data upload first in the copy_uniforms_encoder.
            if uniform_alloc {
                // Use the staging belt to allocate a staging buffer and write to it.
                // This efficiently re-uses the staging buffers.
                let mut staging = self.uniforms_staging_pool.write_buffer(
                    &mut self.copy_uniforms_encoder,
                    &resource.buffer,
                    resource.offset(),
                    size,
                    &self.device,
                );
                staging[0..data.len()].copy_from_slice(data);
            } else {
                // For bigger buffers, we create a new staging buffer and use
                // copy_buffer_to_buffer on the tasks_encoder. The compute pass
                // has to be closed to allow copy_buffer_to_buffer to be called.
                //
                // Note: It is possible to also use the staging belt here, but, that would
                // allocate big staging buffers in the staging belt, which then stick around
                // forever.
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

                self.tasks_encoder.copy_buffer_to_buffer(
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
        self.memory_main
            .memory_usage()
            .combine(self.memory_uniforms.memory_usage())
    }

    fn flush_if_needed(&mut self) {
        // Flush when there are too many tasks, or when too many handles are locked.
        // Locked handles should only accumulate in rare circumstances (where uniforms
        // are being created but no work is submitted).
        if self.tasks_count >= self.tasks_max
            || self.locked_copy_handles.len() >= self.tasks_max * 8
        {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        // End the current compute pass.
        self.compute_pass = None;

        // Mark all the pooled staging buffers as done.
        self.uniforms_staging_pool.finish();

        // Submit the pending actions to the queue. This will _first_ submit the
        // pending uniforms copy operations, then the main tasks.
        let copy_uniforms_encoder = std::mem::replace(&mut self.copy_uniforms_encoder, {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Uniforms Copy Encoder"),
                })
        });
        let tasks_encoder = std::mem::replace(&mut self.tasks_encoder, {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Tasks Encoder"),
                })
        });

        let index = self
            .queue
            .submit([copy_uniforms_encoder.finish(), tasks_encoder.finish()]);

        // Allow the staging buffers in the pool to be used again.
        self.uniforms_staging_pool.recall();

        // Now that the tasks are submitted we can unlock these handles again.
        self.locked_copy_handles.clear();

        self.submission_load
            .regulate(&self.device, self.tasks_count, index);

        // Cleanup allocations and deallocations.
        self.memory_main.cleanup();
        self.memory_main.storage().perform_deallocations();

        self.tasks_count = 0;
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
