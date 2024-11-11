use std::{future::Future, marker::PhantomData, pin::Pin, sync::Arc, time::Duration};
use web_time::Instant;

use crate::compiler::base::WgpuCompiler;

use super::{poll::WgpuPoll, timestamps::KernelTimestamps, WgpuServer};
use cubecl_runtime::{storage::BindingResource, TimestampsError, TimestampsResult};
use wgpu::ComputePipeline;

#[derive(Debug)]
pub struct WgpuStream<C: WgpuCompiler> {
    pass: Option<wgpu::ComputePass<'static>>,
    encoder: wgpu::CommandEncoder,
    pub timestamps: KernelTimestamps,
    tasks_count: usize,
    tasks_max: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    poll: WgpuPoll,
    sync_buffer: Option<wgpu::Buffer>,
    compiler: PhantomData<C>,
}

pub enum PipelineDispatch<C: WgpuCompiler> {
    Static(u32, u32, u32),
    Dynamic(BindingResource<WgpuServer<C>>),
}

impl<C: WgpuCompiler> WgpuStream<C> {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        timestamps: KernelTimestamps,
        tasks_max: usize,
    ) -> Self {
        let poll = WgpuPoll::new(device.clone());
        let encoder = create_encoder(&device);

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

        Self {
            pass: None,
            timestamps,
            device,
            encoder,
            queue,
            tasks_count: 0,
            tasks_max,
            poll,
            sync_buffer,
            compiler: PhantomData,
        }
    }

    pub fn register(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: Vec<BindingResource<WgpuServer<C>>>,
        dispatch: PipelineDispatch<C>,
    ) -> bool {
        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        let pass = self.pass.get_or_insert_with(|| {
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
                resource: r.resource().as_wgpu_bind_resource(),
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
                    &binding_resource.resource().buffer,
                    binding_resource.resource().offset(),
                );
            }
        }

        if self.tasks_count >= self.tasks_max {
            self.flush();
            true
        } else {
            false
        }
    }

    pub fn read_buffer(
        &mut self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> impl Future<Output = Vec<u8>> + 'static {
        self.pass = None;
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
            .copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, aligned_len);

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
            core::mem::drop(poll);

            let result = {
                let data = staging_buffer.slice(..).get_mapped_range();
                bytemuck::cast_slice(&data[0..(size as usize)]).to_vec()
            };

            staging_buffer.unmap();
            result
        }
    }

    pub fn sync_elapsed(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = TimestampsResult> + Send + 'static>> {
        self.pass = None;

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
                let fut = self.read_buffer(&resolved, 0, size);

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
                let fut = self.read_buffer(&buf, 0, 32);
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

    pub fn flush(&mut self) {
        // End the current compute pass.
        self.pass = None;

        let new_encoder = create_encoder(&self.device);
        let encoder = std::mem::replace(&mut self.encoder, new_encoder);

        self.queue.submit([encoder.finish()]);
        self.tasks_count = 0;
    }
}

fn create_encoder(device: &wgpu::Device) -> wgpu::CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("CubeCL Command Encoder"),
    })
}
