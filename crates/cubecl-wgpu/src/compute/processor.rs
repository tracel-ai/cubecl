use async_channel::Sender;
use cubecl_common::stream::StreamId;
use cubecl_core::future::block_on;
use std::{
    collections::BTreeSet,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};
use web_time::Instant;

use crate::compiler::base::WgpuCompiler;

use super::{timestamps::KernelTimestamps, WgpuServer};
use cubecl_runtime::{storage::BindingResource, TimestampsError, TimestampsResult};
use wgpu::ComputePipeline;

#[derive(Debug)]
pub struct WgpuProcessor<C: WgpuCompiler> {
    pass: Option<wgpu::ComputePass<'static>>,
    encoder: wgpu::CommandEncoder,
    pub timestamps: KernelTimestamps,
    tasks_count: usize,
    tasks_max: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sync_buffer: Option<wgpu::Buffer>,
    compiler: PhantomData<C>,
    deffered: Vec<SyncTask>,
    reads: Vec<DeferredJob>,
    should_flush: Arc<AtomicBool>,
    submitmax: usize,
    index: Option<wgpu::SubmissionIndex>,
}

type Callback<M> = async_channel::Sender<M>;

#[derive(new)]
pub struct Message<C: WgpuCompiler> {
    id: StreamId,
    task: Task<C>,
}

pub enum Task<C: WgpuCompiler> {
    EnableTimestamp,
    DisableTimestamp,
    Async(AsyncTask<C>),
    Sync(SyncTask),
}

pub struct AsyncTask<C: WgpuCompiler> {
    pub pipeline: Arc<ComputePipeline>,
    pub resources: Vec<BindingResource<WgpuServer<C>>>,
    pub dispatch: PipelineDispatch<C>,
}

#[derive(Debug)]
pub enum SyncTask {
    Read {
        buffer: Arc<wgpu::Buffer>,
        offset: u64,
        size: u64,
        callback: Callback<Vec<u8>>,
    },
    Sync {
        callback: Callback<()>,
    },
    SyncElapsed {
        callback: Callback<TimestampsResult>,
    },
}

pub enum PipelineDispatch<C: WgpuCompiler> {
    Static(u32, u32, u32),
    Dynamic(BindingResource<WgpuServer<C>>),
}

#[derive(new, Debug)]
struct DeferredJob {
    staging_buffer: wgpu::Buffer,
    callback: Callback<Vec<u8>>,
    size: usize,
}

impl DeferredJob {
    pub async fn execute(self, index: &wgpu::SubmissionIndex, device: &wgpu::Device) {
        let (sender, receiver) = async_channel::bounded(1);

        self.staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |v| {
                sender
                    .try_send(v)
                    .expect("Unable to send buffer slice result to async channel.");
            });

        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(index.clone()));

        receiver
            .recv()
            .await
            .expect("Unable to receive buffer slice result.")
            .expect("Failed to map buffer");

        let data = {
            let data = self.staging_buffer.slice(..).get_mapped_range();
            bytemuck::cast_slice(&data[0..self.size]).to_vec()
        };
        self.staging_buffer.unmap();
        self.callback.send(data).await.unwrap();
    }
}

impl<C: WgpuCompiler> WgpuProcessor<C> {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        timestamps: KernelTimestamps,
        tasks_max: usize,
    ) -> Self {
        let encoder = create_encoder(&device);
        let should_flush = Arc::new(AtomicBool::new(false));

        // #[cfg(target_family = "wasm")]
        let sync_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
        // #[cfg(not(target_family = "wasm"))]
        // let sync_buffer = None;

        Self {
            pass: None,
            timestamps,
            device,
            encoder,
            queue,
            tasks_count: 0,
            tasks_max,
            sync_buffer,
            compiler: PhantomData,
            should_flush,
            deffered: Vec::new(),
            submitmax: 0,
            index: None,
            reads: Vec::new(),
        }
    }

    pub fn start(mut self) -> (Sender<Message<C>>, Arc<AtomicBool>) {
        let (sender, receiver) = async_channel::bounded::<Message<C>>(1);
        let should_flush = self.should_flush.clone();

        let fut = async move {
            let mut reveicing = BTreeSet::<StreamId>::new();

            while let Ok(msg) = receiver.recv().await {
                match &msg.task {
                    Task::Sync(_) => {
                        reveicing.remove(&msg.id);
                    }
                    _ => {
                        reveicing.insert(msg.id);
                    }
                };

                self.on_task(msg.task, false).await;
            }
        };

        std::thread::spawn(move || {
            block_on(fut);
        });

        (sender, should_flush)
    }

    async fn executed_deffered(&mut self) {
        log::info!("Execute deffered {}...", self.deffered.len());
        self.submit();

        if self.deffered.is_empty() {
            return;
        }

        let mut futures: Vec<Pin<Box<dyn Future<Output = ()> + Send + 'static>>> = Vec::new();
        let mut deffered = Vec::new();

        core::mem::swap(&mut deffered, &mut self.deffered);

        for task in deffered {
            match task {
                SyncTask::Read {
                    buffer,
                    offset,
                    size,
                    callback,
                } => {
                    let fut = self.read_buffer(&buffer, offset, size);

                    futures.push(Box::pin(async move {
                        log::info!("Waiting on read");
                        let data = fut.await;
                        log::info!("Read done.");
                        callback.send(data).await.unwrap();
                    }));
                }
                SyncTask::Sync { callback } => {
                    let fut = self.sync();

                    futures.push(Box::pin(async move {
                        log::info!("Waiting on sync");
                        fut.await;
                        log::info!("Sync done.");
                        callback.send(()).await.unwrap();
                    }));
                }
                SyncTask::SyncElapsed { callback } => {
                    let fut = self.sync_elapsed();

                    futures.push(Box::pin(async move {
                        log::info!("Waiting on sync elapsed.");
                        let result = fut.await;
                        log::info!("Sync elapse done.");
                        callback.send(result).await.unwrap();
                    }));
                }
            }
        }

        let device = self.device.clone();

        std::thread::spawn(move || {
            device.poll(wgpu::MaintainBase::Wait);

            block_on(async move {
                for fut in futures {
                    fut.await;
                }
            });
        });
    }

    fn sync_elapsed(&mut self) -> Pin<Box<dyn Future<Output = TimestampsResult> + Send + 'static>> {
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

    fn sync(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
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
                    let device = self.device.clone();
                    Box::pin(async move {
                        device.poll(wgpu::MaintainBase::Wait);
                    })
                }
                #[cfg(target_family = "wasm")]
                {
                    panic!("Only synching from a buffer is supported.");
                }
            }
        }
    }

    fn read_buffer(
        &mut self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> impl Future<Output = Vec<u8>> + 'static {
        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is 32 bytes aligned (see WgpuStorage).
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let aligned_len = size.div_ceil(align) * align;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: aligned_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::info!("Copy.");

        self.encoder
            .copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, aligned_len);

        log::info!("Here.");

        let (sender, receiver) = async_channel::bounded(1);
        staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |v| {
                log::info!("Inside map async.");
                sender
                    .try_send(v)
                    .expect("Unable to send buffer slice result to async channel.");
            });

        async move {
            log::info!("Here awaiting.");
            receiver
                .recv()
                .await
                .expect("Unable to receive buffer slice result.")
                .expect("Failed to map buffer");

            let result = {
                let data = staging_buffer.slice(..).get_mapped_range();
                bytemuck::cast_slice(&data[0..(size as usize)]).to_vec()
            };
            staging_buffer.unmap();
            result
        }
    }

    async fn on_task(&mut self, task: Task<C>, can_deffer: bool) {
        match task {
            Task::Async(task) => {
                self.register_compute(task.pipeline, task.resources, task.dispatch);
            }
            Task::Sync(sync_task) => {
                match sync_task {
                    SyncTask::Read { buffer, offset, size, callback } => {
                    },
                    SyncTask::Sync { callback } => todo!(),
                    SyncTask::SyncElapsed { callback } => todo!(),
                }
                self.deffered.push(sync_task);
                if !can_deffer {
                    self.executed_deffered().await;
                }
            }
            Task::EnableTimestamp => self.timestamps.enable(&self.device),
            Task::DisableTimestamp => self.timestamps.disable(),
        }

        if self.tasks_max <= self.tasks_count {
            log::info!("Task max");
            self.executed_deffered().await;
        }
    }

    fn register_compute(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: Vec<BindingResource<WgpuServer<C>>>,
        dispatch: PipelineDispatch<C>,
    ) {
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
    }

    fn submit(&mut self) {
        // End the current compute pass.
        self.pass = None;
        self.submitmax += 1;

        let new_encoder = create_encoder(&self.device);
        let encoder = std::mem::replace(&mut self.encoder, new_encoder);
        let index = self.queue.submit([encoder.finish()]);

        const MAX_NUM_BATCH: usize = 8;

        if self.index.is_none() {
            self.index = Some(index);
        } else if self.submitmax >= MAX_NUM_BATCH {
            if let Some(index) = self.index.take() {
                log::info!("Waiting");
                self.device
                    .poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));
            }
            self.index = Some(index);
            self.submitmax = 0;
        } else {
            self.submitmax += 1;
        }

        log::info!("Submitted {} tasks", self.tasks_count);
        self.tasks_count = 0;
        self.should_flush
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn create_encoder(device: &wgpu::Device) -> wgpu::CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("CubeCL Command Encoder"),
    })
}
