use super::{mem_manager::WgpuMemManager, poll::WgpuPoll, timings::QueryProfiler};
use crate::{WgpuResource, controller::WgpuAllocController, schedule::ScheduleTask};
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    future::{self, DynFut},
    server::{IoError, ProfileError, ProfilingToken, ServerError, StreamErrorMode},
    zspace::Shape,
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger, memory_management::ManagedMemoryHandle,
    timestamp_profiler::TimestampProfiler,
};
use std::{future::Future, num::NonZero, pin::Pin, sync::Arc};
use wgpu::ComputePipeline;

#[derive(Debug)]
enum Timings {
    Device(QueryProfiler),
    System(TimestampProfiler),
}

#[derive(Debug)]
pub struct WgpuStream {
    pub mem_manage: WgpuMemManager,
    pub device: wgpu::Device,
    pub errors: Vec<ServerError>,
    compute_pass: Option<wgpu::ComputePass<'static>>,
    timings: Timings,
    tasks_count: usize,
    tasks_max: usize,
    queue: wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    poll: WgpuPoll,
    submission_load: SubmissionLoad,
    /// Number of consecutive `write_buffer` calls without a `queue.submit()`.
    /// Used to prevent wgpu staging buffer pool exhaustion during bulk writes
    /// (e.g. model loading with hundreds of tensors).
    pending_write_count: usize,
}

impl WgpuStream {
    /// Creates a new WGPU stream.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
    ) -> Self {
        let timings = if timing_method == TimingMethod::Device {
            Timings::Device(QueryProfiler::new(&queue, &device))
        } else {
            if cfg!(target_family = "wasm") {
                // On WASM, there's not much we can do here anymore. This should be very rare however,
                // all modern GPU's support timestamp queries.
                panic!(
                    "Cannot profile on web assembly without timestamp_query feature as it requires blocking."
                );
            }
            Timings::System(TimestampProfiler::default())
        };

        let poll = WgpuPoll::new(device.clone());

        #[allow(unused_mut)]
        let mut mem_manage =
            WgpuMemManager::new(device.clone(), memory_properties, memory_config, logger);

        Self {
            mem_manage,
            compute_pass: None,
            timings,
            errors: Vec::new(),
            encoder: {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Tasks Encoder"),
                })
            },
            device,
            queue,
            tasks_count: 0,
            tasks_max,
            poll,
            submission_load: SubmissionLoad::default(),
            pending_write_count: 0,
        }
    }

    /// Enqueue a [`ScheduleTask`] on this stream.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to execute.
    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write { data, buffer } => {
                // It is important to flush before writing, as the write operation is inserted
                // into the QUEUE not the encoder. We want to make sure all outstanding work
                // happens _before_ the write operation.
                let _ = self
                    .flush(StreamErrorMode {
                        ignore: true,
                        flush: false,
                    })
                    .ok();
                self.write_to_buffer(&buffer, &data);
            }
            ScheduleTask::Execute {
                pipeline,
                count,
                resources,
            } => {
                let resources = resources.into_resources(self);
                self.register_pipeline(pipeline, resources.iter(), &count);
            }
        }
    }

    /// Read multiple buffers lazily to [Bytes], potentially using pinned memory.
    ///
    /// # Arguments
    ///
    /// * `self` - The current stream.
    /// * `descriptors` - A vector of copy descriptors specifying the source data.
    ///
    /// # Returns
    ///
    /// A [Result] containing a vector of [Bytes] with the copied data, or an [`IoError`] if any copy fails.
    pub fn read_resources(
        &mut self,
        descriptors: Vec<(WgpuResource, Shape, usize)>,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        self.compute_pass = None;
        let mut staging_info = Vec::with_capacity(descriptors.len());
        let mut callbacks = Vec::with_capacity(descriptors.len());

        for (resource, shape, elem_size) in descriptors {
            let size = shape.iter().product::<usize>() * elem_size;

            // Zero-sized resources don't need a GPU copy.
            if resource.size == 0 {
                staging_info.push(None);
                continue;
            }

            // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
            // memory is 32 bytes aligned (see WgpuStorage).
            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let aligned_len = resource.size.div_ceil(align) * align;
            let (staging, binding) = self.mem_manage.reserve_staging(aligned_len).unwrap();

            self.tasks_count += 1;
            self.encoder.copy_buffer_to_buffer(
                &resource.buffer,
                resource.offset,
                &staging.buffer,
                0,
                aligned_len,
            );
            staging_info.push(Some((staging, binding, size)));
        }

        // Flush all commands to the queue, so GPU gets started on copying to the staging buffer.
        let _ = self
            .flush(StreamErrorMode {
                ignore: true,
                flush: false,
            })
            .ok();

        for entry in staging_info.iter() {
            if let Some((staging, _binding, _size)) = entry {
                let (sender, receiver) = async_channel::bounded(1);
                staging
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| {
                        // This might fail if the channel is closed (eg. the future is dropped).
                        // This is fine, just means results aren't needed anymore.
                        let _ = sender.try_send(v);
                    });

                callbacks.push(Some(receiver));
            } else {
                callbacks.push(None);
            }
        }

        let poll = self.poll.start_polling();

        Box::pin(async move {
            for receiver in callbacks.iter().flatten() {
                receiver
                    .recv()
                    .await
                    .expect("Unable to receive buffer slice result.")
                    .expect("Failed to map buffer");
            }

            // Can stop polling now.
            core::mem::drop(poll);

            let result = {
                staging_info
                    .into_iter()
                    .map(|entry| {
                        if let Some((staging, binding, size)) = entry {
                            let controller =
                                Box::new(WgpuAllocController::init(binding, staging.buffer));
                            // SAFETY: The binding has initialized memory for at least `size` bytes.
                            unsafe { Bytes::from_controller(controller, size) }
                        } else {
                            Bytes::from_bytes_vec(vec![])
                        }
                    })
                    .collect()
            };

            Ok(result)
        })
    }

    // Bit silly but needed to make the borrow checker happy.
    fn system_profiler(&mut self) -> &mut TimestampProfiler {
        let Timings::System(timing) = &mut self.timings else {
            panic!("Unexpected timings type");
        };
        timing
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        if matches!(self.timings, Timings::System(_)) {
            cubecl_common::future::block_on(self.sync())?;
        } else {
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
        }

        match &mut self.timings {
            Timings::System(_) => {
                let profiler = self.system_profiler();
                Ok(profiler.start())
            }
            Timings::Device(query) => {
                self.compute_pass = None;
                let token = query.start_profile();
                Ok(token)
            }
        }
    }

    pub fn profile_error(&mut self, error: ProfileError) {
        match &mut self.timings {
            Timings::Device(profiler) => {
                profiler.error(error);
            }
            Timings::System(profiler) => {
                profiler.error(error);
            }
        }
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        match &mut self.timings {
            Timings::System(..) => {
                // Nb: WASM _has_ to use device timing and will panic here if query timestamps are not supported.
                let result = future::block_on(self.sync());
                let profiler = self.system_profiler();

                if let Err(err) = result {
                    profiler.error(ProfileError::Server(Box::new(err)));
                }
                profiler.stop(token)
            }
            Timings::Device(..) => {
                let poll = self.poll.start_polling();
                self.compute_pass = None;

                // Submit commands needed for profiling.
                let buffer = {
                    let Timings::Device(timing) = &mut self.timings else {
                        return Err(ProfileError::Unknown {
                            reason: "Unexpected timings type".to_string(),
                            backtrace: BackTrace::capture(),
                        });
                    };
                    timing.stop_profile_setup(token, &self.device, &mut self.encoder)?
                };

                // This flushes the queue to execute the encoder write command to write the
                // timings.
                self.tasks_count += 1;
                let result = self.flush(StreamErrorMode {
                    ignore: false,
                    flush: true,
                });

                let Timings::Device(timing) = &mut self.timings else {
                    return Err(ProfileError::Unknown {
                        reason: "Unexpected timings type".to_string(),
                        backtrace: BackTrace::capture(),
                    });
                };

                match result {
                    Ok(_) => timing.stop_profile(buffer, poll),
                    Err(err) => {
                        // Just to clean the timing buffer.
                        let _ = timing.stop_profile(buffer, poll).ok();
                        Err(ProfileError::Server(Box::new(err)))
                    }
                }
            }
        }
    }

    pub fn sync(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ServerError>> + Send + 'static>> {
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Internal);

        let flush_error = self
            .flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })
            .err();

        let queue = self.queue.clone();
        let error_future = error_scope.pop();
        let poll = self.poll.start_polling();

        Box::pin(async move {
            let (sender, receiver) = async_channel::bounded::<()>(1);
            queue.on_submitted_work_done(move || {
                // Signal that we're done.
                let _ = sender.try_send(());
                core::mem::drop(poll);
            });
            let _ = receiver.recv().await;

            if let Some(error) = error_future.await {
                return Err(ServerError::Generic {
                    reason: format!("{error}"),
                    backtrace: BackTrace::capture(),
                });
            }

            match flush_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.mem_manage.reserve(size)
    }

    /// Registers a new error into the error sink.
    pub fn error(&mut self, error: ServerError) {
        self.errors.push(error);
    }

    pub(crate) fn create_uniform(&mut self, data: &[u8]) -> WgpuResource {
        let resource = self.mem_manage.reserve_uniform(data.len() as u64);
        self.write_to_buffer(&resource, data);
        resource
    }

    // Nb: this function submits a command to the _queue_ not to the encoder,
    // so you have to be really careful about the ordering of operations here.
    // Any buffer which has outstanding (not yet flushed) compute work should
    // NOT be copied to.
    fn write_to_buffer(&mut self, resource: &WgpuResource, data: &[u8]) {
        // Nothing to write for zero-sized resources.
        if resource.size == 0 {
            return;
        }

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is also aligned (see WgpuStorage). Per the WebGPU spec, this
        // just has to be a multiple of 4: https://www.w3.org/TR/webgpu/#dom-gpuqueue-writebuffer
        let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
        let size = resource.size.next_multiple_of(copy_align);

        if size == data.len() as u64 {
            // write_buffer is the recommended way to write this data, as:
            // - On WebGPU, from WASM, this can save a copy to the JS memory.
            // - On devices with unified memory, this could skip the staging buffer entirely.
            self.queue
                .write_buffer(&resource.buffer, resource.offset, data);
        } else {
            // For sizes not aligned we need to only write a part of the staging buffer, do this
            // with `write_buffer_with`.
            let mut buffer = self
                .queue
                .write_buffer_with(
                    &resource.buffer,
                    resource.offset,
                    NonZero::new(size).unwrap(),
                )
                .expect("Internal error: Failed to call `write_buffer_with`, this likely means no staging buffer could be allocated.");
            buffer.slice(0..data.len()).copy_from_slice(data);
        }

        self.pending_write_count += 1;

        // Prevent wgpu staging buffer pool exhaustion during bulk writes (e.g. model
        // loading with hundreds of tensors). queue.write_buffer() is async — wgpu
        // copies data into an internal staging buffer, then transfers to GPU on the
        // next queue.submit(). Without periodic submits, hundreds of writes accumulate
        // and staging buffers get recycled before the GPU copy completes, silently
        // corrupting early tensors.
        // See: https://github.com/tracel-ai/cubecl/issues/1120
        const MAX_PENDING_WRITES: usize = 64;

        if self.pending_write_count >= MAX_PENDING_WRITES {
            // Submit a fresh, empty command buffer to flush all pending write_buffer work.
            // wgpu flushes its internal staging-buffer copies on any queue.submit(),
            // so we don't need to touch the main compute encoder here.
            let write_flush_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL Write Flush Encoder"),
                    });
            let index = self.queue.submit([write_flush_encoder.finish()]);

            // Wait for the GPU to finish processing these writes before continuing.
            #[cfg(not(target_family = "wasm"))]
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            }) {
                log::warn!("wgpu: write flush poll failed ({e})");
            }

            self.pending_write_count = 0;
        }
    }

    fn flush_if_needed(&mut self) {
        // Flush when there are too many tasks, or when too many handles are locked.
        // Locked handles should only accumulate in rare circumstances (where uniforms
        // are being created but no work is submitted).
        if self.tasks_count >= self.tasks_max {
            let _ = self
                .flush(StreamErrorMode {
                    ignore: true,
                    flush: false,
                })
                .ok();
        }
    }

    pub fn flush(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        if self.tasks_count == 0 {
            return self.flush_errors(mode);
        }

        // End the current compute pass.
        self.compute_pass = None;

        // Submit the pending actions to the queue. This will _first_ submit the
        // pending uniforms copy operations, then the main tasks.
        let tasks_encoder = {
            std::mem::replace(&mut self.encoder, {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL Tasks Encoder"),
                    })
            })
        };

        // This will _first_ fire off all pending write_buffer work.
        let index = self.queue.submit([tasks_encoder.finish()]);

        self.submission_load
            .regulate(&self.device, self.tasks_count, index);

        // Cleanup allocations and deallocations.
        self.mem_manage.memory_cleanup(false);
        self.mem_manage.release_uniforms();

        self.tasks_count = 0;
        self.pending_write_count = 0;

        self.flush_errors(mode)
    }

    fn flush_errors(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        if mode.flush {
            let errors = self.flush_errors_queue();

            if !mode.ignore && !errors.is_empty() {
                let error = ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                };
                return Err(error);
            }
        } else if !mode.ignore && !self.errors.is_empty() {
            let error = ServerError::ServerUnhealthy {
                errors: self.errors.clone(),
                backtrace: BackTrace::capture(),
            };
            return Err(error);
        }

        Ok(())
    }

    fn register_pipeline<'a>(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: impl Iterator<Item = &'a WgpuResource>,
        dispatch: &CubeCount,
    ) {
        if dispatch.is_empty() {
            return;
        }

        let entries = resources
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.as_wgpu_bind_resource(),
            })
            .collect::<Vec<_>>();

        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        let pass = self.compute_pass.get_or_insert_with(|| {
            let writes = if let Timings::Device(query_time) = &mut self.timings {
                query_time
                    .register_profile_device(&self.device)
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    })
            } else {
                None
            };
            self.encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: writes,
                })
                .forget_lifetime()
        });

        self.tasks_count += 1;

        let group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &entries,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        match dispatch.clone() {
            CubeCount::Static(x, y, z) => {
                pass.dispatch_workgroups(x, y, z);
            }
            CubeCount::Dynamic(binding) => {
                let res = self.mem_manage.get_resource(binding).unwrap();
                pass.dispatch_workgroups_indirect(&res.buffer, res.offset);
            }
        }
        self.flush_if_needed();
    }

    pub(crate) fn flush_errors_queue(&mut self) -> Vec<ServerError> {
        let errors = core::mem::take(&mut self.errors);

        if !errors.is_empty() {
            self.profile_error(ProfileError::Unknown {
                reason: alloc::format!("{:?}", errors),
                backtrace: BackTrace::capture(),
            });
        }

        errors
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
                        if let Err(e) = device.poll(wgpu::PollType::Wait {
                            submission_index: Some(index),
                            timeout: None,
                        }) {
                            log::warn!(
                                "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                            )
                        }
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
