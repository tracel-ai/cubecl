use super::{mem_manager::WgpuMemManager, poll::WgpuPoll, timings::QueryProfiler};
use cubecl_common::profile::{ProfileDuration, TimingMethod};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    future::{self, DynFut},
    server::{Binding, Bindings, Handle, ProfileError, ProfilingToken},
};
use cubecl_runtime::{
    memory_management::MemoryDeviceProperties, storage::AllocError,
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
    sync_buffer: Option<Handle>,
    compute_pass: Option<wgpu::ComputePass<'static>>,
    timings: Timings,
    tasks_count: usize,
    tasks_max: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    poll: WgpuPoll,
    submission_load: SubmissionLoad,
}

impl WgpuStream {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        tasks_max: usize,
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
        let mut mem_manage = WgpuMemManager::new(device.clone(), memory_properties, memory_config);

        // Allocate a small buffer to use for synchronization.
        #[cfg(target_family = "wasm")]
        let sync_buffer = Some(mem_manage.reserve(32, false));

        #[cfg(not(target_family = "wasm"))]
        let sync_buffer = None;

        Self {
            mem_manage,
            compute_pass: None,
            timings,
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
            sync_buffer,
            submission_load: SubmissionLoad::default(),
        }
    }

    pub fn register(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        bindings: Bindings,
        dispatch: &CubeCount,
    ) {
        let dispatch_resource = match dispatch.clone() {
            CubeCount::Static(_, _, _) => None,
            CubeCount::Dynamic(binding) => Some(self.mem_manage.get_resource(binding)),
        };

        let info = (!bindings.metadata.data.is_empty()).then(|| {
            self.create(bytemuck::cast_slice(&bindings.metadata.data))
                .expect("Must be able to allocate metadata")
        });
        let scalars = bindings
            .scalars
            .values()
            .map(|s| {
                self.create(s.data())
                    .expect("Must be able to allocate metadata")
            })
            .collect::<Vec<_>>();

        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = bindings
            .buffers
            .iter()
            .map(|b| self.mem_manage.get_resource(b.clone()))
            .collect::<Vec<_>>();

        if let Some(info) = info {
            resources.push(self.mem_manage.get_resource(info.binding()));
        }

        resources.extend(
            scalars
                .iter()
                .map(|s| self.mem_manage.get_resource(s.clone().binding())),
        );

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
            CubeCount::Static(x, y, z) => {
                pass.dispatch_workgroups(*x, *y, *z);
            }
            CubeCount::Dynamic(_) => {
                let res = dispatch_resource.unwrap();
                pass.dispatch_workgroups_indirect(res.buffer(), res.offset());
            }
        }
        self.flush_if_needed();
    }

    pub fn read_buffers(&mut self, bindings: Vec<Binding>) -> DynFut<Vec<Vec<u8>>> {
        self.compute_pass = None;
        let mut staging_buffers = Vec::with_capacity(bindings.len());
        let mut callbacks = Vec::with_capacity(bindings.len());

        for binding in bindings {
            // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
            // memory is 32 bytes aligned (see WgpuStorage).
            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let resource = self.mem_manage.get_resource(binding);
            let aligned_len = resource.size().div_ceil(align) * align;
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: aligned_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.encoder.copy_buffer_to_buffer(
                resource.buffer(),
                resource.offset(),
                &staging_buffer,
                0,
                aligned_len,
            );
            staging_buffers.push((staging_buffer, resource.size()));
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

        Box::pin(async move {
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
        })
    }

    // Bit silly but needed to make the borrow checker happy.
    fn system_profiler(&mut self) -> &mut TimestampProfiler {
        let Timings::System(timing) = &mut self.timings else {
            panic!("Unexpected timings type");
        };
        timing
    }

    pub fn start_profile(&mut self) -> ProfilingToken {
        match &mut self.timings {
            Timings::System(..) => {
                // Sync before profiling as well to get a cleaner measurement, we don't want to
                // include any queued up work so far.
                future::block_on(self.sync());
                self.system_profiler().start()
            }
            Timings::Device(query) => {
                // Close the current compute pass so that we start a new one. This keeps
                // the timestamps separated.
                self.compute_pass = None;
                query.start_profile()
            }
        }
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        match &mut self.timings {
            Timings::System(..) => {
                // Nb: WASM _has_ to use device timing and will panic here if query timestamps are not supported.
                future::block_on(self.sync());
                self.system_profiler().stop(token)
            }
            Timings::Device(..) => {
                let poll = self.poll.start_polling();

                self.compute_pass = None;

                // Submit commands needed for profiling.
                let buffer = {
                    let Timings::Device(timing) = &mut self.timings else {
                        panic!("Unexpected timings type");
                    };
                    timing.stop_profile_setup(token, &self.device, &mut self.encoder)
                };

                // Flush commands.
                self.flush();

                let Timings::Device(timing) = &mut self.timings else {
                    panic!("Unexpected timings type");
                };

                timing.stop_profile(buffer, poll)
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
                let fut = self.read_buffers(vec![buf.clone().binding()]);
                core::mem::swap(&mut buffer, &mut self.sync_buffer);
                Box::pin(async move {
                    fut.await;
                })
            }
            None => {
                #[cfg(not(target_family = "wasm"))]
                {
                    if let Err(e) = self.device.poll(wgpu::PollType::Wait) {
                        log::warn!(
                            "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                        )
                    }
                    Box::pin(async move {})
                }

                #[cfg(target_family = "wasm")]
                {
                    panic!("Only synching from a buffer is supported.");
                }
            }
        }
    }

    pub fn empty(&mut self, size: u64) -> Result<Handle, AllocError> {
        self.mem_manage.reserve(size, false)
    }

    pub fn create(&mut self, data: &[u8]) -> Result<Handle, AllocError> {
        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is 32 bytes aligned (see WgpuStorage).
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let aligned_len = (data.len() as u64).div_ceil(align) * align;

        // We'd like to keep operations as one long ComputePass. To do so, all copy operations happen
        // at the start of the encoder, and all execute operations afterwards. For this re-ordering to be valid,
        // a buffer we copy to MUST not have any outstanding compute work associated with it.
        // Any handles with compute work are kept in pending operations,
        // and the allocation here won't try to use that buffer.
        let alloc = self.mem_manage.reserve(aligned_len, true)?;
        self.copy_to_handle(alloc.clone(), data);
        Ok(alloc)
    }

    pub fn copy_to_handle(&mut self, handle: Handle, data: &[u8]) {
        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is 32 bytes aligned (see WgpuStorage).
        let size = handle.size();

        // We'd like to keep operations as one long ComputePass. To do so, all copy operations happen
        // at the start of the encoder, and all execute operations afterwards. For this re-ordering to be valid,
        // a buffer we copy to MUST not have any outstanding compute work associated with it.
        // Any handles with compute work are kept in pending operations,
        // and the allocation here won't try to use that buffer.
        let resource = self.mem_manage.get_resource(handle.binding());

        // Nb: using write_buffer_with here has no advantages. It'd only be faster if create() would expose
        // its API as a slice to write into.
        //
        // write_buffer is the recommended way to write this data, as:
        // - On WebGPU, from WASM, this can save a copy to the JS memory.
        // - On devices with unified memory, this could skip the staging buffer entirely.
        if size != data.len() as u64 {
            let mut buffer = self
                .queue
                .write_buffer_with(
                    resource.buffer(),
                    resource.offset(),
                    NonZero::new(size).unwrap(),
                )
                .unwrap();
            buffer[0..data.len()].copy_from_slice(data);
        } else {
            self.queue
                .write_buffer(resource.buffer(), resource.offset(), data);
        }

        self.flush_if_needed();
    }

    fn flush_if_needed(&mut self) {
        // Flush when there are too many tasks, or when too many handles are locked.
        // Locked handles should only accumulate in rare circumstances (where uniforms
        // are being created but no work is submitted).
        if self.tasks_count >= self.tasks_max || self.mem_manage.needs_flush(self.tasks_max * 8) {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
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

        // All buffers are submitted, so don't need to exclude them anymore.
        self.mem_manage.clear_pending();
        // Cleanup allocations and deallocations.
        self.mem_manage.memory_cleanup(false);

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
                        if let Err(e) = device.poll(wgpu::PollType::WaitForSubmissionIndex(index)) {
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
