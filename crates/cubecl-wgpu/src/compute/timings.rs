use std::{sync::Arc, time::Duration};
use web_time::Instant;

use cubecl_common::profile::{ProfileDuration, ProfileTicks};
use cubecl_core::server::ProfilingToken;
use hashbrown::HashMap;
use wgpu::{ComputePassDescriptor, ComputePassTimestampWrites, QuerySet};

type QuerySetId = u64;

#[derive(Debug)]
/// Struct encapsulating how timings are captured on wgpu.
pub struct QueryProfiler {
    timestamps: HashMap<ProfilingToken, Timestamp>,
    init_tokens: Vec<ProfilingToken>,
    query_sets: HashMap<QuerySetId, QuerySetItem>,
    current: Option<u64>,
    counter_token: u64,
    counter_query_set: u64,
    cleanups: Vec<QuerySetId>,
    queue_period: f64,
    epoch_tick: u64,
    epoch_instant: Instant,
}

#[derive(Debug)]
pub struct Timestamp {
    start: Option<u64>,
    end: Option<u64>,
}

#[derive(Debug)]
struct QuerySetItem {
    query_set: QuerySet,
    // We only track references to the start query set
    num_ref: u32,
}

fn create_query_set(device: &wgpu::Device, count: u32) -> QuerySet {
    device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("CubeCL gpu -> cpu sync query_set"),
        ty: wgpu::QueryType::Timestamp,
        count,
    })
}

fn create_resolve_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CubeCL gpu -> cpu resolve buffer"),
        size: (wgpu::QUERY_SIZE * count) as _,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_map_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CubeCL gpu -> cpu map buffer"),
        size: (wgpu::QUERY_SIZE * count) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// Measure a timestamp to align the CPU & GPU timelines.
fn synchronize_timestamps(queue: &wgpu::Queue, device: &wgpu::Device) -> (Instant, u64) {
    // Make sure no work is outstanding.
    device.poll(wgpu::PollType::Wait).unwrap();

    // Resolve a timestamp for the query set.
    let query_set = create_query_set(device, 1);
    let resolve_buffer = create_resolve_buffer(device, 1);
    let map_buffer = create_map_buffer(device, 1);

    let mut timestamp_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu-profiler gpu -> cpu query timestamp"),
    });
    // This compute pass is purely to get a timestamp.
    timestamp_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Write timestamp pass"),
        timestamp_writes: Some(ComputePassTimestampWrites {
            query_set: &query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(0),
        }),
    });
    timestamp_encoder.write_timestamp(&query_set, 0);
    timestamp_encoder.resolve_query_set(&query_set, 0..1, &resolve_buffer, 0);
    // Workaround for https://github.com/gfx-rs/wgpu/issues/6406
    // TODO when that bug is fixed, merge these encoders together again
    let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu-profiler gpu -> cpu copy timestamp"),
    });
    copy_encoder.copy_buffer_to_buffer(&resolve_buffer, 0, &map_buffer, 0, wgpu::QUERY_SIZE as _);

    let commands = [timestamp_encoder.finish(), copy_encoder.finish()];

    queue.submit(commands);
    map_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());

    device.poll(wgpu::PollType::Wait).unwrap();
    // Measure CPU timestamp to go along GPU timestamp.
    // This can't be 100% correct as this includes the time to rendezvous the GPU timestamp.
    // Guesstimate by saying the rendesvouz time is twice the submission time.
    let cpu_time = Instant::now();
    let view = map_buffer.slice(..).get_mapped_range();
    (cpu_time, u64::from_le_bytes((*view).try_into().unwrap()))
}

impl QueryProfiler {
    pub fn new(queue: &wgpu::Queue, device: &wgpu::Device) -> Self {
        let sync_timestamps = synchronize_timestamps(queue, device);

        Self {
            cleanups: Vec::new(),
            counter_query_set: 0,
            counter_token: 0,
            query_sets: HashMap::new(),
            current: None,
            timestamps: HashMap::new(),
            init_tokens: Vec::new(),
            queue_period: queue.get_timestamp_period() as f64,
            epoch_instant: sync_timestamps.0,
            epoch_tick: sync_timestamps.1,
        }
    }

    /// Start a new profiling using [device measurement](TimeMeasurement::Device).
    pub fn start_profile(&mut self) -> ProfilingToken {
        let token = ProfilingToken {
            id: self.counter_token,
        };
        self.counter_token += 1;
        self.init_tokens.push(token);
        self.timestamps.insert(
            token,
            Timestamp {
                start: None,
                end: None,
            },
        );
        token
    }

    /// Stop the profiling on a device.
    pub fn stop_profile_setup(
        &mut self,
        token: ProfilingToken,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<wgpu::Buffer> {
        let mut timestamps = self.timestamps.remove(&token).unwrap();
        let Timestamp { start, end } = &mut timestamps;
        *end = self.current;

        // TODO: We could optimize this by having a single handle for both `start` and `end`
        // when a single query_set is used, but it probably doesn't impact much the real
        // performance.
        if let (Some(start), Some(end)) = (start, end) {
            let query_set_start = self.query_sets.get_mut(start).unwrap();
            query_set_start.num_ref -= 1;
            if query_set_start.num_ref == 0 {
                self.cleanups.push(*start);
            }

            // TODO: Maybe could pool these to reduce profiling overhead, or use a wgpu `StagingBelt`.
            let resolve_start = create_resolve_buffer(device, 1);
            let resolve_end = create_resolve_buffer(device, 1);
            let map_buffer = create_map_buffer(device, 2);

            let query_set_start = self.query_sets.get(start).unwrap();
            let query_set_end = self.query_sets.get(end).unwrap();

            encoder.resolve_query_set(&query_set_start.query_set, 0..1, &resolve_start, 0);
            encoder.resolve_query_set(&query_set_end.query_set, 1..2, &resolve_end, 0);

            // Resolve everything else in the future. Don't need this in any particular order,
            // so can do it when whatever thread picks this up.
            encoder.copy_buffer_to_buffer(
                &resolve_start,
                0,
                &map_buffer,
                0,
                wgpu::QUERY_SIZE as u64,
            );
            encoder.copy_buffer_to_buffer(
                &resolve_end,
                0,
                &map_buffer,
                wgpu::QUERY_SIZE as u64,
                wgpu::QUERY_SIZE as u64,
            );

            Some(map_buffer)
        } else {
            None
        }
    }

    pub fn stop_profile(
        &self,
        map_buffer: Option<wgpu::Buffer>,
        poll_signal: Arc<()>,
    ) -> ProfileDuration {
        if let Some(map_buffer) = map_buffer {
            let period = self.queue_period;
            let epoch_tick = self.epoch_tick;
            let epoch_instant = self.epoch_instant;

            ProfileDuration::new_device_time(async move {
                let (sender, rec) = async_channel::bounded(1);
                map_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| {
                        // This might fail if the channel is closed (eg. the future is dropped).
                        // This is fine, just means results aren't needed anymore.
                        let _ = sender.try_send(v);
                    });
                rec.recv()
                    .await
                    .expect("Unable to receive buffer slice result.")
                    .expect("Failed to map buffer");
                // Can stop polling now.
                core::mem::drop(poll_signal);

                let binding = map_buffer.slice(..).get_mapped_range();
                let data: &[u64] = bytemuck::try_cast_slice(&binding).unwrap();

                // Get nr. of ticks since epoch.
                let data_start = data[0].saturating_sub(epoch_tick);
                let data_end = data[1].saturating_sub(epoch_tick);
                drop(binding);

                map_buffer.unmap();
                // Convert to a duration.
                let start_duration = Duration::from_nanos((data_start as f64 * period) as u64);
                let end_duration = Duration::from_nanos((data_end as f64 * period) as u64);

                // Convert to an `Instant`.
                let instant_start = epoch_instant + start_duration;
                let instant_end = epoch_instant + end_duration;

                ProfileTicks::from_start_end(instant_start, instant_end)
            })
        } else {
            // If there was no work done between the start and stop of the profile, logically the
            // time should be 0. We could use a ProfileDuration::from_duration here,
            // but it seems better to always return things as 'device' timing method.
            let now = web_time::Instant::now();
            ProfileDuration::new_device_time(async move { ProfileTicks::from_start_end(now, now) })
        }
    }

    /// Returns the query set to be used by the [wgpu::ComputePass].
    ///
    /// Also performs cleanup of old [query set](QuerySet).
    pub fn register_profile_device(&mut self, device: &wgpu::Device) -> Option<&QuerySet> {
        self.init_query_set().map(|info| {
            let item = self.new_query_set(info, device);
            &item.query_set
        })
    }

    fn new_query_set(
        &mut self,
        query_set_info: (u64, u32),
        device: &wgpu::Device,
    ) -> &mut QuerySetItem {
        let (query_set_id, num_ref) = query_set_info;
        let query_set = create_query_set(device, 2);
        let slot = QuerySetItem { query_set, num_ref };
        self.query_sets.insert(query_set_id, slot);
        self.query_sets.get_mut(&query_set_id).unwrap()
    }

    fn init_query_set(&mut self) -> Option<(QuerySetId, u32)> {
        let mut query_set_id = None;
        let mut count = 0;

        for token in self.init_tokens.drain(..) {
            if let Some(Timestamp { start, .. }) = &mut self.timestamps.get_mut(&token) {
                count += 1;
                let id = match query_set_id {
                    Some(id) => id,
                    None => {
                        let id = self.counter_query_set;
                        self.counter_query_set += 1;
                        self.current = Some(id);
                        query_set_id = Some(id);
                        id
                    }
                };

                *start = Some(id);
            }
        }

        // We only cleanup old query sets when creating a new one, since we don't know if we need
        // the end timing of the last query set.
        self.cleanup_query_sets();

        query_set_id.map(|v| (v, count))
    }

    fn cleanup_query_sets(&mut self) {
        for key in self.cleanups.drain(..) {
            self.query_sets.remove(&key);
        }
    }
}
