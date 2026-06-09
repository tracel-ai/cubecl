use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use cubecl_common::profile::{Duration, Instant, ProfileDuration, ProfileTicks};
use cubecl_core::{
    backtrace::BackTrace,
    server::{ProfileError, ProfilingToken},
};
use hashbrown::HashMap;
use wgpu::{QUERY_SIZE, QuerySet, QuerySetDescriptor, QueryType};

type QuerySetId = u64;

/// Metal caps live `MTLCounterSampleBuffer`s at 32 per device; leave a little headroom.
const DEFAULT_MAX_METAL_TIMING_QUERY_SETS: u32 = 28;

/// Bounds the live timestamp [`QuerySet`]s on a single device.
///
/// Each timestamp query set is backed by a Metal `MTLCounterSampleBuffer`, capped at 32 live
/// per device; exceeding it fails the allocation, poisons profiling, and panics. Shared (via
/// [`Arc`]) by every [`QueryProfiler`] on the device so the total stays under the limit no
/// matter how many streams profile at once. Lock-free. Non-Metal backends use an
/// [unbounded](Self::unbounded) budget and are unaffected.
#[derive(Debug)]
pub struct TimestampQuerySetBudget {
    live: AtomicU32,
    max: u32,
}

impl TimestampQuerySetBudget {
    /// Metal budget, overridable via `CUBECL_MAX_METAL_TIMING_QUERY_SETS`.
    pub fn metal() -> Self {
        let max = std::env::var("CUBECL_MAX_METAL_TIMING_QUERY_SETS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_METAL_TIMING_QUERY_SETS);
        Self {
            live: AtomicU32::new(0),
            max,
        }
    }

    /// Unbounded budget, for backends with no counter-sample-buffer limit.
    pub fn unbounded() -> Self {
        Self {
            live: AtomicU32::new(0),
            max: u32::MAX,
        }
    }

    /// Reserve one slot, lock-free. Returns `false` if already at `max`.
    pub(crate) fn try_acquire(&self) -> bool {
        let mut live = self.live.load(Ordering::Relaxed);
        loop {
            if live >= self.max {
                return false;
            }
            match self.live.compare_exchange_weak(
                live,
                live + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => live = actual,
            }
        }
    }

    /// Return `n` previously-acquired slots to the budget.
    fn release(&self, n: u32) {
        if n > 0 {
            self.live.fetch_sub(n, Ordering::Relaxed);
        }
    }
}

/// Per-profiler allocation of timestamp query sets, backed by a shared device budget.
///
/// Allocates a fresh query set (one more live counter sample buffer) per profile while it can
/// reserve a slot from the device [`budget`](TimestampQuerySetBudget); once the budget is
/// exhausted it reuses its own sets round-robin instead of allocating past the hardware limit.
/// Reuse is safe because all of a device's streams submit to a single ordered queue, so a
/// set's timestamp-write and resolve always execute atomically and in submission order — the
/// only cost is that a profile spanning more compute passes than this allocator has sets may
/// under-measure, which autotune tolerates. Releases every slot it holds on drop.
#[derive(Debug)]
struct QuerySetAllocator {
    /// Shared device budget of live timestamp query sets.
    budget: Arc<TimestampQuerySetBudget>,
    /// Budget slots currently held (released on drop). Starts at 1 — the caller reserves one
    /// slot before constructing the allocator.
    held: u32,
    /// Every distinct query set allocated so far; reused round-robin once the budget is
    /// exhausted.
    created: Vec<QuerySet>,
    /// Round-robin cursor into `created` for reuse.
    reuse_idx: usize,
}

impl QuerySetAllocator {
    /// Creates an allocator owning the one budget slot the caller reserved beforehand.
    fn new(budget: Arc<TimestampQuerySetBudget>) -> Self {
        Self {
            budget,
            held: 1,
            created: Vec::new(),
            reuse_idx: 0,
        }
    }

    /// Returns a query set for a new profile: a freshly allocated one while budget slots are
    /// available, otherwise an existing set reused round-robin.
    fn acquire(&mut self, device: &wgpu::Device) -> QuerySet {
        if (self.created.len() as u32) < self.held || self.budget.try_acquire() {
            // Hold a reserved slot we haven't materialised yet, or just acquired one.
            if self.created.len() as u32 >= self.held {
                self.held += 1;
            }
            let query_set = device.create_query_set(&QuerySetDescriptor {
                label: Some("CubeCL profile queries"),
                ty: QueryType::Timestamp,
                count: 2,
            });
            self.created.push(query_set.clone());
            query_set
        } else {
            let query_set = self.created[self.reuse_idx % self.created.len()].clone();
            self.reuse_idx = self.reuse_idx.wrapping_add(1);
            query_set
        }
    }
}

impl Drop for QuerySetAllocator {
    fn drop(&mut self) {
        // Return every reserved slot to the device budget.
        self.budget.release(self.held);
    }
}

#[derive(Debug)]
/// Struct encapsulating how timings are captured on wgpu.
pub struct QueryProfiler {
    timestamps: HashMap<ProfilingToken, Result<Timestamp, ProfileError>>,
    init_tokens: Vec<ProfilingToken>,
    query_set_pool: Vec<QuerySet>,
    query_sets: HashMap<QuerySetId, QuerySetItem>,
    /// Allocates this profiler's timestamp query sets within the shared device budget.
    allocator: QuerySetAllocator,
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

fn create_resolve_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CubeCL gpu -> cpu resolve buffer"),
        size: (QUERY_SIZE * count) as _,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_map_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CubeCL gpu -> cpu map buffer"),
        size: (QUERY_SIZE * count) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// Measure a timestamp to align the CPU & GPU timelines.
#[cfg(feature = "profile-tracy")]
fn get_cur_timestamp(queue: &wgpu::Queue, device: &wgpu::Device) -> u64 {
    // Make sure no work is outstanding.

    use wgpu::BufferAddress;
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None, // Wait for most recent
            timeout: None,
        })
        .unwrap();

    // Resolve a timestamp for the query set.
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("CubeCL gpu -> cpu sync query_set"),
        ty: wgpu::QueryType::Timestamp,
        count: 1,
    });

    let resolve_buffer = create_resolve_buffer(device, 1);
    let map_buffer = create_map_buffer(device, 1);

    let mut timestamp_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu-profiler gpu -> cpu query timestamp"),
    });
    // This compute pass is purely to get a timestamp.
    timestamp_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Write timestamp pass"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
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
    copy_encoder.copy_buffer_to_buffer(
        &resolve_buffer,
        0,
        &map_buffer,
        0,
        Some(QUERY_SIZE as BufferAddress),
    );

    let commands = [timestamp_encoder.finish(), copy_encoder.finish()];

    queue.submit(commands);
    map_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None, // Wait for most recent
            timeout: None,
        })
        .unwrap();

    let view = map_buffer.slice(..).get_mapped_range().unwrap();
    u64::from_le_bytes((*view).try_into().unwrap())
}

impl QueryProfiler {
    /// Creates a profiler drawing query sets from a shared device `budget`. The caller must
    /// have reserved one slot (via [`TimestampQuerySetBudget::try_acquire`]) beforehand; the
    /// profiler owns that slot and releases all it holds on drop.
    pub fn new(
        queue: &wgpu::Queue,
        #[allow(unused)] device: &wgpu::Device,
        budget: Arc<TimestampQuerySetBudget>,
    ) -> Self {
        #[cfg(feature = "profile-tracy")]
        let sync_timestamps = get_cur_timestamp(queue, device);

        #[cfg(not(feature = "profile-tracy"))]
        let sync_timestamps = 0;

        // Measure CPU timestamp to go along GPU timestamp.
        // This can't be 100% correct as this includes the time to rendezvous the GPU timestamp.
        // Guesstimate by saying the rendesvouz time is twice the submission time.
        let epoch_instant = Instant::now();

        Self {
            cleanups: Vec::new(),
            counter_query_set: 0,
            counter_token: 0,
            query_sets: HashMap::new(),
            query_set_pool: Vec::new(),
            allocator: QuerySetAllocator::new(budget),
            current: None,
            timestamps: HashMap::new(),
            init_tokens: Vec::new(),
            queue_period: queue.get_timestamp_period() as f64,
            epoch_instant,
            epoch_tick: sync_timestamps,
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
            Ok(Timestamp {
                start: None,
                end: None,
            }),
        );
        token
    }

    pub fn error(&mut self, error: ProfileError) {
        self.timestamps.iter_mut().for_each(|(_key, value)| {
            *value = Err(error.clone());
        });
    }

    /// Stop the profiling on a device.
    pub fn stop_profile_setup(
        &mut self,
        token: ProfilingToken,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Option<wgpu::Buffer>, ProfileError> {
        let timestamps =
            self.timestamps
                .remove(&token)
                .ok_or_else(|| ProfileError::NotRegistered {
                    backtrace: BackTrace::capture(),
                })?;
        let mut timestamps = timestamps?;
        let Timestamp { start, end } = &mut timestamps;

        *end = self.current;

        // TODO: We could optimize this by having a single handle for both `start` and `end`
        // when a single query_set is used, but it probably doesn't impact the real
        // performance all that much.
        let (Some(start), Some(end)) = (start, end) else {
            return Ok(None);
        };

        let query_set_error = || ProfileError::Unknown {
            reason: "Can't resolve the query sets".to_string(),
            backtrace: BackTrace::capture(),
        };

        let query_set_start = self.query_sets.get_mut(start).ok_or_else(query_set_error)?;

        query_set_start.num_ref -= 1;
        if query_set_start.num_ref == 0 {
            self.cleanups.push(*start);
        }

        // TODO: Could use a StagingBelt for a small speedup here.
        let resolve_start = create_resolve_buffer(device, 1);
        let resolve_end = create_resolve_buffer(device, 1);
        let map_buffer = create_map_buffer(device, 2);
        let query_set_start = self.query_sets.get(start).ok_or_else(query_set_error)?;

        let query_set_end = self.query_sets.get(end).ok_or_else(query_set_error)?;

        let size = QUERY_SIZE as u64;
        encoder.resolve_query_set(&query_set_start.query_set, 0..1, &resolve_start, 0);
        encoder.resolve_query_set(&query_set_end.query_set, 1..2, &resolve_end, 0);
        encoder.copy_buffer_to_buffer(&resolve_start, 0, &map_buffer, 0, size);
        encoder.copy_buffer_to_buffer(&resolve_end, 0, &map_buffer, size, size);
        Ok(Some(map_buffer))
    }

    pub fn stop_profile(
        &self,
        map_buffer: Option<wgpu::Buffer>,
        poll_signal: Arc<()>,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Some(map_buffer) = map_buffer {
            let period = self.queue_period;
            let epoch_tick = self.epoch_tick;
            let epoch_instant = self.epoch_instant;

            Ok(ProfileDuration::new_device_time(async move {
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

                let binding = map_buffer.slice(..).get_mapped_range().unwrap();
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
            }))
        } else {
            // If there was no work done between the start and stop of the profile, logically the
            // time should be 0. We could use a ProfileDuration::from_duration here,
            // but it seems better to always return things as 'device' timing method.
            let now = Instant::now();
            Ok(ProfileDuration::new_device_time(async move {
                ProfileTicks::from_start_end(now, now)
            }))
        }
    }

    /// Returns the query set to be used by the [`wgpu::ComputePass`].
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
        let query_set = match self.query_set_pool.pop() {
            // Recycle a set we already own and are done with.
            Some(pool) => pool,
            // Otherwise allocate a new set or reuse one within the device budget.
            None => self.allocator.acquire(device),
        };

        let slot = QuerySetItem { query_set, num_ref };
        self.query_sets.insert(query_set_id, slot);
        self.query_sets.get_mut(&query_set_id).unwrap()
    }

    fn init_query_set(&mut self) -> Option<(QuerySetId, u32)> {
        let mut query_set_id = None;
        let mut count = 0;

        for token in self.init_tokens.drain(..) {
            if let Some(Ok(Timestamp { start, .. })) = &mut self.timestamps.get_mut(&token) {
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
            let removed = self
                .query_sets
                .remove(&key)
                .expect("Unknown query set cleaned up");
            self.query_set_pool.push(removed.query_set);
        }
    }
}
