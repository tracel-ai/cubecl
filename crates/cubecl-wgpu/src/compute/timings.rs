use cubecl_common::profile::{ProfileDuration, ProfileTicks};
use cubecl_core::{future::DynFut, server::ProfilingToken};
use hashbrown::HashMap;
use wgpu::{QuerySet, QueryType, wgt::QuerySetDescriptor};

use super::WgpuResource;

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

type QuerySetId = u64;

impl QueryProfiler {
    pub fn new(queue: &wgpu::Queue) -> Self {
        Self {
            cleanups: Vec::new(),
            counter_query_set: 0,
            counter_token: 0,
            query_sets: HashMap::new(),
            current: None,
            timestamps: HashMap::new(),
            init_tokens: Vec::new(),
            queue_period: queue.get_timestamp_period() as f64,
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
    pub fn stop_profile(&mut self, token: ProfilingToken) -> Option<(u64, u64)> {
        let mut timestamps = self.timestamps.remove(&token).unwrap();
        let Timestamp { start, end } = &mut timestamps;
        *end = self.current;

        if let (Some(start), Some(end)) = (start, end) {
            let query_set_start = self.query_sets.get_mut(start).unwrap();
            query_set_start.num_ref -= 1;
            if query_set_start.num_ref == 0 {
                self.cleanups.push(*start);
            }
            Some((*start, *end))
        } else {
            None
        }
    }

    pub fn resolve_profiles(
        &self,
        start: u64,
        end: u64,
        encoder: &mut wgpu::CommandEncoder,
        resource_start: &WgpuResource,
        resource_end: &WgpuResource,
    ) {
        let query_set_start = self.query_sets.get(&start).unwrap();
        let query_set_end = self.query_sets.get(&end).unwrap();
        encoder.resolve_query_set(
            &query_set_start.query_set,
            0..1,
            resource_start.buffer(),
            resource_start.offset(),
        );
        encoder.resolve_query_set(
            &query_set_end.query_set,
            1..2,
            resource_end.buffer(),
            resource_end.offset(),
        );
    }

    pub fn duration_from_fut(&self, fut: DynFut<Vec<Vec<u8>>>) -> ProfileDuration {
        let period = self.queue_period;

        ProfileDuration::from_future(async move {
            let result = fut.await;
            let data_start: u64 = bytemuck::try_cast_slice(&result[0]).unwrap()[0];
            let data_end: u64 = bytemuck::try_cast_slice(&result[1]).unwrap()[0];
            // TODO: Offset by base tick.
            let ns_start = (data_start as f64 * period) as u128;
            let ns_end = (data_end as f64 * period) as u128;
            ProfileTicks::from_start_end(ns_start, ns_end)
        })
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
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("CubeCL profile queries"),
            ty: QueryType::Timestamp,
            count: 2,
        });

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
