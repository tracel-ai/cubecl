use cubecl_core::server::ProfilingToken;
use hashbrown::HashMap;
use web_time::Instant;
use wgpu::{CommandEncoder, QuerySet, QueryType, wgt::QuerySetDescriptor};

use crate::WgpuResource;

#[derive(Debug, Default)]
/// Struct encapsulating how timings are captured on wgpu.
pub struct Timings {
    timestamps: HashMap<ProfilingToken, Timestamp>,
    init_tokens: Vec<ProfilingToken>,
    query_sets: HashMap<QuerySetId, QuerySetItem>,
    current: Option<u64>,
    counter_token: u64,
    counter_query_set: u64,
    cleanups: Vec<QuerySetId>,
}

#[derive(Debug)]
pub enum Timestamp {
    Device {
        start: Option<u64>,
        end: Option<u64>,
    },
    Full {
        start_time: Instant,
    },
}

#[derive(Debug)]
struct QuerySetItem {
    query_set: QuerySet,
    // We only track references to the start query set
    num_ref: u32,
}

type QuerySetId = u64;

impl Timings {
    /// Prepare a new profiling creating a new [token](ProfilingToken).
    pub fn start_profile_prepare(&mut self) -> ProfilingToken {
        let token = ProfilingToken {
            id: self.counter_token,
        };
        self.counter_token += 1;
        token
    }

    /// Start a new profiling using [device measurement](TimeMeasurement::Device).
    pub fn start_profile_device(&mut self, token: ProfilingToken) {
        self.init_tokens.push(token);
        self.timestamps.insert(
            token,
            Timestamp::Device {
                start: None,
                end: None,
            },
        );
    }

    /// Start a new profiling using [system measurement](TimeMeasurement::System).
    pub fn start_profile_system(&mut self, token: ProfilingToken) {
        self.timestamps.insert(
            token,
            Timestamp::Full {
                start_time: Instant::now(),
            },
        );
    }

    /// Prepare to stop a profiling using a [token](ProfilingToken).
    pub fn stop_profile_prepare(&mut self, token: ProfilingToken) -> Timestamp {
        let mut timestamps = self.timestamps.remove(&token).unwrap();

        if let Timestamp::Device { end, .. } = &mut timestamps {
            *end = self.current;
        }

        timestamps
    }

    /// Stop the profiling on a device.
    pub fn stop_profile_device(
        &mut self,
        start: QuerySetId,
        end: QuerySetId,
        encoder: &mut CommandEncoder,
        resource_start: WgpuResource,
        resource_end: WgpuResource,
    ) {
        let query_set_start = self.query_sets.get_mut(&start).unwrap();
        query_set_start.num_ref -= 1;

        if query_set_start.num_ref == 0 {
            self.cleanups.push(start);
        }

        encoder.resolve_query_set(
            &query_set_start.query_set,
            0..1,
            resource_start.buffer(),
            resource_start.offset(),
        );

        let query_set_end = self.query_sets.get_mut(&end).unwrap();

        encoder.resolve_query_set(
            &query_set_end.query_set,
            1..2,
            resource_end.buffer(),
            resource_end.offset(),
        );
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
            if let Some(Timestamp::Device { start, .. }) = &mut self.timestamps.get_mut(&token) {
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
