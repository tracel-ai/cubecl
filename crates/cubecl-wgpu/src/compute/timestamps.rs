use web_time::Instant;
use wgpu::{QuerySet, QuerySetDescriptor, QueryType};

#[derive(Debug)]
pub enum KernelTimestamps {
    Device { query_set: QuerySet, init: bool },
    Full { start_time: Instant },
    Disabled,
}

impl KernelTimestamps {
    pub fn start(&mut self, device: &wgpu::Device) {
        if !matches!(self, Self::Disabled) {
            panic!("Cannot recursively measure timestamps currently.");
        }

        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let query_set = device.create_query_set(&QuerySetDescriptor {
                label: Some("CubeCL profile queries"),
                ty: QueryType::Timestamp,
                count: 2,
            });
            *self = Self::Device {
                query_set,
                init: false,
            };
        } else {
            *self = Self::Full {
                start_time: Instant::now(),
            };
        };
    }

    pub fn stop(&mut self) -> Self {
        std::mem::replace(self, Self::Disabled)
    }
}
