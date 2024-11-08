use web_time::Instant;
use wgpu::{QuerySet, QuerySetDescriptor, QueryType};

#[derive(Debug)]
pub enum KernelTimestamps {
    Native { query_set: QuerySet, init: bool },
    Inferred { start_time: Instant },
    Disabled,
}

impl KernelTimestamps {
    pub fn enable(&mut self, device: &wgpu::Device) {
        if !matches!(self, Self::Disabled) {
            return;
        }

        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let query_set = device.create_query_set(&QuerySetDescriptor {
                label: Some("CubeCL profile queries"),
                ty: QueryType::Timestamp,
                count: 2,
            });

            *self = Self::Native {
                query_set,
                init: false,
            };
        } else {
            *self = Self::Inferred {
                start_time: Instant::now(),
            };
        };
    }

    pub fn disable(&mut self) {
        *self = Self::Disabled;
    }

    pub fn duplicate(&self, device: &wgpu::Device) -> Self {
        match self {
            KernelTimestamps::Native { .. } => {
                let query_set = device.create_query_set(&QuerySetDescriptor {
                    label: Some("CubeCL profile queries"),
                    ty: QueryType::Timestamp,
                    count: 2,
                });
                Self::Native {
                    query_set,
                    init: false,
                }
            }
            KernelTimestamps::Inferred { .. } => Self::Inferred {
                start_time: Instant::now(),
            },
            KernelTimestamps::Disabled => Self::Disabled,
        }
    }
}
