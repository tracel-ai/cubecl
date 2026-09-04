use cubecl_common::device::{Device, DeviceId};

/// A device of the native Metal runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum MetalDevice {
    #[default]
    /// Default Metal device (usually the first GPU)
    DefaultDevice,
    /// Discrete GPU by index
    DiscreteGpu(usize),
    /// Integrated GPU by index
    IntegratedGpu(usize),
    /// Existing device with unique ID, registered through the Metal runtime.
    Existing(u32),
}

impl Device for MetalDevice {
    fn from_id(device_id: DeviceId) -> Self {
        match device_id.type_id {
            0 => Self::DefaultDevice,
            1 => Self::DiscreteGpu(device_id.index_id as usize),
            2 => Self::IntegratedGpu(device_id.index_id as usize),
            3 => Self::Existing(device_id.index_id as u32),
            _ => panic!("Invalid Metal device ID: {:?}", device_id),
        }
    }

    fn to_id(&self) -> DeviceId {
        match self {
            Self::DefaultDevice => DeviceId {
                type_id: 0,
                index_id: 0,
            },
            Self::DiscreteGpu(idx) => DeviceId {
                type_id: 1,
                index_id: *idx as u16,
            },
            Self::IntegratedGpu(idx) => DeviceId {
                type_id: 2,
                index_id: *idx as u16,
            },
            Self::Existing(id) => DeviceId {
                type_id: 3,
                index_id: *id as u16,
            },
        }
    }
}

impl core::fmt::Display for MetalDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DefaultDevice => write!(f, "Metal (default)"),
            Self::DiscreteGpu(idx) => write!(f, "Metal DiscreteGpu {}", idx),
            Self::IntegratedGpu(idx) => write!(f, "Metal IntegratedGpu {}", idx),
            Self::Existing(id) => write!(f, "Metal Device {}", id),
        }
    }
}
