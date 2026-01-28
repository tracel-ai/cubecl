use cubecl_common::device::{Device, DeviceId};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::fmt;

/// Metal device representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetalDevice {
    /// Default Metal device (usually the first GPU)
    DefaultDevice,
    /// Discrete GPU by index
    DiscreteGpu(usize),
    /// Integrated GPU by index
    IntegratedGpu(usize),
    /// Existing device with unique ID
    Existing(u32),
}

impl Default for MetalDevice {
    fn default() -> Self {
        Self::DefaultDevice
    }
}

impl Device for MetalDevice {
    fn from_id(device_id: DeviceId) -> Self {
        match device_id.type_id {
            0 => Self::DefaultDevice,
            1 => Self::DiscreteGpu(device_id.index_id as usize),
            2 => Self::IntegratedGpu(device_id.index_id as usize),
            3 => Self::Existing(device_id.index_id),
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
                index_id: *idx as u32,
            },
            Self::IntegratedGpu(idx) => DeviceId {
                type_id: 2,
                index_id: *idx as u32,
            },
            Self::Existing(id) => DeviceId {
                type_id: 3,
                index_id: *id,
            },
        }
    }

    fn device_count(_type_id: u16) -> usize {
        all_devices().len()
    }
}

impl fmt::Display for MetalDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DefaultDevice => write!(f, "Metal (default)"),
            Self::DiscreteGpu(idx) => write!(f, "Metal DiscreteGpu {}", idx),
            Self::IntegratedGpu(idx) => write!(f, "Metal IntegratedGpu {}", idx),
            Self::Existing(id) => write!(f, "Metal Device {}", id),
        }
    }
}

/// Get the default Metal device
pub fn default_device() -> Option<objc2::rc::Retained<ProtocolObject<dyn MTLDevice>>> {
    objc2_metal::MTLCreateSystemDefaultDevice()
}

/// Get all available Metal devices
pub fn all_devices() -> Vec<objc2::rc::Retained<ProtocolObject<dyn MTLDevice>>> {
    let devices_array = objc2_metal::MTLCopyAllDevices();
    // Convert NSArray to Vec by iterating
    devices_array.iter().map(|d| d.clone()).collect()
}
