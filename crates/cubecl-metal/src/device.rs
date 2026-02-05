use cubecl_common::device::{Device, DeviceId};
use hashbrown::HashMap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::fmt;
use std::sync::{Mutex, OnceLock};

/// Metal device representation
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum MetalDevice {
    #[default]
    /// Default Metal device (usually the first GPU)
    DefaultDevice,
    /// Discrete GPU by index
    DiscreteGpu(usize),
    /// Integrated GPU by index
    IntegratedGpu(usize),
    /// Existing device with unique ID
    Existing(u32),
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

    fn device_count(type_id: u16) -> usize {
        let devices = all_devices();
        match type_id {
            0 => 1,                                                      // Default device
            1 => devices.iter().filter(|d| !(**d).isLowPower()).count(), // Discrete
            2 => devices.iter().filter(|d| (**d).isLowPower()).count(),  // Integrated
            3 => registry().lock().unwrap().devices.len(),               // Existing
            _ => 0,
        }
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
pub fn all_devices() -> Vec<Retained<ProtocolObject<dyn MTLDevice>>> {
    let devices_array = objc2_metal::MTLCopyAllDevices();
    devices_array.to_vec()
}

/// Registry for existing Metal devices.
struct DeviceRegistry {
    devices: HashMap<u32, Retained<ProtocolObject<dyn MTLDevice>>>,
    counter: u32,
}

impl DeviceRegistry {
    fn new() -> Self {
        Self {
            devices: HashMap::new(),
            counter: 0,
        }
    }

    fn register(&mut self, device: Retained<ProtocolObject<dyn MTLDevice>>) -> u32 {
        let id = self.counter;
        self.counter += 1;
        self.devices.insert(id, device);
        id
    }

    fn get(&self, id: u32) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
        self.devices.get(&id).cloned()
    }
}

static DEVICE_REGISTRY: OnceLock<Mutex<DeviceRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<DeviceRegistry> {
    DEVICE_REGISTRY.get_or_init(|| Mutex::new(DeviceRegistry::new()))
}

/// Register an existing Metal device and return a `MetalDevice::Existing` handle.
///
/// This is useful when integrating with existing Metal code that already has a device.
pub fn register_device(device: Retained<ProtocolObject<dyn MTLDevice>>) -> MetalDevice {
    let id = registry().lock().unwrap().register(device);
    MetalDevice::Existing(id)
}

/// Get a registered Metal device by its ID.
pub(crate) fn get_existing_device(id: u32) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    registry().lock().unwrap().get(id)
}
