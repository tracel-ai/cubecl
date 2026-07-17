use cubecl_common::device::{Device, DeviceId};
use cubecl_environment::collections::HashMap;
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
///
/// `MTLCreateSystemDefaultDevice` can return `nil` on some configurations (e.g.
/// certain macOS versions/headless contexts) even when a valid GPU exists, so we
/// fall back to the first device reported by `MTLCopyAllDevices`.
pub fn default_device() -> Option<objc2::rc::Retained<ProtocolObject<dyn MTLDevice>>> {
    objc2_metal::MTLCreateSystemDefaultDevice().or_else(|| all_devices().into_iter().next())
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

/// Registers an existing `MTLDevice` and returns a `MetalDevice::Existing` handle,
/// for integrating with Metal code that already owns a device.
pub fn register_device(device: Retained<ProtocolObject<dyn MTLDevice>>) -> MetalDevice {
    let id = registry().lock().unwrap().register(device);
    MetalDevice::Existing(id)
}

/// Get a registered Metal device by its ID.
pub(crate) fn get_existing_device(id: u32) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    registry().lock().unwrap().get(id)
}
