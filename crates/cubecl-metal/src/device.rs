use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::Mutex;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::sync::OnceLock;

pub use cubecl_server::device::MetalDevice;

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
    let id = registry().lock().register(device);
    MetalDevice::Existing(id)
}

/// Get a registered Metal device by its ID.
pub(crate) fn get_existing_device(id: u32) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    registry().lock().get(id)
}
