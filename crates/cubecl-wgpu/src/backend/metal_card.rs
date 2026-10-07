use cubecl_ir::{PhysicalDevice, RegistryEntryId};
use objc2_metal::MTLDevice;
use wgpu::hal;

pub fn describe_card(adapter: &wgpu::Adapter, physical: &mut PhysicalDevice) {
    // SAFETY: the Metal device is only read, while `adapter` keeps it alive.
    let Some(hal_adapter) = (unsafe { adapter.as_hal::<hal::api::Metal>() }) else {
        return;
    };
    let registry_id = hal_adapter.raw_device().registryID();
    physical.registry_entry_id = Some(RegistryEntryId::new(registry_id));
}
