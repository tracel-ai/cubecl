use cubecl_ir::{AdapterLuid, PhysicalDevice};
use wgpu::hal;

/// Fills the adapter LUID from DXGI, which reports no PCI address.
pub fn describe_card(adapter: &wgpu::Adapter, physical: &mut PhysicalDevice) {
    // SAFETY: the DXGI adapter is only read, while `adapter` keeps it alive.
    let Some(hal_adapter) = (unsafe { adapter.as_hal::<hal::api::Dx12>() }) else {
        return;
    };
    // SAFETY: `GetDesc2` only fills the returned description.
    let Ok(description) = (unsafe { hal_adapter.raw_adapter().GetDesc2() }) else {
        return;
    };
    let luid = description.AdapterLuid;
    physical.luid = Some(AdapterLuid::from_parts(luid.LowPart, luid.HighPart));
}
