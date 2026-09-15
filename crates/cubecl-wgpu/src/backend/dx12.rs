use cubecl_ir::{AdapterLuid, PhysicalDevice};
use wgpu::hal;

/// Fills the dedicated memory and the adapter LUID from DXGI, which reports no PCI address or UUID.
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
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
    bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
    physical.luid = Some(AdapterLuid::new(bytes));
    physical.total_memory = Some(description.DedicatedVideoMemory as u64);
}
