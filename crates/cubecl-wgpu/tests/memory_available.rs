use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

type R = WgpuRuntime;

/// A Vulkan device with `VK_EXT_memory_budget` answers a positive figure; any other device or
/// backend answers `None`, never zero.
#[test]
fn a_device_reports_the_memory_it_has_left_or_nothing() {
    let client = R::client(&WgpuDevice::default());
    let available = client.memory_available();
    if let Some(free) = available {
        assert!(free > 0, "a budget the driver reports is never empty");
    }
}
