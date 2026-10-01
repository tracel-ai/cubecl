//! Native device requests reject mismatched adapters without panicking.
#![cfg(all(
    not(target_family = "wasm"),
    any(feature = "spirv", all(feature = "msl", target_os = "macos"))
))]

use cubecl_environment::future::block_on;
use cubecl_wgpu::{WgpuInitError, wgpu};

fn noop_adapter() -> wgpu::Adapter {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::NOOP;
    descriptor.backend_options.noop.enable = true;
    let instance = wgpu::Instance::new(descriptor);
    block_on(instance.request_adapter(&Default::default())).unwrap()
}

#[cfg(feature = "spirv")]
#[test]
fn vulkan_rejects_a_non_vulkan_adapter() {
    let result = block_on(cubecl_wgpu::vulkan::try_request_vulkan_device(
        &noop_adapter(),
    ));
    assert!(matches!(
        result,
        Err(WgpuInitError::InvalidConfiguration { .. })
    ));
}

#[cfg(all(feature = "msl", target_os = "macos"))]
#[test]
fn metal_rejects_a_non_metal_adapter() {
    let result = block_on(cubecl_wgpu::metal::try_request_metal_device(&noop_adapter()));
    assert!(matches!(
        result,
        Err(WgpuInitError::InvalidConfiguration { .. })
    ));
}
