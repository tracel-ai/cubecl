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

/// Metal reports no GPU core count, so the runtime reads the one the driver states. On Apple
/// silicon it is there, and it is a count a GPU has: a grid sized against a made-up one is wrong
/// everywhere, and silently.
#[cfg(all(feature = "msl", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn a_mac_gpu_reports_its_cores() {
    use cubecl_server::runtime::Runtime;

    let client = <cubecl_wgpu::WgpuRuntime>::client(&Default::default());
    let cores = client.properties().hardware.num_streaming_multiprocessors;
    let cores = cores.expect("an Apple silicon GPU states its core count");
    assert!(
        (1..=1024).contains(&cores),
        "{cores} is not a core count any GPU has"
    );
}

/// A card's description states the capacity a client on it reports, read without opening a
/// device, so a listing can say what fits on a card before anything runs there.
#[cfg(feature = "spirv")]
#[test]
fn a_vulkan_card_describes_the_capacity_its_client_reports() {
    use cubecl_common::device::Device;
    use cubecl_server::runtime::Runtime;
    use cubecl_wgpu::{WgpuBackend, WgpuDevice, WgpuRuntime};

    for id in <WgpuRuntime>::enumerate_all_devices() {
        let device = WgpuDevice::from_id(id).on(WgpuBackend::Vulkan);
        let described = <WgpuRuntime>::description(&device).and_then(|card| card.max_memory);
        let reported = <WgpuRuntime>::client(&device)
            .properties()
            .memory
            .max_memory();

        assert_eq!(described, reported);
    }
}
