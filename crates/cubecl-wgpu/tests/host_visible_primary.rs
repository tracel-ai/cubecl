//! The host-visible primary-memory pool on a real Metal device.
//!
//! Two things the Linux compile cannot show: that Metal actually grants
//! `MAPPABLE_PRIMARY_BUFFERS` for a host-visible primary pool, and that two
//! devices minted for the same graphics API selector are independent contexts
//! that each compute through their own pool.
#![cfg(all(target_os = "macos", not(feature = "spirv")))]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_environment::future::block_on;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{
    MemoryConfiguration, Metal, PrimaryMemoryMode, RuntimeOptions, WgpuDevice, WgpuDeviceKind,
    WgpuRuntime, init_device_for_graphics_api, wgpu,
};

/// Adds one to every element, so the read proves the whole buffer round-tripped
/// through the pool the launch wrote it in.
#[cube(launch)]
fn increment(values: &mut [u32]) {
    if ABSOLUTE_POS < values.len() {
        values[ABSOLUTE_POS] += 1;
    }
}

fn metal_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    !block_on(instance.enumerate_adapters(wgpu::Backends::METAL)).is_empty()
}

fn host_visible_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 32,
        memory_config: MemoryConfiguration::ExclusivePages,
        primary_memory: PrimaryMemoryMode::HostVisible,
    }
}

#[test]
fn independent_host_visible_metal_devices_compute_through_their_own_pool() {
    if !metal_available() {
        eprintln!("skipping host-visible primary test: no Metal adapter is available");
        return;
    }

    let selector = WgpuDevice::new(WgpuDeviceKind::DefaultDevice);
    let first = init_device_for_graphics_api::<Metal>(&selector, host_visible_options());
    let second = init_device_for_graphics_api::<Metal>(&selector, host_visible_options());

    assert_ne!(first, second, "each call mints its own context");
    assert!(matches!(first.kind, WgpuDeviceKind::Existing(_)));
    assert!(matches!(second.kind, WgpuDeviceKind::Existing(_)));

    for device in [first, second] {
        let client = <WgpuRuntime>::client(&device);
        let handle = client.create_from_slice(u32::as_bytes(&[1, 2, 3, 4]));

        increment::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            unsafe { BufferArg::from_raw_parts(handle.clone(), 4) },
        );

        let bytes = client.read_one(handle).unwrap();
        assert_eq!(u32::from_bytes(&bytes), &[2, 3, 4, 5]);
    }
}
