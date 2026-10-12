//! The interop surface a downstream runtime reaches its own server through.
//!
//! Naming the server type, resolving a handle's resource and sealing a stream's
//! queued work are all public, and this drives them from outside the crate on a
//! real Metal device, the way an external library sharing the same device does.
#![cfg(all(target_os = "macos", not(feature = "spirv")))]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_environment::future::block_on;
use cubecl_environment::stream::StreamId;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{
    AutoCompiler, MemoryConfiguration, Metal, PrimaryMemoryMode, RuntimeOptions, WgpuDevice,
    WgpuDeviceKind, WgpuRuntime, WgpuServer, init_device_for_graphics_api, wgpu,
};

/// Adds one to every element, so the read proves the launch's write happened.
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

#[test]
fn the_interop_surface_reaches_the_runtime_server() {
    if !metal_available() {
        eprintln!("skipping wgpu interop test: no Metal adapter is available");
        return;
    }

    let selector = WgpuDevice::new(WgpuDeviceKind::DefaultDevice);
    let options = RuntimeOptions {
        tasks_max: 32,
        memory_config: MemoryConfiguration::ExclusivePages,
        primary_memory: PrimaryMemoryMode::HostVisible,
    };
    let device = init_device_for_graphics_api::<Metal>(&selector, options);
    let client = <WgpuRuntime>::client(&device);
    let stream_id = StreamId::current();
    let handle = client.create_from_slice(u32::as_bytes(&[1, 2, 3, 4]));

    increment::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 4) },
    );

    let managed = client
        .get_resource::<WgpuServer<AutoCompiler>>(handle.clone())
        .unwrap();
    assert_ne!(managed.resource().allocation_id(), 0);

    let completion = client
        .with_server::<WgpuServer<AutoCompiler>, _>(|server| {
            server.submit_stream_completion(stream_id)
        })
        .expect("the client's server is the wgpu one")
        .unwrap();
    completion.wait().unwrap();
    drop(completion);

    let guard = managed.resource().map_read().unwrap();
    assert_eq!(u32::from_bytes(&guard), &[2, 3, 4, 5]);
}
