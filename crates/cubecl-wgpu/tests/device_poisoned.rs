//! Tests for a poisoned wgpu device.
//!
//! wgpu bounds-checks every buffer access, so a kernel cannot fault the way a
//! CUDA kernel does; what it can do is lose the device — a driver reset, a
//! timeout, a GPU that went away. `Device::destroy` loses it on purpose through
//! the same path, and the device-lost callback fires just as it would then.
//!
//! Every sync point after that must fail and say the device is poisoned, never
//! panic.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{
    AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_device, init_setup,
};

#[cube(launch)]
fn fill(out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = 7u32;
    }
}

#[test]
fn a_poisoned_device_surfaces_at_every_sync_point() {
    let setup = init_setup::<AutoGraphicsApi>(&WgpuDevice::default(), RuntimeOptions::default());
    let wgpu_device = setup.device.clone();
    let device = init_device(setup, RuntimeOptions::default());
    let client = <WgpuRuntime>::client(&device);
    let size = core::mem::size_of::<u32>();

    let launch_fill = |out: &cubecl_core::server::Handle| {
        fill::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
        );
    };

    // Healthy first, so the failures below are the poisoning and nothing else.
    let before = client.empty(size);
    launch_fill(&before);
    let bytes = client.read_one(before).expect("the device is healthy");
    assert_eq!(u32::from_bytes(&bytes), &[7]);

    wgpu_device.destroy();
    // `destroy` defers the device-lost notification to the device's next poll, once its queue
    // is empty: wait for it, so the sync points below see the poisoning.
    let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());

    let sync = cubecl_environment::future::block_on(client.sync());
    eprintln!("sync after the poisoning: {sync:?}");
    let sync = sync.expect_err("a sync on a poisoned device must fail");
    assert!(sync.is_device_poisoned(), "got: {sync}");

    let after = client.empty(size);
    launch_fill(&after);
    let read = client.read_one(after);
    eprintln!("read after the poisoning: {read:?}");
    let read = read.expect_err("nothing written after the poisoning can be trusted");
    assert!(read.is_device_poisoned(), "got: {read}");
}
