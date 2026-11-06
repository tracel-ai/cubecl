//! An allocation the device has no room for fails with `IoError::OutOfMemory`, and the device
//! stays usable: the failure is the allocation's, not the device's.
//!
//! The test fills the device's memory, so it runs only when asked (`--ignored`), and on a discrete
//! GPU only: elsewhere buffers come out of system memory.
#![cfg(not(feature = "spirv"))]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::IoError;
use cubecl_server::runtime::Runtime;
use cubecl_server::storage::ComputeStorage;
use cubecl_wgpu::{
    AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, WgpuStorage, init_device, init_setup,
};

#[cube(launch)]
fn fill(out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = 7u32;
    }
}

#[test]
#[ignore = "fills the GPU's memory"]
fn an_allocation_past_the_device_memory_fails_and_the_device_stays_usable() {
    let setup = init_setup::<AutoGraphicsApi>(&WgpuDevice::default(), RuntimeOptions::default());
    if setup.adapter.get_info().device_type != wgpu::DeviceType::DiscreteGpu {
        eprintln!("skipped: not a discrete GPU");
        return;
    }
    let wgpu_device = setup.device.clone();
    // Installs the uncaptured-error handler, which poisons the device on an out-of-memory error
    // that escapes the allocation.
    let device = init_device(setup, RuntimeOptions::default());
    let client = <WgpuRuntime>::client(&device);

    let mut storage = WgpuStorage::new(4, wgpu_device.clone(), wgpu::BufferUsages::STORAGE, false);
    // Fill the device with buffers of the largest size wgpu allows, until one has no room.
    let size = wgpu_device.limits().max_buffer_size;
    let mut held = Vec::new();
    let err = loop {
        match storage.alloc(size) {
            Ok(handle) => held.push(handle),
            Err(err) => break err,
        }
        assert!(held.len() < 1024, "{} buffers of {size} B fit", held.len());
    };
    assert!(
        matches!(err, IoError::OutOfMemory { size: failed, .. } if failed == size),
        "got: {err}"
    );
    assert_eq!(storage.bytes_allocated(), held.len() as u64 * size);

    for handle in held {
        storage.dealloc(handle.id);
    }
    let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());
    let small = storage.alloc(1024).expect("the memory is back");
    storage.dealloc(small.id);

    let out = client.empty(core::mem::size_of::<u32>());
    fill::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
    );
    let bytes = client.read_one(out).expect("the device is still usable");
    assert_eq!(u32::from_bytes(&bytes), &[7]);
}
