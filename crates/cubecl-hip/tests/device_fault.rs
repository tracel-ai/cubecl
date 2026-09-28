//! Forces a real HIP driver fault, then checks how the client reports it.
//!
//! - `out_of_bounds` writes far past the end of its buffer with no bounds
//!   check, which makes the driver raise an illegal address (status 700).
//! - `fill` is a harmless kernel, used to show that work launched after the
//!   fault fails too.
//! - The test launches the fault, then checks that the read, a `sync`, and a
//!   later launch each return an error that `is_device_poisoned`, and none panics.
//!
//! An illegal address poisons the whole HIP context for the rest of the
//! process, so the file holds a single test: a second one in the same test
//! binary would start on a dead device.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;

/// Writes a gigabyte past the end of its buffer: no bounds check, so the
/// access reaches unmapped memory and the device raises an illegal address.
#[cube(launch_unchecked)]
fn out_of_bounds(out: &mut [u32]) {
    out[ABSOLUTE_POS + 268_435_456] = 1u32;
}

#[cube(launch)]
fn fill(out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = 7u32;
    }
}

#[test]
fn a_device_fault_surfaces_at_the_sync_point() {
    let client = HipRuntime::client(&Default::default());
    let size = core::mem::size_of::<u32>();

    let faulted = client.empty(size);
    unsafe {
        out_of_bounds::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(faulted.clone(), 1),
        )
    };

    // The launch was accepted; the fault only exists once the kernel runs,
    // and the read's fence is the first thing to hear about it.
    let read = client
        .read_one(faulted)
        .expect_err("a faulted kernel's output must not read clean");
    assert!(
        read.is_device_poisoned(),
        "an illegal address poisons the device, got: {read}"
    );

    // Every later sync point reports the poisoning — as an error, not as a panic on
    // the server's thread.
    let sync = cubecl_environment::future::block_on(client.sync())
        .expect_err("a sync on a poisoned device must fail");
    assert!(sync.is_device_poisoned(), "got: {sync}");

    let later = client.empty(size);
    fill::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(later.clone(), 1) },
    );
    let later = client
        .read_one(later)
        .expect_err("nothing written after the fault can be trusted");
    assert!(later.is_device_poisoned(), "got: {later}");
}
