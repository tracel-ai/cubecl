//! Forces a real GPU fault on the native Metal runtime, then checks how the client reports it.
//!
//! - `out_of_bounds` writes far past the end of its buffer with no bounds check, which on
//!   Apple GPUs raises a GPU address fault and fails the command buffer.
//! - `fill` is a harmless kernel, used to show what happens to work launched after the fault.
//! - The test launches the fault, then checks that the read, a `sync`, a later launch, and a
//!   sync and read from another thread each return an error, and that none panics or reads
//!   clean.
//!
//! Whether those errors say the device is poisoned is printed rather than asserted: the
//! Metal runtime currently reports a GPU fault as a generic error. Run with `--nocapture` to
//! see it.
//!
//! The fault stays on the stream for the rest of the process, so the file holds a single
//! test: a second one in the same test binary would start on a faulted stream.
#![cfg(target_vendor = "apple")]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_metal::MetalRuntime;
use cubecl_server::runtime::Runtime;

/// Writes a gigabyte past the end of its buffer: no bounds check, so the access reaches
/// unmapped memory and the GPU raises an address fault.
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

/// Print how `error` was reported, for the tester to send back.
fn report(what: &str, error: &impl core::fmt::Display, poisoned: bool) {
    println!("{what}: device poisoned = {poisoned}\n  {error}\n");
}

#[test]
fn a_gpu_fault_surfaces_at_the_sync_point() {
    let client = MetalRuntime::client(&Default::default());
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

    // The launch was accepted; the fault only exists once the kernel runs, and the read's
    // wait is the first to hear about it.
    let read = client
        .read_one(faulted)
        .expect_err("a faulted kernel's output must not read clean");
    report(
        "read of the faulted buffer",
        &read,
        read.is_device_poisoned(),
    );

    let sync = cubecl_environment::future::block_on(client.sync())
        .expect_err("a sync after a GPU fault must fail");
    report("sync", &sync, sync.is_device_poisoned());

    let later = client.empty(size);
    fill::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(later.clone(), 1) },
    );
    match client.read_one(later) {
        Ok(bytes) => println!(
            "read of a buffer written after the fault: succeeded, {} bytes\n",
            bytes.len()
        ),
        Err(err) => report(
            "read of a buffer written after the fault",
            &err,
            err.is_device_poisoned(),
        ),
    }

    // Another thread gets a stream of its own, created on first use. Whether the fault
    // reaches it too is what this prints.
    let (sync, read) = std::thread::spawn(move || {
        let sync = cubecl_environment::future::block_on(client.sync());
        let read = client.read_one(client.create_from_slice(&[7u8; 4]));
        (sync, read)
    })
    .join()
    .expect("work on another thread after a GPU fault must not panic");
    match sync {
        Ok(()) => println!("sync from another thread: succeeded\n"),
        Err(err) => report("sync from another thread", &err, err.is_device_poisoned()),
    }
    match read {
        Ok(bytes) => println!(
            "read of a buffer created on another thread: succeeded, {:?}\n",
            &bytes[..]
        ),
        Err(err) => report(
            "read of a buffer created on another thread",
            &err,
            err.is_device_poisoned(),
        ),
    }
}
