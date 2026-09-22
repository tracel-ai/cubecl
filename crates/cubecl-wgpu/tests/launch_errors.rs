//! A launch the device refuses reaches the caller as an error on the read.
//!
//! wgpu reports a shader module or pipeline it cannot create as an uncaptured
//! device error, which panics the thread that polls the device and leaves the
//! read returning whatever the buffer held. The launch has to catch it instead,
//! so the buffers it never wrote carry the failure, the read of one names it,
//! and a caller such as autotune can tell a refusal from a fault.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;

/// Reads the first element of every input into the output, so every input is a
/// binding of its own.
#[cube(launch_unchecked)]
fn gather_first(inputs: &Sequence<Box<[u32]>>, output: &mut [u32]) {
    let mut total = 0u32;
    #[unroll]
    for i in 0..inputs.len() {
        total += inputs[i][0];
    }
    output[0] = total;
}

/// More storage buffers than the device binds, which no check before wgpu
/// catches: the kernel compiles, and the device refuses its module.
#[test]
fn a_kernel_the_device_refuses_fails_the_read() {
    let client = <WgpuRuntime>::client(&Default::default());
    let over_limit = client.properties().hardware.max_bindings as usize + 8;

    let inputs: Vec<_> = (0..over_limit)
        .map(|_| client.create_from_slice(u32::as_bytes(&[1])))
        .collect();
    let output = client.create_from_slice(u32::as_bytes(&[0]));

    unsafe {
        gather_first::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            inputs
                .iter()
                .map(|input| BufferArg::from_raw_parts(input.clone(), 1))
                .collect(),
            BufferArg::from_raw_parts(output.clone(), 1),
        );
    }

    let err = client
        .read_one(output)
        .expect_err("the launch never wrote the output, so the read fails on it");
    assert!(
        err.is_refusal(),
        "expected the refused kernel as the root cause, got: {err}"
    );
}
