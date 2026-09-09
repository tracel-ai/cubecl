//! The kernels the f16 evaluation modes are judged on.
//!
//! Each mode is read once, when the runtime builds its device, so each has its own test binary
//! and they share this rather than a process.

#![allow(dead_code)]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;
use half::f16;

#[cube(launch_unchecked)]
pub fn product_over(input: &[f16], output: &mut [f16]) {
    output[0] = input[0] * input[1] / input[2];
}

#[cube(launch_unchecked)]
pub fn accumulate(input: &[f16], output: &mut [f16], steps: usize) {
    let mut total = input[0];
    for _ in 0..steps {
        total += input[1];
    }
    output[0] = total;
}

#[cube(launch_unchecked)]
pub fn barrier_smoke(output: &mut [f32]) {
    let barrier = barrier::Barrier::local();
    barrier.arrive_and_wait();
    if UNIT_POS == 0 {
        output[0] = 1.0;
    }
}

/// `a * b / c` where `a * b` is above the f16 maximum, so the answer says whether the
/// intermediate was held.
pub fn product_over_300() -> f32 {
    let client = client();
    let input = [
        f16::from_f32(300.0),
        f16::from_f32(300.0),
        f16::from_f32(300.0),
    ];
    let output = run(&client, &input, |handles| unsafe {
        product_over::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(handles.0, 3),
            BufferArg::from_raw_parts(handles.1, 1),
        )
    });

    output[0].to_f32()
}

/// `start` plus `steps` of `step`, accumulated in whatever the mode holds the total in.
pub fn accumulated(start: f32, step: f32, steps: usize) -> f32 {
    let client = client();
    let input = [f16::from_f32(start), f16::from_f32(step)];
    let output = run(&client, &input, |handles| unsafe {
        accumulate::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(handles.0, 2),
            BufferArg::from_raw_parts(handles.1, 1),
            steps,
        )
    });

    output[0].to_f32()
}

/// A kernel with no f16 in it at all, which still has to reach the end of the pipeline.
pub fn barrier_reaches_the_store() -> f32 {
    let client = client();
    let output = client.empty(size_of::<f32>());

    unsafe {
        barrier_smoke::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 1),
        )
    }

    f32::from_bytes(&client.read_one_unchecked(output))[0]
}

pub fn client() -> Client {
    CpuRuntime::client(&Default::default())
}

fn run(client: &Client, input: &[f16], launch: impl FnOnce((Handle, Handle))) -> Vec<f16> {
    let input = client.create_from_slice(f16::as_bytes(input));
    let output = client.empty(size_of::<f16>());
    launch((input, output.clone()));
    f16::from_bytes(&client.read_one_unchecked(output)).to_vec()
}
