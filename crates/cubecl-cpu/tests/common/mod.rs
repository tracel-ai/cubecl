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

pub const LANES: usize = 4;

#[cube(launch_unchecked)]
pub fn product_over(input: &[f16], output: &mut [f16]) {
    output[0] = input[0] * input[1] / input[2];
}

#[cube(launch_unchecked)]
pub fn product_over_through_a_let(input: &[f16], output: &mut [f16]) {
    let product = input[0] * input[1];
    output[0] = product / input[2];
}

#[allow(unused_mut)]
#[cube(launch_unchecked)]
pub fn product_over_through_a_let_mut(input: &[f16], output: &mut [f16]) {
    let mut product = input[0] * input[1];
    output[0] = product / input[2];
}

#[cube(launch_unchecked)]
pub fn product_on_a_branch(input: &[f16], output: &mut [f16], take: u32) {
    let mut x = input[0];
    if take == 1 {
        x = input[0] * input[1];
    }
    output[0] = x;
    output[1] = x / input[2];
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
pub fn accumulate_vector<N: Size>(
    input: &[Vector<f16, N>],
    output: &mut [Vector<f16, N>],
    steps: usize,
) {
    let mut total = input[0];
    for _ in 0..steps {
        total += input[1];
    }
    output[0] = total;
}

#[cube(launch_unchecked)]
pub fn accumulate_through_a_copy(input: &[f16], output: &mut [f16], steps: usize) {
    let mut total = input[0];
    for _ in 0..steps {
        let carried = total;
        total = carried + input[1];
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
    over_300(1, |client, input, output| unsafe {
        product_over::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            input,
            output,
        )
    })[0]
}

/// `product_over_300`, with the product bound by `let` before it is divided.
pub fn product_over_300_through_a_let() -> f32 {
    over_300(1, |client, input, output| unsafe {
        product_over_through_a_let::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            input,
            output,
        )
    })[0]
}

/// `product_over_300`, with the product bound by `let mut` before it is divided.
pub fn product_over_300_through_a_let_mut() -> f32 {
    over_300(1, |client, input, output| unsafe {
        product_over_through_a_let_mut::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            input,
            output,
        )
    })[0]
}

/// `product_over_300` through a variable the product is stored into under an `if`, then read
/// twice: once stored as it is and once divided, which is what gets reported.
pub fn product_on_a_branch_over_300() -> f32 {
    over_300(2, |client, input, output| unsafe {
        product_on_a_branch::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            input,
            output,
            1,
        )
    })[1]
}

/// `start` plus `steps` of `step`, accumulated in whatever the mode holds the total in.
pub fn accumulated(start: f32, step: f32, steps: usize) -> f32 {
    let client = client();
    let input = [f16::from_f32(start), f16::from_f32(step)];
    let output = run(&client, &input, 1, |input, output| unsafe {
        accumulate::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 2),
            BufferArg::from_raw_parts(output, 1),
            steps,
        )
    });

    output[0].to_f32()
}

/// `accumulated`, in every lane of a vector of `LANES`.
pub fn accumulated_vector(start: f32, step: f32, steps: usize) -> [f32; LANES] {
    let client = client();
    let mut input = [f16::from_f32(start); 2 * LANES];
    input[LANES..].fill(f16::from_f32(step));
    let output = run(&client, &input, LANES, |input, output| unsafe {
        accumulate_vector::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            LANES,
            BufferArg::from_raw_parts(input, 2 * LANES),
            BufferArg::from_raw_parts(output, LANES),
            steps,
        )
    });

    core::array::from_fn(|lane| output[lane].to_f32())
}

/// `accumulated`, with the total passing through a second local on its way around the loop.
pub fn accumulated_through_a_copy(start: f32, step: f32, steps: usize) -> f32 {
    let client = client();
    let input = [f16::from_f32(start), f16::from_f32(step)];
    let output = run(&client, &input, 1, |input, output| unsafe {
        accumulate_through_a_copy::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 2),
            BufferArg::from_raw_parts(output, 1),
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

/// Three f16 inputs of 300, and `outputs` results read back as f32.
fn over_300(outputs: usize, launch: impl FnOnce(&Client, BufferArg, BufferArg)) -> Vec<f32> {
    let client = client();
    let input = [f16::from_f32(300.0); 3];
    let output = run(&client, &input, outputs, |input, output| unsafe {
        launch(
            &client,
            BufferArg::from_raw_parts(input, 3),
            BufferArg::from_raw_parts(output, outputs),
        )
    });

    output.iter().map(|value| value.to_f32()).collect()
}

fn run(
    client: &Client,
    input: &[f16],
    outputs: usize,
    launch: impl FnOnce(Handle, Handle),
) -> Vec<f16> {
    let input = client.create_from_slice(f16::as_bytes(input));
    let output = client.empty(outputs * size_of::<f16>());
    launch(input, output.clone());
    f16::from_bytes(&client.read_one_unchecked(output)).to_vec()
}
