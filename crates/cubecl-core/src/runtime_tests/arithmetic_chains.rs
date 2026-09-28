//! Arithmetic a compiler may carry in a wider type than written: through a binding, a branch, a
//! loop and a vector. The shared tests use operands exact in every float type.

use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_runtime::runtime::Runtime;

pub const LANES: usize = 4;

#[cube(launch_unchecked)]
pub fn kernel_product_over<F: Float>(input: &[F], output: &mut [F]) {
    output[0] = input[0] * input[1] / input[2];
}

#[cube(launch_unchecked)]
pub fn kernel_product_over_through_a_let<F: Float>(input: &[F], output: &mut [F]) {
    let product = input[0] * input[1];
    output[0] = product / input[2];
}

#[allow(unused_mut)]
#[cube(launch_unchecked)]
pub fn kernel_product_over_through_a_let_mut<F: Float>(input: &[F], output: &mut [F]) {
    let mut product = input[0] * input[1];
    output[0] = product / input[2];
}

#[cube(launch_unchecked)]
pub fn kernel_product_on_a_branch<F: Float>(input: &[F], output: &mut [F], take: u32) {
    let mut x = input[0];
    if take == 1 {
        x = input[0] * input[1];
    }
    output[0] = x;
    output[1] = x / input[2];
}

#[cube(launch_unchecked)]
pub fn kernel_accumulate<F: Float>(input: &[F], output: &mut [F], steps: usize) {
    let mut total = input[0];
    for _ in 0..steps {
        total += input[1];
    }
    output[0] = total;
}

#[cube(launch_unchecked)]
pub fn kernel_accumulate_vector<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    steps: usize,
) {
    let mut total = input[0];
    for _ in 0..steps {
        total += input[1];
    }
    output[0] = total;
}

#[cube(launch_unchecked)]
pub fn kernel_accumulate_through_a_copy<F: Float>(input: &[F], output: &mut [F], steps: usize) {
    let mut total = input[0];
    for _ in 0..steps {
        let carried = total;
        total = carried + input[1];
    }
    output[0] = total;
}

pub fn test_products<R: Runtime, F: Float + CubeElement>(client: Client) {
    let operands = [3.0, 4.0, 2.0];
    assert_eq!(product_over::<F>(&client, operands), F::new(6.0));
    assert_eq!(
        product_over_through_a_let::<F>(&client, operands),
        F::new(6.0)
    );
    assert_eq!(
        product_over_through_a_let_mut::<F>(&client, operands),
        F::new(6.0)
    );
    assert_eq!(product_on_a_branch::<F>(&client, operands), F::new(6.0));
}

/// Steps of `+ 1` whose every partial sum is exact in bf16, the narrowest mantissa tested: past
/// 256 a bf16 sum stops moving, and the test would measure rounding instead of the carry.
const ACCUMULATION_STEPS: usize = 256;

pub fn test_accumulations<R: Runtime, F: Float + CubeElement>(client: Client) {
    let steps = ACCUMULATION_STEPS;
    let total = F::new(steps as f32);
    assert_eq!(accumulated::<F>(&client, 0.0, 1.0, steps), total);
    assert_eq!(
        accumulated_through_a_copy::<F>(&client, 0.0, 1.0, steps),
        total
    );
    assert_eq!(
        accumulated_vector::<F>(&client, 0.0, 1.0, steps),
        [total; LANES]
    );
}

/// `a * b / c`, written as one expression.
pub fn product_over<F: Float + CubeElement>(client: &Client, operands: [f32; 3]) -> F {
    run(
        client,
        &operands.map(|x| F::new(x)),
        1,
        |input, output| unsafe {
            kernel_product_over::launch_unchecked::<F>(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                input,
                output,
            )
        },
    )[0]
}

/// `product_over`, with the product bound by `let` before it is divided.
pub fn product_over_through_a_let<F: Float + CubeElement>(
    client: &Client,
    operands: [f32; 3],
) -> F {
    run(
        client,
        &operands.map(|x| F::new(x)),
        1,
        |input, output| unsafe {
            kernel_product_over_through_a_let::launch_unchecked::<F>(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                input,
                output,
            )
        },
    )[0]
}

/// `product_over`, with the product bound by `let mut` before it is divided.
pub fn product_over_through_a_let_mut<F: Float + CubeElement>(
    client: &Client,
    operands: [f32; 3],
) -> F {
    run(
        client,
        &operands.map(|x| F::new(x)),
        1,
        |input, output| unsafe {
            kernel_product_over_through_a_let_mut::launch_unchecked::<F>(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                input,
                output,
            )
        },
    )[0]
}

/// `product_over` through a variable the product is stored into under an `if`, then read twice:
/// once stored as it is and once divided, which is what is returned.
pub fn product_on_a_branch<F: Float + CubeElement>(client: &Client, operands: [f32; 3]) -> F {
    run(
        client,
        &operands.map(|x| F::new(x)),
        2,
        |input, output| unsafe {
            kernel_product_on_a_branch::launch_unchecked::<F>(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                input,
                output,
                1,
            )
        },
    )[1]
}

/// `start` plus `steps` of `step`, carried around a loop.
pub fn accumulated<F: Float + CubeElement>(
    client: &Client,
    start: f32,
    step: f32,
    steps: usize,
) -> F {
    run(
        client,
        &[F::new(start), F::new(step)],
        1,
        |input, output| unsafe {
            kernel_accumulate::launch_unchecked::<F>(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                input,
                output,
                steps,
            )
        },
    )[0]
}

/// `accumulated`, with the total passing through a second local on its way around the loop.
pub fn accumulated_through_a_copy<F: Float + CubeElement>(
    client: &Client,
    start: f32,
    step: f32,
    steps: usize,
) -> F {
    run(
        client,
        &[F::new(start), F::new(step)],
        1,
        |input, output| unsafe {
            kernel_accumulate_through_a_copy::launch_unchecked::<F>(
                client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                input,
                output,
                steps,
            )
        },
    )[0]
}

/// `accumulated`, in every lane of a vector of `LANES`.
pub fn accumulated_vector<F: Float + CubeElement>(
    client: &Client,
    start: f32,
    step: f32,
    steps: usize,
) -> [F; LANES] {
    let mut input = [F::new(start); 2 * LANES];
    input[LANES..].fill(F::new(step));
    let output = run(client, &input, LANES, |input, output| unsafe {
        kernel_accumulate_vector::launch_unchecked::<F>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            LANES,
            input,
            output,
            steps,
        )
    });

    core::array::from_fn(|lane| output[lane])
}

fn run<F: CubeElement>(
    client: &Client,
    input: &[F],
    outputs: usize,
    launch: impl FnOnce(BufferArg, BufferArg),
) -> Vec<F> {
    let input_handle = client.create_from_slice(F::as_bytes(input));
    let output = client.empty(outputs * core::mem::size_of::<F>());
    launch(
        unsafe { BufferArg::from_raw_parts(input_handle, input.len()) },
        unsafe { BufferArg::from_raw_parts(output.clone(), outputs) },
    );
    F::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_arithmetic_chains {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_arithmetic_chain_products() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::arithmetic_chains::test_products::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_arithmetic_chain_accumulations() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::arithmetic_chains::test_accumulations::<
                TestRuntime,
                FloatType,
            >(client);
        }
    };
}
