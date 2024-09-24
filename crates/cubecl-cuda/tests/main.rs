use common::*;
use cubecl_core as cubecl;
use cubecl_core::{prelude::*, CubeCount, CubeDim};
use cubecl_cuda::CudaRuntime;
use pretty_assertions::assert_eq;

mod common;

#[cube(launch_unchecked, create_dummy_kernel)]
pub fn slice_assign_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if UNIT_POS == 0 {
        let slice_1 = output.slice_mut(2, 3);
        slice_1[0] = input[0];
    }
}

#[test]
pub fn slice_assign() {
    let client = client();
    let input = handle(&client);
    let output = handle(&client);

    let kernel = slice_assign_kernel::create_dummy_kernel::<CudaRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        tensor(&input),
        tensor(&output),
    );
    let expected = include_str!("slice_assign.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
pub fn kernel_sum(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = cubecl_core::prelude::subcube_sum(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[test]
pub fn subcube_sum() {
    let client = client();
    let output = handle(&client);

    let kernel = kernel_sum::create_dummy_kernel::<CudaRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::new(4, 1, 1),
        tensor(&output),
    );
    let expected = include_str!("subcube_sum.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
pub fn sequence_for_loop_kernel(output: &mut Array<f32>) {
    if UNIT_POS != 0 {
        return;
    }

    let mut sequence = Sequence::<f32>::new();
    sequence.push(1.0);
    sequence.push(4.0);

    for value in sequence {
        output[0] += value;
    }
}

#[test]
pub fn sequence_for_loop() {
    let client = client();
    let output = handle(&client);

    let kernel = sequence_for_loop_kernel::create_dummy_kernel::<CudaRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        array(&output),
    );
    let expected = include_str!("sequence_for_loop.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
fn execute_unary_kernel<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>, out: &mut Tensor<F>) {
    if ABSOLUTE_POS < out.len() {
        for i in 0..256u32 {
            if i % 2 == 0 {
                out[ABSOLUTE_POS] -= F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            } else {
                out[ABSOLUTE_POS] += F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            }
        }
    }
}

#[test]
pub fn unary_bench() {
    let client = client();
    let lhs = handle(&client);
    let rhs = handle(&client);
    let out = handle(&client);

    let kernel = execute_unary_kernel::create_dummy_kernel::<f32, CudaRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        tensor_vec(&lhs, 4),
        tensor_vec(&rhs, 4),
        tensor_vec(&out, 4),
    );
    let expected = include_str!("unary_bench.cu").replace("\r\n", "\n");
    let expected = expected.trim();
    assert_eq!(compile(kernel), expected);
}

#[cube(launch, create_dummy_kernel)]
fn constant_array_kernel<F: Float>(out: &mut Tensor<F>, #[comptime] data: Vec<u32>) {
    let array = Array::<F>::from_data(data);

    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = array[ABSOLUTE_POS];
    }
}

#[test]
pub fn constant_array() {
    let client = client();
    let out = handle(&client);
    let data: Vec<u32> = vec![3, 5, 1];

    let kernel = constant_array_kernel::create_dummy_kernel::<f32, CudaRuntime>(
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        tensor_vec(&out, 1),
        data,
    );
    let expected = include_str!("constant_array.cu").replace("\r\n", "\n");
    assert_eq!(compile(kernel), expected);
}
