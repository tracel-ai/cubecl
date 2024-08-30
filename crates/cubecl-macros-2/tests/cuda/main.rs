use common::*;
use cubecl_core::{
    new_ir::{element::*, UNIT_POS},
    CubeCount, CubeDim,
};
use cubecl_cuda::CudaRuntime;
use cubecl_macros_2::cube2;
use pretty_assertions::assert_eq;

mod common;

#[cube2(launch_unchecked, create_dummy_kernel)]
pub fn slice_assign_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if UNIT_POS == 0 {
        let slice_1 = &mut output[2..3];
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
    let expected = include_str!("slice_assign.cu");
    assert_eq!(compile(kernel), expected);
}

#[cube2(launch, create_dummy_kernel)]
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
    let expected = include_str!("subcube_sum.cu");
    assert_eq!(compile(kernel), expected);
}

#[cube2(launch, create_dummy_kernel)]
pub fn sequence_for_loop_kernel(output: &mut Array<f32>) {
    if UNIT_POS != 0 {
        return;
    }

    let sequence = Sequence::<f32>::new();
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
    let expected = include_str!("sequence_for_loop.cu");
    assert_eq!(compile(kernel), expected);
}
