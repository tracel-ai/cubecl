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
