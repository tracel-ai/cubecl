use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

use super::test_utils::assert_equals_approx;

#[cube(launch_unchecked)]
fn matmul_instruction_launch<M: MatmulInstruction<I, O>, I: Numeric, O: Numeric>(
    lhs_slice: Array<I>,
    rhs_slice: Array<I>,
    mut out_slice: Array<O>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
    let mut lhs = M::init_lhs(layouts.0);
    let mut rhs = M::init_rhs(layouts.1);
    let mut out = M::init_output();

    M::fill_lhs(lhs_slice.as_slice(), &mut lhs);
    M::fill_rhs(rhs_slice.as_slice(), &mut rhs);

    M::execute(&lhs, &rhs, &mut out);
    M::read_output(&out, out_slice.as_slice_mut());
}

/// Exported test
pub fn test_matmul_instruction<MI, I, O, R>(device: &R::Device)
where
    I: Numeric + CubeElement,
    O: Numeric,
    MI: MatmulInstruction<I, O>,
    R: Runtime,
{
    let client = R::client(device);
    let lhs_size = (MI::M * MI::K) as usize;
    let rhs_size = (MI::K * MI::N) as usize;
    let out_size = (MI::M * MI::N) as usize;

    let lhs_data = vec![1.; lhs_size];
    let rhs_data = vec![1.; rhs_size];

    let lhs = client.create(I::as_bytes(&I::from_values(&lhs_data)));
    let rhs = client.create(I::as_bytes(&I::from_values(&rhs_data)));
    let out = client.empty(out_size);

    let cube_dim = CubeDim::new(32, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        matmul_instruction_launch::launch_unchecked::<MI, I, O, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(&lhs, lhs_size, 1),
            ArrayArg::from_raw_parts(&rhs, rhs_size, 1),
            ArrayArg::from_raw_parts(&out, out_size, 1),
            (MatrixLayout::Row, MatrixLayout::Row),
        );
    }

    let expected = vec![16.; out_size];
    if let Err(e) = assert_equals_approx::<I, R>(&client, out, &expected, 10e-3) {
        panic!("{}", e);
    }
}
