use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_different_rank(lhs: &Tensor<f32>, rhs: &Tensor<f32>, output: &mut Tensor<f32>) {
    output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
}

pub fn test_kernel_different_rank_first_biggest<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let shape_lhs = vec![2, 2, 2];
    let shape_rhs = vec![8];
    let shape_out = vec![2, 4];

    let strides_lhs = vec![8, 4, 1];
    let strides_rhs = vec![1];
    let strides_out = vec![4, 1];

    test_kernel_different_rank::<R>(
        client,
        (shape_lhs, shape_rhs, shape_out),
        (strides_lhs, strides_rhs, strides_out),
    );
}

pub fn test_kernel_different_rank_last_biggest<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let shape_lhs = vec![2, 4];
    let shape_rhs = vec![8];
    let shape_out = vec![2, 2, 2];

    let strides_lhs = vec![4, 1];
    let strides_rhs = vec![1];
    let strides_out = vec![8, 4, 1];

    test_kernel_different_rank::<R>(
        client,
        (shape_lhs, shape_rhs, shape_out),
        (strides_lhs, strides_rhs, strides_out),
    );
}

fn test_kernel_different_rank<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    (shape_lhs, shape_rhs, shape_out): (Vec<usize>, Vec<usize>, Vec<usize>),
    (strides_lhs, strides_rhs, strides_out): (Vec<usize>, Vec<usize>, Vec<usize>),
) {
    let vectorisation = 2;

    let handle_lhs = client.create(f32::as_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]));
    let handle_rhs = client.create(f32::as_bytes(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]));
    let handle_out = client.create(f32::as_bytes(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));

    let lhs =
        unsafe { TensorArg::from_raw_parts(&handle_lhs, &strides_lhs, &shape_lhs, vectorisation) };
    let rhs =
        unsafe { TensorArg::from_raw_parts(&handle_rhs, &strides_rhs, &shape_rhs, vectorisation) };
    let out =
        unsafe { TensorArg::from_raw_parts(&handle_out, &strides_out, &shape_out, vectorisation) };

    kernel_different_rank::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        lhs,
        rhs,
        out,
    );

    let actual = client.read(handle_out.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual, &[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_different_rank {
    () => {
        use super::*;

        #[test]
        fn test_kernel_different_rank_first_biggest() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::different_rank::test_kernel_different_rank_first_biggest::<
                TestRuntime,
            >(client);
        }

        #[test]
        fn test_kernel_different_rank_last_biggest() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::different_rank::test_kernel_different_rank_last_biggest::<
                TestRuntime,
            >(client);
        }
    };
}
