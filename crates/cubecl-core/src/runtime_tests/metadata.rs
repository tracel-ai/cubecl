use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_shape_dim_4(lhs: &Tensor<f32>, rhs: &Tensor<f32>, out: &mut Tensor<u32>) {
    if ABSOLUTE_POS >= out.len() {
        return;
    }

    let _ = lhs[0];
    let _ = rhs[0];

    out[0] = lhs.shape(0);
    out[1] = lhs.shape(1);
    out[2] = lhs.shape(2);
    out[3] = lhs.shape(3);
    out[4] = rhs.shape(0);
    out[5] = rhs.shape(1);
    out[6] = rhs.shape(2);
    out[7] = rhs.shape(3);
    out[8] = out.shape(0);
    out[9] = out.shape(1);
    out[10] = out.shape(2);
    out[11] = out.shape(3);
}

pub fn test_shape_dim_4<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client.empty(12 * core::mem::size_of::<u32>());
    let handle2 = client.empty(12 * core::mem::size_of::<u32>());
    let handle3 = client.empty(12 * core::mem::size_of::<u32>());

    unsafe {
        kernel_shape_dim_4::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts(&handle1, &[1, 1, 1, 1], &[2, 3, 4, 5], 1),
            TensorArg::from_raw_parts(&handle2, &[1, 1, 1, 1], &[9, 8, 7, 6], 1),
            TensorArg::from_raw_parts(&handle3, &[1, 1, 1, 1], &[10, 11, 12, 13], 1),
        )
    };

    let actual = client.read(handle3.binding());
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = vec![2, 3, 4, 5, 9, 8, 7, 6, 10, 11, 12, 13];

    assert_eq!(actual, &expect);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_metadata {
    () => {
        use super::*;

        #[test]
        fn test_shape() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::metadata::test_shape_dim_4::<TestRuntime>(client);
        }
    };
}
