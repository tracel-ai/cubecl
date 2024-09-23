use cubecl_core::Runtime;

use crate::matmul::{
    tests::{matmul_test_case::MatmulTestCase, test_utils::assert_equals_approx},
    tiling2d,
};

pub fn test_matmul_tiling2d_one_cube<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
    };

    test_tiling2d::<R>(case, device);
}

pub fn test_matmul_tiling2d_several_cubes<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
    };

    test_tiling2d::<R>(case, device);
}

pub fn test_matmul_tiling2d_with_check_bounds<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
    };

    test_tiling2d::<R>(case, device);
}

pub fn test_matmul_tiling2d_with_batches<R: Runtime>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
    };

    test_tiling2d::<R>(case, device);
}

fn test_tiling2d<R: Runtime>(case: MatmulTestCase, device: &R::Device) {
    let client = R::client(device);
    let lhs = case.random_lhs::<R>(&client);
    let rhs = case.random_rhs::<R>(&client);

    let expected = case.matmul_cpu(&lhs, &rhs, &client);

    let out = tiling2d::launch::<R, f32>(
        &client,
        lhs,
        rhs,
        case.empty_out(&client),
        Default::default(),
    );

    if let Err(e) = assert_equals_approx::<R>(&client, out.handle, &expected, 10e-3) {
        panic!("{}", e);
    }
}
