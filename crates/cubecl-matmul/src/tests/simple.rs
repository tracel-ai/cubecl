use std::fmt::Display;

use cubecl_core::{CubeElement, Runtime, prelude::Float};

use crate::kernels::naive;
use cubecl_std::tensor::TensorHandle;

use super::test_utils::{MatmulTestCase, Sample, assert_equals_approx};

pub fn test_small<R: Runtime, F: Float + CubeElement + Display + Sample>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
    };

    test_simple::<R, F>(case, device);
}

pub fn test_large<R: Runtime, F: Float + CubeElement + Display + Sample>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
    };

    test_simple::<R, F>(case, device);
}

pub fn test_with_check_bounds<R: Runtime, F: Float + CubeElement + Display + Sample>(
    device: &R::Device,
) {
    let case = MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
    };

    test_simple::<R, F>(case, device);
}

pub fn test_with_batches<R: Runtime, F: Float + CubeElement + Display + Sample>(
    device: &R::Device,
) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
    };

    test_simple::<R, F>(case, device);
}

fn test_simple<R: Runtime, F: Float + CubeElement + Display + Sample>(
    case: MatmulTestCase,
    device: &R::Device,
) {
    let client = R::client(device);
    let lhs = case.random_lhs::<R, F>(&client);
    let rhs = case.random_rhs::<R, F>(&client);

    let expected = case.matmul_cpu::<R, F>(&lhs, &rhs, &client);

    let out: TensorHandle<R, F> = case.empty_out(&client);
    naive::launch::<R, F>(&client, lhs, rhs, &out.as_ref()).unwrap();

    if let Err(e) = assert_equals_approx::<R, F>(
        &client,
        out.handle,
        &out.shape,
        &out.strides,
        &expected,
        10e-4,
    ) {
        panic!("{}", e);
    }
}
