use std::fmt::Display;

use cubecl_core::{CubeElement, Runtime, prelude::Float};

use crate::{
    MatmulInputHandle,
    components::MatmulElems,
    kernels::naive,
    tests::{
        naive::utils::MatmulTestCase,
        test_utils::{Sample, assert_equals_approx},
    },
    tune_key::MatmulElemType,
};
use cubecl_std::tensor::TensorHandle;

pub fn test_small<R: Runtime, F: Float + CubeElement + Display + Sample>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
    };

    test_simple::<R, F>(case, device);
}

pub fn test_odd<R: Runtime, F: Float + CubeElement + Display + Sample>(device: &R::Device) {
    let case = MatmulTestCase {
        m: 1,
        k: 101,
        n: 255,
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

    let out: TensorHandle<R> = case.empty_out::<R, F>(&client);
    let dtypes = MatmulElems {
        lhs_global: MatmulElemType::new(F::as_type_native_unchecked(), false),
        rhs_global: MatmulElemType::new(F::as_type_native_unchecked(), false),
        acc_global: MatmulElemType::new(F::as_type_native_unchecked(), false),
        lhs_stage: MatmulElemType::new(F::as_type_native_unchecked(), false),
        rhs_stage: MatmulElemType::new(F::as_type_native_unchecked(), false),
        acc_stage: MatmulElemType::new(F::as_type_native_unchecked(), false),
        lhs_register: MatmulElemType::new(F::as_type_native_unchecked(), false),
        rhs_register: MatmulElemType::new(F::as_type_native_unchecked(), false),
        acc_register: MatmulElemType::new(F::as_type_native_unchecked(), false),
    };
    naive::launch(
        &client,
        MatmulInputHandle::Normal(lhs),
        MatmulInputHandle::Normal(rhs),
        &out.as_ref(),
        dtypes,
    )
    .unwrap();

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
