use crate::{self as cubecl, as_bytes};

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_switch_simple<F: Float>(output: &mut Array<F>, case: u32) {
    match case {
        0 => {
            output[0] = F::new(1.0);
        }
        1 => {
            output[0] = F::new(3.0);
        }
        _ => {
            output[0] = F::new(5.0);
        }
    }
}

#[cube(launch)]
pub fn kernel_switch_value_expr<F: Float>(output: &mut Array<F>, case: u32) {
    if UNIT_POS == 0 {
        let value = match case {
            0 => F::new(1.0f32),
            1 => F::new(3.0f32),
            _ => F::new(5.0f32),
        };
        output[0] = value;
    }
}

#[cube(launch)]
pub fn kernel_switch_or_arm<F: Float>(output: &mut Array<F>, case: u32) {
    if UNIT_POS == 0 {
        let value = match case {
            0 => F::new(1.0f32),
            1 | 2 => F::new(3.0f32),
            _ => F::new(5.0f32),
        };
        output[0] = value;
    }
}

#[cube(launch)]
pub fn kernel_select<F: Float>(output: &mut Array<F>, cond: u32) {
    if UNIT_POS == 0 {
        output[0] = select(cond == 1, F::new(3.0), F::new(5.0));
    }
}

pub fn test_switch_statement<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0, 1.0]).expect("Alloc failed");

    let vectorization = 1;

    unsafe {
        kernel_switch_simple::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::default(),
            ArrayArg::from_raw_parts::<F>(&handle, 2, vectorization),
            ScalarArg::new(0),
        );
    }

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1.0));
}

pub fn test_switch_used_as_value<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0, 1.0]).expect("Alloc failed");

    let vectorization = 2;

    kernel_switch_value_expr::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, vectorization) },
        ScalarArg::new(1),
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(3.0));
}

pub fn test_switch_default<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0, 1.0]).expect("Alloc failed");

    let vectorization = 2;

    kernel_switch_value_expr::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, vectorization) },
        ScalarArg::new(5),
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_switch_or_branch<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0, 1.0]).expect("Alloc failed");

    let vectorization = 2;

    kernel_switch_or_arm::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, vectorization) },
        ScalarArg::new(2),
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(3.0));
}

pub fn test_select<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    cond: bool,
) {
    let handle = client.create(as_bytes![F: 0.0]).expect("Alloc failed");

    let vectorization = 1;

    let cond_u32 = if cond { 1 } else { 0 };

    kernel_select::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 1, vectorization) },
        ScalarArg::new(cond_u32),
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    if cond {
        assert_eq!(actual[0], F::new(3.0));
    } else {
        assert_eq!(actual[0], F::new(5.0));
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_branch {
    () => {
        use super::*;

        #[test]
        fn test_switch_statement() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_statement::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_switch_used_as_value() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_used_as_value::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_switch_default() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_default::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_switch_or_branch() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_or_branch::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_select_true() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_select::<TestRuntime, FloatType>(client, true);
        }

        #[test]
        fn test_select_false() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_select::<TestRuntime, FloatType>(
                client, false,
            );
        }
    };
}
