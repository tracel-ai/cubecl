use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_switch_simple(output: &mut Array<f32>, case: u32) {
    if UNIT_POS == 0 {
        match case {
            0 => {
                output[0] = 1.0;
            }
            1 => {
                output[0] = 3.0;
            }
            _ => {
                output[0] = 5.0;
            }
        }
    }
}

#[cube(launch)]
pub fn kernel_switch_value_expr(output: &mut Array<f32>, case: u32) {
    if UNIT_POS == 0 {
        let value = match case {
            0 => 1.0f32,
            1 => 3.0f32,
            _ => 5.0f32,
        };
        output[0] = value;
    }
}

#[cube(launch)]
pub fn kernel_switch_or_arm(output: &mut Array<f32>, case: u32) {
    if UNIT_POS == 0 {
        let value = match case {
            0 => 1.0f32,
            1 | 2 => 3.0f32,
            _ => 5.0f32,
        };
        output[0] = value;
    }
}

#[cube(launch)]
pub fn kernel_select(output: &mut Array<f32>, cond: u32) {
    if UNIT_POS == 0 {
        output[0] = select(cond == 1, 3.0, 5.0);
    }
}

pub fn test_switch_statement<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0, 1.0]));

    let vectorization = 2;

    kernel_switch_simple::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts(&handle, 2, vectorization) },
        ScalarArg::new(0),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 1.0);
}

pub fn test_switch_used_as_value<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0, 1.0]));

    let vectorization = 2;

    kernel_switch_value_expr::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts(&handle, 2, vectorization) },
        ScalarArg::new(1),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 3.0);
}

pub fn test_switch_default<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0, 1.0]));

    let vectorization = 2;

    kernel_switch_value_expr::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts(&handle, 2, vectorization) },
        ScalarArg::new(5),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

pub fn test_switch_or_branch<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0, 1.0]));

    let vectorization = 2;

    kernel_switch_or_arm::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts(&handle, 2, vectorization) },
        ScalarArg::new(2),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 3.0);
}

pub fn test_select<R: Runtime>(client: ComputeClient<R::Server, R::Channel>, cond: bool) {
    let handle = client.create(f32::as_bytes(&[0.0]));

    let vectorization = 1;

    let cond_u32 = if cond { 1 } else { 0 };

    kernel_select::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts(&handle, 1, vectorization) },
        ScalarArg::new(cond_u32),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    if cond {
        assert_eq!(actual[0], 3.0);
    } else {
        assert_eq!(actual[0], 5.0);
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
            cubecl_core::runtime_tests::branch::test_switch_statement::<TestRuntime>(client);
        }

        #[test]
        fn test_switch_used_as_value() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_used_as_value::<TestRuntime>(client);
        }

        #[test]
        fn test_switch_default() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_default::<TestRuntime>(client);
        }

        #[test]
        fn test_switch_or_branch() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_or_branch::<TestRuntime>(client);
        }

        #[test]
        fn test_select_true() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_select::<TestRuntime>(client, true);
        }

        #[test]
        fn test_select_false() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_select::<TestRuntime>(client, false);
        }
    };
}
