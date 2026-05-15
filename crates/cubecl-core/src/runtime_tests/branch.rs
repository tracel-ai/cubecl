use alloc::{vec, vec::Vec};

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

const CASE_0: u32 = 0;
const CASE_1: u32 = 1;

#[cube(launch)]
pub fn kernel_switch_const<F: Float>(output: &mut Array<F>, case: u32) {
    if UNIT_POS == 0 {
        let value = match case {
            CASE_0 => F::new(1.0f32),
            CASE_1 => F::new(3.0f32),
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

#[cube(launch)]
pub fn kernel_if_literal_u8(output: &mut Array<u8>, cond: u32) {
    if UNIT_POS == 0 {
        let value = if cond == 1 { 1u8 } else { 0u8 };
        output[0] = value;
    }
}

#[cube(launch)]
pub fn kernel_for_loop_with_break<F: Float>(output: &mut Array<F>) {
    let max_iterations = comptime!(20_i32);
    for i in 0..max_iterations {
        if i > 3 {
            break;
        }
        output[i as usize] = F::new(1.0);
    }
}

pub fn test_switch_const<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    kernel_switch_const::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
        1,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(3.0));
}

pub fn test_switch_statement<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    unsafe {
        kernel_switch_simple::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(handle.clone(), 2),
            0,
        );
    }

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1.0));
}

pub fn test_switch_used_as_value<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    kernel_switch_value_expr::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
        1,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(3.0));
}

pub fn test_switch_default<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    kernel_switch_value_expr::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
        5,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_switch_or_branch<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    kernel_switch_or_arm::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
        2,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(3.0));
}

pub fn test_select<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>, cond: bool) {
    let handle = client.create_from_slice(as_bytes![F: 0.0]);

    let cond_u32 = if cond { 1 } else { 0 };

    kernel_select::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 1) },
        cond_u32,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    if cond {
        assert_eq!(actual[0], F::new(3.0));
    } else {
        assert_eq!(actual[0], F::new(5.0));
    }
}

pub fn test_if_literal_u8<R: Runtime>(client: ComputeClient<R>, cond: bool) {
    let handle = client.create_from_slice(u8::as_bytes(&[9u8]));

    kernel_if_literal_u8::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 1) },
        if cond { 1 } else { 0 },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u8::from_bytes(&actual);

    assert_eq!(actual[0], if cond { 1 } else { 0 });
}

pub fn test_for_loop_with_break<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let zeros = vec![F::new(0.0); 20];
    let handle = client.create_from_slice(F::as_bytes(&zeros));

    kernel_for_loop_with_break::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 20) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    let expected: Vec<F> = (0..20)
        .map(|i| if i < 4 { F::new(1.0) } else { F::new(0.0) })
        .collect();
    assert_eq!(actual, expected.as_slice());
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_branch {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_switch_statement() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_statement::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_switch_used_as_value() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_used_as_value::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_switch_default() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_default::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_switch_or_branch() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_or_branch::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_select_true() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_select::<TestRuntime, FloatType>(client, true);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_select_false() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_select::<TestRuntime, FloatType>(
                client, false,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_if_literal_u8_true() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_if_literal_u8::<TestRuntime>(client, true);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_if_literal_u8_false() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_if_literal_u8::<TestRuntime>(client, false);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_switch_const() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_switch_const::<TestRuntime, FloatType>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_for_loop_with_break() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_for_loop_with_break::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
