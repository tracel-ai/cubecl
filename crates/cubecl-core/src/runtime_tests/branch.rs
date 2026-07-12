use alloc::{vec, vec::Vec};

use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

// Regression: a value-form if/else with literal-constant arms must honor the runtime
// condition (it used to select at comptime and always return the else value). One kernel
// per type, plus a checker asserting the two conditions produce different output.
macro_rules! lit_value_form {
    ($kernel:ident, $check:ident, $ty:ty, $a:expr, $b:expr) => {
        #[cube(launch)]
        pub fn $kernel(output: &mut [$ty], cond: u32) {
            if UNIT_POS == 0 {
                let flag = cond != 0u32;
                output[0] = if flag { $a } else { $b };
            }
        }

        fn $check<R: Runtime>(client: &ComputeClient<R>) {
            let run = |cond: u32| -> Vec<u8> {
                let handle = client.empty(core::mem::size_of::<$ty>());
                $kernel::launch::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
                    cond,
                );
                client.read_one_unchecked(handle).to_vec()
            };
            assert_ne!(
                run(1),
                run(0),
                "value-form if/else miscompiled for {}",
                stringify!($ty)
            );
        }
    };
}

lit_value_form!(kernel_lit_i8, check_i8, i8, 30i8, 50i8);
lit_value_form!(kernel_lit_i16, check_i16, i16, 30i16, 50i16);
lit_value_form!(kernel_lit_i32, check_i32, i32, 30i32, 50i32);
lit_value_form!(kernel_lit_i64, check_i64, i64, 30i64, 50i64);
lit_value_form!(kernel_lit_u8, check_u8, u8, 30u8, 50u8);
lit_value_form!(kernel_lit_u16, check_u16, u16, 30u16, 50u16);
lit_value_form!(kernel_lit_u32, check_u32, u32, 30u32, 50u32);
lit_value_form!(kernel_lit_u64, check_u64, u64, 30u64, 50u64);
lit_value_form!(kernel_lit_f32, check_f32, f32, 30.0f32, 50.0f32);
lit_value_form!(kernel_lit_f64, check_f64, f64, 30.0f64, 50.0f64);

pub fn test_value_form_literals<R: Runtime>(client: ComputeClient<R>) {
    check_i8(&client);
    check_i16(&client);
    check_i32(&client);
    check_i64(&client);
    check_u8(&client);
    check_u16(&client);
    check_u32(&client);
    check_u64(&client);
    check_f32(&client);
    check_f64(&client);
}

// Subset using only types every backend supports (wgpu rejects i8/i16/u8/u16/f64).
pub fn test_value_form_literals_portable<R: Runtime>(client: ComputeClient<R>) {
    check_f32(&client);
    check_i32(&client);
    check_u32(&client);
}

// Regression: `match` used as a value with literal arms had the same comptime-select
// bug as if/else, and should act the same way now.
// One kernel per type, with a single-pattern arm, an or-pattern arm, and a default; the
// checker asserts each selector picks the right arm's value.
macro_rules! lit_match_value {
    ($kernel:ident, $check:ident, $ty:ty, $a:expr, $b:expr, $c:expr) => {
        #[cube(launch)]
        pub fn $kernel(output: &mut [$ty], sel: u32) {
            if UNIT_POS == 0 {
                output[0] = match sel {
                    0 => $a,
                    1 | 2 => $b,
                    _ => $c,
                };
            }
        }

        fn $check<R: Runtime>(client: &ComputeClient<R>) {
            let run = |sel: u32| -> Vec<u8> {
                let handle = client.empty(core::mem::size_of::<$ty>());
                $kernel::launch::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
                    sel,
                );
                client.read_one_unchecked(handle).to_vec()
            };
            let ty = stringify!($ty);
            assert_eq!(run(0), <$ty>::as_bytes(&[$a]).to_vec(), "{ty}: arm 0");
            assert_eq!(
                run(1),
                <$ty>::as_bytes(&[$b]).to_vec(),
                "{ty}: or-arm sel=1"
            );
            assert_eq!(
                run(2),
                <$ty>::as_bytes(&[$b]).to_vec(),
                "{ty}: or-arm sel=2"
            );
            assert_eq!(run(3), <$ty>::as_bytes(&[$c]).to_vec(), "{ty}: default");
        }
    };
}

lit_match_value!(kernel_match_i8, match_check_i8, i8, 10i8, 20i8, 30i8);
lit_match_value!(kernel_match_i16, match_check_i16, i16, 10i16, 20i16, 30i16);
lit_match_value!(kernel_match_i32, match_check_i32, i32, 10i32, 20i32, 30i32);
lit_match_value!(kernel_match_i64, match_check_i64, i64, 10i64, 20i64, 30i64);
lit_match_value!(kernel_match_u8, match_check_u8, u8, 10u8, 20u8, 30u8);
lit_match_value!(kernel_match_u16, match_check_u16, u16, 10u16, 20u16, 30u16);
lit_match_value!(kernel_match_u32, match_check_u32, u32, 10u32, 20u32, 30u32);
lit_match_value!(kernel_match_u64, match_check_u64, u64, 10u64, 20u64, 30u64);
lit_match_value!(
    kernel_match_f32,
    match_check_f32,
    f32,
    10.0f32,
    20.0f32,
    30.0f32
);
lit_match_value!(
    kernel_match_f64,
    match_check_f64,
    f64,
    10.0f64,
    20.0f64,
    30.0f64
);

pub fn test_match_value_literals<R: Runtime>(client: ComputeClient<R>) {
    match_check_i8(&client);
    match_check_i16(&client);
    match_check_i32(&client);
    match_check_i64(&client);
    match_check_u8(&client);
    match_check_u16(&client);
    match_check_u32(&client);
    match_check_u64(&client);
    match_check_f32(&client);
    match_check_f64(&client);
}

// Subset using only types every backend supports (wgpu rejects i8/i16/u8/u16/f64).
pub fn test_match_value_literals_portable<R: Runtime>(client: ComputeClient<R>) {
    match_check_f32(&client);
    match_check_i32(&client);
    match_check_u32(&client);
}

#[cube(launch_unchecked)]
pub fn kernel_switch_simple<F: Float>(output: &mut [F], case: u32) {
    match case {
        0 => {
            output[0] = F::new(1f32);
        }
        1 => {
            output[0] = F::new(3f32);
        }
        _ => {
            output[0] = F::new(5f32);
        }
    }
}

#[cube(launch)]
pub fn kernel_switch_value_expr<F: Float>(output: &mut [F], case: u32) {
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
pub fn kernel_switch_or_arm<F: Float>(output: &mut [F], case: u32) {
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
pub fn kernel_switch_const<F: Float>(output: &mut [F], case: u32) {
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
pub fn kernel_select<F: Float>(output: &mut [F], cond: u32) {
    if UNIT_POS == 0 {
        output[0] = select(cond == 1, F::new(3f32), F::new(5f32));
    }
}

#[cube(launch)]
pub fn kernel_for_loop_with_break<F: Float>(output: &mut [F]) {
    let max_iterations = comptime!(20_i32);
    for i in 0..max_iterations {
        if i > 3 {
            break;
        }
        output[i as usize] = F::new(1f32);
    }
}

#[cube(launch)]
pub fn kernel_for_loop_with_return<F: Float>(output: &mut [F]) {
    let max_iterations = comptime!(20_i32);
    for i in 0..max_iterations {
        if i > 3 {
            terminate!();
        }
        output[i as usize] = F::new(1f32);
    }
    output[0] = F::new(5f32);
}

pub fn test_switch_const<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    kernel_switch_const::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
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
            BufferArg::from_raw_parts(handle.clone(), 2),
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
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
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
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
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
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
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
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
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

pub fn test_for_loop_with_break<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let zeros = vec![F::new(0.0); 20];
    let handle = client.create_from_slice(F::as_bytes(&zeros));

    kernel_for_loop_with_break::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 20) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    let expected: Vec<F> = (0..20)
        .map(|i| if i < 4 { F::new(1.0) } else { F::new(0.0) })
        .collect();
    assert_eq!(actual, expected.as_slice());
}

pub fn test_for_loop_with_return<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let zeros = vec![F::new(0.0); 20];
    let handle = client.create_from_slice(F::as_bytes(&zeros));

    kernel_for_loop_with_return::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 20) },
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
        fn test_value_form_literals_portable() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_value_form_literals_portable::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_match_value_literals_portable() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_match_value_literals_portable::<TestRuntime>(
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

        #[$crate::runtime_tests::test_log::test]
        fn test_for_loop_with_return() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::branch::test_for_loop_with_return::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}

/// All-types literal value-form if/else test, for backends supporting every scalar type
/// (CUDA, CPU). Kept out of `testgen_branch` since wgpu rejects i8/i16/u8/u16/f64.
#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_value_form_literals {
    () => {
        mod value_form_literals {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_value_form_literals() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::branch::test_value_form_literals::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_match_value_literals() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::branch::test_match_value_literals::<TestRuntime>(
                    client,
                );
            }
        }
    };
}
