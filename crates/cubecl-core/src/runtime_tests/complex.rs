use crate::{self as cubecl};
use alloc::vec;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_complex_add<C: Complex>(output: &mut Array<C>, lhs: &Array<C>, rhs: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
    }
}

pub fn test_complex_add_cf32<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    let lhs = vec![C::new(1.0f32, 2.0f32), C::new(3.0f32, 4.0f32)];
    let rhs = vec![C::new(5.0f32, 6.0f32), C::new(7.0f32, 8.0f32)];
    let expected = vec![C::new(6.0f32, 8.0f32), C::new(10.0f32, 12.0f32)];

    let handle_output = client.empty(2 * core::mem::size_of::<C>());
    let handle_lhs = client.create_from_slice(C::as_bytes(&lhs));
    let handle_rhs = client.create_from_slice(C::as_bytes(&rhs));

    unsafe {
        kernel_complex_add::launch_unchecked::<C, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            ArrayArg::from_raw_parts(handle_lhs, 2),
            ArrayArg::from_raw_parts(handle_rhs, 2),
        )
    };

    let actual = client.read_one_unchecked(handle_output);
    let actual = C::from_bytes(&actual);

    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[1], expected[1]);
}

pub fn test_complex_add_cf64<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f64>;
    let lhs = vec![C::new(1.0f64, 2.0f64), C::new(3.0f64, 4.0f64)];
    let rhs = vec![C::new(5.0f64, 6.0f64), C::new(7.0f64, 8.0f64)];
    let expected = vec![C::new(6.0f64, 8.0f64), C::new(10.0f64, 12.0f64)];

    let handle_output = client.empty(2 * core::mem::size_of::<C>());
    let handle_lhs = client.create_from_slice(C::as_bytes(&lhs));
    let handle_rhs = client.create_from_slice(C::as_bytes(&rhs));

    unsafe {
        kernel_complex_add::launch_unchecked::<C, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            ArrayArg::from_raw_parts(handle_lhs, 2),
            ArrayArg::from_raw_parts(handle_rhs, 2),
        )
    };

    let actual = client.read_one_unchecked(handle_output);
    let actual = C::from_bytes(&actual);

    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[1], expected[1]);
}

#[cube(launch_unchecked)]
pub fn kernel_complex_mul<C: Complex>(output: &mut Array<C>, lhs: &Array<C>, rhs: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS];
    }
}

pub fn test_complex_mul_cf32<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    let lhs = vec![C::new(1.0f32, 2.0f32), C::new(3.0f32, 4.0f32)];
    let rhs = vec![C::new(3.0f32, 4.0f32), C::new(5.0f32, 6.0f32)];
    let expected = vec![C::new(-5.0f32, 10.0f32), C::new(-9.0f32, 38.0f32)];

    let handle_output = client.empty(2 * core::mem::size_of::<C>());
    let handle_lhs = client.create_from_slice(C::as_bytes(&lhs));
    let handle_rhs = client.create_from_slice(C::as_bytes(&rhs));

    unsafe {
        kernel_complex_mul::launch_unchecked::<C, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            ArrayArg::from_raw_parts(handle_lhs, 2),
            ArrayArg::from_raw_parts(handle_rhs, 2),
        )
    };

    let actual = client.read_one_unchecked(handle_output);
    let actual = C::from_bytes(&actual);

    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[1], expected[1]);
}

#[cube(launch_unchecked)]
pub fn kernel_complex_conj<C: Complex>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].conj();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_constant<C: Complex + cubecl::ScalarArgType>(
    output: &mut Array<C>,
    scale: C,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = output[ABSOLUTE_POS] * scale;
    }
}

pub fn test_complex_conj_cf32<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    let input = vec![C::new(1.0f32, 2.0f32), C::new(3.0f32, -4.0f32)];
    let expected = vec![C::new(1.0f32, -2.0f32), C::new(3.0f32, 4.0f32)];

    let handle_output = client.empty(2 * core::mem::size_of::<C>());
    let handle_input = client.create_from_slice(C::as_bytes(&input));

    unsafe {
        kernel_complex_conj::launch_unchecked::<C, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            ArrayArg::from_raw_parts(handle_input, 2),
        )
    };

    let actual = client.read_one_unchecked(handle_output);
    let actual = C::from_bytes(&actual);

    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[1], expected[1]);
}

pub fn test_complex_constant_cf32<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    let input = vec![C::new(1.0f32, 2.0f32), C::new(3.0f32, 4.0f32)];
    let scale = C::new(2.0f32, -1.0f32);
    let expected = vec![input[0] * scale, input[1] * scale];

    let handle_output = client.create_from_slice(C::as_bytes(&input));

    unsafe {
        kernel_complex_constant::launch_unchecked::<C, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            scale,
        )
    };

    let actual = client.read_one_unchecked(handle_output);
    let actual = C::from_bytes(&actual);

    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[1], expected[1]);
}

pub fn test_complex_constant_cf64<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f64>;
    let input = vec![C::new(1.0f64, 2.0f64), C::new(3.0f64, 4.0f64)];
    let scale = C::new(2.0f64, -1.0f64);
    let expected = vec![input[0] * scale, input[1] * scale];

    let handle_output = client.create_from_slice(C::as_bytes(&input));

    unsafe {
        kernel_complex_constant::launch_unchecked::<C, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            scale,
        )
    };

    let actual = client.read_one_unchecked(handle_output);
    let actual = C::from_bytes(&actual);

    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[1], expected[1]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_complex {
    () => {
        use super::*;

        mod complex {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_add_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_add_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_add_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_add_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_mul_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_mul_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_conj_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_conj_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_constant_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_constant_cf32::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_constant_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_constant_cf64::<TestRuntime>(
                    client,
                );
            }
        }
    };
}
