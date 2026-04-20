use crate::{self as cubecl};
use alloc::{format, vec};
use core::fmt::{Debug, Display};
use cubecl::prelude::*;
use cubecl_runtime::server::ServerError;

fn assert_exact_eq<R: Runtime, E: CubeElement + Debug + PartialEq>(
    client: &ComputeClient<R>,
    output: cubecl_runtime::server::Handle,
    expected: &[E],
) {
    let actual = client.read_one_unchecked(output);
    let actual = E::from_bytes(&actual);

    assert_eq!(actual, expected);
}

fn assert_real_approx_eq<R: Runtime, F: num_traits::Float + CubeElement + Display>(
    client: &ComputeClient<R>,
    output: cubecl_runtime::server::Handle,
    expected: &[F],
    epsilon: F,
) {
    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= epsilon
                || (actual.is_nan() && expected.is_nan())
                || (actual.is_infinite()
                    && expected.is_infinite()
                    && actual.is_sign_positive() == expected.is_sign_positive()),
            "Values differ more than epsilon: actual={}, expected={}, difference={}, epsilon={}, index={}",
            actual,
            expected,
            (*actual - *expected).abs(),
            epsilon,
            index
        );
    }
}

fn assert_complex_approx_eq<R: Runtime, F: num_traits::Float + CubeElement + Display>(
    client: &ComputeClient<R>,
    output: cubecl_runtime::server::Handle,
    expected: &[num_complex::Complex<F>],
    epsilon: F,
) where
    num_complex::Complex<F>: CubeElement,
{
    let actual = client.read_one_unchecked(output);
    let actual = <num_complex::Complex<F> as CubeElement>::from_bytes(&actual);

    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let real_matches = (actual.re - expected.re).abs() <= epsilon
            || (actual.re.is_nan() && expected.re.is_nan())
            || (actual.re.is_infinite()
                && expected.re.is_infinite()
                && actual.re.is_sign_positive() == expected.re.is_sign_positive());
        let imag_matches = (actual.im - expected.im).abs() <= epsilon
            || (actual.im.is_nan() && expected.im.is_nan())
            || (actual.im.is_infinite()
                && expected.im.is_infinite()
                && actual.im.is_sign_positive() == expected.im.is_sign_positive());

        assert!(
            real_matches && imag_matches,
            "Complex values differ more than epsilon: actual={:?}, expected={:?}, epsilon={}, index={}",
            actual,
            expected,
            epsilon,
            index
        );
    }
}

fn complex_abs_value<T: num_traits::Float>(value: num_complex::Complex<T>) -> T {
    value.re.hypot(value.im)
}

fn complex_exp_value<T: num_traits::Float>(
    value: num_complex::Complex<T>,
) -> num_complex::Complex<T> {
    let magnitude = value.re.exp();
    num_complex::Complex::new(magnitude * value.im.cos(), magnitude * value.im.sin())
}

fn complex_ln_value<T: num_traits::Float>(
    value: num_complex::Complex<T>,
) -> num_complex::Complex<T> {
    num_complex::Complex::new(complex_abs_value(value).ln(), value.im.atan2(value.re))
}

fn complex_powc_value<T: num_traits::Float>(
    value: num_complex::Complex<T>,
    exp: num_complex::Complex<T>,
) -> num_complex::Complex<T> {
    if exp.re.is_zero() && exp.im.is_zero() {
        return num_complex::Complex::new(T::one(), T::zero());
    }

    complex_exp_value(exp * complex_ln_value(value))
}

fn assert_complex_validation_error(result: Result<(), ServerError>, expected_fragment: &str) {
    match result {
        Err(ServerError::ServerUnhealthy { errors, .. }) => {
            assert!(
                errors
                    .iter()
                    .any(|error| format!("{error:?}").contains(expected_fragment)),
                "Expected validation error containing `{expected_fragment}`, got: {errors:?}",
            );
        }
        other => panic!("Expected validation error, got {other:?}"),
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_add<C: ComplexCore>(output: &mut Array<C>, lhs: &Array<C>, rhs: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_sub<C: ComplexCore>(output: &mut Array<C>, lhs: &Array<C>, rhs: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] - rhs[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_mul<C: ComplexCore>(output: &mut Array<C>, lhs: &Array<C>, rhs: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_div<C: ComplexCore>(output: &mut Array<C>, lhs: &Array<C>, rhs: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] / rhs[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_neg<C: ComplexCore>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = -input[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_conj<C: ComplexCore>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].conj();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_real_cf32(output: &mut Array<f32>, input: &Array<num_complex::Complex<f32>>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_real_cf64(output: &mut Array<f64>, input: &Array<num_complex::Complex<f64>>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_imag_cf32(output: &mut Array<f32>, input: &Array<num_complex::Complex<f32>>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].imag_val();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_imag_cf64(output: &mut Array<f64>, input: &Array<num_complex::Complex<f64>>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].imag_val();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_eq<C: ComplexCompare>(
    output: &mut Array<u8>,
    lhs: &Array<C>,
    rhs: &Array<C>,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = if lhs[ABSOLUTE_POS] == rhs[ABSOLUTE_POS] {
            1u8
        } else {
            0u8
        };
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_ne<C: ComplexCompare>(
    output: &mut Array<u8>,
    lhs: &Array<C>,
    rhs: &Array<C>,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = if lhs[ABSOLUTE_POS] != rhs[ABSOLUTE_POS] {
            1u8
        } else {
            0u8
        };
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_constant<C: ComplexCore + cubecl::ScalarArgType>(
    output: &mut Array<C>,
    scale: C,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = output[ABSOLUTE_POS] * scale;
    }
}

macro_rules! test_complex_binary_eq_op {
    (
        $test_name:ident,
        $kernel:ident,
        $ty:ty,
        lhs: [$($lhs:expr),+ $(,)?],
        rhs: [$($rhs:expr),+ $(,)?],
        expect: |$lhs_var:ident, $rhs_var:ident| $expected:expr
    ) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            type C = $ty;
            let lhs = vec![$($lhs),+];
            let rhs = vec![$($rhs),+];
            let expected = lhs
                .iter()
                .copied()
                .zip(rhs.iter().copied())
                .map(|($lhs_var, $rhs_var)| $expected)
                .collect::<vec::Vec<_>>();

            let handle_output = client.empty(lhs.len() * core::mem::size_of::<C>());
            let handle_lhs = client.create_from_slice(C::as_bytes(&lhs));
            let handle_rhs = client.create_from_slice(C::as_bytes(&rhs));

            unsafe {
                $kernel::launch_unchecked::<C, R>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(lhs.len() as u32),
                    ArrayArg::from_raw_parts(handle_output.clone(), lhs.len()),
                    ArrayArg::from_raw_parts(handle_lhs, lhs.len()),
                    ArrayArg::from_raw_parts(handle_rhs, rhs.len()),
                )
            };

            assert_exact_eq(&client, handle_output, &expected);
        }
    };
}

macro_rules! test_complex_unary_eq_op {
    (
        $test_name:ident,
        $kernel:ident,
        $ty:ty,
        input: [$($value:expr),+ $(,)?],
        expect: |$value_var:ident| $expected:expr
    ) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            type C = $ty;
            let input = vec![$($value),+];
            let expected = input
                .iter()
                .copied()
                .map(|$value_var| $expected)
                .collect::<vec::Vec<_>>();

            let handle_output = client.empty(input.len() * core::mem::size_of::<C>());
            let handle_input = client.create_from_slice(C::as_bytes(&input));

            unsafe {
                $kernel::launch_unchecked::<C, R>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(input.len() as u32),
                    ArrayArg::from_raw_parts(handle_output.clone(), input.len()),
                    ArrayArg::from_raw_parts(handle_input, input.len()),
                )
            };

            assert_exact_eq(&client, handle_output, &expected);
        }
    };
}

macro_rules! test_complex_scalar_eq_op {
    (
        $test_name:ident,
        $kernel:ident,
        $ty:ty,
        input: [$($value:expr),+ $(,)?],
        scalar: $scalar:expr,
        expect: |$value_var:ident, $scale_var:ident| $expected:expr
    ) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            type C = $ty;
            let input = vec![$($value),+];
            let scale = $scalar;
            let expected = input
                .iter()
                .copied()
                .map(|$value_var| {
                    let $scale_var = scale;
                    $expected
                })
                .collect::<vec::Vec<_>>();

            let handle_output = client.create_from_slice(C::as_bytes(&input));

            unsafe {
                $kernel::launch_unchecked::<C, R>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(input.len() as u32),
                    ArrayArg::from_raw_parts(handle_output.clone(), input.len()),
                    scale,
                )
            };

            assert_exact_eq(&client, handle_output, &expected);
        }
    };
}

macro_rules! test_complex_unary_scalar_eq_op {
    (
        $test_name:ident,
        $kernel:ident,
        $input_ty:ty,
        $output_ty:ty,
        input: [$($value:expr),+ $(,)?],
        expect: |$value_var:ident| $expected:expr
    ) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            type C = $input_ty;
            type O = $output_ty;
            let input = vec![$($value),+];
            let expected = input
                .iter()
                .copied()
                .map(|$value_var| $expected)
                .collect::<vec::Vec<_>>();

            let handle_output = client.empty(input.len() * core::mem::size_of::<O>());
            let handle_input = client.create_from_slice(C::as_bytes(&input));

            unsafe {
                $kernel::launch_unchecked::<R>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(input.len() as u32),
                    ArrayArg::from_raw_parts(handle_output.clone(), input.len()),
                    ArrayArg::from_raw_parts(handle_input, input.len()),
                )
            };

            assert_exact_eq(&client, handle_output, &expected);
        }
    };
}

macro_rules! test_complex_binary_bool_eq_op {
    (
        $test_name:ident,
        $kernel:ident,
        $ty:ty,
        lhs: [$($lhs:expr),+ $(,)?],
        rhs: [$($rhs:expr),+ $(,)?],
        expect: |$lhs_var:ident, $rhs_var:ident| $expected:expr
    ) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            type C = $ty;
            let lhs = vec![$($lhs),+];
            let rhs = vec![$($rhs),+];
            let expected = lhs
                .iter()
                .copied()
                .zip(rhs.iter().copied())
                .map(|($lhs_var, $rhs_var)| $expected)
                .collect::<vec::Vec<_>>();

            let handle_output = client.empty(lhs.len() * core::mem::size_of::<u8>());
            let handle_lhs = client.create_from_slice(C::as_bytes(&lhs));
            let handle_rhs = client.create_from_slice(C::as_bytes(&rhs));

            unsafe {
                $kernel::launch_unchecked::<C, R>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(lhs.len() as u32),
                    ArrayArg::from_raw_parts(handle_output.clone(), lhs.len()),
                    ArrayArg::from_raw_parts(handle_lhs, lhs.len()),
                    ArrayArg::from_raw_parts(handle_rhs, rhs.len()),
                )
            };

            assert_exact_eq(&client, handle_output, &expected);
        }
    };
}

test_complex_binary_eq_op!(
    test_complex_add_cf32,
    kernel_complex_add,
    num_complex::Complex<f32>,
    lhs: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, 4.0f32),
    ],
    rhs: [
        num_complex::Complex::new(5.0f32, 6.0f32),
        num_complex::Complex::new(7.0f32, 8.0f32),
    ],
    expect: |lhs, rhs| lhs + rhs
);
test_complex_binary_eq_op!(
    test_complex_add_cf64,
    kernel_complex_add,
    num_complex::Complex<f64>,
    lhs: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, 4.0f64),
    ],
    rhs: [
        num_complex::Complex::new(5.0f64, 6.0f64),
        num_complex::Complex::new(7.0f64, 8.0f64),
    ],
    expect: |lhs, rhs| lhs + rhs
);
test_complex_binary_eq_op!(
    test_complex_sub_cf32,
    kernel_complex_sub,
    num_complex::Complex<f32>,
    lhs: [
        num_complex::Complex::new(4.0f32, 3.0f32),
        num_complex::Complex::new(1.0f32, -2.0f32),
    ],
    rhs: [
        num_complex::Complex::new(1.0f32, 1.0f32),
        num_complex::Complex::new(0.5f32, 2.0f32),
    ],
    expect: |lhs, rhs| lhs - rhs
);
test_complex_binary_eq_op!(
    test_complex_sub_cf64,
    kernel_complex_sub,
    num_complex::Complex<f64>,
    lhs: [
        num_complex::Complex::new(4.0f64, 3.0f64),
        num_complex::Complex::new(1.0f64, -2.0f64),
    ],
    rhs: [
        num_complex::Complex::new(1.0f64, 1.0f64),
        num_complex::Complex::new(0.5f64, 2.0f64),
    ],
    expect: |lhs, rhs| lhs - rhs
);
test_complex_binary_eq_op!(
    test_complex_mul_cf32,
    kernel_complex_mul,
    num_complex::Complex<f32>,
    lhs: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, 4.0f32),
    ],
    rhs: [
        num_complex::Complex::new(3.0f32, 4.0f32),
        num_complex::Complex::new(5.0f32, 6.0f32),
    ],
    expect: |lhs, rhs| lhs * rhs
);
test_complex_binary_eq_op!(
    test_complex_mul_cf64,
    kernel_complex_mul,
    num_complex::Complex<f64>,
    lhs: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, 4.0f64),
    ],
    rhs: [
        num_complex::Complex::new(3.0f64, 4.0f64),
        num_complex::Complex::new(5.0f64, 6.0f64),
    ],
    expect: |lhs, rhs| lhs * rhs
);
test_complex_binary_eq_op!(
    test_complex_div_cf32,
    kernel_complex_div,
    num_complex::Complex<f32>,
    lhs: [
        num_complex::Complex::new(4.0f32, 2.0f32),
        num_complex::Complex::new(-3.0f32, 1.5f32),
    ],
    rhs: [
        num_complex::Complex::new(2.0f32, 0.0f32),
        num_complex::Complex::new(0.5f32, 0.0f32),
    ],
    expect: |lhs, rhs| lhs / rhs
);
test_complex_binary_eq_op!(
    test_complex_div_cf64,
    kernel_complex_div,
    num_complex::Complex<f64>,
    lhs: [
        num_complex::Complex::new(4.0f64, 2.0f64),
        num_complex::Complex::new(-3.0f64, 1.5f64),
    ],
    rhs: [
        num_complex::Complex::new(2.0f64, 0.0f64),
        num_complex::Complex::new(0.5f64, 0.0f64),
    ],
    expect: |lhs, rhs| lhs / rhs
);
test_complex_unary_eq_op!(
    test_complex_neg_cf32,
    kernel_complex_neg,
    num_complex::Complex<f32>,
    input: [
        num_complex::Complex::new(1.0f32, -2.0f32),
        num_complex::Complex::new(-3.0f32, 4.0f32),
    ],
    expect: |value| -value
);
test_complex_unary_eq_op!(
    test_complex_neg_cf64,
    kernel_complex_neg,
    num_complex::Complex<f64>,
    input: [
        num_complex::Complex::new(1.0f64, -2.0f64),
        num_complex::Complex::new(-3.0f64, 4.0f64),
    ],
    expect: |value| -value
);
test_complex_unary_eq_op!(
    test_complex_conj_cf32,
    kernel_complex_conj,
    num_complex::Complex<f32>,
    input: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, -4.0f32),
    ],
    expect: |value| num_complex::Complex::new(value.re, -value.im)
);
test_complex_unary_eq_op!(
    test_complex_conj_cf64,
    kernel_complex_conj,
    num_complex::Complex<f64>,
    input: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, -4.0f64),
    ],
    expect: |value| num_complex::Complex::new(value.re, -value.im)
);
test_complex_unary_scalar_eq_op!(
    test_complex_real_cf32,
    kernel_complex_real_cf32,
    num_complex::Complex<f32>,
    f32,
    input: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(-3.5f32, -4.0f32),
    ],
    expect: |value| value.re
);
test_complex_unary_scalar_eq_op!(
    test_complex_real_cf64,
    kernel_complex_real_cf64,
    num_complex::Complex<f64>,
    f64,
    input: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(-3.5f64, -4.0f64),
    ],
    expect: |value| value.re
);
test_complex_unary_scalar_eq_op!(
    test_complex_imag_cf32,
    kernel_complex_imag_cf32,
    num_complex::Complex<f32>,
    f32,
    input: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(-3.5f32, -4.0f32),
    ],
    expect: |value| value.im
);
test_complex_unary_scalar_eq_op!(
    test_complex_imag_cf64,
    kernel_complex_imag_cf64,
    num_complex::Complex<f64>,
    f64,
    input: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(-3.5f64, -4.0f64),
    ],
    expect: |value| value.im
);
test_complex_binary_bool_eq_op!(
    test_complex_eq_cf32,
    kernel_complex_eq,
    num_complex::Complex<f32>,
    lhs: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, 4.0f32),
    ],
    rhs: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, -4.0f32),
    ],
    expect: |lhs, rhs| if lhs == rhs { 1u8 } else { 0u8 }
);
test_complex_binary_bool_eq_op!(
    test_complex_eq_cf64,
    kernel_complex_eq,
    num_complex::Complex<f64>,
    lhs: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, 4.0f64),
    ],
    rhs: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, -4.0f64),
    ],
    expect: |lhs, rhs| if lhs == rhs { 1u8 } else { 0u8 }
);
test_complex_binary_bool_eq_op!(
    test_complex_ne_cf32,
    kernel_complex_ne,
    num_complex::Complex<f32>,
    lhs: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, 4.0f32),
    ],
    rhs: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, -4.0f32),
    ],
    expect: |lhs, rhs| if lhs != rhs { 1u8 } else { 0u8 }
);
test_complex_binary_bool_eq_op!(
    test_complex_ne_cf64,
    kernel_complex_ne,
    num_complex::Complex<f64>,
    lhs: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, 4.0f64),
    ],
    rhs: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, -4.0f64),
    ],
    expect: |lhs, rhs| if lhs != rhs { 1u8 } else { 0u8 }
);
test_complex_scalar_eq_op!(
    test_complex_constant_cf32,
    kernel_complex_constant,
    num_complex::Complex<f32>,
    input: [
        num_complex::Complex::new(1.0f32, 2.0f32),
        num_complex::Complex::new(3.0f32, 4.0f32),
    ],
    scalar: num_complex::Complex::new(2.0f32, -1.0f32),
    expect: |value, scale| value * scale
);
test_complex_scalar_eq_op!(
    test_complex_constant_cf64,
    kernel_complex_constant,
    num_complex::Complex<f64>,
    input: [
        num_complex::Complex::new(1.0f64, 2.0f64),
        num_complex::Complex::new(3.0f64, 4.0f64),
    ],
    scalar: num_complex::Complex::new(2.0f64, -1.0f64),
    expect: |value, scale| value * scale
);

#[cube(launch_unchecked)]
pub fn kernel_complex_abs_cf32(output: &mut Array<f32>, input: &Array<num_complex::Complex<f32>>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].abs();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_abs_cf64(output: &mut Array<f64>, input: &Array<num_complex::Complex<f64>>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].abs();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_exp<C: ComplexMath + Exp>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].exp();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_log<C: ComplexMath + Log>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].ln();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_sin<C: ComplexMath + Sin>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].sin();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_cos<C: ComplexMath + Cos>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].cos();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_sqrt<C: ComplexMath + Sqrt>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].sqrt();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_tanh<C: ComplexMath + Tanh>(output: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS].tanh();
    }
}

#[cube(launch_unchecked)]
pub fn kernel_complex_powf<C: ComplexMath + Powf>(
    output: &mut Array<C>,
    lhs: &Array<C>,
    rhs: &Array<C>,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = <C as Powf>::powf(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

pub fn test_complex_abs_cf32<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    let input = vec![C::new(3.0f32, 4.0f32), C::new(5.0f32, -12.0f32)];
    let expected = vec![complex_abs_value(input[0]), complex_abs_value(input[1])];

    let handle_output = client.empty(2 * core::mem::size_of::<f32>());
    let handle_input = client.create_from_slice(C::as_bytes(&input));

    unsafe {
        kernel_complex_abs_cf32::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            ArrayArg::from_raw_parts(handle_input, 2),
        )
    };

    assert_real_approx_eq::<R, f32>(&client, handle_output, &expected, 1.0e-5f32);
}

pub fn test_complex_abs_cf64<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f64>;
    let input = vec![C::new(3.0f64, 4.0f64), C::new(5.0f64, -12.0f64)];
    let expected = vec![complex_abs_value(input[0]), complex_abs_value(input[1])];

    let handle_output = client.empty(2 * core::mem::size_of::<f64>());
    let handle_input = client.create_from_slice(C::as_bytes(&input));

    unsafe {
        kernel_complex_abs_cf64::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(handle_output.clone(), 2),
            ArrayArg::from_raw_parts(handle_input, 2),
        )
    };

    assert_real_approx_eq::<R, f64>(&client, handle_output, &expected, 1.0e-12f64);
}

macro_rules! test_complex_unary_op {
    ($test_name:ident, $kernel:ident, $method:ident, $ty:ty, $epsilon:expr, [$($value:expr),+ $(,)?]) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            type C = $ty;
            let input = vec![$($value),+];
            let expected = input.iter().copied().map(|value| value.$method()).collect::<vec::Vec<_>>();

            let handle_output = client.empty(input.len() * core::mem::size_of::<C>());
            let handle_input = client.create_from_slice(C::as_bytes(&input));

            unsafe {
                $kernel::launch_unchecked::<C, R>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(input.len() as u32),
                    ArrayArg::from_raw_parts(handle_output.clone(), input.len()),
                    ArrayArg::from_raw_parts(handle_input, input.len()),
                )
            };

            assert_complex_approx_eq::<R, _>(&client, handle_output, &expected, $epsilon);
        }
    };
}

macro_rules! test_complex_powf_op {
    ($test_name:ident, $ty:ty, $epsilon:expr, lhs: [$($lhs:expr),+ $(,)?], rhs: [$($rhs:expr),+ $(,)?]) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            type C = $ty;
            let lhs = vec![$($lhs),+];
            let rhs = vec![$($rhs),+];
            let expected = lhs
                .iter()
                .copied()
                .zip(rhs.iter().copied())
                .map(|(lhs, rhs)| complex_powc_value(lhs, rhs))
                .collect::<vec::Vec<_>>();

            let handle_output = client.empty(lhs.len() * core::mem::size_of::<C>());
            let handle_lhs = client.create_from_slice(C::as_bytes(&lhs));
            let handle_rhs = client.create_from_slice(C::as_bytes(&rhs));

            unsafe {
                kernel_complex_powf::launch_unchecked::<C, R>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(lhs.len() as u32),
                    ArrayArg::from_raw_parts(handle_output.clone(), lhs.len()),
                    ArrayArg::from_raw_parts(handle_lhs, lhs.len()),
                    ArrayArg::from_raw_parts(handle_rhs, rhs.len()),
                )
            };

            assert_complex_approx_eq::<R, _>(&client, handle_output, &expected, $epsilon);
        }
    };
}

test_complex_unary_op!(
    test_complex_exp_cf32,
    kernel_complex_exp,
    exp,
    num_complex::Complex<f32>,
    1.0e-4f32,
    [
        num_complex::Complex::new(0.5f32, -0.75f32),
        num_complex::Complex::new(-1.25f32, 0.25f32),
    ]
);
test_complex_unary_op!(
    test_complex_exp_cf64,
    kernel_complex_exp,
    exp,
    num_complex::Complex<f64>,
    1.0e-12f64,
    [
        num_complex::Complex::new(0.5f64, -0.75f64),
        num_complex::Complex::new(-1.25f64, 0.25f64),
    ]
);
test_complex_unary_op!(
    test_complex_log_cf32,
    kernel_complex_log,
    ln,
    num_complex::Complex<f32>,
    1.0e-4f32,
    [
        num_complex::Complex::new(0.5f32, -0.75f32),
        num_complex::Complex::new(-1.25f32, 0.25f32),
    ]
);
test_complex_unary_op!(
    test_complex_log_cf64,
    kernel_complex_log,
    ln,
    num_complex::Complex<f64>,
    1.0e-12f64,
    [
        num_complex::Complex::new(0.5f64, -0.75f64),
        num_complex::Complex::new(-1.25f64, 0.25f64),
    ]
);
test_complex_unary_op!(
    test_complex_sin_cf32,
    kernel_complex_sin,
    sin,
    num_complex::Complex<f32>,
    1.0e-4f32,
    [
        num_complex::Complex::new(0.5f32, -0.75f32),
        num_complex::Complex::new(-1.25f32, 0.25f32),
    ]
);
test_complex_unary_op!(
    test_complex_sin_cf64,
    kernel_complex_sin,
    sin,
    num_complex::Complex<f64>,
    1.0e-12f64,
    [
        num_complex::Complex::new(0.5f64, -0.75f64),
        num_complex::Complex::new(-1.25f64, 0.25f64),
    ]
);
test_complex_unary_op!(
    test_complex_cos_cf32,
    kernel_complex_cos,
    cos,
    num_complex::Complex<f32>,
    1.0e-4f32,
    [
        num_complex::Complex::new(0.5f32, -0.75f32),
        num_complex::Complex::new(-1.25f32, 0.25f32),
    ]
);
test_complex_unary_op!(
    test_complex_cos_cf64,
    kernel_complex_cos,
    cos,
    num_complex::Complex<f64>,
    1.0e-12f64,
    [
        num_complex::Complex::new(0.5f64, -0.75f64),
        num_complex::Complex::new(-1.25f64, 0.25f64),
    ]
);
test_complex_unary_op!(
    test_complex_sqrt_cf32,
    kernel_complex_sqrt,
    sqrt,
    num_complex::Complex<f32>,
    1.0e-4f32,
    [
        num_complex::Complex::new(0.5f32, -0.75f32),
        num_complex::Complex::new(-1.25f32, 0.25f32),
    ]
);
test_complex_unary_op!(
    test_complex_sqrt_cf64,
    kernel_complex_sqrt,
    sqrt,
    num_complex::Complex<f64>,
    1.0e-12f64,
    [
        num_complex::Complex::new(0.5f64, -0.75f64),
        num_complex::Complex::new(-1.25f64, 0.25f64),
    ]
);
test_complex_unary_op!(
    test_complex_tanh_cf32,
    kernel_complex_tanh,
    tanh,
    num_complex::Complex<f32>,
    1.0e-4f32,
    [
        num_complex::Complex::new(0.5f32, -0.75f32),
        num_complex::Complex::new(-1.25f32, 0.25f32),
    ]
);
test_complex_unary_op!(
    test_complex_tanh_cf64,
    kernel_complex_tanh,
    tanh,
    num_complex::Complex<f64>,
    1.0e-12f64,
    [
        num_complex::Complex::new(0.5f64, -0.75f64),
        num_complex::Complex::new(-1.25f64, 0.25f64),
    ]
);
test_complex_powf_op!(
    test_complex_powf_cf32,
    num_complex::Complex<f32>,
    1.0e-4f32,
    lhs: [
        num_complex::Complex::new(0.5f32, -0.75f32),
        num_complex::Complex::new(-1.25f32, 0.25f32),
    ],
    rhs: [
        num_complex::Complex::new(0.25f32, 0.5f32),
        num_complex::Complex::new(-0.75f32, 0.125f32),
    ]
);
test_complex_powf_op!(
    test_complex_powf_cf64,
    num_complex::Complex<f64>,
    1.0e-12f64,
    lhs: [
        num_complex::Complex::new(0.5f64, -0.75f64),
        num_complex::Complex::new(-1.25f64, 0.25f64),
    ],
    rhs: [
        num_complex::Complex::new(0.25f64, 0.5f64),
        num_complex::Complex::new(-0.75f64, 0.125f64),
    ]
);

#[cube(launch)]
pub fn kernel_complex_validation_core<C: ComplexCore>(
    output: &mut Array<C>,
    lhs: &Array<C>,
    rhs: &Array<C>,
) {
    if UNIT_POS == 0 {
        output[0] = lhs[0] + rhs[0];
    }
}

#[cube(launch)]
pub fn kernel_complex_validation_compare<C: ComplexCompare>(
    output: &mut Array<u8>,
    lhs: &Array<C>,
    rhs: &Array<C>,
) {
    if UNIT_POS == 0 {
        output[0] = if lhs[0] == rhs[0] { 1u8 } else { 0u8 };
    }
}

#[cube(launch)]
pub fn kernel_complex_validation_math<C: ComplexMath>(output: &mut Array<C>, input: &Array<C>) {
    if UNIT_POS == 0 {
        output[0] = input[0].exp();
    }
}

pub fn test_complex_validation_core<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    if C::supported_complex_uses(&client).contains(cubecl::ir::features::ComplexUsage::Core) {
        return;
    }

    let output = client.empty(core::mem::size_of::<C>());
    let lhs = client.create_from_slice(C::as_bytes(&[C::new(1.0, 2.0)]));
    let rhs = client.create_from_slice(C::as_bytes(&[C::new(3.0, 4.0)]));

    kernel_complex_validation_core::launch::<C, R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(output, 1) },
        unsafe { ArrayArg::from_raw_parts(lhs, 1) },
        unsafe { ArrayArg::from_raw_parts(rhs, 1) },
    );

    assert_complex_validation_error(client.flush(), "Complex operation");
}

pub fn test_complex_validation_compare<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    if C::supported_complex_uses(&client).contains(cubecl::ir::features::ComplexUsage::Compare) {
        return;
    }

    let output = client.empty(core::mem::size_of::<u8>());
    let lhs = client.create_from_slice(C::as_bytes(&[C::new(1.0, 2.0)]));
    let rhs = client.create_from_slice(C::as_bytes(&[C::new(1.0, 2.0)]));

    kernel_complex_validation_compare::launch::<C, R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(output, 1) },
        unsafe { ArrayArg::from_raw_parts(lhs, 1) },
        unsafe { ArrayArg::from_raw_parts(rhs, 1) },
    );

    assert_complex_validation_error(client.flush(), "Complex operation");
}

pub fn test_complex_validation_math<R: Runtime>(client: ComputeClient<R>) {
    type C = num_complex::Complex<f32>;
    if C::supported_complex_uses(&client).contains(cubecl::ir::features::ComplexUsage::Math) {
        return;
    }

    let output = client.empty(core::mem::size_of::<C>());
    let input = client.create_from_slice(C::as_bytes(&[C::new(1.0, 2.0)]));

    kernel_complex_validation_math::launch::<C, R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(output, 1) },
        unsafe { ArrayArg::from_raw_parts(input, 1) },
    );

    assert_complex_validation_error(client.flush(), "Complex operation");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_complex_core {
    () => {
        use super::*;

        mod complex_core {
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
            fn test_complex_sub_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_sub_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_sub_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_sub_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_mul_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_mul_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_mul_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_mul_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_div_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_div_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_div_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_div_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_neg_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_neg_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_neg_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_neg_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_conj_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_conj_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_conj_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_conj_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_real_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_real_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_real_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_real_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_imag_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_imag_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_imag_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_imag_cf64::<TestRuntime>(client);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_complex_compare {
    () => {
        use super::*;

        mod complex_compare {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_eq_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_eq_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_eq_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_eq_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_ne_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_ne_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_ne_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_ne_cf64::<TestRuntime>(client);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_complex_math {
    () => {
        use super::*;

        mod complex_math {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_abs_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_abs_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_abs_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_abs_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_exp_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_exp_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_exp_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_exp_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_log_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_log_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_log_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_log_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_sin_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_sin_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_sin_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_sin_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_cos_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_cos_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_cos_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_cos_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_sqrt_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_sqrt_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_sqrt_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_sqrt_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_tanh_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_tanh_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_tanh_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_tanh_cf64::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_powf_cf32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_powf_cf32::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_powf_cf64() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_powf_cf64::<TestRuntime>(client);
            }
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_complex_validation {
    () => {
        use super::*;

        mod complex_validation {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_validation_core() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_validation_core::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_validation_compare() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_validation_compare::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_complex_validation_math() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::complex::test_complex_validation_math::<TestRuntime>(
                    client,
                );
            }
        }
    };
}
