#![allow(clippy::approx_constant)]

use core::f32;

use std::{fmt::Display, sync::LazyLock};

use crate::{self as cubecl, as_type};

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use enumset::EnumSet;

#[track_caller]
pub(crate) fn assert_equals_approx<
    R: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: &ComputeClient<R>,
    output: Handle,
    expected: &[F],
    epsilon: f32,
) {
    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual[0..expected.len()]
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        // account for lower precision at higher values
        let allowed_error = F::new((epsilon * e.to_f32().unwrap().abs()).max(epsilon));
        assert!(
            (*a - *e).abs() < allowed_error
                || (a.is_nan() && e.is_nan())
                || (a.is_infinite()
                    && e.is_infinite()
                    && a.is_sign_positive() == e.is_sign_positive()),
            "Values differ more than epsilon: actual={}, expected={}, difference={}, epsilon={}
index: {}
actual: {:?}
expected: {:?}",
            a,
            e,
            (*a - *e).abs(),
            epsilon,
            i,
            actual,
            expected
        );
    }
}

// Needs lazy because const trait fns aren't stable
static FAST_MATH: LazyLock<EnumSet<FastMath>> =
    LazyLock::new(|| FastMath::all().difference(FastMath::NotNaN.into()));

macro_rules! test_binary_impl {
    (
        $test_name:ident,
        $float_type:ident,
        $binary_func:expr,
        [$({
            input_vectorization: $input_vectorization:expr,
            out_vectorization: $out_vectorization:expr,
            lhs: $lhs:expr,
            rhs: $rhs:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R>) {
            #[cube(launch_unchecked, fast_math = *FAST_MATH)]
            fn test_function<$float_type: Float>(lhs: &Array<$float_type>, rhs: &Array<$float_type>, output: &mut Array<$float_type>) {
                if ABSOLUTE_POS < rhs.len() {
                    output[ABSOLUTE_POS] = $binary_func(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
                }
            }

            $(
            {
                let lhs = $lhs;
                let rhs = $rhs;
                let output_handle = client.empty($expected.len() * core::mem::size_of::<$float_type>());
                let lhs_handle = client.create_from_slice($float_type::as_bytes(lhs));
                let rhs_handle = client.create_from_slice($float_type::as_bytes(rhs));

                unsafe {
                    test_function::launch_unchecked::<$float_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((lhs.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$float_type>(&lhs_handle, lhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$float_type>(&rhs_handle, rhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$float_type>(&output_handle, $expected.len(), $out_vectorization),
                    ).unwrap()
                };

                assert_equals_approx::<R, F>(&client, output_handle, $expected, 0.001);
            }
            )*
        }
    };
}

test_binary_impl!(
    test_dot,
    F,
    F::dot,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: -1.0, -71.61, 3.36, 77.01]
        },
        {
            input_vectorization: 2,
            out_vectorization: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: -72.61, 80.37]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: 7.76]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1, -1., 23.1, -1.4, 5.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1, 1., -3.1, -2.4, 15.1],
            expected: as_type![F: 7.76, 7.76]
        }

    ]
);

test_binary_impl!(
    test_powf,
    F,
    F::powf,
    [
        {
            input_vectorization: 2,
            out_vectorization: 2,
            lhs: as_type![F: 2., -3., 2., 81.],
            rhs: as_type![F: 3., 2., -1., 0.5],
            expected: as_type![F: 8., 9., 0.5, 9.]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            lhs: as_type![F: 2., -3., 2., 81.],
            rhs: as_type![F: 3., 2., -1., 0.5],
            expected: as_type![F: 8., 9., 0.5, 9.]
        }
    ]
);

test_binary_impl!(
    test_atan2,
    F,
    F::atan2,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            lhs: as_type![F: 0., 1., -1., 1., -1.],
            rhs: as_type![F: 1., 0., 0., 1., -1.],
            expected: as_type![F: 0., 1.570_796_4, -1.570_796_4, 0.785_398_2, -2.356_194_5]
        },
        {
            input_vectorization: 2,
            out_vectorization: 2,
            lhs: as_type![F: 0., 1., -1., 1.],
            rhs: as_type![F: 1., 0., 0., 1.],
            expected: as_type![F: 0., 1.570_796_4, -1.570_796_4, 0.785_398_2]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            lhs: as_type![F: 0., 1., -1., 1.],
            rhs: as_type![F: 1., 0., 0., 1.],
            expected: as_type![F: 0., 1.570_796_4, -1.570_796_4, 0.785_398_2]
        }
    ]
);

test_binary_impl!(
    test_hypot,
    F,
    F::hypot,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            lhs: as_type![F: 3., 0., 5., 0.],
            rhs: as_type![F: 4., 5., 0., 0.],
            expected: as_type![F: 5., 5., 5., 0.]
        },
        {
            input_vectorization: 2,
            out_vectorization: 2,
            lhs: as_type![F: 3., 0., 5., 8.],
            rhs: as_type![F: 4., 5., 0., 15.],
            expected: as_type![F: 5., 5., 5., 17.]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            lhs: as_type![F: -3., 0., -5., -8.],
            rhs: as_type![F: -4., -5., 0., 15.],
            expected: as_type![F: 5., 5., 5., 17.]
        }
    ]
);

test_binary_impl!(
    test_rhypot,
    F,
    F::rhypot,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            lhs: as_type![F: 3., 0., 5., 0.],
            rhs: as_type![F: 4., 5., 0., 0.],
            expected: &[F::new(0.2), F::new(0.2), F::new(0.2), F::INFINITY]
        },
        {
            input_vectorization: 2,
            out_vectorization: 2,
            lhs: as_type![F: 3., 0., 5., 0.3],
            rhs: as_type![F: 4., 5., 0., 0.4],
            expected: as_type![F: 0.2, 0.2, 0.2, 2.]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            lhs: as_type![F: 0., 0., -5., -0.3],
            rhs: as_type![F: -1., -5., 0., -0.4],
            expected: as_type![F: 1., 0.2, 0.2, 2.]
        }
    ]
);

#[cube(launch_unchecked)]
fn test_powi_kernel<F: Float>(
    lhs: &Array<Line<F>>,
    rhs: &Array<Line<i32>>,
    output: &mut Array<Line<F>>,
) {
    if ABSOLUTE_POS < rhs.len() {
        output[ABSOLUTE_POS] = Powi::powi(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

macro_rules! test_powi_impl {
    (
        $test_name:ident,
        $float_type:ident,
        [$({
            input_vectorization: $input_vectorization:expr,
            out_vectorization: $out_vectorization:expr,
            lhs: $lhs:expr,
            rhs: $rhs:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R>) {
            $(
            {
                let lhs = $lhs;
                let rhs = $rhs;
                let output_handle = client.empty($expected.len() * core::mem::size_of::<$float_type>());
                let lhs_handle = client.create_from_slice($float_type::as_bytes(lhs));
                let rhs_handle = client.create_from_slice(i32::as_bytes(rhs));

                unsafe {
                    test_powi_kernel::launch_unchecked::<F, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((lhs.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$float_type>(&lhs_handle, lhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<i32>(&rhs_handle, rhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$float_type>(&output_handle, $expected.len(), $out_vectorization),
                    ).unwrap()
                };

                assert_equals_approx::<R, F>(&client, output_handle, $expected, 0.001);
            }
            )*
        }
    };
}

test_powi_impl!(
    test_powi,
    F,
    [
        {
            input_vectorization: 2,
            out_vectorization: 2,
            lhs: as_type![F: 2., -3., 2., 81.],
            rhs: as_type![i32: 3, 2, -1, 1],
            expected: as_type![F: 8., 9., 0.5, 81.]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            lhs: as_type![F: 2., -3., 2., 81.],
            rhs: as_type![i32: 3, 2, -1, 1],
            expected: as_type![F: 8., 9., 0.5, 81.]
        }
    ]
);

#[cube(launch_unchecked)]
fn test_mulhi_kernel(
    lhs: &Array<Line<u32>>,
    rhs: &Array<Line<u32>>,
    output: &mut Array<Line<u32>>,
) {
    if ABSOLUTE_POS < rhs.len() {
        output[ABSOLUTE_POS] = MulHi::mul_hi(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

macro_rules! test_mulhi_impl {
    (
        $test_name:ident,
        [$({
            input_vectorization: $input_vectorization:expr,
            out_vectorization: $out_vectorization:expr,
            lhs: $lhs:expr,
            rhs: $rhs:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            $(
            {
                let lhs = $lhs;
                let rhs = $rhs;
                let output_handle = client.empty($expected.len() * core::mem::size_of::<u32>());
                let lhs_handle = client.create_from_slice(u32::as_bytes(lhs));
                let rhs_handle = client.create_from_slice(u32::as_bytes(rhs));

                unsafe {
                    test_mulhi_kernel::launch_unchecked(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((lhs.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(&lhs_handle, lhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<u32>(&rhs_handle, rhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<u32>(&output_handle, $expected.len(), $out_vectorization),
                    ).unwrap()
                };

                let actual = client.read_one(output_handle);
                let actual = u32::from_bytes(&actual);
                let expected: &[u32] = $expected;

                assert_eq!(actual, expected);
            }
            )*
        }
    };
}

test_mulhi_impl!(
    test_mulhi,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            lhs: &[1, 2, 3, 4],
            rhs: &[5, 6, 7, 8],
            expected: &[0, 0, 0, 0]
        },
        {
            input_vectorization: 1,
            out_vectorization: 1,
            lhs: &[0xFFFFFFFF, 0x80000000, 0x55555555, 0x10000000],
            rhs: &[0x10000, 2, 4, 0x20000000],
            expected: &[0x0000FFFF, 1, 1, 0x2000000]
        },
        {
            input_vectorization: 1,
            out_vectorization: 1,
            lhs: &[0xFFFFFFFF, 0xFFFFFFFF, 0x80000000, 0x10000],
            rhs: &[0xFFFFFFFF, 2, 0x80000000, 0x10000],
            expected: &[0xFFFFFFFEu32, 1, 0x40000000, 1]
        }
    ]
);

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_binary {
    () => {
        mod binary {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::binary::$test_name::<TestRuntime, FloatType>(
                            client,
                        );
                    }
                };
            }

            add_test!(test_dot);
            add_test!(test_powf);
            add_test!(test_hypot);
            add_test!(test_rhypot);
            add_test!(test_powi);
            add_test!(test_atan2);
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_binary_untyped {
    () => {
        mod binary_untyped {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::binary::$test_name::<TestRuntime>(client);
                    }
                };
                ($test_name:ident, $ty:ty) => {
                    #[test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::binary::$test_name::<TestRuntime, $ty>(client);
                    }
                };
            }

            add_test!(test_mulhi);
        }
    };
}
