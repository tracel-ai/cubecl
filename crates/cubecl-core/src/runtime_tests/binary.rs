use std::fmt::Display;

use crate::{self as cubecl, as_type};

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

#[track_caller]
pub(crate) fn assert_equals_approx<
    R: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[F],
    epsilon: f32,
) {
    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual[0..expected.len()]
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        // account for lower precision at higher values
        let allowed_error = F::new((epsilon * e.to_f32().unwrap()).max(epsilon));
        assert!(
            (*a - *e).abs() < allowed_error || (a.is_nan() && e.is_nan()),
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
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R::Server, R::Channel>) {
            #[cube(launch_unchecked, fast_math = FastMath::all())]
            fn test_function<$float_type: Float>(lhs: &Array<$float_type>, rhs: &Array<$float_type>, output: &mut Array<$float_type>) {
                if ABSOLUTE_POS < rhs.len() {
                    output[ABSOLUTE_POS] = $binary_func(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
                }
            }

            $(
            {
                let lhs = $lhs;
                let rhs = $rhs;
                let output_handle = client.empty($expected.len() * core::mem::size_of::<$float_type>()).expect("Alloc failed");
                let lhs_handle = client.create($float_type::as_bytes(lhs)).expect("Alloc failed");
                let rhs_handle = client.create($float_type::as_bytes(rhs)).expect("Alloc failed");

                unsafe {
                    test_function::launch_unchecked::<$float_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((lhs.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$float_type>(&lhs_handle, lhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$float_type>(&rhs_handle, rhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$float_type>(&output_handle, $expected.len(), $out_vectorization),
                    )
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
        pub fn $test_name<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
            $(
            {
                let lhs = $lhs;
                let rhs = $rhs;
                let output_handle = client.empty($expected.len() * core::mem::size_of::<u32>()).expect("Alloc failed");
                let lhs_handle = client.create(u32::as_bytes(lhs)).expect("Alloc failed");
                let rhs_handle = client.create(u32::as_bytes(rhs)).expect("Alloc failed");

                unsafe {
                    test_mulhi_kernel::launch_unchecked::<R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((lhs.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(&lhs_handle, lhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<u32>(&rhs_handle, rhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<u32>(&output_handle, $expected.len(), $out_vectorization),
                    )
                };

                let actual = client.read_one(output_handle.binding());
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
            }

            add_test!(test_mulhi);
        }
    };
}
