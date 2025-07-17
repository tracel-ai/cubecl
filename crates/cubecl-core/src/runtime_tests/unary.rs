use std::fmt::Display;

use crate::{self as cubecl, as_type};

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

pub(crate) fn assert_equals_approx<
    R: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[F],
    epsilon: F,
) {
    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*a - *e).abs() < epsilon || (a.is_nan() && e.is_nan()),
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

macro_rules! test_unary_impl {
    (
        $test_name:ident,
        $float_type:ident,
        $unary_func:expr,
        [$({
            input_vectorization: $input_vectorization:expr,
            out_vectorization: $out_vectorization:expr,
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R::Server, R::Channel>) {
            #[cube(launch_unchecked)]
            fn test_function<$float_type: Float>(input: &Array<$float_type>, output: &mut Array<$float_type>) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = $unary_func(input[ABSOLUTE_POS]);
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$float_type>());
                let input_handle = client.create($float_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$float_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((input.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$float_type>(&input_handle, input.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$float_type>(&output_handle, $expected.len(), $out_vectorization),
                    )
                };

                assert_equals_approx::<R, $float_type>(&client, output_handle, $expected, $float_type::new(0.02));
            }
            )*
        }
    };
}

macro_rules! test_unary_impl_int {
    (
        $test_name:ident,
        $int_type:ident,
        $unary_func:expr,
        [$({
            input_vectorization: $input_vectorization:expr,
            out_vectorization: $out_vectorization:expr,
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $int_type: Int + CubeElement>(client: ComputeClient<R::Server, R::Channel>) {
            #[cube(launch_unchecked)]
            fn test_function<$int_type: Int>(input: &Array<$int_type>, output: &mut Array<$int_type>) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = $unary_func(input[ABSOLUTE_POS]);
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$int_type>());
                let input_handle = client.create($int_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$int_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((input.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$int_type>(&input_handle, input.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$int_type>(&output_handle, $expected.len(), $out_vectorization),
                    )
                };

                let actual = client.read_one(output_handle.binding());
                let actual = $int_type::from_bytes(&actual);

                assert_eq!(actual, $expected);
            }
            )*
        }
    };
}

macro_rules! test_unary_impl_int_fixed {
    (
        $test_name:ident,
        $int_type:ident,
        $out_type:ident,
        $unary_func:expr,
        [$({
            input_vectorization: $input_vectorization:expr,
            out_vectorization: $out_vectorization:expr,
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $int_type: Int + CubeElement>(client: ComputeClient<R::Server, R::Channel>) {
            #[cube(launch_unchecked)]
            fn test_function<$int_type: Int>(input: &Array<$int_type>, output: &mut Array<$out_type>) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = $unary_func(input[ABSOLUTE_POS]);
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$out_type>());
                let input_handle = client.create($int_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$int_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((input.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$int_type>(&input_handle, input.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$out_type>(&output_handle, $expected.len(), $out_vectorization),
                    )
                };

                let actual = client.read_one(output_handle.binding());
                let actual = $out_type::from_bytes(&actual);

                assert_eq!(actual, $expected);
            }
            )*
        }
    };
}

test_unary_impl!(
    test_magnitude,
    F,
    F::magnitude,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            input: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: 1., 23.1, 1.4, 5.1]
        },
        {
            input_vectorization: 2,
            out_vectorization: 1,
            input: as_type![F: -1., 0., 1., 5.],
            expected: as_type![F: 1.0, 5.099]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            input: as_type![F: -1., 0., 1., 5.],
            expected: as_type![F: 5.196]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            input: as_type![F: 0., 0., 0., 0.],
            expected: as_type![F: 0.]
        }
    ]
);

test_unary_impl!(test_abs, F, F::abs, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: -1., 0., 2., -3.],
        expected: as_type![F: 1., 0., 2., 3.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: -1., 0., 2., -3.],
        expected: as_type![F: 1., 0., 2., 3.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: -1., 0., 2., -3.],
        expected: as_type![F: 1., 0., 2., 3.]
    }
]);

test_unary_impl!(
    test_normalize,
    F,
    F::normalize,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            input: as_type![F: -1., 0., 1., 5.],
            expected: as_type![F: -1., f32::NAN, 1., 1.]
        },
        {
            input_vectorization: 2,
            out_vectorization: 2,
            input: as_type![F: -1., 0., 1., 5.],
            expected: as_type![F: -1.0, 0.0, 0.196, 0.981]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            input: as_type![F: -1., 0., 1., 5.],
            expected: as_type![F: -0.192, 0.0, 0.192, 0.962]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            input: as_type![F: 0., 0., 0., 0.],
            expected: as_type![F: f32::NAN, f32::NAN, f32::NAN, f32::NAN]
        },
        {
            input_vectorization: 2,
            out_vectorization: 2,
            input: as_type![F: 0., 0., 1., 0.],
            expected: as_type![F: f32::NAN, f32::NAN, 1., 0.]
        }
    ]
);

test_unary_impl_int_fixed!(test_count_ones, I, u32, I::count_ones, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111],
        expected: &[4, 1, 8]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111, 0b1100_0001],
        expected: &[4, 1, 8, 3]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111, 0b1100_0001],
        expected: &[4, 1, 8, 3]
    }
]);

macro_rules! shift {
    ($value:expr) => {{
        let shift = (size_of::<I>() - 1) * 8;
        $value << shift
    }};
}

test_unary_impl_int!(test_reverse_bits, I, I::reverse_bits, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111],
        expected: as_type![I: shift!(0b0100_0111), shift!(0b0000_0001), shift!(0b1111_1111)]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111, 0b1100_0001],
        expected: as_type![I: shift!(0b0100_0111), shift!(0b0000_0001), shift!(0b1111_1111), shift!(0b1000_0011)]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111, 0b1100_0001],
        expected: as_type![I: shift!(0b0100_0111), shift!(0b0000_0001), shift!(0b1111_1111), shift!(0b1000_0011)]
    }
]);

macro_rules! norm_lead {
    ($value:expr) => {{
        let shift = (size_of::<I>() - 1) * 8;
        $value + shift as u32
    }};
}

test_unary_impl_int_fixed!(test_leading_zeros, I, u32, I::leading_zeros, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1111],
        expected: &[norm_lead!(0), norm_lead!(8), norm_lead!(2)]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1111, 0b1111_1111],
        expected: &[norm_lead!(0), norm_lead!(8), norm_lead!(2), norm_lead!(0)]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1111, 0b1111_1111],
        expected: &[norm_lead!(0), norm_lead!(8), norm_lead!(2), norm_lead!(0)]
    }
]);

test_unary_impl_int_fixed!(test_find_first_set, I, u32, I::find_first_set, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b1111_1111],
        expected: &[2, 0, 1]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b1111_1111, 0b1000_0000],
        expected: &[2, 0, 1, 8]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b1111_1111, 0b1000_0000],
        expected: &[2, 0, 1, 8]
    }
]);

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_unary {
    () => {
        mod unary {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::unary::$test_name::<TestRuntime, FloatType>(
                            client,
                        );
                    }
                };
            }

            add_test!(test_normalize);
            add_test!(test_magnitude);
            add_test!(test_abs);
        }
    };
}

test_unary_impl_int!(test_abs_int, I, I::abs, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![I: 3, -5, 0, -127],
        expected: as_type![I: 3, 5, 0, 127]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![I: 3, -5, 0, -127],
        expected: as_type![I: 3, 5, 0, 127]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![I: 3, -5, 0, -127],
        expected: as_type![I: 3, 5, 0, 127]
    }
]);

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_unary_int {
    () => {
        mod unary_int {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::unary::$test_name::<TestRuntime, IntType>(
                            client,
                        );
                    }
                };
            }

            add_test!(test_abs_int);
            add_test!(test_count_ones);
            add_test!(test_reverse_bits);
            add_test!(test_leading_zeros);
            add_test!(test_find_first_set);
        }
    };
}
