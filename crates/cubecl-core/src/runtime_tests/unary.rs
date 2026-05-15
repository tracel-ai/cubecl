#![allow(clippy::approx_constant)]

use core::f32;
use core::f32::consts::PI;

use core::fmt::Display;

use crate::{self as cubecl, as_type};

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

pub(crate) fn assert_equals_approx<R: Runtime, F: num_traits::Float + CubeElement + Display>(
    client: &ComputeClient<R>,
    output: Handle,
    expected: &[F],
    epsilon: F,
) {
    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*a - *e).abs() < epsilon
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

fn ordered_f32_bits(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    if bits < 0 { i32::MIN - bits } else { bits }
}

fn assert_f32_ulp_le(actual: f32, expected: f32, max_ulp: u32) {
    if actual == expected {
        return;
    }

    let diff = ordered_f32_bits(actual).abs_diff(ordered_f32_bits(expected));
    assert!(
        diff <= max_ulp,
        "Values differ by more than {max_ulp} ulp: actual={actual}, expected={expected}, ulp_diff={diff}"
    );
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
        test_unary_impl!($test_name, $float_type, $unary_func, [$({
            input_vectorization: $input_vectorization,
            out_vectorization: $out_vectorization,
            input: $input,
            expected: $expected
        }),*], 0.02);
    };
    (
        $test_name:ident,
        $float_type:ident,
        $unary_func:expr,
        [$({
            input_vectorization: $input_vectorization:expr,
            out_vectorization: $out_vectorization:expr,
            input: $input:expr,
            expected: $expected:expr
        }),*],
        $epsilon:expr) => {
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R>) {
            #[cube(launch_unchecked, fast_math = FastMath::all())]
            fn test_function<$float_type: Float, In: Size, Out: Size>(
                input: &Array<Vector<$float_type, In>>, output: &mut Array<Vector<$float_type, Out>>
            ) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = Vector::cast_from($unary_func(input[ABSOLUTE_POS]));
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$float_type>());
                let input_handle = client.create_from_slice($float_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$float_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d((input.len() / $input_vectorization as usize) as u32),
                        $input_vectorization,
                        $out_vectorization,
                        ArrayArg::from_raw_parts(input_handle, input.len()),
                        ArrayArg::from_raw_parts(output_handle.clone(), $expected.len()),
                    )
                };

                assert_equals_approx::<R, $float_type>(&client, output_handle, $expected, $float_type::new($epsilon));
            }
            )*
        }
    };
}

macro_rules! test_unary_impl_fixed {
    (
        $test_name:ident,
        $float_type:ident,
        $out_type:ident,
        $unary_func:expr,
        [$({
            input_vectorization: $input_vectorization:expr,
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R>) {
            #[cube(launch_unchecked)]
            fn test_function<$float_type: Float, N: Size>(
                input: &Array<Vector<$float_type, N>>,
                output: &mut Array<Vector<$out_type, N>>
            ) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = Vector::cast_from($unary_func(input[ABSOLUTE_POS]));
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$out_type>());
                let input_handle = client.create_from_slice($float_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$float_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d((input.len() / $input_vectorization as usize) as u32),
                        $input_vectorization,
                        ArrayArg::from_raw_parts(input_handle, input.len()),
                        ArrayArg::from_raw_parts(output_handle.clone(), $expected.len()),
                    )
                };

                let actual = client.read_one_unchecked(output_handle);
                let actual = $out_type::from_bytes(&actual);

                assert_eq!(actual, $expected);
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
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $int_type: Int + CubeElement>(client: ComputeClient<R>) {
            #[cube(launch_unchecked)]
            fn test_function<$int_type: Int, N: Size>(
                input: &Array<Vector<$int_type, N>>,
                output: &mut Array<Vector<$int_type, N>>
            ) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = Vector::cast_from($unary_func(input[ABSOLUTE_POS]));
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$int_type>());
                let input_handle = client.create_from_slice($int_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$int_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d((input.len() / $input_vectorization as usize) as u32),
                        $input_vectorization,
                        ArrayArg::from_raw_parts(input_handle, input.len()),
                        ArrayArg::from_raw_parts(output_handle.clone(), $expected.len()),
                    )
                };

                let actual = client.read_one_unchecked(output_handle);
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
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $int_type: Int + CubeElement>(client: ComputeClient<R>) {
            #[cube(launch_unchecked)]
            fn test_function<$int_type: Int, N: Size>(
                input: &Array<Vector<$int_type, N>>,
                output: &mut Array<Vector<$out_type, N>>
            ) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = Vector::cast_from($unary_func(input[ABSOLUTE_POS]));
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$out_type>());
                let input_handle = client.create_from_slice($int_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$int_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d((input.len() / $input_vectorization as usize) as u32),
                        $input_vectorization,
                        ArrayArg::from_raw_parts(input_handle, input.len()),
                        ArrayArg::from_raw_parts(output_handle.clone(), $expected.len()),
                    )
                };

                let actual = client.read_one_unchecked(output_handle);
                let actual = $out_type::from_bytes(&actual);

                assert_eq!(actual, $expected);
            }
            )*
        }
    };
}

test_unary_impl!(test_sin, F, Vector::sin, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1.570_796_4, 3.141_592_7, -1.570_796_4],
        expected: as_type![F: 0., 1., 0., -1.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1.570_796_4, 3.141_592_7, -1.570_796_4],
        expected: as_type![F: 0., 1., 0., -1.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1.570_796_4, 3.141_592_7, -1.570_796_4],
        expected: as_type![F: 0., 1., 0., -1.]
    }
]);

test_unary_impl!(test_cos, F, Vector::cos, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1.570_796_4, 3.141_592_7, -1.570_796_4],
        expected: as_type![F: 1., 0., -1., 0.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1.570_796_4, 3.141_592_7, -1.570_796_4],
        expected: as_type![F: 1., 0., -1., 0.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1.570_796_4, 3.141_592_7, -1.570_796_4],
        expected: as_type![F: 1., 0., -1., 0.]
    }
]);

test_unary_impl!(test_tan, F, Vector::tan, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 0.785_398_2, 1.047_197_6, -0.785_398_2],
        expected: as_type![F: 0., 1., 1.732_050_8, -1.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 0.785_398_2, 1.047_197_6, -0.785_398_2],
        expected: as_type![F: 0., 1., 1.732_050_8, -1.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 0.785_398_2, 1.047_197_6, -0.785_398_2],
        expected: as_type![F: 0., 1., 1.732_050_8, -1.]
    }
]);

test_unary_impl!(test_asin, F, Vector::asin, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 0.5, 1., -0.5, -1.],
        expected: as_type![F: 0., 0.523_598_8, 1.570_796_4, -0.523_598_8, -1.570_796_4]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 0.5, 1., -0.5],
        expected: as_type![F: 0., 0.523_598_8, 1.570_796_4, -0.523_598_8]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 0.5, 1., -0.5],
        expected: as_type![F: 0., 0.523_598_8, 1.570_796_4, -0.523_598_8]
    }
]);

test_unary_impl!(test_acos, F, Vector::acos, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 1., 0.5, 0., -0.5, -1.],
        expected: as_type![F: 0., 1.047_197_6, 1.570_796_4, 2.094_395_2, 3.141_592_7]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 1., 0.5, 0., -0.5],
        expected: as_type![F: 0., 1.047_197_6, 1.570_796_4, 2.094_395_2]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 1., 0.5, 0., -0.5],
        expected: as_type![F: 0., 1.047_197_6, 1.570_796_4, 2.094_395_2]
    }
]);

test_unary_impl!(test_atan, F, Vector::atan, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 1000., -1000.],
        expected: as_type![F: 0., 0.785_398_2, -0.785_398_2, 1.569_796_3, -1.569_796_3]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 1000.],
        expected: as_type![F: 0., 0.785_398_2, -0.785_398_2, 1.569_796_3]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 1000.],
        expected: as_type![F: 0., 0.785_398_2, -0.785_398_2, 1.569_796_3]
    }
]);

test_unary_impl!(test_sinh, F, Vector::sinh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 0., 1.175_201_2, -1.175_201_2, 3.626_860_4, -3.626_860_4]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 1.175_201_2, -1.175_201_2, 3.626_860_4]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 1.175_201_2, -1.175_201_2, 3.626_860_4]
    }
]);

test_unary_impl!(test_cosh, F, Vector::cosh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 1., 1.543_080_7, 1.543_080_7, 3.762_195_6, 3.762_195_6]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 1., 1.543_080_7, 1.543_080_7, 3.762_195_6]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 1., 1.543_080_7, 1.543_080_7, 3.762_195_6]
    }
]);

test_unary_impl!(test_tanh, F, Vector::tanh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 0., 0.761_594_2, -0.761_594_2, 0.964_027_6, -0.964_027_6]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.761_594_2, -0.761_594_2, 0.964_027_6]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.761_594_2, -0.761_594_2, 0.964_027_6]
    }
]);

test_unary_impl!(test_asinh, F, Vector::asinh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 0., 0.881_373_6, -0.881_373_6, 1.443_635_5, -1.443_635_5]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.881_373_6, -0.881_373_6, 1.443_635_5]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.881_373_6, -0.881_373_6, 1.443_635_5]
    }
]);

test_unary_impl!(test_acosh, F, Vector::acosh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 1., 2., 3., 10.],
        expected: as_type![F: 0., 1.316_958, 1.762_747_2, 2.993_223]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 1., 2., 3., 10.],
        expected: as_type![F: 0., 1.316_958, 1.762_747_2, 2.993_223]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 1., 2., 3., 10.],
        expected: as_type![F: 0., 1.316_958, 1.762_747_2, 2.993_223]
    }
]);

test_unary_impl!(test_atanh, F, Vector::atanh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 0.5, -0.5, 0.9, -0.9],
        expected: as_type![F: 0., 0.549_306_15, -0.549_306_15, 1.472_219_5, -1.472_219_5]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 0.5, -0.5, 0.9],
        expected: as_type![F: 0., 0.549_306_15, -0.549_306_15, 1.472_219_5]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 0.5, -0.5, 0.9],
        expected: as_type![F: 0., 0.549_306_15, -0.549_306_15, 1.472_219_5]
    }
]);

test_unary_impl!(test_sqrt, F, Vector::sqrt, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., 4., 9., 16., 25.],
        expected: as_type![F: 0., 1., 2., 3., 4., 5.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., 4., 9.],
        expected: as_type![F: 0., 1., 2., 3.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., 4., 9.],
        expected: as_type![F: 0., 1., 2., 3.]
    }
]);

test_unary_impl!(test_degrees, F, Vector::to_degrees, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., PI / 2., PI, PI * 2., -PI / 2., -PI, -PI * 2.],
        expected: as_type![F: 0., 90., 180., 360., -90., -180., -360.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., PI / 2., PI, -PI / 2.],
        expected: as_type![F: 0., 90., 180., -90.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., PI / 2., PI, -PI / 2.],
        expected: as_type![F: 0., 90., 180., -90.]
    }
], 0.3);

test_unary_impl!(test_radians, F, Vector::to_radians, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 90., 180., 360., -90., -180., -360.],
        expected: as_type![F: 0., PI / 2., PI, PI * 2., -PI / 2., -PI, -PI * 2.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 90., 180., -90.],
        expected: as_type![F: 0., PI / 2., PI, -PI / 2.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 90., 180., -90.],
        expected: as_type![F: 0., PI / 2., PI, -PI / 2.]
    }
]);

test_unary_impl!(
    test_magnitude,
    F,
    Vector::magnitude,
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

test_unary_impl!(
    test_vector_sum,
    F,
    Vector::vector_sum,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            input: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: -1., 23.1, -1.4, 5.1]
        },
        {
            input_vectorization: 2,
            out_vectorization: 1,
            input: as_type![F: 1., 3., 2., 5.],
            expected: as_type![F: 4., 7.]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            input: as_type![F: 1., 2., 3., 4.],
            expected: as_type![F: 10.]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            input: as_type![F: 0., 0., 0., 0.],
            expected: as_type![F: 0.]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            input: as_type![F: -1., 1., -2., 2.],
            expected: as_type![F: 0.]
        }
    ]
);

test_unary_impl!(test_abs, F, Vector::abs, [
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

test_unary_impl!(test_inverse_sqrt, F, Vector::inverse_sqrt, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 1.0, 4.0, 16.0, 0.25],
        expected: as_type![F: 1.0, 0.5, 0.25, 2.0]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 9.0, 25.0, 0.0625, 100.0],
        expected: as_type![F: 0.333_333_34, 0.2, 4.0, 0.1]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0.0, 0.01, 64.0, 0.111111],
        expected: as_type![F: f32::INFINITY, 10.0, 0.125, 3.0]
    }
]);

test_unary_impl!(
    test_normalize,
    F,
    Vector::normalize,
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

test_unary_impl!(
    test_trunc,
    F,
    Vector::trunc,
    [{
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: -1.2, -1., -0., 0.],
        expected: as_type![F: -1., -1., -0., 0.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: f32::NAN, 1., 1.2, 1.9],
        expected: as_type![F: f32::NAN, 1., 1., 1.0]
    },{
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: -0.9, 0.2, f32::NAN, 1.99],
        expected: as_type![F: -0., 0., f32::NAN, 1.]
    }]
);

test_unary_impl_fixed!(
    test_is_nan,
    F,
    u32,
    IsNan::is_nan,
    [
        {
            input_vectorization: 1,
            input: &[F::new(0.), F::NAN, F::INFINITY, F::NEG_INFINITY],
            expected: as_type![u32: false as i64, true as i64, false as i64, false as i64]
        },
        {
            input_vectorization: 2,
            input: &[F::INFINITY, F::new(-100.), F::NAN, F::NEG_INFINITY],
            expected: as_type![u32: false as i64, false as i64, true as i64, false as i64]
        },
        {
            input_vectorization: 4,
            input: &[F::NEG_INFINITY, F::INFINITY, F::new(100.), F::NAN],
            expected: as_type![u32: false as i64, false as i64, false as i64, true as i64]
        }
    ]
);

test_unary_impl_fixed!(
    test_is_inf,
    F,
    u32,
    IsInf::is_inf,
    [
        {
            input_vectorization: 1,
            input: as_type![F: 0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            expected: as_type![u32: false as i64, false as i64, true as i64, true as i64]
        },
        {
            input_vectorization: 2,
            input: as_type![F: f32::INFINITY, -100., f32::NAN, f32::NEG_INFINITY],
            expected: as_type![u32: true as i64, false as i64, false as i64, true as i64]
        },
        {
            input_vectorization: 4,
            input: as_type![F: f32::NEG_INFINITY, f32::INFINITY, 100., f32::NAN],
            expected: as_type![u32: true as i64, true as i64, false as i64, false as i64]
        }
    ]
);

pub fn test_expm1_f32<R: Runtime>(client: ComputeClient<R>) {
    #[cube(launch_unchecked)]
    fn test_function<In: Size, Out: Size>(
        input: &Array<Vector<f32, In>>,
        output: &mut Array<Vector<f32, Out>>,
    ) {
        if ABSOLUTE_POS < input.len() {
            output[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS].exp_m1());
        }
    }

    let input = &[0.0f32, 1.0e-7, 10.0, -1.0e-7];
    let expected = [
        0.0f32.exp_m1(),
        1.0e-7f32.exp_m1(),
        10.0f32.exp_m1(),
        (-1.0e-7f32).exp_m1(),
    ];

    for &(input_vectorization, out_vectorization) in &[(1usize, 1usize), (2, 2), (4, 4)] {
        let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
        let input_handle = client.create_from_slice(f32::as_bytes(input));

        unsafe {
            test_function::launch_unchecked::<R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d((input.len() / input_vectorization) as u32),
                input_vectorization,
                out_vectorization,
                ArrayArg::from_raw_parts(input_handle, input.len()),
                ArrayArg::from_raw_parts(output_handle.clone(), expected.len()),
            )
        };

        let actual = client.read_one_unchecked(output_handle);
        let actual = f32::from_bytes(&actual);

        assert_eq!(actual[0], 0.0);
        assert_f32_ulp_le(actual[1], expected[1], 2);
        assert_f32_ulp_le(actual[2], expected[2], 2);
        assert_f32_ulp_le(actual[3], expected[3], 2);
    }
}

test_unary_impl_int_fixed!(test_count_ones, I, u32, Vector::count_ones, [
    {
        input_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111],
        expected: &[4, 1, 8]
    },
    {
        input_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111, 0b1100_0001],
        expected: &[4, 1, 8, 3]
    },
    {
        input_vectorization: 4,
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

test_unary_impl_int!(test_reverse_bits, I, ReverseBits::reverse_bits, [
    {
        input_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111],
        expected: as_type![I: shift!(0b0100_0111), shift!(0b0000_0001), shift!(0b1111_1111)]
    },
    {
        input_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b1000_0000, 0b1111_1111, 0b1100_0001],
        expected: as_type![I: shift!(0b0100_0111), shift!(0b0000_0001), shift!(0b1111_1111), shift!(0b1000_0011)]
    },
    {
        input_vectorization: 4,
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

test_unary_impl_int_fixed!(test_leading_zeros, I, u32, Vector::leading_zeros, [
    {
        input_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1111],
        expected: &[norm_lead!(0), norm_lead!(8), norm_lead!(2)]
    },
    {
        input_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1111, 0b1111_1111],
        expected: &[norm_lead!(0), norm_lead!(8), norm_lead!(2), norm_lead!(0)]
    },
    {
        input_vectorization: 4,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1111, 0b1111_1111],
        expected: &[norm_lead!(0), norm_lead!(8), norm_lead!(2), norm_lead!(0)]
    }
]);

test_unary_impl_int_fixed!(test_find_first_set, I, u32, Vector::find_first_set, [
    {
        input_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b1111_1111],
        expected: &[2, 0, 1]
    },
    {
        input_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b1111_1111, 0b1000_0000],
        expected: &[2, 0, 1, 8]
    },
    {
        input_vectorization: 4,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b1111_1111, 0b1000_0000],
        expected: &[2, 0, 1, 8]
    }
]);

test_unary_impl_int_fixed!(test_trailing_zeros, I, u32, Vector::trailing_zeros, [
    {
        input_vectorization: 1,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1000],
        // trailing zeros: 1, all bits (size*8), 3
        expected: &[1, size_of::<I>() as u32 * 8, 3]
    },
    {
        input_vectorization: 2,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1000, 0b1111_1111],
        expected: &[1, size_of::<I>() as u32 * 8, 3, 0]
    },
    {
        input_vectorization: 4,
        input: as_type![I: 0b1110_0010, 0b0000_0000, 0b0010_1000, 0b1111_1111],
        expected: &[1, size_of::<I>() as u32 * 8, 3, 0]
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
                    #[$crate::runtime_tests::test_log::test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::unary::$test_name::<TestRuntime, FloatType>(
                            client,
                        );
                    }
                };
            }

            add_test!(test_sin);
            add_test!(test_cos);
            add_test!(test_tan);
            add_test!(test_sinh);
            add_test!(test_cosh);
            add_test!(test_tanh);
            add_test!(test_asin);
            add_test!(test_acos);
            add_test!(test_atan);
            add_test!(test_asinh);
            add_test!(test_acosh);
            add_test!(test_atanh);
            add_test!(test_degrees);
            add_test!(test_radians);
            add_test!(test_normalize);
            add_test!(test_magnitude);
            add_test!(test_vector_sum);
            add_test!(test_sqrt);
            add_test!(test_inverse_sqrt);
            add_test!(test_abs);
            add_test!(test_trunc);
            add_test!(test_is_nan);
            add_test!(test_is_inf);

            #[$crate::runtime_tests::test_log::test]
            fn test_expm1_f32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::unary::test_expm1_f32::<TestRuntime>(client);
            }
        }
    };
}

test_unary_impl_int!(test_abs_int, I, Abs::abs, [
    {
        input_vectorization: 1,
        input: as_type![I: 3, -5, 0, -127],
        expected: as_type![I: 3, 5, 0, 127]
    },
    {
        input_vectorization: 2,
        input: as_type![I: 3, -5, 0, -127],
        expected: as_type![I: 3, 5, 0, 127]
    },
    {
        input_vectorization: 4,
        input: as_type![I: 3, -5, 0, -127],
        expected: as_type![I: 3, 5, 0, 127]
    }
]);

pub fn test_vector_sum_int<R: Runtime, I: Int + CubeElement>(client: ComputeClient<R>) {
    #[cube(launch_unchecked)]
    fn test_function<I: Int, In: Size, Out: Size>(
        input: &Array<Vector<I, In>>,
        output: &mut Array<Vector<I, Out>>,
    ) {
        if ABSOLUTE_POS < input.len() {
            output[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS].vector_sum());
        }
    }

    // vec1: identity
    {
        let input = as_type![I: 3, -5, 7, -2];
        let output_handle = client.empty(input.len() * core::mem::size_of::<I>());
        let input_handle = client.create_from_slice(I::as_bytes(input));

        unsafe {
            test_function::launch_unchecked::<I, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(input.len() as u32),
                1usize,
                1usize,
                ArrayArg::from_raw_parts(input_handle, input.len()),
                ArrayArg::from_raw_parts(output_handle.clone(), input.len()),
            )
        };

        let actual = client.read_one_unchecked(output_handle);
        let actual = I::from_bytes(&actual);
        assert_eq!(actual, as_type![I: 3, -5, 7, -2]);
    }

    // vec2: sum pairs
    {
        let input = as_type![I: 1, 3, 2, 5];
        let output_handle = client.empty(2 * core::mem::size_of::<I>());
        let input_handle = client.create_from_slice(I::as_bytes(input));

        unsafe {
            test_function::launch_unchecked::<I, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                2usize,
                1usize,
                ArrayArg::from_raw_parts(input_handle, input.len()),
                ArrayArg::from_raw_parts(output_handle.clone(), 2),
            )
        };

        let actual = client.read_one_unchecked(output_handle);
        let actual = I::from_bytes(&actual);
        assert_eq!(actual, as_type![I: 4, 7]);
    }

    // vec4: sum all 4
    {
        let input = as_type![I: 1, 2, 3, 4];
        let output_handle = client.empty(core::mem::size_of::<I>());
        let input_handle = client.create_from_slice(I::as_bytes(input));

        unsafe {
            test_function::launch_unchecked::<I, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                4usize,
                1usize,
                ArrayArg::from_raw_parts(input_handle, input.len()),
                ArrayArg::from_raw_parts(output_handle.clone(), 1),
            )
        };

        let actual = client.read_one_unchecked(output_handle);
        let actual = I::from_bytes(&actual);
        assert_eq!(actual, as_type![I: 10]);
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_unary_int {
    () => {
        mod unary_int {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[$crate::runtime_tests::test_log::test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::unary::$test_name::<TestRuntime, IntType>(
                            client,
                        );
                    }
                };
            }

            add_test!(test_abs_int);
            add_test!(test_vector_sum_int);
            add_test!(test_count_ones);
            add_test!(test_reverse_bits);
            add_test!(test_leading_zeros);
            add_test!(test_trailing_zeros);
            add_test!(test_find_first_set);
        }
    };
}
