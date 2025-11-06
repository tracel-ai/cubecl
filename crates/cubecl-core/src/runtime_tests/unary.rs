use std::f32::consts::PI;
use std::fmt::Display;

use crate::{self as cubecl, as_type};

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

pub(crate) fn assert_equals_approx<
    R: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: &ComputeClient<R::Server>,
    output: Handle,
    expected: &[F],
    epsilon: F,
) {
    let actual = client.read_one(output);
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
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R::Server>) {
            #[cube(launch_unchecked, fast_math = FastMath::all())]
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
            out_vectorization: $out_vectorization:expr,
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R::Server>) {
            #[cube(launch_unchecked)]
            fn test_function<$float_type: Float>(input: &Array<$float_type>, output: &mut Array<$out_type>) {
                if ABSOLUTE_POS < input.len() {
                    output[ABSOLUTE_POS] = $unary_func(input[ABSOLUTE_POS]) as $out_type;
                }
            }

            $(
            {
                let input = $input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<$out_type>());
                let input_handle = client.create($float_type::as_bytes(input));

                unsafe {
                    test_function::launch_unchecked::<$float_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((input.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$float_type>(&input_handle, input.len(), $input_vectorization),
                        ArrayArg::from_raw_parts::<$out_type>(&output_handle, $expected.len(), $out_vectorization),
                    )
                };

                let actual = client.read_one(output_handle);
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
            out_vectorization: $out_vectorization:expr,
            input: $input:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $int_type: Int + CubeElement>(client: ComputeClient<R::Server>) {
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

                let actual = client.read_one(output_handle);
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
        pub fn $test_name<R: Runtime, $int_type: Int + CubeElement>(client: ComputeClient<R::Server>) {
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

                let actual = client.read_one(output_handle);
                let actual = $out_type::from_bytes(&actual);

                assert_eq!(actual, $expected);
            }
            )*
        }
    };
}

test_unary_impl!(test_sin, F, F::sin, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1.57079632679, 3.14159265359, -1.57079632679],
        expected: as_type![F: 0., 1., 0., -1.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1.57079632679, 3.14159265359, -1.57079632679],
        expected: as_type![F: 0., 1., 0., -1.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1.57079632679, 3.14159265359, -1.57079632679],
        expected: as_type![F: 0., 1., 0., -1.]
    }
]);

test_unary_impl!(test_cos, F, F::cos, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1.57079632679, 3.14159265359, -1.57079632679],
        expected: as_type![F: 1., 0., -1., 0.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1.57079632679, 3.14159265359, -1.57079632679],
        expected: as_type![F: 1., 0., -1., 0.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1.57079632679, 3.14159265359, -1.57079632679],
        expected: as_type![F: 1., 0., -1., 0.]
    }
]);

test_unary_impl!(test_tan, F, F::tan, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 0.78539816339, 1.04719755119, -0.78539816339],
        expected: as_type![F: 0., 1., 1.73205080757, -1.]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 0.78539816339, 1.04719755119, -0.78539816339],
        expected: as_type![F: 0., 1., 1.73205080757, -1.]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 0.78539816339, 1.04719755119, -0.78539816339],
        expected: as_type![F: 0., 1., 1.73205080757, -1.]
    }
]);

test_unary_impl!(test_asin, F, F::asin, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 0.5, 1., -0.5, -1.],
        expected: as_type![F: 0., 0.52359877559, 1.57079632679, -0.52359877559, -1.57079632679]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 0.5, 1., -0.5],
        expected: as_type![F: 0., 0.52359877559, 1.57079632679, -0.52359877559]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 0.5, 1., -0.5],
        expected: as_type![F: 0., 0.52359877559, 1.57079632679, -0.52359877559]
    }
]);

test_unary_impl!(test_acos, F, F::acos, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 1., 0.5, 0., -0.5, -1.],
        expected: as_type![F: 0., 1.04719755119, 1.57079632679, 2.09439510239, 3.14159265359]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 1., 0.5, 0., -0.5],
        expected: as_type![F: 0., 1.04719755119, 1.57079632679, 2.09439510239]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 1., 0.5, 0., -0.5],
        expected: as_type![F: 0., 1.04719755119, 1.57079632679, 2.09439510239]
    }
]);

test_unary_impl!(test_atan, F, F::atan, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 1000., -1000.],
        expected: as_type![F: 0., 0.78539816339, -0.78539816339, 1.56979632472, -1.56979632472]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 1000.],
        expected: as_type![F: 0., 0.78539816339, -0.78539816339, 1.56979632472]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 1000.],
        expected: as_type![F: 0., 0.78539816339, -0.78539816339, 1.56979632472]
    }
]);

test_unary_impl!(test_sinh, F, F::sinh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 0., 1.1752011936, -1.1752011936, 3.6268604078, -3.6268604078]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 1.1752011936, -1.1752011936, 3.6268604078]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 1.1752011936, -1.1752011936, 3.6268604078]
    }
]);

test_unary_impl!(test_cosh, F, F::cosh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 1., 1.5430806348, 1.5430806348, 3.7621956911, 3.7621956911]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 1., 1.5430806348, 1.5430806348, 3.7621956911]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 1., 1.5430806348, 1.5430806348, 3.7621956911]
    }
]);

test_unary_impl!(test_tanh, F, F::tanh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 0., 0.7615941559, -0.7615941559, 0.9640275801, -0.9640275801]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.7615941559, -0.7615941559, 0.9640275801]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.7615941559, -0.7615941559, 0.9640275801]
    }
]);

test_unary_impl!(test_asinh, F, F::asinh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 1., -1., 2., -2.],
        expected: as_type![F: 0., 0.88137358702, -0.88137358702, 1.44363547517, -1.44363547517]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.88137358702, -0.88137358702, 1.44363547517]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 1., -1., 2.],
        expected: as_type![F: 0., 0.88137358702, -0.88137358702, 1.44363547517]
    }
]);

test_unary_impl!(test_acosh, F, F::acosh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 1., 2., 3., 10.],
        expected: as_type![F: 0., 1.31695789692, 1.76274717404, 2.99322284612]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 1., 2., 3., 10.],
        expected: as_type![F: 0., 1.31695789692, 1.76274717404, 2.99322284612]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 1., 2., 3., 10.],
        expected: as_type![F: 0., 1.31695789692, 1.76274717404, 2.99322284612]
    }
]);

test_unary_impl!(test_atanh, F, F::atanh, [
    {
        input_vectorization: 1,
        out_vectorization: 1,
        input: as_type![F: 0., 0.5, -0.5, 0.9, -0.9],
        expected: as_type![F: 0., 0.54930614433, -0.54930614433, 1.47221948958, -1.47221948958]
    },
    {
        input_vectorization: 2,
        out_vectorization: 2,
        input: as_type![F: 0., 0.5, -0.5, 0.9],
        expected: as_type![F: 0., 0.54930614433, -0.54930614433, 1.47221948958]
    },
    {
        input_vectorization: 4,
        out_vectorization: 4,
        input: as_type![F: 0., 0.5, -0.5, 0.9],
        expected: as_type![F: 0., 0.54930614433, -0.54930614433, 1.47221948958]
    }
]);

test_unary_impl!(test_sqrt, F, F::sqrt, [
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

test_unary_impl!(test_degrees, F, F::to_degrees, [
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

test_unary_impl!(test_radians, F, F::to_radians, [
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

test_unary_impl!(test_inverse_sqrt, F, F::inverse_sqrt, [
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

test_unary_impl!(
    test_trunc,
    F,
    F::trunc,
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
    F::is_nan,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            input: &[F::new(0.), F::NAN, F::INFINITY, F::NEG_INFINITY],
            expected: as_type![u32: false as i64, true as i64, false as i64, false as i64]
        },
        {
            input_vectorization: 2,
            out_vectorization: 2,
            input: &[F::INFINITY, F::new(-100.), F::NAN, F::NEG_INFINITY],
            expected: as_type![u32: false as i64, false as i64, true as i64, false as i64]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            input: &[F::NEG_INFINITY, F::INFINITY, F::new(100.), F::NAN],
            expected: as_type![u32: false as i64, false as i64, false as i64, true as i64]
        }
    ]
);

test_unary_impl_fixed!(
    test_is_inf,
    F,
    u32,
    F::is_inf,
    [
        {
            input_vectorization: 1,
            out_vectorization: 1,
            input: as_type![F: 0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            expected: as_type![u32: false as i64, false as i64, true as i64, true as i64]
        },
        {
            input_vectorization: 2,
            out_vectorization: 2,
            input: as_type![F: f32::INFINITY, -100., f32::NAN, f32::NEG_INFINITY],
            expected: as_type![u32: true as i64, false as i64, false as i64, true as i64]
        },
        {
            input_vectorization: 4,
            out_vectorization: 4,
            input: as_type![F: f32::NEG_INFINITY, f32::INFINITY, 100., f32::NAN],
            expected: as_type![u32: true as i64, true as i64, false as i64, false as i64]
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
            add_test!(test_inverse_sqrt);
            add_test!(test_magnitude);
            add_test!(test_sqrt);
            add_test!(test_inverse_sqrt);
            add_test!(test_abs);
            add_test!(test_trunc);
            add_test!(test_is_nan);
            add_test!(test_is_inf);
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
