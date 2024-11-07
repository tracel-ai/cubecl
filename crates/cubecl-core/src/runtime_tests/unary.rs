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
    let actual = client.read(output.binding());
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

                assert_equals_approx::<R, F>(&client, output_handle, $expected, $float_type::new(0.02));
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
        }
    };
}
