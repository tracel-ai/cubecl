use crate as cubecl;

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

pub(crate) fn assert_equals_approx<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[f32],
    epsilon: f32,
) {
    let actual = client.read(output.binding());
    let actual = f32::from_bytes(&actual);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < epsilon || (a.is_nan() && e.is_nan()),
            "Values differ more than epsilon: actual={}, expected={}, difference={}, epsilon={}
index: {}
actual: {:?}
expected: {:?}",
            a,
            e,
            (a - e).abs(),
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
        pub fn $test_name<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
            #[cube(launch_unchecked)]
            fn test_function<$float_type: Float>(lhs: &Array<$float_type>, rhs: &Array<$float_type>, output: &mut Array<$float_type>) {
                if ABSOLUTE_POS < rhs.len() {
                    output[ABSOLUTE_POS] = $binary_func(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
                }
            }

            $(
            {
                let lhs = &$lhs;
                let rhs = &$rhs;
                let output_handle = client.empty($expected.len() * core::mem::size_of::<f32>());
                let lhs_handle = client.create(f32::as_bytes(lhs));
                let rhs_handle = client.create(f32::as_bytes(rhs));

                unsafe {
                    test_function::launch_unchecked::<f32, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((lhs.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts(&lhs_handle, lhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts(&rhs_handle, rhs.len(), $input_vectorization),
                        ArrayArg::from_raw_parts(&output_handle, $expected.len(), $out_vectorization),
                    )
                };

                assert_equals_approx::<R>(&client, output_handle, &$expected, 0.001);
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
            lhs: [1., -3.1, -2.4, 15.1],
            rhs: [-1., 23.1, -1.4, 5.1],
            expected: [-1.0, -71.61, 3.36, 77.01]
        },
        {
            input_vectorization: 2,
            out_vectorization: 1,
            lhs: [1., -3.1, -2.4, 15.1],
            rhs: [-1., 23.1, -1.4, 5.1],
            expected: [-72.61, 80.37]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            lhs: [1., -3.1, -2.4, 15.1],
            rhs: [-1., 23.1, -1.4, 5.1],
            expected: [7.76]
        },
        {
            input_vectorization: 4,
            out_vectorization: 1,
            lhs: [1., -3.1, -2.4, 15.1, -1., 23.1, -1.4, 5.1],
            rhs: [-1., 23.1, -1.4, 5.1, 1., -3.1, -2.4, 15.1],
            expected: [7.76, 7.76]
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
                        cubecl_core::runtime_tests::binary::$test_name::<TestRuntime>(client);
                    }
                };
            }

            add_test!(test_dot);
        }
    };
}
