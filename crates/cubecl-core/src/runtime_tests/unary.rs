use crate as cubecl;

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

pub(crate) fn assert_equals_approx<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle<<R as Runtime>::Server>,
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
lhs: {:?}
rhs: {:?}",
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

macro_rules! test_unary_impl {
    ($test_name:ident, $unary_func:ident, [$($input:expr, $input_vectorization:expr => $expected:expr, $output_vectorization:expr);*]) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
            $(
            {
                let input = &$input;
                let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
                let input_handle = client.create(f32::as_bytes(input));

                unsafe {
                    $unary_func::launch_unchecked::<F32, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((input.len() / $input_vectorization as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts(&input_handle, input.len(), $input_vectorization),
                        ArrayArg::from_raw_parts(&output_handle, input.len(), $output_vectorization),
                    )
                };

                assert_equals_approx::<R>(&client, output_handle, &$expected, 0.001);
            }
            )*
        }
    };
}

#[cube(launch_unchecked)]
fn normalize<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = F::normalize(input[ABSOLUTE_POS]);
    }
}
test_unary_impl!(
    test_normalize,
    normalize,
    [
        [-1., 0., 1., 5.],4 => [-0.192, 0.0, 0.192, 0.9623],4;
        [0., 0., 0., 0.],4 => [f32::NAN, f32::NAN, f32::NAN, f32::NAN],4
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
                        cubecl_core::runtime_tests::unary::$test_name::<TestRuntime>(client);
                    }
                };
            }

            add_test!(test_normalize);
        }
    };
}
