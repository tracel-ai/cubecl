use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

macro_rules! impl_line_comparison {
    ($cmp:ident, $expected:expr) => {
        ::paste::paste! {
            #[cube(launch)]
            pub fn [< kernel_line_ $cmp >]<F: Float>(
                lhs: &Array<Line<F>>,
                rhs: &Array<Line<F>>,
                output: &mut Array<Line<u32>>,
            ) {
                if UNIT_POS == 0 {
                    output[0] = Line::cast_from(lhs[0].$cmp(rhs[0]));
                }
            }

            pub fn [< test_line_ $cmp >] <R: Runtime, F: Float + CubeElement>(
                client: ComputeClient<R::Server, R::Channel>,
            ) {
                let lhs = client.create(as_bytes![F: 0.0, 1.0, 2.0, 3.0]);
                let rhs = client.create(as_bytes![F: 0.0, 2.0, 1.0, 3.0]);
                let output = client.empty(16);

                unsafe {
                    [< kernel_line_ $cmp >]::launch::<F, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<F>(&lhs, 1, 4),
                        ArrayArg::from_raw_parts::<F>(&rhs, 1, 4),
                        ArrayArg::from_raw_parts::<u32>(&output, 1, 4),
                    )
                };

                let actual = client.read_one(output.binding());
                let actual = u32::from_bytes(&actual);

                assert_eq!(actual, $expected);
            }
        }
    };
}

impl_line_comparison!(equal, [1, 0, 0, 1]);
impl_line_comparison!(not_equal, [0, 1, 1, 0]);
impl_line_comparison!(less_than, [0, 1, 0, 0]);
impl_line_comparison!(greater_than, [0, 0, 1, 0]);
impl_line_comparison!(less_equal, [1, 1, 0, 1]);
impl_line_comparison!(greater_equal, [1, 0, 1, 1]);

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_line {
    () => {
        use super::*;

        #[test]
        fn test_line_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_equal::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_not_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_not_equal::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_less_than() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_less_than::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_greater_than() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_greater_than::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_line_less_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_less_equal::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_line_greater_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_greater_equal::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
