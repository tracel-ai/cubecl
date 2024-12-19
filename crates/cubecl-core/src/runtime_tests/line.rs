use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_line_index<F: Float>(output: &mut Array<F>, #[comptime] line_size: u32) {
    if UNIT_POS == 0 {
        let line = Line::empty(line_size).fill(F::new(5.0));
        output[0] = line[0];
    }
}

pub fn test_line_index<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem()) {
        let handle = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_line_index::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&handle, line_size as usize, 1),
                line_size as u32,
            );
        }
        let actual = client.read_one(handle.binding());
        let actual = F::from_bytes(&actual);

        let mut expected = vec![F::new(0.0); line_size as usize];
        expected[0] = F::new(5.0);

        assert_eq!(actual, expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_line_index_assign<F: Float>(output: &mut Array<Line<F>>) {
    if UNIT_POS == 0 {
        let mut line = output[0];
        line[0] = F::new(5.0);
        output[0] = line;
    }
}

pub fn test_line_index_assign<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem()) {
        let handle = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_line_index_assign::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&handle, 1, line_size),
            );
        }

        let actual = client.read_one(handle.binding());
        let actual = F::from_bytes(&actual);

        let mut expected = vec![F::new(0.0); line_size as usize];
        expected[0] = F::new(5.0);

        assert_eq!(actual, expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_line_loop_unroll(output: &mut Array<Line<u32>>, #[comptime] line_size: u32) {
    if UNIT_POS == 0 {
        let mut line = output[0];
        #[unroll]
        for k in 0..line_size {
            line[k] += k;
        }
        output[0] = line;
    }
}

pub mod kernel_line_loop_unrolld {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand(
        context: &mut cubecl::prelude::CubeContext,
        output: <Array<Line<u32>> as cubecl::prelude::CubeType>::ExpandType,
        line_size: u32,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _cond = {
                    let _lhs = UNIT_POS::expand(context);
                    let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(0);
                    cubecl::frontend::eq::expand(context, _lhs, _rhs)
                };
                cubecl::frontend::branch::if_expand(
                    context,
                    _cond.into(),
                    |context| {
                        let mut line = {
                            let _init = {
                                let _array = output.clone();
                                let _index = cubecl::frontend::ExpandElementTyped::from_lit(
                                    0,
                                );
                                cubecl::frontend::index::expand(context, _array, _index)
                            };
                            cubecl::frontend::Init::init(_init, context)
                        };
                        {
                            let _range = {
                                let _start = 0;
                                let _end = line_size.clone();
                                cubecl::frontend::RangeExpand::new(
                                    _start.into(),
                                    _end.into(),
                                    false,
                                )
                            };
                            let _unroll = true;
                            cubecl::frontend::branch::for_expand(
                                context,
                                _range,
                                _unroll,
                                |context, k| {
                                    {
                                        let _array = line.clone();
                                        let _index = k.clone();
                                        let _value = k;
                                        cubecl::frontend::add_assign_array_op::expand(
                                            context,
                                            _array,
                                            _index,
                                            _value,
                                        )
                                    };
                                    ()
                                },
                            );
                        };
                        {
                            let _array = output;
                            let _index = cubecl::frontend::ExpandElementTyped::from_lit(
                                0,
                            );
                            let _value = line;
                            cubecl::frontend::index_assign::expand(
                                context,
                                _array,
                                _index,
                                _value,
                            )
                        };
                        ()
                    },
                );
            };
            ()
        }
    }
}

pub fn test_line_loop_unroll<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem()) {
        let handle = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_line_loop_unroll::launch_unchecked::<R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&handle, 1, line_size),
                line_size as u32,
            );
        }

        let actual = client.read_one(handle.binding());
        let actual = F::from_bytes(&actual);

        let expected = (0..line_size as i64).map(|x| F::from_int(x)).collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }
}

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
        fn test_line_index() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_index::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_index_assign() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_index_assign::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_line_loop_unroll() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_loop_unroll::<TestRuntime, FloatType>(
                client,
            );
        }

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
