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
    for line_size in R::line_size_elem(&F::as_elem_native().unwrap()) {
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

        assert_eq!(&actual[..line_size as usize], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_line_index_assign<F: Float>(output: &mut Array<Line<F>>) {
    if UNIT_POS == 0 {
        let mut line = RuntimeCell::<Line<F>>::new(output[0]);
        line.store_at(0, F::new(5.0));
        output[0] = line.consume();
    }
}

pub fn test_line_index_assign<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem_native().unwrap()) {
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

        assert_eq!(&actual[..line_size as usize], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_line_loop_unroll<F: Float>(output: &mut Array<Line<F>>, #[comptime] line_size: u32) {
    if UNIT_POS == 0 {
        let mut line = output[0];
        #[unroll]
        for k in 0..line_size {
            line[k] += F::cast_from(k);
        }
        output[0] = line;
    }
}

pub fn test_line_loop_unroll<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem_native_unchecked()) {
        let handle = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_line_loop_unroll::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&handle, 1, line_size),
                line_size as u32,
            );
        }

        let actual = client.read_one(handle.binding());
        let actual = F::from_bytes(&actual);

        let expected = (0..line_size as i64)
            .map(|x| F::from_int(x))
            .collect::<Vec<_>>();

        assert_eq!(&actual[..line_size as usize], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_shared_memory<F: Float>(output: &mut Array<Line<F>>) {
    let mut smem1 = SharedMemory::<F>::new_lined(8, output.line_size());
    smem1[0] = Line::new(F::new(42.0));
    output[0] = smem1[0];
}

pub fn test_shared_memory<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem_native().unwrap()) {
        let output = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_shared_memory::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&output, 1, line_size),
            );
        }

        let actual = client.read_one(output.binding());
        let actual = F::from_bytes(&actual);

        assert_eq!(actual[0], F::new(42.0));
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
        fn test_shared_memory() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_shared_memory::<TestRuntime, FloatType>(client);
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
