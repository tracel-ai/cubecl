use alloc::{vec, vec::Vec};

use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_vector_index<F: Float, N: Size>(output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let vector = Vector::<F, N>::new(F::new(5.0));
        for i in 0..4 {
            output[i] = vector[i];
        }
    }
}

#[allow(clippy::needless_range_loop)]
pub fn test_vector_index<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    for vector_size in client.io_optimized_vector_sizes(size_of::<F>()) {
        if vector_size < 4 {
            continue;
        }
        let handle = client.create_from_slice(F::as_bytes(&vec![F::new(0.0); vector_size]));
        unsafe {
            kernel_vector_index::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                vector_size,
                ArrayArg::from_raw_parts(handle.clone(), vector_size),
            )
        }
        let actual = client.read_one_unchecked(handle);
        let actual = F::from_bytes(&actual);

        let mut expected = vec![F::new(0.0); vector_size];
        for i in 0..4 {
            expected[i] = F::new(5.0);
        }

        assert_eq!(&actual[..vector_size], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_vector_index_assign<F: Float, N: Size>(output: &mut Array<Vector<F, N>>) {
    if UNIT_POS == 0 {
        let mut vector = RuntimeCell::<Vector<F, N>>::new(output[0]);
        vector.store_at(0, F::new(5.0));
        output[0] = vector.consume();
    }
}

pub fn test_vector_index_assign<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    for vector_size in client.io_optimized_vector_sizes(size_of::<F>()) {
        let handle = client.create_from_slice(F::as_bytes(&vec![F::new(0.0); vector_size]));
        unsafe {
            kernel_vector_index_assign::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                vector_size,
                ArrayArg::from_raw_parts(handle.clone(), 1),
            )
        }

        let actual = client.read_one_unchecked(handle);
        let actual = F::from_bytes(&actual);

        let mut expected = vec![F::new(0.0); vector_size];
        expected[0] = F::new(5.0);

        assert_eq!(&actual[..vector_size], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_vector_loop_unroll<F: Float, N: Size>(output: &mut Array<Vector<F, N>>) {
    if UNIT_POS == 0 {
        let mut vector = output[0];
        #[unroll]
        for k in 0..N::value() {
            vector[k] += F::cast_from(k);
        }
        output[0] = vector;
    }
}

pub fn test_vector_loop_unroll<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    for vector_size in client.io_optimized_vector_sizes(size_of::<F>()) {
        let handle = client.create_from_slice(F::as_bytes(&vec![F::new(0.0); vector_size]));
        unsafe {
            kernel_vector_loop_unroll::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                vector_size,
                ArrayArg::from_raw_parts(handle.clone(), 1),
            )
        }

        let actual = client.read_one_unchecked(handle);
        let actual = F::from_bytes(&actual);

        let expected = (0..vector_size as i64)
            .map(|x| F::from_int(x))
            .collect::<Vec<_>>();

        assert_eq!(&actual[..vector_size], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_vector_conditional<F: Float, N: Size>(
    input: &Array<Vector<F, N>>,
    flag: &Array<u32>,
    output: &mut Array<Vector<F, N>>,
) {
    let cond = flag[0] == u32::new(0);
    let vector = if cond { input[0] } else { input[1] };
    output[0] = vector;
}

pub fn test_vector_conditional<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let vector_size = 8usize;
    let mut input_data = vec![F::new(1.0); vector_size];
    input_data.extend(vec![F::new(2.0); vector_size]);
    let input = client.create_from_slice(F::as_bytes(&input_data));
    let output = client.create_from_slice(F::as_bytes(&vec![F::new(0.0); vector_size]));

    let flag = client.create_from_slice(u32::as_bytes(&[0u32]));
    unsafe {
        kernel_vector_conditional::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            vector_size,
            ArrayArg::from_raw_parts(input.clone(), 2),
            ArrayArg::from_raw_parts(flag, 1),
            ArrayArg::from_raw_parts(output.clone(), 1),
        )
    }
    let actual = client.read_one_unchecked(output.clone());
    let actual = F::from_bytes(&actual);
    assert_eq!(actual, vec![F::new(1.0); vector_size]);

    let flag = client.create_from_slice(u32::as_bytes(&[1u32]));
    unsafe {
        kernel_vector_conditional::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            vector_size,
            ArrayArg::from_raw_parts(input, 2),
            ArrayArg::from_raw_parts(flag, 1),
            ArrayArg::from_raw_parts(output.clone(), 1),
        )
    }
    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);
    assert_eq!(actual, vec![F::new(2.0); vector_size]);
}

#[cube(launch_unchecked)]
pub fn kernel_shared_memory<F: Float, N: Size>(output: &mut Array<Vector<F, N>>) {
    let mut smem1 = SharedMemory::<Vector<F, N>>::new(8usize);
    smem1[0] = Vector::new(F::new(42.0));
    output[0] = smem1[0];
}

pub fn test_shared_memory<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    for vector_size in client.io_optimized_vector_sizes(size_of::<F>()) {
        let output = client.create_from_slice(F::as_bytes(&vec![F::new(0.0); vector_size]));
        unsafe {
            kernel_shared_memory::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                vector_size,
                ArrayArg::from_raw_parts(output.clone(), vector_size),
            )
        }

        let actual = client.read_one_unchecked(output);
        let actual = F::from_bytes(&actual);

        assert_eq!(actual[0], F::new(42.0));
    }
}

macro_rules! impl_vector_comparison {
    ($cmp:ident, $expected:expr) => {
        ::paste::paste! {
            #[cube(launch)]
            pub fn [< kernel_vector_ $cmp >]<F: Float, N: Size>(
                lhs: &Array<Vector<F, N>>,
                rhs: &Array<Vector<F, N>>,
                output: &mut Array<Vector<u32, N>>,
            ) {
                if UNIT_POS == 0 {
                    output[0] = Vector::cast_from(lhs[0].$cmp(rhs[0]));
                }
            }

            pub fn [< test_vector_ $cmp >] <R: Runtime, F: Float + CubeElement>(
                client: ComputeClient<R>,
            ) {
                let lhs = client.create_from_slice(as_bytes![F: 0.0, 1.0, 2.0, 3.0]);
                let rhs = client.create_from_slice(as_bytes![F: 0.0, 2.0, 1.0, 3.0]);
                let output = client.empty(16);

                unsafe {
                    [< kernel_vector_ $cmp >]::launch::<F, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        4,
                        ArrayArg::from_raw_parts(lhs, 1),
                        ArrayArg::from_raw_parts(rhs, 1),
                        ArrayArg::from_raw_parts(output.clone(), 1),
                    )
                };

                let actual = client.read_one_unchecked(output);
                let actual = u32::from_bytes(&actual);

                assert_eq!(actual, $expected);
            }
        }
    };
}

impl_vector_comparison!(equal, [1, 0, 0, 1]);
impl_vector_comparison!(not_equal, [0, 1, 1, 0]);
impl_vector_comparison!(less_than, [0, 1, 0, 0]);
impl_vector_comparison!(greater_than, [0, 0, 1, 0]);
impl_vector_comparison!(less_equal, [1, 1, 0, 1]);
impl_vector_comparison!(greater_equal, [1, 0, 1, 1]);

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_vector {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_index() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_index::<TestRuntime, FloatType>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_index_assign() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_index_assign::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_loop_unroll() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_loop_unroll::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_conditional() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_conditional::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_shared_memory() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_shared_memory::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_equal::<TestRuntime, FloatType>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_not_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_not_equal::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_less_than() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_less_than::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_greater_than() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_greater_than::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_less_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_less_equal::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_vector_greater_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::vector::test_vector_greater_equal::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
