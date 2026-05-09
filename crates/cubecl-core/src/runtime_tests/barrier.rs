use crate::{self as cubecl, as_bytes};
use alloc::vec::Vec;
use barrier::Barrier;
use cubecl::prelude::*;
use cubecl_ir::OpaqueType;
use num_traits::Zero;

#[cube(launch)]
pub fn async_copy_test<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut [Vector<F, N>]) {
    let barrier = Barrier::local();
    let mut smem = SharedMemory::<Vector<F, N>>::new(1usize);

    let source = &input[2..3];
    let destination = &mut smem[..1];

    barrier.memcpy_async(source, destination);

    barrier.arrive_and_wait();
    output[0] = smem[0];
}

pub fn test_async_copy<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    if !client
        .properties()
        .supports_type(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Unit))
    {
        // We can't execute the test, skip.
        return;
    }

    let input = client.create_from_slice(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        async_copy_test::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            1,
            BufferArg::from_raw_parts(input, 5),
            BufferArg::from_raw_parts(output.clone(), 1),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(2.0));
}

#[cube(launch)]
fn one_load<F: Float, N: Size>(lhs: &Tensor<Vector<F, N>>, output: &mut Tensor<Vector<F, N>>) {
    let mut lhs_smem = SharedMemory::<Vector<F, N>>::new(4usize);

    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_cube();

    // Can't use lhs.as_slice() because then generated input_length will not exist
    barrier.memcpy_async(&lhs[0..4], lhs_smem.as_mut_slice());

    barrier.arrive_and_wait();

    let start = UNIT_POS_X as usize * 2;
    let end = start + 2;
    for i in start..end {
        output[i] = lhs_smem[i];
    }
}

#[cube(launch)]
fn two_loads<F: Float, N: Size>(
    lhs: &Tensor<Vector<F, N>>,
    rhs: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    #[comptime] num_data: usize, // should be even
) {
    let mut lhs_smem = SharedMemory::<Vector<F, N>>::new(num_data);
    let mut rhs_smem = SharedMemory::<Vector<F, N>>::new(num_data);

    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_cube();

    let start = UNIT_POS_X as usize * num_data / 2;
    let end = start + num_data / 2;

    barrier.memcpy_async(&lhs[start..end], &mut lhs_smem[start..end]);
    barrier.memcpy_async(&rhs[start..end], &mut rhs_smem[start..end]);

    barrier.arrive_and_wait();
    let mut dot = Vector::default();
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }

    output[UNIT_POS_X as usize] = dot;
}

#[cube(launch)]
fn two_independent_loads<F: Float, N: Size>(
    lhs: &Tensor<Vector<F, N>>,
    rhs: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    #[comptime] num_data: usize,
) {
    let mut lhs_smem = SharedMemory::<Vector<F, N>>::new(num_data);
    let mut rhs_smem = SharedMemory::<Vector<F, N>>::new(num_data);

    let barrier_0 = barrier::Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    let barrier_1 = barrier::Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    // At the Cube level, we must sync after barrier creation to make sure they
    // exist for all units
    sync_cube();

    let start = UNIT_POS_X as usize * num_data / 2;
    let end = start + num_data / 2;

    for i in start..end {
        lhs_smem[i] = Vector::zeroed();
        rhs_smem[i] = Vector::zeroed();
        output[i] = Vector::zeroed();
    }

    barrier_0.memcpy_async(&lhs[start..end], &mut lhs_smem[start..end]);
    barrier_1.memcpy_async(&rhs[start..end], &mut rhs_smem[start..end]);

    let mut dot = Vector::zero();

    barrier_0.arrive_and_wait();
    barrier_1.arrive_and_wait();
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }

    output[UNIT_POS_X as usize] = dot;
}

pub fn test_memcpy_one_load<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    if !client
        .properties()
        .supports_type(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Cube))
    {
        // We can't execute the test, skip.
        return;
    }

    let lhs = client.create_from_slice(as_bytes![F: 10., 11., 12., 13.]);
    let output = client.empty(4 * core::mem::size_of::<F>());

    unsafe {
        one_load::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(2),
            1,
            TensorArg::from_raw_parts(lhs, [4, 1].into(), [4, 4].into()),
            TensorArg::from_raw_parts(output.clone(), [4, 1].into(), [4, 4].into()),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);
    let expected = [F::new(10.0), F::new(11.0), F::new(12.0), F::new(13.0)];

    assert_eq!(actual, expected);
}

pub fn test_memcpy_two_loads<R: Runtime, F: Float + CubeElement>(
    independent: bool,
    client: ComputeClient<R>,
) {
    if !client
        .properties()
        .supports_type(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Cube))
    {
        // We can't execute the test, skip.
        return;
    }

    let num_data = 4;
    let lhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();
    let rhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();

    let lhs = client.create_from_slice(F::as_bytes(&lhs_data));
    let rhs = client.create_from_slice(F::as_bytes(&rhs_data));
    let output = client.empty(2 * core::mem::size_of::<F>());

    if independent {
        unsafe {
            two_independent_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                1,
                TensorArg::from_raw_parts(lhs, [1].into(), [num_data].into()),
                TensorArg::from_raw_parts(rhs, [1].into(), [num_data].into()),
                TensorArg::from_raw_parts(output.clone(), [1].into(), [2].into()),
                num_data,
            )
        };
    } else {
        unsafe {
            two_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                1,
                TensorArg::from_raw_parts(lhs, [1].into(), [num_data].into()),
                TensorArg::from_raw_parts(rhs, [1].into(), [num_data].into()),
                TensorArg::from_raw_parts(output.clone(), [1].into(), [2].into()),
                num_data,
            )
        };
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    let middle = num_data / 2;
    let expected = [
        dot(&lhs_data[..middle], &rhs_data[..middle]),
        dot(&lhs_data[middle..], &rhs_data[middle..]),
    ];

    assert_eq!(actual, expected);
}

fn dot<F: Float>(vec1: &[F], vec2: &[F]) -> F {
    let mut sum = F::from_int(0);
    for i in 0..vec1.len() {
        sum += vec1[i] * vec2[i];
    }
    sum
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_barrier {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_barrier_async_copy() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_async_copy::<TestRuntime, FloatType>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_barrier_memcpy_async_one_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_one_load::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_barrier_memcpy_async_two_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_two_loads::<TestRuntime, FloatType>(
                false, client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_barrier_memcpy_async_two_independent_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_two_loads::<TestRuntime, FloatType>(
                true, client,
            );
        }
    };
}
