use crate::{self as cubecl, Feature, as_bytes, prelude::barrier::BarrierLevel};
use barrier::Barrier;
use cubecl::prelude::*;

#[cube(launch)]
pub fn async_copy_test<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    let barrier = Barrier::<F>::new(BarrierLevel::unit());
    let mut smem = SharedMemory::<F>::new_lined(1u32, 1u32);

    let source = input.slice(2, 3);
    let mut destination = smem.slice_mut(0, 1);

    barrier.memcpy_async(&source, &mut destination);

    barrier.arrive_and_wait();
    output[0] = smem[0];
}

pub fn test_async_copy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Barrier) {
        // We can't execute the test, skip.
        return;
    }

    let input = client
        .create(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0])
        .expect("Alloc failed");
    let output = client
        .empty(core::mem::size_of::<F>())
        .expect("Alloc failed");

    unsafe {
        async_copy_test::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 5, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(2.0));
}

#[cube(launch)]
fn one_load<F: Float>(lhs: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(4u32, 1u32);

    let barrier = Barrier::<F>::new(BarrierLevel::cube_manual(0u32));
    sync_cube();

    // Can't use lhs.to_slice() because then generated input_length will not exist
    barrier.memcpy_async(&lhs.slice(0u32, 4u32), &mut lhs_smem.to_slice_mut());

    barrier.arrive_and_wait();

    let start = UNIT_POS_X * 2u32;
    let end = start + 2u32;
    for i in start..end {
        output[i] = lhs_smem[i];
    }
}

#[cube(launch)]
fn two_loads<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] num_data: u32, // should be even
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);

    let barrier = Barrier::<F>::new(BarrierLevel::cube_manual(0u32));
    sync_cube();

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    barrier.memcpy_async(&lhs.slice(start, end), &mut lhs_smem.slice_mut(start, end));
    barrier.memcpy_async(&rhs.slice(start, end), &mut rhs_smem.slice_mut(start, end));

    barrier.arrive_and_wait();
    let mut dot = Line::cast_from(0u32);
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }

    output[UNIT_POS_X] = dot;
}

#[cube(launch)]
fn two_independent_loads<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] num_data: u32,
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);

    let barrier_0 = barrier::Barrier::new(BarrierLevel::cube_manual(0u32));
    let barrier_1 = barrier::Barrier::new(BarrierLevel::cube_manual(0u32));
    // At the Cube level, we must sync after barrier creation to make sure they
    // exist for all units
    sync_cube();

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    for i in start..end {
        lhs_smem[i] = Line::cast_from(0u32);
        rhs_smem[i] = Line::cast_from(0u32);
        output[i] = Line::cast_from(0u32);
    }

    barrier_0.memcpy_async(&lhs.slice(start, end), &mut lhs_smem.slice_mut(start, end));
    barrier_1.memcpy_async(&rhs.slice(start, end), &mut rhs_smem.slice_mut(start, end));

    let mut dot = Line::cast_from(0u32);

    barrier_0.arrive_and_wait();
    barrier_1.arrive_and_wait();
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }

    output[UNIT_POS_X] = dot;
}

pub fn test_memcpy_one_load<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Barrier) {
        // We can't execute the test, skip.
        return;
    }

    let lhs = client
        .create(as_bytes![F: 10., 11., 12., 13.])
        .expect("Alloc failed");
    let output = client
        .empty(4 * core::mem::size_of::<F>())
        .expect("Alloc failed");

    unsafe {
        one_load::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(2, 1, 1),
            TensorArg::from_raw_parts::<F>(&lhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&output, &[4, 1], &[4, 4], 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);
    let expected = [F::new(10.0), F::new(11.0), F::new(12.0), F::new(13.0)];

    assert_eq!(actual, expected);
}

pub fn test_memcpy_two_loads<R: Runtime, F: Float + CubeElement>(
    independent: bool,
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Barrier) {
        // We can't execute the test, skip.
        return;
    }

    let num_data = 4;
    let lhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();
    let rhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();

    let lhs = client.create(F::as_bytes(&lhs_data)).expect("Alloc failed");
    let rhs = client.create(F::as_bytes(&rhs_data)).expect("Alloc failed");
    let output = client
        .empty(2 * core::mem::size_of::<F>())
        .expect("Alloc failed");

    if independent {
        unsafe {
            two_independent_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(2, 1, 1),
                TensorArg::from_raw_parts::<F>(&lhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&rhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&output, &[1], &[2], 1),
                num_data as u32,
            )
        };
    } else {
        unsafe {
            two_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(2, 1, 1),
                TensorArg::from_raw_parts::<F>(&lhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&rhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&output, &[1], &[2], 1),
                num_data as u32,
            )
        };
    }

    let actual = client.read_one(output.binding());
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

        #[test]
        fn test_barrier_async_copy() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_async_copy::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_barrier_memcpy_async_one_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_one_load::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_barrier_memcpy_async_two_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_two_loads::<TestRuntime, FloatType>(
                false, client,
            );
        }

        #[test]
        fn test_barrier_memcpy_async_two_independent_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_two_loads::<TestRuntime, FloatType>(
                true, client,
            );
        }
    };
}
