use std::fmt::Display;

use crate::{self as cubecl};
use crate::{Feature, runtime_tests::binary::assert_equals_approx};
use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_sum<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_sum(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_inclusive_sum<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_inclusive_sum(val);

    output[UNIT_POS] = val2;
}

#[cube(launch)]
pub fn kernel_exclusive_sum<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_exclusive_sum(val);

    output[UNIT_POS] = val2;
}

#[cube(launch)]
pub fn kernel_prod<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_prod(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_inclusive_prod<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_inclusive_prod(val);

    output[UNIT_POS] = val2;
}

#[cube(launch)]
pub fn kernel_exclusive_prod<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_exclusive_prod(val);

    output[UNIT_POS] = val2;
}

#[cube(launch)]
pub fn kernel_max<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_max(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_min<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_min(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_all<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_all(val < F::new(5.0));
    output[UNIT_POS] = F::cast_from(val2);
}

#[cube(launch)]
pub fn kernel_any<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_any(val > F::new(5.0));
    output[UNIT_POS] = F::cast_from(val2);
}

#[cube(launch)]
pub fn kernel_elect<F: Float>(output: &mut Tensor<F>) {
    let elect = plane_elect();
    if elect {
        output[20] += F::new(1.0);
    }
}

#[cube(launch)]
pub fn kernel_broadcast<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = plane_broadcast(val, 2);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_ballot(output: &mut Tensor<Line<u32>>) {
    let val2 = plane_ballot(UNIT_POS < 8);

    if UNIT_POS == 0 {
        output[0] = Line::cast_from(val2);
    }
}

pub fn test_plane_sum<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        for v in 0..vectorization as usize {
            expected[v] += input[v + k * vectorization as usize];
        }
    }
    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_sum::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_inclusive_sum<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        let offs_out = k * vectorization as usize;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization as usize;
            for v in 0..vectorization as usize {
                expected[v + offs_out] += input[v + offs_in];
            }
        }
    }

    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_inclusive_sum::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_exclusive_sum<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = vec![0.0; input.len()];

    for k in 1..plane_size as usize {
        let offs_out = k * vectorization as usize;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization as usize;
            for v in 0..vectorization as usize {
                expected[v + offs_out] += input[v + offs_in];
            }
        }
    }

    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_exclusive_sum::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_prod<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| match x % 3 {
            0 => 0.5,
            1 => 1.25,
            2 => 1.75,
            _ => unreachable!(),
        }) // keep the values relatively close to 1 to avoid overflow.
        .collect();
    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        for v in 0..vectorization as usize {
            expected[v] *= input[v + k * vectorization as usize];
        }
    }
    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_prod::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_inclusive_prod<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| match x % 3 {
            0 => 0.5,
            1 => 1.25,
            2 => 1.75,
            _ => unreachable!(),
        }) // keep the values relatively close to 1 to avoid overflow.
        .collect();
    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        let offs_out = k * vectorization as usize;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization as usize;
            for v in 0..vectorization as usize {
                expected[v + offs_out] *= input[v + offs_in];
            }
        }
    }
    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_inclusive_prod::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_exclusive_prod<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| match x % 3 {
            0 => 0.5,
            1 => 1.25,
            2 => 1.75,
            _ => unreachable!(),
        }) // keep the values relatively close to 1 to avoid overflow.
        .collect();
    let mut expected = vec![1.0; input.len()];

    for k in 1..plane_size as usize {
        let offs_out = k * vectorization as usize;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization as usize;
            for v in 0..vectorization as usize {
                expected[v + offs_out] *= input[v + offs_in];
            }
        }
    }
    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_exclusive_prod::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_max<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    input[16] = 999.0; // I don't want the max to always be the last element.

    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        for v in 0..vectorization as usize {
            expected[v] = expected[v].max(input[v + k * vectorization as usize]);
        }
    }
    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_max::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_min<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    input[16] = -5.0; // I don't want the min to always be the first element.

    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        for v in 0..vectorization as usize {
            expected[v] = expected[v].min(input[v + k * vectorization as usize]);
        }
    }
    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_min::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_all<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| (x % 5) as f32) // the predicate is x < 5 which is always satisfied at this step.
        .collect();

    for k in 0..vectorization as usize {
        if k % 2 == 0 {
            input[4 * vectorization as usize + k] = 10.0; // Make all even batches false by setting an element to be > 5.
        }
    }

    let expected: Vec<f32> = (0..input.len())
        .map(|x| ((x % vectorization as usize) % 2) as f32)
        .collect();

    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_all::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_any<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| (x % 5) as f32) // the predicate is x > 5 which is never satisfied at this step.
        .collect();

    for k in 0..vectorization as usize {
        if k % 2 == 0 {
            input[4 * vectorization as usize + k] = 10.0; // Make all even batches true by setting an element to be > 5.
        }
    }

    let expected: Vec<f32> = (0..input.len())
        .map(|x| (1 - (x % vectorization as usize) % 2) as f32)
        .collect();

    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_any::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_ballot<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Plane) {
        // Can't execute the test.
        return;
    }

    let handle = client.empty(size_of::<u32>() * 4).expect("Alloc failed");
    let (shape, strides) = ([1], [1]);

    unsafe {
        kernel_ballot::launch::<TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(32, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle, &strides, &shape, 4),
        );
    }

    let expected = [0b1111_1111, 0, 0, 0];
    let actual = client.read_one(handle.binding());

    assert_eq!(u32::from_bytes(&actual), &expected);
}

pub fn test_plane_elect<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input = vec![0.0; plane_size as usize * vectorization as usize];

    let mut expected = input.clone();
    expected[20] = vectorization as f32;

    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_any::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

pub fn test_plane_broadcast<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    vectorization: u8,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    for v in 0..vectorization as usize {
        expected[v] = input[v + 2 * vectorization as usize];
    }
    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_broadcast::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new(plane_size, 1, 1),
                handle,
            )
        },
    );
}

fn test_plane_operation<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
    Launch,
>(
    input: &[F],
    expected: &[F],
    vectorization: u8,
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    launch: Launch,
) where
    Launch: Fn(CubeCount, TensorArg<'_, TestRuntime>),
{
    if !client.properties().feature_enabled(Feature::Plane) {
        // Can't execute the test.
        return;
    }

    let handle = client.create(F::as_bytes(input)).expect("Alloc failed");
    let (shape, strides) = ([input.len()], [1]);

    unsafe {
        launch(
            CubeCount::Static(1, 1, 1),
            TensorArg::from_raw_parts::<F>(&handle, &strides, &shape, vectorization),
        );
    }

    assert_equals_approx::<TestRuntime, F>(&client, handle, expected, 1e-5);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_plane {
    () => {
        use super::*;

        fn impl_test_plane_sum(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_sum::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_sum_vec1() {
            impl_test_plane_sum(1);
        }
        #[test]
        fn test_plane_sum_vec2() {
            impl_test_plane_sum(2);
        }
        #[test]
        fn test_plane_sum_vec4() {
            impl_test_plane_sum(4);
        }

        fn impl_test_plane_inclusive_sum(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_inclusive_sum::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_inclusive_sum_vec1() {
            impl_test_plane_inclusive_sum(1);
        }
        #[test]
        fn test_plane_inclusive_sum_vec2() {
            impl_test_plane_inclusive_sum(2);
        }
        #[test]
        fn test_plane_inclusive_sum_vec4() {
            impl_test_plane_inclusive_sum(4);
        }

        fn impl_test_plane_exclusive_sum(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_exclusive_sum::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_exclusive_sum_vec1() {
            impl_test_plane_exclusive_sum(1);
        }
        #[test]
        fn test_plane_exclusive_sum_vec2() {
            impl_test_plane_exclusive_sum(2);
        }
        #[test]
        fn test_plane_exclusive_sum_vec4() {
            impl_test_plane_exclusive_sum(4);
        }

        fn impl_test_plane_prod(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_prod::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_prod_vec1() {
            impl_test_plane_prod(1);
        }
        #[test]
        fn test_plane_prod_vec2() {
            impl_test_plane_prod(2);
        }
        #[test]
        fn test_plane_prod_vec4() {
            impl_test_plane_prod(4);
        }

        fn impl_test_plane_inclusive_prod(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_inclusive_prod::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_inclusive_prod_vec1() {
            impl_test_plane_inclusive_prod(1);
        }
        #[test]
        fn test_plane_inclusive_prod_vec2() {
            impl_test_plane_inclusive_prod(2);
        }
        #[test]
        fn test_plane_inclusive_prod_vec4() {
            impl_test_plane_inclusive_prod(4);
        }

        fn impl_test_plane_exclusive_prod(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_exclusive_prod::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_exclusive_prod_vec1() {
            impl_test_plane_exclusive_prod(1);
        }
        #[test]
        fn test_plane_exclusive_prod_vec2() {
            impl_test_plane_exclusive_prod(2);
        }
        #[test]
        fn test_plane_exclusive_prod_vec4() {
            impl_test_plane_exclusive_prod(4);
        }

        fn impl_test_plane_max(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_max::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_max_vec1() {
            impl_test_plane_max(1);
        }
        #[test]
        fn test_plane_max_vec2() {
            impl_test_plane_max(2);
        }
        #[test]
        fn test_plane_max_vec4() {
            impl_test_plane_max(4);
        }

        fn impl_test_plane_min(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_min::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_min_vec1() {
            impl_test_plane_min(1);
        }
        #[test]
        fn test_plane_min_vec2() {
            impl_test_plane_min(2);
        }
        #[test]
        fn test_plane_min_vec4() {
            impl_test_plane_min(4);
        }

        fn impl_test_plane_all(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_all::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_all_vec1() {
            impl_test_plane_all(1);
        }
        #[test]
        fn test_plane_all_vec2() {
            impl_test_plane_all(2);
        }
        #[test]
        fn test_plane_all_vec4() {
            impl_test_plane_all(4);
        }

        fn impl_test_plane_any(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_any::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_any_vec1() {
            impl_test_plane_any(1);
        }
        #[test]
        fn test_plane_any_vec2() {
            impl_test_plane_any(2);
        }
        #[test]
        fn test_plane_any_vec4() {
            impl_test_plane_any(4);
        }

        fn impl_test_plane_elect(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_elect::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[ignore]
        #[test]
        fn test_plane_elect_vec1() {
            impl_test_plane_elect(1);
        }
        #[ignore]
        #[test]
        fn test_plane_elect_vec2() {
            impl_test_plane_elect(2);
        }
        #[ignore]
        #[test]
        fn test_plane_elect_vec4() {
            impl_test_plane_elect(4);
        }

        fn impl_test_plane_broadcast(vectorization: u8) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_broadcast::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[test]
        fn test_plane_broadcast_vec1() {
            impl_test_plane_broadcast(1);
        }
        #[test]
        fn test_plane_broadcast_vec2() {
            impl_test_plane_broadcast(2);
        }
        #[test]
        fn test_plane_broadcast_vec4() {
            impl_test_plane_broadcast(4);
        }

        #[test]
        fn test_plane_ballot() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_ballot::<TestRuntime>(client.clone());
        }
    };
}
