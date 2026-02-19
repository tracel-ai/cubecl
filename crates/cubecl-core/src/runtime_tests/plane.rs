use crate::runtime_tests::binary::assert_equals_approx;
use crate::{self as cubecl};
use alloc::{fmt::Display, vec, vec::Vec};
use cubecl::prelude::*;
use cubecl_ir::features::Plane;

#[cube(launch)]
pub fn kernel_sum<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_sum(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_inclusive_sum<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_inclusive_sum(val);

    output[UNIT_POS as usize] = val2;
}

#[cube(launch)]
pub fn kernel_exclusive_sum<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_exclusive_sum(val);

    output[UNIT_POS as usize] = val2;
}

#[cube(launch)]
pub fn kernel_prod<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_prod(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_inclusive_prod<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_inclusive_prod(val);

    output[UNIT_POS as usize] = val2;
}

#[cube(launch)]
pub fn kernel_exclusive_prod<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_exclusive_prod(val);

    output[UNIT_POS as usize] = val2;
}

#[cube(launch)]
pub fn kernel_max<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_max(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_min<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_min(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_all<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_all(val < F::new(5.0));
    output[UNIT_POS as usize] = F::cast_from(val2);
}

#[cube(launch)]
pub fn kernel_any<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_any(val > F::new(5.0));
    output[UNIT_POS as usize] = F::cast_from(val2);
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
    let val = output[UNIT_POS as usize];
    let val2 = plane_broadcast(val, 2u32);

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

#[cube(launch)]
pub fn kernel_shuffle<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_shuffle(val, 0); // All lanes read from lane 0

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_shuffle_xor<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_shuffle_xor(val, 1);

    output[UNIT_POS as usize] = val2;
}

#[cube(launch)]
pub fn kernel_shuffle_up<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_shuffle_up(val, 1);

    output[UNIT_POS as usize] = val2;
}

#[cube(launch)]
pub fn kernel_shuffle_down<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS as usize];
    let val2 = plane_shuffle_down(val, 1);

    output[UNIT_POS as usize] = val2;
}

pub fn test_plane_sum<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        for v in 0..vectorization {
            expected[v] += input[v + k * vectorization];
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_inclusive_sum<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        let offs_out = k * vectorization;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization;
            for v in 0..vectorization {
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_exclusive_sum<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = vec![0.0; input.len()];

    for k in 1..plane_size as usize {
        let offs_out = k * vectorization;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization;
            for v in 0..vectorization {
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_prod<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
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
        for v in 0..vectorization {
            expected[v] *= input[v + k * vectorization];
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_inclusive_prod<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
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
        let offs_out = k * vectorization;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization;
            for v in 0..vectorization {
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_exclusive_prod<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
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
        let offs_out = k * vectorization;
        for k_1 in 0..k {
            let offs_in = k_1 * vectorization;
            for v in 0..vectorization {
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_max<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    input[16] = 999.0; // I don't want the max to always be the last element.

    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        for v in 0..vectorization {
            expected[v] = expected[v].max(input[v + k * vectorization]);
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_min<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    input[16] = -5.0; // I don't want the min to always be the first element.

    let mut expected = input.clone();

    for k in 1..plane_size as usize {
        for v in 0..vectorization {
            expected[v] = expected[v].min(input[v + k * vectorization]);
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_all<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| (x % 5) as f32) // the predicate is x < 5 which is always satisfied at this step.
        .collect();

    for k in 0..vectorization {
        if k % 2 == 0 {
            input[4 * vectorization + k] = 10.0; // Make all even batches false by setting an element to be > 5.
        }
    }

    let expected: Vec<f32> = (0..input.len())
        .map(|x| ((x % vectorization) % 2) as f32)
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_any<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let mut input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| (x % 5) as f32) // the predicate is x > 5 which is never satisfied at this step.
        .collect();

    for k in 0..vectorization {
        if k % 2 == 0 {
            input[4 * vectorization + k] = 10.0; // Make all even batches true by setting an element to be > 5.
        }
    }

    let expected: Vec<f32> = (0..input.len())
        .map(|x| (1 - (x % vectorization) % 2) as f32)
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_ballot<TestRuntime: Runtime>(client: ComputeClient<TestRuntime>) {
    if !client.properties().features.plane.contains(Plane::Ops) {
        // Can't execute the test.
        return;
    }

    let handle = client.empty(size_of::<u32>() * 4);
    let (shape, strides) = ([1], [1]);

    unsafe {
        kernel_ballot::launch::<TestRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            TensorArg::from_raw_parts::<u32>(&handle, strides.into(), shape.into(), 4),
        )
    }

    let expected = [0b1111_1111, 0, 0, 0];
    let actual = client.read_one_unchecked(handle);

    assert_eq!(u32::from_bytes(&actual), &expected);
}

pub fn test_plane_elect<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let input = vec![0.0; plane_size as usize * vectorization];

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
                CubeDim::new_1d(plane_size),
                handle,
            );
        },
    );
}

pub fn test_plane_broadcast<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    for v in 0..vectorization {
        expected[v] = input[v + 2 * vectorization];
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
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_shuffle<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = client.properties().hardware.plane_size_max;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    // All lanes read from lane 0 (same as broadcast(value, 0))
    expected[..vectorization].copy_from_slice(&input[..vectorization]);

    let input: Vec<F> = input.into_iter().map(|x| F::new(x)).collect();
    let expected: Vec<F> = expected.into_iter().map(|x| F::new(x)).collect();

    test_plane_operation::<TestRuntime, F, _>(
        &input,
        &expected,
        vectorization,
        client.clone(),
        |cube_count, handle| {
            kernel_shuffle::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_shuffle_xor<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    // XOR with mask=1: lane i gets value from lane (i XOR 1)
    // So lane 0 <-> 1, lane 2 <-> 3, lane 4 <-> 5, etc.
    for lane in 0..plane_size as usize {
        let partner = lane ^ 1;
        for v in 0..vectorization {
            expected[lane * vectorization + v] = input[partner * vectorization + v];
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
            kernel_shuffle_xor::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_shuffle_up<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = 32;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    // Shuffle up with delta=1: lane i gets value from lane (i - 1)
    // Lane 0 stays the same, lanes 1..31 shift down
    for lane in 1..plane_size as usize {
        for v in 0..vectorization {
            expected[lane * vectorization + v] = input[(lane - 1) * vectorization + v];
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
            kernel_shuffle_up::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new_1d(plane_size),
                handle,
            )
        },
    );
}

pub fn test_plane_shuffle_down<
    TestRuntime: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: ComputeClient<TestRuntime>,
    vectorization: LineSize,
) {
    let plane_size = client.properties().hardware.plane_size_max;
    let input: Vec<f32> = (0..plane_size * vectorization as u32)
        .map(|x| x as f32)
        .collect();
    let mut expected = input.clone();

    // Shuffle down with delta=1: lane i gets value from lane (i + 1)
    // Lanes 0..30 shift up, lane 31 stays the same
    for lane in 0..(plane_size - 1) as usize {
        for v in 0..vectorization {
            expected[lane * vectorization + v] = input[(lane + 1) * vectorization + v];
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
            kernel_shuffle_down::launch::<F, TestRuntime>(
                &client,
                cube_count,
                CubeDim::new_1d(plane_size),
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
    line_size: LineSize,
    client: ComputeClient<TestRuntime>,
    launch: Launch,
) where
    Launch: Fn(CubeCount, TensorArg<'_, TestRuntime>),
{
    if !client.properties().features.plane.contains(Plane::Ops) {
        // Can't execute the test.
        return;
    }

    let handle = client.create_from_slice(F::as_bytes(input));
    let (shape, strides) = ([input.len()], [1]);

    unsafe {
        launch(
            CubeCount::Static(1, 1, 1),
            TensorArg::from_raw_parts::<F>(&handle, strides.into(), shape.into(), line_size),
        )
    }

    assert_equals_approx::<TestRuntime, F>(&client, handle, expected, 1e-5);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_plane {
    () => {
        use super::*;

        fn impl_test_plane_sum(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_sum::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_sum_vec1() {
            impl_test_plane_sum(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_sum_vec2() {
            impl_test_plane_sum(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_sum_vec4() {
            impl_test_plane_sum(4);
        }

        fn impl_test_plane_inclusive_sum(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_inclusive_sum::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_inclusive_sum_vec1() {
            impl_test_plane_inclusive_sum(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_inclusive_sum_vec2() {
            impl_test_plane_inclusive_sum(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_inclusive_sum_vec4() {
            impl_test_plane_inclusive_sum(4);
        }

        fn impl_test_plane_exclusive_sum(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_exclusive_sum::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_exclusive_sum_vec1() {
            impl_test_plane_exclusive_sum(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_exclusive_sum_vec2() {
            impl_test_plane_exclusive_sum(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_exclusive_sum_vec4() {
            impl_test_plane_exclusive_sum(4);
        }

        fn impl_test_plane_prod(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_prod::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_prod_vec1() {
            impl_test_plane_prod(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_prod_vec2() {
            impl_test_plane_prod(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_prod_vec4() {
            impl_test_plane_prod(4);
        }

        fn impl_test_plane_inclusive_prod(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_inclusive_prod::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_inclusive_prod_vec1() {
            impl_test_plane_inclusive_prod(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_inclusive_prod_vec2() {
            impl_test_plane_inclusive_prod(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_inclusive_prod_vec4() {
            impl_test_plane_inclusive_prod(4);
        }

        fn impl_test_plane_exclusive_prod(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_exclusive_prod::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_exclusive_prod_vec1() {
            impl_test_plane_exclusive_prod(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_exclusive_prod_vec2() {
            impl_test_plane_exclusive_prod(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_exclusive_prod_vec4() {
            impl_test_plane_exclusive_prod(4);
        }

        fn impl_test_plane_max(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_max::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_max_vec1() {
            impl_test_plane_max(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_max_vec2() {
            impl_test_plane_max(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_max_vec4() {
            impl_test_plane_max(4);
        }

        fn impl_test_plane_min(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_min::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_min_vec1() {
            impl_test_plane_min(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_min_vec2() {
            impl_test_plane_min(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_min_vec4() {
            impl_test_plane_min(4);
        }

        fn impl_test_plane_all(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_all::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_all_vec1() {
            impl_test_plane_all(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_all_vec2() {
            impl_test_plane_all(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_all_vec4() {
            impl_test_plane_all(4);
        }

        fn impl_test_plane_any(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_any::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_any_vec1() {
            impl_test_plane_any(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_any_vec2() {
            impl_test_plane_any(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_any_vec4() {
            impl_test_plane_any(4);
        }

        fn impl_test_plane_elect(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_elect::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[ignore]
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_elect_vec1() {
            impl_test_plane_elect(1);
        }
        #[ignore]
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_elect_vec2() {
            impl_test_plane_elect(2);
        }
        #[ignore]
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_elect_vec4() {
            impl_test_plane_elect(4);
        }

        fn impl_test_plane_broadcast(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_broadcast::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_broadcast_vec1() {
            impl_test_plane_broadcast(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_broadcast_vec2() {
            impl_test_plane_broadcast(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_broadcast_vec4() {
            impl_test_plane_broadcast(4);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_plane_ballot() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_ballot::<TestRuntime>(client.clone());
        }

        fn impl_test_plane_shuffle(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_shuffle::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_vec1() {
            impl_test_plane_shuffle(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_vec2() {
            impl_test_plane_shuffle(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_vec4() {
            impl_test_plane_shuffle(4);
        }

        fn impl_test_plane_shuffle_xor(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_shuffle_xor::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_xor_vec1() {
            impl_test_plane_shuffle_xor(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_xor_vec2() {
            impl_test_plane_shuffle_xor(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_xor_vec4() {
            impl_test_plane_shuffle_xor(4);
        }

        fn impl_test_plane_shuffle_up(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_shuffle_up::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_up_vec1() {
            impl_test_plane_shuffle_up(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_up_vec2() {
            impl_test_plane_shuffle_up(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_up_vec4() {
            impl_test_plane_shuffle_up(4);
        }

        fn impl_test_plane_shuffle_down(vectorization: LineSize) {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::plane::test_plane_shuffle_down::<TestRuntime, FloatType>(
                client.clone(),
                vectorization,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_down_vec1() {
            impl_test_plane_shuffle_down(1);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_down_vec2() {
            impl_test_plane_shuffle_down(2);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_plane_shuffle_down_vec4() {
            impl_test_plane_shuffle_down(4);
        }
    };
}
