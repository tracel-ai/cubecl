use crate::Feature;
use crate::{self as cubecl, as_type};
use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_sum<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = subcube_sum(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_prod<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = subcube_prod(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_max<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = subcube_max(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_min<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = subcube_min(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_all<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = subcube_all(val < F::new(5.0));
    output[UNIT_POS] = F::cast_from(val2);
}

#[cube(launch)]
pub fn kernel_any<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = subcube_any(val < F::new(5.0));
    output[UNIT_POS] = F::cast_from(val2);
}

#[cube(launch)]
pub fn kernel_elect<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let elect = subcube_elect();
    if elect {
        output[4] += val;
    }
}

#[cube(launch)]
pub fn kernel_broadcast<F: Float>(output: &mut Tensor<F>) {
    let val = output[UNIT_POS];
    let val2 = subcube_broadcast(val, 2);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

pub fn test_subcube_sum<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 4.0, 5.0, 7.0, 1.0],
        as_type![F: 17.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_count: CubeCount, cube_dim, handle| {
            kernel_sum::launch::<F, TestRuntime>(&client, cube_count, cube_dim, handle)
        },
    );
}

pub fn test_subcube_prod<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 4.0, 5.0, 7.0, 1.0],
        as_type![F: 140.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_prod::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}
pub fn test_subcube_max<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 4.0, 5.0, 7.0, 1.0],
        as_type![F: 7.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_max::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_min<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 4.0, 5.0, 7.0, 1.0],
        as_type![F: 1.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_min::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_all<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 2.0, 1.0, -6.0, 3.0],
        as_type![F: 1.0, 1.0, 1.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_all::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 2.0, -10.0, 2.0, 7.0],
        as_type![F: 0.0, 0.0, 0.0, 0.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_all::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_any<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 2.0, 1.0, -6.0, 3.0],
        as_type![F: 1.0, 1.0, 1.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_any::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 8.0, 10.0, 20.0, 7.0],
        as_type![F: 0.0, 0.0, 0.0, 0.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_any::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_elect<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 2.0, 1.0, -6.0, 3.0],
        as_type![F: 2.0, 1.0, 1.0, 5.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_elect::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_broadcast<TestRuntime: Runtime, F: Float + CubeElement>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, F, _>(
        as_type![F: 2.0, 1.0, -6.0, 3.0],
        as_type![F: -6.0, 1.0, -6.0, 3.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_broadcast::launch::<F, TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

fn test_subcube_operation<TestRuntime: Runtime, F: Float + CubeElement, Launch>(
    input: &[F],
    expected: &[F],
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    launch: Launch,
) where
    Launch: Fn(CubeCount, CubeDim, TensorArg<'_, TestRuntime>),
{
    if !client.properties().feature_enabled(Feature::Subcube) {
        // Can't execute the test.
        return;
    }

    let handle = client.create(F::as_bytes(input));
    let (shape, strides) = ([input.len()], [1]);

    unsafe {
        launch(
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32, 1, 1),
            TensorArg::from_raw_parts::<F>(&handle, &strides, &shape, 1),
        );
    }

    let actual = client.read(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual, expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_subcube {
    () => {
        use super::*;

        #[test]
        fn test_subcube_sum() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_sum::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_subcube_prod() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_prod::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_subcube_max() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_max::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_subcube_min() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_min::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_subcube_all() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_all::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_subcube_any() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_any::<TestRuntime, FloatType>(client);
        }

        #[ignore]
        #[test]
        fn test_subcube_elect() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_elect::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_subcube_broadcast() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_broadcast::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
