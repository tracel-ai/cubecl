use crate as cubecl;
use crate::Feature;
use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_sum(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = subcube_sum(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_prod(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = subcube_prod(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_max(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = subcube_max(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_min(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = subcube_min(val);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

#[cube(launch)]
pub fn kernel_all(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = subcube_all(val < 5.0);
    output[UNIT_POS] = val2 as u32 as f32;
}

#[cube(launch)]
pub fn kernel_any(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = subcube_any(val < 5.0);
    output[UNIT_POS] = val2 as u32 as f32;
}

#[cube(launch)]
pub fn kernel_elect(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let elect = subcube_elect();
    if elect {
        output[4] += val;
    }
}

#[cube(launch)]
pub fn kernel_broadcast(output: &mut Tensor<f32>) {
    let val = output[UNIT_POS];
    let val2 = subcube_broadcast(val, 2);

    if UNIT_POS == 0 {
        output[0] = val2;
    }
}

pub fn test_subcube_sum<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[4.0, 5.0, 7.0, 1.0],
        &[17.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_count: CubeCount, cube_dim, handle| {
            kernel_sum::launch::<TestRuntime>(&client, cube_count, cube_dim, handle)
        },
    );
}

pub fn test_subcube_prod<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[4.0, 5.0, 7.0, 1.0],
        &[140.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_prod::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}
pub fn test_subcube_max<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[4.0, 5.0, 7.0, 1.0],
        &[7.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_max::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_min<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[4.0, 5.0, 7.0, 1.0],
        &[1.0, 5.0, 7.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_min::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_all<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[2.0, 1.0, -6.0, 3.0],
        &[1.0, 1.0, 1.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_all::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
    test_subcube_operation::<TestRuntime, _>(
        &[2.0, -10.0, 2.0, 7.0],
        &[0.0, 0.0, 0.0, 0.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_all::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_any<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[2.0, 1.0, -6.0, 3.0],
        &[1.0, 1.0, 1.0, 1.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_any::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
    test_subcube_operation::<TestRuntime, _>(
        &[8.0, 10.0, 20.0, 7.0],
        &[0.0, 0.0, 0.0, 0.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_any::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_elect<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[2.0, 1.0, -6.0, 3.0],
        &[2.0, 1.0, 1.0, 5.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_elect::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

pub fn test_subcube_broadcast<TestRuntime: Runtime>(
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
) {
    test_subcube_operation::<TestRuntime, _>(
        &[2.0, 1.0, -6.0, 3.0],
        &[-6.0, 1.0, -6.0, 3.0],
        client.clone(),
        |cube_dim, settings, handle| {
            kernel_broadcast::launch::<TestRuntime>(&client, cube_dim, settings, handle)
        },
    );
}

fn test_subcube_operation<TestRuntime: Runtime, Launch>(
    input: &[f32],
    expected: &[f32],
    client: ComputeClient<TestRuntime::Server, TestRuntime::Channel>,
    launch: Launch,
) where
    Launch: Fn(CubeCount, CubeDim, TensorArg<'_, TestRuntime>),
{
    if !client.properties().feature_enabled(Feature::Subcube) {
        // Can't execute the test.
        return;
    }

    let handle = client.create(f32::as_bytes(input));
    let (shape, strides) = ([input.len()], [1]);

    unsafe {
        launch(
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32, 1, 1),
            TensorArg::from_raw_parts(&handle, &strides, &shape, 1),
        );
    }

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

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
            cubecl_core::runtime_tests::subcube::test_subcube_sum::<TestRuntime>(client);
        }

        #[test]
        fn test_subcube_prod() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_prod::<TestRuntime>(client);
        }

        #[test]
        fn test_subcube_max() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_max::<TestRuntime>(client);
        }

        #[test]
        fn test_subcube_min() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_min::<TestRuntime>(client);
        }

        #[test]
        fn test_subcube_all() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_all::<TestRuntime>(client);
        }

        #[test]
        fn test_subcube_any() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_any::<TestRuntime>(client);
        }

        #[ignore]
        #[test]
        fn test_subcube_elect() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_elect::<TestRuntime>(client);
        }

        #[test]
        fn test_subcube_broadcast() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::subcube::test_subcube_broadcast::<TestRuntime>(client);
        }
    };
}
