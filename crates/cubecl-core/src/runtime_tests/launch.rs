use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[derive(CubeLaunch, CubeType)]
pub struct ComptimeTag {
    array: Array<f32>,
    #[cube(comptime)]
    tag: String,
}

#[cube(launch)]
pub fn kernel_with_comptime_tag(output: &mut ComptimeTag) {
    if UNIT_POS == 0 {
        if comptime![&output.tag == "zero"] {
            output.array[0] = f32::new(0.0);
        } else {
            output.array[0] = f32::new(1.0);
        }
    }
}

#[cube(launch)]
pub fn kernel_with_generics<F: Float>(output: &mut Array<F>) {
    if UNIT_POS == 0 {
        output[0] = F::new(5.0);
    }
}

#[cube(launch)]
pub fn kernel_without_generics(output: &mut Array<f32>) {
    if UNIT_POS == 0 {
        output[0] = 5.0;
    }
}

#[cube(launch)]
pub fn kernel_with_max_shared(
    output: &mut Array<u32>,
    #[comptime] shared_size_1: u32,
    #[comptime] shared_size_2: u32,
) {
    let mut shared_1 = SharedMemory::<u32>::new(shared_size_1);
    let mut shared_2 = SharedMemory::<u32>::new(shared_size_2);
    if UNIT_POS < 8 {
        shared_1[shared_size_1 - UNIT_POS - 1] = output[UNIT_POS];
        shared_2[shared_size_2 - UNIT_POS - 1] = output[8 - UNIT_POS];
    }
    sync_cube();
    if UNIT_POS < 8 {
        let a = shared_1[shared_size_1 - UNIT_POS - 2];
        let b = shared_2[shared_size_2 - UNIT_POS - 1];
        output[UNIT_POS] = a + b;
    }
}

pub fn test_kernel_with_comptime_tag<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[5.0])).expect("Alloc failed");
    let array_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 1, 1) };

    kernel_with_comptime_tag::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        ComptimeTagLaunch::new(array_arg, &"zero".to_string()),
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(0.0));

    let handle = client.create(f32::as_bytes(&[5.0])).expect("Alloc failed");
    let array_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 1, 1) };

    kernel_with_comptime_tag::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        ComptimeTagLaunch::new(array_arg, &"not_zero".to_string()),
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(1.0));
}

pub fn test_kernel_with_generics<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0, 1.0]).expect("Alloc failed");

    kernel_with_generics::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_kernel_without_generics<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client
        .create(f32::as_bytes(&[0.0, 1.0]))
        .expect("Alloc failed");

    kernel_without_generics::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

pub fn test_kernel_max_shared<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let total_shared_size = client.properties().hardware.max_shared_memory_size;

    let handle = client
        .create(u32::as_bytes(&[0, 1, 2, 3, 4, 5, 6, 7]))
        .expect("Alloc failed");

    // Allocate 24kibi to a check buffer, and the rest to the second buffer
    let shared_size_1 = 24576 / size_of::<u32>();
    let shared_size_2 = (total_shared_size - 24576) / size_of::<u32>();

    kernel_with_max_shared::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 8, 1) },
        shared_size_1 as u32,
        shared_size_2 as u32,
    );

    let actual = client.read_one(handle.binding());
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, &[1, 9, 9, 9, 9, 9, 9, 1]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_launch {
    () => {
        use super::*;

        #[test]
        fn test_launch_with_generics() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_with_generics::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_launch_without_generics() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_without_generics::<TestRuntime>(client);
        }

        #[test]
        fn test_launch_with_comptime_tag() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_with_comptime_tag::<TestRuntime>(
                client,
            );
        }

        #[ignore = "Seemingly flaky with CPU emulation"]
        #[test]
        fn test_launch_with_max_shared() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_max_shared::<TestRuntime>(client);
        }
    };
}
