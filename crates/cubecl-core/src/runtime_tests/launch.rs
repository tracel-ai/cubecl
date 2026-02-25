use std::println;

use alloc::string::{String, ToString};

use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;
use cubecl_runtime::server::{ResourceLimitError, ServerError};

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

#[cube(launch, address_type = "dynamic")]
pub fn kernel_dynamic_addressing(output: &mut Array<f32>) {
    if UNIT_POS == 0 {
        output[0] = 5.0;
    }
}

#[cube(launch)]
pub fn kernel_with_max_shared(
    output: &mut Array<u32>,
    #[comptime] shared_size_1: usize,
    #[comptime] shared_size_2: usize,
) {
    let mut shared_1 = SharedMemory::<u32>::new(shared_size_1);
    let mut shared_2 = SharedMemory::<u32>::new(shared_size_2);
    if UNIT_POS < 8 {
        shared_1[shared_size_1 - UNIT_POS as usize - 1] = output[UNIT_POS as usize];
        shared_2[shared_size_2 - UNIT_POS as usize - 1] = output[8 - UNIT_POS as usize];
    }
    sync_cube();
    if UNIT_POS < 8 {
        let a = shared_1[shared_size_1 - UNIT_POS as usize - 2];
        let b = shared_2[shared_size_2 - UNIT_POS as usize - 1];
        output[UNIT_POS as usize] = a + b;
    }
}

#[cube(launch)]
pub fn kernel_resource_errors(output: &mut Array<u32>, #[comptime] shared_size: usize) {
    let mut shared = SharedMemory::<u32>::new(shared_size);
    // Add some dummy code to prevent smem from being optimized out
    shared[0] = 0;
    sync_cube();
    output[0] = shared[0];
}

pub fn test_kernel_with_comptime_tag<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(f32::as_bytes(&[5.0]));
    let array_arg = unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 1, 1) };

    kernel_with_comptime_tag::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        ComptimeTagLaunch::new(array_arg, "zero".to_string()),
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(0.0));

    let handle = client.create_from_slice(f32::as_bytes(&[5.0]));
    let array_arg = unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 1, 1) };

    kernel_with_comptime_tag::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        ComptimeTagLaunch::new(array_arg, "not_zero".to_string()),
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(1.0));
}

pub fn test_kernel_with_generics<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    kernel_with_generics::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts::<F>(handle.clone(), 2, 1) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_kernel_without_generics<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(f32::as_bytes(&[0.0, 1.0]));

    kernel_without_generics::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 2, 1) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

pub fn test_kernel_max_shared<R: Runtime>(client: ComputeClient<R>) {
    let total_shared_size = client.properties().hardware.max_shared_memory_size;

    let handle = client.create_from_slice(u32::as_bytes(&[0, 1, 2, 3, 4, 5, 6, 7]));

    // Allocate 24kibi to a check buffer, and the rest to the second buffer
    let shared_size_1 = 24576 / size_of::<u32>();
    let shared_size_2 = (total_shared_size - 24576) / size_of::<u32>();

    kernel_with_max_shared::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 8, 1) },
        shared_size_1,
        shared_size_2,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, &[1, 9, 9, 9, 9, 9, 9, 1]);
}

pub fn test_shared_memory_error<R: Runtime>(client: ComputeClient<R>) {
    // No real limit on CPU, so ignore
    if client.properties().hardware.num_cpu_cores.is_some() {
        return;
    }

    let max_shared_size = client.properties().hardware.max_shared_memory_size;

    let shared_size = (max_shared_size + 1).div_ceil(size_of::<u32>());
    let requested_bytes = shared_size * size_of::<u32>();

    let handle = client.create_from_slice(u32::as_bytes(&[0]));
    let error = client
        .clone()
        .exclusive(move || {
            kernel_resource_errors::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 1, 1) },
                shared_size,
            );

            client.flush_errors().remove(0)
        })
        .unwrap();

    match error {
        ServerError::Launch(LaunchError::TooManyResources(inner)) => match inner {
            ResourceLimitError::SharedMemory { requested, max, .. } => {
                assert_eq!(
                    requested_bytes, requested,
                    "Requested should be equal to requested size"
                );
                assert_eq!(
                    max_shared_size, max,
                    "Max should be equal to max shared size"
                );
            }
            other => panic!("Should be shared memory resource error, is {other:?}"),
        },
        other => panic!("Should be resource error, is {other:?}"),
    }
}

pub fn test_cube_dim_error<R: Runtime>(client: ComputeClient<R>) {
    let max_cube_dim = client.properties().hardware.max_cube_dim;
    let max_units = client.properties().hardware.max_units_per_cube;

    // CPU has no limit, and + 1 will overflow
    if max_cube_dim.2 == u32::MAX {
        return;
    }

    let handle = client.create_from_slice(u32::as_bytes(&[0]));

    let error = client
        .clone()
        .exclusive(move || {
            kernel_resource_errors::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_3d(1, 1, max_cube_dim.2 + 1),
                unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 1, 1) },
                1,
            );
            client.flush_errors().pop().unwrap()
        })
        .unwrap();

    match error {
        ServerError::Launch(LaunchError::TooManyResources(inner)) => match inner {
            ResourceLimitError::CubeDim { requested, max, .. } => {
                assert_eq!((1, 1, max_cube_dim.2 + 1), requested);
                assert_eq!(max_cube_dim, max);
            }
            // Could also be valid
            ResourceLimitError::Units { requested, max, .. } if max_cube_dim.2 >= max_units => {
                assert_eq!(max_cube_dim.2 + 1, requested);
                assert_eq!(max_units, max);
            }
            other => panic!("Should be shared memory resource error, is {other:?}"),
        },
        other => panic!("Should be resource error, is {other:?}"),
    }
}

pub fn test_max_units_error<R: Runtime>(client: ComputeClient<R>) {
    let max_cube_dim = client.properties().hardware.max_cube_dim;
    // CPU has no limit, and num_elems will overflow
    if max_cube_dim.2 == u32::MAX {
        return;
    }

    let max_units = client.properties().hardware.max_units_per_cube;
    let cube_dim: CubeDim = max_cube_dim.into();
    let requested_units = cube_dim.num_elems();

    let handle = client.create_from_slice(u32::as_bytes(&[0]));

    let error = client
        .clone()
        .exclusive(move || {
            kernel_resource_errors::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                cube_dim,
                unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 1, 1) },
                1,
            );

            client.flush_errors().remove(0)
        })
        .unwrap();

    match error {
        ServerError::Launch(LaunchError::TooManyResources(inner)) => match inner {
            ResourceLimitError::Units { requested, max, .. } => {
                assert_eq!(requested_units, requested);
                assert_eq!(max_units, max);
            }
            other => panic!("Should be shared memory resource error, is {other:?}"),
        },
        other => panic!("Should be resource error, is {other:?}"),
    }
}

pub fn test_kernel_dynamic_addressing<R: Runtime>(
    client: ComputeClient<R>,
    address_type: AddressType,
) {
    let handle = client.create_from_slice(f32::as_bytes(&[0.0, 1.0]));

    if !client.properties().supports_address(address_type) {
        println!("Skipping dynamic addressing kernel, no type support");
        return;
    }

    kernel_dynamic_addressing::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        address_type,
        unsafe { ArrayArg::from_raw_parts::<f32>(handle.clone(), 2, 1) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_launch {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_launch_with_generics() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_with_generics::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_launch_without_generics() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_without_generics::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_launch_with_comptime_tag() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_with_comptime_tag::<TestRuntime>(
                client,
            );
        }

        #[ignore = "Seemingly flaky with CPU emulation"]
        #[$crate::runtime_tests::test_log::test]
        fn test_launch_with_max_shared() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_max_shared::<TestRuntime>(client);
        }
    };
}

#[macro_export]
macro_rules! testgen_launch_untyped {
    () => {
        #[test]
        fn test_launch_dynamic_addressing_32() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_dynamic_addressing::<TestRuntime>(
                client.clone(),
                AddressType::U32,
            );
        }

        #[test]
        fn test_launch_dynamic_addressing_64() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_kernel_dynamic_addressing::<TestRuntime>(
                client,
                AddressType::U64,
            );
        }

        #[test]
        fn test_launch_shared_memory_error() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_shared_memory_error::<TestRuntime>(client);
        }

        #[test]
        fn test_launch_cube_dim_error() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_cube_dim_error::<TestRuntime>(client);
        }

        #[test]
        fn test_launch_units_error() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::launch::test_max_units_error::<TestRuntime>(client);
        }
    };
}
