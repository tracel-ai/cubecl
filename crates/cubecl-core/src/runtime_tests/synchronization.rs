use alloc::{vec, vec::Vec};

use crate::prelude::*;
use crate::{self as cubecl};
use cubecl_ir::features::Plane;

#[cube(launch)]
/// First 32 elements should be 1, while last 32 elements may or may not be 1
fn kernel_test_sync_cube(buffer: &mut Array<u32>, out: &mut Array<u32>) {
    let unit_pos = UNIT_POS as usize;
    buffer[unit_pos] = UNIT_POS;
    sync_cube();
    if unit_pos != 0 {
        out[unit_pos] = buffer[unit_pos - 1] + buffer[unit_pos];
    }
}

pub fn test_sync_cube<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.empty(32 * core::mem::size_of::<u32>());
    let test = client.empty(32 * core::mem::size_of::<u32>());

    let vectorization = 1;

    kernel_test_sync_cube::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(8, 2),
        unsafe { ArrayArg::from_raw_parts::<u32>(&test, 32, vectorization) },
        unsafe { ArrayArg::from_raw_parts::<u32>(&handle, 32, vectorization) },
    )
    .unwrap();

    let actual = client.read_one(handle);
    let actual = u32::from_bytes(&actual);

    let expected: Vec<u32> = (0..16)
        .map(|i| core::cmp::max(2 * i - 1, 0) as u32)
        .collect();

    assert_eq!(&actual[1..16], &expected[1..16]);
}

#[cube(launch)]
/// First 32 elements should be 1, while last 32 elements may or may not be 1
fn kernel_test_finished_sync_cube(buffer: &mut Array<u32>, out: &mut Array<u32>) {
    let unit_pos = UNIT_POS as usize;
    buffer[unit_pos] = UNIT_POS;
    if UNIT_POS > 16 {
        terminate!();
    }
    sync_cube();
    sync_cube();
    if UNIT_POS != 0 {
        out[unit_pos] = buffer[unit_pos - 1] + buffer[unit_pos];
    }
    sync_cube();
}

pub fn test_finished_sync_cube<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.empty(32 * core::mem::size_of::<u32>());
    let test = client.empty(32 * core::mem::size_of::<u32>());

    let vectorization = 1;

    kernel_test_finished_sync_cube::launch(
        &client,
        CubeCount::Static(2, 1, 1),
        CubeDim::new_2d(8, 2),
        unsafe { ArrayArg::from_raw_parts::<u32>(&test, 32, vectorization) },
        unsafe { ArrayArg::from_raw_parts::<u32>(&handle, 32, vectorization) },
    )
    .unwrap();

    let actual = client.read_one(handle);
    let actual = u32::from_bytes(&actual);

    let expected: Vec<u32> = (0..8)
        .map(|i| core::cmp::max(2 * i - 1, 0) as u32)
        .collect();

    assert_eq!(&actual[1..8], &expected[1..8]);
}

#[cube(launch)]
/// First 32 elements should be 1, while last 32 elements may or may not be 1
fn kernel_test_sync_plane<F: Float>(out: &mut Array<F>) {
    let mut shared_memory = Shared::<F>::new();

    if UNIT_POS == 0 {
        *shared_memory.as_mut() = F::from_int(1);
    }

    sync_plane();

    out[UNIT_POS as usize] = *shared_memory.as_ref();
}

pub fn test_sync_plane<R: Runtime>(client: ComputeClient<R>) {
    if !client.properties().features.plane.contains(Plane::Sync) {
        // We can't execute the test, skip.
        return;
    }

    let handle = client.empty(64 * core::mem::size_of::<f32>());

    let vectorization = 1;

    kernel_test_sync_plane::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 2),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, vectorization) },
    )
    .unwrap();

    let actual = client.read_one(handle);
    let actual = f32::from_bytes(&actual);
    let expected = &[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    assert_eq!(&actual[0..32], expected);
}

#[cube(launch)]
/// All 64 elements should be 1
fn kernel_test_sync_cube_shared<F: Float>(out: &mut Array<F>) {
    let mut shared_memory = Shared::<F>::new();

    if UNIT_POS == 0 {
        *shared_memory.as_mut() = F::from_int(1);
    }

    sync_cube();

    out[UNIT_POS as usize] = *shared_memory.as_ref();
}

pub fn test_sync_cube_shared<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.empty(64 * core::mem::size_of::<f32>());

    let vectorization = 1;

    kernel_test_sync_cube_shared::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 2),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, vectorization) },
    )
    .unwrap();

    let actual = client.read_one(handle);
    let actual = f32::from_bytes(&actual);
    let expected = vec![1.0; 64];

    assert_eq!(&actual[0..64], expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_sync_plane {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_sync_plane() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_sync_plane::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_sync_cube() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_sync_cube::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_finished_sync_cube() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_finished_sync_cube::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_sync_cube_shared() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_sync_cube_shared::<TestRuntime>(
                client,
            );
        }
    };
}
