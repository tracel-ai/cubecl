//! Launch-validation tests for the Metal backend.
//!
//! Resource-limit violations must surface as `LaunchError::TooManyResources`, not as the
//! opaque `CompilationError` Metal's pipeline creation produces when a kernel genuinely
//! uses more threadgroup memory than the device allows.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_core::{
    Runtime,
    server::{LaunchError, ResourceLimitError, ServerError},
};

type R = crate::MetalRuntime;

#[cube(launch)]
fn oversized_smem_kernel(output: &mut [u32], #[comptime] shared_size: usize) {
    let mut shared = Shared::new_slice(shared_size);
    // Runtime-dependent indices so the MSL compiler can't shrink the allocation.
    let idx = output[0] as usize;
    shared[idx] = output[0];
    sync_cube();
    output[0] = shared[idx + 1];
}

#[test]
fn oversized_shared_memory_is_a_resource_limit_error() {
    let client = R::client(&Default::default());
    let max = client.properties().hardware.max_shared_memory_size;
    let shared_size = (max + 1).div_ceil(size_of::<u32>());
    let requested_bytes = shared_size * size_of::<u32>();

    let handle = client.create_from_slice(u32::as_bytes(&[0]));
    oversized_smem_kernel::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        shared_size,
    );

    let err = client
        .flush()
        .expect_err("a launch requesting more shared memory than the device limit must fail");
    let ServerError::ServerUnhealthy { errors, .. } = err else {
        panic!("expected ServerUnhealthy, got: {err}");
    };
    match &errors[0] {
        ServerError::Launch(LaunchError::TooManyResources(ResourceLimitError::SharedMemory {
            requested,
            max: reported_max,
            ..
        })) => {
            assert_eq!(*requested, requested_bytes);
            assert_eq!(*reported_max, max);
        }
        other => panic!("expected a shared memory resource limit error, got: {other}"),
    }
}
