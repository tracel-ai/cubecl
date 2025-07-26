use crate::prelude::*;
use crate::{self as cubecl, debug_print};

#[cube]
fn helper_fn<F: Float>(num: F) -> F {
    num * num
}

#[cube(launch)]
fn simple_call_kernel<F: Float>(out: &mut Array<F>) {
    if UNIT_POS == 0 {
        out[0] = helper_fn::<F>(out[0]);
    }
}

pub fn test_simple_call<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client
        .create(f32::as_bytes(&[10.0, 1.0]))
        .expect("Alloc failed");

    let vectorization = 1;

    simple_call_kernel::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 100.0);
}

#[cube]
fn nested_helper<F: Float>(num: F) -> F {
    helper_fn::<F>(num) * num
}

#[cube(launch)]
fn nested_call_kernel<F: Float>(out: &mut Array<F>) {
    if UNIT_POS == 0 {
        out[0] = nested_helper::<F>(out[0]);
    }
}

pub fn test_nested_call<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client
        .create(f32::as_bytes(&[10.0, 1.0]))
        .expect("Alloc failed");

    let vectorization = 1;

    nested_call_kernel::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 1000.0);
}

#[cube(launch)]
fn debug_print_kernel<F: Float>(out: &mut Array<F>) {
    if UNIT_POS == 0 {
        let val = out[0];
        debug_print!("Test value: %f\n", val);
        out[0] = helper_fn::<F>(val);
    }
}

#[cfg(not(all(target_os = "macos")))]
pub fn test_debug_print<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    //let logger = MemoryLogger::setup(log::Level::Info);
    let handle = client
        .create(f32::as_bytes(&[10.0, 1.0]))
        .expect("Alloc failed");

    let vectorization = 1;

    debug_print_kernel::launch::<f32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<f32>(&handle, 2, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = f32::from_bytes(&actual);

    // No way to assert the log is happening right now because CUDA prints to stdout, which can't be
    // easily captured
    assert_eq!(actual[0], 100.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_debug {
    () => {
        use super::*;

        #[test]
        fn test_simple_call_debug() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::debug::test_simple_call::<TestRuntime>(client);
        }

        #[test]
        fn test_nested_call_debug() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::debug::test_nested_call::<TestRuntime>(client);
        }

        #[cfg(not(all(target_os = "macos")))]
        #[test]
        fn test_debug_print() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::debug::test_debug_print::<TestRuntime>(client);
        }
    };
}
