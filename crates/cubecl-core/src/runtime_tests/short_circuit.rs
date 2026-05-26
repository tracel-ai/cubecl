use crate::prelude::*;
use crate::{self as cubecl};
use cubecl_runtime::server::Handle;

// Pin `||` / `&&` short-circuit semantics. The RHS mutates a side-channel.
// If short-circuit holds, the channel is never written.

#[cube]
fn mark_side_channel(side_channel: &mut Array<u32>) -> bool {
    side_channel[0] = 1u32;
    false.runtime()
}

#[cube(launch)]
pub fn kernel_short_circuit_or(output: &mut [u32], left_input: u32) {
    if UNIT_POS == 0 {
        let mut side_channel = Array::<u32>::new(1usize);
        side_channel[0] = 0u32;
        let flag = (left_input != 0u32) || mark_side_channel(&mut side_channel);
        if flag {
            output[0] = side_channel[0];
        } else {
            output[0] = 999u32;
        }
    }
}

#[cube(launch)]
pub fn kernel_short_circuit_and(output: &mut [u32], left_input: u32) {
    if UNIT_POS == 0 {
        let mut side_channel = Array::<u32>::new(1usize);
        side_channel[0] = 0u32;
        let flag = (left_input != 0u32) && mark_side_channel(&mut side_channel);
        if !flag {
            output[0] = side_channel[0];
        } else {
            output[0] = 999u32;
        }
    }
}

// Pure operands take the eager path. The logical result must still be correct.

#[cube(launch)]
pub fn kernel_pure_or(output: &mut [u32], a: u32, b: u32) {
    if UNIT_POS == 0 {
        let flag = (a != 0u32) || (b != 0u32);
        if flag {
            output[0] = 1u32;
        } else {
            output[0] = 0u32;
        }
    }
}

#[cube(launch)]
pub fn kernel_pure_and(output: &mut [u32], a: u32, b: u32) {
    if UNIT_POS == 0 {
        let flag = (a != 0u32) && (b != 0u32);
        if flag {
            output[0] = 1u32;
        } else {
            output[0] = 0u32;
        }
    }
}

pub fn test_short_circuit_or<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.empty(core::mem::size_of::<u32>());
    kernel_short_circuit_or::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        1u32,
    );
    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    assert_eq!(actual[0], 0, "`||` did not short-circuit");
}

pub fn test_short_circuit_and<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.empty(core::mem::size_of::<u32>());
    kernel_short_circuit_and::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        0u32,
    );
    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    assert_eq!(actual[0], 0, "`&&` did not short-circuit");
}

fn run_pure<R: Runtime>(
    client: &ComputeClient<R>,
    launch: impl Fn(&ComputeClient<R>, Handle, u32, u32),
    a: u32,
    b: u32,
) -> u32 {
    let handle = client.empty(core::mem::size_of::<u32>());
    launch(client, handle.clone(), a, b);
    let actual = client.read_one_unchecked(handle);
    u32::from_bytes(&actual)[0]
}

pub fn test_pure_or<R: Runtime>(client: ComputeClient<R>) {
    let launch = |client: &ComputeClient<R>, handle: Handle, a, b| {
        kernel_pure_or::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(handle, 1) },
            a,
            b,
        );
    };
    assert_eq!(run_pure(&client, launch, 0, 0), 0, "0 || 0");
    assert_eq!(run_pure(&client, launch, 0, 7), 1, "0 || 7");
    assert_eq!(run_pure(&client, launch, 5, 0), 1, "5 || 0");
    assert_eq!(run_pure(&client, launch, 5, 7), 1, "5 || 7");
}

pub fn test_pure_and<R: Runtime>(client: ComputeClient<R>) {
    let launch = |client: &ComputeClient<R>, handle: Handle, a, b| {
        kernel_pure_and::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(handle, 1) },
            a,
            b,
        );
    };
    assert_eq!(run_pure(&client, launch, 0, 0), 0, "0 && 0");
    assert_eq!(run_pure(&client, launch, 0, 7), 0, "0 && 7");
    assert_eq!(run_pure(&client, launch, 5, 0), 0, "5 && 0");
    assert_eq!(run_pure(&client, launch, 5, 7), 1, "5 && 7");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_short_circuit {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_short_circuit_or() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_short_circuit_or::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_short_circuit_and() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_short_circuit_and::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_pure_or() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_pure_or::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_pure_and() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_pure_and::<TestRuntime>(client);
        }
    };
}
