use crate::prelude::*;
use crate::{self as cubecl};

// Pin `||` / `&&` short-circuit semantics. The RHS mutates a side-channel.
// If short-circuit holds, the channel is never written.

#[cube]
fn mark_side_channel(side_channel: &mut Array<u32>) -> bool {
    side_channel[0] = 1u32;
    false.into()
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
    };
}
