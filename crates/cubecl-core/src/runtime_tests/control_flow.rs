use crate::prelude::*;
use crate::{self as cubecl};
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;

// Pin `||` / `&&` short-circuit semantics. The RHS mutates a side-channel.
// If short-circuit holds, the channel is never written.

#[cube]
fn mark_side_channel(side_channel: &mut Array<u32>) -> bool {
    side_channel[0] = 1u32;
    false
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

#[cube(launch)]
pub fn kernel_short_circuit_and_while(output: &mut [u32], left_input: u32) {
    if UNIT_POS == 0 {
        let mut side_channel = Array::<u32>::new(1usize);
        side_channel[0] = 0u32;
        while (left_input != 0u32) && mark_side_channel(&mut side_channel) {
            output[0] = side_channel[0];
        }
        output[0] = side_channel[0];
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

#[cube(launch)]
pub fn kernel_early_return(output: &mut [i32], divisor: i32) {
    if UNIT_POS == 0 {
        output[0] = early_returning(divisor);
    }
}

#[cube]
fn early_returning(divisor: i32) -> i32 {
    if divisor == 0 {
        return 0;
    }
    4 / divisor
}

#[cube(launch)]
pub fn kernel_early_return_no_value(output: &mut [i32], divisor: i32) {
    if UNIT_POS == 0 {
        early_returning_no_value(output, divisor);
    }
}

#[cube]
fn early_returning_no_value(output: &mut [i32], divisor: i32) {
    if divisor == 0 {
        return;
    }
    output[0] = 4;
}

#[cube(launch)]
pub fn kernel_terminate_nested(output: &mut [i32], cond: u32) {
    if UNIT_POS < output.len() as u32 {
        nested_terminate(cond != 0);
        output[UNIT_POS as usize] = UNIT_POS as i32;
    }
}

#[cube]
fn nested_terminate(cond: bool) {
    // Ensure return flag and terminate flag are not conflated
    if !cond {
        return;
    }
    terminate!();
}

#[cube(launch)]
pub fn kernel_continue(output: &mut [i32]) {
    if UNIT_POS == 0 {
        for i in 0..4 {
            if i != 1 {
                continue;
            }
            output[0] = i;
        }
    }
}

#[cube(launch)]
pub fn kernel_continue_and_break(output: &mut [i32]) {
    if UNIT_POS == 0 {
        for i in 0..4 {
            if i > 2 {
                break;
            }
            if i != 1 {
                continue;
            }
            output[0] = i;
        }
    }
}

#[allow(clippy::while_immutable_condition)]
#[allow(unused_mut, reason = "otherwise it's comptime")]
#[cube(launch)]
pub fn kernel_while_terminate(output: &mut [i32]) {
    if UNIT_POS == 0 {
        let mut cond = true;
        while cond {
            if cond {
                terminate!();
            }
            output[0] = 10;
        }
        output[0] = 20;
    }
}

#[cube(launch)]
pub fn kernel_comptime_terminate_ends_scope(output: &mut [i32], #[comptime] terminate: bool) {
    if terminate {
        terminate!();
    }
    output[0] = 10;
}

pub fn test_short_circuit_or<R: Runtime>(client: Client) {
    let handle = client.empty(core::mem::size_of::<u32>());
    kernel_short_circuit_or::launch(
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

pub fn test_short_circuit_and<R: Runtime>(client: Client) {
    let handle = client.empty(core::mem::size_of::<u32>());
    kernel_short_circuit_and::launch(
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

pub fn test_short_circuit_and_while<R: Runtime>(client: Client) {
    let handle = client.empty(core::mem::size_of::<u32>());
    kernel_short_circuit_and_while::launch(
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

fn run_pure(client: &Client, launch: impl Fn(&Client, Handle, u32, u32), a: u32, b: u32) -> u32 {
    let handle = client.empty(core::mem::size_of::<u32>());
    launch(client, handle.clone(), a, b);
    let actual = client.read_one_unchecked(handle);
    u32::from_bytes(&actual)[0]
}

pub fn test_pure_or<R: Runtime>(client: Client) {
    let launch = |client: &Client, handle: Handle, a, b| {
        kernel_pure_or::launch(
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

pub fn test_pure_and<R: Runtime>(client: Client) {
    let launch = |client: &Client, handle: Handle, a, b| {
        kernel_pure_and::launch(
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

pub fn test_early_return<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(i32::as_bytes(&[5]));

    kernel_early_return::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        0,
    );

    let actual = client.read_one(handle.clone()).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 0);

    kernel_early_return::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        2,
    );

    let actual = client.read_one(handle).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 2);
}

pub fn test_early_return_no_value<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(i32::as_bytes(&[5]));

    kernel_early_return_no_value::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        0,
    );

    let actual = client.read_one(handle.clone()).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 5);

    kernel_early_return_no_value::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        2,
    );

    let actual = client.read_one(handle).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 4);
}

pub fn test_nested_terminate<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(i32::as_bytes(&[0, 0, 0, 0]));

    kernel_terminate_nested::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 4),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 4) },
        1,
    );

    let actual = client.read_one(handle.clone()).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual, [0, 0, 0, 0]);

    kernel_terminate_nested::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 4),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 4) },
        0,
    );

    let actual = client.read_one(handle).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual, [0, 1, 2, 3]);
}

pub fn test_continue<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(i32::as_bytes(&[0]));

    kernel_continue::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
    );

    let actual = client.read_one(handle.clone()).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 1);
}

pub fn test_continue_and_break<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(i32::as_bytes(&[0]));

    kernel_continue_and_break::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
    );

    let actual = client.read_one(handle.clone()).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 1);
}

pub fn test_while_terminate<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(i32::as_bytes(&[5]));

    kernel_while_terminate::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
    );

    let actual = client.read_one(handle.clone()).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 5);
}

pub fn comptime_terminate_ends_scope<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(i32::as_bytes(&[5]));

    kernel_comptime_terminate_ends_scope::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        true,
    );

    let actual = client.read_one(handle.clone()).unwrap();
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 5);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_control_flow {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_short_circuit_or() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_short_circuit_or::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_short_circuit_and() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_short_circuit_and::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_short_circuit_and_while() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_short_circuit_and_while::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_pure_or() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_pure_or::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_pure_and() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_pure_and::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_early_return() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_early_return::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_early_return_no_value() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_early_return_no_value::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_nested_terminate() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_nested_terminate::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_continue() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_continue::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_continue_and_break() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_continue_and_break::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_while_terminate() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::test_while_terminate::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn comptime_terminate_ends_scope() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::control_flow::comptime_terminate_ends_scope::<TestRuntime>(
                client,
            );
        }
    };
}
