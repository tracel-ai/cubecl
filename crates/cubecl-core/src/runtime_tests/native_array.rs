use crate as cubecl;
use cubecl::prelude::*;
use cubecl_runtime::runtime::Runtime;

// Distinct surrounding scalars catch incorrect argument registration order.
#[cube(launch)]
pub fn native_array_launch_arg(prefix: f32, values: [f32; 2], suffix: f32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    output[0] = prefix;
    output[1] = values[0];
    output[2] = values[1];
    output[3] = suffix;
}

#[cube(launch)]
pub fn native_array_repeat(value: f32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    // The repeated value is a runtime input.
    let values = [value; 10];

    #[allow(clippy::manual_memcpy)]
    for i in 0..10usize {
        output[i] = values[i];
    }
}

#[cube(launch)]
pub fn native_array_literal(first: f32, second: f32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let values = [first, second];

    output[0] = values[0];
    output[1] = values[1];
}

#[cube(launch)]
pub fn native_array_index_assign(value: f32, index: u32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let mut values = [1.0f32, 2.0, 3.0].runtime();

    values[0] = value;
    values[index as usize] += 5.0;

    output[0] = values[0];
    output[1] = values[1];
    output[2] = values[2];
}

#[cube(launch)]
pub fn native_array_assign(value: f32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let mut source = [value, value + 1.0];
    let mut destination = [0.0f32; 2];

    output[0] = destination[0];

    // Whole-array assignment must copy the elements.
    destination = source;

    // Mutate both arrays to check independence in both directions.
    source[0] += 10.0;
    destination[1] += 20.0;

    output[1] = destination[0];
    output[2] = destination[1];
    output[3] = source[0];
    output[4] = source[1];

    // Assignment directly from an element-list expression.
    destination = [value + 30.0, value + 40.0];

    output[5] = destination[0];
    output[6] = destination[1];

    // Assignment directly from a repeat expression.
    destination = [value + 50.0; 2];

    output[7] = destination[0];
    output[8] = destination[1];
}

#[cube(launch)]
pub fn native_array_branch_assign(condition: u32, value: f32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    #[allow(clippy::needless_late_init, reason = "testing")]
    let values: [f32; 2];

    if condition != 0 {
        values = [value + 1.0, value + 2.0];
    } else {
        values = [value + 3.0; 2];
    }

    output[0] = values[0];
    output[1] = values[1];
}

#[cube(launch)]
pub fn native_array_copy_init(value: f32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    // Runtime input prevents this from becoming a comptime array.
    let mut source = [value, value + 1.0];
    let mut copy = source;

    copy[0] += 10.0;
    source[1] += 20.0;

    output[0] = source[0];
    output[1] = source[1];
    output[2] = copy[0];
    output[3] = copy[1];
}

#[allow(clippy::clone_on_copy, unused_assignments)]
#[cube(launch)]
pub fn native_array_clone_init(value: f32, output: &mut [f32]) {
    if UNIT_POS != 0 {
        terminate!();
    }

    // Runtime input prevents this from becoming a comptime array.
    let mut source = [value, value + 1.0];
    let snapshot = source.clone();
    let snapshot2 = source;

    source[0] += 10.0;
    output[0] = snapshot[0];
    output[1] = snapshot2[0];
}

fn assert_output(client: &Client, expected: &[f32], launch: impl FnOnce(BufferArg)) {
    // A sentinel makes missing writes visible.
    let initial = alloc::vec![-999.0f32; expected.len()];
    let handle = client.create_from_slice(f32::as_bytes(&initial));

    launch(unsafe { BufferArg::from_raw_parts(handle.clone(), expected.len()) });

    let actual = client.read_one(handle).unwrap();
    assert_eq!(f32::from_bytes(&actual), expected);
}

pub fn test_native_array_launch_arg<R: Runtime>(client: Client) {
    // Reuse the same kernel with different values to catch accidental
    // specialization or stale scalar values in the kernel cache.
    for (prefix, values, suffix) in [
        (7.0f32, [2.0f32, 5.0], 11.0f32),
        (19.0f32, [13.0f32, 17.0], 23.0f32),
    ] {
        let expected = [prefix, values[0], values[1], suffix];

        assert_output(&client, &expected, |output| {
            native_array_launch_arg::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                prefix,
                values,
                suffix,
                output,
            );
        });
    }
}

pub fn test_native_array_repeat<R: Runtime>(client: Client) {
    for value in [3.0f32, -2.0] {
        assert_output(&client, &[value; 10], |output| {
            native_array_repeat::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                value,
                output,
            );
        });
    }
}

pub fn test_native_array_literal<R: Runtime>(client: Client) {
    for (first, second) in [(-3.0f32, 7.0f32), (11.0f32, -5.0f32)] {
        assert_output(&client, &[first, second], |output| {
            native_array_literal::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                first,
                second,
                output,
            );
        });
    }
}

pub fn test_native_array_index_assign<R: Runtime>(client: Client) {
    for (index, expected) in [
        (0u32, [12.0f32, 2.0, 3.0]),
        (1u32, [7.0f32, 7.0, 3.0]),
        (2u32, [7.0f32, 2.0, 8.0]),
    ] {
        assert_output(&client, &expected, |output| {
            native_array_index_assign::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                7.0,
                index,
                output,
            );
        });
    }
}

pub fn test_native_array_assign<R: Runtime>(client: Client) {
    let expected = [
        0.0f32, 3.0, 24.0, // Destination after copying and mutation.
        13.0, 4.0, // Source remains independent.
        33.0, 43.0, // Element-list reassignment.
        53.0, 53.0, // Repeat reassignment.
    ];

    assert_output(&client, &expected, |output| {
        native_array_assign::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            3.0,
            output,
        );
    });
}

pub fn test_native_array_branch_assign<R: Runtime>(client: Client) {
    for (condition, expected) in [(0u32, [6.0f32, 6.0]), (1u32, [4.0f32, 5.0])] {
        assert_output(&client, &expected, |output| {
            native_array_branch_assign::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                condition,
                3.0,
                output,
            );
        });
    }
}

pub fn test_native_array_copy_init<R: Runtime>(client: Client) {
    assert_output(&client, &[3.0, 24.0, 13.0, 4.0], |output| {
        native_array_copy_init::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            3.0,
            output,
        );
    });
}

pub fn test_native_array_clone_init<R: Runtime>(client: Client) {
    assert_output(&client, &[3.0, 3.0], |output| {
        native_array_clone_init::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            3.0,
            output,
        );
    });
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_native_array {
    () => {
        mod native_array {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_launch_arg() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_launch_arg::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_repeat() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_repeat::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_literal() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_literal::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_index_assign() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_index_assign::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_assign() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_assign::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_branch_assign() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_branch_assign::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_copy_init() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_copy_init::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_native_array_clone_init() {
                let client = TestRuntime::client(&Default::default());
                $crate::runtime_tests::native_array::test_native_array_clone_init::<TestRuntime>(
                    client,
                );
            }
        }
    };
}
