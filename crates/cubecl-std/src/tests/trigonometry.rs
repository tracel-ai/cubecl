use core::f32::consts::{PI, TAU};
use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::trigonometry::*;

#[cube(launch_unchecked)]
fn kernel_to_degrees(input: &Array<f32>, output: &mut Array<f32>) {
    if (UNIT_POS as usize) < input.len() {
        output[UNIT_POS as usize] = to_degrees::<f32>(input[UNIT_POS as usize]);
    }
}

pub fn test_to_degrees<R: Runtime>(client: ComputeClient<R>) {
    let input_data = vec![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, TAU];
    let expected = [0.0, 30.0, 45.0, 90.0, 180.0, 360.0];

    let input = client.create_from_slice(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_to_degrees::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(input_data.len() as u32),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_to_radians(input: &Array<f32>, output: &mut Array<f32>) {
    if (UNIT_POS as usize) < input.len() {
        output[UNIT_POS as usize] = to_radians::<f32>(input[UNIT_POS as usize]);
    }
}

pub fn test_to_radians<R: Runtime>(client: ComputeClient<R>) {
    let input_data = vec![0.0, 30.0, 45.0, 90.0, 180.0, 360.0];
    let expected = [0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, TAU];

    let input = client.create_from_slice(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_to_radians::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(input_data.len() as u32),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[macro_export]
macro_rules! testgen_trigonometry {
    () => {
        mod trigonometry {
            use super::*;
            use $crate::tests::trigonometry::*;

            #[$crate::tests::test_log::test]
            fn test_to_degrees_conversion() {
                let client = TestRuntime::client(&Default::default());
                test_to_degrees::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_to_radians_conversion() {
                let client = TestRuntime::client(&Default::default());
                test_to_radians::<TestRuntime>(client);
            }
        }
    };
}
