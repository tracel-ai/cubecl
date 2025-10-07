use cubecl::prelude::*;
use cubecl_core as cubecl;
use std::f32::consts::{PI, TAU};

use crate::trigonometry::*;

#[cube(launch_unchecked)]
fn kernel_to_degrees(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < input.len() {
        output[UNIT_POS] = to_degrees::<f32>(input[UNIT_POS]);
    }
}

pub fn test_to_degrees<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, TAU];
    let expected = vec![0.0, 30.0, 45.0, 90.0, 180.0, 360.0];

    let input = client.create(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_to_degrees::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
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
    if UNIT_POS < input.len() {
        output[UNIT_POS] = to_radians::<f32>(input[UNIT_POS]);
    }
}

pub fn test_to_radians<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![0.0, 30.0, 45.0, 90.0, 180.0, 360.0];
    let expected = vec![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, TAU];

    let input = client.create(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_to_radians::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
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
fn kernel_hypot(x: &Array<f32>, y: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < x.len() {
        output[UNIT_POS] = hypot::<f32>(x[UNIT_POS], y[UNIT_POS]);
    }
}

pub fn test_hypot<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let x_data = vec![3.0, 0.0, 1.0, 5.0, 0.0];
    let y_data = vec![4.0, 1.0, 1.0, 12.0, 0.0];
    let expected = vec![5.0, 1.0, 1.4142135623730951, 13.0, 0.0];

    let x = client.create(f32::as_bytes(&x_data));
    let y = client.create(f32::as_bytes(&y_data));
    let output = client.empty(x_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_hypot::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(x_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&x, x_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&y, y_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, x_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Hypot test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}
